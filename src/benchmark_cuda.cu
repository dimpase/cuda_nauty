#include "cli_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nauty_cuda::Graph;
using nauty_cuda::Result;
using nauty_cuda::Status;

void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflow");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { if (event_ != nullptr) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

__global__ void canonicalize_batch(const Graph* graphs, Result* results,
                                   std::size_t count, std::uint64_t max_nodes)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) results[index] = nauty_cuda::canonicalize(graphs[index], max_nodes);
}

struct Options {
    std::size_t repeats = 5;
    std::size_t warmup = 1;
    std::size_t loops = 1;
    std::size_t batch_size = 4096;
    std::uint64_t max_nodes = 1000000;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if ((option != "--repeats" && option != "--warmup" && option != "--loops" &&
             option != "--batch-size" && option != "--max-nodes") || i + 1 == argc) {
            throw std::invalid_argument("unknown or incomplete option: " + option);
        }
        const std::uint64_t value = nauty_cuda::number(argv[++i]);
        if (option == "--max-nodes") {
            options.max_nodes = value;
            continue;
        }
        if (value > std::numeric_limits<std::size_t>::max() ||
            (value == 0 && option != "--warmup")) {
            throw std::invalid_argument(option + " is out of range");
        }
        if (option == "--repeats") options.repeats = static_cast<std::size_t>(value);
        else if (option == "--warmup") options.warmup = static_cast<std::size_t>(value);
        else if (option == "--loops") options.loops = static_cast<std::size_t>(value);
        else options.batch_size = static_cast<std::size_t>(value);
    }
    return options;
}

std::vector<Graph> read_graphs()
{
    std::vector<Graph> graphs;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(std::cin, line)) {
        ++line_number;
        if (line.empty() || line == ">>graph6<<" || line == ">>graph6<<\r") continue;
        Graph graph{};
        std::string error;
        if (!nauty_cuda::decode_graph6(line, graph, error)) {
            throw std::invalid_argument("input line " + std::to_string(line_number) + ": " + error);
        }
        graphs.push_back(graph);
    }
    if (std::cin.bad()) throw std::runtime_error("failed to read graph input");
    if (graphs.empty()) throw std::invalid_argument("benchmark requires at least one graph");
    return graphs;
}

void validate_result(const Graph& graph, const Result& result, std::size_t index)
{
    const std::string context = "graph " + std::to_string(index) + ": ";
    if (result.status != Status::ok) {
        throw std::runtime_error(context + (result.status == Status::node_limit ?
            "node_limit; no completed-graph timing reported" : "invalid result status"));
    }
    if (result.n != graph.n) throw std::runtime_error(context + "result order mismatch");
    std::uint32_t seen = 0;
    for (unsigned v = 0; v < result.n; ++v) {
        if (result.order[v] >= graph.n || (seen & (std::uint32_t{1} << result.order[v]))) {
            throw std::runtime_error(context + "result witness is not a permutation");
        }
        seen |= std::uint32_t{1} << result.order[v];
    }
    for (unsigned v = 0; v < result.n; ++v) {
        std::uint32_t row = 0;
        for (unsigned u = 0; u < result.n; ++u) {
            row |= ((graph.rows[result.order[v]] >> result.order[u]) & 1u) << u;
        }
        if (result.rows[v] != row) throw std::runtime_error(context + "incorrect canonical witness");
    }
}

bool same_result(const Result& left, const Result& right)
{
    if (left.status != right.status || left.n != right.n || left.nodes != right.nodes) return false;
    for (unsigned i = 0; i < left.n; ++i) {
        if (left.order[i] != right.order[i] || left.rows[i] != right.rows[i]) return false;
    }
    return true;
}

void hash_word(std::uint64_t& hash, std::uint32_t word)
{
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash ^= (word >> shift) & 255u;
        hash *= UINT64_C(1099511628211);
    }
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

void print_times(const std::vector<double>& values)
{
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

int benchmark(const Options& options, const std::vector<Graph>& graphs)
{
    if (options.loops > std::numeric_limits<std::size_t>::max() / graphs.size()) {
        throw std::invalid_argument("graphs times loops is out of range");
    }
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability != cudaSuccess || device_count == 0) {
        throw std::runtime_error(std::string("CUDA benchmark unavailable: ") +
            (availability == cudaSuccess ? "no NVIDIA GPU detected" : cudaGetErrorString(availability)));
    }
    int device = 0, driver_version = 0, runtime_version = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    cudaFuncAttributes kernel{};
    check_cuda(cudaFuncGetAttributes(&kernel, canonicalize_batch), "cudaFuncGetAttributes");
    const int threads = std::min({128, properties.maxThreadsPerBlock,
                                  properties.maxThreadsDim[0], kernel.maxThreadsPerBlock});
    if (threads <= 0 || properties.maxGridSize[0] <= 0) {
        throw std::runtime_error("unsupported CUDA launch dimensions");
    }
    const std::size_t capacity = std::min(options.batch_size, graphs.size());
    const std::size_t max_blocks = 1 + (capacity - 1) / static_cast<std::size_t>(threads);
    if (max_blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        max_blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("batch size exceeds CUDA grid limits");
    }
    constexpr std::size_t bytes_per_graph = sizeof(Graph) + sizeof(Result);
    if (capacity > std::numeric_limits<std::size_t>::max() / bytes_per_graph) {
        throw std::runtime_error("CUDA batch allocation size overflow");
    }
    std::size_t available_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&available_bytes, &total_bytes), "cudaMemGetInfo");
    if (capacity * bytes_per_graph > available_bytes) {
        throw std::runtime_error("batch input and output exceed available CUDA memory");
    }

    DeviceBuffer<Graph> device_graphs(capacity);
    DeviceBuffer<Result> device_results(capacity);
    Event start_event, stop_event;
    std::vector<Result> batch_results(capacity), reference(graphs.size());
    std::vector<double> trials, kernel_trials;
    trials.reserve(options.repeats);
    kernel_trials.reserve(options.repeats);
    bool reference_ready = false;
    std::uint64_t checksum = 0;
    std::uint64_t nodes = 0, max_nodes_observed = 0;
    check_cuda(cudaDeviceSynchronize(), "finish CUDA initialization");

    const auto trial = [&](bool measured) {
        double host_seconds = 0, kernel_seconds = 0;
        std::uint64_t hash = UINT64_C(14695981039346656037);
        nodes = max_nodes_observed = 0;
        for (std::size_t loop = 0; loop < options.loops; ++loop) {
            for (std::size_t offset = 0; offset < graphs.size();) {
              const std::size_t count = std::min(capacity, graphs.size() - offset);
              const unsigned int blocks = static_cast<unsigned int>(
                  1 + (count - 1) / static_cast<std::size_t>(threads));
              const auto begin = std::chrono::steady_clock::now();
              check_cuda(cudaMemcpy(device_graphs.get(), graphs.data() + offset, count * sizeof(Graph),
                                    cudaMemcpyHostToDevice), "copy graph batch to CUDA");
              check_cuda(cudaEventRecord(start_event.get()), "record kernel start");
              canonicalize_batch<<<blocks, threads>>>(device_graphs.get(), device_results.get(),
                                                       count, options.max_nodes);
              check_cuda(cudaGetLastError(), "launch benchmark kernel");
              check_cuda(cudaEventRecord(stop_event.get()), "record kernel stop");
              check_cuda(cudaEventSynchronize(stop_event.get()), "synchronize benchmark kernel");
              check_cuda(cudaMemcpy(batch_results.data(), device_results.get(), count * sizeof(Result),
                                    cudaMemcpyDeviceToHost), "copy benchmark results from CUDA");
              const auto end = std::chrono::steady_clock::now();
              float milliseconds = 0;
              check_cuda(cudaEventElapsedTime(&milliseconds, start_event.get(), stop_event.get()),
                         "measure CUDA kernel interval");
              host_seconds += std::chrono::duration<double>(end - begin).count();
              kernel_seconds += static_cast<double>(milliseconds) / 1000;

              // Validation and hashes are outside both timers. Compare fields,
              // not struct padding, and retain the first pass for exact checks.
              for (std::size_t i = 0; i < count; ++i) {
                  const Result& result = batch_results[i];
                  validate_result(graphs[offset + i], result, offset + i);
                  if (reference_ready && !same_result(reference[offset + i], result)) {
                      throw std::runtime_error("CUDA results changed between benchmark passes");
                  }
                  if (!reference_ready) reference[offset + i] = result;
                  if (result.nodes > std::numeric_limits<std::uint64_t>::max() - nodes) {
                      throw std::runtime_error("aggregate search node count overflow");
                  }
                  nodes += result.nodes;
                  max_nodes_observed = std::max(max_nodes_observed, result.nodes);
                  if (loop == 0) {
                      hash_word(hash, result.n);
                      for (unsigned v = 0; v < result.n; ++v) hash_word(hash, result.rows[v]);
                  }
              }
              offset += count;
            }
            reference_ready = true;
        }
        checksum = hash;
        if (measured) {
            trials.push_back(host_seconds);
            kernel_trials.push_back(kernel_seconds);
        }
    };
    for (std::size_t i = 0; i < options.warmup; ++i) trial(false);
    for (std::size_t i = 0; i < options.repeats; ++i) trial(true);

    std::ostringstream checksum_text;
    checksum_text << std::hex << std::setfill('0') << std::setw(16) << checksum;
    std::cout << std::setprecision(17)
              << "{\"status\":\"ok\",\"backend\":\"cuda\",\"graphs\":" << graphs.size()
              << ",\"completed\":" << graphs.size() << ",\"batch_size\":" << options.batch_size
              << ",\"repeats\":" << options.repeats << ",\"warmup\":" << options.warmup
              << ",\"loops\":" << options.loops << ",\"operations\":" << graphs.size() * options.loops
              << ",\"nodes\":" << nodes << ",\"max_nodes_observed\":" << max_nodes_observed
              << ",\"max_nodes\":" << options.max_nodes << ",\"checksum\":\"" << checksum_text.str()
              << "\",\"trials_seconds\":";
    print_times(trials);
    std::cout << ",\"kernel_trials_seconds\":";
    print_times(kernel_trials);
    std::cout << ",\"median_seconds\":" << median(trials)
              << ",\"kernel_median_seconds\":" << median(kernel_trials)
              << ",\"timing_scope\":\"batched H2D + kernel + synchronization + D2H; excludes parsing, allocation and validation\""
              << ",\"device\":{\"index\":" << device << ",\"name\":";
    nauty_cuda::json_string(properties.name);
    std::cout << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
              << "\",\"driver_version\":" << driver_version << ",\"runtime_version\":" << runtime_version
              << "},\"threads_per_block\":" << threads << "}\n";
    if (!std::cout) throw std::runtime_error("failed to write benchmark output");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--repeats N] [--warmup N] [--loops N] [--batch-size N] [--max-nodes N] < graphs.g6\n"
                      << "Reports CUDA event kernel timing and host transfer-inclusive timing as JSON.\n"
                      << "Requires an NVIDIA GPU; incomplete or unavailable runs produce no timing report.\n";
            return 0;
        }
        const Options options = parse_options(argc, argv);
        const std::vector<Graph> graphs = read_graphs();
        return benchmark(options, graphs);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
}
