#include "cli_common.hpp"

#include <nauty.h>
// nauty's generated C header can define TLS_ATTR as _Thread_local, which
// GCC rejects in C++. Keep the library's TLS configuration and use the
// equivalent C++ spelling before its dependent headers declare TLS objects.
#if HAVE_TLS
#undef TLS_ATTR
#define TLS_ATTR thread_local
#endif
#include <naurng.h>
#include <traces.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace {

using nauty_cuda::Graph;
using nauty_cuda::Result;
constexpr int capacity = nauty_cuda::max_vertices;
constexpr int dense_words = capacity * SETWORDSNEEDED(capacity);

struct Options {
    std::string backend = "prototype";
    std::uint64_t repeats = 5, warmup = 1, loops = 1, max_nodes = 1000000;
};

struct Counts {
    std::uint64_t completed = 0, incomplete = 0, errors = 0;
    std::uint64_t nodes = 0, max_nodes = 0;
    bool overflow = false;

    void add(bool ok, bool limited, std::uint64_t count) {
        completed += ok;
        incomplete += limited;
        errors += !ok && !limited;
        if (count > std::numeric_limits<std::uint64_t>::max() - nodes) overflow = true;
        else nodes += count;
        max_nodes = std::max(max_nodes, count);
    }
};

struct Trial {
    double seconds = 0;
    Counts counts;
    std::uint64_t checksum = 0;
};

struct DenseRecord {
    std::array<graph, dense_words> input{}, canonical{};
    int n = 0, m = 1;
};

struct SparseBuffer {
    std::vector<std::size_t> offsets;
    std::vector<int> degrees, edges;
    sparsegraph sg{};

    // Allocate from the actual graph size before timing, and never resize
    // afterwards. These capacities prevent Traces from reallocating outputs
    // without imposing maximum-order storage on every small graph.
    void initialize(int n, std::size_t nde) {
        offsets.resize(static_cast<std::size_t>(n));
        degrees.resize(static_cast<std::size_t>(n));
        edges.resize(nde);
        sg = {nde, offsets.data(), n, degrees.data(), edges.data(), nullptr,
              offsets.size(), degrees.size(), edges.size(), 0};
    }
};

struct SparseRecord {
    SparseBuffer input, canonical;
};

// Each pass must remain in the timed region, even when the inline prototype
// overwrites identical results. The checksum is deliberately outside timing.
inline void escape_results(const void *results) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(results) : "memory");
#else
    (void)results;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

void hash_word(std::uint64_t &hash, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        hash ^= (value >> shift) & 255;
        hash *= UINT64_C(1099511628211);
    }
}

std::string hex_checksum(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

void json_string(const std::string &value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else std::cout << c;
    }
    std::cout << '"';
}

class Benchmark {
    const std::vector<Graph> &graphs_;
    const Options &options_;
    std::vector<Result> results_;
    std::vector<DenseRecord> dense_;
    std::vector<SparseRecord> sparse_;

public:
    Benchmark(const std::vector<Graph> &graphs, const Options &options)
        : graphs_(graphs), options_(options) {
        if (options.backend == "prototype") results_.resize(graphs.size());
        else {
            nauty_check(WORDSIZE, SETWORDSNEEDED(capacity), capacity, NAUTYVERSIONID);
            if (options.backend == "nauty") {
                dense_.resize(graphs.size());
                for (std::size_t i = 0; i < graphs.size(); ++i) {
                    auto &record = dense_[i];
                    record.n = static_cast<int>(graphs[i].n);
                    record.m = std::max(1, SETWORDSNEEDED(record.n));
                    for (int u = 0; u < record.n; ++u)
                        for (int v = u + 1; v < record.n; ++v)
                            if ((graphs[i].rows[u] >> v) & 1u)
                                ADDONEEDGE(record.input.data(), u, v, record.m);
                }
            } else {
                sparse_.resize(graphs.size());
                for (std::size_t i = 0; i < graphs.size(); ++i) {
                    auto &record = sparse_[i];
                    std::size_t nde = 0;
                    for (unsigned u = 0; u < graphs[i].n; ++u)
                        nde += nauty_cuda::detail::popcount(graphs[i].rows[u]);
                    record.input.initialize(static_cast<int>(graphs[i].n), nde);
                    record.canonical.initialize(static_cast<int>(graphs[i].n), nde);
                    std::size_t offset = 0;
                    for (unsigned u = 0; u < graphs[i].n; ++u) {
                        record.input.offsets[u] = offset;
                        for (unsigned v = 0; v < graphs[i].n; ++v) {
                            if ((graphs[i].rows[u] >> v) & 1u) {
                                record.input.edges[offset++] = static_cast<int>(v);
                                ++record.input.degrees[u];
                            }
                        }
                    }
                }
            }
        }
    }

    Trial run() {
        Trial trial;
        // Reset the library RNG before each whole trial, outside timing.
        if (options_.backend != "prototype") ran_init(1);
        DEFAULTOPTIONS_GRAPH(dense_options);
        DEFAULTOPTIONS_TRACES(sparse_options);
        dense_options.getcanon = TRUE;
        sparse_options.getcanon = TRUE;
        // defaultptn=TRUE makes the libraries initialize lab and ptn inside
        // every call. Neither lab nor ptn from a previous search is reused.
        std::array<int, capacity> lab{}, ptn{}, orbits{};
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint64_t pass = 0; pass < options_.loops; ++pass) {
            if (options_.backend == "prototype") {
                for (std::size_t i = 0; i < graphs_.size(); ++i) {
                    auto &result = results_[i];
                    result = nauty_cuda::canonicalize(graphs_[i], options_.max_nodes);
                    trial.counts.add(result.status == nauty_cuda::Status::ok,
                                     result.status == nauty_cuda::Status::node_limit,
                                     result.nodes);
                }
                escape_results(results_.data());
            } else if (options_.backend == "nauty") {
                for (auto &record : dense_) {
                    statsblk stats{};
                    densenauty(record.input.data(), lab.data(), ptn.data(), orbits.data(),
                               &dense_options, &stats, record.m, record.n,
                               record.canonical.data());
                    trial.counts.add(stats.errstatus == 0, false, stats.numnodes);
                }
                escape_results(dense_.data());
            } else {
                for (auto &record : sparse_) {
                    // Traces accesses ptn[n-1] and divides by n, so its API
                    // cannot accept order zero. The empty canonical graph
                    // is already initialized and requires no search nodes.
                    TracesStats stats{};
                    if (record.input.sg.nv)
                        Traces(&record.input.sg, lab.data(), ptn.data(), orbits.data(),
                               &sparse_options, &stats, &record.canonical.sg);
                    trial.counts.add(stats.errstatus == 0, false, stats.numnodes);
                }
                escape_results(sparse_.data());
            }
        }
        const auto end = std::chrono::steady_clock::now();
        trial.seconds = std::chrono::duration<double>(end - begin).count();

        // Hash canonical adjacency rows, not labelling permutations or sparse
        // neighbour-list order. Different algorithms may choose different
        // canonical forms, so checksums are compared only within one backend.
        trial.checksum = UINT64_C(14695981039346656037);
        if (!trial.counts.errors && !trial.counts.incomplete) {
            for (std::size_t i = 0; i < graphs_.size(); ++i) {
                hash_word(trial.checksum, graphs_[i].n);
                for (unsigned u = 0; u < graphs_[i].n; ++u) {
                    std::uint32_t row = 0;
                    if (options_.backend == "prototype") row = results_[i].rows[u];
                    else if (options_.backend == "nauty") {
                        auto &record = dense_[i];
                        const auto *native_row = GRAPHROW(record.canonical.data(), u, record.m);
                        for (unsigned v = 0; v < graphs_[i].n; ++v)
                            if (ISELEMENT(native_row, v)) row |= std::uint32_t{1} << v;
                    } else {
                        const auto &sg = sparse_[i].canonical.sg;
                        for (int j = 0; j < sg.d[u]; ++j)
                            row |= std::uint32_t{1} << sg.e[sg.v[u] + j];
                    }
                    hash_word(trial.checksum, row);
                }
            }
        }
        return trial;
    }
};

}  // namespace

int main(int argc, char **argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--help") {
                std::cout << "Usage: " << argv[0]
                          << " [--backend prototype|nauty|traces] [--repeats N]"
                             " [--warmup N] [--loops N] [--max-nodes N] < graphs.g6\n"
                             "Single-thread CPU benchmark for simple graphs of order 0..32.\n"
                             "Defaults: prototype, 5 repeats, 1 warmup, 1 loop, 1000000 nodes.\n"
                             "Times include all loops; operations = graphs * loops.\n"
                             "The per-graph node limit applies only to the prototype.\n"
                             "Exit 0: complete; 2: incomplete; 3: benchmark/input error.\n";
                return 0;
            }
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + argument);
            if (argument == "--backend") options.backend = argv[++i];
            else if (argument == "--repeats") options.repeats = nauty_cuda::number(argv[++i]);
            else if (argument == "--warmup") options.warmup = nauty_cuda::number(argv[++i]);
            else if (argument == "--loops") options.loops = nauty_cuda::number(argv[++i]);
            else if (argument == "--max-nodes") options.max_nodes = nauty_cuda::number(argv[++i]);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (options.backend != "prototype" && options.backend != "nauty" && options.backend != "traces")
            throw std::invalid_argument("backend must be prototype, nauty, or traces");
        if (!options.repeats || !options.loops)
            throw std::invalid_argument("repeats and loops must be positive");
        if (options.repeats > std::vector<Trial>().max_size())
            throw std::invalid_argument("repeat count is out of range");

        std::vector<Graph> graphs;
        std::string line;
        std::uint64_t line_number = 0;
        while (std::getline(std::cin, line)) {
            ++line_number;
            if (line.empty() || line == ">>graph6<<" || line == ">>graph6<<\r") continue;
            Graph input{};
            std::string error;
            if (!nauty_cuda::decode_graph6(line, input, error))
                throw std::invalid_argument("line " + std::to_string(line_number) + ": " + error);
            graphs.push_back(input);
        }
        if (std::cin.bad()) throw std::runtime_error("failed to read input");
        if (graphs.empty()) throw std::invalid_argument("input contains no graphs");
        if (options.loops > std::numeric_limits<std::uint64_t>::max() / graphs.size())
            throw std::invalid_argument("graphs * loops is out of range");
        const std::uint64_t operations = graphs.size() * options.loops;

        Benchmark benchmark(graphs, options);
        std::vector<Trial> trials;
        trials.reserve(static_cast<std::size_t>(options.repeats));
        bool incomplete = false, error = false, have_checksum = false;
        std::uint64_t checksum = 0;
        Trial last;
        auto check = [&](const Trial &trial) {
            incomplete |= trial.counts.incomplete != 0;
            error |= trial.counts.errors != 0 || trial.counts.overflow;
            if (trial.counts.completed == operations && !error && !incomplete) {
                if (have_checksum && checksum != trial.checksum) error = true;
                checksum = trial.checksum;
                have_checksum = true;
            }
            last = trial;
        };
        for (std::uint64_t i = 0; i < options.warmup && !error && !incomplete; ++i)
            check(benchmark.run());
        for (std::uint64_t i = 0; i < options.repeats && !error && !incomplete; ++i) {
            trials.push_back(benchmark.run());
            check(trials.back());
        }
        const bool complete = !error && !incomplete;
        std::vector<double> seconds;
        for (const auto &trial : trials) seconds.push_back(trial.seconds);
        std::sort(seconds.begin(), seconds.end());
        double median = 0;
        if (!seconds.empty()) {
            const auto middle = seconds.size() / 2;
            median = seconds[middle];
            if (seconds.size() % 2 == 0) median = (median + seconds[middle - 1]) / 2;
        }
        std::cout << std::setprecision(17) << "{\"status\":\""
                  << (error ? "benchmark_error" : incomplete ? "incomplete" : "ok")
                  << "\",\"backend\":\"" << options.backend
                  << "\",\"device\":\"cpu\",\"threads\":1,\"library_version\":";
        json_string(options.backend == "prototype" ? "prototype 0.1.0" : NAUTYVERSION);
        std::cout << ",\"graphs\":" << graphs.size() << ",\"loops\":" << options.loops
                  << ",\"operations\":" << operations
                  << ",\"completed\":" << last.counts.completed / options.loops
                  << ",\"completed_operations\":" << last.counts.completed
                  << ",\"incomplete_operations\":" << last.counts.incomplete
                  << ",\"error_operations\":" << last.counts.errors
                  << ",\"repeats\":" << options.repeats << ",\"warmup\":" << options.warmup
                  << ",\"max_nodes\":" << options.max_nodes
                  << ",\"node_limit_applied\":" << (options.backend == "prototype" ? "true" : "false")
                  << ",\"nodes\":" << last.counts.nodes
                  << ",\"max_nodes_observed\":" << last.counts.max_nodes
                  << ",\"trials_seconds\":[";
        for (std::size_t i = 0; i < trials.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << trials[i].seconds;
        }
        std::cout << "],\"median_seconds\":";
        if (complete && median > 0) std::cout << median;
        else std::cout << "null";
        std::cout << ",\"graphs_per_second\":";
        if (complete && median > 0) std::cout << operations / median;
        else std::cout << "null";
        std::cout << ",\"checksum\":";
        if (complete) json_string(hex_checksum(checksum));
        else std::cout << "null";
        if (error) std::cout << ",\"error\":\"library error, node-count overflow, or inconsistent canonical output\"";
        else if (incomplete) std::cout << ",\"error\":\"prototype node limit reached; timings are not valid comparisons\"";
        std::cout << "}\n";
        if (!std::cout) throw std::runtime_error("failed to write output");
        return error ? 3 : incomplete ? 2 : 0;
    } catch (const std::exception &error) {
        std::cout << "{\"status\":\"benchmark_error\",\"error\":";
        json_string(error.what());
        std::cout << "}\n";
        std::cerr << error.what() << '\n';
        return 3;
    }
}
