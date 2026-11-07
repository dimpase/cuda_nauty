#include "cli_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

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

    ~DeviceBuffer()
    {
        // Cleanup must not hide an earlier launch, synchronization, or copy error.
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

__global__ void canonicalize_batch(const nauty_cuda::Graph* graphs,
                                   nauty_cuda::Result* results,
                                   std::size_t count, std::uint64_t max_nodes)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        results[index] = nauty_cuda::canonicalize(graphs[index], max_nodes);
    }
}

void run_cuda(const nauty_cuda::Graph* graphs, nauty_cuda::Result* results,
              std::size_t count, std::uint64_t max_nodes)
{
    if (count == 0) {
        return;
    }
    if (graphs == nullptr || results == nullptr) {
        throw std::runtime_error("nonempty CUDA batch requires input and output buffers");
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable GPU is available");
    }
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    cudaFuncAttributes kernel{};
    check_cuda(cudaFuncGetAttributes(&kernel, canonicalize_batch), "cudaFuncGetAttributes");
    const int threads = std::min({128, properties.maxThreadsPerBlock,
                                  properties.maxThreadsDim[0], kernel.maxThreadsPerBlock});
    if (threads <= 0 || properties.maxGridSize[0] <= 0) {
        throw std::runtime_error("CUDA device does not support the required launch dimensions");
    }
    const std::size_t blocks = 1 + (count - 1) / static_cast<std::size_t>(threads);
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("CUDA batch exceeds the device's maximum grid size");
    }

    constexpr std::size_t bytes_per_graph = sizeof(nauty_cuda::Graph) + sizeof(nauty_cuda::Result);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_graph) {
        throw std::runtime_error("CUDA batch allocation size overflow");
    }
    std::size_t available_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&available_bytes, &total_bytes), "cudaMemGetInfo");
    if (count * bytes_per_graph > available_bytes) {
        throw std::runtime_error("CUDA batch input and output exceed available device memory");
    }

    DeviceBuffer<nauty_cuda::Graph> device_graphs(count);
    DeviceBuffer<nauty_cuda::Result> device_results(count);
    check_cuda(cudaMemcpy(device_graphs.get(), graphs, count * sizeof(nauty_cuda::Graph),
                          cudaMemcpyHostToDevice), "copy graphs to CUDA device");
    canonicalize_batch<<<static_cast<unsigned int>(blocks), threads>>>(
        device_graphs.get(), device_results.get(), count, max_nodes);
    check_cuda(cudaGetLastError(), "launch canonicalize_batch");
    check_cuda(cudaDeviceSynchronize(), "synchronize canonicalize_batch");
    check_cuda(cudaMemcpy(results, device_results.get(), count * sizeof(nauty_cuda::Result),
                          cudaMemcpyDeviceToHost), "copy canonical results from CUDA device");
}

} // namespace

int main(int argc, char** argv)
{
    return nauty_cuda::run_cli(argc, argv, run_cuda);
}
