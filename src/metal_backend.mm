#include "metal_backend.hpp"
#include "metal_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace nauty_cuda {
namespace {

struct MetalParams {
    std::uint64_t max_nodes;
    std::uint32_t count;
    std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<Graph> && std::is_trivially_copyable_v<Graph>);
static_assert(std::is_standard_layout_v<Result> && std::is_trivially_copyable_v<Result>);
static_assert(sizeof(Graph) == 132 && alignof(Graph) == 4);
static_assert(sizeof(Result) == 272 && alignof(Result) == 8);
static_assert(offsetof(Result, nodes) == 264);
static_assert(sizeof(MetalParams) == 16 && alignof(MetalParams) == 8);
static_assert(offsetof(MetalParams, count) == 8);

std::runtime_error metal_error(const char* operation, NSError* error)
{
    const char* detail = error.localizedDescription.UTF8String;
    return std::runtime_error(std::string(operation) + ": " +
                              (detail != nullptr ? detail : "no Metal error details available"));
}

} // namespace

struct MetalRunner::Impl {
    std::size_t capacity;
    std::size_t group_size = 0;
    std::string name;
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> graphs = nil;
    id<MTLBuffer> results = nil;
    id<MTLBuffer> params = nil;

    explicit Impl(std::size_t requested_capacity) : capacity(requested_capacity)
    {
        if (capacity == 0 || capacity > std::numeric_limits<std::uint32_t>::max() ||
            capacity > std::numeric_limits<std::size_t>::max() / sizeof(Graph) ||
            capacity > std::numeric_limits<std::size_t>::max() / sizeof(Result)) {
            throw std::invalid_argument("Metal batch capacity is out of range");
        }
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("Metal unavailable: no Metal device found");
            const char* device_name = device.name.UTF8String;
            name = device_name != nullptr ? device_name : "unnamed Metal device";
            if (capacity * sizeof(Graph) > device.maxBufferLength ||
                capacity * sizeof(Result) > device.maxBufferLength) {
                throw std::invalid_argument("Metal batch capacity exceeds the device buffer limit");
            }

            NSString* source = [[NSString alloc] initWithUTF8String:metal_source];
            if (source == nil) throw std::runtime_error("embedded Metal source is not valid UTF-8");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
            if (library == nil) throw metal_error("compile canonicalization Metal library", error);
            id<MTLFunction> function = [library newFunctionWithName:@"canonicalize_batch"];
            if (function == nil) throw std::runtime_error("Metal library has no canonicalize_batch kernel");
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (pipeline == nil) throw metal_error("create Metal compute pipeline", error);

            group_size = std::min({std::size_t{128},
                static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup),
                static_cast<std::size_t>(device.maxThreadsPerThreadgroup.width)});
            if (group_size == 0) throw std::runtime_error("Metal pipeline supports no compute threads");
            const std::size_t width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
            if (width != 0 && group_size >= width) group_size -= group_size % width;

            queue = [device newCommandQueue];
            graphs = [device newBufferWithLength:capacity * sizeof(Graph)
                                        options:MTLResourceStorageModeShared];
            results = [device newBufferWithLength:capacity * sizeof(Result)
                                         options:MTLResourceStorageModeShared];
            params = [device newBufferWithLength:sizeof(MetalParams)
                                        options:MTLResourceStorageModeShared];
            if (queue == nil) throw std::runtime_error("failed to allocate Metal command queue");
            if (graphs == nil || results == nil || params == nil ||
                graphs.contents == nullptr || results.contents == nullptr || params.contents == nullptr) {
                throw std::runtime_error("failed to allocate shared Metal buffers");
            }
        }
    }

    MetalTiming run(const Graph* input, Result* output, std::size_t count, std::uint64_t max_nodes)
    {
        if (count == 0) return {0, 0};
        if (count > capacity) throw std::invalid_argument("Metal batch exceeds preallocated capacity");
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("nonempty Metal batch requires input and output buffers");
        }
        const auto begin = std::chrono::steady_clock::now();
        double gpu_seconds = 0;
        @autoreleasepool {
            // Command buffers and encoders are single-use; include their
            // mandatory creation in the batch API cost. Persistent queue,
            // pipeline and data-buffer allocation stays outside this timer.
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (command == nil) throw std::runtime_error("failed to allocate Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil) throw std::runtime_error("failed to allocate Metal compute encoder");
            const MetalParams arguments{max_nodes, static_cast<std::uint32_t>(count), 0};
            const std::size_t groups = 1 + (count - 1) / group_size;

            std::memcpy(graphs.contents, input, count * sizeof(Graph));
            std::memcpy(params.contents, &arguments, sizeof(arguments));
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:graphs offset:0 atIndex:0];
            [encoder setBuffer:results offset:0 atIndex:1];
            [encoder setBuffer:params offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw metal_error("Metal canonicalization command failed", command.error);
            }
            std::memcpy(output, results.contents, count * sizeof(Result));

            const double gpu_start = command.GPUStartTime;
            const double gpu_end = command.GPUEndTime;
            if (!std::isfinite(gpu_start) || !std::isfinite(gpu_end) ||
                gpu_start <= 0 || gpu_end <= gpu_start) {
                throw std::runtime_error("Metal command completed without usable GPU timestamps");
            }
            gpu_seconds = gpu_end - gpu_start;
        }
        // Include release of the per-dispatch command objects as part of the
        // synchronous API cost, in addition to their creation and submission.
        const auto end = std::chrono::steady_clock::now();
        return {std::chrono::duration<double>(end - begin).count(), gpu_seconds};
    }
};

MetalRunner::MetalRunner(std::size_t capacity) : impl_(std::make_unique<Impl>(capacity)) {}
MetalRunner::~MetalRunner() = default;

MetalTiming MetalRunner::run(const Graph* graphs, Result* results,
                             std::size_t count, std::uint64_t max_nodes)
{
    return impl_->run(graphs, results, count, max_nodes);
}

const std::string& MetalRunner::device_name() const { return impl_->name; }
std::size_t MetalRunner::threads_per_group() const { return impl_->group_size; }

} // namespace nauty_cuda
