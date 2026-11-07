#pragma once

#include "canonical.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nauty_cuda {

struct MetalTiming {
    double host_seconds;
    double gpu_seconds;
};

// A synchronous, reusable runner. Each instance owns its Metal buffers and is
// not safe for simultaneous calls from multiple threads.
class MetalRunner {
public:
    explicit MetalRunner(std::size_t capacity);
    ~MetalRunner();

    MetalRunner(const MetalRunner&) = delete;
    MetalRunner& operator=(const MetalRunner&) = delete;

    // A zero count is a no-op. Nonempty batches must fit the constructor's
    // capacity. Result statuses are preserved for the caller to interpret.
    MetalTiming run(const Graph* graphs, Result* results,
                    std::size_t count, std::uint64_t max_nodes);
    const std::string& device_name() const;
    std::size_t threads_per_group() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nauty_cuda
