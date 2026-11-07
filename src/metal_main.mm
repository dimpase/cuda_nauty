#include "cli_common.hpp"
#include "metal_backend.hpp"

#include <memory>

int main(int argc, char** argv)
{
    std::unique_ptr<nauty_cuda::MetalRunner> runner;
    const auto run = [&](const nauty_cuda::Graph* graphs, nauty_cuda::Result* results,
                         std::size_t count, std::uint64_t max_nodes) {
        if (count == 0) return;
        // run_cli flushes full batches before its final partial batch, so the
        // first nonempty call establishes sufficient capacity for this stream.
        if (!runner) runner = std::make_unique<nauty_cuda::MetalRunner>(count);
        runner->run(graphs, results, count, max_nodes);
    };
    return nauty_cuda::run_cli(argc, argv, run);
}
