#include "cli_common.hpp"

int main(int argc, char **argv) {
    return nauty_cuda::run_cli(argc, argv,
        [](const nauty_cuda::Graph *graphs, nauty_cuda::Result *results,
           std::size_t count, std::uint64_t max_nodes) {
            for (std::size_t i = 0; i < count; ++i) {
                results[i] = nauty_cuda::canonicalize(graphs[i], max_nodes);
            }
        });
}
