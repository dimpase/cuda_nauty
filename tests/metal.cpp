#include "metal_backend.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace nauty_cuda;

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void compare(const Result &cpu, const Result &gpu) {
    check(cpu.status == gpu.status && cpu.n == gpu.n && cpu.nodes == gpu.nodes,
          "Metal status/order/node count differs from CPU");
    for (unsigned i = 0; i < max_vertices; ++i) {
        check(cpu.rows[i] == gpu.rows[i] && cpu.order[i] == gpu.order[i],
              "Metal canonical rows or witness differ from CPU");
    }
}

int main() {
    try {
        std::vector<Graph> fixtures(8);
        fixtures[1].n = 33;  // Reject before any out-of-bounds access.
        fixtures[2].n = 2;
        fixtures[2].rows[0] = 1;  // Self-loop.
        fixtures[3].n = 2;
        fixtures[3].rows[0] = 2;  // Asymmetric adjacency.
        fixtures[4].n = 2;
        fixtures[4].rows[0] = std::uint32_t{1} << 31;  // Vertex outside n.
        fixtures[5].n = 32;  // Empty graph, maximal search depth.
        fixtures[6].n = 32;
        for (unsigned i = 0; i < 32; ++i) fixtures[6].rows[i] = ~(std::uint32_t{1} << i);
        fixtures[7].n = 32;
        fixtures[7].rows[0] = std::uint32_t{1} << 31;
        fixtures[7].rows[31] = 1;

        MetalRunner runner(129);
        std::vector<Result> output(fixtures.size());
        const std::array<std::uint64_t, 7> budgets{
            0, 1, 31, 32, 1000, std::uint64_t{1} << 32,
            std::numeric_limits<std::uint64_t>::max()};
        for (auto budget : budgets) {
            runner.run(fixtures.data(), output.data(), fixtures.size(), budget);
            for (std::size_t i = 0; i < fixtures.size(); ++i)
                compare(canonicalize(fixtures[i], budget), output[i]);
        }
        // Reuse one runner/buffer allocation across both whole and partial
        // threadgroups. Interleave invalid input to catch stale output slots.
        for (std::size_t count : {1u, 31u, 32u, 33u, 127u, 128u, 129u}) {
            std::vector<Graph> input(count);
            output.resize(count);
            for (std::size_t i = 0; i < count; ++i) input[i] = fixtures[i % fixtures.size()];
            runner.run(input.data(), output.data(), count, 1000);
            for (std::size_t i = 0; i < count; ++i)
                compare(canonicalize(input[i], 1000), output[i]);
        }
        const auto empty = runner.run(nullptr, nullptr, 0, 0);
        check(empty.host_seconds == 0 && empty.gpu_seconds == 0, "empty batch was dispatched");
        for (unsigned test = 0; test < 3; ++test) {
            bool rejected = false;
            try {
                runner.run(test == 0 ? nullptr : fixtures.data(),
                           test == 1 ? nullptr : output.data(), test == 2 ? 130 : 1, 1000);
            } catch (const std::exception &) {
                rejected = true;
            }
            check(rejected, "invalid Metal batch arguments were accepted");
        }
        std::cout << runner.device_name()
                  << ": Metal ABI, invalid-input, 64-bit budget, depth/bit31 and partial-batch checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
