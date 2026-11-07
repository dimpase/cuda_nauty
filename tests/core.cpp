#include "canonical.hpp"

#include <iostream>
#include <stdexcept>

using namespace nauty_cuda;

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void incomplete(const Result &result) {
    check(result.status == Status::node_limit, "expected incomplete result");
    for (int i = 0; i < max_vertices; ++i) {
        check(result.rows[i] == 0 && result.order[i] == 0,
              "incomplete result exposed a provisional certificate");
    }
}

int main() {
    try {
        Graph graph{};
        check(canonicalize(graph, 1).status == Status::ok, "empty graph rejected");
        incomplete(canonicalize(graph, 0));

        graph.n = 33;
        check(canonicalize(graph, 100).status == Status::invalid_input, "oversized graph accepted");
        graph.n = 2;
        graph.rows[0] = 1;
        check(canonicalize(graph, 100).status == Status::invalid_input, "loop accepted");
        graph.rows[0] = 2;
        check(canonicalize(graph, 100).status == Status::invalid_input, "asymmetric graph accepted");
        graph.rows[0] = 1u << 31;
        check(canonicalize(graph, 100).status == Status::invalid_input, "out-of-range vertex accepted");

        graph = Graph{};
        graph.n = 32;
        auto empty = canonicalize(graph, 32);
        check(empty.status == Status::ok && empty.nodes == 32, "32-vertex twin pruning failed");
        incomplete(canonicalize(graph, 31));
        for (unsigned v = 0; v < graph.n; ++v) graph.rows[v] = ~(1u << v);
        auto complete = canonicalize(graph, 32);
        check(complete.status == Status::ok && complete.nodes == 32, "complete graph depth boundary failed");
        for (unsigned v = 0; v < graph.n; ++v) {
            check(complete.rows[v] == ~(1u << v), "complete graph certificate failed");
        }

        graph = Graph{};
        graph.n = 32;
        graph.rows[0] = 1u << 31;
        graph.rows[31] = 1;
        auto high_bit = canonicalize(graph, 1000);
        check(high_bit.status == Status::ok, "vertex31 boundary failed");
        bool seen[max_vertices]{};
        for (unsigned v = 0; v < graph.n; ++v) {
            check(high_bit.order[v] < graph.n && !seen[high_bit.order[v]], "invalid witness permutation");
            seen[high_bit.order[v]] = true;
            for (unsigned w = 0; w < graph.n; ++w) {
                check(((high_bit.rows[v] >> w) & 1) ==
                      ((graph.rows[high_bit.order[v]] >> high_bit.order[w]) & 1),
                      "witness does not reproduce input adjacency");
            }
        }
        std::cout << "Core invalid-input, budget, twin and bit31/depth boundary tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
