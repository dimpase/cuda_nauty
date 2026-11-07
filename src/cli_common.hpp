#pragma once

#include "canonical.hpp"

#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nauty_cuda {

inline bool decode_graph6(std::string text, Graph &graph, std::string &error) {
    if (text.rfind(">>graph6<<", 0) == 0) text.erase(0, 10);
    if (!text.empty() && text.back() == '\r') text.pop_back();
    graph = Graph{};
    auto invalid = [&](const char *reason) {
        error = reason;
        graph.n = max_vertices + 1;  // Reject malformed records in either backend.
        return false;
    };
    if (text.empty()) return invalid("empty graph6 record");
    for (char raw : text) {
        auto c = static_cast<unsigned char>(raw);
        if (c < 63 || c > 126) return invalid("invalid graph6 character");
    }
    graph.n = static_cast<unsigned char>(text[0]) - 63;
    if (graph.n > max_vertices) return invalid("only graph orders 0 through 32 are supported");
    const unsigned bits = graph.n * (graph.n ? graph.n - 1 : 0) / 2;
    if (text.size() != 1 + (bits + 5) / 6) return invalid("incorrect graph6 record length");
    unsigned index = 0;
    for (unsigned v = 1; v < graph.n; ++v) {
        for (unsigned u = 0; u < v; ++u, ++index) {
            if (((text[1 + index / 6] - 63) >> (5 - index % 6)) & 1) {
                graph.rows[v] |= std::uint32_t{1} << u;
                graph.rows[u] |= std::uint32_t{1} << v;
            }
        }
    }
    if (bits % 6 && (static_cast<unsigned>(text.back() - 63) & ((1u << (6 - bits % 6)) - 1))) {
        return invalid("nonzero graph6 padding bits");
    }
    return true;
}

inline std::string encode_graph6(const Result &result) {
    std::string text(1, static_cast<char>(result.n + 63));
    unsigned value = 0, count = 0;
    for (unsigned v = 1; v < result.n; ++v) {
        for (unsigned u = 0; u < v; ++u) {
            value = (value << 1) | ((result.rows[v] >> u) & 1);
            if (++count == 6) {
                text.push_back(static_cast<char>(value + 63));
                count = value = 0;
            }
        }
    }
    if (count) text.push_back(static_cast<char>((value << (6 - count)) + 63));
    return text;
}

inline void json_string(const std::string &value) {
    std::cout << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\';
        std::cout << c;
    }
    std::cout << '"';
}

inline std::uint64_t number(const char *text) {
    std::string value(text);
    std::uint64_t parsed = 0;
    auto conversion = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (conversion.ec != std::errc{} || conversion.ptr != value.data() + value.size()) {
        throw std::invalid_argument("expected a nonnegative integer");
    }
    return parsed;
}

// Run a streaming graph6 -> JSONL frontend with bounded batches. The callback
// must produce one Result per input Graph, preserving input order.
template <class Run>
int run_cli(int argc, char **argv, Run run) {
    try {
        std::uint64_t max_nodes = 1000000;
        std::size_t batch_size = 4096;
        for (int i = 1; i < argc; ++i) {
            std::string argument(argv[i]);
            if (argument == "--help") {
                std::cout << "Usage: " << argv[0] << " [--max-nodes N] [--batch-size N] < graphs.g6\n"
                          << "Exact uncoloured simple graphs of order 0..32. Output: JSONL.\n"
                          << "Node budget is per graph; zero permits no search.\n"
                          << "Exit 0: all complete; 2: invalid/incomplete records; 3: runtime error.\n";
                return 0;
            }
            if ((argument != "--max-nodes" && argument != "--batch-size") || i + 1 == argc) {
                throw std::invalid_argument("unknown or incomplete option: " + argument);
            }
            auto value = number(argv[++i]);
            if (argument == "--max-nodes") max_nodes = value;
            else {
                if (!value || value > std::numeric_limits<std::size_t>::max() / sizeof(Result)) {
                    throw std::invalid_argument("batch size is out of range");
                }
                batch_size = static_cast<std::size_t>(value);
            }
        }
        std::vector<Graph> graphs;
        std::vector<std::string> errors;
        int exit_status = 0;
        auto flush = [&]() {
            if (graphs.empty()) return;
            std::vector<Result> results(graphs.size());
            run(graphs.data(), results.data(), graphs.size(), max_nodes);
            for (std::size_t i = 0; i < results.size(); ++i) {
                const auto &result = results[i];
                bool complete = result.status == Status::ok;
                const char *status = complete ? "ok" :
                    result.status == Status::node_limit ? "node_limit" : "invalid_input";
                std::cout << "{\"status\":\"" << status << "\",\"n\":" << result.n
                          << ",\"nodes\":" << result.nodes << ",\"order\":[";
                if (complete) {
                    for (unsigned j = 0; j < result.n; ++j) {
                        if (j) std::cout << ',';
                        std::cout << result.order[j];
                    }
                } else exit_status = 2;
                std::cout << "],\"graph6\":";
                if (complete) json_string(encode_graph6(result));
                else std::cout << "null";
                if (!errors[i].empty()) {
                    std::cout << ",\"error\":";
                    json_string(errors[i]);
                }
                std::cout << "}\n";
            }
            if (!std::cout) throw std::runtime_error("failed to write output");
            graphs.clear();
            errors.clear();
        };
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty() || line == ">>graph6<<" || line == ">>graph6<<\r") continue;
            Graph graph{};
            std::string error;
            decode_graph6(line, graph, error);
            graphs.push_back(graph);
            errors.push_back(error);
            if (graphs.size() == batch_size) flush();
        }
        if (std::cin.bad()) throw std::runtime_error("failed to read input");
        flush();
        return exit_status;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
}

}  // namespace nauty_cuda
