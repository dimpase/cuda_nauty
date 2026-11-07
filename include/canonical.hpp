#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define NAUTY_CUDA_THREAD thread
#define NAUTY_CUDA_CONSTANT constant
#else
#include <cstdint>
#define NAUTY_CUDA_THREAD
#define NAUTY_CUDA_CONSTANT
#endif

#ifdef __CUDACC__
#define NAUTY_CUDA_HD __host__ __device__
#else
#define NAUTY_CUDA_HD
#endif

namespace nauty_cuda {

#ifdef __METAL_VERSION__
using u8 = unsigned char;
using u32 = unsigned int;
using u64 = unsigned long;
#else
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
#endif

NAUTY_CUDA_CONSTANT constexpr int max_vertices = 32;

struct Graph {
    u32 n;
    u32 rows[max_vertices];
};

enum class Status : u32 { ok, invalid_input, node_limit };

struct Result {
    Status status;
    u32 n;
    // Canonical position -> vertex number in the input graph.
    u32 order[max_vertices];
    u32 rows[max_vertices];
    u64 nodes;
};

namespace detail {

NAUTY_CUDA_HD inline u32 bit(unsigned v) {
    return u32{1} << v;
}

NAUTY_CUDA_HD inline unsigned popcount(u32 word) {
#ifdef __METAL_VERSION__
    return metal::popcount(word);
#elif defined(__CUDA_ARCH__)
    return __popc(word);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcount(word));
#else
    unsigned count = 0;
    while (word) {
        word &= word - 1;
        ++count;
    }
    return count;
#endif
}

// Call only with a nonzero word.
NAUTY_CUDA_HD inline unsigned first_vertex(u32 word) {
#ifdef __METAL_VERSION__
    return metal::ctz(word);
#elif defined(__CUDA_ARCH__)
    return static_cast<unsigned>(__ffs(static_cast<int>(word)) - 1);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctz(word));
#else
    unsigned v = 0;
    while (!(word & 1)) {
        word >>= 1;
        ++v;
    }
    return v;
#endif
}

struct Partition {
    u32 cells[max_vertices];
    unsigned count;
};

NAUTY_CUDA_HD inline int compare_signature(
    NAUTY_CUDA_THREAD const u8 *a, NAUTY_CUDA_THREAD const u8 *b, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

// Preserve ordered parent cells; order their subdivisions by exact neighbour
// count tuples. This rule commutes with every relabelling of the input graph.
NAUTY_CUDA_HD inline void refine(
    NAUTY_CUDA_THREAD const Graph &graph, NAUTY_CUDA_THREAD Partition &partition) {
    for (;;) {
        u8 signatures[max_vertices][max_vertices];
        for (unsigned v = 0; v < graph.n; ++v) {
            for (unsigned c = 0; c < partition.count; ++c) {
                signatures[v][c] = static_cast<u8>(
                    popcount(graph.rows[v] & partition.cells[c]));
            }
        }
        Partition next{};
        for (unsigned c = 0; c < partition.count; ++c) {
            auto remaining = partition.cells[c];
            while (remaining) {
                unsigned least = first_vertex(remaining);
                for (auto scan = remaining; scan; scan &= scan - 1) {
                    unsigned v = first_vertex(scan);
                    if (compare_signature(signatures[v], signatures[least],
                                          partition.count) < 0) least = v;
                }
                u32 group = 0;
                for (auto scan = remaining; scan; scan &= scan - 1) {
                    unsigned v = first_vertex(scan);
                    if (compare_signature(signatures[v], signatures[least],
                                          partition.count) == 0) group |= bit(v);
                }
                next.cells[next.count++] = group;
                remaining &= ~group;
            }
        }
        bool stable = next.count == partition.count;
        partition = next;
        if (stable) return;
    }
}

struct Frame {
    Partition partition;
    u32 remaining;
    int target;  // -2 means not yet refined; otherwise a nonsingleton cell.
};

NAUTY_CUDA_HD inline bool valid_graph(NAUTY_CUDA_THREAD const Graph &graph) {
    if (graph.n > max_vertices) return false;
    const auto mask = graph.n == 32 ? ~u32{0} : bit(graph.n) - 1;
    for (unsigned v = 0; v < graph.n; ++v) {
        if ((graph.rows[v] & ~mask) || (graph.rows[v] & bit(v))) return false;
        for (unsigned w = 0; w < v; ++w) {
            if (!!(graph.rows[v] & bit(w)) != !!(graph.rows[w] & bit(v))) return false;
        }
    }
    return true;
}

}  // namespace detail

// Exact canonical labelling of an uncoloured simple undirected graph.
// The canonical certificate is the lexicographically least sequence of numeric
// adjacency rows among the leaves of this invariant refinement/search tree.
// It is not specified to equal the canonical representative selected by nauty.
NAUTY_CUDA_HD inline Result canonicalize(NAUTY_CUDA_THREAD const Graph &graph, u64 max_nodes) {
    using namespace detail;
    Result result{};
    result.n = graph.n;
    if (!valid_graph(graph)) {
        result.status = Status::invalid_input;
        return result;
    }

    Frame stack[max_vertices];
    stack[0] = Frame{};
    stack[0].target = -2;
    if (graph.n) {
        stack[0].partition.count = 1;
        stack[0].partition.cells[0] = graph.n == 32 ? ~u32{0} : bit(graph.n) - 1;
    }
    int depth = 0;
    bool found = false;
    while (depth >= 0) {
        NAUTY_CUDA_THREAD Frame &frame = stack[depth];
        if (frame.target == -2) {
            if (result.nodes == max_nodes) {
                result.status = Status::node_limit;
                // Never expose an unproved provisional certificate.
                for (int i = 0; i < max_vertices; ++i) result.rows[i] = result.order[i] = 0;
                return result;
            }
            ++result.nodes;
            refine(graph, frame.partition);
            if (frame.partition.count == graph.n) {
                u32 order[max_vertices];
                u32 rows[max_vertices];
                for (unsigned i = 0; i < graph.n; ++i) order[i] = first_vertex(frame.partition.cells[i]);
                int comparison = found ? 0 : -1;
                for (unsigned i = 0; i < graph.n; ++i) {
                    rows[i] = 0;
                    for (unsigned j = 0; j < graph.n; ++j) {
                        if (graph.rows[order[i]] & bit(order[j])) rows[i] |= bit(j);
                    }
                    if (!comparison && rows[i] != result.rows[i]) {
                        comparison = rows[i] < result.rows[i] ? -1 : 1;
                    }
                }
                if (!found || comparison < 0) {
                    for (unsigned i = 0; i < graph.n; ++i) {
                        result.rows[i] = rows[i];
                        result.order[i] = order[i];
                    }
                    found = true;
                }
                --depth;
                continue;
            }
            for (unsigned c = 0; c < frame.partition.count; ++c) {
                auto cell = frame.partition.cells[c];
                if (cell & (cell - 1)) {
                    frame.target = static_cast<int>(c);
                    frame.remaining = cell;
                    break;
                }
            }
        }
        if (!frame.remaining) {
            --depth;
            continue;
        }
        unsigned selected_vertex = first_vertex(frame.remaining);
        frame.remaining &= ~bit(selected_vertex);
        // Same-cell twins have an actual transposition automorphism preserving
        // this partition. Their search subtrees have identical certificates.
        for (auto scan = frame.remaining; scan; scan &= scan - 1) {
            unsigned other = first_vertex(scan);
            if (((graph.rows[selected_vertex] ^ graph.rows[other]) & ~(bit(selected_vertex) | bit(other))) == 0) {
                frame.remaining &= ~bit(other);
            }
        }
        NAUTY_CUDA_THREAD Frame &child = stack[depth + 1];
        child = Frame{};
        child.target = -2;
        for (unsigned c = 0; c < frame.partition.count; ++c) {
            if (static_cast<int>(c) == frame.target) {
                child.partition.cells[child.partition.count++] = bit(selected_vertex);
                child.partition.cells[child.partition.count++] = frame.partition.cells[c] & ~bit(selected_vertex);
            } else {
                child.partition.cells[child.partition.count++] = frame.partition.cells[c];
            }
        }
        ++depth;
    }
    result.status = Status::ok;
    return result;
}

}  // namespace nauty_cuda

#undef NAUTY_CUDA_HD
#undef NAUTY_CUDA_THREAD
#undef NAUTY_CUDA_CONSTANT
