#include "canonical.hpp"

// The host supplies this exact layout at buffer(2).
struct MetalParams {
    nauty_cuda::u64 max_nodes;
    nauty_cuda::u32 count;
    nauty_cuda::u32 reserved;
};

static_assert(sizeof(nauty_cuda::u8) == 1, "u8 must be 8 bits");
static_assert(sizeof(nauty_cuda::u32) == 4, "u32 must be 32 bits");
static_assert(sizeof(nauty_cuda::u64) == 8, "u64 must be 64 bits");
static_assert(sizeof(nauty_cuda::Graph) == 132, "Graph host/shader layout mismatch");
static_assert(alignof(nauty_cuda::Graph) == 4, "Graph host/shader alignment mismatch");
static_assert(sizeof(nauty_cuda::Result) == 272, "Result host/shader layout mismatch");
static_assert(alignof(nauty_cuda::Result) == 8, "Result host/shader alignment mismatch");
static_assert(__builtin_offsetof(nauty_cuda::Result, nodes) == 264,
              "Result node counter host/shader offset mismatch");
static_assert(sizeof(MetalParams) == 16, "MetalParams host/shader layout mismatch");
static_assert(alignof(MetalParams) == 8, "MetalParams host/shader alignment mismatch");
static_assert(__builtin_offsetof(MetalParams, count) == 8,
              "MetalParams count host/shader offset mismatch");

kernel void canonicalize_batch(
    device const nauty_cuda::Graph *graphs [[buffer(0)]],
    device nauty_cuda::Result *results [[buffer(1)]],
    constant MetalParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.count) return;
    // The search's references and scratch arrays consistently use thread
    // storage; only this boundary accesses the shared input/output buffers.
    nauty_cuda::Graph graph = graphs[index];
    results[index] = nauty_cuda::canonicalize(graph, params.max_nodes);
}
