Canonical-labelling timing results, 2026-11-07
============================================

CUDA speedup: not measured. No NVIDIA GPU is available on this workstation
or through SSH. CUDA compilation and a working no-device error path do not
establish GPU correctness or performance.

CPU results
-----------

Apple M4 Max, macOS 27.2 ARM64, one CPU thread per backend. Release build
with Apple Clang 21.0.0.21000401, ``-O3 -DNDEBUG``; Homebrew nauty/Traces
2.9.3 (64-bit words). This was a workstation run with ordinary background
activity, including macOS media analysis and an existing macOS VM. The
Linux compilation VM was stopped before timing. CPU load averages before
and after the main suite were approximately 2.26 and 2.62 on 14 logical CPUs.

Values below are median microseconds per graph, excluding parsing and
output, from five trials after warmup. Each backend processed the identical
base corpus, with repetitions calibrated independently. The comparison is
single-thread CPU batch throughput, not whole-machine throughput or
single-graph latency. All calls completed without a node-limit failure.

=================================  ==========  ==========  ==========
Corpus                              Prototype       nauty      Traces
=================================  ==========  ==========  ==========
All 12,346 unlabelled graphs, n=8       0.514       0.505       1.445
1,024 random G(32, 0.1) graphs          4.771       2.386       2.568
1,024 random G(32, 0.5) graphs          4.903       1.364       2.238
64 relabellings of cycle C32          664.023       6.351       1.641
64 relabellings of hypercube Q5     15153.547       8.138      19.568
64 rook/Shrikhande graphs, n=16       801.897       2.732      11.971
64 empty/complete graphs, n=32         14.939      56.350      44.335
=================================  ==========  ==========  ==========

The rook/Shrikhande corpus alternates seeded relabellings of the two graphs.
The empty/complete corpus alternates the two trivial isomorphism classes.
Random seed: 1729. Canonical byte strings can differ between algorithms;
checksums are checked for stability within each backend. A future CUDA run
must match the CPU prototype's checksum.

The initial Traces cycle trials were about 90 ms, slightly below the 100 ms
calibration target. A follow-up with longer trials measured 1.611 us/graph
(median trial 257 ms), within 2% of the initial result. The table retains the
original run. All raw samples and this follow-up are saved below; these
workstation measurements are not formal confidence intervals.

Interpretation
--------------

The prototype is about 2.0x and 3.6x slower than dense nauty on the two random
32-vertex families. On symmetric examples the gap is much larger: roughly
405x behind Traces on C32, 1,862x behind nauty on Q5, and 294x behind nauty on
the mixed rook/Shrikhande corpus. On Q5 the prototype visits 6,113 search
nodes per graph. Its lack of general automorphism-group pruning is a likely
major contributor to the symmetric-graph gap.

The prototype is faster on the alternating empty/complete corpus, where
its twin rule handles the symmetry cheaply. This is a narrow special case:
nauty and Traces also compute automorphism information that the prototype
does not provide. The small difference on n=8 is not evidence of a decisive
performance advantage either way.

These results do not predict CUDA speedup. A GPU improvement relative to
this prototype's CPU implementation must be distinguished from an
improvement relative to optimized nauty/Traces. For example, a large gain
over the prototype on Q5 could still leave it slower than nauty. The current
64-graph symmetric batches may also underutilize a GPU. Real NVIDIA tests
need larger batches, several batch sizes, transfer-inclusive measurements,
and CPU baselines on the same NVIDIA host. This is canonical labelling only;
there are no graph-generation speed measurements yet.
Future GPU comparisons should also include a multicore CPU batch baseline
to distinguish gains over one CPU thread from gains over the whole CPU.

Reproduce and inspect
---------------------

Benchmark sources: commit ``f5cb001f9d6736d8196a3360bc4d6fe3b6c23059``.
The main run completed all 21 backend/workload combinations in about 22 s::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNAUTY_CUDA_ENABLE_CUDA=OFF
    cmake --build build -j
    python3 benchmarks/run.py --cpu build/nauty-benchmark-cpu

The longer Traces cycle check used::

    python3 benchmarks/run.py --cpu build/nauty-benchmark-cpu \
        --datasets cycles32 --backends traces --target-seconds 0.3

* `Main raw results <results/20261107T180345.356577Z/results.json>`_
  include commands, samples, node counts, checksums, source and executable
  hashes, compiler settings, and host metadata.
* `Main summary <results/20261107T180345.356577Z/summary.txt>`_
  and `CSV <results/20261107T180345.356577Z/summary.csv>`_.
* `Corpus manifest <results/20261107T180345.356577Z/corpora.json>`_; the exact
  graph6 inputs are in the adjacent ``corpora`` directory.
* `Longer cycle check <results/20261107T180430.183004Z/results.json>`_.

CUDA timing support and validation
---------------------------------

The CUDA benchmark compiled and linked using NVCC 12.9.86 on Ubuntu ARM64,
targeting SM80. It measures kernel time with CUDA events and host time for
H2D transfers, launch, execution, synchronization, and D2H transfers. Buffers
are allocated before timing. It validates every result witness, checks
repeatability, and returns no timing report if a GPU is unavailable or a
graph fails to complete. The driver requires matching canonical-output
checksums from the CPU prototype before reporting GPU speedups.

Compilation used strict host warnings. Help and six invalid-input/no-GPU
checks passed. The actual no-GPU invocation exited 3 with an availability
error and empty stdout. The `compile and validation log
<results/20261107T180345.356577Z/cuda-compile.txt>`_ is included for inspection.
There was no NVIDIA kernel execution, Compute Sanitizer run, or GPU timing.

On a machine with an NVIDIA GPU, first run the correctness suite described
in the project README, then run both benchmark backends together::

    python3 benchmarks/run.py \
        --cpu build-cuda/nauty-benchmark-cpu \
        --cuda build-cuda/nauty-benchmark-cuda
