Metal validation and timing results, 2026-11-07
==============================================

The optional Metal backend runs on this machine's Apple M4 Max GPU. These
are real native GPU measurements, separate from the still-unmeasured CUDA
backend. Enable it with ``-DNAUTY_CUDA_ENABLE_METAL=ON``; see the project
README for build, validation, and benchmark commands.

With batches of 4,096 dense random 32-vertex graphs, Metal measured 4.67x
the throughput of single-threaded nauty, including host overhead. Sparse
random graphs improved by 1.34x. The direct port remains much slower than
optimized CPU algorithms on the symmetric examples. These results do not
establish an advantage over a multicore CPU or measure graph generation.

Correctness and compatibility
-----------------------------

All 55,636 valid inputs passed exact CPU/Metal comparisons of canonical
graphs, witness permutations, and search-node counts:

* 54,396 inputs covering every unlabelled graph through order 8 and three
  seeded relabellings per graph, checked against nauty's 13,599 classes.
* All 1,100 labelled graphs through order 5, covering 53 classes.
* 140 symmetric/regular fixtures and relabellings through order 32,
  covering 35 classes.

Malformed graph6 input and exhausted budgets also passed. Direct GPU tests
covered invalid adjacency matrices, 64-bit budgets including 2^32 and
UINT64_MAX, vertex 31, maximal search depth, empty calls, and batches crossing
threadgroup boundaries. They passed with Apple's Shader Validation enabled,
including abort-on-fault and stderr reporting. CLI help and empty input do
not initialize the GPU; GPU unavailability produces an explicit error and
no benchmark timing output.

The CPU build still works with Metal disabled. Both CUDA executables and
the CPU targets compiled with NVCC 12.9.86/GCC 13.3 on Ubuntu ARM64 after the
shared-header changes. Linux CPU CTest and CUDA CLI/no-device checks passed.
The Linux build VM was synced and stopped before timing. Actual NVIDIA
execution and Compute Sanitizer validation remain pending.

Measurement protocol
--------------------

Host: Apple M4 Max, 32 GPU cores, 14 logical CPU cores, macOS 27.2 ARM64.
Release build, Apple Clang 21.0.0.21000401, ``-O3 -DNDEBUG``. CPU baselines
use one thread and nauty/Traces 2.9.3. Ordinary workstation background
activity was not controlled. No other benchmark or compilation ran in
parallel; the existing macOS VM was left running.

Each value is the median microseconds per graph from five trials after
warmup, normalized by the actual number of calls. Calibration selected
repetitions separately for each backend. Both runs used seed 1729 and a
one-million-node budget per graph. Every call completed, and every Metal
corpus checksum matched the CPU prototype. Raw samples, hashes, commands,
build settings, host metadata, and graph6 corpora are saved with each run.

The primary Metal timer includes command/encoder creation and cleanup,
input/output memcpy, dispatch, and waiting for completion. Persistent buffer
allocation, runtime shader compilation, parsing, and result validation are
excluded. GPU command duration is also recorded separately using
``GPUEndTime - GPUStartTime``. Host and GPU buffers use shared memory;
the adapter copies into/out of them to preserve the common batch API.

Calibration targeted 0.2 s per trial in the first run and 0.3 s in the second.
Some warmed measurements fell below those targets; all measured trials were
longer than 0.1 s. The reports flag those cases. Five workstation samples
are not formal confidence intervals.

Initial seven-workload comparison
--------------------------------

Batch capacity was 4,096 graphs. Random corpora contained 1,024 graphs each;
each symmetric corpus contained 64. Values are microseconds per graph, with
the Metal column including host overhead.

.. list-table:: Initial CPU and native GPU timings
   :header-rows: 1

   * - Corpus
     - Prototype CPU
     - nauty CPU
     - Traces CPU
     - Metal
   * - All unlabelled n=8 (12,346 graphs)
     - 0.503
     - 0.504
     - 1.401
     - 0.890
   * - Random G(32, 0.1)
     - 4.670
     - 2.381
     - 2.539
     - 5.170
   * - Random G(32, 0.5)
     - 5.060
     - 1.568
     - 2.351
     - 1.172
   * - Cycle C32
     - 697.372
     - 7.097
     - 1.908
     - 3393.600
   * - Hypercube Q5
     - 15358.005
     - 8.169
     - 19.457
     - 77735.967
   * - Alternating rook/Shrikhande, n=16
     - 776.356
     - 2.744
     - 12.147
     - 6820.193
   * - Alternating empty/complete, n=32
     - 13.063
     - 56.103
     - 44.116
     - 64.149

The kernel used 128 threads per threadgroup. A 64-graph corpus therefore
occupied only one threadgroup, with half its threads exiting immediately.
This prompted a larger-batch follow-up; the first run is retained in full.

Larger-batch follow-up
---------------------

Random corpora contained 8,192 graphs, processed in two batches of 4,096.
The cycle corpus contained 4,096 seeded relabellings processed in one batch.
Each backend used exactly the same corpus within each comparison.

.. list-table:: Larger batches, microseconds per graph
   :header-rows: 1

   * - Corpus
     - Prototype CPU
     - nauty CPU
     - Traces CPU
     - Metal host-inclusive
     - Metal GPU command
   * - Random G(32, 0.1)
     - 4.802
     - 2.484
     - 2.657
     - 1.855
     - 1.796
   * - Random G(32, 0.5)
     - 5.004
     - 1.498
     - 2.404
     - 0.321
     - 0.260
   * - Cycle C32
     - 678.755
     - 6.650
     - 1.729
     - 54.700
     - 54.633

Dense random graphs achieved 15.60x the prototype CPU throughput, 4.67x
nauty, and 7.49x Traces. Sparse random graphs achieved 2.59x, 1.34x, and 1.43x,
respectively. All these ratios use the host-inclusive Metal timer.

For C32, Metal was 12.41x faster than the prototype CPU, but still 8.23x
slower than nauty and 31.63x slower than Traces. Batching helps utilization;
it does not remove the algorithm's missing general symmetry pruning.
The GPU command accounts for almost all the cost of the large cycle batch,
so submission overhead alone cannot explain that remaining gap. The other
symmetric families were not repeated with larger batches.

Artifacts and reproduction
--------------------------

Both runs measured source commit
``733d9b1b6fb25769366f26f1ccc2bdd129f39220``. After building as described in
the README, the initial suite used::

    python3 benchmarks/run.py \
        --cpu build-metal/nauty-benchmark-cpu \
        --metal build-metal/nauty-benchmark-metal \
        --target-seconds 0.2 --timeout 60 --max-total-seconds 240

The larger-batch follow-up used::

    python3 benchmarks/run.py \
        --cpu build-metal/nauty-benchmark-cpu \
        --metal build-metal/nauty-benchmark-metal \
        --datasets random32-p010 random32-p050 cycles32 \
        --random-count 8192 --symmetric-count 4096 --batch-size 4096 \
        --target-seconds 0.3 --timeout 120 --max-total-seconds 300

* Initial run: `Initial JSON <results/20261107T185728.084432Z/results.json>`_,
  `Initial CSV <results/20261107T185728.084432Z/summary.csv>`_,
  `Initial text summary <results/20261107T185728.084432Z/summary.txt>`_.
* Larger batches: `Larger-batch JSON <results/20261107T190854.522292Z/results.json>`_,
  `Larger-batch CSV <results/20261107T190854.522292Z/summary.csv>`_,
  `Larger-batch text summary <results/20261107T190854.522292Z/summary.txt>`_.
* Each results directory contains a corpus manifest and the exact inputs.
* `CUDA regression compilation log
  <results/20261107T185728.084432Z/cuda-after-metal.txt>`_.
