GPU canonical labelling prototype
=================================

This is an initial, independent C++/CUDA/Metal implementation of exact
individualization--refinement canonical labelling, with unmodified nauty as
the test oracle. It is a starting point for the requested nauty/Traces GPU
work, rather than a complete port of either program.

Scope
-----

* Simple, undirected, uncoloured graphs with 0 through 32 vertices.
* Exact canonical forms and a permutation witness mapping canonical positions
  to input vertex numbers.
* A shared host/device implementation, a CUDA batch runner, and an optional
  native Apple Metal backend, assigning one graph to each GPU thread.
* Explicit ``node_limit`` results when a search cannot finish within its budget.
  Incomplete results contain no canonical graph and must never be used for
  isomorphism rejection, deduplication, or graph-generation acceptance.

The initial code does not compute full automorphism groups, implement nauty's
group-based pruning or Traces' trace pruning, or generate graphs. Graph6 input
and output allow comparisons with the existing nauty tools.

Algorithm and correctness
-------------------------

Adjacency rows are exact 32-bit sets. Refinement preserves the order of parent
partition cells and subdivides them by lexicographically ordered tuples of
neighbour counts into the current cells. Search individualizes each candidate
in the first nonsingleton cell and compares complete leaf certificates.
The certificate ordering is lexicographic on unsigned numeric adjacency rows.

Within a target cell, vertices with identical neighbours outside their pair
are interchangeable by an actual transposition automorphism. Only one of
these twins is explored. Other symmetry pruning is left for subsequent work.

These rules commute with vertex relabelling. The selected form need not equal
nauty's canonical bytes, nor the global minimum over all permutations.
Validation compares isomorphism classes with nauty and checks each returned
permutation against the original adjacency matrix.

Build and test on the CPU
-------------------------

Requirements: CMake 3.24+, a C++17 compiler, and Python 3 for the oracle tests.
The latter also require nauty's ``geng`` and ``labelg`` executables::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNAUTY_CUDA_ENABLE_CUDA=OFF
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    python3 tests/validate.py --cpu build/nauty-canonical-cpu --max-order 8

The test script accepts ``--geng`` and ``--labelg`` paths when the tools are
not on PATH. It checks all unlabelled classes through the selected order,
seeded relabellings, every labelled graph through order 5, symmetric examples
through order 32, witness permutations, malformed input, and exhausted budgets.

CUDA build and GPU validation
-----------------------------

Use a supported NVIDIA CUDA toolkit and host compiler on Linux. Compilation
can be checked without a GPU; running the CUDA executable requires one.
Choose ``CMAKE_CUDA_ARCHITECTURES`` for the GPU being tested; 80 below is an
example targeting Ampere::

    cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
        -DNAUTY_CUDA_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
    cmake --build build-cuda -j
    ctest --test-dir build-cuda --output-on-failure
    python3 tests/validate.py \
        --cpu build-cuda/nauty-canonical-cpu \
        --cuda build-cuda/nauty-canonical-cuda --max-order 8

``NAUTY_CUDA_REQUIRE_CUDA=ON`` makes a missing CUDA compiler a configuration
failure. The ordinary optional build prints a warning when CUDA targets
cannot be built.

Optional native Apple GPU backend
---------------------------------

Enable Metal explicitly on macOS with an Apple silicon GPU. It is disabled
by default and can be configured independently of CUDA::

    cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
        -DNAUTY_CUDA_ENABLE_CUDA=OFF -DNAUTY_CUDA_ENABLE_METAL=ON
    cmake --build build-metal -j
    ctest --test-dir build-metal --output-on-failure
    python3 tests/validate.py \
        --cpu build-metal/nauty-canonical-cpu \
        --metal build-metal/nauty-canonical-metal --max-order 8

The compiler needs Objective-C++ and the Foundation/Metal frameworks, supplied
by Apple's developer tools. The executable embeds the shared algorithm and
shader source, then compiles the shader through Metal at runtime. An offline
Metal compiler and the original source checkout are not needed to run it.
Pipeline creation and buffer allocation happen once per CLI invocation, when
the first nonempty batch arrives. An unavailable GPU or shader/dispatch error
is reported explicitly; no CPU fallback is used.

The CPU, CUDA and Metal implementations share the same refinement/search
code and result layout. The test suite compares canonical graphs, witness
permutations, and node counts exactly. Additional direct Metal tests cover
malformed adjacency matrices, 64-bit budgets, maximal search depth, vertex 31,
and batches crossing threadgroup boundaries. To check GPU memory accesses::

    MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
        MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
        ctest --test-dir build-metal --output-on-failure -R metal-core -V

Metal support has been tested on an M4 Max. CUDA execution still requires an
NVIDIA GPU; Metal measurements are not CUDA measurements.

Interface
---------

All backend executables read graph6 records from standard input and write JSONL::

    geng -q 8 | build/nauty-canonical-cpu --batch-size 4096 --max-nodes 1000000

Each result has ``status``, ``n``, ``nodes``, ``order``, and ``graph6`` fields.
Successful records have status ``ok``. An invalid input or exhausted search
has ``graph6: null`` and ``order: []``. Malformed graph6 records use n=33 as an
internal invalid-input sentinel and include an explanatory ``error`` field.

The default node budget is one million per graph; zero allows no search.
Exit status 0 means all records completed, 2 means some were invalid or
incomplete, and 3 means an option, I/O, or backend error occurred. Empty input
does not initialize the GPU. Header-only lines are accepted.

Timing canonical labelling
--------------------------

Install the nauty library and its pkg-config metadata (``libnauty.pc``), then
configure and build as above. With ``NAUTY_CUDA_BUILD_BENCHMARKS=ON`` (the
default), CMake builds ``nauty-benchmark-cpu`` when it finds that library,
and the corresponding GPU benchmark when CUDA or Metal is enabled.
Run the CPU comparison::

    python3 benchmarks/run.py --cpu build/nauty-benchmark-cpu

The driver saves the generated graph6 corpora, hashes, commands, raw timing
samples, and JSON/CSV/text summaries in a timestamped directory under
``benchmarks/results``. It compares the prototype, dense nauty, and Traces
on identical inputs: all unlabelled 8-vertex graphs, seeded sparse/dense
random 32-vertex graphs, and several symmetric families. Each CPU backend
uses one thread. These are batch throughput measurements, not single-graph
latencies or measurements of a complete graph generator.

Each backend gets a warmup and five measured trials. Calibration chooses
the number of corpus repetitions per trial, targeting at least 0.1 seconds.
The reported microseconds per graph are median trial seconds divided by
the number of completed calls. Graph generation, parsing, input conversion,
process startup, output, and checksums are outside timing. Required per-call
library initialization and result bookkeeping are included. The prototype's
node budget is one million per graph; incomplete runs have no comparable
throughput. Nauty and Traces compute additional automorphism information,
which the prototype does not yet provide.

On an NVIDIA machine, build both benchmark executables and run them together::

    python3 benchmarks/run.py \
        --cpu build-cuda/nauty-benchmark-cpu \
        --cuda build-cuda/nauty-benchmark-cuda

CUDA results must match the CPU prototype's canonical-output checksum.
The primary CUDA timer includes host-to-device copies, kernel launch and
execution, synchronization, and device-to-host copies. CUDA events also
report kernel-only time. Buffers are allocated before timing and reused;
host buffers use ordinary pageable memory. No transfer/compute overlap is
attempted. Result validation is outside both timers, between batches.
Ratios against nauty and Traces are reported separately from ratios against
the prototype CPU code. Run the correctness suite and Compute Sanitizer
on that NVIDIA machine before relying on its timing results.

For meaningful CPU/GPU ratios, collect both sides on the same machine.
The default batch capacity is 4096; small corpora can underutilize a GPU.
Use ``--random-count``, ``--symmetric-count``, and ``--batch-size`` to study
batch-size effects, and increase ``--timeout``/``--max-total-seconds`` for
longer runs. CUDA speedup is unmeasured until such a run succeeds.

On the Apple GPU, run the corresponding Metal benchmark::

    python3 benchmarks/run.py \
        --cpu build-metal/nauty-benchmark-cpu \
        --metal build-metal/nauty-benchmark-metal

The primary Metal timer includes command/encoder creation and cleanup,
input/output copies into shared buffers, dispatch, and synchronization.
Shader compilation, persistent buffer allocation, parsing and validation are
outside timing. A separate metric records the GPU command duration from
``GPUEndTime - GPUStartTime``; this differs from CUDA event timing.
The driver requires a matching CPU prototype checksum before reporting any
GPU speedup. CUDA and Metal can both be supplied to the driver when available.
Compare against optimized nauty/Traces as well as the prototype, and include
a multicore CPU baseline before making whole-machine performance claims.

Performance status and next steps
---------------------------------

The `initial timing report <benchmarks/RESULTS.rst>`_ records seven CPU
workloads against nauty and Traces, with raw samples and reproducible corpora.
The `Metal timing report <benchmarks/METAL_RESULTS.rst>`_ adds native M4 Max
GPU measurements and a larger-batch follow-up. CUDA speedup remains unmeasured.

Verified during development on 2026-11-07:

* 55,636 valid CPU inputs passed class/witness checks against nauty 2.9.3:
  every unlabelled graph through order 8 with seeded relabellings, every
  labelled graph through order 5, and selected symmetric graphs through 32.
* Core boundary tests and 6,252 oracle inputs passed with AddressSanitizer
  and UndefinedBehaviorSanitizer on macOS ARM64.
* NVIDIA NVCC 12.9.86 with GCC 13.3 on Ubuntu ARM64 compiled and linked the
  CUDA runner for SM80 and generated PTX. The Linux CPU core test passed.
  Both CUDA targets also rebuilt successfully after the Metal compatibility
  changes to the shared header.
* All 55,636 valid cases passed exact CPU/Metal comparisons on the M4 Max,
  alongside invalid-input and node-budget cases. Metal boundary tests also
  passed with Apple's GPU Shader Validation enabled.
* NVIDIA GPU execution, Compute Sanitizer, and CUDA timings remain untested.

No NVIDIA GPU is available in the current development environment, so CUDA
correctness and performance have not been established. The CPU implementation
is a correctness reference, not a competitive replacement for optimized nauty.
The current per-thread search stack and variable search sizes may cause spills
and poor load balance; a CUDA build alone does not demonstrate acceleration.
For SM80, the compiler reports 78 registers and a 6,176-byte stack frame per
thread, with no register spill loads/stores. These are resource diagnostics,
not performance measurements.

Next steps are real CUDA validation (including Compute Sanitizer), profiling
representative graph families, then choosing cooperative refinement, device
work queues, and stronger automorphism pruning based on those measurements.
Canonical construction-path generation requires additional orbit and
augmentation logic beyond this labelling prototype.

References
----------

* `nauty/Traces <https://pallini.di.uniroma1.it/>`_
* `Practical Graph Isomorphism, II <https://arxiv.org/abs/1301.1493>`_
* `nauty user guide <https://users.cecs.anu.edu.au/~bdm/nauty/nug29.pdf>`_
