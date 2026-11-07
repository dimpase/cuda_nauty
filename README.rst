CUDA canonical labelling prototype
=================================

This is an initial, independent C++/CUDA implementation of exact
individualization--refinement canonical labelling, with unmodified nauty as
the test oracle. It is a starting point for the requested nauty/Traces GPU
work, rather than a complete port of either program.

Scope
-----

* Simple, undirected, uncoloured graphs with 0 through 32 vertices.
* Exact canonical forms and a permutation witness mapping canonical positions
  to input vertex numbers.
* A shared host/device implementation and a CUDA batch runner, initially
  assigning one graph to each GPU thread.
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
failure. The ordinary optional build prints a warning when it builds only
the CPU executable.

Interface
---------

Both executables read graph6 records from standard input and write JSONL::

    geng -q 8 | build/nauty-canonical-cpu --batch-size 4096 --max-nodes 1000000

Each result has ``status``, ``n``, ``nodes``, ``order``, and ``graph6`` fields.
Successful records have status ``ok``. An invalid input or exhausted search
has ``graph6: null`` and ``order: []``. Malformed graph6 records use n=33 as an
internal invalid-input sentinel and include an explanatory ``error`` field.

The default node budget is one million per graph; zero allows no search.
Exit status 0 means all records completed, 2 means some were invalid or
incomplete, and 3 means an option, I/O, or backend error occurred. Empty input
does not initialize CUDA. Header-only lines are accepted.

Performance status and next steps
---------------------------------

Verified during development on 2026-11-07:

* 55,636 valid CPU inputs passed class/witness checks against nauty 2.9.3:
  every unlabelled graph through order 8 with seeded relabellings, every
  labelled graph through order 5, and selected symmetric graphs through 32.
* Core boundary tests and 6,252 oracle inputs passed with AddressSanitizer
  and UndefinedBehaviorSanitizer on macOS ARM64.
* NVIDIA NVCC 12.9.86 with GCC 13.3 on Ubuntu ARM64 compiled and linked the
  CUDA runner for SM80 and generated PTX. The Linux CPU core test passed.
* NVIDIA GPU execution, Compute Sanitizer, and GPU timings remain untested.

No NVIDIA GPU is available in the current development environment, so GPU
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
