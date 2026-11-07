cmake_minimum_required(VERSION 3.24)

project(nauty_cuda_prototype VERSION 0.1.0 LANGUAGES CXX)

option(NAUTY_CUDA_ENABLE_CUDA "Build the CUDA executable when a CUDA compiler is available" ON)
option(NAUTY_CUDA_REQUIRE_CUDA "Fail configuration unless the CUDA executable can be built" OFF)
option(NAUTY_CUDA_BUILD_BENCHMARKS "Build timing tools when their dependencies are available" ON)

if(NAUTY_CUDA_REQUIRE_CUDA AND NOT NAUTY_CUDA_ENABLE_CUDA)
    message(FATAL_ERROR "NAUTY_CUDA_REQUIRE_CUDA requires NAUTY_CUDA_ENABLE_CUDA=ON")
endif()

add_executable(nauty-canonical-cpu src/cpu_main.cpp)
target_include_directories(nauty-canonical-cpu PRIVATE include)
target_compile_features(nauty-canonical-cpu PRIVATE cxx_std_17)
set_target_properties(nauty-canonical-cpu PROPERTIES CXX_EXTENSIONS OFF)

if(NAUTY_CUDA_BUILD_BENCHMARKS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(NAUTY QUIET IMPORTED_TARGET libnauty)
    endif()
    if(TARGET PkgConfig::NAUTY)
        add_executable(nauty-benchmark-cpu src/benchmark_cpu.cpp)
        target_include_directories(nauty-benchmark-cpu PRIVATE include)
        target_link_libraries(nauty-benchmark-cpu PRIVATE PkgConfig::NAUTY)
        target_compile_features(nauty-benchmark-cpu PRIVATE cxx_std_17)
        set_target_properties(nauty-benchmark-cpu PROPERTIES CXX_EXTENSIONS OFF)
    else()
        message(STATUS "nauty pkg-config library not found: CPU comparison benchmark is unavailable")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(nauty-canonical-core-test tests/core.cpp)
    target_include_directories(nauty-canonical-core-test PRIVATE include)
    target_compile_features(nauty-canonical-core-test PRIVATE cxx_std_17)
    set_target_properties(nauty-canonical-core-test PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME canonical-core COMMAND nauty-canonical-core-test)
endif()

if(NAUTY_CUDA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_executable(nauty-canonical-cuda src/cuda_main.cu)
        target_include_directories(nauty-canonical-cuda PRIVATE include)
        set_target_properties(nauty-canonical-cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS OFF
        )
        if(NAUTY_CUDA_BUILD_BENCHMARKS)
            add_executable(nauty-benchmark-cuda src/benchmark_cuda.cu)
            target_include_directories(nauty-benchmark-cuda PRIVATE include)
            set_target_properties(nauty-benchmark-cuda PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_EXTENSIONS OFF
            )
        endif()
        # Keep the compiler/toolchain default, or the user's explicit
        # CMAKE_CUDA_ARCHITECTURES. Do not query a GPU on compile-only hosts.
        message(STATUS "CUDA executable enabled; architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    elseif(NAUTY_CUDA_REQUIRE_CUDA)
        message(FATAL_ERROR "A CUDA compiler is required but was not found. Set CUDACXX or CMAKE_CUDA_COMPILER.")
    else()
        message(WARNING
            "No CUDA compiler found: building CPU targets only. "
            "CUDA compilation and execution are not validated. "
            "Use -DNAUTY_CUDA_REQUIRE_CUDA=ON to require the CUDA target."
        )
    endif()
else()
    message(STATUS "CUDA disabled: building CPU targets only")
endif()
