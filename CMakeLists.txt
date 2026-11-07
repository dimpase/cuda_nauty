cmake_minimum_required(VERSION 3.24)

project(nauty_cuda_prototype VERSION 0.1.0 LANGUAGES CXX)

option(NAUTY_CUDA_ENABLE_CUDA "Build the CUDA executable when a CUDA compiler is available" ON)
option(NAUTY_CUDA_REQUIRE_CUDA "Fail configuration unless the CUDA executable can be built" OFF)
option(NAUTY_CUDA_ENABLE_METAL "Build the optional native Apple Metal backend" OFF)
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
            "No CUDA compiler found: CUDA targets are disabled. "
            "CUDA compilation and execution are not validated. "
            "Use -DNAUTY_CUDA_REQUIRE_CUDA=ON to require the CUDA target."
        )
    endif()
else()
    message(STATUS "CUDA targets disabled")
endif()

if(NAUTY_CUDA_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "NAUTY_CUDA_ENABLE_METAL requires macOS and the Apple Metal framework")
    endif()
    enable_language(OBJCXX)
    find_library(NAUTY_METAL_FRAMEWORK Metal REQUIRED)
    find_library(NAUTY_FOUNDATION_FRAMEWORK Foundation REQUIRED)

    # Embed the shared algorithm and kernel, so the executable can compile its
    # shader at runtime without an offline Metal toolchain or a source checkout.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/canonical.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/canonical_metal.metal")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/canonical.hpp" NAUTY_METAL_CORE)
    string(REPLACE "#pragma once" "" NAUTY_METAL_CORE "${NAUTY_METAL_CORE}")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/canonical_metal.metal" NAUTY_METAL_KERNEL)
    string(REPLACE "#include \"canonical.hpp\"" "" NAUTY_METAL_KERNEL "${NAUTY_METAL_KERNEL}")
    configure_file(src/metal_source.hpp.in generated/metal_source.hpp @ONLY)

    add_library(nauty-metal-backend STATIC src/metal_backend.mm)
    target_include_directories(nauty-metal-backend PUBLIC include src
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_features(nauty-metal-backend PUBLIC cxx_std_17)
    target_compile_options(nauty-metal-backend PRIVATE -fobjc-arc)
    target_link_libraries(nauty-metal-backend PRIVATE
        "${NAUTY_METAL_FRAMEWORK}" "${NAUTY_FOUNDATION_FRAMEWORK}")
    set_target_properties(nauty-metal-backend PROPERTIES
        OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES OBJCXX_EXTENSIONS OFF)

    add_executable(nauty-canonical-metal src/metal_main.mm)
    target_link_libraries(nauty-canonical-metal PRIVATE nauty-metal-backend)
    set_target_properties(nauty-canonical-metal PROPERTIES
        OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES OBJCXX_EXTENSIONS OFF)
    if(NAUTY_CUDA_BUILD_BENCHMARKS)
        add_executable(nauty-benchmark-metal src/benchmark_metal.mm)
        target_link_libraries(nauty-benchmark-metal PRIVATE nauty-metal-backend)
        set_target_properties(nauty-benchmark-metal PROPERTIES
            OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES OBJCXX_EXTENSIONS OFF)
    endif()
    if(BUILD_TESTING)
        add_executable(nauty-metal-core-test tests/metal.cpp)
        target_link_libraries(nauty-metal-core-test PRIVATE nauty-metal-backend)
        add_test(NAME metal-core COMMAND nauty-metal-core-test)
        set_tests_properties(metal-core PROPERTIES TIMEOUT 120)
    endif()
    message(STATUS "Metal enabled: shaders compile at runtime on the selected GPU")
endif()
