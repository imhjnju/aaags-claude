cmake_minimum_required(VERSION 3.16)
project(harmonyos_3dgs CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_OPENCL "Build with OpenCL GPU backend" OFF)

if(CMAKE_CROSSCOMPILING)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    # Disable FMA contraction on ARM to match x86 float rounding behavior.
    # FMA does mul+add with single rounding; without FMA, mul and add each round
    # separately, matching x86 SSE behavior for cross-platform reproducibility.
    add_compile_options(-ffp-contract=off)
endif()

# Ensure consistent floating-point behavior across platforms
add_compile_options(-ffp-contract=off)

add_library(gs3d_core
    src/types.cpp
    src/math_utils.cpp
    src/sh_eval.cpp
    src/ply_loader.cpp
    src/cpu/preprocessor_cpu.cpp
    src/cpu/tile_binner_cpu.cpp
    src/cpu/sorter_cpu.cpp
    src/cpu/rasterizer_cpu.cpp
    src/cpu/rasterizer_backward_cpu.cpp
    src/cpu/preprocessor_backward_cpu.cpp
    src/train_types.cpp
    src/renderer.cpp
    src/image_io.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/trainer.cpp
    src/density_controller.cpp
)
target_include_directories(gs3d_core PUBLIC include src)

if(ENABLE_OPENCL)
    target_compile_definitions(gs3d_core PUBLIC ENABLE_OPENCL=1)
    target_sources(gs3d_core PRIVATE
        src/gpu/opencl_loader.cpp
        src/gpu/opencl_context.cpp
        src/gpu/preprocessor_gpu.cpp
        src/gpu/tile_binner_gpu.cpp
        src/gpu/sorter_gpu.cpp
        src/gpu/rasterizer_gpu.cpp
        src/gpu/rasterizer_backward_gpu.cpp
        src/gpu/preprocessor_backward_gpu.cpp
        src/trainer_gpu.cpp
    )
    target_include_directories(gs3d_core PUBLIC third_party/OpenCL-Headers)
    target_link_libraries(gs3d_core PUBLIC dl)
endif()

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(gs3d_tests
        tests/test_types.cpp
        tests/test_math_utils.cpp
        tests/test_sh_eval.cpp
        tests/test_ply_loader.cpp
        tests/test_preprocessor.cpp
        tests/test_tile_binner.cpp
        tests/test_sorter.cpp
        tests/test_rasterizer.cpp
        tests/test_rasterizer_gpu.cpp
        tests/test_preprocessor_gpu.cpp
        tests/test_renderer_e2e.cpp
        tests/test_train_types.cpp
        tests/test_forward_cache.cpp
        tests/test_loss.cpp
        tests/test_rasterizer_backward.cpp
        tests/test_preprocessor_backward.cpp
        tests/test_optimizer.cpp
        tests/test_trainer_e2e.cpp
        tests/test_train_pipeline.cpp
        tests/test_train_loop.cpp
        tests/test_rasterizer_backward_gpu.cpp
        tests/test_preprocessor_backward_gpu.cpp
        tests/test_trainer_gpu.cpp
        tests/test_density_controller.cpp
    )
    target_link_libraries(gs3d_tests gs3d_core GTest::gtest_main)
    target_compile_definitions(gs3d_tests PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
    include(GoogleTest)
    gtest_discover_tests(gs3d_tests)
endif()

add_executable(gs3d_render src/main.cpp)
target_link_libraries(gs3d_render gs3d_core)

add_executable(gs3d_train src/train_main.cpp)
target_link_libraries(gs3d_train gs3d_core)

add_executable(gs3d_train_demo src/train_demo.cpp)
target_link_libraries(gs3d_train_demo gs3d_core)

add_executable(gs3d_verify_grads src/verify_grads_main.cpp)
target_link_libraries(gs3d_verify_grads gs3d_core)

add_executable(gs3d_train_compare src/train_compare_cpp.cpp)
target_link_libraries(gs3d_train_compare gs3d_core)
