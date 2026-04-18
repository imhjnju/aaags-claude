cmake_minimum_required(VERSION 3.16)
project(harmonyos_3dgs CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_OPENCL "Build with OpenCL GPU backend" OFF)
option(ENABLE_VULKAN "Build Vulkan GPU backend" ON)

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

    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    FetchContent_MakeAvailable(nlohmann_json)

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
        tests/test_npy_reader.cpp
        tests/test_compare.cpp
        tests/test_manifest.cpp
        tests/test_golden_roundtrip.cpp
    )
    target_link_libraries(gs3d_tests gs3d_core GTest::gtest_main nlohmann_json::nlohmann_json)
    target_include_directories(gs3d_tests PRIVATE tests)
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

# ---------------------------------------------------------------------------
# Vulkan Phase 1: abstractions + TDD gate.
#   gs3d_vk_core       -- RAII wrappers (VulkanContext/Buffer/Shader/Pipeline)
#   gs3d_vulkan_hello  -- standalone compute smoke binary (uses gs3d_vk_core)
#   gs3d_vk_tests      -- gtest TDD gate (uses gs3d_vk_core)
# Skipped when cross-compiling, or when Vulkan/glslangValidator is missing.
# ---------------------------------------------------------------------------
if(ENABLE_VULKAN AND NOT CMAKE_CROSSCOMPILING)
    find_package(Vulkan REQUIRED)
    find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator
        HINTS $ENV{VULKAN_SDK}/bin ~/.local/bin REQUIRED)

    if(Vulkan_FOUND AND GLSLANG_VALIDATOR)
        set(_vk_shader_dir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
        set(_vk_shader_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders)

        function(gs3d_compile_glsl SHADER_NAME OUT_SPV_VAR)
            set(_src ${_vk_shader_src_dir}/${SHADER_NAME}.comp)
            set(_spv ${_vk_shader_dir}/${SHADER_NAME}.spv)
            add_custom_command(
                OUTPUT ${_spv}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${_vk_shader_dir}
                COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                        ${_src} -o ${_spv}
                DEPENDS ${_src}
                COMMENT "Compiling ${SHADER_NAME}.comp -> SPIR-V (vulkan1.1)"
                VERBATIM)
            set(${OUT_SPV_VAR} ${_spv} PARENT_SCOPE)
        endfunction()

        # Embed an already-compiled .spv as a C header via xxd -i.
        # Produces ${CMAKE_CURRENT_BINARY_DIR}/shaders/<shader>_spv.h
        # with `unsigned char <shader>_spv[]` and `unsigned int <shader>_spv_len`.
        find_program(XXD xxd REQUIRED)
        function(gs3d_embed_spirv SHADER_NAME OUT_HEADER_VAR)
            set(_spv    ${_vk_shader_dir}/${SHADER_NAME}.spv)
            set(_header ${_vk_shader_dir}/${SHADER_NAME}_spv.h)
            add_custom_command(
                OUTPUT ${_header}
                COMMAND ${CMAKE_COMMAND} -E chdir ${_vk_shader_dir}
                        ${XXD} -i ${SHADER_NAME}.spv > ${_header}
                DEPENDS ${_spv}
                COMMENT "Embedding ${SHADER_NAME}.spv as C header via xxd -i"
                VERBATIM)
            set(${OUT_HEADER_VAR} ${_header} PARENT_SCOPE)
        endfunction()

        gs3d_compile_glsl(add_one    _spv_add_one)
        gs3d_compile_glsl(preprocess _spv_preprocess)
        gs3d_embed_spirv(add_one _hdr_add_one)
        gs3d_compile_glsl(hello _spv_hello)
        gs3d_embed_spirv(hello _hdr_hello)
        add_custom_target(gs3d_vulkan_shaders
            DEPENDS ${_spv_add_one} ${_spv_preprocess} ${_hdr_add_one}
                    ${_spv_hello} ${_hdr_hello})

        add_library(gs3d_vk_core STATIC
            src/vulkan/vk_context.cpp
            src/vulkan/vk_capabilities.cpp
            src/vulkan/vk_buffer.cpp
            src/vulkan/vk_shader.cpp
            src/vulkan/vk_pipeline.cpp
            src/vulkan/preprocessor_vk.cpp)
        target_include_directories(gs3d_vk_core PUBLIC include)
        target_link_libraries(gs3d_vk_core PUBLIC Vulkan::Vulkan)

        add_executable(gs3d_vulkan_hello src/vulkan/hello_vulkan_main.cpp)
        add_dependencies(gs3d_vulkan_hello gs3d_vulkan_shaders)
        target_link_libraries(gs3d_vulkan_hello PRIVATE gs3d_vk_core)
        target_compile_definitions(gs3d_vulkan_hello PRIVATE
            ADD_ONE_SPV_PATH="${_spv_add_one}")

        if(BUILD_TESTS)
            add_executable(gs3d_vk_tests
                tests/test_vk_compute.cpp
                tests/test_vk_capabilities.cpp
                tests/test_vk_device_selection.cpp
                tests/test_preprocessor_vk.cpp)
            add_dependencies(gs3d_vk_tests gs3d_vulkan_shaders)
            # gs3d_core provides transformPoint4x3 (CPU oracle) and types.
            target_link_libraries(gs3d_vk_tests PRIVATE
                gs3d_vk_core gs3d_core GTest::gtest_main)
            target_compile_definitions(gs3d_vk_tests PRIVATE
                ADD_ONE_SPV_PATH="${_spv_add_one}"
                SHADER_DIR="${_vk_shader_dir}")
            gtest_discover_tests(gs3d_vk_tests)
        endif()
        message(STATUS "Vulkan enabled: gs3d_vk_core, gs3d_vulkan_hello, gs3d_vk_tests (shaders: add_one, preprocess)")
    else()
        message(STATUS "Vulkan SKIPPED (Vulkan=${Vulkan_FOUND}, glslangValidator=${GLSLANG_VALIDATOR})")
    endif()
endif()
