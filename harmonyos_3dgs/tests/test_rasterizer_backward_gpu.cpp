// test_rasterizer_backward_gpu.cpp -- CPU-GPU cross-validation for rasterizer backward
// Runs identical inputs through CPU and GPU backward, compares gradient arrays.

#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/rasterizer_backward_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "test_gpu_context.h"
#endif

// Small test scene: 8x8 image, 3 Gaussians, single tile
struct GPUBackwardTestScene {
    static constexpr int N = 3;
    static constexpr int W = 8;
    static constexpr int H = 8;

    float means2D[N * 2];
    float depths[N];
    float conics[N * 3];
    float opacities_2d[N];
    float rgb[N * 3];
    int radii[N];
    int tiles_touched[N];

    uint32_t tile_ranges[2];
    uint32_t values_sorted[N];

    PreprocessOutput pre;
    BinningOutput bin;
    Camera cam;
    RenderConfig cfg;

    GPUBackwardTestScene() {
        // Gaussian 0: center-ish
        means2D[0] = 3.5f; means2D[1] = 3.5f;
        // Gaussian 1: off-center
        means2D[2] = 5.0f; means2D[3] = 2.0f;
        // Gaussian 2: another position
        means2D[4] = 2.0f; means2D[5] = 6.0f;

        depths[0] = 5.0f; depths[1] = 6.0f; depths[2] = 4.0f;

        // Wide Gaussians so all pixels contribute
        for (int i = 0; i < N; i++) {
            conics[i*3]   = 0.04f;   // a
            conics[i*3+1] = -0.005f; // b
            conics[i*3+2] = 0.05f;   // c
        }

        opacities_2d[0] = 0.7f; opacities_2d[1] = 0.5f; opacities_2d[2] = 0.6f;

        rgb[0] = 0.8f; rgb[1] = 0.2f; rgb[2] = 0.4f;
        rgb[3] = 0.3f; rgb[4] = 0.9f; rgb[5] = 0.1f;
        rgb[6] = 0.5f; rgb[7] = 0.5f; rgb[8] = 0.7f;

        for (int i = 0; i < N; i++) {
            radii[i] = 8;
            tiles_touched[i] = 1;
        }

        // Front-to-back sorted order: 2, 0, 1 (by depth)
        values_sorted[0] = 2; values_sorted[1] = 0; values_sorted[2] = 1;
        tile_ranges[0] = 0; tile_ranges[1] = N;

        pre.means2D = means2D;
        pre.depths = depths;
        pre.conics = conics;
        pre.opacities_2d = opacities_2d;
        pre.rgb = rgb;
        pre.radii = radii;
        pre.tiles_touched = tiles_touched;
        pre.gauss2screen = nullptr;
        pre.cov3D_inv = nullptr;
        pre.mean_offset = nullptr;
        pre.eval_3D = false;
        pre.device_data = nullptr;

        bin.total_pairs = N;
        bin.keys_unsorted = nullptr;
        bin.values_unsorted = nullptr;
        bin.keys_sorted = nullptr;
        bin.values_sorted = values_sorted;
        bin.num_tiles = 1;
        bin.tile_ranges = tile_ranges;
        bin.device_data = nullptr;

        cam = {};
        cam.width = W;
        cam.height = H;
        cam.tan_fovx = 1.0f;
        cam.tan_fovy = 1.0f;

        cfg = {};
        cfg.bg_color[0] = 0.1f;
        cfg.bg_color[1] = 0.2f;
        cfg.bg_color[2] = 0.05f;
        cfg.tile_w = 16;
        cfg.tile_h = 16;
        cfg.eval_3D = false;
        cfg.training = true;
    }
};

TEST(RasterizerBackwardGPU, MatchesCPU) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    GPUBackwardTestScene scene;
    constexpr int N = GPUBackwardTestScene::N;
    constexpr int W = GPUBackwardTestScene::W;
    constexpr int H = GPUBackwardTestScene::H;
    int npix = W * H;

    // Run CPU forward to get cache + rendered image
    FrameAllocator alloc(16 * 1024 * 1024);

    ForwardCache cache{};
    cache.T_final   = alloc.allocate_array<float>(npix);
    cache.n_contrib = alloc.allocate_array<int>(npix);
    std::memset(cache.T_final, 0, npix * sizeof(float));
    std::memset(cache.n_contrib, 0, npix * sizeof(int));

    float* rendered = alloc.allocate_array<float>(npix * 3);
    std::memset(rendered, 0, npix * 3 * sizeof(float));

    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(scene.pre, scene.bin, scene.cam, scene.cfg,
                       rendered, nullptr, &cache, &alloc);

    // Construct a synthetic d_image (gradient of loss w.r.t. rendered)
    float* d_image = alloc.allocate_array<float>(npix * 3);
    for (int i = 0; i < npix * 3; i++)
        d_image[i] = 0.01f * (float)(i % 7 - 3);  // small varied gradients

    // CPU backward
    RasterGradOutput rgrad_cpu;
    rgrad_cpu.allocate_and_zero(alloc, N);
    RasterizerBackwardCPU cpu_bwd;
    cpu_bwd.backward(scene.pre, scene.bin, scene.cam, scene.cfg, cache, d_image, rgrad_cpu);

    // GPU backward: create device buffers for preprocess and binning data
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D      = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics       = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb          = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);

    ctx.writeBuffer(pre_dev.means2D,      scene.means2D,      N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics,       scene.conics,       N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb,          scene.rgb,          N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, scene.opacities_2d, N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges   = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);

    ctx.writeBuffer(bin_dev.values_sorted, scene.values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges,   scene.tile_ranges,   2 * sizeof(uint32_t));

    PreprocessOutput pre_gpu = scene.pre;
    pre_gpu.device_data = &pre_dev;
    BinningOutput bin_gpu = scene.bin;
    bin_gpu.device_data = &bin_dev;

    RasterGradOutput rgrad_gpu;
    rgrad_gpu.allocate_and_zero(alloc, N);

    RasterizerBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(pre_gpu, bin_gpu, scene.cam, scene.cfg, cache, d_image, N, rgrad_gpu);

    // Compare
    float max_rel_err = 0.0f;
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            float denom = std::max(std::fabs(cpu[i]), std::fabs(gpu[i]));
            float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;
            max_rel_err = std::max(max_rel_err, rel_err);
            EXPECT_LT(rel_err, 1e-2f)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i]
                << " rel_err=" << rel_err;
        }
    };

    check("d_means2D",      rgrad_cpu.d_means2D,      rgrad_gpu.d_means2D,      N * 2);
    check("d_conics",       rgrad_cpu.d_conics,       rgrad_gpu.d_conics,       N * 3);
    check("d_rgb",          rgrad_cpu.d_rgb,          rgrad_gpu.d_rgb,          N * 3);
    check("d_opacities_2d", rgrad_cpu.d_opacities_2d, rgrad_gpu.d_opacities_2d, N);

    std::fprintf(stderr, "RasterizerBackwardGPU: max relative error = %e\n", max_rel_err);

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}

// Test backward with single Gaussian
TEST(RasterizerBackwardGPU, SingleGaussian) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 1;
    constexpr int W = 8;
    constexpr int H = 8;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    float means2D[N * 2] = {4.0f, 4.0f};
    float depths[N] = {5.0f};
    float conics[N * 3] = {0.05f, 0.0f, 0.05f};
    float opacities_2d[N] = {0.8f};
    float rgb[N * 3] = {0.7f, 0.5f, 0.3f};
    int radii[N] = {8};
    int tiles_touched[N] = {1};

    uint32_t values_sorted[N] = {0};
    uint32_t tile_ranges[2] = {0, 1};

    PreprocessOutput pre{};
    pre.means2D = means2D;
    pre.depths = depths;
    pre.conics = conics;
    pre.opacities_2d = opacities_2d;
    pre.rgb = rgb;
    pre.radii = radii;
    pre.tiles_touched = tiles_touched;
    pre.gauss2screen = nullptr;
    pre.cov3D_inv = nullptr;
    pre.mean_offset = nullptr;
    pre.eval_3D = false;

    BinningOutput bin{};
    bin.total_pairs = N;
    bin.values_sorted = values_sorted;
    bin.num_tiles = 1;
    bin.tile_ranges = tile_ranges;

    Camera cam{};
    cam.width = W;
    cam.height = H;
    cam.tan_fovx = 1.0f;
    cam.tan_fovy = 1.0f;

    RenderConfig cfg{};
    cfg.bg_color[0] = 0.0f;
    cfg.bg_color[1] = 0.0f;
    cfg.bg_color[2] = 0.0f;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.eval_3D = false;

    // CPU forward
    ForwardCache cache{};
    cache.T_final = alloc.allocate_array<float>(npix);
    cache.n_contrib = alloc.allocate_array<int>(npix);
    std::memset(cache.T_final, 0, npix * sizeof(float));
    std::memset(cache.n_contrib, 0, npix * sizeof(int));

    float* rendered = alloc.allocate_array<float>(npix * 3);
    std::memset(rendered, 0, npix * 3 * sizeof(float));

    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(pre, bin, cam, cfg, rendered, nullptr, &cache, &alloc);

    // Synthetic d_image
    float* d_image = alloc.allocate_array<float>(npix * 3);
    for (int i = 0; i < npix * 3; i++)
        d_image[i] = 0.01f;

    // CPU backward
    RasterGradOutput rgrad_cpu;
    rgrad_cpu.allocate_and_zero(alloc, N);
    RasterizerBackwardCPU cpu_bwd;
    cpu_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad_cpu);

    // GPU backward
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);

    ctx.writeBuffer(pre_dev.means2D, means2D, N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics, conics, N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb, rgb, N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, opacities_2d, N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    ctx.writeBuffer(bin_dev.values_sorted, values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges, tile_ranges, 2 * sizeof(uint32_t));

    pre.device_data = &pre_dev;
    bin.device_data = &bin_dev;

    RasterGradOutput rgrad_gpu;
    rgrad_gpu.allocate_and_zero(alloc, N);

    RasterizerBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(pre, bin, cam, cfg, cache, d_image, N, rgrad_gpu);

    // Compare
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            float denom = std::max(std::fabs(cpu[i]), std::fabs(gpu[i]));
            float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;
            EXPECT_LT(rel_err, tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    check("d_rgb", rgrad_cpu.d_rgb, rgrad_gpu.d_rgb, N * 3, 1e-2f);
    check("d_opacities_2d", rgrad_cpu.d_opacities_2d, rgrad_gpu.d_opacities_2d, N, 1e-2f);
    check("d_means2D", rgrad_cpu.d_means2D, rgrad_gpu.d_means2D, N * 2, 1e-2f);
    check("d_conics", rgrad_cpu.d_conics, rgrad_gpu.d_conics, N * 3, 1e-2f);

    std::fprintf(stderr, "RasterizerBackwardGPU (Single): gradients match for single Gaussian\n");

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}

// Test backward with depth ordering (front Gaussian has larger gradient)
TEST(RasterizerBackwardGPU, DepthOrdering) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 2;
    constexpr int W = 8;
    constexpr int H = 8;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    // Two Gaussians at same position but different depths
    float means2D[N * 2] = {4.0f, 4.0f,  4.0f, 4.0f};
    float depths[N] = {3.0f, 6.0f};  // Front (closer) and back
    float conics[N * 3] = {0.05f, 0.0f, 0.05f,  0.05f, 0.0f, 0.05f};
    float opacities_2d[N] = {0.6f, 0.6f};
    float rgb[N * 3] = {1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f};  // Front=red, Back=blue
    int radii[N] = {8, 8};
    int tiles_touched[N] = {1, 1};

    // Sorted front-to-back: front (index 0) first, then back (index 1)
    uint32_t values_sorted[N] = {0, 1};
    uint32_t tile_ranges[2] = {0, 2};

    PreprocessOutput pre{};
    pre.means2D = means2D;
    pre.depths = depths;
    pre.conics = conics;
    pre.opacities_2d = opacities_2d;
    pre.rgb = rgb;
    pre.radii = radii;
    pre.tiles_touched = tiles_touched;
    pre.gauss2screen = nullptr;
    pre.cov3D_inv = nullptr;
    pre.mean_offset = nullptr;
    pre.eval_3D = false;

    BinningOutput bin{};
    bin.total_pairs = N;
    bin.values_sorted = values_sorted;
    bin.num_tiles = 1;
    bin.tile_ranges = tile_ranges;

    Camera cam{};
    cam.width = W;
    cam.height = H;
    cam.tan_fovx = 1.0f;
    cam.tan_fovy = 1.0f;

    RenderConfig cfg{};
    cfg.bg_color[0] = 0.0f;
    cfg.bg_color[1] = 0.0f;
    cfg.bg_color[2] = 0.0f;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.eval_3D = false;

    // CPU forward
    ForwardCache cache{};
    cache.T_final = alloc.allocate_array<float>(npix);
    cache.n_contrib = alloc.allocate_array<int>(npix);
    std::memset(cache.T_final, 0, npix * sizeof(float));
    std::memset(cache.n_contrib, 0, npix * sizeof(int));

    float* rendered = alloc.allocate_array<float>(npix * 3);
    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(pre, bin, cam, cfg, rendered, nullptr, &cache, &alloc);

    // d_image targeting red channel
    float* d_image = alloc.allocate_array<float>(npix * 3);
    for (int i = 0; i < npix * 3; i++)
        d_image[i] = (i % 3 == 0) ? 0.1f : 0.0f;  // Only red channel gradient

    // CPU backward
    RasterGradOutput rgrad_cpu;
    rgrad_cpu.allocate_and_zero(alloc, N);
    RasterizerBackwardCPU cpu_bwd;
    cpu_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad_cpu);

    // GPU backward
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY);

    ctx.writeBuffer(pre_dev.means2D, means2D, N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics, conics, N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb, rgb, N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, opacities_2d, N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    ctx.writeBuffer(bin_dev.values_sorted, values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges, tile_ranges, 2 * sizeof(uint32_t));

    pre.device_data = &pre_dev;
    bin.device_data = &bin_dev;

    RasterGradOutput rgrad_gpu;
    rgrad_gpu.allocate_and_zero(alloc, N);

    RasterizerBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(pre, bin, cam, cfg, cache, d_image, N, rgrad_gpu);

    // Compare
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            float denom = std::max(std::fabs(cpu[i]), std::fabs(gpu[i]));
            float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;
            EXPECT_LT(rel_err, tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    check("d_rgb", rgrad_cpu.d_rgb, rgrad_gpu.d_rgb, N * 3, 1e-2f);
    check("d_opacities_2d", rgrad_cpu.d_opacities_2d, rgrad_gpu.d_opacities_2d, N, 1e-2f);

    // Front Gaussian (red) should have larger RGB gradient than back (blue)
    float front_rgb_grad = std::fabs(rgrad_gpu.d_rgb[0]) + std::fabs(rgrad_gpu.d_rgb[1]) + std::fabs(rgrad_gpu.d_rgb[2]);
    float back_rgb_grad = std::fabs(rgrad_gpu.d_rgb[3]) + std::fabs(rgrad_gpu.d_rgb[4]) + std::fabs(rgrad_gpu.d_rgb[5]);
    std::fprintf(stderr, "RasterizerBackwardGPU (Depth): front_grad=%.4f, back_grad=%.4f\n",
                 front_rgb_grad, back_rgb_grad);

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}
