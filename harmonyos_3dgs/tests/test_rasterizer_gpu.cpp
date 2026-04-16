// test_rasterizer_gpu.cpp -- CPU-GPU cross-validation for rasterizer forward pass.
// Runs identical inputs through CPU and GPU rasterizer, compares output images.

#include <gtest/gtest.h>
#include "types.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/preprocessor_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/rasterizer_gpu.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/sorter_gpu.h"
#include "test_gpu_context.h"
#endif

// Small test scene: 16x16 image, single Gaussian
struct RasterizerGpuTestScene {
    static constexpr int N = 1;
    static constexpr int W = 16;
    static constexpr int H = 16;

    float means2D[N * 2];
    float depths[N];
    float conics[N * 3];
    float opacities_2d[N];
    float rgb[N * 3];
    int radii[N];
    int tiles_touched[N];

    PreprocessOutput pre;
    BinningOutput bin;
    Camera cam;
    RenderConfig cfg;

    RasterizerGpuTestScene() {
        // Gaussian at center
        means2D[0] = 8.0f;
        means2D[1] = 8.0f;
        depths[0] = 5.0f;

        // Conic for a circular Gaussian
        conics[0] = 0.05f;  // a
        conics[1] = 0.0f;   // b
        conics[2] = 0.05f;  // c

        opacities_2d[0] = 0.8f;
        rgb[0] = 0.7f;
        rgb[1] = 0.4f;
        rgb[2] = 0.2f;

        radii[0] = 8;
        tiles_touched[0] = 1;

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

        // Binning: single tile
        bin.total_pairs = N;
        bin.keys_unsorted = nullptr;
        bin.values_unsorted = nullptr;
        bin.keys_sorted = nullptr;
        bin.values_sorted = new uint32_t[N]{0};
        bin.num_tiles = 1;
        bin.tile_ranges = new uint32_t[2]{0, 1};
        bin.device_data = nullptr;

        cam = {};
        cam.width = W;
        cam.height = H;
        cam.tan_fovx = 1.0f;
        cam.tan_fovy = 1.0f;

        cfg = {};
        cfg.bg_color[0] = 0.1f;
        cfg.bg_color[1] = 0.1f;
        cfg.bg_color[2] = 0.1f;
        cfg.tile_w = 16;
        cfg.tile_h = 16;
        cfg.eval_3D = false;
        cfg.training = true;
    }

    ~RasterizerGpuTestScene() {
        delete[] bin.values_sorted;
        delete[] bin.tile_ranges;
    }
};

TEST(RasterizerGPU, MatchesCPU_SingleGaussian) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    RasterizerGpuTestScene scene;
    constexpr int N = RasterizerGpuTestScene::N;
    constexpr int W = RasterizerGpuTestScene::W;
    constexpr int H = RasterizerGpuTestScene::H;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU rasterizer
    std::vector<float> img_cpu(npix * 3, 0.0f);
    ForwardCache cache_cpu{};
    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(scene.pre, scene.bin, scene.cam, scene.cfg,
                       img_cpu.data(), nullptr, &cache_cpu, &alloc);

    // GPU rasterizer: create device buffers
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D      = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics       = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb          = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.depths       = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.gauss2screen = ctx.createBuffer(N * 16 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.cov3D_inv    = ctx.createBuffer(N * 6 * sizeof(float),  CL_MEM_READ_ONLY);
    pre_dev.mean_offset  = ctx.createBuffer(N * 3 * sizeof(float),  CL_MEM_READ_ONLY);

    ctx.writeBuffer(pre_dev.means2D,      scene.means2D,      N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics,       scene.conics,       N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb,          scene.rgb,          N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, scene.opacities_2d, N * sizeof(float));
    ctx.writeBuffer(pre_dev.depths,       scene.depths,       N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges   = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);

    ctx.writeBuffer(bin_dev.values_sorted, scene.bin.values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges,   scene.bin.tile_ranges,   2 * sizeof(uint32_t));

    PreprocessOutput pre_gpu = scene.pre;
    pre_gpu.device_data = &pre_dev;
    BinningOutput bin_gpu = scene.bin;
    bin_gpu.device_data = &bin_dev;

    std::vector<float> img_gpu(npix * 3, 0.0f);
    RasterizerGPU gpu_rast(ctx);
    gpu_rast.rasterize(pre_gpu, bin_gpu, scene.cam, scene.cfg,
                       img_gpu.data(), nullptr, nullptr, nullptr);

    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < npix * 3; i++) {
        float abs_err = std::fabs(img_cpu[i] - img_gpu[i]);
        float denom = std::max(std::fabs(img_cpu[i]), std::fabs(img_gpu[i]));
        float rel_err = (denom > 1e-5f) ? abs_err / denom : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    std::fprintf(stderr, "RasterizerGPU: max abs error = %e, max rel error = %e\n",
                 max_abs_err, max_rel_err);

    // Allow for small floating-point differences
    for (int i = 0; i < npix * 3; i++) {
        EXPECT_NEAR(img_cpu[i], img_gpu[i], 1e-4f)
            << "Pixel " << i << " channel mismatch";
    }

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(pre_dev.depths);
    cl.clReleaseMemObject(pre_dev.gauss2screen);
    cl.clReleaseMemObject(pre_dev.cov3D_inv);
    cl.clReleaseMemObject(pre_dev.mean_offset);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}

TEST(RasterizerGPU, MatchesCPU_MultipleGaussians) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 5;
    constexpr int W = 32;
    constexpr int H = 32;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    // Create multiple Gaussians at different positions
    std::vector<float> means2D(N * 2);
    std::vector<float> depths(N);
    std::vector<float> conics(N * 3);
    std::vector<float> opacities_2d(N);
    std::vector<float> rgb(N * 3);
    std::vector<int> radii(N);
    std::vector<int> tiles_touched(N);

    for (int i = 0; i < N; i++) {
        means2D[i * 2] = 8.0f + i * 3.0f;
        means2D[i * 2 + 1] = 8.0f + (i % 3) * 4.0f;
        depths[i] = 4.0f + i * 0.5f;
        conics[i * 3] = 0.03f + i * 0.01f;
        conics[i * 3 + 1] = 0.0f;
        conics[i * 3 + 2] = 0.03f + i * 0.01f;
        opacities_2d[i] = 0.5f + i * 0.1f;
        rgb[i * 3] = 0.3f + i * 0.1f;
        rgb[i * 3 + 1] = 0.5f - i * 0.05f;
        rgb[i * 3 + 2] = 0.7f - i * 0.1f;
        radii[i] = 10;
        tiles_touched[i] = 1;
    }

    PreprocessOutput pre{};
    pre.means2D = means2D.data();
    pre.depths = depths.data();
    pre.conics = conics.data();
    pre.opacities_2d = opacities_2d.data();
    pre.rgb = rgb.data();
    pre.radii = radii.data();
    pre.tiles_touched = tiles_touched.data();
    pre.gauss2screen = nullptr;
    pre.cov3D_inv = nullptr;
    pre.mean_offset = nullptr;
    pre.eval_3D = false;
    pre.device_data = nullptr;

    // Binning output
    uint32_t* values_sorted = alloc.allocate_array<uint32_t>(N);
    for (int i = 0; i < N; i++) values_sorted[i] = i;
    uint32_t* tile_ranges = alloc.allocate_array<uint32_t>(8);
    tile_ranges[0] = 0; tile_ranges[1] = N;
    for (int i = 1; i < 4; i++) {
        tile_ranges[i * 2] = 0;
        tile_ranges[i * 2 + 1] = 0;
    }

    BinningOutput bin{};
    bin.total_pairs = N;
    bin.keys_unsorted = nullptr;
    bin.values_unsorted = nullptr;
    bin.keys_sorted = nullptr;
    bin.values_sorted = values_sorted;
    bin.num_tiles = 4;
    bin.tile_ranges = tile_ranges;
    bin.device_data = nullptr;

    Camera cam{};
    cam.width = W;
    cam.height = H;
    cam.tan_fovx = 1.0f;
    cam.tan_fovy = 1.0f;

    RenderConfig cfg{};
    cfg.bg_color[0] = 0.05f;
    cfg.bg_color[1] = 0.05f;
    cfg.bg_color[2] = 0.05f;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.eval_3D = false;
    cfg.training = true;

    // CPU rasterizer
    std::vector<float> img_cpu(npix * 3, 0.0f);
    ForwardCache cache_cpu{};
    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(pre, bin, cam, cfg, img_cpu.data(), nullptr, &cache_cpu, &alloc);

    // GPU rasterizer
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D      = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics       = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb          = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.depths       = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.gauss2screen = ctx.createBuffer(N * 16 * sizeof(float), CL_MEM_READ_WRITE);
    pre_dev.cov3D_inv    = ctx.createBuffer(N * 6 * sizeof(float),  CL_MEM_READ_WRITE);
    pre_dev.mean_offset  = ctx.createBuffer(N * 3 * sizeof(float),  CL_MEM_READ_WRITE);

    ctx.writeBuffer(pre_dev.means2D,      means2D.data(),      N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics,       conics.data(),       N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb,          rgb.data(),          N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, opacities_2d.data(), N * sizeof(float));
    ctx.writeBuffer(pre_dev.depths,       depths.data(),       N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges   = ctx.createBuffer(8 * sizeof(uint32_t), CL_MEM_READ_ONLY);

    ctx.writeBuffer(bin_dev.values_sorted, values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges,   tile_ranges,   8 * sizeof(uint32_t));

    pre.device_data = &pre_dev;
    bin.device_data = &bin_dev;

    std::vector<float> img_gpu(npix * 3, 0.0f);
    RasterizerGPU gpu_rast(ctx);
    gpu_rast.rasterize(pre, bin, cam, cfg, img_gpu.data(), nullptr, nullptr, nullptr);

    // Compare
    float max_abs_err = 0.0f;
    for (int i = 0; i < npix * 3; i++) {
        float abs_err = std::fabs(img_cpu[i] - img_gpu[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        EXPECT_NEAR(img_cpu[i], img_gpu[i], 1e-4f)
            << "Pixel " << i << " channel mismatch";
    }

    std::fprintf(stderr, "RasterizerGPU (Multi): max abs error = %e\n", max_abs_err);

    // Cleanup
    auto& cl = ctx.cl();
    for (int i = 0; i < 7; i++) {
        if (i == 0) cl.clReleaseMemObject(pre_dev.means2D);
        else if (i == 1) cl.clReleaseMemObject(pre_dev.conics);
        else if (i == 2) cl.clReleaseMemObject(pre_dev.rgb);
        else if (i == 3) cl.clReleaseMemObject(pre_dev.opacities_2d);
        else if (i == 4) cl.clReleaseMemObject(pre_dev.depths);
        else if (i == 5) cl.clReleaseMemObject(pre_dev.gauss2screen);
        else if (i == 6) cl.clReleaseMemObject(pre_dev.cov3D_inv);
    }
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}

// Test with very low alpha Gaussian (should be culled)
TEST(RasterizerGPU, LowAlphaGaussian) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 2;
    constexpr int W = 16;
    constexpr int H = 16;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    std::vector<float> means2D(N * 2);
    std::vector<float> depths(N);
    std::vector<float> conics(N * 3);
    std::vector<float> opacities_2d(N);
    std::vector<float> rgb(N * 3);
    std::vector<int> radii(N);
    std::vector<int> tiles_touched(N);

    // Gaussian 0: normal
    means2D[0] = 8.0f; means2D[1] = 8.0f;
    depths[0] = 5.0f;
    conics[0] = 0.05f; conics[1] = 0.0f; conics[2] = 0.05f;
    opacities_2d[0] = 0.8f;
    rgb[0] = 0.7f; rgb[1] = 0.4f; rgb[2] = 0.2f;
    radii[0] = 8; tiles_touched[0] = 1;

    // Gaussian 1: very low alpha (below 1/255 threshold)
    means2D[2] = 8.0f; means2D[3] = 8.0f;
    depths[1] = 4.0f;
    conics[3] = 0.05f; conics[4] = 0.0f; conics[5] = 0.05f;
    opacities_2d[1] = 0.001f;  // Very low
    rgb[3] = 1.0f; rgb[4] = 0.0f; rgb[5] = 0.0f;
    radii[1] = 8; tiles_touched[1] = 1;

    PreprocessOutput pre{};
    pre.means2D = means2D.data();
    pre.depths = depths.data();
    pre.conics = conics.data();
    pre.opacities_2d = opacities_2d.data();
    pre.rgb = rgb.data();
    pre.radii = radii.data();
    pre.tiles_touched = tiles_touched.data();
    pre.gauss2screen = nullptr;
    pre.cov3D_inv = nullptr;
    pre.mean_offset = nullptr;
    pre.eval_3D = false;
    pre.device_data = nullptr;

    // Sorted order: Gaussian 1 (front) then Gaussian 0
    uint32_t* values_sorted = alloc.allocate_array<uint32_t>(N);
    values_sorted[0] = 1; values_sorted[1] = 0;
    uint32_t* tile_ranges = alloc.allocate_array<uint32_t>(2);
    tile_ranges[0] = 0; tile_ranges[1] = N;

    BinningOutput bin{};
    bin.total_pairs = N;
    bin.values_sorted = values_sorted;
    bin.num_tiles = 1;
    bin.tile_ranges = tile_ranges;
    bin.device_data = nullptr;

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

    // CPU
    std::vector<float> img_cpu(npix * 3, 0.0f);
    ForwardCache cache_cpu{};
    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(pre, bin, cam, cfg, img_cpu.data(), nullptr, &cache_cpu, &alloc);

    // GPU
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D      = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics       = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb          = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.depths       = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.gauss2screen = ctx.createBuffer(N * 16 * sizeof(float), CL_MEM_READ_WRITE);
    pre_dev.cov3D_inv    = ctx.createBuffer(N * 6 * sizeof(float),  CL_MEM_READ_WRITE);
    pre_dev.mean_offset  = ctx.createBuffer(N * 3 * sizeof(float),  CL_MEM_READ_WRITE);

    ctx.writeBuffer(pre_dev.means2D, means2D.data(), N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics, conics.data(), N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb, rgb.data(), N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, opacities_2d.data(), N * sizeof(float));
    ctx.writeBuffer(pre_dev.depths, depths.data(), N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges   = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    ctx.writeBuffer(bin_dev.values_sorted, values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges, tile_ranges, 2 * sizeof(uint32_t));

    pre.device_data = &pre_dev;
    bin.device_data = &bin_dev;

    std::vector<float> img_gpu(npix * 3, 0.0f);
    RasterizerGPU gpu_rast(ctx);
    gpu_rast.rasterize(pre, bin, cam, cfg, img_gpu.data(), nullptr, nullptr, nullptr);

    // Compare - low alpha Gaussian should be skipped in both
    float max_abs_err = 0.0f;
    for (int i = 0; i < npix * 3; i++) {
        float abs_err = std::fabs(img_cpu[i] - img_gpu[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        EXPECT_NEAR(img_cpu[i], img_gpu[i], 1e-4f);
    }
    std::fprintf(stderr, "RasterizerGPU (LowAlpha): max abs error = %e\n", max_abs_err);

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(pre_dev.depths);
    cl.clReleaseMemObject(pre_dev.gauss2screen);
    cl.clReleaseMemObject(pre_dev.cov3D_inv);
    cl.clReleaseMemObject(pre_dev.mean_offset);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}

// Test T saturation (T < 0.0001 early termination)
TEST(RasterizerGPU, TSaturation) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 20;  // Many overlapping Gaussians
    constexpr int W = 16;
    constexpr int H = 16;
    int npix = W * H;

    FrameAllocator alloc(16 * 1024 * 1024);

    std::vector<float> means2D(N * 2, 8.0f);  // All at center
    std::vector<float> depths(N);
    std::vector<float> conics(N * 3);
    std::vector<float> opacities_2d(N);
    std::vector<float> rgb(N * 3);
    std::vector<int> radii(N, 8);
    std::vector<int> tiles_touched(N, 1);

    for (int i = 0; i < N; i++) {
        depths[i] = 2.0f + i * 0.1f;
        conics[i * 3] = 0.02f;
        conics[i * 3 + 1] = 0.0f;
        conics[i * 3 + 2] = 0.02f;
        opacities_2d[i] = 0.5f;  // Medium opacity
        rgb[i * 3] = 1.0f;
        rgb[i * 3 + 1] = 0.0f;
        rgb[i * 3 + 2] = 0.0f;
    }

    PreprocessOutput pre{};
    pre.means2D = means2D.data();
    pre.depths = depths.data();
    pre.conics = conics.data();
    pre.opacities_2d = opacities_2d.data();
    pre.rgb = rgb.data();
    pre.radii = radii.data();
    pre.tiles_touched = tiles_touched.data();
    pre.gauss2screen = nullptr;
    pre.cov3D_inv = nullptr;
    pre.mean_offset = nullptr;
    pre.eval_3D = false;
    pre.device_data = nullptr;

    // Front to back order
    uint32_t* values_sorted = alloc.allocate_array<uint32_t>(N);
    for (int i = 0; i < N; i++) values_sorted[i] = i;
    uint32_t* tile_ranges = alloc.allocate_array<uint32_t>(2);
    tile_ranges[0] = 0; tile_ranges[1] = N;

    BinningOutput bin{};
    bin.total_pairs = N;
    bin.values_sorted = values_sorted;
    bin.num_tiles = 1;
    bin.tile_ranges = tile_ranges;
    bin.device_data = nullptr;

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

    // CPU
    std::vector<float> img_cpu(npix * 3, 0.0f);
    ForwardCache cache_cpu{};
    RasterizerCPU cpu_rast;
    cpu_rast.rasterize(pre, bin, cam, cfg, img_cpu.data(), nullptr, &cache_cpu, &alloc);

    // GPU
    PreprocessDeviceBuffers pre_dev{};
    pre_dev.means2D      = ctx.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.conics       = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.rgb          = ctx.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    pre_dev.opacities_2d = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.depths       = ctx.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);
    pre_dev.gauss2screen = ctx.createBuffer(N * 16 * sizeof(float), CL_MEM_READ_WRITE);
    pre_dev.cov3D_inv    = ctx.createBuffer(N * 6 * sizeof(float),  CL_MEM_READ_WRITE);
    pre_dev.mean_offset  = ctx.createBuffer(N * 3 * sizeof(float),  CL_MEM_READ_WRITE);

    ctx.writeBuffer(pre_dev.means2D, means2D.data(), N * 2 * sizeof(float));
    ctx.writeBuffer(pre_dev.conics, conics.data(), N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.rgb, rgb.data(), N * 3 * sizeof(float));
    ctx.writeBuffer(pre_dev.opacities_2d, opacities_2d.data(), N * sizeof(float));
    ctx.writeBuffer(pre_dev.depths, depths.data(), N * sizeof(float));

    BinningDeviceBuffers bin_dev{};
    bin_dev.values_sorted = ctx.createBuffer(N * sizeof(uint32_t), CL_MEM_READ_ONLY);
    bin_dev.tile_ranges   = ctx.createBuffer(2 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    ctx.writeBuffer(bin_dev.values_sorted, values_sorted, N * sizeof(uint32_t));
    ctx.writeBuffer(bin_dev.tile_ranges, tile_ranges, 2 * sizeof(uint32_t));

    pre.device_data = &pre_dev;
    bin.device_data = &bin_dev;

    std::vector<float> img_gpu(npix * 3, 0.0f);
    RasterizerGPU gpu_rast(ctx);
    gpu_rast.rasterize(pre, bin, cam, cfg, img_gpu.data(), nullptr, nullptr, nullptr);

    // Compare
    float max_abs_err = 0.0f;
    for (int i = 0; i < npix * 3; i++) {
        float abs_err = std::fabs(img_cpu[i] - img_gpu[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        EXPECT_NEAR(img_cpu[i], img_gpu[i], 1e-4f);
    }
    std::fprintf(stderr, "RasterizerGPU (TSaturation): max abs error = %e\n", max_abs_err);

    // T should saturate before all 20 Gaussians are processed
    // (0.5)^12 ≈ 0.0002, so ~12-13 Gaussians should contribute
    int center_pix = 8 * 16 + 8;
    float T_final = cache_cpu.T_final[center_pix];
    int n_contrib = cache_cpu.n_contrib[center_pix];
    std::fprintf(stderr, "  Center pixel: T_final=%.6f, n_contrib=%d\n", T_final, n_contrib);
    EXPECT_LT(T_final, 0.001f);  // T should be very small

    // Cleanup
    auto& cl = ctx.cl();
    cl.clReleaseMemObject(pre_dev.means2D);
    cl.clReleaseMemObject(pre_dev.conics);
    cl.clReleaseMemObject(pre_dev.rgb);
    cl.clReleaseMemObject(pre_dev.opacities_2d);
    cl.clReleaseMemObject(pre_dev.depths);
    cl.clReleaseMemObject(pre_dev.gauss2screen);
    cl.clReleaseMemObject(pre_dev.cov3D_inv);
    cl.clReleaseMemObject(pre_dev.mean_offset);
    cl.clReleaseMemObject(bin_dev.values_sorted);
    cl.clReleaseMemObject(bin_dev.tile_ranges);
#endif
}
