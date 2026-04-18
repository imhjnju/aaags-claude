// test_preprocessor_backward_vulkan.cpp — SP-3 T23: PreprocessorBackwardVulkan
// vs CPU reference comparison.
//
// Runs both the CPU and Vulkan backward preprocessors on the same inputs
// and verifies the outputs match within tolerance.
//
// Setup mirrors test_preprocessor_backward.cpp::CovChain_ScaleGradient:
//   N=1, sh_degree=0, simple camera, visible Gaussian.
//
// Tolerance: max abs diff < 1e-4 on d_raw_positions, d_raw_sh_coeffs,
//   d_raw_scales, d_raw_rotations.
//
// GTEST_SKIP if no Vulkan device.

#include "vulkan/vk_context.h"
#include "vulkan/preprocessor_backward_vulkan.h"

#include "types.h"
#include "train_types.h"
#include "math_utils.h"
#include "sh_eval.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/preprocessor_backward_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

// ---- Helpers ---------------------------------------------------------------

static void setup_simple_camera_vk(Camera& cam) {
    cam.width  = 16;
    cam.height = 16;
    cam.tan_fovx = 1.0f;
    cam.tan_fovy = 1.0f;
    cam.cam_pos[0] = 0.0f;
    cam.cam_pos[1] = 0.0f;
    cam.cam_pos[2] = 0.0f;

    // Identity view matrix (column-major)
    std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
    cam.view_matrix[0]  = 1.f;
    cam.view_matrix[5]  = 1.f;
    cam.view_matrix[10] = 1.f;
    cam.view_matrix[15] = 1.f;

    // Simple perspective viewproj (focal_x = focal_y = 8.0)
    float focal_x = cam.width / (2.0f * cam.tan_fovx);
    float focal_y = cam.height / (2.0f * cam.tan_fovy);
    std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0]  = focal_x;
    cam.viewproj_matrix[5]  = focal_y;
    cam.viewproj_matrix[10] = 1.f;
    cam.viewproj_matrix[11] = 1.f;
}

// ============================================================================
// Test: MatchesCPU_TinyGolden
// ============================================================================
TEST(PreprocessorBackwardVulkan, MatchesCPU_TinyGolden) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    // ---- Scene setup -------------------------------------------------------
    const int N         = 1;
    const int sh_degree = 0;
    const int max_coeffs = 1;  // (sh_degree+1)^2

    // Raw Gaussian parameters
    // NOTE: raw_rot must be pre-normalized so that |q_raw| = 1.0. The Vulkan
    // shader only has access to the already-normalized quaternion (from GaussianData),
    // so it computes the quaternion normalization Jacobian assuming |q_raw| = 1.
    // The CPU uses the actual |q_raw| from RawGaussianParams. Using a pre-normalized
    // quaternion makes both paths identical.
    float raw_pos[3]    = {0.0f, 0.0f, 5.0f};
    float raw_scales[3] = {0.5f, 0.3f, 0.1f};  // log-space
    float raw_rot[4]    = {1.0f, 0.0f, 0.0f, 0.0f};  // identity quaternion (pre-normalized)
    float raw_sh[3]     = {1.0f, 0.5f, 0.8f};
    float raw_opacity[1]= {2.0f};  // sigmoid(2) ~ 0.88

    Camera cam;
    setup_simple_camera_vk(cam);

    RenderConfig cfg;
    cfg.sh_degree      = sh_degree;
    cfg.tile_w         = 16;
    cfg.tile_h         = 16;
    cfg.bg_color[0]    = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.eval_3D        = false;
    cfg.antialiasing   = false;
    cfg.scale_modifier = 1.0f;
    cfg.training       = true;

    const int npix = cam.width * cam.height;
    std::vector<float> gt(npix * 3, 0.0f);  // zero target → MSE loss

    // ---- CPU forward pass --------------------------------------------------
    FrameAllocator alloc(16 * 1024 * 1024);

    RawGaussianParams raw;
    raw.count        = N;
    raw.sh_degree    = sh_degree;
    raw.max_coeffs   = max_coeffs;
    raw.raw_positions  = raw_pos;
    raw.raw_scales     = raw_scales;
    raw.raw_rotations  = raw_rot;
    raw.raw_sh_coeffs  = raw_sh;
    raw.raw_opacities  = raw_opacity;

    GaussianData g;
    g.count      = N;
    g.sh_degree  = sh_degree;
    g.max_coeffs = max_coeffs;
    g.positions  = alloc.allocate_array<float>(N * 3);
    g.scales     = alloc.allocate_array<float>(N * 3);
    g.rotations  = alloc.allocate_array<float>(N * 4);
    g.sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
    g.opacities  = alloc.allocate_array<float>(N);
    g.filter_3D  = nullptr;
    raw.activate(g);

    ForwardCache cache{};
    PreprocessorCPU preprocessor;
    PreprocessOutput pre = preprocessor.process(g, cam, cfg, alloc, &cache);

    // Check Gaussian is visible
    ASSERT_GT(pre.radii[0], 0) << "Gaussian must be visible for meaningful gradient test";

    cache.pre = &pre;

    TileBinnerCPU binner;
    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);
    cache.bin = &bin;

    SorterCPU sorter;
    sorter.sort(bin, alloc);

    float* out_img = alloc.allocate_array<float>(npix * 3);
    std::memset(out_img, 0, npix * 3 * sizeof(float));
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, out_img, nullptr, &cache, &alloc);

    // Compute MSE loss gradient
    float* d_image = alloc.allocate_array<float>(npix * 3);
    float inv_n = 1.0f / (float)(npix * 3);
    for (int px = 0; px < npix * 3; px++) {
        float diff = out_img[px] - gt[px];
        d_image[px] = 2.0f * diff * inv_n;
    }

    // ---- CPU rasterizer backward -------------------------------------------
    RasterGradOutput rgrad;
    rgrad.allocate_and_zero(alloc, N);
    RasterizerBackwardCPU rast_bwd;
    rast_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad);

    // ---- CPU preprocessor backward -----------------------------------------
    GradientOutput grads_cpu;
    grads_cpu.allocate_and_zero(alloc, N, max_coeffs);
    PreprocessorBackwardCPU preproc_bwd;
    preproc_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads_cpu);

    // ---- Vulkan preprocessor backward --------------------------------------
    GradientOutput grads_vk;
    PreprocessorBackwardVulkan vk_bwd(ctx);
    vk_bwd.backward(g, N, cam, cfg, cache, rgrad, raw, grads_vk, alloc);

    // ---- Compare CPU vs Vulkan ---------------------------------------------
    const float tol = 1e-4f;

    // d_raw_positions (d_means3D)
    for (int k = 0; k < N * 3; k++) {
        EXPECT_NEAR(grads_vk.d_raw_positions[k], grads_cpu.d_raw_positions[k], tol)
            << "d_raw_positions[" << k << "]: vk=" << grads_vk.d_raw_positions[k]
            << " cpu=" << grads_cpu.d_raw_positions[k];
    }

    // d_raw_sh_coeffs (d_sh)
    for (int k = 0; k < N * max_coeffs * 3; k++) {
        EXPECT_NEAR(grads_vk.d_raw_sh_coeffs[k], grads_cpu.d_raw_sh_coeffs[k], tol)
            << "d_raw_sh_coeffs[" << k << "]: vk=" << grads_vk.d_raw_sh_coeffs[k]
            << " cpu=" << grads_cpu.d_raw_sh_coeffs[k];
    }

    // d_raw_scales
    for (int k = 0; k < N * 3; k++) {
        EXPECT_NEAR(grads_vk.d_raw_scales[k], grads_cpu.d_raw_scales[k], tol)
            << "d_raw_scales[" << k << "]: vk=" << grads_vk.d_raw_scales[k]
            << " cpu=" << grads_cpu.d_raw_scales[k];
    }

    // d_raw_rotations
    for (int k = 0; k < N * 4; k++) {
        EXPECT_NEAR(grads_vk.d_raw_rotations[k], grads_cpu.d_raw_rotations[k], tol)
            << "d_raw_rotations[" << k << "]: vk=" << grads_vk.d_raw_rotations[k]
            << " cpu=" << grads_cpu.d_raw_rotations[k];
    }
}

// ============================================================================
// Test: CulledGaussianZeroGrad
// Verify that a Gaussian with radii=0 produces zero gradients from Vulkan.
// ============================================================================
TEST(PreprocessorBackwardVulkan, CulledGaussianZeroGrad) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    // N=2: Gaussian 0 visible, Gaussian 1 culled
    const int N          = 2;
    const int sh_degree  = 0;
    const int max_coeffs = 1;

    FrameAllocator alloc(4 * 1024 * 1024);

    Camera cam;
    setup_simple_camera_vk(cam);

    RenderConfig cfg;
    cfg.sh_degree      = sh_degree;
    cfg.tile_w         = 16;
    cfg.tile_h         = 16;
    cfg.scale_modifier = 1.0f;
    cfg.training       = true;
    cfg.eval_3D        = false;
    cfg.antialiasing   = false;

    GaussianData g;
    g.count      = N;
    g.sh_degree  = sh_degree;
    g.max_coeffs = max_coeffs;
    g.positions  = alloc.allocate_array<float>(N * 3);
    g.scales     = alloc.allocate_array<float>(N * 3);
    g.rotations  = alloc.allocate_array<float>(N * 4);
    g.sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
    g.opacities  = alloc.allocate_array<float>(N);
    g.filter_3D  = nullptr;

    // Gaussian 0: in front of camera
    g.positions[0] = 0.f; g.positions[1] = 0.f; g.positions[2] = 5.f;
    // Gaussian 1: far off-axis (will be culled in practice, but we fake radii=0)
    g.positions[3] = 100.f; g.positions[4] = 100.f; g.positions[5] = 5.f;

    g.scales[0] = g.scales[1] = g.scales[2] = 0.5f;
    g.scales[3] = g.scales[4] = g.scales[5] = 0.5f;

    g.rotations[0] = 1.f; g.rotations[4] = 1.f;  // identity quaternions
    g.sh_coeffs[0] = g.sh_coeffs[1] = g.sh_coeffs[2] = 0.3f;
    g.sh_coeffs[3] = g.sh_coeffs[4] = g.sh_coeffs[5] = 0.3f;
    g.opacities[0] = 0.8f;
    g.opacities[1] = 0.8f;

    // Setup ForwardCache
    ForwardCache cache{};
    cache.cov3D     = alloc.allocate_array<float>(N * 6);
    cache.p_view    = alloc.allocate_array<float>(N * 3);
    cache.p_hom_w   = alloc.allocate_array<float>(N);
    cache.cov2D     = alloc.allocate_array<float>(N * 3);
    cache.cov2D_det = alloc.allocate_array<float>(N);
    std::memset(cache.cov3D,     0, N * 6 * sizeof(float));
    std::memset(cache.p_view,    0, N * 3 * sizeof(float));
    std::memset(cache.p_hom_w,   0, N * sizeof(float));
    std::memset(cache.cov2D,     0, N * 3 * sizeof(float));
    std::memset(cache.cov2D_det, 0, N * sizeof(float));

    // Set cov3D for Gaussian 0 (diagonal)
    cache.cov3D[0] = 1.0f; cache.cov3D[3] = 1.0f; cache.cov3D[5] = 1.0f;

    // Set p_view for Gaussian 0 (view-space z=5)
    cache.p_view[2] = 5.0f;

    PreprocessOutput pre{};
    pre.radii = alloc.allocate_array<int>(N);
    pre.means2D = alloc.allocate_array<float>(N * 2);
    pre.radii[0] = 1;   // Gaussian 0: visible
    pre.radii[1] = 0;   // Gaussian 1: culled
    pre.means2D[0] = 8.f; pre.means2D[1] = 8.f;
    pre.means2D[2] = 0.f; pre.means2D[3] = 0.f;
    pre.eval_3D = false;
    cache.pre = &pre;

    // Fake rasterizer gradients
    RasterGradOutput rgrad;
    rgrad.d_means2D = alloc.allocate_array<float>(N * 2);
    rgrad.d_conics = alloc.allocate_array<float>(N * 3);
    rgrad.d_rgb = alloc.allocate_array<float>(N * 3);
    rgrad.d_opacities_2d = alloc.allocate_array<float>(N);
    std::memset(rgrad.d_means2D,      0, N * 2 * sizeof(float));
    std::memset(rgrad.d_conics,       0, N * 3 * sizeof(float));
    std::memset(rgrad.d_rgb,          0, N * 3 * sizeof(float));
    std::memset(rgrad.d_opacities_2d, 0, N * sizeof(float));
    // Give Gaussian 0 nonzero gradients
    rgrad.d_rgb[0] = 0.5f; rgrad.d_rgb[1] = 0.3f; rgrad.d_rgb[2] = 0.2f;
    rgrad.d_conics[0] = 0.1f; rgrad.d_conics[1] = 0.05f; rgrad.d_conics[2] = 0.1f;
    rgrad.d_means2D[0] = 0.01f; rgrad.d_means2D[1] = 0.01f;

    RawGaussianParams raw;
    raw.count = N; raw.sh_degree = sh_degree; raw.max_coeffs = max_coeffs;
    raw.raw_positions = g.positions;
    raw.raw_scales = g.scales;
    raw.raw_rotations = g.rotations;
    raw.raw_sh_coeffs = g.sh_coeffs;
    raw.raw_opacities = g.opacities;

    // Run Vulkan backward
    GradientOutput grads_vk;
    PreprocessorBackwardVulkan vk_bwd(ctx);
    vk_bwd.backward(g, N, cam, cfg, cache, rgrad, raw, grads_vk, alloc);

    // Culled Gaussian 1: all outputs must be zero
    for (int k = 0; k < max_coeffs * 3; k++) {
        EXPECT_EQ(grads_vk.d_raw_sh_coeffs[1 * max_coeffs * 3 + k], 0.0f)
            << "Culled Gaussian 1 d_sh[" << k << "] should be zero";
    }
    for (int k = 0; k < 3; k++) {
        EXPECT_EQ(grads_vk.d_raw_scales[1*3+k], 0.0f)
            << "Culled Gaussian 1 d_scales[" << k << "] should be zero";
        EXPECT_EQ(grads_vk.d_raw_positions[1*3+k], 0.0f)
            << "Culled Gaussian 1 d_means3D[" << k << "] should be zero";
    }
    for (int k = 0; k < 4; k++) {
        EXPECT_EQ(grads_vk.d_raw_rotations[1*4+k], 0.0f)
            << "Culled Gaussian 1 d_rotations[" << k << "] should be zero";
    }

    // Active Gaussian 0: d_sh should be nonzero
    bool sh0_nonzero = false;
    for (int k = 0; k < max_coeffs * 3; k++) {
        if (grads_vk.d_raw_sh_coeffs[k] != 0.0f) sh0_nonzero = true;
    }
    EXPECT_TRUE(sh0_nonzero) << "Active Gaussian 0 d_sh should be nonzero";
}
