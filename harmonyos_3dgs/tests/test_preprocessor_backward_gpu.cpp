// test_preprocessor_backward_gpu.cpp -- CPU-GPU cross-validation for preprocessor backward
// Runs identical inputs through CPU and GPU backward, compares gradient arrays.

#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include "cpu/preprocessor_backward_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/preprocessor_backward_gpu.h"
#include "test_gpu_context.h"
#endif

// Build a small test scenario with known forward cache values
struct PreprocessBwdTestData {
    static constexpr int N = 2;
    static constexpr int SH_DEGREE = 1;
    static constexpr int MAX_COEFFS = 4;  // (1+1)^2
    static constexpr int MC3 = MAX_COEFFS * 3;

    // Activated model data
    float positions[N * 3];
    float sh_coeffs[N * MC3];
    float scales[N * 3];
    float rotations[N * 4];  // normalized quaternions
    float opacities[N];

    // Raw params (for quaternion normalization backward)
    float raw_rotations[N * 4];

    // Camera
    Camera cam;
    RenderConfig cfg;

    // Forward cache
    float cov2D[N * 3];
    float cov2D_det[N];
    float p_view[N * 3];
    float p_hom_w[N];
    float means2D[N * 2];
    int radii[N];

    // Rasterizer gradients (input to preprocessor backward)
    float d_means2D[N * 2];
    float d_conics[N * 3];
    float d_rgb[N * 3];
    float d_opacities_2d[N];

    GaussianData g;
    ForwardCache cache;
    PreprocessOutput pre;
    RasterGradOutput rgrad;
    RawGaussianParams raw;

    PreprocessBwdTestData() {
        // Positions
        positions[0] = 1.0f; positions[1] = 0.5f; positions[2] = 5.0f;
        positions[3] = -0.5f; positions[4] = 1.0f; positions[5] = 7.0f;

        // SH coefficients (degree 1, 4 basis functions, 3 channels)
        for (int i = 0; i < N * MC3; i++)
            sh_coeffs[i] = 0.1f * (float)(i % 5 + 1);

        // Scales (already exp-activated)
        scales[0] = 0.1f; scales[1] = 0.2f; scales[2] = 0.15f;
        scales[3] = 0.12f; scales[4] = 0.18f; scales[5] = 0.1f;

        // Normalized quaternions
        rotations[0] = 0.9239f; rotations[1] = 0.0f; rotations[2] = 0.3827f; rotations[3] = 0.0f;
        rotations[4] = 0.7071f; rotations[5] = 0.7071f; rotations[6] = 0.0f; rotations[7] = 0.0f;

        // Raw rotations (before normalization - use same as normalized for simplicity)
        for (int i = 0; i < N * 4; i++)
            raw_rotations[i] = rotations[i] * 1.1f;  // slightly unnormalized

        opacities[0] = 0.8f; opacities[1] = 0.6f;

        // Camera
        cam = {};
        cam.width = 32; cam.height = 32;
        cam.tan_fovx = 0.5f; cam.tan_fovy = 0.5f;

        // Simple view matrix (identity rotation, slight translation)
        std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
        cam.view_matrix[0] = 1.0f; cam.view_matrix[5] = 1.0f;
        cam.view_matrix[10] = 1.0f; cam.view_matrix[15] = 1.0f;

        // Simple viewproj (perspective-like)
        std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
        float fx = cam.width / (2.0f * cam.tan_fovx);
        float fy = cam.height / (2.0f * cam.tan_fovy);
        cam.viewproj_matrix[0] = fx / 16.0f;  // scaled
        cam.viewproj_matrix[5] = fy / 16.0f;
        cam.viewproj_matrix[10] = 1.0f;
        cam.viewproj_matrix[11] = 1.0f;

        cam.cam_pos[0] = 0.0f; cam.cam_pos[1] = 0.0f; cam.cam_pos[2] = 0.0f;

        cfg = {};
        cfg.scale_modifier = 1.0f;
        cfg.sh_degree = SH_DEGREE;

        // Forward cache: synthetic values
        cov2D[0] = 5.0f; cov2D[1] = 0.5f; cov2D[2] = 4.0f;
        cov2D[3] = 3.0f; cov2D[4] = -0.3f; cov2D[5] = 6.0f;

        for (int i = 0; i < N; i++)
            cov2D_det[i] = cov2D[i*3] * cov2D[i*3+2] - cov2D[i*3+1] * cov2D[i*3+1];

        p_view[0] = 1.0f; p_view[1] = 0.5f; p_view[2] = 5.0f;
        p_view[3] = -0.5f; p_view[4] = 1.0f; p_view[5] = 7.0f;

        p_hom_w[0] = 5.0f; p_hom_w[1] = 7.0f;

        means2D[0] = 16.0f; means2D[1] = 16.5f;
        means2D[2] = 14.0f; means2D[3] = 18.0f;

        radii[0] = 8; radii[1] = 6;

        // Rasterizer gradients (synthetic)
        for (int i = 0; i < N * 2; i++) d_means2D[i] = 0.01f * (float)(i + 1);
        for (int i = 0; i < N * 3; i++) d_conics[i] = 0.005f * (float)(i + 1);
        for (int i = 0; i < N * 3; i++) d_rgb[i] = 0.02f * (float)(i + 1);
        d_opacities_2d[0] = 0.03f; d_opacities_2d[1] = 0.04f;

        // Wire up structs
        g.count = N;
        g.sh_degree = SH_DEGREE;
        g.max_coeffs = MAX_COEFFS;
        g.positions = positions;
        g.sh_coeffs = sh_coeffs;
        g.scales = scales;
        g.rotations = rotations;
        g.opacities = opacities;
        g.filter_3D = nullptr;

        pre.means2D = means2D;
        pre.radii = radii;

        cache.cov2D = cov2D;
        cache.cov2D_det = cov2D_det;
        cache.p_view = p_view;
        cache.p_hom_w = p_hom_w;
        cache.pre = &pre;

        rgrad.d_means2D = d_means2D;
        rgrad.d_conics = d_conics;
        rgrad.d_rgb = d_rgb;
        rgrad.d_opacities_2d = d_opacities_2d;

        raw.count = N;
        raw.sh_degree = SH_DEGREE;
        raw.max_coeffs = MAX_COEFFS;
        raw.raw_positions = positions;  // identity activation for positions
        raw.raw_scales = scales;
        raw.raw_rotations = raw_rotations;
        raw.raw_sh_coeffs = sh_coeffs;
        raw.raw_opacities = opacities;
    }
};

TEST(PreprocessorBackwardGPU, MatchesCPU) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    PreprocessBwdTestData data;
    constexpr int N = PreprocessBwdTestData::N;
    constexpr int MC3 = PreprocessBwdTestData::MC3;
    constexpr int MAX_COEFFS = PreprocessBwdTestData::MAX_COEFFS;

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU backward
    GradientOutput grads_cpu;
    grads_cpu.allocate_and_zero(alloc, N, MAX_COEFFS);
    PreprocessorBackwardCPU cpu_bwd;
    cpu_bwd.backward(data.g, data.cam, data.cfg, data.cache, data.rgrad, data.raw, grads_cpu);

    // GPU backward
    GradientOutput grads_gpu;
    grads_gpu.allocate_and_zero(alloc, N, MAX_COEFFS);
    PreprocessorBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(data.g, data.cam, data.cfg, data.cache, data.rgrad, data.raw, grads_gpu);

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

    check("d_raw_positions",  grads_cpu.d_raw_positions,  grads_gpu.d_raw_positions,  N * 3);
    check("d_raw_scales",     grads_cpu.d_raw_scales,     grads_gpu.d_raw_scales,     N * 3);
    check("d_raw_rotations",  grads_cpu.d_raw_rotations,  grads_gpu.d_raw_rotations,  N * 4);
    check("d_raw_sh_coeffs",  grads_cpu.d_raw_sh_coeffs,  grads_gpu.d_raw_sh_coeffs,  N * MC3);
    check("d_raw_opacities",  grads_cpu.d_raw_opacities,  grads_gpu.d_raw_opacities,  N);

    std::fprintf(stderr, "PreprocessorBackwardGPU: max relative error = %e\n", max_rel_err);
#endif
}

// Test SH degree 0 backward (simplest case)
TEST(PreprocessorBackwardGPU, SH_Degree0) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 2;
    constexpr int SH_DEGREE = 0;
    constexpr int MAX_COEFFS = 1;
    constexpr int MC3 = MAX_COEFFS * 3;

    float positions[N * 3] = {0, 0, 5,  1, 0, 6};
    float sh_coeffs[N * MC3] = {0.5f, 0.4f, 0.3f,  0.3f, 0.5f, 0.4f};
    float scales[N * 3] = {0.1f, 0.1f, 0.1f,  0.15f, 0.15f, 0.15f};
    float rotations[N * 4] = {1, 0, 0, 0,  1, 0, 0, 0};  // Identity
    float opacities[N] = {0.8f, 0.7f};

    Camera cam{};
    cam.width = 32; cam.height = 32;
    cam.tan_fovx = 0.5f; cam.tan_fovy = 0.5f;
    cam.view_matrix[0] = 1; cam.view_matrix[5] = 1; cam.view_matrix[10] = 1; cam.view_matrix[15] = 1;
    float fx = 32 / (2.0f * 0.5f);
    float fy = 32 / (2.0f * 0.5f);
    cam.viewproj_matrix[0] = fx; cam.viewproj_matrix[5] = fy; cam.viewproj_matrix[10] = 1; cam.viewproj_matrix[11] = 1;

    RenderConfig cfg{};
    cfg.scale_modifier = 1.0f;
    cfg.sh_degree = SH_DEGREE;

    // Forward cache
    float cov2D[N * 3] = {4, 0, 4,  5, 0, 5};
    float cov2D_det[N] = {16, 25};
    float p_view[N * 3] = {0, 0, 5,  1, 0, 6};
    float p_hom_w[N] = {5, 6};
    float means2D[N * 2] = {16, 16,  18, 16};
    int radii[N] = {8, 8};

    // Rasterizer gradients
    float d_means2D[N * 2] = {0.01f, 0.02f,  0.03f, 0.04f};
    float d_conics[N * 3] = {0.001f, 0, 0.001f,  0.002f, 0, 0.002f};
    float d_rgb[N * 3] = {0.1f, 0.1f, 0.1f,  0.2f, 0.2f, 0.2f};
    float d_opacities_2d[N] = {0.05f, 0.04f};

    GaussianData g{};
    g.count = N;
    g.sh_degree = SH_DEGREE;
    g.max_coeffs = MAX_COEFFS;
    g.positions = positions;
    g.sh_coeffs = sh_coeffs;
    g.scales = scales;
    g.rotations = rotations;
    g.opacities = opacities;
    g.filter_3D = nullptr;

    PreprocessOutput pre{};
    pre.means2D = means2D;
    pre.radii = radii;

    ForwardCache cache{};
    cache.cov2D = cov2D;
    cache.cov2D_det = cov2D_det;
    cache.p_view = p_view;
    cache.p_hom_w = p_hom_w;
    cache.pre = &pre;

    RasterGradOutput rgrad{};
    rgrad.d_means2D = d_means2D;
    rgrad.d_conics = d_conics;
    rgrad.d_rgb = d_rgb;
    rgrad.d_opacities_2d = d_opacities_2d;

    RawGaussianParams raw{};
    raw.count = N;
    raw.sh_degree = SH_DEGREE;
    raw.max_coeffs = MAX_COEFFS;
    raw.raw_positions = positions;
    raw.raw_scales = scales;
    raw.raw_rotations = rotations;
    raw.raw_sh_coeffs = sh_coeffs;
    raw.raw_opacities = opacities;

    FrameAllocator alloc(16 * 1024 * 1024);

    GradientOutput grads_cpu, grads_gpu;
    grads_cpu.allocate_and_zero(alloc, N, MAX_COEFFS);
    grads_gpu.allocate_and_zero(alloc, N, MAX_COEFFS);

    PreprocessorBackwardCPU cpu_bwd;
    cpu_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads_cpu);

    PreprocessorBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads_gpu);

    float max_rel_err = 0.0f;
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            float denom = std::max(std::fabs(cpu[i]), std::fabs(gpu[i]));
            float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;
            max_rel_err = std::max(max_rel_err, rel_err);
            EXPECT_LT(rel_err, tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    check("d_raw_sh_coeffs", grads_cpu.d_raw_sh_coeffs, grads_gpu.d_raw_sh_coeffs, N * MC3, 1e-3f);
    check("d_raw_opacities", grads_cpu.d_raw_opacities, grads_gpu.d_raw_opacities, N, 1e-2f);
    check("d_raw_scales", grads_cpu.d_raw_scales, grads_gpu.d_raw_scales, N * 3, 1e-2f);
    check("d_raw_rotations", grads_cpu.d_raw_rotations, grads_gpu.d_raw_rotations, N * 4, 1e-2f);

    std::fprintf(stderr, "PreprocessorBackwardGPU (SH0): max rel error = %e\n", max_rel_err);
#endif
}

// Test with culled Gaussians (radii <= 0 should produce zero gradients)
TEST(PreprocessorBackwardGPU, CulledGaussians) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    constexpr int N = 3;
    constexpr int SH_DEGREE = 0;
    constexpr int MAX_COEFFS = 1;

    float positions[N * 3] = {0, 0, 5,  1, 0, -1,  2, 0, 6};  // Middle one behind camera
    float sh_coeffs[N * 3] = {0.5f, 0.4f, 0.3f,  0.3f, 0.5f, 0.4f,  0.4f, 0.4f, 0.4f};
    float scales[N * 3] = {0.1f, 0.1f, 0.1f,  0.1f, 0.1f, 0.1f,  0.1f, 0.1f, 0.1f};
    float rotations[N * 4] = {1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0};
    float opacities[N] = {0.8f, 0.7f, 0.6f};

    Camera cam{};
    cam.width = 32; cam.height = 32;
    cam.tan_fovx = 0.5f; cam.tan_fovy = 0.5f;
    cam.view_matrix[0] = 1; cam.view_matrix[5] = 1; cam.view_matrix[10] = 1; cam.view_matrix[15] = 1;
    float fx = 32;
    float fy = 32;
    cam.viewproj_matrix[0] = fx; cam.viewproj_matrix[5] = fy; cam.viewproj_matrix[10] = 1; cam.viewproj_matrix[11] = 1;

    RenderConfig cfg{};
    cfg.scale_modifier = 1.0f;
    cfg.sh_degree = SH_DEGREE;

    // Forward cache - middle Gaussian culled (radius = 0)
    float cov2D[N * 3] = {4, 0, 4,  0, 0, 0,  5, 0, 5};
    float cov2D_det[N] = {16, 0, 25};
    float p_view[N * 3] = {0, 0, 5,  1, 0, -1,  2, 0, 6};
    float p_hom_w[N] = {5, -1, 6};
    float means2D[N * 2] = {16, 16,  0, 0,  18, 16};
    int radii[N] = {8, 0, 8};  // Middle culled

    float d_means2D[N * 2] = {0.01f, 0.02f,  0, 0,  0.03f, 0.04f};
    float d_conics[N * 3] = {0.001f, 0, 0.001f,  0, 0, 0,  0.002f, 0, 0.002f};
    float d_rgb[N * 3] = {0.1f, 0.1f, 0.1f,  0, 0, 0,  0.2f, 0.2f, 0.2f};
    float d_opacities_2d[N] = {0.05f, 0, 0.04f};

    GaussianData g{};
    g.count = N;
    g.sh_degree = SH_DEGREE;
    g.max_coeffs = MAX_COEFFS;
    g.positions = positions;
    g.sh_coeffs = sh_coeffs;
    g.scales = scales;
    g.rotations = rotations;
    g.opacities = opacities;

    PreprocessOutput pre{};
    pre.means2D = means2D;
    pre.radii = radii;

    ForwardCache cache{};
    cache.cov2D = cov2D;
    cache.cov2D_det = cov2D_det;
    cache.p_view = p_view;
    cache.p_hom_w = p_hom_w;
    cache.pre = &pre;

    RasterGradOutput rgrad{};
    rgrad.d_means2D = d_means2D;
    rgrad.d_conics = d_conics;
    rgrad.d_rgb = d_rgb;
    rgrad.d_opacities_2d = d_opacities_2d;

    RawGaussianParams raw{};
    raw.count = N;
    raw.sh_degree = SH_DEGREE;
    raw.max_coeffs = MAX_COEFFS;
    raw.raw_positions = positions;
    raw.raw_scales = scales;
    raw.raw_rotations = rotations;
    raw.raw_sh_coeffs = sh_coeffs;
    raw.raw_opacities = opacities;

    FrameAllocator alloc(16 * 1024 * 1024);

    GradientOutput grads_cpu, grads_gpu;
    grads_cpu.allocate_and_zero(alloc, N, MAX_COEFFS);
    grads_gpu.allocate_and_zero(alloc, N, MAX_COEFFS);

    PreprocessorBackwardCPU cpu_bwd;
    cpu_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads_cpu);

    PreprocessorBackwardGPU gpu_bwd(ctx);
    gpu_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads_gpu);

    // Culled Gaussian should have zero gradients in both CPU and GPU
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(grads_cpu.d_raw_positions[1 * 3 + i], 0, 1e-6f) << "culled position grad";
        EXPECT_NEAR(grads_gpu.d_raw_positions[1 * 3 + i], 0, 1e-6f) << "culled position grad GPU";
    }
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(grads_cpu.d_raw_sh_coeffs[1 * 3 + i], 0, 1e-6f) << "culled SH grad";
        EXPECT_NEAR(grads_gpu.d_raw_sh_coeffs[1 * 3 + i], 0, 1e-6f) << "culled SH grad GPU";
    }

    std::fprintf(stderr, "PreprocessorBackwardGPU (Culled): culled Gaussian gradients are zero as expected\n");
#endif
}
