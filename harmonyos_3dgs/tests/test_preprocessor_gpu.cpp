// test_preprocessor_gpu.cpp -- CPU-GPU cross-validation for preprocessor forward pass.
// Runs identical inputs through CPU and GPU preprocessor, compares output arrays.

#include <gtest/gtest.h>
#include "types.h"
#include "cpu/preprocessor_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/preprocessor_gpu.h"
#include "test_gpu_context.h"
#endif

// Small test scene: 32x32 image, 5 Gaussians
struct PreprocessorGpuTestScene {
    static constexpr int N = 5;
    static constexpr int W = 32;
    static constexpr int H = 32;
    static constexpr int SH_DEGREE = 1;
    static constexpr int MAX_COEFFS = 4;  // (1+1)^2

    GaussianData g;
    Camera cam;
    RenderConfig cfg;

    // Raw model data
    float positions[N * 3];
    float sh_coeffs[N * MAX_COEFFS * 3];
    float scales[N * 3];
    float rotations[N * 4];
    float opacities[N];

    PreprocessorGpuTestScene() {
        // Initialize positions - spread across scene
        for (int i = 0; i < N; i++) {
            positions[i * 3] = -2.0f + i * 1.0f;
            positions[i * 3 + 1] = -1.0f + i * 0.5f;
            positions[i * 3 + 2] = 4.0f + i * 0.5f;
        }

        // SH coefficients (degree 1)
        for (int i = 0; i < N * MAX_COEFFS * 3; i++) {
            sh_coeffs[i] = 0.1f * (float)((i % 7) + 1);
        }

        // Scales
        for (int i = 0; i < N; i++) {
            scales[i * 3] = 0.1f + i * 0.02f;
            scales[i * 3 + 1] = 0.12f + i * 0.02f;
            scales[i * 3 + 2] = 0.08f + i * 0.02f;
        }

        // Rotations (normalized quaternions)
        for (int i = 0; i < N; i++) {
            float angle = 0.1f * i;
            rotations[i * 4] = std::cos(angle);
            rotations[i * 4 + 1] = std::sin(angle) * 0.3f;
            rotations[i * 4 + 2] = std::sin(angle) * 0.5f;
            rotations[i * 4 + 3] = std::sin(angle) * 0.7f;
            // Normalize
            float len = std::sqrt(rotations[i * 4] * rotations[i * 4] +
                                  rotations[i * 4 + 1] * rotations[i * 4 + 1] +
                                  rotations[i * 4 + 2] * rotations[i * 4 + 2] +
                                  rotations[i * 4 + 3] * rotations[i * 4 + 3]);
            for (int j = 0; j < 4; j++) rotations[i * 4 + j] /= len;
        }

        // Opacities (sigmoid-activated)
        for (int i = 0; i < N; i++) {
            opacities[i] = 0.5f + i * 0.1f;
        }

        // Wire up GaussianData
        g.count = N;
        g.sh_degree = SH_DEGREE;
        g.max_coeffs = MAX_COEFFS;
        g.positions = positions;
        g.sh_coeffs = sh_coeffs;
        g.scales = scales;
        g.rotations = rotations;
        g.opacities = opacities;
        g.filter_3D = nullptr;

        // Camera
        cam = {};
        cam.width = W;
        cam.height = H;
        cam.tan_fovx = 0.5f;
        cam.tan_fovy = 0.5f;

        // View matrix (identity)
        std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
        cam.view_matrix[0] = 1.0f; cam.view_matrix[5] = 1.0f;
        cam.view_matrix[10] = 1.0f; cam.view_matrix[15] = 1.0f;

        // ViewProj matrix
        float fx = W / (2.0f * cam.tan_fovx);
        float fy = H / (2.0f * cam.tan_fovy);
        std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
        cam.viewproj_matrix[0] = fx;
        cam.viewproj_matrix[5] = fy;
        cam.viewproj_matrix[10] = 1.0f;
        cam.viewproj_matrix[11] = 1.0f;

        cam.cam_pos[0] = 0.0f;
        cam.cam_pos[1] = 0.0f;
        cam.cam_pos[2] = 0.0f;

        // Render config
        cfg = {};
        cfg.scale_modifier = 1.0f;
        cfg.sh_degree = SH_DEGREE;
        cfg.tile_w = 16;
        cfg.tile_h = 16;
        cfg.eval_3D = false;
        cfg.antialiasing = false;
        cfg.training = true;
    }
};

TEST(PreprocessorGPU, MatchesCPU_Basic) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    PreprocessorGpuTestScene scene;
    constexpr int N = PreprocessorGpuTestScene::N;

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU preprocessor
    PreprocessOutput out_cpu;
    {
        PreprocessorCPU cpu_proc;
        out_cpu = cpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // GPU preprocessor
    PreprocessOutput out_gpu;
    {
        PreprocessorGPU gpu_proc(ctx);
        out_gpu = gpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // Compare arrays
    float max_rel_err = 0.0f;
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            float denom = std::max(std::fabs(cpu[i]), std::fabs(gpu[i]));
            float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;
            max_rel_err = std::max(max_rel_err, rel_err);
            EXPECT_NEAR(cpu[i], gpu[i], tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    // Check means2D
    check("means2D", out_cpu.means2D, out_gpu.means2D, N * 2, 1e-4f);

    // Check depths
    check("depths", out_cpu.depths, out_gpu.depths, N, 1e-5f);

    // Check conics
    check("conics", out_cpu.conics, out_gpu.conics, N * 3, 1e-4f);

    // Check rgb
    check("rgb", out_cpu.rgb, out_gpu.rgb, N * 3, 1e-4f);

    // Check opacities_2d
    check("opacities_2d", out_cpu.opacities_2d, out_gpu.opacities_2d, N, 1e-4f);

    // Check radii (should be exact match)
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(out_cpu.radii[i], out_gpu.radii[i])
            << "radii[" << i << "] mismatch";
    }

    // Check tiles_touched
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(out_cpu.tiles_touched[i], out_gpu.tiles_touched[i])
            << "tiles_touched[" << i << "] mismatch";
    }

    std::fprintf(stderr, "PreprocessorGPU: max relative error = %e\n", max_rel_err);
#endif
}

TEST(PreprocessorGPU, MatchesCPU_2DGaussian) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    // Test with elongated 2D Gaussian
    PreprocessorGpuTestScene scene;
    constexpr int N = PreprocessorGpuTestScene::N;

    // Make first Gaussian highly elongated
    scene.scales[0] = 0.5f;   // elongated in x
    scene.scales[1] = 0.05f;  // narrow in y
    scene.scales[2] = 0.1f;

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU
    PreprocessOutput out_cpu;
    {
        PreprocessorCPU cpu_proc;
        out_cpu = cpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // GPU
    PreprocessOutput out_gpu;
    {
        PreprocessorGPU gpu_proc(ctx);
        out_gpu = gpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // Compare
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            EXPECT_NEAR(cpu[i], gpu[i], tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    check("means2D", out_cpu.means2D, out_gpu.means2D, N * 2, 1e-4f);
    check("depths", out_cpu.depths, out_gpu.depths, N, 1e-5f);
    check("conics", out_cpu.conics, out_gpu.conics, N * 3, 1e-4f);
    check("rgb", out_cpu.rgb, out_gpu.rgb, N * 3, 1e-4f);
#endif
}

// Test scene for eval_3D mode
struct PreprocessorEval3DTestScene {
    static constexpr int N = 3;
    static constexpr int W = 32;
    static constexpr int H = 32;
    static constexpr int SH_DEGREE = 0;  // Use degree 0 for simplicity
    static constexpr int MAX_COEFFS = 1;

    GaussianData g;
    Camera cam;
    RenderConfig cfg;

    float positions[N * 3];
    float sh_coeffs[N * MAX_COEFFS * 3];
    float scales[N * 3];
    float rotations[N * 4];
    float opacities[N];
    float filter_3D[N];

    PreprocessorEval3DTestScene() {
        // Positions in front of camera
        for (int i = 0; i < N; i++) {
            positions[i * 3] = i * 0.5f - 0.5f;
            positions[i * 3 + 1] = i * 0.3f - 0.3f;
            positions[i * 3 + 2] = 3.0f + i * 0.5f;
        }

        // SH coefficients (degree 0, DC only)
        for (int i = 0; i < N * 3; i++) {
            sh_coeffs[i] = 0.5f;
        }

        // Scales
        for (int i = 0; i < N; i++) {
            scales[i * 3] = 0.1f;
            scales[i * 3 + 1] = 0.1f;
            scales[i * 3 + 2] = 0.1f;
        }

        // Identity rotations
        for (int i = 0; i < N; i++) {
            rotations[i * 4] = 1.0f;
            rotations[i * 4 + 1] = 0.0f;
            rotations[i * 4 + 2] = 0.0f;
            rotations[i * 4 + 3] = 0.0f;
        }

        // Opacities
        for (int i = 0; i < N; i++) {
            opacities[i] = 0.8f;
        }

        // Filter 3D (dilation factor)
        for (int i = 0; i < N; i++) {
            filter_3D[i] = 0.0f;
        }

        g.count = N;
        g.sh_degree = SH_DEGREE;
        g.max_coeffs = MAX_COEFFS;
        g.positions = positions;
        g.sh_coeffs = sh_coeffs;
        g.scales = scales;
        g.rotations = rotations;
        g.opacities = opacities;
        g.filter_3D = filter_3D;

        cam = {};
        cam.width = W;
        cam.height = H;
        cam.tan_fovx = 0.5f;
        cam.tan_fovy = 0.5f;

        // Identity view matrix
        std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
        cam.view_matrix[0] = 1.0f; cam.view_matrix[5] = 1.0f;
        cam.view_matrix[10] = 1.0f; cam.view_matrix[15] = 1.0f;

        // ViewProj
        float fx = W / (2.0f * cam.tan_fovx);
        float fy = H / (2.0f * cam.tan_fovy);
        std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
        cam.viewproj_matrix[0] = fx;
        cam.viewproj_matrix[5] = fy;
        cam.viewproj_matrix[10] = 1.0f;
        cam.viewproj_matrix[11] = 1.0f;

        cam.cam_pos[0] = 0.0f;
        cam.cam_pos[1] = 0.0f;
        cam.cam_pos[2] = 0.0f;

        cfg = {};
        cfg.scale_modifier = 1.0f;
        cfg.sh_degree = SH_DEGREE;
        cfg.tile_w = 16;
        cfg.tile_h = 16;
        cfg.eval_3D = true;  // Enable 3D evaluation
        cfg.antialiasing = false;
        cfg.training = true;
    }
};

TEST(PreprocessorGPU, MatchesCPU_Eval3D) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    PreprocessorEval3DTestScene scene;
    constexpr int N = PreprocessorEval3DTestScene::N;

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU preprocessor with eval_3D
    PreprocessOutput out_cpu;
    {
        PreprocessorCPU cpu_proc;
        out_cpu = cpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // GPU preprocessor with eval_3D
    PreprocessOutput out_gpu;
    {
        PreprocessorGPU gpu_proc(ctx);
        out_gpu = gpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // Compare arrays
    auto check = [&](const char* name, const float* cpu, const float* gpu, int count, float tol) {
        for (int i = 0; i < count; i++) {
            float abs_err = std::fabs(cpu[i] - gpu[i]);
            EXPECT_NEAR(cpu[i], gpu[i], tol)
                << name << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i];
        }
    };

    // Check radii (should be exact match)
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(out_cpu.radii[i], out_gpu.radii[i])
            << "radii[" << i << "] mismatch";
    }

    // Check tiles_touched
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(out_cpu.tiles_touched[i], out_gpu.tiles_touched[i])
            << "tiles_touched[" << i << "] mismatch";
    }

    // Only compare valid Gaussians (radii > 0)
    int valid_count = 0;
    for (int i = 0; i < N; i++) {
        if (out_cpu.radii[i] > 0) {
            valid_count++;
            // Check means2D for valid Gaussians
            check("means2D", out_cpu.means2D + i * 2, out_gpu.means2D + i * 2, 2, 1e-3f);
            check("depths", out_cpu.depths + i, out_gpu.depths + i, 1, 1e-4f);
            check("rgb", out_cpu.rgb + i * 3, out_gpu.rgb + i * 3, 3, 1e-3f);
            check("opacities_2d", out_cpu.opacities_2d + i, out_gpu.opacities_2d + i, 1, 1e-3f);
        }
    }

    std::fprintf(stderr, "PreprocessorGPU (Eval3D): %d/%d valid Gaussians\n", valid_count, N);
#endif
}

TEST(PreprocessorGPU, Culling_BehindCamera) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    PreprocessorGpuTestScene scene;
    constexpr int N = PreprocessorGpuTestScene::N;

    // Put first Gaussian behind camera (negative z in view space)
    scene.positions[2] = -1.0f;  // z = -1, behind camera

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU
    PreprocessOutput out_cpu;
    {
        PreprocessorCPU cpu_proc;
        out_cpu = cpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // GPU
    PreprocessOutput out_gpu;
    {
        PreprocessorGPU gpu_proc(ctx);
        out_gpu = gpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // First Gaussian should be culled (radius = 0)
    EXPECT_EQ(out_cpu.radii[0], 0);
    EXPECT_EQ(out_gpu.radii[0], 0);

    // Other Gaussians should still be valid
    for (int i = 1; i < N; i++) {
        EXPECT_GT(out_cpu.radii[i], 0);
        EXPECT_GT(out_gpu.radii[i], 0);
    }
#endif
}

TEST(PreprocessorGPU, Culling_NearPlane) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    PreprocessorGpuTestScene scene;
    constexpr int N = PreprocessorGpuTestScene::N;

    // Put first Gaussian very close to near plane (z = 0.2)
    scene.positions[2] = 0.25f;  // Just above near plane

    FrameAllocator alloc(16 * 1024 * 1024);

    // CPU
    PreprocessOutput out_cpu;
    {
        PreprocessorCPU cpu_proc;
        out_cpu = cpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // GPU
    PreprocessOutput out_gpu;
    {
        PreprocessorGPU gpu_proc(ctx);
        out_gpu = gpu_proc.process(scene.g, scene.cam, scene.cfg, alloc, nullptr);
    }

    // Compare radii
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(out_cpu.radii[i], out_gpu.radii[i])
            << "radii[" << i << "] mismatch near plane";
    }
#endif
}
