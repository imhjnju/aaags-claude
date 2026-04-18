// test_preprocessor_vk.cpp -- Incremental TDD gate for PreprocessorVK.
// Each slice compares GPU output against a tight CPU oracle (direct calls
// to the reference helper functions in math_utils.h), NOT against
// PreprocessorCPU end-to-end, to isolate each slice.

#include "math_utils.h"
#include "types.h"
#include "vulkan/preprocessor_vk.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Owns backing storage for a GaussianData view. Default scales=(1,1,1),
// rotations=(1,0,0,0)=identity quat, opacities=1. Positions start at 0 so
// callers overwrite them.
struct TestGaussians {
    std::vector<float>    positions;
    std::vector<float>    scales;
    std::vector<float>    rotations;
    std::vector<float>    opacities;
    int                   count = 0;

    GaussianData view() {
        GaussianData g{};
        g.count       = count;
        g.sh_degree   = 0;
        g.max_coeffs  = 1;
        g.positions   = positions.data();
        g.scales      = scales.data();
        g.rotations   = rotations.data();
        g.opacities   = opacities.data();
        // sh_coeffs, filter_3D left null (unused by current slices)
        return g;
    }
};

TestGaussians make_default_gaussians(int N) {
    TestGaussians g;
    g.count = N;
    g.positions.assign(N * 3, 0.0f);
    g.scales.assign(N * 3, 1.0f);
    g.rotations.assign(N * 4, 0.0f);
    for (int i = 0; i < N; ++i) g.rotations[i * 4] = 1.0f;  // quat r=1 (identity)
    g.opacities.assign(N, 1.0f);
    return g;
}

Camera make_camera_identity_view() {
    Camera c{};
    const float I[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    std::memcpy(c.view_matrix,     I, sizeof(I));
    std::memcpy(c.viewproj_matrix, I, sizeof(I));
    c.cam_pos[0] = c.cam_pos[1] = c.cam_pos[2] = 0.0f;
    c.tan_fovx = c.tan_fovy = 1.0f;
    c.width = c.height = 512;
    return c;
}

Camera make_camera_translated_view() {
    Camera c = make_camera_identity_view();
    c.view_matrix[12] = 0.0f;
    c.view_matrix[13] = 0.0f;
    c.view_matrix[14] = -1.0f;
    c.view_matrix[15] = 1.0f;
    return c;
}

Camera make_camera_rotated_view() {
    Camera c{};
    const float ang = 0.5235987756f;
    const float co = std::cos(ang), si = std::sin(ang);
    const float V[16] = {
         co, si, 0, 0,
        -si, co, 0, 0,
          0,  0, 1, 0,
         0.1f, -0.2f, -0.5f, 1,
    };
    const float I[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    std::memcpy(c.view_matrix,     V, sizeof(V));
    std::memcpy(c.viewproj_matrix, I, sizeof(I));
    c.cam_pos[0] = c.cam_pos[1] = c.cam_pos[2] = 0.0f;
    c.tan_fovx = c.tan_fovy = 1.0f;
    c.width = c.height = 512;
    return c;
}

Camera make_camera_trivial_perspective() {
    Camera c = make_camera_identity_view();
    float P[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 1,
        0, 0, 0, 0,
    };
    std::memcpy(c.viewproj_matrix, P, sizeof(P));
    c.width = 800;
    c.height = 600;
    return c;
}

}  // namespace

// ============================================================================
// Slice 2a — world->view transform + near-plane cull
// ============================================================================
TEST(PreprocessorVK_Slice2a, IdentityTranslate_MatchesCPU) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    constexpr int N = 100;
    auto g = make_default_gaussians(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (float& v : g.positions) v = d(rng);

    Camera cam = make_camera_translated_view();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);

    ASSERT_EQ((int)out.depths.size(), N);
    ASSERT_EQ((int)out.active.size(), N);
    for (int i = 0; i < N; ++i) {
        float p_view[3];
        transformPoint4x3(&g.positions[i * 3], cam.view_matrix, p_view);
        EXPECT_NEAR(out.depths[i], p_view[2], 1e-5f) << "i=" << i;
        uint32_t expected = (p_view[2] > 0.2f) ? 1u : 0u;
        EXPECT_EQ(out.active[i], expected) << "i=" << i;
    }
}

TEST(PreprocessorVK_Slice2a, RotatedView_MatchesCPU) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    constexpr int N = 256;
    auto g = make_default_gaussians(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    for (float& v : g.positions) v = d(rng);

    Camera cam = make_camera_rotated_view();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);

    int active_count = 0;
    for (int i = 0; i < N; ++i) {
        float p_view[3];
        transformPoint4x3(&g.positions[i * 3], cam.view_matrix, p_view);
        EXPECT_NEAR(out.depths[i], p_view[2], 1e-5f) << "i=" << i;
        uint32_t expected = (p_view[2] > 0.2f) ? 1u : 0u;
        EXPECT_EQ(out.active[i], expected) << "i=" << i;
        active_count += static_cast<int>(expected);
    }
    EXPECT_GT(active_count, 0);
    EXPECT_LT(active_count, N);
}

TEST(PreprocessorVK_Slice2a, NearPlaneBoundary_StrictCull) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    auto g = make_default_gaussians(4);
    g.positions = {
        0.0f, 0.0f, 0.199999f,
        0.0f, 0.0f, 0.2f,
        0.0f, 0.0f, 0.200001f,
        0.0f, 0.0f, 1.0f,
    };
    Camera cam = make_camera_identity_view();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);
    EXPECT_EQ(out.active[0], 0u);
    EXPECT_EQ(out.active[1], 0u);
    EXPECT_EQ(out.active[2], 1u);
    EXPECT_EQ(out.active[3], 1u);
}

// ============================================================================
// Slice 2b — viewproj, NDC, pixel-space position
// ============================================================================
TEST(PreprocessorVK_Slice2b, PerspectiveViewproj_MatchesCPU) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    constexpr int N = 128;
    auto g = make_default_gaussians(N);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> xy(-1.5f, 1.5f);
    std::uniform_real_distribution<float> z (-1.0f, 5.0f);
    for (int i = 0; i < N; ++i) {
        g.positions[i*3+0] = xy(rng);
        g.positions[i*3+1] = xy(rng);
        g.positions[i*3+2] = z (rng);
    }

    Camera cam = make_camera_trivial_perspective();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);
    ASSERT_EQ((int)out.means2D.size(), N * 2);
    ASSERT_EQ((int)out.p_hom_w.size(), N);

    int active_count = 0;
    for (int i = 0; i < N; ++i) {
        float p_view[3];
        transformPoint4x3(&g.positions[i * 3], cam.view_matrix, p_view);
        float p_hom[4];
        transformPoint4x4(&g.positions[i * 3], cam.viewproj_matrix, p_hom);
        EXPECT_NEAR(out.p_hom_w[i], p_hom[3], 1e-5f) << "i=" << i;

        bool active = (p_view[2] > 0.2f);
        if (active) {
            float p_w = 1.0f / (p_hom[3] + 1e-7f);
            float expected_x = ndc2Pix(p_hom[0] * p_w, cam.width);
            float expected_y = ndc2Pix(p_hom[1] * p_w, cam.height);
            EXPECT_NEAR(out.means2D[i*2+0], expected_x, 1e-3f) << "i=" << i;
            EXPECT_NEAR(out.means2D[i*2+1], expected_y, 1e-3f) << "i=" << i;
            ++active_count;
        } else {
            EXPECT_EQ(out.means2D[i*2+0], 0.0f) << "i=" << i;
            EXPECT_EQ(out.means2D[i*2+1], 0.0f) << "i=" << i;
        }
    }
    EXPECT_GT(active_count, 0);
    EXPECT_LT(active_count, N);
}

TEST(PreprocessorVK_Slice2b, CenterPixel_LandsAtImageCenter) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    auto g = make_default_gaussians(1);
    g.positions = {0.0f, 0.0f, 1.0f};
    Camera cam = make_camera_trivial_perspective();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);

    ASSERT_EQ(out.active[0], 1u);
    float expected_x = 0.5f * cam.width  - 0.5f;
    float expected_y = 0.5f * cam.height - 0.5f;
    EXPECT_NEAR(out.means2D[0], expected_x, 1e-3f);
    EXPECT_NEAR(out.means2D[1], expected_y, 1e-3f);
}

// ============================================================================
// Slice 2c — 3D covariance, 2D covariance projection, conic, opacity
// (no antialiasing yet — that's Slice 2d)
// ============================================================================
namespace {

// CPU oracle: replicate the conic computation exactly as preprocessor_cpu.cpp
// does for the 2D path.
void compute_expected_conic(const float pos[3],
                            const float scale[3], float scale_mod,
                            const float rot[4],
                            const Camera& cam,
                            float expected_conic[3]) {
    float cov3d[6];
    computeCov3D(scale, scale_mod, rot, cov3d);
    float focal_x = cam.width  / (2.0f * cam.tan_fovx);
    float focal_y = cam.height / (2.0f * cam.tan_fovy);
    float cov2d[3];
    computeCov2D(pos, cov3d, cam.view_matrix, focal_x, focal_y,
                 cam.tan_fovx, cam.tan_fovy, cov2d);
    cov2d[0] += 0.3f;
    cov2d[2] += 0.3f;
    float det = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];
    float inv_det = 1.0f / det;
    expected_conic[0] =  cov2d[2] * inv_det;
    expected_conic[1] = -cov2d[1] * inv_det;
    expected_conic[2] =  cov2d[0] * inv_det;
}

// Relative tolerance — conic magnitudes vary with focal^2 and opacity.
// 1e-4 relative passes for all well-behaved inputs; 1e-7 floor prevents
// divide-by-zero when expected is effectively zero.
void expect_conic_near(const float* gpu, const float* cpu, int i) {
    for (int k = 0; k < 3; ++k) {
        float a = gpu[k], b = cpu[k];
        float abs_err = std::fabs(a - b);
        float rel_err = abs_err / std::max(std::fabs(b), 1e-7f);
        EXPECT_LT(rel_err, 1e-4f)
            << "i=" << i << " k=" << k << " gpu=" << a << " cpu=" << b;
    }
}

}  // namespace

TEST(PreprocessorVK_Slice2c, IdentityGaussian_KnownConic) {
    // Unit-scale identity-quat Gaussian at (0,0,1) with identity view +
    // identity viewproj. Σ3D = I, J = diag(fx, fy, 0) at z=1, so
    // cov2D = diag(fx², fy²). After +0.3, conic = diag(1/(c+0.3), 1/(a+0.3)).
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    auto g = make_default_gaussians(1);
    g.positions = {0.0f, 0.0f, 1.0f};
    Camera cam = make_camera_identity_view();  // W=512, H=512, tan_fov=1
    RenderConfig cfg;  // scale_modifier=1, antialiasing=false

    auto out = pre.process(g.view(), cam, cfg);
    ASSERT_EQ(out.active[0], 1u);

    float focal_x = cam.width  / (2.0f * cam.tan_fovx);  // 256
    float focal_y = cam.height / (2.0f * cam.tan_fovy);  // 256
    float a = focal_x * focal_x + 0.3f;
    float c = focal_y * focal_y + 0.3f;
    float det = a * c;
    float expected[3] = { c / det, 0.0f, a / det };

    expect_conic_near(out.conics.data(), expected, 0);
    EXPECT_EQ(out.opacities_2d[0], 1.0f);
}

TEST(PreprocessorVK_Slice2c, RandomScalesRotations_MatchesCPU) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    constexpr int N = 64;
    TestGaussians g = make_default_gaussians(N);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> xy(-0.5f, 0.5f);
    std::uniform_real_distribution<float> z (0.5f, 3.0f);
    std::uniform_real_distribution<float> s (0.05f, 0.3f);
    std::uniform_real_distribution<float> q (-1.0f, 1.0f);
    std::uniform_real_distribution<float> op(0.2f, 1.0f);
    for (int i = 0; i < N; ++i) {
        g.positions[i*3+0] = xy(rng);
        g.positions[i*3+1] = xy(rng);
        g.positions[i*3+2] = z (rng);
        g.scales   [i*3+0] = s (rng);
        g.scales   [i*3+1] = s (rng);
        g.scales   [i*3+2] = s (rng);
        float qr=q(rng), qx=q(rng), qy=q(rng), qz=q(rng);
        float n = std::sqrt(qr*qr + qx*qx + qy*qy + qz*qz);
        g.rotations[i*4+0] = qr/n;
        g.rotations[i*4+1] = qx/n;
        g.rotations[i*4+2] = qy/n;
        g.rotations[i*4+3] = qz/n;
        g.opacities[i]     = op(rng);
    }

    Camera cam = make_camera_identity_view();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);

    int active_count = 0;
    for (int i = 0; i < N; ++i) {
        if (out.active[i] == 0u) continue;
        ++active_count;
        float expected[3];
        compute_expected_conic(&g.positions[i*3], &g.scales[i*3],
                               cfg.scale_modifier, &g.rotations[i*4],
                               cam, expected);
        expect_conic_near(&out.conics[i*3], expected, i);
        EXPECT_NEAR(out.opacities_2d[i], g.opacities[i], 1e-6f) << "i=" << i;
    }
    EXPECT_EQ(active_count, N);  // All test Gaussians are in front of camera
}

TEST(PreprocessorVK_Slice2c, CulledGaussians_HaveZeroConic) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    auto g = make_default_gaussians(3);
    // Behind camera, at near plane, in front.
    g.positions = {
        0.0f, 0.0f, -0.5f,   // culled
        0.0f, 0.0f, 0.1f,    // culled (<= 0.2)
        0.0f, 0.0f, 1.0f,    // active
    };
    Camera cam = make_camera_identity_view();
    RenderConfig cfg;
    auto out = pre.process(g.view(), cam, cfg);

    EXPECT_EQ(out.active[0], 0u);
    EXPECT_EQ(out.active[1], 0u);
    EXPECT_EQ(out.active[2], 1u);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(out.conics[i*3+0], 0.0f) << "i=" << i;
        EXPECT_EQ(out.conics[i*3+1], 0.0f) << "i=" << i;
        EXPECT_EQ(out.conics[i*3+2], 0.0f) << "i=" << i;
        EXPECT_EQ(out.opacities_2d[i], 0.0f) << "i=" << i;
    }
}

TEST(PreprocessorVK_Slice2c, ScaleModifier_ScalesCovariance) {
    // scale_modifier multiplies each axis, so cov3D scales by s². Conic
    // scales by 1/s² roughly (before +0.3 filter). We just verify output
    // matches CPU under a non-default scale_modifier to lock in the plumbing.
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    PreprocessorVK pre(ctx, SHADER_DIR);

    auto g = make_default_gaussians(16);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> z(0.5f, 2.0f);
    for (int i = 0; i < 16; ++i) g.positions[i*3+2] = z(rng);

    Camera cam = make_camera_identity_view();
    RenderConfig cfg;
    cfg.scale_modifier = 0.5f;
    auto out = pre.process(g.view(), cam, cfg);

    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(out.active[i], 1u) << "i=" << i;
        float expected[3];
        compute_expected_conic(&g.positions[i*3], &g.scales[i*3],
                               cfg.scale_modifier, &g.rotations[i*4],
                               cam, expected);
        expect_conic_near(&out.conics[i*3], expected, i);
    }
}
