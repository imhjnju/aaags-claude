// test_training_step_vk.cpp — SP-4 Task 8: VulkanTrainer integration tests.
//
// Three tests:
//   T-step-count  : VulkanTrainer.StepCountIncreases
//       Load tiny golden fixture. Run 3 steps, assert step_count() == 3.
//
//   T-step1       : VulkanTrainer.GradientsNonZeroAfterStep
//       One step with all-zeros target. Assert that raw_scales, raw_opacities,
//       and/or raw_rotations change from their initial values (gradients flow).
//
//   T-step10      : VulkanTrainer.LossDecreasesOverSteps
//       All-zeros target. Loss = mean(|rendered|). Adam should drive it lower
//       over 10 steps. Assert loss_step10 < loss_step1.
//
// All tests use the tiny golden fixture (N=103, 64x64) so the scene is
// guaranteed to produce visible Gaussians without needing to hand-craft a
// valid camera + projection matrix.
//
// GTEST_SKIP if no Vulkan device.

#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"

#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

// logit: inverse of sigmoid.  logit(p) = log(p / (1 - p))
// Clamps to avoid log(0).
float logit(float p) {
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));
}

// SceneFixture loads the tiny golden scene and exposes GaussianData,
// RawGaussianParams, Camera, and RenderConfig, all with owned storage.
struct SceneFixture {
    int N = 0, H = 0, W = 0, sh_degree = 0, max_coeffs = 0;

    // Activated values (as stored in the npy — already exp/sigmoid activated).
    std::vector<float> positions;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> opacities;
    std::vector<float> sh_coeffs;
    std::vector<float> filter_3d;

    // Raw (pre-activation) values.
    std::vector<float> raw_positions;   // = positions (no activation)
    std::vector<float> raw_scales;      // = log(scales)
    std::vector<float> raw_rotations;  // = rotations (already normalized)
    std::vector<float> raw_opacities;  // = logit(opacities)
    std::vector<float> raw_sh;         // = sh_coeffs (no activation)

    GaussianData       g{};
    RawGaussianParams  raw{};
    Camera             cam{};
    RenderConfig       cfg{};

    bool load() {
        const std::string root = tiny_cam0_dir();
        try {
            auto pos_npy  = load_npy(root + "/input_positions.npy");
            auto scl_npy  = load_npy(root + "/input_scales.npy");
            auto rot_npy  = load_npy(root + "/input_rotations.npy");
            auto opa_npy  = load_npy(root + "/input_opacities.npy");
            auto sh_npy   = load_npy(root + "/input_sh.npy");
            auto f3d_npy  = load_npy(root + "/input_filter_3D.npy");
            auto vm_npy   = load_npy(root + "/input_viewmatrix.npy");
            auto pm_npy   = load_npy(root + "/input_projmatrix.npy");
            auto fov_npy  = load_npy(root + "/input_fov_size.npy");
            auto cp_npy   = load_npy(root + "/input_campos.npy");
            auto meta_npy = load_npy(root + "/input_meta.npy");

            sh_degree = static_cast<int>(meta_npy.f32()[0]);
            max_coeffs = static_cast<int>(meta_npy.f32()[1]);
            H          = static_cast<int>(meta_npy.f32()[2]);
            W          = static_cast<int>(meta_npy.f32()[3]);
            N          = static_cast<int>(pos_npy.shape[0]);

            positions = npy_to_f32_vec(pos_npy);
            scales    = npy_to_f32_vec(scl_npy);
            rotations = npy_to_f32_vec(rot_npy);
            opacities = npy_to_f32_vec(opa_npy);
            sh_coeffs = npy_to_f32_vec(sh_npy);
            filter_3d = npy_to_f32_vec(f3d_npy);

            // Build raw vectors.
            raw_positions = positions;
            raw_rotations = rotations;  // already normalized quaternions
            raw_sh        = sh_coeffs;

            raw_scales.resize(static_cast<size_t>(N) * 3);
            for (int i = 0; i < N * 3; ++i) {
                raw_scales[i] = std::log(scales[i]);
            }

            raw_opacities.resize(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i) {
                raw_opacities[i] = logit(opacities[i]);
            }

            // Build GaussianData (activated).
            g.count      = N;
            g.sh_degree  = sh_degree;
            g.max_coeffs = max_coeffs;
            g.positions  = positions.data();
            g.scales     = scales.data();
            g.rotations  = rotations.data();
            g.opacities  = opacities.data();
            g.sh_coeffs  = sh_coeffs.data();
            g.filter_3D  = filter_3d.data();

            // Build RawGaussianParams.
            raw.count         = N;
            raw.sh_degree     = sh_degree;
            raw.max_coeffs    = max_coeffs;
            raw.raw_positions = raw_positions.data();
            raw.raw_scales    = raw_scales.data();
            raw.raw_rotations = raw_rotations.data();
            raw.raw_sh_coeffs = raw_sh.data();
            raw.raw_opacities = raw_opacities.data();

            // Build Camera.
            std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
            std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
            cam.cam_pos[0] = cp_npy.f32()[0];
            cam.cam_pos[1] = cp_npy.f32()[1];
            cam.cam_pos[2] = cp_npy.f32()[2];
            cam.tan_fovx   = fov_npy.f32()[0];
            cam.tan_fovy   = fov_npy.f32()[1];
            cam.width      = static_cast<int>(fov_npy.f32()[2]);
            cam.height     = static_cast<int>(fov_npy.f32()[3]);

            // Build RenderConfig.
            cfg.sh_degree      = sh_degree;
            cfg.training       = true;
            cfg.eval_3D        = false;
            cfg.tile_w         = 16;
            cfg.tile_h         = 16;
            cfg.antialiasing   = false;
            cfg.scale_modifier = 1.0f;
            cfg.bg_color[0]    = 0.0f;
            cfg.bg_color[1]    = 0.0f;
            cfg.bg_color[2]    = 0.0f;

            return true;
        } catch (...) {
            return false;
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// T-step-count : step_count() increments correctly
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, StepCountIncreases) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H);

    EXPECT_EQ(trainer.step_count(), 0);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    EXPECT_EQ(trainer.step_count(), 1);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    EXPECT_EQ(trainer.step_count(), 2);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    EXPECT_EQ(trainer.step_count(), 3);
}

// ---------------------------------------------------------------------------
// T-step1 : gradients non-zero after one step
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, GradientsNonZeroAfterStep) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    // Record initial raw values before training.
    std::vector<float> init_raw_scales(scene.raw_scales);
    std::vector<float> init_raw_opacities(scene.raw_opacities);
    std::vector<float> init_raw_rotations(scene.raw_rotations);

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H);

    // Zero-image target: loss = mean(|rendered|), gradient = sign(rendered)/n.
    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    // After one step, Adam should have updated raw params where gradients flow.
    const RawGaussianParams& after = trainer.raw_params();

    float max_scale_delta    = 0.0f;
    float max_opacity_delta  = 0.0f;
    float max_rotation_delta = 0.0f;

    for (int i = 0; i < scene.N * 3; ++i) {
        max_scale_delta = std::max(max_scale_delta,
                                   std::fabs(after.raw_scales[i] - init_raw_scales[i]));
    }
    for (int i = 0; i < scene.N; ++i) {
        max_opacity_delta = std::max(max_opacity_delta,
                                     std::fabs(after.raw_opacities[i] - init_raw_opacities[i]));
    }
    for (int i = 0; i < scene.N * 4; ++i) {
        max_rotation_delta = std::max(max_rotation_delta,
                                      std::fabs(after.raw_rotations[i] - init_raw_rotations[i]));
    }

    // Scales and opacities have strong, direct gradients from the L1 loss
    // on a non-zero rendered image. At least one of the three must move.
    const bool any_moved = (max_scale_delta > 0.0f) ||
                           (max_opacity_delta > 0.0f) ||
                           (max_rotation_delta > 0.0f);
    EXPECT_TRUE(any_moved)
        << "No raw parameters changed after one step — gradient did not flow.\n"
        << "  max_scale_delta="    << max_scale_delta    << "\n"
        << "  max_opacity_delta="  << max_opacity_delta  << "\n"
        << "  max_rotation_delta=" << max_rotation_delta;

    // Also verify loss was computed (non-NaN, non-negative).
    EXPECT_GE(trainer.last_loss(), 0.0f);
    EXPECT_FALSE(std::isnan(trainer.last_loss()));
    // The scene renders non-black Gaussians, so L1 loss against zero must be > 0.
    EXPECT_GT(trainer.last_loss(), 0.0f) << "Loss is zero — no visible Gaussians?";
}

// ---------------------------------------------------------------------------
// T-step10 : loss decreases over 10 steps toward all-zero target
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, LossDecreasesOverSteps) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H);

    // Target: all zeros. Loss = mean(|rendered|). Adam drives colors to zero.
    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    float loss_step1  = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    float loss_step10 = loss_step1;
    for (int i = 1; i < 10; ++i) {
        loss_step10 = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    }

    // Loss must be strictly decreasing over 10 Adam steps.
    EXPECT_LT(loss_step10, loss_step1)
        << "Loss did not decrease after 10 steps: step1=" << loss_step1
        << " step10=" << loss_step10;

    EXPECT_GE(loss_step10, 0.0f);
    EXPECT_FALSE(std::isnan(loss_step10));
}
