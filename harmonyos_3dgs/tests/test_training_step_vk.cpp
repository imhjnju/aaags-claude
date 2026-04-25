// test_training_step_vk.cpp — SP-4 Task 8 + SP-5 Task 3: VulkanTrainer integration tests.
//
// Five tests:
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
//   T-lr-decay    : VulkanTrainer.LRDecayReducesPositionUpdate
//       Fast-decay config (max_steps=10, init=1e-2, final=1e-5). Assert the
//       position update magnitude at step 10 < step 1 (lr decayed).
//
//   T-sh-schedule : VulkanTrainer.SHDegreeSchedule
//       warmup=5 steps per increment. Assert active_sh_degree()==1 after 5
//       steps and ==2 after 10 steps.
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
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    assert_dtype(a, NpyDtype::float32);
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

// ---------------------------------------------------------------------------
// T-lr-decay : position update shrinks as LR decays exponentially
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, LRDecayReducesPositionUpdate) {
    // Test that position LR decays over steps: the position update at step 1
    // should be larger than at step 10 (for fixed gradient, smaller lr => smaller update).
    // Use a fast-decay config: pos_lr_init=1e-2, pos_lr_final=1e-5, max_steps=10.
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VkTrainingConfig tcfg;
    tcfg.pos_lr_init      = 1e-2f;
    tcfg.pos_lr_final     = 1e-5f;
    tcfg.max_steps        = 10;
    tcfg.sh_degree_warmup = 10000;  // disable SH scheduling for this test
    tcfg.sh_degree_max    = 3;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Step 1: record positions before and after to get delta1.
    std::vector<float> pos_before(scene.raw_positions);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    const float* pos_after1 = trainer.raw_params().raw_positions;
    float delta1 = 0.0f;
    for (int i = 0; i < scene.N * 3; ++i) {
        delta1 += std::fabs(pos_after1[i] - pos_before[i]);
    }

    // Steps 2–9 (intermediate).
    for (int i = 1; i < 9; ++i) {
        trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    }

    // Step 10: record positions before and after to get delta10.
    std::vector<float> pos_before10(trainer.raw_params().raw_positions,
                                    trainer.raw_params().raw_positions + scene.N * 3);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    const float* pos_after10 = trainer.raw_params().raw_positions;
    float delta10 = 0.0f;
    for (int i = 0; i < scene.N * 3; ++i) {
        delta10 += std::fabs(pos_after10[i] - pos_before10[i]);
    }

    // Position update at step 10 must be smaller than at step 1 (LR decayed).
    EXPECT_LT(delta10, delta1)
        << "LR decay did not reduce position update: delta1=" << delta1
        << " delta10=" << delta10;
}

// ---------------------------------------------------------------------------
// T-sh-schedule : active_sh_degree increments by warmup schedule
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, SHDegreeSchedule) {
    // After warmup steps, active_sh_degree should increment.
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VkTrainingConfig tcfg;
    tcfg.sh_degree_max    = 3;
    tcfg.sh_degree_warmup = 5;   // increment every 5 steps for fast test
    tcfg.pos_lr_init      = 1.6e-4f;
    tcfg.pos_lr_final     = 1.6e-6f;
    tcfg.max_steps        = 30000;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Run 5 steps: step_count_ == 5, active_sh_degree_ = 5 / 5 = 1.
    for (int i = 0; i < 5; ++i) {
        trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    }
    EXPECT_EQ(trainer.active_sh_degree(), 1)
        << "Expected active_sh_degree==1 after 5 steps (warmup=5).";

    // Run 5 more steps (total 10): active_sh_degree_ = 10 / 5 = 2.
    for (int i = 0; i < 5; ++i) {
        trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    }
    EXPECT_EQ(trainer.active_sh_degree(), 2)
        << "Expected active_sh_degree==2 after 10 steps (warmup=5).";
}
