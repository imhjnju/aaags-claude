// test_training_step_vk.cpp — SP-4 Task 8 + SP-5 Task 3: VulkanTrainer integration tests.
//
// Nine tests:
//   T-step-count  : VulkanTrainer.StepCountIncreases
//       Load tiny golden fixture. Run 3 steps, assert step_count() == 3.
//
//   T-step1       : VulkanTrainer.GradientsNonZeroAfterStep
//       One step with all-zeros target. Assert that raw_scales, raw_opacities,
//       and/or raw_rotations change from their initial values (gradients flow).
//
//   T-no-update   : VulkanTrainer.StepCanSkipAdamUpdate
//       One step with apply_update=false. Assert gradients/render are produced
//       while raw parameters remain unchanged.
//
//   T-eval3d      : VulkanTrainer.Eval3DOneStepSmoke
//       One eval_3D step with all-zeros target. Assert finite loss and at least
//       one raw parameter changes.
//
//   T-eval3d-replay: VulkanTrainer.Eval3DNonParityRecordsReplayOrder
//       eval_3D training with parity_mode=false records replay-order sideband data.
//
//   T-eval3d-grad : VulkanTrainer.Eval3DStep1RawGradientParity
//       One eval_3D step with all-zeros target. Compare captured raw gradients
//       against CUDA autograd goldens from dump_eval3d_golden.py.
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
#include <filesystem>
#include <cstdio>
#include <stdexcept>

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

struct GradStats {
    double l2_rel = 0.0;
    double max_abs = 0.0;
    size_t worst = 0;
    float vk_at_worst = 0.0f;
    float cuda_at_worst = 0.0f;
};

GradStats compare_grad(const float* vk, const float* cuda, size_t n) {
    GradStats s;
    double sum_diff2 = 0.0;
    double sum_ref2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(vk[i]) - static_cast<double>(cuda[i]);
        sum_diff2 += d * d;
        sum_ref2 += static_cast<double>(cuda[i]) * static_cast<double>(cuda[i]);
        const double ad = std::fabs(d);
        if (ad > s.max_abs) {
            s.max_abs = ad;
            s.worst = i;
            s.vk_at_worst = vk[i];
            s.cuda_at_worst = cuda[i];
        }
    }
    s.l2_rel = std::sqrt(sum_diff2) / std::max(std::sqrt(sum_ref2), 1e-30);
    return s;
}

void print_grad_stats(const char* name, const GradStats& s) {
    std::printf("[eval3D grad] %-8s l2_rel=%.6e max_abs=%.6e worst=%zu vk=%.8g cuda=%.8g\n",
                name, s.l2_rel, s.max_abs, s.worst,
                static_cast<double>(s.vk_at_worst),
                static_cast<double>(s.cuda_at_worst));
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
// T-no-update : final-step mode runs backward but skips Adam
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, StepCanSkipAdamUpdate) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H);
    trainer.enable_gradient_capture(true);

    std::vector<std::vector<float>> init_adam_m(6), init_adam_v(6);
    for (int group = 0; group < 6; ++group) {
        trainer.download_adam_moments(group, init_adam_m[group], init_adam_v[group]);
    }

    const RawGaussianParams& before = trainer.raw_params();
    std::vector<float> init_raw_positions(before.raw_positions, before.raw_positions + scene.N * 3);
    std::vector<float> init_raw_scales(before.raw_scales, before.raw_scales + scene.N * 3);
    std::vector<float> init_raw_rotations(before.raw_rotations, before.raw_rotations + scene.N * 4);
    std::vector<float> init_raw_sh(before.raw_sh_coeffs, before.raw_sh_coeffs + scene.N * scene.max_coeffs * 3);
    std::vector<float> init_raw_opacities(before.raw_opacities, before.raw_opacities + scene.N);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    const float loss = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H, false);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GE(loss, 0.0f);
    EXPECT_EQ(trainer.step_count(), 1);
    ASSERT_NE(trainer.rendered_image(), nullptr);

    auto max_abs = [](const std::vector<float>& v) {
        float m = 0.0f;
        for (float x : v) m = std::max(m, std::fabs(x));
        return m;
    };
    ASSERT_EQ(trainer.captured_grad_positions().size(), static_cast<size_t>(scene.N) * 3);
    ASSERT_EQ(trainer.captured_grad_scales().size(), static_cast<size_t>(scene.N) * 3);
    ASSERT_EQ(trainer.captured_grad_rotations().size(), static_cast<size_t>(scene.N) * 4);
    ASSERT_EQ(trainer.captured_grad_sh().size(), static_cast<size_t>(scene.N) * scene.max_coeffs * 3);
    ASSERT_EQ(trainer.captured_grad_opacities().size(), static_cast<size_t>(scene.N));
    const float max_grad = std::max({
        max_abs(trainer.captured_grad_positions()),
        max_abs(trainer.captured_grad_scales()),
        max_abs(trainer.captured_grad_rotations()),
        max_abs(trainer.captured_grad_sh()),
        max_abs(trainer.captured_grad_opacities())});
    EXPECT_GT(max_grad, 0.0f);

    const RawGaussianParams& after = trainer.raw_params();
    for (int i = 0; i < scene.N * 3; ++i) {
        EXPECT_FLOAT_EQ(after.raw_positions[i], init_raw_positions[i]);
        EXPECT_FLOAT_EQ(after.raw_scales[i], init_raw_scales[i]);
    }
    for (int i = 0; i < scene.N * 4; ++i) {
        EXPECT_FLOAT_EQ(after.raw_rotations[i], init_raw_rotations[i]);
    }
    for (int i = 0; i < scene.N * scene.max_coeffs * 3; ++i) {
        EXPECT_FLOAT_EQ(after.raw_sh_coeffs[i], init_raw_sh[i]);
    }
    for (int i = 0; i < scene.N; ++i) {
        EXPECT_FLOAT_EQ(after.raw_opacities[i], init_raw_opacities[i]);
    }

    for (int group = 0; group < 6; ++group) {
        std::vector<float> after_m, after_v;
        trainer.download_adam_moments(group, after_m, after_v);
        ASSERT_EQ(after_m.size(), init_adam_m[group].size()) << "group " << group;
        ASSERT_EQ(after_v.size(), init_adam_v[group].size()) << "group " << group;
        for (size_t i = 0; i < after_m.size(); ++i) {
            EXPECT_FLOAT_EQ(after_m[i], init_adam_m[group][i]) << "group " << group << " m[" << i << "]";
        }
        for (size_t i = 0; i < after_v.size(); ++i) {
            EXPECT_FLOAT_EQ(after_v[i], init_adam_v[group][i]) << "group " << group << " v[" << i << "]";
        }
    }
}

// ---------------------------------------------------------------------------
// T-eval3d : eval_3D one-step smoke
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, Eval3DOneStepSmoke) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    scene.cfg.eval_3D = true;

    VkTrainingConfig tcfg;
    tcfg.eval_3D = true;
    tcfg.parity_mode = true;
    tcfg.noise_lr = 0.0f;
    tcfg.sh_degree_max = scene.sh_degree;
    tcfg.sh_degree_warmup = 0;
    tcfg.densify_from_step = 0;

    std::vector<float> init_raw_positions(scene.raw_positions);
    std::vector<float> init_raw_scales(scene.raw_scales);
    std::vector<float> init_raw_rotations(scene.raw_rotations);
    std::vector<float> init_raw_sh(scene.raw_sh);
    std::vector<float> init_raw_opacities(scene.raw_opacities);

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    const float loss = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GE(loss, 0.0f);
    EXPECT_EQ(trainer.step_count(), 1);

    const RawGaussianParams& after = trainer.raw_params();
    float max_delta = 0.0f;
    for (int i = 0; i < scene.N * 3; ++i) {
        max_delta = std::max(max_delta, std::fabs(after.raw_positions[i] - init_raw_positions[i]));
        max_delta = std::max(max_delta, std::fabs(after.raw_scales[i] - init_raw_scales[i]));
    }
    for (int i = 0; i < scene.N * 4; ++i) {
        max_delta = std::max(max_delta, std::fabs(after.raw_rotations[i] - init_raw_rotations[i]));
    }
    for (int i = 0; i < scene.N * scene.max_coeffs * 3; ++i) {
        max_delta = std::max(max_delta, std::fabs(after.raw_sh_coeffs[i] - init_raw_sh[i]));
    }
    for (int i = 0; i < scene.N; ++i) {
        max_delta = std::max(max_delta, std::fabs(after.raw_opacities[i] - init_raw_opacities[i]));
    }

    EXPECT_GT(max_delta, 0.0f) << "eval_3D step produced no raw parameter update.";
}

// ---------------------------------------------------------------------------
// T-eval3d-replay : non-parity eval_3D training records exact replay order
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, Eval3DNonParityRecordsReplayOrder) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    scene.cfg.eval_3D = true;

    VkTrainingConfig tcfg;
    tcfg.eval_3D = true;
    tcfg.parity_mode = false;
    tcfg.noise_lr = 0.0f;
    tcfg.sh_degree_max = scene.sh_degree;
    tcfg.sh_degree_warmup = 0;
    tcfg.densify_from_step = 0;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);
    trainer.enable_intermediate_capture(true);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    float loss = 0.0f;
    ASSERT_NO_THROW(loss = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H));
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GE(loss, 0.0f);
    EXPECT_EQ(trainer.step_count(), 1);

    size_t blended = 0;
    for (int n : trainer.captured_n_contrib()) {
        ASSERT_GE(n, 0);
        blended += static_cast<size_t>(n);
    }
    EXPECT_GT(blended, 0u);
    EXPECT_EQ(trainer.last_replay_order_count_for_test(), blended);
}

// ---------------------------------------------------------------------------
// T-eval3d-grad : eval_3D raw gradients match CUDA autograd goldens
// ---------------------------------------------------------------------------

TEST(VulkanTrainer, Eval3DStep1RawGradientParity) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    SceneFixture scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    const std::string root = tiny_cam0_dir();
    const char* required_goldens[] = {
        "eval3d_loss.npy",
        "eval3d_grad_pos.npy",
        "eval3d_grad_sca.npy",
        "eval3d_grad_rot.npy",
        "eval3d_grad_sh.npy",
        "eval3d_grad_op.npy",
        "eval3d_bwd_d_rgb.npy",
        "eval3d_bwd_d_opacity.npy",
        "eval3d_bwd_d_gauss2screen.npy",
    };
    for (const char* filename : required_goldens) {
        if (!std::filesystem::exists(root + "/" + filename)) {
            GTEST_SKIP() << "CUDA eval_3D gradient goldens missing; run tools/dump_eval3d_golden.py --dump-backward";
        }
    }

    auto load_grad = [&](const char* filename, size_t expected_numel) {
        const std::string path = root + "/" + filename;
        NpyArray arr = load_npy(path);
        assert_dtype(arr, NpyDtype::float32);
        EXPECT_EQ(arr.numel(), expected_numel) << path;
        return npy_to_f32_vec(arr);
    };

    scene.cfg.eval_3D = true;

    VkTrainingConfig tcfg;
    tcfg.eval_3D = true;
    tcfg.parity_mode = true;
    tcfg.lambda_dssim = 0.0f;
    tcfg.opacity_reg = 0.0f;
    tcfg.scale_reg = 0.0f;
    tcfg.noise_lr = 0.0f;
    tcfg.sh_degree_max = scene.sh_degree;
    tcfg.sh_degree_warmup = 0;
    tcfg.densify_from_step = 0;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);
    trainer.enable_gradient_capture(true);
    trainer.enable_backward_diagnostic_capture(true);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    const float vk_loss = trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    ASSERT_EQ(trainer.captured_bwd_d_rgb().size(), static_cast<size_t>(scene.N) * 3);
    ASSERT_EQ(trainer.captured_bwd_d_opacity().size(), static_cast<size_t>(scene.N));
    ASSERT_EQ(trainer.captured_bwd_d_gauss2screen().size(), static_cast<size_t>(scene.N) * 16);
    const float max_g2s = *std::max_element(
        trainer.captured_bwd_d_gauss2screen().begin(),
        trainer.captured_bwd_d_gauss2screen().end(),
        [](float a, float b) { return std::fabs(a) < std::fabs(b); });
    EXPECT_GT(std::fabs(max_g2s), 0.0f);

    const std::vector<float> cuda_loss = load_grad("eval3d_loss.npy", 1);
    EXPECT_NEAR(vk_loss, cuda_loss[0], 1e-5f);

    const std::vector<float> cg_pos = load_grad("eval3d_grad_pos.npy", static_cast<size_t>(scene.N) * 3);
    const std::vector<float> cg_sca = load_grad("eval3d_grad_sca.npy", static_cast<size_t>(scene.N) * 3);
    const std::vector<float> cg_rot = load_grad("eval3d_grad_rot.npy", static_cast<size_t>(scene.N) * 4);
    const std::vector<float> cg_sh  = load_grad("eval3d_grad_sh.npy",  static_cast<size_t>(scene.N) * scene.max_coeffs * 3);
    const std::vector<float> cg_op  = load_grad("eval3d_grad_op.npy",  static_cast<size_t>(scene.N));
    const std::vector<float> cg_bwd_rgb = load_grad("eval3d_bwd_d_rgb.npy", static_cast<size_t>(scene.N) * 3);
    const std::vector<float> cg_bwd_opacity = load_grad("eval3d_bwd_d_opacity.npy", static_cast<size_t>(scene.N));
    const std::vector<float> cg_bwd_g2s = load_grad("eval3d_bwd_d_gauss2screen.npy", static_cast<size_t>(scene.N) * 16);

    const GradStats bwd_rgb = compare_grad(trainer.captured_bwd_d_rgb().data(), cg_bwd_rgb.data(), cg_bwd_rgb.size());
    const GradStats bwd_opacity = compare_grad(trainer.captured_bwd_d_opacity().data(), cg_bwd_opacity.data(), cg_bwd_opacity.size());
    const GradStats bwd_g2s = compare_grad(trainer.captured_bwd_d_gauss2screen().data(), cg_bwd_g2s.data(), cg_bwd_g2s.size());
    const GradStats pos = compare_grad(trainer.captured_grad_positions().data(), cg_pos.data(), cg_pos.size());
    const GradStats sca = compare_grad(trainer.captured_grad_scales().data(),    cg_sca.data(), cg_sca.size());
    const GradStats rot = compare_grad(trainer.captured_grad_rotations().data(), cg_rot.data(), cg_rot.size());
    const GradStats sh  = compare_grad(trainer.captured_grad_sh().data(),        cg_sh.data(),  cg_sh.size());
    const GradStats op  = compare_grad(trainer.captured_grad_opacities().data(), cg_op.data(),  cg_op.size());

    print_grad_stats("bwd_rgb", bwd_rgb);
    print_grad_stats("bwd_op", bwd_opacity);
    print_grad_stats("bwd_g2s", bwd_g2s);
    print_grad_stats("pos", pos);
    print_grad_stats("scale", sca);
    print_grad_stats("rot", rot);
    print_grad_stats("sh", sh);
    print_grad_stats("opacity", op);

    constexpr double kRelTol = 2e-3;
    constexpr double kAbsTol = 2e-5;
    constexpr double kG2SAbsTol = 3e-5;
    EXPECT_LT(bwd_rgb.l2_rel, kRelTol);
    EXPECT_LT(bwd_opacity.l2_rel, kRelTol);
    EXPECT_LT(bwd_g2s.l2_rel, kRelTol);
    EXPECT_LT(pos.l2_rel, kRelTol);
    EXPECT_LT(sca.l2_rel, kRelTol);
    EXPECT_LT(rot.l2_rel, kRelTol);
    EXPECT_LT(sh.l2_rel,  kRelTol);
    EXPECT_LT(op.l2_rel,  kRelTol);
    EXPECT_LT(bwd_rgb.max_abs, kAbsTol);
    EXPECT_LT(bwd_opacity.max_abs, kAbsTol);
    EXPECT_LT(bwd_g2s.max_abs, kG2SAbsTol);
    EXPECT_LT(pos.max_abs, kAbsTol);
    EXPECT_LT(sca.max_abs, kAbsTol);
    EXPECT_LT(rot.max_abs, kAbsTol);
    EXPECT_LT(sh.max_abs,  kAbsTol);
    EXPECT_LT(op.max_abs,  kAbsTol);
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
