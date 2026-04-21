// test_densification.cpp — SP-5 Task 5: MCMC densification unit tests.
//
// Three tests:
//   T-dense1 : Clone test — small Gaussians with high gradient cloned
//   T-dense2 : Prune test — low-opacity Gaussians removed
//   T-dense3 : Integration — VulkanTrainer survives a densification step

#include "densification.h"
#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"
#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Build a minimal OwnedRawParams with N Gaussians.
// Scales in raw (log) space: raw_scale_val applied to all 3 axes.
// Opacity in raw (logit) space: raw_opacity_val.
// Positions set to grid along X axis (0, 1, 2, ...).
// Rotations set to identity quaternion (1, 0, 0, 0).
// SH coefficients set to zero.
OwnedRawParams make_params(int N, float raw_scale_val, float raw_opacity_val) {
    OwnedRawParams p;
    p.sh_degree  = 0;
    p.max_coeffs = 1;
    p.resize(N);
    for (int i = 0; i < N; ++i) {
        p.positions[i * 3 + 0] = static_cast<float>(i) * 0.1f;
        p.positions[i * 3 + 1] = 0.0f;
        p.positions[i * 3 + 2] = 0.0f;
        for (int k = 0; k < 3; ++k) p.scales[i * 3 + k] = raw_scale_val;
        p.rotations[i * 4 + 0] = 1.0f;  // identity quaternion
        p.rotations[i * 4 + 1] = 0.0f;
        p.rotations[i * 4 + 2] = 0.0f;
        p.rotations[i * 4 + 3] = 0.0f;
        p.opacities[i] = raw_opacity_val;
        // SH coefficients zero (already from resize)
    }
    return p;
}

// Default VkTrainingConfig for densification tests.
VkTrainingConfig dense_cfg() {
    VkTrainingConfig cfg;
    cfg.densify_grad_thresh   = 2e-4f;
    cfg.opacity_thresh        = 0.005f;
    cfg.densify_percent_dense = 0.01f;
    cfg.densify_from_step     = 500;
    cfg.densify_until_step    = 15000;
    cfg.densify_interval      = 100;
    return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// T-dense1 : Clone test
// N=50, all grad_norms > thresh, all small scale => expect new_N > 50 (cloned)
// ---------------------------------------------------------------------------

TEST(Densification, CloneSmallGaussians) {
    const int N = 50;

    // raw_scale = -10 => exp(-10) ≈ 4.5e-5
    // scene_extent = 1.0 => size_threshold = 0.01 * 1.0 = 0.01
    // 4.5e-5 < 0.01 => is_small = true
    const float raw_scale   = -10.0f;

    // raw_opacity = 2.0 => sigmoid(2.0) ≈ 0.88 > 0.005 => not pruned
    const float raw_opacity = 2.0f;

    OwnedRawParams p = make_params(N, raw_scale, raw_opacity);

    // All gradients above threshold.
    std::vector<float> grad_norms(N, 1e-2f);  // >> 2e-4

    VkTrainingConfig cfg = dense_cfg();
    const float scene_extent = 1.0f;

    int new_N = densify_and_prune(p, grad_norms.data(), 1000, cfg, scene_extent);

    // Each Gaussian should be: kept + cloned → 2× at minimum.
    EXPECT_GT(new_N, N)
        << "Clone test: expected new_N > " << N << " but got " << new_N;
    // With all 50 cloned, expect exactly 100 (no splits — is_small=true).
    EXPECT_EQ(new_N, N * 2)
        << "Clone test: expected exactly " << N * 2 << " (keep + clone) but got " << new_N;
}

// ---------------------------------------------------------------------------
// T-dense2 : Prune test
// N=30, all opacities below thresh, no densification => expect new_N == 0
// ---------------------------------------------------------------------------

TEST(Densification, PruneAllOpacityBelowThresh) {
    const int N = 30;

    // raw_scale = -5 (moderate size, doesn't matter — grad_norm = 0 so no densify)
    // raw_opacity = -10 => sigmoid(-10) ≈ 4.5e-5 < 0.005 => pruned
    OwnedRawParams p = make_params(N, -5.0f, -10.0f);

    // Zero gradients — no densification.
    std::vector<float> grad_norms(N, 0.0f);

    VkTrainingConfig cfg = dense_cfg();
    const float scene_extent = 1.0f;

    int new_N = densify_and_prune(p, grad_norms.data(), 1000, cfg, scene_extent);

    EXPECT_EQ(new_N, 0)
        << "Prune test: expected all " << N << " Gaussians pruned but got " << new_N;
}

// ---------------------------------------------------------------------------
// T-dense3 : Integration — VulkanTrainer survives a densification step
// Uses the tiny golden fixture (N=103, 64x64).
// Config: densify_from_step=1, densify_until_step=5, densify_interval=1
// Run 2 steps, assert step_count()==2 (no crash, no assertion failure).
// ---------------------------------------------------------------------------

namespace {

std::string tiny_cam0_dir_dense() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_dense(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

float logit_dense(float p) {
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));
}

struct DenseSceneFixture {
    int N = 0, H = 0, W = 0, sh_degree = 0, max_coeffs = 0;

    std::vector<float> positions, scales, rotations, opacities, sh_coeffs, filter_3d;
    std::vector<float> raw_positions, raw_scales, raw_rotations, raw_opacities, raw_sh;

    GaussianData      g{};
    RawGaussianParams raw{};
    Camera            cam{};
    RenderConfig      cfg{};

    bool load() {
        const std::string root = tiny_cam0_dir_dense();
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

            sh_degree  = static_cast<int>(meta_npy.f32()[0]);
            max_coeffs = static_cast<int>(meta_npy.f32()[1]);
            H          = static_cast<int>(meta_npy.f32()[2]);
            W          = static_cast<int>(meta_npy.f32()[3]);
            N          = static_cast<int>(pos_npy.shape[0]);

            positions = npy_to_f32_dense(pos_npy);
            scales    = npy_to_f32_dense(scl_npy);
            rotations = npy_to_f32_dense(rot_npy);
            opacities = npy_to_f32_dense(opa_npy);
            sh_coeffs = npy_to_f32_dense(sh_npy);
            filter_3d = npy_to_f32_dense(f3d_npy);

            raw_positions = positions;
            raw_rotations = rotations;
            raw_sh        = sh_coeffs;

            raw_scales.resize(static_cast<size_t>(N) * 3);
            for (int i = 0; i < N * 3; ++i)
                raw_scales[i] = std::log(scales[i]);

            raw_opacities.resize(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i)
                raw_opacities[i] = logit_dense(opacities[i]);

            g.count      = N;
            g.sh_degree  = sh_degree;
            g.max_coeffs = max_coeffs;
            g.positions  = positions.data();
            g.scales     = scales.data();
            g.rotations  = rotations.data();
            g.opacities  = opacities.data();
            g.sh_coeffs  = sh_coeffs.data();
            g.filter_3D  = filter_3d.data();

            raw.count         = N;
            raw.sh_degree     = sh_degree;
            raw.max_coeffs    = max_coeffs;
            raw.raw_positions = raw_positions.data();
            raw.raw_scales    = raw_scales.data();
            raw.raw_rotations = raw_rotations.data();
            raw.raw_sh_coeffs = raw_sh.data();
            raw.raw_opacities = raw_opacities.data();

            std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
            std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
            cam.cam_pos[0] = cp_npy.f32()[0];
            cam.cam_pos[1] = cp_npy.f32()[1];
            cam.cam_pos[2] = cp_npy.f32()[2];
            cam.tan_fovx   = fov_npy.f32()[0];
            cam.tan_fovy   = fov_npy.f32()[1];
            cam.width      = static_cast<int>(fov_npy.f32()[2]);
            cam.height     = static_cast<int>(fov_npy.f32()[3]);

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

TEST(Densification, VulkanTrainerSurvivesDensificationStep) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    DenseSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    // Enable densification starting at step 1, running every step up to step 5.
    VkTrainingConfig tcfg;
    tcfg.densify_from_step     = 1;
    tcfg.densify_until_step    = 5;
    tcfg.densify_interval      = 1;
    tcfg.densify_grad_thresh   = 2e-4f;
    tcfg.opacity_thresh        = 0.005f;
    tcfg.densify_percent_dense = 0.01f;
    tcfg.pos_lr_init           = 1.6e-4f;
    tcfg.pos_lr_final          = 1.6e-6f;
    tcfg.max_steps             = 30000;
    tcfg.sh_degree_max         = scene.sh_degree;
    tcfg.sh_degree_warmup      = 10000;  // keep SH at 0 during test

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Run 2 steps — both will trigger densification (steps 1 and 2 are in [1..5]).
    // Test passes as long as no crash, assertion, or exception occurs.
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    EXPECT_EQ(trainer.step_count(), 2)
        << "Expected step_count==2 after 2 steps.";

    // Loss should be non-NaN and non-negative.
    EXPECT_GE(trainer.last_loss(), 0.0f);
    EXPECT_FALSE(std::isnan(trainer.last_loss()));
}

// ---------------------------------------------------------------------------
// T-dense4 : Split test
// N=10, all large scale (exp(2.0) >> percent_dense*scene_extent),
// all high gradient, moderate opacity (not pruned).
// Expected: new_N = 10 originals + 2*10 split children = 30.
// Also verify scale shrank: raw_scale ≈ 2.0 - log(1.6) for split children.
// ---------------------------------------------------------------------------

TEST(Densification, SplitLargeGaussians) {
    OwnedRawParams p;
    p.sh_degree = 0;
    p.max_coeffs = 1;
    const int N = 10;
    p.resize(N);

    // Large scale: exp(2.0) ≈ 7.39 >> percent_dense * scene_extent = 0.01 * 1.0 = 0.01
    for (int i = 0; i < N; ++i) {
        p.scales[i * 3 + 0] = 2.0f;
        p.scales[i * 3 + 1] = 2.0f;
        p.scales[i * 3 + 2] = 2.0f;
        // Rotations: identity quaternion [0,0,0,1] (x,y,z,w convention)
        p.rotations[i * 4 + 0] = 0.0f;
        p.rotations[i * 4 + 1] = 0.0f;
        p.rotations[i * 4 + 2] = 0.0f;
        p.rotations[i * 4 + 3] = 1.0f;
        // Moderate opacity: sigmoid(2.0) ≈ 0.88, well above 0.005 threshold
        p.opacities[i] = 2.0f;
    }

    std::vector<float> grad_norms(N, 1e-2f);  // all >> 2e-4 threshold

    VkTrainingConfig cfg;
    cfg.densify_grad_thresh   = 2e-4f;
    cfg.opacity_thresh        = 0.005f;
    cfg.densify_percent_dense = 0.01f;

    float scene_extent = 1.0f;
    int new_N = densify_and_prune(p, grad_norms.data(), 1000, cfg, scene_extent);

    // 10 originals kept + 2 split children each = 10 + 20 = 30
    EXPECT_EQ(new_N, 30) << "Expected 10 originals + 20 split children";

    // Verify output layout and split child scales.
    //
    // densify_and_prune emits per Gaussian (in order): keep + up to 2 children.
    // For each input Gaussian with high gradient and large scale:
    //   - index 3*i+0 : original (kept)          — scale unchanged = 2.0
    //   - index 3*i+1 : split child A             — scale shrunken = 2.0 - log(1.6)
    //   - index 3*i+2 : split child B             — scale shrunken = 2.0 - log(1.6)
    //
    // Note: MCMC noise perturbs positions only, not scales, so the scale check is exact.
    const float original_raw_scale = 2.0f;
    const float split_raw_scale    = 2.0f - std::log(1.6f);

    for (int i = 0; i < N; ++i) {
        const int orig_idx  = 3 * i + 0;
        const int childA_idx = 3 * i + 1;
        const int childB_idx = 3 * i + 2;

        // Original should be unchanged.
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(p.scales[orig_idx * 3 + k], original_raw_scale, 1e-5f)
                << "Original scale changed for Gaussian " << i << " axis " << k;
        }
        // Split children should have shrunk scale.
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(p.scales[childA_idx * 3 + k], split_raw_scale, 1e-5f)
                << "Scale wrong for split child A of Gaussian " << i << " axis " << k;
            EXPECT_NEAR(p.scales[childB_idx * 3 + k], split_raw_scale, 1e-5f)
                << "Scale wrong for split child B of Gaussian " << i << " axis " << k;
        }
    }
}
