// test_mcmc_trainer_integration.cpp — Phase 4 of MCMC densification port.
//
// Verifies VulkanTrainer.step() works end-to-end with MCMC densification
// enabled (cap_max > 0). Mirrors the structure of
// test_densification.cpp::VulkanTrainerSurvivesDensificationStep but routes
// through the new mcmc::densify() path.

#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"
#include "vulkan/vulkan_adam.h"
#include "golden/compare.h"
#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

float logit_mcmc_int(float p) {
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));
}

std::string tiny_cam0_dir_mcmc() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_mcmc(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

std::vector<int> npy_to_i32_mcmc(const NpyArray& a) {
    std::vector<int> v(a.numel());
    const int32_t* p = reinterpret_cast<const int32_t*>(a.raw.data());
    for (size_t i = 0; i < a.numel(); ++i) v[i] = static_cast<int>(p[i]);
    return v;
}

float sigmoid_mcmc_int(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void expect_f32_close_mcmc(const std::vector<float>& got,
                           const std::vector<float>& expected,
                           float abs_tol,
                           float rel_tol,
                           const char* label) {
    CompareResult r = compare_f32(got, expected, abs_tol, rel_tol);
    EXPECT_TRUE(r.passed)
        << label << " max_abs=" << r.max_abs_err
        << " max_rel=" << r.max_rel_err
        << " first_bad=" << r.first_bad_index
        << " num_bad=" << r.num_bad;
}

// Same fixture shape as DenseSceneFixture in test_densification.cpp; renamed
// to avoid linker collision because both files are compiled into gs3d_vk_tests.
struct McmcSceneFixture {
    int N = 0, H = 0, W = 0, sh_degree = 0, max_coeffs = 0;

    std::vector<float> positions, scales, rotations, opacities, sh_coeffs, filter_3d;
    std::vector<float> raw_positions, raw_scales, raw_rotations, raw_opacities, raw_sh;

    GaussianData      g{};
    RawGaussianParams raw{};
    Camera            cam{};
    RenderConfig      cfg{};

    bool load() {
        const std::string root = tiny_cam0_dir_mcmc();
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

            positions = npy_to_f32_mcmc(pos_npy);
            scales    = npy_to_f32_mcmc(scl_npy);
            rotations = npy_to_f32_mcmc(rot_npy);
            opacities = npy_to_f32_mcmc(opa_npy);
            sh_coeffs = npy_to_f32_mcmc(sh_npy);
            filter_3d = npy_to_f32_mcmc(f3d_npy);

            raw_positions = positions;
            raw_rotations = rotations;
            raw_sh        = sh_coeffs;

            raw_scales.resize(static_cast<size_t>(N) * 3);
            for (int i = 0; i < N * 3; ++i)
                raw_scales[i] = std::log(scales[i]);

            raw_opacities.resize(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i)
                raw_opacities[i] = logit_mcmc_int(opacities[i]);

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

TEST(McmcDensify, VulkanTrainerStepsWithMcmcEnabled) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    McmcSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    // Enable MCMC densification: cap_max > 0 routes through mcmc::densify.
    VkTrainingConfig tcfg;
    tcfg.densify_from_step     = 1;
    tcfg.densify_until_step    = 5;
    tcfg.densify_interval      = 1;
    tcfg.opacity_thresh        = 0.005f;
    tcfg.pos_lr_init           = 1.6e-4f;
    tcfg.pos_lr_final          = 1.6e-6f;
    tcfg.max_steps             = 30000;
    tcfg.sh_degree_max         = scene.sh_degree;
    tcfg.sh_degree_warmup      = 10000;
    tcfg.cap_max               = 200;  // grow target cap

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Run 2 steps — both trigger MCMC densification (steps 1 and 2 in [1..5]).
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    EXPECT_EQ(trainer.step_count(), 2)
        << "Expected step_count==2 after 2 steps.";

    EXPECT_GE(trainer.last_loss(), 0.0f);
    EXPECT_FALSE(std::isnan(trainer.last_loss()));
}

// ---------------------------------------------------------------------------
// Relocated dead slots must not inherit stale Adam state from the slots they
// overwrite.
// ---------------------------------------------------------------------------
TEST(McmcDensify, RelocatedDestinationsHaveAdamStateZeroed) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    McmcSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    scene.raw_opacities[0] = logit_mcmc_int(0.9f);
    for (int i = 1; i < scene.N; ++i) {
        scene.raw_opacities[i] = logit_mcmc_int(0.00499f);
        scene.opacities[i] = 0.00499f;
    }
    scene.opacities[0] = 0.9f;

    VkTrainingConfig tcfg;
    tcfg.densify_from_step     = 2;
    tcfg.densify_until_step    = 5;
    tcfg.densify_interval      = 1;
    tcfg.opacity_thresh        = 0.005f;
    tcfg.pos_lr_init           = 1.6e-4f;
    tcfg.pos_lr_final          = 1.6e-6f;
    tcfg.max_steps             = 30000;
    tcfg.sh_degree_max         = scene.sh_degree;
    tcfg.sh_degree_warmup      = 10000;
    tcfg.cap_max               = scene.N;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    std::vector<float> m_before, v_before;
    trainer.download_adam_moments(0, m_before, v_before);
    int nonzero_dead_before = 0;
    for (int i = 1; i < scene.N; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (m_before[static_cast<size_t>(i) * 3 + k] != 0.0f) ++nonzero_dead_before;
        }
    }
    ASSERT_GT(nonzero_dead_before, 0) << "fixture must create destination Adam state before relocation";

    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    const int strides[6] = {3, 3, (scene.max_coeffs - 1) * 3, 1, 3, 4};
    for (int group = 0; group < 6; ++group) {
        if (strides[group] == 0) continue;
        std::vector<float> m_after, v_after;
        trainer.download_adam_moments(group, m_after, v_after);
        ASSERT_GE(static_cast<int>(m_after.size()), scene.N * strides[group]);
        for (int i = 0; i < scene.N * strides[group]; ++i) {
            EXPECT_FLOAT_EQ(m_after[i], 0.0f) << "group=" << group << " i=" << i;
            EXPECT_FLOAT_EQ(v_after[i], 0.0f) << "group=" << group << " i=" << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Untouched Gaussian Adam state should survive MCMC growth; a full reset would
// zero every original moment.
// ---------------------------------------------------------------------------
TEST(McmcDensify, AdamStatePreservedAcrossDensification) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    McmcSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    // Densify from step 2 onward so step 1 runs the Adam update WITHOUT
    // densification. After step 1, moments are non-zero. After step 2,
    // densification fires — we verify the original Gaussians' moments survive.
    VkTrainingConfig tcfg;
    tcfg.densify_from_step     = 2;
    tcfg.densify_until_step    = 5;
    tcfg.densify_interval      = 1;
    tcfg.opacity_thresh        = 0.005f;
    tcfg.pos_lr_init           = 1.6e-4f;
    tcfg.pos_lr_final          = 1.6e-6f;
    tcfg.max_steps             = 30000;
    tcfg.sh_degree_max         = scene.sh_degree;
    tcfg.sh_degree_warmup      = 10000;
    tcfg.cap_max               = 200;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Step 1: Adam runs, NO densification.
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    // Download group 0 (positions) m/v after step 1.
    std::vector<float> m_after_step1, v_after_step1;
    trainer.download_adam_moments(0, m_after_step1, v_after_step1);
    ASSERT_FALSE(m_after_step1.empty());

    // Count non-zero m entries — should be most of them.
    int nonzero_before = 0;
    for (float x : m_after_step1) if (x != 0.0f) ++nonzero_before;
    EXPECT_GT(nonzero_before, 0) << "Adam m should be non-zero after step 1";

    const int N_before_densify = static_cast<int>(m_after_step1.size());

    // Step 2: densification fires.
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);
    EXPECT_EQ(trainer.step_count(), 2);

    // Download group 0 m/v after step 2 (with densification).
    std::vector<float> m_after_step2, v_after_step2;
    trainer.download_adam_moments(0, m_after_step2, v_after_step2);
    ASSERT_FALSE(m_after_step2.empty());

    // The group must be at least as large as before (MCMC only grows).
    EXPECT_GE(static_cast<int>(m_after_step2.size()), N_before_densify)
        << "Adam group 0 should not shrink after MCMC densification";

    // The first N_before_densify elements must NOT all be zero —
    // existing Gaussians' Adam state must have been preserved.
    int nonzero_after = 0;
    for (int i = 0; i < N_before_densify; ++i)
        if (m_after_step2[i] != 0.0f) ++nonzero_after;
    EXPECT_GT(nonzero_after, 0)
        << "Adam state for original Gaussians should be non-zero after densification "
           "(preserve, not reset)";
}

TEST(McmcDensify, OpacityResetIntervalResetsRawOpacity) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    McmcSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    VkTrainingConfig tcfg;
    tcfg.densify_from_step = 1;
    tcfg.densify_until_step = 1;
    tcfg.densify_interval = 1;
    tcfg.opacity_reset_interval = 1;
    tcfg.opacity_thresh = 0.005f;
    tcfg.cap_max = scene.N;
    tcfg.lambda_dssim = 0.0f;
    tcfg.opacity_reg = 0.0f;
    tcfg.scale_reg = 0.0f;
    tcfg.noise_lr = 0.0f;
    tcfg.sh_degree_max = scene.sh_degree;
    tcfg.sh_degree_warmup = 0;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);

    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);
    trainer.step(scene.cam, scene.cfg, target.data(), scene.W, scene.H);

    const float expected = logit_mcmc_int(0.01f);
    const RawGaussianParams& raw = trainer.raw_params();
    ASSERT_EQ(raw.count, scene.N);
    for (int i = 0; i < raw.count; ++i) {
        EXPECT_NEAR(raw.raw_opacities[i], expected, 1e-6f) << "i=" << i;
    }

    std::vector<float> m_opacity, v_opacity;
    trainer.download_adam_moments(3, m_opacity, v_opacity);
    ASSERT_EQ(static_cast<int>(m_opacity.size()), raw.count);
    for (int i = 0; i < raw.count; ++i) {
        EXPECT_EQ(m_opacity[i], 0.0f) << "m i=" << i;
        EXPECT_EQ(v_opacity[i], 0.0f) << "v i=" << i;
    }
}

TEST(McmcDensify, VulkanTrainerSupportsEval3DMcmcWithFilterPropagation) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    McmcSceneFixture scene;
    if (!scene.load()) {
        GTEST_SKIP() << "Could not load tiny golden fixture — skipping.";
    }

    VkTrainingConfig tcfg;
    tcfg.eval_3D = true;
    tcfg.parity_mode = true;
    tcfg.cap_max = 200;
    tcfg.densify_from_step = 1;
    tcfg.densify_until_step = 1;
    tcfg.densify_interval = 1;
    tcfg.lambda_dssim = 0.0f;
    tcfg.opacity_reg = 0.0f;
    tcfg.scale_reg = 0.0f;
    tcfg.noise_lr = 0.0f;
    tcfg.sh_degree_max = scene.sh_degree;
    tcfg.sh_degree_warmup = 0;

    VulkanTrainer trainer(ctx, scene.g, scene.raw,
                          scene.sh_degree, scene.W, scene.H, tcfg);
    RenderConfig cfg = scene.cfg;
    cfg.eval_3D = true;
    cfg.eval_3D_parity_mode = true;
    std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    EXPECT_NO_THROW(trainer.step(scene.cam, cfg, target.data(), scene.W, scene.H));
    ASSERT_EQ(static_cast<int>(trainer.filter_3D_for_test().size()), trainer.raw_params().count);
}

TEST(McmcDensify, VulkanTrainerReplayMatchesCudaGolden) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const std::string root = std::string(TEST_DATA_DIR) + "/golden/mcmc_cuda/combined_relocate_add";
    std::vector<float> raw_positions = npy_to_f32_mcmc(load_npy(root + "/input_raw_positions.npy"));
    std::vector<float> raw_scales = npy_to_f32_mcmc(load_npy(root + "/input_raw_scales.npy"));
    std::vector<float> raw_rotations = npy_to_f32_mcmc(load_npy(root + "/input_raw_rotations.npy"));
    std::vector<float> raw_opacities = npy_to_f32_mcmc(load_npy(root + "/input_raw_opacities.npy"));
    std::vector<float> raw_sh = npy_to_f32_mcmc(load_npy(root + "/input_raw_sh.npy"));
    std::vector<int> relocate_sources = npy_to_i32_mcmc(load_npy(root + "/relocate_src_indices.npy"));
    std::vector<int> add_sources = npy_to_i32_mcmc(load_npy(root + "/add_src_indices.npy"));
    std::vector<int> expected_modified = npy_to_i32_mcmc(load_npy(root + "/expected_modified_sources.npy"));
    std::vector<int> expected_dsts = npy_to_i32_mcmc(load_npy(root + "/expected_replaced_dsts.npy"));

    std::vector<float> expected_positions = npy_to_f32_mcmc(load_npy(root + "/expected_raw_positions.npy"));
    std::vector<float> expected_scales = npy_to_f32_mcmc(load_npy(root + "/expected_raw_scales.npy"));
    std::vector<float> expected_rotations = npy_to_f32_mcmc(load_npy(root + "/expected_raw_rotations.npy"));
    std::vector<float> expected_opacities = npy_to_f32_mcmc(load_npy(root + "/expected_raw_opacities.npy"));
    std::vector<float> expected_sh = npy_to_f32_mcmc(load_npy(root + "/expected_raw_sh.npy"));

    const int N = static_cast<int>(raw_opacities.size());
    const int max_coeffs = static_cast<int>(raw_sh.size() / (static_cast<size_t>(N) * 3));

    std::vector<float> act_scales(raw_scales.size());
    std::vector<float> act_opacities(raw_opacities.size());
    std::vector<float> filter_3d(static_cast<size_t>(N));
    for (size_t i = 0; i < raw_scales.size(); ++i) act_scales[i] = std::exp(raw_scales[i]);
    for (size_t i = 0; i < raw_opacities.size(); ++i) act_opacities[i] = sigmoid_mcmc_int(raw_opacities[i]);
    for (int i = 0; i < N; ++i) filter_3d[static_cast<size_t>(i)] = 500.0f + static_cast<float>(i);

    GaussianData g{};
    g.count = N;
    g.sh_degree = 0;
    g.max_coeffs = max_coeffs;
    g.positions = raw_positions.data();
    g.scales = act_scales.data();
    g.rotations = raw_rotations.data();
    g.opacities = act_opacities.data();
    g.sh_coeffs = raw_sh.data();
    g.filter_3D = filter_3d.data();

    RawGaussianParams raw{};
    raw.count = N;
    raw.sh_degree = 0;
    raw.max_coeffs = max_coeffs;
    raw.raw_positions = raw_positions.data();
    raw.raw_scales = raw_scales.data();
    raw.raw_rotations = raw_rotations.data();
    raw.raw_sh_coeffs = raw_sh.data();
    raw.raw_opacities = raw_opacities.data();

    VkTrainingConfig tcfg;
    tcfg.cap_max = 80;
    tcfg.sh_degree_max = 0;

    VulkanTrainer trainer(ctx, g, raw, 0, 1, 1, tcfg);

    mcmc::DensifySamplePlan plan;
    plan.relocate_sources = relocate_sources;
    plan.add_sources = add_sources;
    mcmc::DensifyResult result = trainer.apply_mcmc_densification_for_test(0.005f, 80, plan);

    std::vector<float> expected_filter = filter_3d;
    std::vector<int> dead_indices;
    for (int i = 0; i < N; ++i) {
        if (sigmoid_mcmc_int(raw_opacities[static_cast<size_t>(i)]) <= 0.005f) {
            dead_indices.push_back(i);
        }
    }
    ASSERT_EQ(dead_indices.size(), relocate_sources.size());
    for (size_t k = 0; k < dead_indices.size(); ++k) {
        expected_filter[static_cast<size_t>(dead_indices[k])] = filter_3d[static_cast<size_t>(relocate_sources[k])];
    }
    std::vector<float> filter_after_reloc = expected_filter;
    expected_filter.resize(expected_opacities.size());
    for (size_t k = 0; k < add_sources.size(); ++k) {
        expected_filter[static_cast<size_t>(N) + k] = filter_after_reloc[static_cast<size_t>(add_sources[k])];
    }

    EXPECT_EQ(result.final_count, static_cast<int>(expected_opacities.size()));
    EXPECT_EQ(result.modified_source_indices, expected_modified);
    EXPECT_EQ(result.replaced_destination_indices, expected_dsts);

    const RawGaussianParams& out = trainer.raw_params();
    std::vector<float> got_positions(out.raw_positions, out.raw_positions + out.count * 3);
    std::vector<float> got_scales(out.raw_scales, out.raw_scales + out.count * 3);
    std::vector<float> got_rotations(out.raw_rotations, out.raw_rotations + out.count * 4);
    std::vector<float> got_opacities(out.raw_opacities, out.raw_opacities + out.count);
    std::vector<float> got_sh(out.raw_sh_coeffs, out.raw_sh_coeffs + out.count * max_coeffs * 3);

    expect_f32_close_mcmc(got_positions, expected_positions, 1e-6f, 1e-6f, "positions");
    expect_f32_close_mcmc(got_rotations, expected_rotations, 1e-6f, 1e-6f, "rotations");
    expect_f32_close_mcmc(got_sh, expected_sh, 1e-6f, 1e-6f, "sh");
    expect_f32_close_mcmc(got_opacities, expected_opacities, 2e-5f, 2e-5f, "opacities");
    expect_f32_close_mcmc(got_scales, expected_scales, 2e-5f, 2e-5f, "scales");
    EXPECT_EQ(trainer.filter_3D_for_test(), expected_filter);
}
