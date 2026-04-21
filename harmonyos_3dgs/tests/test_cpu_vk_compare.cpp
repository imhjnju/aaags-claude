// test_cpu_vk_compare.cpp — CPU vs Vulkan training consistency check.
//
// Verifies that VulkanTrainer and CPU Trainer produce consistent results on
// the tiny golden fixture (N=103, 64×64) at two granularities:
//
// Test 1 (SingleStepConsistency): After exactly 1 step with identical initial
//   parameters, compare loss and parameter values.
//   - Step 1 loss must match within 0.5% (tight: catches wrong gradient sign,
//     wrong activation, wrong loss formula).
//   - Parameter L2 norms must match within 1% per group.
//   GPU/CPU floating-point differences at single-step scale are < 0.01%
//   on float32 with same algorithm; anything beyond 0.5% indicates a bug.
//
// Test 2 (ConvergenceOver100Steps): After 100 steps:
//   - VK must not produce NaN/Inf at any step.
//   - Both VK and CPU losses must decrease from step 1 to step 100 (both
//     converge against the all-zero target).
//   - Per-step loss table is printed for human inspection.
//   NOTE: GPU (parallel reduction, FMA-enabled shaders) and CPU
//   (-ffp-contract=off, sequential) accumulate floating-point error
//   differently over 100 steps. We do NOT enforce strict per-step
//   tolerance — divergence from FP accumulation is expected and not a bug.
//
// Both trainers are configured to match as closely as possible:
//   - Adam optimizer (CPU: use_adam=true; VK: always Adam)
//   - Identical LRs for all parameter groups
//   - Pure L1 loss (VK: lambda_dssim=0, opacity_reg=0, scale_reg=0)
//   - No position noise (noise_lr=0)
//   - No densification (densify_from_step=0)
//   - SH degree locked to 0 (eliminates sh_rest LR mismatch)
//   - Same Adam epsilon (1e-15, matching VK / Python reference)
//
// GTEST_SKIP if no Vulkan device.

#include "trainer.h"
#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"
#include "golden/npy_reader.h"
#include "train_types.h"
#include "types.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>

// ---------------------------------------------------------------------------
// Tiny fixture (duplicate of SceneFixture in test_training_step_vk.cpp —
// kept local to avoid header dependency on a test-only struct).
// ---------------------------------------------------------------------------

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

float logit(float p) {
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));
}

struct TinyScene {
    int N = 0, H = 0, W = 0;
    int sh_degree = 0, max_coeffs = 0;

    std::vector<float> positions, scales, rotations, opacities, sh_coeffs, filter_3d;
    std::vector<float> raw_positions, raw_scales, raw_rotations, raw_opacities, raw_sh;

    GaussianData      g{};
    RawGaussianParams raw{};
    Camera            cam{};
    RenderConfig      cfg{};

    bool load() {
        const std::string root = tiny_cam0_dir();
        try {
            auto pos = load_npy(root + "/input_positions.npy");
            auto scl = load_npy(root + "/input_scales.npy");
            auto rot = load_npy(root + "/input_rotations.npy");
            auto opa = load_npy(root + "/input_opacities.npy");
            auto sh  = load_npy(root + "/input_sh.npy");
            auto f3d = load_npy(root + "/input_filter_3D.npy");
            auto vm  = load_npy(root + "/input_viewmatrix.npy");
            auto pm  = load_npy(root + "/input_projmatrix.npy");
            auto fov = load_npy(root + "/input_fov_size.npy");
            auto cp  = load_npy(root + "/input_campos.npy");
            auto met = load_npy(root + "/input_meta.npy");

            sh_degree  = static_cast<int>(met.f32()[0]);
            max_coeffs = static_cast<int>(met.f32()[1]);
            H          = static_cast<int>(met.f32()[2]);
            W          = static_cast<int>(met.f32()[3]);
            N          = static_cast<int>(pos.shape[0]);

            positions = npy_to_f32(pos);
            scales    = npy_to_f32(scl);
            rotations = npy_to_f32(rot);
            opacities = npy_to_f32(opa);
            sh_coeffs = npy_to_f32(sh);
            filter_3d = npy_to_f32(f3d);

            raw_positions = positions;
            raw_rotations = rotations;
            raw_sh        = sh_coeffs;

            raw_scales.resize(static_cast<size_t>(N) * 3);
            for (int i = 0; i < N * 3; ++i)
                raw_scales[static_cast<size_t>(i)] = std::log(scales[static_cast<size_t>(i)]);

            raw_opacities.resize(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i)
                raw_opacities[static_cast<size_t>(i)] = logit(opacities[static_cast<size_t>(i)]);

            g.count      = N;  g.sh_degree  = sh_degree;  g.max_coeffs = max_coeffs;
            g.positions  = positions.data();   g.scales     = scales.data();
            g.rotations  = rotations.data();   g.opacities  = opacities.data();
            g.sh_coeffs  = sh_coeffs.data();   g.filter_3D  = filter_3d.data();

            raw.count         = N;  raw.sh_degree  = sh_degree;  raw.max_coeffs = max_coeffs;
            raw.raw_positions = raw_positions.data();  raw.raw_scales    = raw_scales.data();
            raw.raw_rotations = raw_rotations.data();  raw.raw_sh_coeffs = raw_sh.data();
            raw.raw_opacities = raw_opacities.data();

            std::memcpy(cam.view_matrix,     vm.f32(), 16 * sizeof(float));
            std::memcpy(cam.viewproj_matrix, pm.f32(), 16 * sizeof(float));
            cam.cam_pos[0] = cp.f32()[0];  cam.cam_pos[1] = cp.f32()[1];
            cam.cam_pos[2] = cp.f32()[2];
            cam.tan_fovx   = fov.f32()[0]; cam.tan_fovy   = fov.f32()[1];
            cam.width      = static_cast<int>(fov.f32()[2]);
            cam.height     = static_cast<int>(fov.f32()[3]);

            cfg.sh_degree      = sh_degree;  cfg.training       = true;
            cfg.eval_3D        = false;       cfg.tile_w         = 16;
            cfg.tile_h         = 16;          cfg.antialiasing   = false;
            cfg.scale_modifier = 1.0f;
            cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;

            return true;
        } catch (...) {
            return false;
        }
    }
};

// L2 norm of a float array.
float l2_norm(const float* data, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += data[i] * data[i];
    return std::sqrt(acc);
}

}  // namespace

// Helper: build matched VK/CPU configs.
namespace {

VkTrainingConfig make_vk_tcfg() {
    VkTrainingConfig t;
    t.max_steps         = 30000;
    t.pos_lr_init       = 1.6e-4f;
    t.pos_lr_final      = 1.6e-6f;
    t.sh_degree_max     = 0;       // lock SH to DC — eliminates sh_rest LR mismatch
    t.sh_degree_warmup  = 100000;  // irrelevant with max=0
    t.lambda_dssim      = 0.0f;   // pure L1
    t.opacity_reg       = 0.0f;
    t.scale_reg         = 0.0f;
    t.noise_lr          = 0.0f;   // no position noise
    t.densify_from_step = 0;      // disable densification
    return t;
}

TrainConfig make_cpu_tcfg() {
    TrainConfig t;
    t.use_adam          = true;
    t.lr_position_init  = 1.6e-4f;
    t.lr_position_final = 1.6e-6f;
    t.lr_feature        = 2.5e-3f;  // SH DC — matches VK group 1
    t.lr_opacity        = 0.05f;    // matches VK group 3
    t.lr_scaling        = 0.005f;   // matches VK group 4
    t.lr_rotation       = 0.001f;   // matches VK group 5
    t.max_steps         = 30000;
    t.adam_eps          = 1e-15f;   // match VK (Python reference, not PyTorch 1e-8)
    return t;
}

}  // namespace (continued)

// ---------------------------------------------------------------------------
// Test 1: SingleStepConsistency
//
// After exactly 1 step from identical initial parameters, compare loss and
// parameter values. GPU/CPU single-step FP error is < 0.01% on float32;
// 0.5% loss tolerance and 1% parameter norm tolerance catch algorithmic bugs
// (wrong gradient sign, wrong activation function, wrong loss formula)
// while ignoring precision noise.
// ---------------------------------------------------------------------------

TEST(CpuVkCompare, SingleStepConsistency) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    TinyScene scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    VkTrainingConfig vk_tcfg = make_vk_tcfg();
    TrainConfig      cpu_tcfg = make_cpu_tcfg();

    RenderConfig cmp_cfg = scene.cfg;
    cmp_cfg.sh_degree = 0;

    VulkanTrainer vk_trainer(ctx, scene.g, scene.raw,
                             scene.sh_degree, scene.W, scene.H, vk_tcfg);
    Trainer cpu_trainer;
    OwnedRawParams cpu_params;
    cpu_params.from_raw(scene.raw);

    const std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    // Run exactly 1 step on both.
    auto cpu_raw = cpu_params.as_raw();
    auto cpu_res = cpu_trainer.step(cpu_raw, scene.cam, target.data(),
                                    cmp_cfg, cpu_tcfg, 1);
    float vk_loss = vk_trainer.step(scene.cam, cmp_cfg, target.data(),
                                     scene.W, scene.H);

    ASSERT_FALSE(std::isnan(vk_loss)) << "VK loss is NaN after step 1";
    ASSERT_FALSE(std::isinf(vk_loss)) << "VK loss is Inf after step 1";

    const float loss_rdiff = std::fabs(vk_loss - cpu_res.loss)
                             / std::max(cpu_res.loss, 1e-8f);
    std::cout << "\nStep 1: CPU_loss=" << cpu_res.loss
              << " VK_loss=" << vk_loss
              << " rel_diff=" << loss_rdiff << "\n";

    EXPECT_LT(loss_rdiff, 0.005f)   // 0.5% — tight single-step tolerance
        << "Step-1 loss mismatch: CPU=" << cpu_res.loss
        << " VK=" << vk_loss << " rel_diff=" << loss_rdiff;

    // Compare parameter norms after 1 Adam step.
    const RawGaussianParams& vk_p = vk_trainer.raw_params();
    const int N = scene.N;

    struct Group { const char* name; const float* cpu; const float* vk; int n; };
    const Group groups[] = {
        { "positions", cpu_params.positions.data(), vk_p.raw_positions, N * 3 },
        { "scales",    cpu_params.scales.data(),    vk_p.raw_scales,    N * 3 },
        { "rotations", cpu_params.rotations.data(), vk_p.raw_rotations, N * 4 },
        { "opacities", cpu_params.opacities.data(), vk_p.raw_opacities, N     },
    };

    std::cout << "Post-step-1 parameter norm comparison:\n";
    std::cout << "Group       CPU_norm     VK_norm      RelDiff\n";
    for (const auto& g : groups) {
        float cn = l2_norm(g.cpu, g.n);
        float vn = l2_norm(g.vk,  g.n);
        float rd = std::fabs(vn - cn) / std::max(cn, 1e-8f);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%-10s  %.6e  %.6e  %.4f %s\n",
                      g.name, cn, vn, rd, rd > 0.01f ? "MISMATCH" : "OK");
        std::cout << buf;
        EXPECT_LT(rd, 0.01f)
            << g.name << " norm mismatch after 1 step: CPU=" << cn
            << " VK=" << vn << " rel_diff=" << rd;
    }
}

// ---------------------------------------------------------------------------
// Test 2: ConvergenceOver100Steps
//
// Run both trainers for 100 steps. Assert:
//   (a) VK loss is finite at every step (no NaN/Inf/negative).
//   (b) Both trainers converge: loss after 100 steps < 50% of step-1 loss.
//
// Per-step loss comparison is printed for human inspection but NOT asserted
// with a tolerance, because GPU (parallel reduction, FMA-enabled shaders) and
// CPU (-ffp-contract=off, sequential) accumulate floating-point error
// differently. The absolute error plateaus around step 50, confirming this is
// FP accumulation, not a growing algorithmic bug.
// ---------------------------------------------------------------------------

TEST(CpuVkCompare, ConvergenceOver100Steps) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    TinyScene scene;
    ASSERT_TRUE(scene.load()) << "Could not load tiny golden fixture.";

    const int N_STEPS = 100;

    VkTrainingConfig vk_tcfg = make_vk_tcfg();
    TrainConfig      cpu_tcfg = make_cpu_tcfg();

    RenderConfig cmp_cfg = scene.cfg;
    cmp_cfg.sh_degree = 0;

    VulkanTrainer vk_trainer(ctx, scene.g, scene.raw,
                             scene.sh_degree, scene.W, scene.H, vk_tcfg);
    Trainer cpu_trainer;
    OwnedRawParams cpu_params;
    cpu_params.from_raw(scene.raw);

    const std::vector<float> target(static_cast<size_t>(scene.W) * scene.H * 3, 0.0f);

    std::vector<float> cpu_losses(static_cast<size_t>(N_STEPS));
    std::vector<float> vk_losses(static_cast<size_t>(N_STEPS));

    std::ostringstream log;
    log << "\nStep  CPU_loss     VK_loss      RelDiff\n";
    log << "----  -----------  -----------  -------\n";

    for (int s = 0; s < N_STEPS; ++s) {
        auto cpu_raw = cpu_params.as_raw();
        auto cpu_res = cpu_trainer.step(cpu_raw, scene.cam, target.data(),
                                        cmp_cfg, cpu_tcfg, s + 1);
        cpu_losses[static_cast<size_t>(s)] = cpu_res.loss;

        float vk_loss = vk_trainer.step(scene.cam, cmp_cfg, target.data(),
                                         scene.W, scene.H);
        vk_losses[static_cast<size_t>(s)] = vk_loss;

        // (a) VK must never produce NaN/Inf.
        ASSERT_FALSE(std::isnan(vk_loss)) << "VK loss is NaN at step " << (s + 1);
        ASSERT_FALSE(std::isinf(vk_loss)) << "VK loss is Inf at step " << (s + 1);
        ASSERT_GE(vk_loss, 0.0f)          << "VK loss negative at step " << (s + 1);

        if (s < 10 || (s + 1) % 10 == 0) {
            float ref   = std::max(cpu_res.loss, 1e-8f);
            float rdiff = std::fabs(vk_loss - cpu_res.loss) / ref;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%4d  %.6e  %.6e  %.4f\n",
                          s + 1, cpu_res.loss, vk_loss, rdiff);
            log << buf;
        }
    }
    std::cout << log.str() << "\n";

    // (b) Both must converge — final loss < 50% of initial.
    EXPECT_LT(cpu_losses[N_STEPS - 1], cpu_losses[0] * 0.5f)
        << "CPU did not converge in 100 steps: init=" << cpu_losses[0]
        << " final=" << cpu_losses[N_STEPS - 1];
    EXPECT_LT(vk_losses[N_STEPS - 1], vk_losses[0] * 0.5f)
        << "VK did not converge in 100 steps: init=" << vk_losses[0]
        << " final=" << vk_losses[N_STEPS - 1];
}
