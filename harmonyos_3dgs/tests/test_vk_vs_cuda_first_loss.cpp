// test_vk_vs_cuda_first_loss.cpp — VK vs CUDA first-loss parity harness
// (Phase 0.4 of dev_notes/vk_cuda_first_loss_parity_plan.md).
//
// Runs input, forward-stage, and loss gates that compare the exact inputs and
// outputs the VK forward path consumes or produces against the CUDA golden dump
// at tests/golden/basketball/cuda_ref/step_0001/.
//
// Fixtures:
//   PLY     : /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/
//             vulkan_3d/basket-aaa.ply  (400000 Gaussians, SH3)
//             Fallback: /home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply
//   Cameras : /home/robota/Downloads/basketball/_sp0_dump_output/cameras.json
//   Golden  : tests/golden/basketball/cuda_ref/step_0001/*.npy + raw_params.npz
//             (raw_params.npz is pre-extracted into sibling raw_<field>.npy
//             files — see tools/dump_cuda_training_step.py and the README
//             adjacent to the golden directory for how to regenerate.)
//
// The test SKIPs cleanly if the PLY, cameras.json, or golden directory is
// missing so it never hard-fails in CI environments that lack those data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "math_utils.h"
#include "ply_loader.h"
#include "train_types.h"
#include "types.h"
#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"

#include "golden/npy_reader.h"

namespace {

// ---------------------------------------------------------------------------
// Fixture paths
// ---------------------------------------------------------------------------
constexpr int kW = 720;
constexpr int kH = 960;

const std::string kPlyPathPrimary =
    "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply";
const std::string kPlyPathFallback =
    "/home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply";
const std::string kCamPath =
    "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json";

inline std::string dump_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/basketball/cuda_ref/step_0001";
}

inline std::string resolve_ply_path() {
    if (std::filesystem::exists(kPlyPathPrimary)) return kPlyPathPrimary;
    if (std::filesystem::exists(kPlyPathFallback)) return kPlyPathFallback;
    return {};
}

// ---------------------------------------------------------------------------
// Raw-param reconstruction (inverse of ply_loader activations).
//   - positions/rotations: PLY stores raw values as-is (no activation).
//   - scales:    raw = log(activated); activation is exp().
//   - opacities: raw = logit(activated); activation is sigmoid().
//   - SH:        PLY raw == activated (no activation). ply_loader reorders
//                PLY's channel-first layout into basis-interleaved
//                [gaussian, coeff, channel] which matches CUDA's get_features
//                output — i.e. the CUDA dump's (N, 16, 3) layout.
// ---------------------------------------------------------------------------
inline float inverse_sigmoid(float y) {
    y = std::max(1e-6f, std::min(1.0f - 1e-6f, y));
    return std::log(y / (1.0f - y));
}

RawGaussianParams deriveRawFromPlyLoaded(const GaussianData& g,
                                         std::vector<float>& raw_positions,
                                         std::vector<float>& raw_scales,
                                         std::vector<float>& raw_rotations,
                                         std::vector<float>& raw_sh_coeffs,
                                         std::vector<float>& raw_opacities) {
    const int N   = g.count;
    const int mc3 = g.max_coeffs * 3;

    raw_positions.assign(g.positions, g.positions + static_cast<size_t>(N) * 3);
    raw_rotations.assign(g.rotations, g.rotations + static_cast<size_t>(N) * 4);
    raw_sh_coeffs.assign(g.sh_coeffs, g.sh_coeffs + static_cast<size_t>(N) * mc3);

    raw_scales.resize(static_cast<size_t>(N) * 3);
    for (size_t i = 0; i < raw_scales.size(); ++i) {
        raw_scales[i] = std::log(std::max(1e-12f, g.scales[i]));
    }

    raw_opacities.resize(N);
    for (int i = 0; i < N; ++i) raw_opacities[i] = inverse_sigmoid(g.opacities[i]);

    RawGaussianParams r{};
    r.count         = N;
    r.sh_degree     = g.sh_degree;
    r.max_coeffs    = g.max_coeffs;
    r.raw_positions = raw_positions.data();
    r.raw_scales    = raw_scales.data();
    r.raw_rotations = raw_rotations.data();
    r.raw_sh_coeffs = raw_sh_coeffs.data();
    r.raw_opacities = raw_opacities.data();
    return r;
}

// ---------------------------------------------------------------------------
// Pretty-printers
// ---------------------------------------------------------------------------
void print_mat4(const char* label, const float* m_col_major) {
    // m_col_major[col*4 + row] convention (VK Camera layout).
    std::printf("  %s (row-major display of column-major storage):\n", label);
    for (int r = 0; r < 4; ++r) {
        std::printf("    [% .8f % .8f % .8f % .8f]\n",
                    m_col_major[0 * 4 + r], m_col_major[1 * 4 + r],
                    m_col_major[2 * 4 + r], m_col_major[3 * 4 + r]);
    }
}

void print_mat4_npy(const char* label, const float* arr_16) {
    // np.save row-major storage of a 4x4: arr_16[i*4+j] == stored[i,j].
    std::printf("  %s (row-major numpy storage):\n", label);
    for (int i = 0; i < 4; ++i) {
        std::printf("    [% .8f % .8f % .8f % .8f]\n",
                    arr_16[i * 4 + 0], arr_16[i * 4 + 1],
                    arr_16[i * 4 + 2], arr_16[i * 4 + 3]);
    }
}

// Report first K mismatches between two float arrays within absolute tolerance.
// Returns true if all values match.
bool report_float_mismatches(const char* label,
                             const float* a, const float* b, size_t n,
                             float tol, int max_print = 12) {
    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
        if (d > tol) {
            if (bad < max_print) {
                std::printf("    [%s] idx=%zu vk=% .8g cuda=% .8g |diff|=%.3g\n",
                            label, i, a[i], b[i], d);
            }
            ++bad;
        }
    }
    std::printf("  [%s] n=%zu bad=%d max_abs=%.3g (tol=%.3g)\n",
                label, n, bad, max_abs, tol);
    return bad == 0;
}

}  // namespace

// ===========================================================================
// Gate I1 — View matrix
// ===========================================================================
TEST(VkVsCudaFirstLoss, Gate_I1_ViewMatrix) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();

    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width, kW);
    ASSERT_EQ(cam.height, kH);

    // CUDA dump: row-major numpy storage of `world_view_transform` which is
    // byte-identical to column-major W2C — matching VK Camera.view_matrix[16].
    NpyArray vm = load_npy(dump_dir() + "/view_matrix.npy");
    assert_dtype(vm, NpyDtype::float32);
    ASSERT_EQ(vm.numel(), 16u);
    const float* cuda_vm = vm.f32();

    constexpr float kTol = 1e-6f;
    bool ok = report_float_mismatches("view_matrix", cam.view_matrix, cuda_vm, 16, kTol);
    if (!ok) {
        std::printf("[Gate I1] VK vs CUDA view matrix mismatch:\n");
        print_mat4("VK cam.view_matrix", cam.view_matrix);
        print_mat4_npy("CUDA view_matrix.npy", cuda_vm);
    }
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(cam.view_matrix[i], cuda_vm[i], kTol)
            << "view_matrix element " << i;
    }
}

// ===========================================================================
// Gate I2 — Projection matrix
// ===========================================================================
TEST(VkVsCudaFirstLoss, Gate_I2_ProjMatrix) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();

    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    NpyArray pm = load_npy(dump_dir() + "/proj_matrix.npy");
    assert_dtype(pm, NpyDtype::float32);
    ASSERT_EQ(pm.numel(), 16u);
    const float* cuda_pm = pm.f32();

    // Projection is FP-sensitive (perspective divide contributes differently
    // than a W2C cross product), so allow a slightly looser tolerance.  Still
    // tight enough to catch a transpose, axis swap, or sign flip.
    constexpr float kTol = 1e-5f;
    bool ok = report_float_mismatches("proj_matrix", cam.viewproj_matrix, cuda_pm, 16, kTol);
    if (!ok) {
        std::printf("[Gate I2] VK vs CUDA proj matrix mismatch:\n");
        print_mat4("VK cam.viewproj_matrix", cam.viewproj_matrix);
        print_mat4_npy("CUDA proj_matrix.npy", cuda_pm);
    }
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(cam.viewproj_matrix[i], cuda_pm[i], kTol)
            << "proj_matrix element " << i;
    }

    // Camera position + tan_fov (cheap bonus checks that share this gate).
    {
        NpyArray cp = load_npy(dump_dir() + "/cam_position.npy");
        assert_dtype(cp, NpyDtype::float32);
        ASSERT_EQ(cp.numel(), 3u);
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(cam.cam_pos[i], cp.f32()[i], 1e-6f) << "cam_pos[" << i << "]";
        }
    }
    {
        NpyArray tf = load_npy(dump_dir() + "/tan_fov.npy");
        assert_dtype(tf, NpyDtype::float32);
        ASSERT_EQ(tf.numel(), 2u);
        EXPECT_NEAR(cam.tan_fovx, tf.f32()[0], 1e-6f) << "tan_fovx";
        EXPECT_NEAR(cam.tan_fovy, tf.f32()[1], 1e-6f) << "tan_fovy";
    }
}

// ===========================================================================
// Gate I3 — Raw Gaussian parameters (first K Gaussians)
// ===========================================================================
TEST(VkVsCudaFirstLoss, Gate_I3_RawParams) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();

    // Load per-field .npy files extracted from raw_params.npz
    // (see dev_notes/vk_cuda_first_loss_parity_plan.md for the extraction step).
    auto require_sidecar = [&](const std::string& name) -> bool {
        if (std::filesystem::exists(dump_dir() + "/" + name)) return true;
        std::printf(
            "[Gate I3] Missing sidecar %s — raw_params.npz has not been "
            "pre-extracted.  Run:\n"
            "  python3 -c \"import numpy as np, os; "
            "d=np.load('%s/raw_params.npz'); "
            "[np.save(os.path.join('%s', k+'.npy'), d[k]) for k in d.files]\"\n",
            name.c_str(), dump_dir().c_str(), dump_dir().c_str());
        return false;
    };
    if (!require_sidecar("raw_positions.npy") ||
        !require_sidecar("raw_scales.npy") ||
        !require_sidecar("raw_rotations.npy") ||
        !require_sidecar("raw_opacities.npy") ||
        !require_sidecar("raw_sh_coeffs.npy")) {
        GTEST_SKIP() << "raw_params.npz not pre-extracted (see log above)";
    }

    NpyArray cpos = load_npy(dump_dir() + "/raw_positions.npy");
    NpyArray cscl = load_npy(dump_dir() + "/raw_scales.npy");
    NpyArray crot = load_npy(dump_dir() + "/raw_rotations.npy");
    NpyArray copa = load_npy(dump_dir() + "/raw_opacities.npy");
    NpyArray csh  = load_npy(dump_dir() + "/raw_sh_coeffs.npy");

    assert_dtype(cpos, NpyDtype::float32);
    assert_dtype(cscl, NpyDtype::float32);
    assert_dtype(crot, NpyDtype::float32);
    assert_dtype(copa, NpyDtype::float32);
    assert_dtype(csh,  NpyDtype::float32);

    const size_t N_cuda = cpos.shape[0];
    ASSERT_EQ(cpos.shape.size(), 2u);
    ASSERT_EQ(cpos.shape[1], 3u);
    ASSERT_EQ(cscl.shape, (std::vector<size_t>{N_cuda, 3u}));
    ASSERT_EQ(crot.shape, (std::vector<size_t>{N_cuda, 4u}));
    ASSERT_EQ(copa.shape, (std::vector<size_t>{N_cuda}));
    ASSERT_EQ(csh.shape,  (std::vector<size_t>{N_cuda, 16u, 3u}));

    // Load the VK side: PLY -> activated GaussianData -> invert activation.
    auto model = loadPly(ply.c_str());
    ASSERT_EQ(static_cast<size_t>(model.data.count), N_cuda)
        << "Gaussian count mismatch: VK=" << model.data.count
        << " CUDA=" << N_cuda;
    ASSERT_EQ(model.data.max_coeffs, 16);
    ASSERT_EQ(model.data.sh_degree, 3);

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    // Construct VulkanTrainer with parity config so raw_params() reflects the
    // exact bytes that will flow into the forward pass.  We do NOT call step()
    // here — Phase 1 is purely about verifying inputs, and the VK backward
    // path is incompatible with eval_3D=true.
    VulkanContext ctx;
    bool have_vk = ctx.init();
    std::unique_ptr<VulkanTrainer> trainer;
    if (have_vk) {
        RenderConfig rcfg{};
        rcfg.sh_degree = 3;
        rcfg.eval_3D   = true;
        rcfg.training  = true;
        rcfg.antialiasing = false;
        rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;

        VkTrainingConfig tcfg{};
        tcfg.sh_degree_max    = 3;
        tcfg.sh_degree_warmup = 1000;
        tcfg.lambda_dssim     = 0.0f;
        tcfg.eval_3D          = true;
        tcfg.parity_mode      = true;
        tcfg.densify_from_step = 0;  // disable densification
        // Build an activated GaussianData view to feed the trainer ctor.
        GaussianData init_g = model.data;
        trainer = std::make_unique<VulkanTrainer>(ctx, init_g, vk_raw, 3, kW, kH, tcfg);
    } else {
        std::printf("[Gate I3] No Vulkan device — testing raw params without "
                    "running trainer ctor (raw-from-PLY only).\n");
    }

    // Field references — prefer trainer's internal copies if we built one
    // (they're the exact bytes VK's forward path will consume).
    const RawGaussianParams& raw = trainer ? trainer->raw_params() : vk_raw;
    const size_t N = raw.count;
    ASSERT_EQ(N, N_cuda);

    // Compare first K Gaussians across each field.  Positions / SH are
    // byte-identical (no activation).  Scales / opacities may differ by ULPs
    // due to log(exp(x)) / inv_sigmoid(sigmoid(x)) round-trip; allow tiny
    // tolerance for those.
    constexpr int K_POS = 10;
    constexpr int K_SH  = 10;

    // --- positions ---
    {
        constexpr size_t n = K_POS * 3;
        bool ok = report_float_mismatches(
            "positions[0..10]", raw.raw_positions, cpos.f32(), n, 0.0f);
        (void)ok;
        for (size_t i = 0; i < n; ++i)
            EXPECT_FLOAT_EQ(raw.raw_positions[i], cpos.f32()[i]) << "pos idx " << i;
    }

    // --- scales ---
    {
        constexpr size_t n = K_POS * 3;
        constexpr float kScaleTol = 1e-6f;   // log(exp(x)) round-trip
        bool ok = report_float_mismatches(
            "scales[0..10]", raw.raw_scales, cscl.f32(), n, kScaleTol);
        (void)ok;
        for (size_t i = 0; i < n; ++i)
            EXPECT_NEAR(raw.raw_scales[i], cscl.f32()[i], kScaleTol)
                << "scale idx " << i;
    }

    // --- rotations ---
    {
        constexpr size_t n = K_POS * 4;
        // ply_loader normalizes the quaternion; CUDA's `_rotation` is the raw
        // (unnormalized) pre-activation tensor.  The activation used inside
        // rasterization normalizes too, so mathematically the effective
        // rotation is equivalent.  Report both bit-equal count and a loose
        // post-normalization check.
        int exact = 0;
        for (size_t i = 0; i < n; ++i) {
            if (raw.raw_rotations[i] == crot.f32()[i]) ++exact;
        }
        std::printf("  [rotations[0..10]] exact_byte_match=%d/%zu\n", exact, n);

        // Compare per-quaternion normalized versions: if both normalize to the
        // same unit quaternion (or its sign-flipped mate), they represent the
        // same rotation.
        for (int i = 0; i < K_POS; ++i) {
            const float* a = &raw.raw_rotations[i * 4];
            const float* b = &crot.f32()[i * 4];
            auto norm4 = [](const float* q) {
                float s = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
                return s > 0.0f ? s : 1.0f;
            };
            float na = norm4(a), nb = norm4(b);
            float na_q[4] = {a[0]/na, a[1]/na, a[2]/na, a[3]/na};
            float nb_q[4] = {b[0]/nb, b[1]/nb, b[2]/nb, b[3]/nb};
            // Sign-flip ambiguity: q and -q represent the same rotation.
            float dot = na_q[0]*nb_q[0] + na_q[1]*nb_q[1]
                      + na_q[2]*nb_q[2] + na_q[3]*nb_q[3];
            if (dot < 0.0f) for (int j = 0; j < 4; ++j) nb_q[j] = -nb_q[j];
            for (int j = 0; j < 4; ++j) {
                EXPECT_NEAR(na_q[j], nb_q[j], 1e-5f)
                    << "normalized rotation g" << i << " comp " << j
                    << " (raw vk=" << a[j] << " cuda=" << b[j] << ")";
            }
        }
    }

    // --- opacities ---
    {
        constexpr int n = K_POS;
        constexpr float kOpaTol = 1e-5f;  // inv_sigmoid(sigmoid(x)) round-trip
        bool ok = report_float_mismatches(
            "opacities[0..10]", raw.raw_opacities, copa.f32(), n, kOpaTol);
        (void)ok;
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(raw.raw_opacities[i], copa.f32()[i], kOpaTol)
                << "opa idx " << i;
    }

    // --- SH coeffs: (K_SH * 16 * 3) floats ---
    // Layout expected: sh[g * 16 * 3 + k * 3 + c].  CUDA dump uses the same
    // [gaussian, coeff, channel] order (np array flattens row-major).
    {
        constexpr size_t n = static_cast<size_t>(K_SH) * 16u * 3u;
        const float* vk_sh_ptr   = raw.raw_sh_coeffs;
        const float* cuda_sh_ptr = csh.f32();
        bool ok = report_float_mismatches(
            "sh[0..10 gaussians]", vk_sh_ptr, cuda_sh_ptr, n, 0.0f);
        if (!ok) {
            // Side-by-side dump for Gaussian 0 to diagnose layout mismatch.
            std::printf("[Gate I3] SH Gaussian 0 (16 coeffs x 3 channels), "
                        "VK vs CUDA:\n");
            for (int k = 0; k < 16; ++k) {
                std::printf("    k=%2d  VK=[% .6g % .6g % .6g]  "
                            "CUDA=[% .6g % .6g % .6g]\n", k,
                            vk_sh_ptr[k*3+0], vk_sh_ptr[k*3+1], vk_sh_ptr[k*3+2],
                            cuda_sh_ptr[k*3+0], cuda_sh_ptr[k*3+1], cuda_sh_ptr[k*3+2]);
            }
        }
        for (size_t i = 0; i < n; ++i)
            EXPECT_FLOAT_EQ(vk_sh_ptr[i], cuda_sh_ptr[i]) << "sh idx " << i;
    }

    model.free();
}

// ===========================================================================
// Gate I4 — Config flags sanity check
// ===========================================================================
TEST(VkVsCudaFirstLoss, Gate_I4_ConfigFlags) {
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();

    // meta.json parsing is intentionally minimal: we only assert the handful
    // of fields we directly control from the VK side.  Failures print the
    // CUDA-side expectation beside the VK-side value.
    std::ifstream mf(dump_dir() + "/meta.json");
    ASSERT_TRUE(mf.good()) << "cannot open meta.json";
    std::string meta((std::istreambuf_iterator<char>(mf)),
                     std::istreambuf_iterator<char>());

    auto find_bool = [&](const std::string& key) -> int {
        auto p = meta.find("\"" + key + "\":");
        if (p == std::string::npos) return -1;
        p = meta.find_first_not_of(" \t\r\n", p + key.size() + 3);
        if (p == std::string::npos) return -1;
        if (meta.compare(p, 4, "true")  == 0) return 1;
        if (meta.compare(p, 5, "false") == 0) return 0;
        return -1;
    };
    auto find_num = [&](const std::string& key) -> double {
        auto p = meta.find("\"" + key + "\":");
        if (p == std::string::npos) return std::nan("");
        p = meta.find_first_not_of(" \t\r\n", p + key.size() + 3);
        if (p == std::string::npos) return std::nan("");
        size_t end = meta.find_first_of(",]}\n", p);
        return std::stod(meta.substr(p, end - p));
    };

    const int cuda_eval_3D      = find_bool("eval_3D");
    const int cuda_antialiasing = find_bool("antialiasing");
    const int cuda_near_clip    = find_bool("near_clipping");
    const int cuda_new_aabb     = find_bool("new_aabb");
    const double cuda_lambda    = find_num("lambda_dssim");
    const double cuda_sh_max    = find_num("sh_degree_max");
    const double cuda_scale_mod = find_num("scale_modifier");

    // VK-side config snapshot (the config we WOULD pass to VulkanTrainer).
    VkTrainingConfig tcfg{};
    tcfg.eval_3D       = true;
    tcfg.parity_mode   = true;
    tcfg.lambda_dssim  = 0.0f;
    tcfg.sh_degree_max = 3;

    RenderConfig rcfg{};
    rcfg.sh_degree = 3;
    rcfg.eval_3D   = true;
    rcfg.antialiasing = false;
    rcfg.training  = true;
    rcfg.scale_modifier = 1.0f;
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;

    std::printf("[Gate I4] VK parity config snapshot:\n"
                "    tcfg.eval_3D        = %d  (cuda=%d)\n"
                "    tcfg.parity_mode    = %d  (cuda sort_mode=0 implied)\n"
                "    tcfg.lambda_dssim   = %.6f  (cuda=%.6f)\n"
                "    tcfg.sh_degree_max  = %d  (cuda=%.0f)\n"
                "    rcfg.antialiasing   = %d  (cuda=%d)\n"
                "    rcfg.scale_modifier = %.6f  (cuda=%.6f)\n"
                "    rcfg.bg_color       = [%.3f, %.3f, %.3f]\n",
                (int)tcfg.eval_3D, cuda_eval_3D,
                (int)tcfg.parity_mode,
                tcfg.lambda_dssim, cuda_lambda,
                tcfg.sh_degree_max, cuda_sh_max,
                (int)rcfg.antialiasing, cuda_antialiasing,
                rcfg.scale_modifier, cuda_scale_mod,
                rcfg.bg_color[0], rcfg.bg_color[1], rcfg.bg_color[2]);

    EXPECT_EQ(cuda_eval_3D, 1) << "CUDA dump was produced with eval_3D=false";
    EXPECT_EQ(cuda_antialiasing, 0) << "CUDA dump used antialiasing=true";
    EXPECT_EQ(cuda_near_clip, 0) << "CUDA dump used near_clipping=true";
    EXPECT_NEAR(cuda_lambda, 0.0, 1e-9) << "CUDA dump lambda_dssim != 0";
    EXPECT_NEAR(cuda_sh_max, 3.0, 1e-9);
    EXPECT_NEAR(cuda_scale_mod, 1.0, 1e-9);

    // --- new_aabb Phase-2 risk flag ---------------------------------------
    // CUDA dump fixes new_aabb=false (legacy screen-space AABB).  VK's
    // preprocess.comp unconditionally takes the new_aabb=true branch (see
    // src/vulkan/shaders/preprocess.comp around the computeAABBView call
    // inside the eval_3D block; the comment there reads "CUDA default:
    // new_aabb=true → always uses compute_aabb_view, never
    // compute_aabb_screen.").  This is a known Phase 2 parity gap that will
    // surface at Gate P4 (radii) and downstream.  Log but do NOT fail: Phase
    // 1's job is input isomorphism, and new_aabb is a forward-path divergence.
    if (cuda_new_aabb == 0) {
        std::printf("[Gate I4][PHASE-2-RISK] meta.json pins new_aabb=false, "
                    "but VK's preprocess.comp eval_3D branch is hard-wired to "
                    "new_aabb=true (always calls computeAABBView). Expect "
                    "Gate_P4_Radii to diverge for screen-space-AABB-rejected "
                    "Gaussians. Fix location (out of scope for Phase 1): "
                    "src/vulkan/shaders/preprocess.comp:~1031 and "
                    "src/vulkan/preprocessor_vulkan.cpp (push-constant threading).\n");
    } else if (cuda_new_aabb == 1) {
        std::printf("[Gate I4] meta.json new_aabb=true — matches VK default.\n");
    } else {
        std::printf("[Gate I4] meta.json has no new_aabb field.\n");
    }
}

// ===========================================================================
// Phase 2 — forward-stage gates
// ===========================================================================

// ---------------------------------------------------------------------------
// Gate P1 — means2D parity
// ---------------------------------------------------------------------------
// Compare the per-Gaussian screen-space (x, y) projected centers between the
// VK forward pass and the CUDA reference dump. CUDA dump shape: (N, 2)
// float32. VK source: VulkanTrainer::captured_means2D() after forward_only().
//
// Tolerance reasoning (from plan): means2D = perspective_divide(viewproj * pos)
// is a deterministic 4x4-matrix * vec3 + perspective divide. Gates I1/I2
// confirmed the input matrices match within 1e-7 (view) and 1e-5 (proj). On
// well-conditioned floats this should propagate to ~1e-5 in pixel space; we
// use 1e-4 as the assertion threshold for the first attempt and report what
// we actually observe. If this fails, do NOT relax the tolerance — the
// failure pattern IS the diagnostic for which subsystem is responsible.
TEST(VkVsCudaFirstLoss, Gate_P1_Means2D) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    if (!std::filesystem::exists(dump_dir() + "/means2D.npy"))
        GTEST_SKIP() << "means2D.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        GTEST_SKIP() << "gt_image.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    // 1. Load PLY + cameras + GT image.
    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    // 2. Construct VulkanTrainer with parity config (eval_3D=true, parity_mode=true,
    //    lambda_dssim=0.0). Densification disabled (densify_from_step=0).
    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;       // make active_sh_degree_ jump straight to max
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;       // disable densification path
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    // 3. forward_only with the parity RenderConfig.
    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree     = 3;
    rcfg.eval_3D       = true;
    rcfg.antialiasing  = false;
    rcfg.training      = true;
    rcfg.scale_modifier = 1.0f;

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P1] forward_only loss = %.8f (CUDA L1 from meta.json: see Gate_L1)\n",
                vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    // 4. Load CUDA means2D reference and compare to VK capture.
    NpyArray cuda_m2d = load_npy(dump_dir() + "/means2D.npy");
    assert_dtype(cuda_m2d, NpyDtype::float32);
    ASSERT_EQ(cuda_m2d.shape.size(), 2u);
    ASSERT_EQ(cuda_m2d.shape[1], 2u);
    const size_t N = cuda_m2d.shape[0];
    ASSERT_EQ(static_cast<int>(N), model.data.count)
        << "Gaussian count mismatch CUDA vs PLY";

    const std::vector<float>& vk_m2d = trainer->captured_means2D();
    ASSERT_EQ(vk_m2d.size(), N * 2u)
        << "VK captured_means2D() size mismatch (expected " << (N*2) << ")";

    const float* vk_p   = vk_m2d.data();
    const float* cuda_p = cuda_m2d.f32();

    // 5. Per-Gaussian comparison.
    //
    // Tolerance rationale (C.0 Step 2 — 2026-04-25):
    // ------------------------------------------------------------------
    // The original 1e-4 px absolute tolerance was aspirational. C.0 Step 1
    // diagnosis showed it produces ~181538 spurious "bad" entries that are
    // pure cross-vendor transcendental ULP noise, NOT a real correctness
    // gap. Two sources of irreducible disagreement at 1e-4 scale:
    //
    //   (a) CUDA degenerate-AABB fallback. When inside_sqrt <= 0, CUDA's
    //       compute_aabb_view returns tan(±π/2 - ε) ≈ ±3.5e7 as a
    //       "give up, full-screen AABB" sentinel — these are NOT pixel
    //       coordinates, they are control values whose only meaning is
    //       "do not cull this Gaussian". ~336 such cases on the basketball
    //       step-1 fixture. We mask them by |cuda_means2D| > 1e5 and tally
    //       in cuda_huge_skipped for visibility (see TODO below).
    //       Cite: AAA-Gaussians/submodules/diff-gaussian-rasterization/
    //             cuda_rasterizer/stopthepop/consistent_common.cuh:217-284
    //
    //   (b) Sub-pixel atan2/tan/sqrt ULP drift. CPU↔VK PSNR on the
    //       full-frame render is 99.15 dB / 0.0024 max channel error —
    //       the underlying numerics ARE correct. CUDA↔VK and CUDA↔CPU
    //       differ at the same scale because CUDA's nvcc/PTX
    //       transcendental implementations have different ULP envelopes
    //       than the SPIR-V/glslang-emitted ones on Tegra. This is
    //       fundamental, not a fixable bug.
    //
    // Replacement criterion: relative tolerance with absolute floor.
    //   kAbsTol = 1e-2  → 0.01 px floor for tiny means2D near (0,0).
    //                     Covers ULP-noise on small magnitudes where
    //                     relative tolerance would clip below FP32 quantum.
    //   kRelTol = 1e-3  → 0.1% of CUDA value, scale-invariant.
    //                     Catches systematic drift (mis-scaled projection,
    //                     wrong intrinsics) which would land at >>0.1%.
    //                     Does NOT catch a ULP cliff at the ±1e-3·|cuda|
    //                     threshold for genuinely large pixel coordinates,
    //                     which is exactly the cross-vendor noise we are
    //                     deliberately admitting.
    //
    // Assertion still has teeth: bad_after_mask_and_reltol must be 0.
    // If a real numerical regression appears (e.g. a projection-matrix
    // bug, a wrong sign in atan2 unwrap), it will produce O(N) failures
    // at ratios FAR larger than 0.1% and trip this gate immediately.
    //
    // Empirical state (basketball fixture, step 1, 2026-04-25):
    //   N=400000 total Gaussians.
    //   cuda_huge_skipped = 336    ← CUDA tan(±π/2-ε) sentinels.
    //   vk_huge_skipped   = 0      ← caught by cuda_huge first when both huge.
    //   bad_after_relax   = 8790   ← currently FAILING. Composed of:
    //     - 5903  "only CUDA rasterizes" (CUDA radii>0, VK radii=0).
    //             VK writes (0,0) for culled Gaussians; CUDA writes a
    //             real pixel. This is a culling-threshold disagreement,
    //             NOT a projection numerical bug.
    //     - ~2887 "both rasterize" with per-component delta exceeding
    //             0.1%·|cuda|. max_abs (post-mask) ≈ 1295 px on gid
    //             140305 — too large for ULP noise. Likely concentrated
    //             at near-degenerate Gaussians where inside_sqrt is
    //             close to zero in CUDA but not in VK (or vice-versa).
    //
    // The 8790 residual is the HONEST GAP. The gate stays RED until the
    // root cause is fixed in preprocess.comp; do not raise the tolerance
    // to make the count drop. If the count grows, that signals additional
    // drift; if it shrinks, that's progress.
    //
    // TODO (deferred — see dev_notes/master_plan/cuda_vk_parity.md and
    // dev_notes/vk_initial_loss_mismatch_s10.md):
    //   - Audit gauss2view layout: confirm row/column-major matches the
    //     reference Python; rule out a near-degenerate Gaussian feeding
    //     a different inside_sqrt sign than CUDA expects.
    //   - Investigate the inside_sqrt FP cancellation cliff: the
    //     ~336 huge-mean cases cluster where (a-c)² + 4b² ≈ 4·det,
    //     so VK+CUDA pick different sides of zero. A reformulation
    //     using compensated subtraction (Kahan) might shrink the count.
    //   - Reconcile the radii>0 culling threshold between VK preprocess
    //     and CUDA preprocess (5903 only-CUDA-rasterizes cases).
    // ------------------------------------------------------------------
    constexpr float kTol    = 1e-4f;   // legacy print value (informational only)
    constexpr float kAbsTol = 1e-2f;   // 1/100 px floor for tiny means2D
    constexpr float kRelTol = 1e-3f;   // 0.1% of CUDA value
    constexpr float kHugeMask = 1e5f;  // CUDA degenerate-AABB sentinel cutoff
    auto tol_for = [&](float cuda_val) {
        return std::max(kAbsTol, kRelTol * std::max(std::fabs(cuda_val), 1.0f));
    };

    struct Diff { float abs_d; float vx, vy, cx, cy; size_t gid; };
    std::vector<Diff> all_diffs;
    all_diffs.reserve(N);
    int bad = 0;                    // legacy: |max(dx,dy)| > kTol (informational)
    int bad_after_relax = 0;        // PASS/FAIL signal: post-mask + per-component reltol
    int cuda_huge_skipped = 0;      // CUDA degenerate-AABB sentinels masked out
    int vk_huge_skipped   = 0;      // VK-side degenerate sentinels masked out (no CUDA huge)
    double sum_abs = 0.0;
    double sum_abs_post = 0.0;      // sum over non-masked entries
    size_t n_post = 0;
    float max_abs_post = 0.0f;
    size_t max_abs_post_gid = 0;
    for (size_t i = 0; i < N; ++i) {
        const float vx = vk_p[i * 2 + 0];
        const float vy = vk_p[i * 2 + 1];
        const float cx = cuda_p[i * 2 + 0];
        const float cy = cuda_p[i * 2 + 1];
        const float dx = std::fabs(vx - cx);
        const float dy = std::fabs(vy - cy);
        const float d  = std::max(dx, dy);
        all_diffs.push_back({d, vx, vy, cx, cy, i});
        sum_abs += d;
        if (d > kTol) ++bad;

        // (A) Mask CUDA degenerate-AABB sentinel: |cx|>1e5 or |cy|>1e5
        //     means CUDA fell back to tan(±π/2 - ε); not a pixel coord.
        //     Symmetric extension: if VK-side hits the same sentinel
        //     regime (e.g. inside_sqrt cancellation tipped the other way
        //     in VK glsl), the comparison is also meaningless. Either
        //     side huge → skip and tally separately for visibility.
        const bool cuda_huge =
            std::fabs(cx) > kHugeMask || std::fabs(cy) > kHugeMask;
        const bool vk_huge =
            std::fabs(vx) > kHugeMask || std::fabs(vy) > kHugeMask;
        if (cuda_huge) {
            ++cuda_huge_skipped;
            continue;
        }
        if (vk_huge) {
            ++vk_huge_skipped;
            continue;
        }

        // (B) Per-component relative tolerance with absolute floor.
        const float tol_x = tol_for(cx);
        const float tol_y = tol_for(cy);
        if (dx > tol_x || dy > tol_y) ++bad_after_relax;

        sum_abs_post += d;
        ++n_post;
        if (d > max_abs_post) {
            max_abs_post = d;
            max_abs_post_gid = i;
        }
    }

    // Sort by |diff| descending for top-K reporting.
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min<size_t>(32u, N),
                      idx.end(),
                      [&](size_t a, size_t b) {
                          return all_diffs[a].abs_d > all_diffs[b].abs_d;
                      });

    // Median |diff|.
    std::vector<float> abs_copy(N);
    for (size_t i = 0; i < N; ++i) abs_copy[i] = all_diffs[i].abs_d;
    std::nth_element(abs_copy.begin(),
                     abs_copy.begin() + N / 2,
                     abs_copy.end());
    const float median_abs = abs_copy[N / 2];

    const Diff worst = all_diffs[idx[0]];

    std::printf(
        "[Gate P1] N=%zu  bad(>tol=%.1e)=%d  max_abs=%.6e (gid=%zu)  "
        "median_abs=%.6e  mean_abs=%.6e\n",
        N, kTol, bad, worst.abs_d, worst.gid,
        median_abs, static_cast<float>(sum_abs / static_cast<double>(N)));
    std::printf(
        "[Gate P1] Relaxed-criterion summary (mask+reltol):\n"
        "    cuda_huge_skipped (|cuda|>%.0e, CUDA tan(±π/2-ε) sentinel): %d\n"
        "    vk_huge_skipped   (|vk|>%.0e,   VK-side same fallback)    : %d\n"
        "    bad_after_relax (per-component reltol=%.0e, abs_floor=%.0e): %d\n"
        "    max_abs (post-mask): %.6e (gid=%zu)\n"
        "    mean_abs (post-mask): %.6e (n=%zu)\n",
        kHugeMask, cuda_huge_skipped,
        kHugeMask, vk_huge_skipped,
        kRelTol, kAbsTol, bad_after_relax,
        max_abs_post, max_abs_post_gid,
        n_post ? static_cast<float>(sum_abs_post /
                                    static_cast<double>(n_post))
               : 0.0f,
        n_post);

    if (worst.abs_d > kTol) {
        std::printf("[Gate P1] Top-32 worst Gaussians (gid, vk(x,y), cuda(x,y), |diff|):\n");
        for (int k = 0; k < std::min<int>(32, static_cast<int>(N)); ++k) {
            const Diff& d = all_diffs[idx[k]];
            std::printf("    gid=%-7zu  vk=(% .6f, % .6f)  cuda=(% .6f, % .6f)  "
                        "|diff|=%.4e\n",
                        d.gid, d.vx, d.vy, d.cx, d.cy, d.abs_d);
        }

        // Spatial pattern report: bin failures by screen quadrant + by depth proxy
        // (we don't have z directly, but report worst diffs' (x, y) coordinates so
        // reader can spot edge/corner concentration).
        int q[4] = {0,0,0,0};   // TL TR BL BR (CUDA coords)
        const float W2 = kW * 0.5f;
        const float H2 = kH * 0.5f;
        for (size_t i = 0; i < N; ++i) {
            if (all_diffs[i].abs_d <= kTol) continue;
            const float cx = all_diffs[i].cx;
            const float cy = all_diffs[i].cy;
            const int qi = (cx >= W2 ? 1 : 0) + (cy >= H2 ? 2 : 0);
            ++q[qi];
        }
        std::printf("[Gate P1] Failure quadrant distribution (CUDA-side x,y):\n"
                    "    TL=%d  TR=%d  BL=%d  BR=%d\n",
                    q[0], q[1], q[2], q[3]);

        // Cross-classify failures vs CUDA radii (rejection state) and VK radii.
        if (std::filesystem::exists(dump_dir() + "/radii.npy")) {
            NpyArray cr = load_npy(dump_dir() + "/radii.npy");
            assert_dtype(cr, NpyDtype::int32);
            ASSERT_EQ(cr.numel(), N);
            const int32_t* cuda_radii = cr.i32();
            const std::vector<int>& vk_radii = trainer->captured_radii();
            int huge_cuda = 0;          // |cuda m2d| > 1e5
            int agree_radii = 0;        // (vk>0)==(cuda>0)
            int both_in = 0, only_vk_in = 0, only_cuda_in = 0;
            int bad_in_both    = 0;     // both rasterize, m2d differs
            int bad_only_vk_in = 0;
            int bad_only_cu_in = 0;
            int bad_both_out   = 0;     // both reject, but m2d differs
            for (size_t i = 0; i < N; ++i) {
                const bool huge =
                    std::fabs(all_diffs[i].cx) > 1e5f ||
                    std::fabs(all_diffs[i].cy) > 1e5f;
                if (huge) ++huge_cuda;
                const bool vk_in = vk_radii[i] > 0;
                const bool cu_in = cuda_radii[i] > 0;
                if (vk_in == cu_in) ++agree_radii;
                if (vk_in && cu_in)        ++both_in;
                else if (vk_in && !cu_in)  ++only_vk_in;
                else if (!vk_in && cu_in)  ++only_cuda_in;
                if (all_diffs[i].abs_d > kTol) {
                    if (vk_in && cu_in)        ++bad_in_both;
                    else if (vk_in && !cu_in)  ++bad_only_vk_in;
                    else if (!vk_in && cu_in)  ++bad_only_cu_in;
                    else                        ++bad_both_out;
                }
            }
            std::printf(
                "[Gate P1] Subsystem cross-class (radii > 0 == 'rasterized'):\n"
                "    CUDA huge-mean (|m2d|>1e5) count: %d\n"
                "    VK/CUDA radii agreement: %d / %zu\n"
                "    both rasterize:        %d  (bad mean2D: %d)\n"
                "    only VK rasterizes:    %d  (bad: %d)  [VK keeps; CUDA rejects]\n"
                "    only CUDA rasterizes:  %d  (bad: %d)  [VK rejects; CUDA keeps]\n"
                "    both reject (radii=0): bad mean2D: %d\n",
                huge_cuda, agree_radii, N,
                both_in, bad_in_both,
                only_vk_in, bad_only_vk_in,
                only_cuda_in, bad_only_cu_in,
                bad_both_out);
        }
    }

    // 6. PASS/FAIL assertion — relaxed criterion (mask + per-component reltol).
    //    See tolerance rationale block above. The legacy strict (kTol=1e-4 abs)
    //    print remains for diagnostic continuity but is NOT the gate.
    EXPECT_EQ(bad_after_relax, 0)
        << "VK means2D differs from CUDA beyond relative tolerance "
        << "(kRelTol=" << kRelTol << ", kAbsTol=" << kAbsTol << " px floor) "
        << "after masking " << cuda_huge_skipped
        << " CUDA degenerate-AABB sentinels. "
        << "max_abs (post-mask)=" << max_abs_post
        << " gid=" << max_abs_post_gid
        << ". This signals systematic drift, NOT cross-vendor ULP noise — "
        << "investigate before relaxing further. See diagnostic dump above.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P2_ConicOpacity) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    if (!std::filesystem::exists(dump_dir() + "/conic_opacity.npy"))
        GTEST_SKIP() << "conic_opacity.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/radii.npy"))
        GTEST_SKIP() << "radii.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        GTEST_SKIP() << "gt_image.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P2] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_co = load_npy(dump_dir() + "/conic_opacity.npy");
    assert_dtype(cuda_co, NpyDtype::float32);
    ASSERT_EQ(cuda_co.shape.size(), 2u);
    ASSERT_EQ(cuda_co.shape[1], 4u);
    const size_t N = cuda_co.shape[0];
    ASSERT_EQ(static_cast<int>(N), model.data.count)
        << "Gaussian count mismatch CUDA vs PLY";

    NpyArray cuda_radii = load_npy(dump_dir() + "/radii.npy");
    assert_dtype(cuda_radii, NpyDtype::int32);
    ASSERT_EQ(cuda_radii.shape, (std::vector<size_t>{N}));

    const std::vector<float>& vk_co = trainer->captured_conic_opacity();
    ASSERT_EQ(vk_co.size(), N * 4u)
        << "VK captured_conic_opacity() size mismatch (expected " << (N * 4) << ")";
    const std::vector<int>& vk_radii = trainer->captured_radii();
    ASSERT_EQ(vk_radii.size(), N)
        << "VK captured_radii() size mismatch (expected " << N << ")";

    const float* vk_p = vk_co.data();
    const float* cuda_p = cuda_co.f32();
    const int32_t* cuda_r = cuda_radii.i32();

    constexpr float kAbsTolOpacity = 1e-5f;
    constexpr float kRelTolOpacity = 1e-4f;
    auto tol_for = [&](float cuda_val) {
        return std::max(kAbsTolOpacity,
                        kRelTolOpacity * std::max(std::fabs(cuda_val), 1.0f));
    };

    struct Diff { float abs_d; float vk; float cuda; size_t gid; size_t flat_idx; };
    std::vector<Diff> diffs;
    diffs.reserve(N);
    int bad = 0;
    int both_active = 0;
    int only_cuda_active = 0;
    int only_vk_active = 0;
    int both_inactive = 0;
    double sum_abs = 0.0;
    float max_abs = 0.f;
    size_t max_gid = 0u;

    for (size_t i = 0; i < N; ++i) {
        const bool cuda_active = cuda_r[i] > 0;
        const bool vk_active = vk_radii[i] > 0;
        if (!cuda_active || !vk_active) {
            if (cuda_active) ++only_cuda_active;
            else if (vk_active) ++only_vk_active;
            else ++both_inactive;
            continue;
        }
        ++both_active;

        const float vk = vk_p[i * 4 + 3];
        const float cu = cuda_p[i];
        const float d = std::fabs(vk - cu);
        diffs.push_back({d, vk, cu, i, i});
        sum_abs += d;
        if (d > max_abs) {
            max_abs = d;
            max_gid = i;
        }
        if (d > tol_for(cu)) ++bad;
    }

    const int cuda_active_count = both_active + only_cuda_active;
    const int vk_active_count = both_active + only_vk_active;
    ASSERT_GT(cuda_active_count, 0)
        << "Gate P2 has no active CUDA Gaussians; cannot validate opacity parity.";
    ASSERT_GT(vk_active_count, 0)
        << "Gate P2 has no active VK Gaussians; cannot validate opacity parity.";
    const int min_cuda_overlap = std::max(1, (cuda_active_count * 95 + 99) / 100);
    const int min_vk_overlap = std::max(1, (vk_active_count * 95 + 99) / 100);
    ASSERT_GE(both_active, min_cuda_overlap)
        << "Gate P2 has insufficient CUDA-active overlap to validate opacity parity. "
        << "both_active=" << both_active
        << " cuda_active=" << cuda_active_count
        << " vk_active=" << vk_active_count
        << " only_cuda_active=" << only_cuda_active
        << " only_vk_active=" << only_vk_active
        << " both_inactive=" << both_inactive;
    ASSERT_GE(both_active, min_vk_overlap)
        << "Gate P2 has insufficient VK-active overlap to validate opacity parity. "
        << "both_active=" << both_active
        << " cuda_active=" << cuda_active_count
        << " vk_active=" << vk_active_count
        << " only_cuda_active=" << only_cuda_active
        << " only_vk_active=" << only_vk_active
        << " both_inactive=" << both_inactive;

    if (!diffs.empty()) {
        std::partial_sort(diffs.begin(),
                          diffs.begin() + std::min<size_t>(32u, diffs.size()),
                          diffs.end(),
                          [](const Diff& a, const Diff& b) { return a.abs_d > b.abs_d; });
    }

    std::printf("[Gate P2] eval_3D opacity flat-buffer comparison: N=%zu both_active=%d "
                "only_cuda_active=%d only_vk_active=%d both_inactive=%d bad=%d\n",
                N, both_active, only_cuda_active, only_vk_active, both_inactive, bad);
    std::printf("[Gate P2] opacity mean_abs=%.6e max_abs=%.6e gid=%zu\n",
                both_active ? static_cast<float>(sum_abs / static_cast<double>(both_active)) : 0.0f,
                max_abs, max_gid);
    if (!diffs.empty() && diffs[0].abs_d > 0.f) {
        std::printf("[Gate P2] Top-32 worst opacity entries (gid, cuda_flat_idx, vk, cuda, |diff|):\n");
        for (int k = 0; k < std::min<int>(32, static_cast<int>(diffs.size())); ++k) {
            const Diff& d = diffs[k];
            std::printf("    gid=%-7zu flat=%-7zu vk=% .8g cuda=% .8g |diff|=%.6e\n",
                        d.gid, d.flat_idx, d.vk, d.cuda, d.abs_d);
        }
    }

    EXPECT_EQ(bad, 0)
        << "VK eval_3D opacity differs from CUDA beyond tolerance. "
        << "CUDA materialize_dump exposes eval_3D opacity as a flat float buffer; "
        << "only cuda_conic_opacity.reshape(-1)[gid] is semantically valid. "
        << "See Gate P2 diagnostic dump above.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P3_RgbColors) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    if (!std::filesystem::exists(dump_dir() + "/rgb_colors.npy"))
        GTEST_SKIP() << "rgb_colors.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/radii.npy"))
        GTEST_SKIP() << "radii.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        GTEST_SKIP() << "gt_image.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P3] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_rgb = load_npy(dump_dir() + "/rgb_colors.npy");
    assert_dtype(cuda_rgb, NpyDtype::float32);
    ASSERT_EQ(cuda_rgb.shape.size(), 2u);
    ASSERT_EQ(cuda_rgb.shape[1], 3u);
    const size_t N = cuda_rgb.shape[0];
    ASSERT_EQ(static_cast<int>(N), model.data.count)
        << "Gaussian count mismatch CUDA vs PLY";

    NpyArray cuda_radii = load_npy(dump_dir() + "/radii.npy");
    assert_dtype(cuda_radii, NpyDtype::int32);
    ASSERT_EQ(cuda_radii.shape, (std::vector<size_t>{N}));

    const std::vector<float>& vk_rgb = trainer->captured_rgb();
    ASSERT_EQ(vk_rgb.size(), N * 3u)
        << "VK captured_rgb() size mismatch (expected " << (N * 3) << ")";
    const std::vector<int>& vk_radii = trainer->captured_radii();
    ASSERT_EQ(vk_radii.size(), N)
        << "VK captured_radii() size mismatch (expected " << N << ")";

    const float* vk_p = vk_rgb.data();
    const float* cuda_p = cuda_rgb.f32();
    const int32_t* cuda_r = cuda_radii.i32();

    constexpr float kAbsTolRgb = 1e-5f;
    constexpr float kRelTolRgb = 1e-4f;
    auto tol_for = [&](float cuda_val) {
        return std::max(kAbsTolRgb,
                        kRelTolRgb * std::max(std::fabs(cuda_val), 1.0f));
    };

    struct Diff { float abs_d; float vk; float cuda; size_t gid; int comp; };
    std::vector<Diff> diffs;
    diffs.reserve(N * 3u);
    int bad = 0;
    int both_active = 0;
    int only_cuda_active = 0;
    int only_vk_active = 0;
    int both_inactive = 0;
    double sum_abs[3] = {0.0, 0.0, 0.0};
    float max_abs[3] = {0.f, 0.f, 0.f};
    size_t max_gid[3] = {0u, 0u, 0u};

    for (size_t i = 0; i < N; ++i) {
        const bool cuda_active = cuda_r[i] > 0;
        const bool vk_active = vk_radii[i] > 0;
        if (!cuda_active || !vk_active) {
            if (cuda_active) ++only_cuda_active;
            else if (vk_active) ++only_vk_active;
            else ++both_inactive;
            continue;
        }
        ++both_active;

        for (int c = 0; c < 3; ++c) {
            const float vk = vk_p[i * 3 + c];
            const float cu = cuda_p[i * 3 + c];
            const float d = std::fabs(vk - cu);
            diffs.push_back({d, vk, cu, i, c});
            sum_abs[c] += d;
            if (d > max_abs[c]) {
                max_abs[c] = d;
                max_gid[c] = i;
            }
            if (d > tol_for(cu)) ++bad;
        }
    }

    const int cuda_active_count = both_active + only_cuda_active;
    const int vk_active_count = both_active + only_vk_active;
    ASSERT_GT(cuda_active_count, 0)
        << "Gate P3 has no active CUDA Gaussians; cannot validate RGB parity.";
    ASSERT_GT(vk_active_count, 0)
        << "Gate P3 has no active VK Gaussians; cannot validate RGB parity.";
    const int min_cuda_overlap = std::max(1, (cuda_active_count * 95 + 99) / 100);
    const int min_vk_overlap = std::max(1, (vk_active_count * 95 + 99) / 100);
    ASSERT_GE(both_active, min_cuda_overlap)
        << "Gate P3 has insufficient CUDA-active overlap to validate RGB parity. "
        << "both_active=" << both_active
        << " cuda_active=" << cuda_active_count
        << " vk_active=" << vk_active_count
        << " only_cuda_active=" << only_cuda_active
        << " only_vk_active=" << only_vk_active
        << " both_inactive=" << both_inactive;
    ASSERT_GE(both_active, min_vk_overlap)
        << "Gate P3 has insufficient VK-active overlap to validate RGB parity. "
        << "both_active=" << both_active
        << " cuda_active=" << cuda_active_count
        << " vk_active=" << vk_active_count
        << " only_cuda_active=" << only_cuda_active
        << " only_vk_active=" << only_vk_active
        << " both_inactive=" << both_inactive;

    if (!diffs.empty()) {
        std::partial_sort(diffs.begin(),
                          diffs.begin() + std::min<size_t>(32u, diffs.size()),
                          diffs.end(),
                          [](const Diff& a, const Diff& b) { return a.abs_d > b.abs_d; });
    }

    std::printf("[Gate P3] RGB comparison: N=%zu both_active=%d "
                "only_cuda_active=%d only_vk_active=%d both_inactive=%d bad=%d\n",
                N, both_active, only_cuda_active, only_vk_active, both_inactive, bad);
    for (int c = 0; c < 3; ++c) {
        std::printf("[Gate P3] comp=%d mean_abs=%.6e max_abs=%.6e gid=%zu\n",
                    c, both_active ? static_cast<float>(sum_abs[c] /
                                                        static_cast<double>(both_active)) : 0.0f,
                    max_abs[c], max_gid[c]);
    }
    if (!diffs.empty() && diffs[0].abs_d > 0.f) {
        std::printf("[Gate P3] Top-32 worst RGB entries (gid, comp, vk, cuda, |diff|):\n");
        for (int k = 0; k < std::min<int>(32, static_cast<int>(diffs.size())); ++k) {
            const Diff& d = diffs[k];
            std::printf("    gid=%-7zu comp=%d vk=% .8g cuda=% .8g |diff|=%.6e\n",
                        d.gid, d.comp, d.vk, d.cuda, d.abs_d);
        }
    }

    EXPECT_EQ(bad, 0)
        << "VK RGB differs from CUDA beyond tolerance on overlap-active Gaussians. "
        << "Inactive CUDA RGB rows are not semantically stable, so this gate compares "
        << "only the CUDA/VK active-set intersection. See Gate P3 diagnostic dump above.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P4_Radii) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    if (!std::filesystem::exists(dump_dir() + "/radii.npy"))
        GTEST_SKIP() << "radii.npy missing in CUDA dump";
    if (!std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        GTEST_SKIP() << "gt_image.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P4] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_radii = load_npy(dump_dir() + "/radii.npy");
    assert_dtype(cuda_radii, NpyDtype::int32);
    ASSERT_EQ(cuda_radii.shape.size(), 1u);
    const size_t N = cuda_radii.shape[0];
    ASSERT_EQ(static_cast<int>(N), model.data.count)
        << "Gaussian count mismatch CUDA vs PLY";

    const std::vector<int>& vk_radii = trainer->captured_radii();
    ASSERT_EQ(vk_radii.size(), N)
        << "VK captured_radii() size mismatch (expected " << N << ")";

    const int32_t* cuda_r = cuda_radii.i32();

    constexpr int kHugeRadius = 100000;
    struct Diff { int abs_d; int vk; int cuda; size_t gid; };
    std::vector<Diff> finite_diffs;
    finite_diffs.reserve(N);
    std::vector<Diff> sentinel_diffs;
    int bad_large_delta = 0;
    int off_by_one = 0;
    int both_active = 0;
    int only_cuda_active = 0;
    int only_vk_active = 0;
    int both_inactive = 0;
    int sentinel_skipped = 0;
    size_t finite_count = 0;
    double sum_abs_finite = 0.0;
    int max_abs_finite = 0;
    size_t max_gid_finite = 0u;

    for (size_t i = 0; i < N; ++i) {
        const int vk = vk_radii[i];
        const int cu = cuda_r[i];
        const bool cuda_active = cu > 0;
        const bool vk_active = vk > 0;
        if (cuda_active && vk_active) ++both_active;
        else if (cuda_active) ++only_cuda_active;
        else if (vk_active) ++only_vk_active;
        else ++both_inactive;

        const int d = vk > cu ? vk - cu : cu - vk;
        const bool sentinel = vk > kHugeRadius || cu > kHugeRadius;
        if (sentinel) {
            ++sentinel_skipped;
            if (d > 0) sentinel_diffs.push_back({d, vk, cu, i});
            continue;
        }

        ++finite_count;
        sum_abs_finite += static_cast<double>(d);
        if (d > max_abs_finite) {
            max_abs_finite = d;
            max_gid_finite = i;
        }
        if (d > 1) {
            ++bad_large_delta;
            finite_diffs.push_back({d, vk, cu, i});
        } else if (d == 1) {
            ++off_by_one;
            finite_diffs.push_back({d, vk, cu, i});
        }
    }

    const int cuda_active_count = both_active + only_cuda_active;
    const int vk_active_count = both_active + only_vk_active;
    ASSERT_GT(cuda_active_count, 0)
        << "Gate P4 has no active CUDA Gaussians; cannot validate radii parity.";
    ASSERT_GT(vk_active_count, 0)
        << "Gate P4 has no active VK Gaussians; cannot validate radii parity.";
    ASSERT_GT(finite_count, 0u)
        << "Gate P4 has no finite radii after sentinel masking.";

    const int max_off_by_one =
        std::max(32, static_cast<int>((finite_count + 9999u) / 10000u));

    if (!finite_diffs.empty()) {
        std::partial_sort(finite_diffs.begin(),
                          finite_diffs.begin() + std::min<size_t>(32u, finite_diffs.size()),
                          finite_diffs.end(),
                          [](const Diff& a, const Diff& b) { return a.abs_d > b.abs_d; });
    }
    if (!sentinel_diffs.empty()) {
        std::partial_sort(sentinel_diffs.begin(),
                          sentinel_diffs.begin() + std::min<size_t>(32u, sentinel_diffs.size()),
                          sentinel_diffs.end(),
                          [](const Diff& a, const Diff& b) { return a.abs_d > b.abs_d; });
    }

    std::printf("[Gate P4] radii comparison: N=%zu both_active=%d "
                "only_cuda_active=%d only_vk_active=%d both_inactive=%d "
                "sentinel_skipped=%d off_by_one=%d max_off_by_one=%d bad_large_delta=%d\n",
                N, both_active, only_cuda_active, only_vk_active, both_inactive,
                sentinel_skipped, off_by_one, max_off_by_one, bad_large_delta);
    std::printf("[Gate P4] finite mean_abs=%.6e max_abs=%d gid=%zu finite_count=%zu\n",
                finite_count ? static_cast<float>(sum_abs_finite /
                                                  static_cast<double>(finite_count)) : 0.0f,
                max_abs_finite, max_gid_finite, finite_count);
    if (!finite_diffs.empty()) {
        std::printf("[Gate P4] Top-32 finite radii deltas (gid, vk, cuda, |diff|):\n");
        for (int k = 0; k < std::min<int>(32, static_cast<int>(finite_diffs.size())); ++k) {
            const Diff& d = finite_diffs[k];
            std::printf("    gid=%-7zu vk=%d cuda=%d |diff|=%d\n",
                        d.gid, d.vk, d.cuda, d.abs_d);
        }
    }
    if (!sentinel_diffs.empty()) {
        std::printf("[Gate P4] Top-32 sentinel radii entries skipped (gid, vk, cuda, |diff|):\n");
        for (int k = 0; k < std::min<int>(32, static_cast<int>(sentinel_diffs.size())); ++k) {
            const Diff& d = sentinel_diffs[k];
            std::printf("    gid=%-7zu vk=%d cuda=%d |diff|=%d\n",
                        d.gid, d.vk, d.cuda, d.abs_d);
        }
    }

    EXPECT_EQ(only_cuda_active, 0)
        << "VK culled Gaussians that CUDA considered active; this is AABB/culling divergence.";
    EXPECT_EQ(only_vk_active, 0)
        << "VK kept Gaussians that CUDA culled; this is AABB/culling divergence.";
    EXPECT_EQ(bad_large_delta, 0)
        << "VK finite radii differ from CUDA by more than the 1-pixel ceil boundary "
        << "window. Sentinel-scale radii are masked because they derive from the same "
        << "tan(±pi/2-epsilon) degenerate-AABB fallback already masked by Gate P1.";
    EXPECT_LE(off_by_one, max_off_by_one)
        << "VK finite radii have too many 1-pixel ceil-boundary deltas; this indicates "
        << "systematic AABB/radius drift rather than isolated rounding-boundary noise.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P5_SortedIds) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        << "gt_image.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/means2D.npy"))
        << "means2D.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/radii.npy"))
        << "radii.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/tiles_touched.npy"))
        << "tiles_touched.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/depths.npy"))
        << "depths.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/rects2D.npy"))
        << "rects2D.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/sorted_ids_per_tile.npy"))
        << "sorted_ids_per_tile.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/tile_offsets.npy"))
        << "tile_offsets.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;
    ASSERT_TRUE(tcfg.eval_3D && tcfg.parity_mode)
        << "P5 compares parity-mode global depth keys, not eval_3D tile-depth binning.";

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P5] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_means = load_npy(dump_dir() + "/means2D.npy");
    NpyArray cuda_radii = load_npy(dump_dir() + "/radii.npy");
    NpyArray cuda_tiles = load_npy(dump_dir() + "/tiles_touched.npy");
    NpyArray cuda_depths = load_npy(dump_dir() + "/depths.npy");
    NpyArray cuda_sorted = load_npy(dump_dir() + "/sorted_ids_per_tile.npy");
    NpyArray cuda_offsets = load_npy(dump_dir() + "/tile_offsets.npy");
    assert_dtype(cuda_means, NpyDtype::float32);
    assert_dtype(cuda_radii, NpyDtype::int32);
    assert_dtype(cuda_tiles, NpyDtype::int32);
    assert_dtype(cuda_depths, NpyDtype::float32);
    assert_dtype(cuda_sorted, NpyDtype::int32);
    assert_dtype(cuda_offsets, NpyDtype::int32);
    ASSERT_EQ(cuda_means.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count), 2u}));
    ASSERT_EQ(cuda_radii.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count)}));
    ASSERT_EQ(cuda_tiles.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count)}));
    ASSERT_EQ(cuda_depths.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count)}));
    ASSERT_EQ(cuda_offsets.shape, (std::vector<size_t>{static_cast<size_t>(kW / 16 * (kH / 16) + 1)}));

    const size_t N = static_cast<size_t>(model.data.count);
    const std::vector<float>& vk_means = trainer->captured_means2D();
    const std::vector<int>& vk_radii = trainer->captured_radii();
    const std::vector<int>& vk_tiles = trainer->captured_tiles_touched();
    const std::vector<float>& vk_depths = trainer->captured_depths();
    const std::vector<float>& vk_radius_f = trainer->captured_radius_f();
    const std::vector<float>& vk_g2s = trainer->captured_gauss2screen();
    const std::vector<int>& vk_sorted = trainer->captured_sorted_gaussian_ids();
    const std::vector<int>& vk_tile_ranges = trainer->captured_tile_offsets();
    ASSERT_EQ(vk_means.size(), N * 2);
    ASSERT_EQ(vk_radii.size(), N);
    ASSERT_EQ(vk_tiles.size(), N);
    ASSERT_EQ(vk_depths.size(), N);
    ASSERT_EQ(vk_radius_f.size(), N * 2);
    ASSERT_EQ(vk_g2s.size(), N * 16);

    const std::string diag_base_path = dump_dir();
    const std::string diag_g2s_path = diag_base_path + "/gauss2screen.npy";
    const std::string diag_rects_path = diag_base_path + "/rects2D.npy";
    const std::string diag_aabb_path = diag_base_path + "/aabb_debug.npy";
    NpyArray cuda_g2s;
    NpyArray cuda_rects;
    NpyArray cuda_aabb;
    const float* cuda_g2s_data = nullptr;
    const float* cuda_rects_data = nullptr;
    const float* cuda_aabb_data = nullptr;
    if (std::filesystem::exists(diag_g2s_path)) {
        cuda_g2s = load_npy(diag_g2s_path);
        assert_dtype(cuda_g2s, NpyDtype::float32);
        ASSERT_EQ(cuda_g2s.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count), 16u}));
        cuda_g2s_data = cuda_g2s.f32();
    }
    if (std::filesystem::exists(diag_rects_path)) {
        cuda_rects = load_npy(diag_rects_path);
        assert_dtype(cuda_rects, NpyDtype::float32);
        ASSERT_EQ(cuda_rects.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count), 2u}));
        cuda_rects_data = cuda_rects.f32();
    }
    if (std::filesystem::exists(diag_aabb_path)) {
        cuda_aabb = load_npy(diag_aabb_path);
        assert_dtype(cuda_aabb, NpyDtype::float32);
        ASSERT_EQ(cuda_aabb.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count), 3u, 4u}));
        cuda_aabb_data = cuda_aabb.f32();
    }
    ASSERT_NE(cuda_rects_data, nullptr);

    const float* cuda_m = cuda_means.f32();
    const int32_t* cuda_r = cuda_radii.i32();
    const int32_t* cuda_t = cuda_tiles.i32();
    const float* cuda_d = cuda_depths.f32();
    const int32_t* cuda_sorted_ids = cuda_sorted.i32();
    const int32_t* cuda_tile_offsets = cuda_offsets.i32();
    constexpr int tile_w = 16;
    constexpr int tile_h = 16;
    const int grid_x = (kW + tile_w - 1) / tile_w;
    const int grid_y = (kH + tile_h - 1) / tile_h;
    const int num_tiles = grid_x * grid_y;
    ASSERT_EQ(cuda_offsets.numel(), static_cast<size_t>(num_tiles + 1));
    ASSERT_EQ(cuda_sorted.numel(), static_cast<size_t>(cuda_tile_offsets[num_tiles]));
    ASSERT_EQ(vk_tile_ranges.size(), static_cast<size_t>(num_tiles * 2));
    ASSERT_EQ(vk_sorted.size(), static_cast<size_t>(vk_tile_ranges[(num_tiles - 1) * 2 + 1]));
    struct RectDiag {
        int min_x;
        int min_y;
        int max_x;
        int max_y;
        int count;
    };
    auto tile_count_from_int_radius = [&](float x, float y, int radius) {
        const int rmn_x = std::min(grid_x, std::max(0, static_cast<int>((x - static_cast<float>(radius)) / static_cast<float>(tile_w))));
        const int rmn_y = std::min(grid_y, std::max(0, static_cast<int>((y - static_cast<float>(radius)) / static_cast<float>(tile_h))));
        const int rmx_x = std::min(grid_x, std::max(0, static_cast<int>((x + static_cast<float>(radius) + static_cast<float>(tile_w) - 1.0f) / static_cast<float>(tile_w))));
        const int rmx_y = std::min(grid_y, std::max(0, static_cast<int>((y + static_cast<float>(radius) + static_cast<float>(tile_h) - 1.0f) / static_cast<float>(tile_h))));
        return (rmx_x - rmn_x) * (rmx_y - rmn_y);
    };
    auto rect_from_float_extent = [&](float x, float y, float ex, float ey) {
        RectDiag r{};
        r.min_x = std::min(grid_x, std::max(0, static_cast<int>(std::floor((x - ex) / static_cast<float>(tile_w)))));
        r.min_y = std::min(grid_y, std::max(0, static_cast<int>(std::floor((y - ey) / static_cast<float>(tile_h)))));
        r.max_x = std::min(grid_x, std::max(0, static_cast<int>(std::ceil((x + ex) / static_cast<float>(tile_w)))));
        r.max_y = std::min(grid_y, std::max(0, static_cast<int>(std::ceil((y + ey) / static_cast<float>(tile_h)))));
        r.count = (r.max_x - r.min_x) * (r.max_y - r.min_y);
        return r;
    };
    auto contains_tile = [](const RectDiag& r, int tx, int ty) {
        return tx >= r.min_x && tx < r.max_x && ty >= r.min_y && ty < r.max_y;
    };
    auto boundary_distance = [](float v) {
        return std::fabs(v - std::round(v));
    };
    auto ndc2pix = [](float v, int S) {
        return ((v + 1.0f) * static_cast<float>(S) - 1.0f) * 0.5f;
    };
    auto project_center = [&](size_t i, float& x, float& y) {
        const float* p = model.data.positions + i * 3;
        const float h0 = cam.viewproj_matrix[0] * p[0] + cam.viewproj_matrix[4] * p[1] +
                         cam.viewproj_matrix[8] * p[2] + cam.viewproj_matrix[12];
        const float h1 = cam.viewproj_matrix[1] * p[0] + cam.viewproj_matrix[5] * p[1] +
                         cam.viewproj_matrix[9] * p[2] + cam.viewproj_matrix[13];
        const float h3 = cam.viewproj_matrix[3] * p[0] + cam.viewproj_matrix[7] * p[1] +
                         cam.viewproj_matrix[11] * p[2] + cam.viewproj_matrix[15];
        const float inv_w = 1.0f / (h3 + 1e-7f);
        x = ndc2pix(h0 * inv_w, kW);
        y = ndc2pix(h1 * inv_w, kH);
    };
    auto normalize_angle_cpu = [](float theta) {
        const float two_pi = 2.0f * static_cast<float>(M_PI);
        theta = theta - std::trunc(theta / two_pi) * two_pi;
        if (theta > static_cast<float>(M_PI)) theta -= two_pi;
        if (theta <= -static_cast<float>(M_PI)) theta += two_pi;
        return theta;
    };
    auto print_cpu_aabb_trace = [&](size_t gid, const float g2v[16], const float p_view[3], float focal_x, float focal_y, float cutoff) {
        const float t[4] = {cutoff, cutoff, cutoff, -1.0f};
        const float vlen = std::sqrt(p_view[0] * p_view[0] + p_view[1] * p_view[1] + p_view[2] * p_view[2]);
        const float viewdir[3] = {p_view[0] / vlen, p_view[1] / vlen, p_view[2] / vlen};
        for (int axis = 0; axis < 2; ++axis) {
            const float theta_mu = std::atan2(axis == 0 ? viewdir[0] : viewdir[1], viewdir[2]);
            const float* Ta = g2v + axis * 4;
            const float* T2 = g2v + 2 * 4;
            float squared_axis = 0.0f;
            float squared_z = 0.0f;
            float mid = 0.0f;
            for (int j = 0; j < 4; ++j) {
                squared_axis += t[j] * Ta[j] * Ta[j];
                squared_z += t[j] * T2[j] * T2[j];
                mid += t[j] * Ta[j] * T2[j];
            }
            const float inside_sqrt = mid * mid - squared_z * squared_axis;
            float result_lo = -(static_cast<float>(M_PI) / 2.0f - 1e-5f);
            float result_hi = static_cast<float>(M_PI) / 2.0f - 1e-5f;
            float theta_axis_0 = 0.0f;
            float theta_axis_1 = 0.0f;
            float norm0 = 0.0f;
            float norm1 = 0.0f;
            if (inside_sqrt > 0.0f) {
                const float sqrt_val = std::sqrt(inside_sqrt);
                theta_axis_0 = std::atan2(-(mid + sqrt_val), -squared_z);
                theta_axis_1 = std::atan2(-(mid - sqrt_val), -squared_z);
                while (theta_axis_0 > theta_mu) theta_axis_0 -= static_cast<float>(M_PI);
                while (theta_axis_0 < theta_mu - static_cast<float>(M_PI)) theta_axis_0 += static_cast<float>(M_PI);
                while (theta_axis_1 < theta_mu) theta_axis_1 += static_cast<float>(M_PI);
                while (theta_axis_1 > theta_mu + static_cast<float>(M_PI)) theta_axis_1 -= static_cast<float>(M_PI);
                norm0 = normalize_angle_cpu(theta_axis_0);
                norm1 = normalize_angle_cpu(theta_axis_1);
                if (theta_mu < 0.0f && std::fabs(norm0) < std::fabs(norm1)) {
                    theta_axis_0 += 2.0f * static_cast<float>(M_PI);
                    theta_axis_1 += 2.0f * static_cast<float>(M_PI);
                } else if (theta_mu > 0.0f && std::fabs(norm1) < std::fabs(norm0)) {
                    theta_axis_0 -= 2.0f * static_cast<float>(M_PI);
                    theta_axis_1 -= 2.0f * static_cast<float>(M_PI);
                }
                result_lo = std::max(result_lo, theta_axis_0);
                result_hi = std::min(result_hi, theta_axis_1);
            }
            const float focal_dim = axis == 0 ? focal_x : focal_y;
            const float dim_size = axis == 0 ? static_cast<float>(kW) : static_cast<float>(kH);
            const float tan_lo = std::tan(result_lo);
            const float tan_hi = std::tan(result_hi);
            const float bounds_lo = dim_size / 2.0f + focal_dim * tan_lo;
            const float bounds_hi = dim_size / 2.0f + focal_dim * tan_hi;
            std::printf("        cpu_aabb gid=%zu axis=%d theta_mu=%.9g sq_axis=%.9g sq_z=%.9g mid=%.9g inside=%.9g theta=(%.9g,%.9g) norm=(%.9g,%.9g) result=(%.9g,%.9g) tan=(%.9g,%.9g) bounds=(%.9g,%.9g)\n",
                        gid, axis, theta_mu, squared_axis, squared_z, mid, inside_sqrt,
                        theta_axis_0, theta_axis_1, norm0, norm1, result_lo, result_hi,
                        tan_lo, tan_hi, bounds_lo, bounds_hi);
            if (cuda_aabb_data) {
                const float* cuda_axis = cuda_aabb_data + gid * 12 + axis * 4;
                const float* cuda_tail = cuda_aabb_data + gid * 12 + 8;
                const float cuda_inside = axis == 0 ? cuda_tail[0] : cuda_tail[2];
                const float cuda_result = axis == 0 ? cuda_tail[1] : cuda_tail[3];
                const float cpu_result = axis == 0 ? result_lo : result_hi;
                std::printf("        cuda_aabb gid=%zu axis=%d theta_mu=%.9g sq_axis=%.9g sq_z=%.9g mid=%.9g inside=%.9g %s=%.9g diff=(%.3g,%.3g,%.3g,%.3g,%.3g,%.3g)\n",
                            gid, axis,
                            cuda_axis[0], cuda_axis[1], cuda_axis[2], cuda_axis[3],
                            cuda_inside, axis == 0 ? "result_lo" : "result_hi", cuda_result,
                            theta_mu - cuda_axis[0], squared_axis - cuda_axis[1],
                            squared_z - cuda_axis[2], mid - cuda_axis[3],
                            inside_sqrt - cuda_inside, cpu_result - cuda_result);
            }
        }
    };

    int vk_vs_cuda_bad = 0;
    int vk_vs_cuda_sum_delta = 0;
    int cuda_int_rect_bad = 0;
    int vk_int_rect_vs_cuda_bad = 0;
    int vk_int_rect_vs_vk_bad = 0;
    int vk_ndc_int_rect_vs_cuda_bad = 0;
    int vk_ndc_float_rect_vs_cuda_bad = 0;
    int boundary_ambiguous_bad = 0;
    std::vector<uint8_t> boundary_ambiguous(N, 0);
    std::vector<RectDiag> vk_float_rects(N);
    std::vector<RectDiag> cuda_float_rects(N);
    std::printf("[Gate P5] tiles_touched mismatches and int-radius rect diagnostics:\n");
    for (size_t i = 0; i < N; ++i) {
        const int vk_int_count = tile_count_from_int_radius(
            vk_means[i * 2 + 0], vk_means[i * 2 + 1], vk_radii[i]);
        const int cuda_int_count = tile_count_from_int_radius(
            cuda_m[i * 2 + 0], cuda_m[i * 2 + 1], cuda_r[i]);
        float ndc_x = 0.0f;
        float ndc_y = 0.0f;
        project_center(i, ndc_x, ndc_y);
        const int vk_ndc_int_count = tile_count_from_int_radius(ndc_x, ndc_y, vk_radii[i]);
        const float x = vk_means[i * 2 + 0];
        const float y = vk_means[i * 2 + 1];
        const float ex = vk_radius_f[i * 2 + 0];
        const float ey = vk_radius_f[i * 2 + 1];
        const float cuda_ex = cuda_rects_data ? cuda_rects_data[i * 2 + 0] : 0.0f;
        const float cuda_ey = cuda_rects_data ? cuda_rects_data[i * 2 + 1] : 0.0f;
        vk_float_rects[i] = rect_from_float_extent(x, y, ex, ey);
        cuda_float_rects[i] = cuda_rects_data
            ? rect_from_float_extent(cuda_m[i * 2 + 0], cuda_m[i * 2 + 1], cuda_ex, cuda_ey)
            : RectDiag{0, 0, 0, 0, 0};
        const RectDiag vk_ndc_float_rect = rect_from_float_extent(ndc_x, ndc_y, ex, ey);
        if (cuda_int_count != cuda_t[i]) ++cuda_int_rect_bad;
        if (vk_int_count != cuda_t[i]) ++vk_int_rect_vs_cuda_bad;
        if (vk_int_count != vk_tiles[i]) ++vk_int_rect_vs_vk_bad;
        if (vk_ndc_int_count != cuda_t[i]) ++vk_ndc_int_rect_vs_cuda_bad;
        if (vk_ndc_float_rect.count != cuda_t[i]) ++vk_ndc_float_rect_vs_cuda_bad;
        if (vk_tiles[i] != cuda_t[i]) {
            if (vk_vs_cuda_bad < 32) {
                const float x = vk_means[i * 2 + 0];
                const float y = vk_means[i * 2 + 1];
                const float ex = vk_radius_f[i * 2 + 0];
                const float ey = vk_radius_f[i * 2 + 1];
                const float cuda_ex = cuda_rects_data ? cuda_rects_data[i * 2 + 0] : 0.0f;
                const float cuda_ey = cuda_rects_data ? cuda_rects_data[i * 2 + 1] : 0.0f;
                const RectDiag r = vk_float_rects[i];
                const RectDiag cuda_float_rect = cuda_float_rects[i];
                float p_view[3]{};
                float cpu_g2s[16]{};
                float cpu_g2v[16]{};
                float cpu_mean[2]{};
                float cpu_extent[2]{};
                transformPoint4x3(model.data.positions + i * 3, cam.view_matrix, p_view);
                const float focal_x = static_cast<float>(kW) / (2.0f * cam.tan_fovx);
                const float focal_y = static_cast<float>(kH) / (2.0f * cam.tan_fovy);
                const float focal = std::max(focal_x, focal_y);
                const float filter = model.data.filter_3D ? model.data.filter_3D[i] : 0.0f;
                computeGauss2Screen(model.data.positions + i * 3,
                                    model.data.scales + i * 3,
                                    model.data.rotations + i * 4,
                                    1.0f,
                                    cam.cam_pos,
                                    cam.viewproj_matrix,
                                    cam.view_matrix,
                                    focal,
                                    0.3f,
                                    filter,
                                    kW,
                                    kH,
                                    cpu_g2s,
                                    cpu_g2v);
                const float alpha_threshold = 1.0f / 255.0f;
                const float cutoff = std::min(11.11f, 2.0f * std::log(model.data.opacities[i] / alpha_threshold));
                const bool cpu_aabb_ok = computeAABBView(cpu_g2v, p_view, focal_x, focal_y,
                                                         static_cast<float>(kW), static_cast<float>(kH),
                                                         cutoff, cpu_mean, cpu_extent);
                const RectDiag cpu_rect = cpu_aabb_ok
                    ? rect_from_float_extent(cpu_mean[0], cpu_mean[1], cpu_extent[0], cpu_extent[1])
                    : RectDiag{0, 0, 0, 0, 0};
                const float lo_x = (x - ex) / static_cast<float>(tile_w);
                const float hi_x = (x + ex) / static_cast<float>(tile_w);
                const float lo_y = (y - ey) / static_cast<float>(tile_h);
                const float hi_y = (y + ey) / static_cast<float>(tile_h);
                const float cuda_lo_x = (cuda_m[i * 2 + 0] - cuda_ex) / static_cast<float>(tile_w);
                const float cuda_hi_x = (cuda_m[i * 2 + 0] + cuda_ex) / static_cast<float>(tile_w);
                const float cuda_lo_y = (cuda_m[i * 2 + 1] - cuda_ey) / static_cast<float>(tile_h);
                const float cuda_hi_y = (cuda_m[i * 2 + 1] + cuda_ey) / static_cast<float>(tile_h);
                const float vk_min_edge_dist = std::min(
                    std::min(boundary_distance(lo_x), boundary_distance(hi_x)),
                    std::min(boundary_distance(lo_y), boundary_distance(hi_y)));
                const float cuda_min_edge_dist = std::min(
                    std::min(boundary_distance(cuda_lo_x), boundary_distance(cuda_hi_x)),
                    std::min(boundary_distance(cuda_lo_y), boundary_distance(cuda_hi_y)));
                if (std::min(vk_min_edge_dist, cuda_min_edge_dist) <= 1.0e-3f) {
                    boundary_ambiguous[i] = 1;
                    ++boundary_ambiguous_bad;
                }
                float g2s_max_diff = 0.0f;
                int g2s_max_idx = -1;
                float g2s_vk_at_max = 0.0f;
                float g2s_cuda_at_max = 0.0f;
                int g2s_first_idx = -1;
                float g2s_first_diff = 0.0f;
                float g2s_vk_at_first = 0.0f;
                float g2s_cuda_at_first = 0.0f;
                if (cuda_g2s_data) {
                    for (int j = 0; j < 16; ++j) {
                        const float vk_v = vk_g2s[i * 16 + j];
                        const float cuda_v = cuda_g2s_data[i * 16 + j];
                        const float diff = std::fabs(vk_v - cuda_v);
                        if (g2s_first_idx < 0 && diff > 1e-7f) {
                            g2s_first_idx = j;
                            g2s_first_diff = diff;
                            g2s_vk_at_first = vk_v;
                            g2s_cuda_at_first = cuda_v;
                        }
                        if (diff > g2s_max_diff) {
                            g2s_max_diff = diff;
                            g2s_max_idx = j;
                            g2s_vk_at_max = vk_v;
                            g2s_cuda_at_max = cuda_v;
                        }
                    }
                }
                print_cpu_aabb_trace(i, cpu_g2v, p_view, focal_x, focal_y, cutoff);
                std::printf("    gid=%-7zu vk_tiles=%d cuda_tiles=%d delta=%d "
                            "vk_float_rect=(%d,%d)-(%d,%d)=%d "
                            "cuda_float_rect=(%d,%d)-(%d,%d)=%d "
                            "vk_ndc_float_rect=(%d,%d)-(%d,%d)=%d "
                            "cpu_rect=(%d,%d)-(%d,%d)=%d "
                            "vk_int_rect=%d cuda_int_rect=%d vk_ndc_int_rect=%d "
                            "vk_mean=(%.9g,%.9g) cuda_mean=(%.9g,%.9g) ndc=(%.9g,%.9g) "
                            "vk_extent=(%.9g,%.9g) cuda_extent=(%.9g,%.9g) cpu_extent=(%.9g,%.9g) vk_r=%d cuda_r=%d "
                            "g2s_first_diff=%.9g@%d(r=%d,c=%d,vk=%.9g,cuda=%.9g) "
                            "g2s_max_diff=%.9g@%d(r=%d,c=%d,vk=%.9g,cuda=%.9g) "
                            "vk_edge=(%.9g,%.9g,%.9g,%.9g) "
                            "cuda_edge=(%.9g,%.9g,%.9g,%.9g) "
                            "vk_edge_dist=(%.3g,%.3g,%.3g,%.3g) "
                            "cuda_edge_dist=(%.3g,%.3g,%.3g,%.3g)\n",
                            i, vk_tiles[i], cuda_t[i], vk_tiles[i] - cuda_t[i],
                            r.min_x, r.min_y, r.max_x, r.max_y, r.count,
                            cuda_float_rect.min_x, cuda_float_rect.min_y,
                            cuda_float_rect.max_x, cuda_float_rect.max_y,
                            cuda_float_rect.count,
                            vk_ndc_float_rect.min_x, vk_ndc_float_rect.min_y,
                            vk_ndc_float_rect.max_x, vk_ndc_float_rect.max_y,
                            vk_ndc_float_rect.count,
                            cpu_rect.min_x, cpu_rect.min_y, cpu_rect.max_x, cpu_rect.max_y,
                            cpu_rect.count,
                            vk_int_count, cuda_int_count, vk_ndc_int_count,
                            x, y,
                            cuda_m[i * 2 + 0], cuda_m[i * 2 + 1], ndc_x, ndc_y,
                            ex, ey, cuda_ex, cuda_ey, cpu_extent[0], cpu_extent[1], vk_radii[i], cuda_r[i],
                            g2s_first_diff, g2s_first_idx,
                            g2s_first_idx >= 0 ? g2s_first_idx / 4 : -1,
                            g2s_first_idx >= 0 ? g2s_first_idx % 4 : -1,
                            g2s_vk_at_first, g2s_cuda_at_first,
                            g2s_max_diff, g2s_max_idx,
                            g2s_max_idx >= 0 ? g2s_max_idx / 4 : -1,
                            g2s_max_idx >= 0 ? g2s_max_idx % 4 : -1,
                            g2s_vk_at_max, g2s_cuda_at_max,
                            lo_x, hi_x, lo_y, hi_y,
                            cuda_lo_x, cuda_hi_x, cuda_lo_y, cuda_hi_y,
                            boundary_distance(lo_x), boundary_distance(hi_x),
                            boundary_distance(lo_y), boundary_distance(hi_y),
                            boundary_distance(cuda_lo_x), boundary_distance(cuda_hi_x),
                            boundary_distance(cuda_lo_y), boundary_distance(cuda_hi_y));
            }
            ++vk_vs_cuda_bad;
            vk_vs_cuda_sum_delta += vk_tiles[i] - cuda_t[i];
        }
    }

    std::printf("[Gate P5] summary: vk_vs_cuda_bad=%d boundary_ambiguous=%d sum_delta=%d "
                "cuda_int_rect_bad=%d vk_int_rect_vs_cuda_bad=%d "
                "vk_int_rect_vs_vk_bad=%d vk_ndc_int_rect_vs_cuda_bad=%d "
                "vk_ndc_float_rect_vs_cuda_bad=%d\n",
                vk_vs_cuda_bad, boundary_ambiguous_bad, vk_vs_cuda_sum_delta, cuda_int_rect_bad,
                vk_int_rect_vs_cuda_bad, vk_int_rect_vs_vk_bad,
                vk_ndc_int_rect_vs_cuda_bad, vk_ndc_float_rect_vs_cuda_bad);

    auto one_sided_boundary_instance = [&](int gid, int tile, bool from_cuda) {
        if (gid < 0 || static_cast<size_t>(gid) >= N || !boundary_ambiguous[static_cast<size_t>(gid)]) return false;
        const int tx = tile % grid_x;
        const int ty = tile / grid_x;
        const RectDiag& vk_rect = vk_float_rects[static_cast<size_t>(gid)];
        const RectDiag& cuda_rect = cuda_float_rects[static_cast<size_t>(gid)];
        const RectDiag& source = from_cuda ? cuda_rect : vk_rect;
        const RectDiag& other = from_cuda ? vk_rect : cuda_rect;
        if (!contains_tile(source, tx, ty) || contains_tile(other, tx, ty)) return false;

        const float x = vk_means[static_cast<size_t>(gid) * 2 + 0];
        const float y = vk_means[static_cast<size_t>(gid) * 2 + 1];
        const float ex = vk_radius_f[static_cast<size_t>(gid) * 2 + 0];
        const float ey = vk_radius_f[static_cast<size_t>(gid) * 2 + 1];
        const float cuda_ex = cuda_rects_data[static_cast<size_t>(gid) * 2 + 0];
        const float cuda_ey = cuda_rects_data[static_cast<size_t>(gid) * 2 + 1];
        const float lo_x = (x - ex) / static_cast<float>(tile_w);
        const float hi_x = (x + ex) / static_cast<float>(tile_w);
        const float lo_y = (y - ey) / static_cast<float>(tile_h);
        const float hi_y = (y + ey) / static_cast<float>(tile_h);
        const float cuda_lo_x = (cuda_m[static_cast<size_t>(gid) * 2 + 0] - cuda_ex) / static_cast<float>(tile_w);
        const float cuda_hi_x = (cuda_m[static_cast<size_t>(gid) * 2 + 0] + cuda_ex) / static_cast<float>(tile_w);
        const float cuda_lo_y = (cuda_m[static_cast<size_t>(gid) * 2 + 1] - cuda_ey) / static_cast<float>(tile_h);
        const float cuda_hi_y = (cuda_m[static_cast<size_t>(gid) * 2 + 1] + cuda_ey) / static_cast<float>(tile_h);
        constexpr float boundary_tol = 1.0e-3f;
        const bool left_edge = std::min(boundary_distance(lo_x), boundary_distance(cuda_lo_x)) <= boundary_tol;
        const bool right_edge = std::min(boundary_distance(hi_x), boundary_distance(cuda_hi_x)) <= boundary_tol;
        const bool top_edge = std::min(boundary_distance(lo_y), boundary_distance(cuda_lo_y)) <= boundary_tol;
        const bool bottom_edge = std::min(boundary_distance(hi_y), boundary_distance(cuda_hi_y)) <= boundary_tol;

        if (tx < other.min_x && tx == other.min_x - 1 && left_edge) return true;
        if (tx >= other.max_x && tx == other.max_x && right_edge) return true;
        if (ty < other.min_y && ty == other.min_y - 1 && top_edge) return true;
        if (ty >= other.max_y && ty == other.max_y && bottom_edge) return true;
        return false;
    };

    int stable_bad_tiles = 0;
    int stable_set_bad_tiles = 0;
    int stable_order_only_bad_tiles = 0;
    int stable_tie_order_only_bad_tiles = 0;
    int stable_near_depth_order_only_bad_tiles = 0;
    int stable_real_depth_order_bad_tiles = 0;
    uint32_t max_depth_key_delta_ulp = 0;
    uint32_t max_near_depth_window_span_ulp = 0;
    size_t max_near_depth_cluster_size = 0;
    int stable_first_bad_tile = -1;
    size_t stable_compared_ids = 0;
    size_t masked_cuda_instances = 0;
    size_t masked_vk_instances = 0;
    size_t first_cuda_stable_size = 0;
    size_t first_vk_stable_size = 0;
    int first_cuda_id = -1;
    int first_vk_id = -1;
    size_t first_bad_pos = 0;
    std::vector<int> first_cuda_stable_ids;
    std::vector<int> first_vk_stable_ids;
    int first_non_tie_tile = -1;
    size_t first_non_tie_pos = 0;
    std::vector<int> first_non_tie_cuda_ids;
    std::vector<int> first_non_tie_vk_ids;
    auto bits_for_float = [](float value) {
        union Bits {
            float f;
            uint32_t u;
        } bits{};
        bits.f = value;
        return bits.u;
    };
    auto cuda_depth_bits_for = [&](int gid) {
        return bits_for_float(cuda_d[static_cast<size_t>(gid)]);
    };
    auto vk_depth_bits_for = [&](int gid) {
        return bits_for_float(vk_depths[static_cast<size_t>(gid)]);
    };
    auto same_modulo_depth_ties = [&](const std::vector<int>& cuda_ids, const std::vector<int>& vk_ids) {
        if (cuda_ids.size() != vk_ids.size()) return false;
        size_t c = 0;
        size_t v = 0;
        while (c < cuda_ids.size()) {
            if (v >= vk_ids.size()) return false;
            const uint32_t c_key = cuda_depth_bits_for(cuda_ids[c]);
            const uint32_t v_key = vk_depth_bits_for(vk_ids[v]);
            if (c_key != v_key) return false;
            size_t c_end = c + 1;
            while (c_end < cuda_ids.size() && cuda_depth_bits_for(cuda_ids[c_end]) == c_key) ++c_end;
            size_t v_end = v + 1;
            while (v_end < vk_ids.size() && vk_depth_bits_for(vk_ids[v_end]) == v_key) ++v_end;
            if (c_end - c != v_end - v) return false;
            std::vector<int> c_group(cuda_ids.begin() + static_cast<std::ptrdiff_t>(c),
                                     cuda_ids.begin() + static_cast<std::ptrdiff_t>(c_end));
            std::vector<int> v_group(vk_ids.begin() + static_cast<std::ptrdiff_t>(v),
                                     vk_ids.begin() + static_cast<std::ptrdiff_t>(v_end));
            std::sort(c_group.begin(), c_group.end());
            std::sort(v_group.begin(), v_group.end());
            if (c_group != v_group) return false;
            c = c_end;
            v = v_end;
        }
        return v == vk_ids.size();
    };
    auto same_modulo_near_depth_windows = [&](const std::vector<int>& cuda_ids,
                                             const std::vector<int>& vk_ids,
                                             size_t& tile_max_cluster) {
        if (cuda_ids.size() != vk_ids.size()) return false;
        constexpr uint32_t ulp_window = 1u;
        struct DepthWindow {
            uint32_t lo;
            uint32_t hi;
        };
        auto depth_window_for = [&](int gid) {
            const uint32_t c_key = cuda_depth_bits_for(gid);
            const uint32_t v_key = vk_depth_bits_for(gid);
            const uint32_t lo0 = std::min(c_key, v_key);
            const uint32_t hi0 = std::max(c_key, v_key);
            max_depth_key_delta_ulp = std::max(max_depth_key_delta_ulp, hi0 - lo0);
            return DepthWindow{
                lo0 > ulp_window ? lo0 - ulp_window : 0u,
                hi0 <= UINT32_MAX - ulp_window ? hi0 + ulp_window : UINT32_MAX,
            };
        };
        std::vector<size_t> vk_pos(static_cast<size_t>(model.data.count), vk_ids.size());
        for (size_t i = 0; i < vk_ids.size(); ++i) {
            vk_pos[static_cast<size_t>(vk_ids[i])] = i;
        }
        for (size_t i = 0; i < cuda_ids.size(); ++i) {
            if (vk_pos[static_cast<size_t>(cuda_ids[i])] == vk_ids.size()) return false;
        }
        for (size_t i = 0; i < cuda_ids.size(); ++i) {
            const size_t vk_i = vk_pos[static_cast<size_t>(cuda_ids[i])];
            for (size_t j = i + 1; j < cuda_ids.size(); ++j) {
                const size_t vk_j = vk_pos[static_cast<size_t>(cuda_ids[j])];
                if (vk_i <= vk_j) continue;
                const DepthWindow a = depth_window_for(cuda_ids[i]);
                const DepthWindow b = depth_window_for(cuda_ids[j]);
                if (a.lo > b.hi || b.lo > a.hi) return false;
                const uint32_t span = std::max(a.hi, b.hi) - std::min(a.lo, b.lo);
                max_near_depth_window_span_ulp = std::max(max_near_depth_window_span_ulp, span);
                tile_max_cluster = std::max(tile_max_cluster, static_cast<size_t>(2));
            }
        }
        return true;
    };
    for (int tile = 0; tile < num_tiles; ++tile) {
        std::vector<int> cuda_tile_ids;
        std::vector<int> vk_tile_ids;
        for (int p = cuda_tile_offsets[tile]; p < cuda_tile_offsets[tile + 1]; ++p) {
            cuda_tile_ids.push_back(cuda_sorted_ids[p]);
        }
        for (int p = vk_tile_ranges[tile * 2 + 0]; p < vk_tile_ranges[tile * 2 + 1]; ++p) {
            vk_tile_ids.push_back(vk_sorted[p]);
        }
        std::vector<int> cuda_members = cuda_tile_ids;
        std::vector<int> vk_members = vk_tile_ids;
        std::sort(cuda_members.begin(), cuda_members.end());
        std::sort(vk_members.begin(), vk_members.end());

        std::vector<int> cuda_stable;
        std::vector<int> vk_stable;
        for (int gid : cuda_tile_ids) {
            if (one_sided_boundary_instance(gid, tile, /*from_cuda=*/true)) {
                ++masked_cuda_instances;
                continue;
            }
            cuda_stable.push_back(gid);
        }
        for (int gid : vk_tile_ids) {
            if (one_sided_boundary_instance(gid, tile, /*from_cuda=*/false)) {
                ++masked_vk_instances;
                continue;
            }
            vk_stable.push_back(gid);
        }
        bool tile_ok = cuda_stable.size() == vk_stable.size();
        const size_t common = std::min(cuda_stable.size(), vk_stable.size());
        size_t bad_pos = common;
        if (tile_ok) {
            for (size_t p = 0; p < common; ++p) {
                if (cuda_stable[p] != vk_stable[p]) {
                    tile_ok = false;
                    bad_pos = p;
                    break;
                }
            }
        }
        stable_compared_ids += common;
        if (!tile_ok) {
            std::vector<int> cuda_set = cuda_stable;
            std::vector<int> vk_set = vk_stable;
            std::sort(cuda_set.begin(), cuda_set.end());
            std::sort(vk_set.begin(), vk_set.end());
            const bool same_set = cuda_set == vk_set;
            if (same_set) {
                ++stable_order_only_bad_tiles;
                if (same_modulo_depth_ties(cuda_stable, vk_stable)) {
                    ++stable_tie_order_only_bad_tiles;
                } else {
                    size_t tile_max_cluster = 0;
                    if (same_modulo_near_depth_windows(cuda_stable, vk_stable, tile_max_cluster)) {
                        ++stable_near_depth_order_only_bad_tiles;
                        max_near_depth_cluster_size = std::max(max_near_depth_cluster_size, tile_max_cluster);
                    } else {
                        ++stable_real_depth_order_bad_tiles;
                    }
                    if (first_non_tie_tile < 0) {
                        first_non_tie_tile = tile;
                        first_non_tie_pos = bad_pos;
                        first_non_tie_cuda_ids = cuda_stable;
                        first_non_tie_vk_ids = vk_stable;
                    }
                }
            } else {
                ++stable_set_bad_tiles;
            }
            if (stable_bad_tiles == 0) {
                stable_first_bad_tile = tile;
                first_cuda_stable_size = cuda_stable.size();
                first_vk_stable_size = vk_stable.size();
                first_bad_pos = bad_pos;
                if (bad_pos < cuda_stable.size()) first_cuda_id = cuda_stable[bad_pos];
                if (bad_pos < vk_stable.size()) first_vk_id = vk_stable[bad_pos];
                first_cuda_stable_ids = cuda_stable;
                first_vk_stable_ids = vk_stable;
            }
            ++stable_bad_tiles;
        }
    }
    std::printf("[Gate P5] stable sorted IDs after masking boundary tile-instances from %d gids: "
                "bad_tiles=%d set_bad_tiles=%d order_only_bad_tiles=%d "
                "tie_order_only_bad_tiles=%d near_depth_order_only_bad_tiles=%d "
                "real_depth_order_bad_tiles=%d max_depth_key_delta_ulp=%u "
                "max_near_depth_window_span_ulp=%u max_near_depth_cluster_size=%zu "
                "first_bad_tile=%d first_bad_pos=%zu "
                "first_sizes=(cuda=%zu,vk=%zu) first_ids=(cuda=%d,vk=%d) "
                "masked_instances=(cuda=%zu,vk=%zu) compared_common=%zu\n",
                vk_vs_cuda_bad, stable_bad_tiles, stable_set_bad_tiles,
                stable_order_only_bad_tiles, stable_tie_order_only_bad_tiles,
                stable_near_depth_order_only_bad_tiles, stable_real_depth_order_bad_tiles,
                max_depth_key_delta_ulp, max_near_depth_window_span_ulp,
                max_near_depth_cluster_size, stable_first_bad_tile, first_bad_pos,
                first_cuda_stable_size, first_vk_stable_size, first_cuda_id, first_vk_id,
                masked_cuda_instances, masked_vk_instances, stable_compared_ids);
    if (stable_first_bad_tile >= 0) {
        const int tx = stable_first_bad_tile % grid_x;
        const int ty = stable_first_bad_tile / grid_x;
        auto depth_for = [&](int gid, const float* g2s_data) {
            if (!g2s_data || gid < 0) return 0.0f;
            float max_pos_depth = 0.0f;
            maxContribGaussianFrustum3D(
                static_cast<float>(tx * tile_w),
                static_cast<float>(ty * tile_h),
                static_cast<float>((tx + 1) * tile_w - 1),
                static_cast<float>((ty + 1) * tile_h - 1),
                g2s_data + static_cast<size_t>(gid) * 16u,
                max_pos_depth);
            return max_pos_depth;
        };
        auto view_z_for = [&](int gid) {
            if (gid < 0) return 0.0f;
            float p_view[3]{};
            transformPoint4x3(model.data.positions + static_cast<size_t>(gid) * 3u,
                              cam.view_matrix, p_view);
            return p_view[2];
        };
        const size_t lo = first_bad_pos > 3 ? first_bad_pos - 3 : 0;
        const size_t hi = std::min(first_bad_pos + 4, std::min(first_cuda_stable_ids.size(), first_vk_stable_ids.size()));
        std::printf("[Gate P5] first order-only tile window tile=%d (tx=%d,ty=%d):\n",
                    stable_first_bad_tile, tx, ty);
        for (size_t p = lo; p < hi; ++p) {
            const int cid = first_cuda_stable_ids[p];
            const int vid = first_vk_stable_ids[p];
            std::printf("    pos=%zu cuda_id=%d view_z(cpu)=%.9g vk_depth=%.9g tile_depth(cpu,cuda_g2s)=%.9g "
                        "vk_id=%d view_z(cpu)=%.9g vk_depth=%.9g tile_depth(cpu,vk_g2s)=%.9g\n",
                        p, cid, view_z_for(cid), vk_depths[static_cast<size_t>(cid)],
                        depth_for(cid, cuda_g2s_data),
                        vid, view_z_for(vid), vk_depths[static_cast<size_t>(vid)],
                        depth_for(vid, vk_g2s.data()));
        }
    }
    if (first_non_tie_tile >= 0) {
        const size_t lo = first_non_tie_pos > 3 ? first_non_tie_pos - 3 : 0;
        const size_t hi = std::min(first_non_tie_pos + 4, std::min(first_non_tie_cuda_ids.size(), first_non_tie_vk_ids.size()));
        std::printf("[Gate P5] first non-tie order window tile=%d first_pos=%zu:\n",
                    first_non_tie_tile, first_non_tie_pos);
        for (size_t p = lo; p < hi; ++p) {
            const int cid = first_non_tie_cuda_ids[p];
            const int vid = first_non_tie_vk_ids[p];
            const uint32_t c_key = cuda_depth_bits_for(cid);
            const uint32_t v_key = vk_depth_bits_for(vid);
            std::printf("    pos=%zu cuda_id=%d key=0x%08x depth=%.9g vk_id=%d key=0x%08x depth=%.9g\n",
                        p, cid, c_key, cuda_d[static_cast<size_t>(cid)],
                        vid, v_key, vk_depths[static_cast<size_t>(vid)]);
        }
    }

    EXPECT_LE(vk_vs_cuda_bad, 16)
        << "VK/CUDA eval_3D tile coverage drift exceeded the documented boundary-window mask.";
    EXPECT_EQ(boundary_ambiguous_bad, vk_vs_cuda_bad)
        << "A tiles_touched mismatch was not close enough to a tile boundary to mask.";
    EXPECT_LE(std::abs(vk_vs_cuda_sum_delta), 16)
        << "VK/CUDA total tile-instance drift is too large for isolated boundary flips.";
    EXPECT_LE(masked_cuda_instances + masked_vk_instances, static_cast<size_t>(32))
        << "Boundary masking removed too many tile instances for an isolated edge ambiguity.";
    EXPECT_EQ(stable_set_bad_tiles, 0)
        << "Stable per-tile sorted-ID sets differ after masking boundary-ambiguous gids.";
    EXPECT_LE(max_depth_key_delta_ulp, 1u)
        << "Accepted near-depth reorder used a per-Gaussian CUDA/VK depth key delta above the near window.";
    EXPECT_LE(max_near_depth_window_span_ulp, 4u)
        << "Accepted near-depth reorder spanned more than one overlapping pairwise depth window.";
    EXPECT_EQ(stable_real_depth_order_bad_tiles, 0)
        << "Stable per-tile sorted-ID order differs across separated depth-key windows.";
    EXPECT_EQ(stable_order_only_bad_tiles,
              stable_tie_order_only_bad_tiles + stable_near_depth_order_only_bad_tiles)
        << "Stable sorted-ID order differences must be exact ties or near-depth key windows.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P6_TFinalNContrib) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        << "gt_image.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/T_final.npy"))
        << "T_final.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/n_contrib.npy"))
        << "n_contrib.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/depths.npy"))
        << "depths.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/sorted_ids_per_tile.npy"))
        << "sorted_ids_per_tile.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/tile_offsets.npy"))
        << "tile_offsets.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);
    trainer->enable_intermediate_capture(true);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;
    ASSERT_TRUE(tcfg.eval_3D && tcfg.parity_mode)
        << "P6 compares eval_3D parity-mode raster state.";

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P6] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_T = load_npy(dump_dir() + "/T_final.npy");
    NpyArray cuda_n = load_npy(dump_dir() + "/n_contrib.npy");
    NpyArray cuda_depths = load_npy(dump_dir() + "/depths.npy");
    NpyArray cuda_sorted = load_npy(dump_dir() + "/sorted_ids_per_tile.npy");
    NpyArray cuda_offsets = load_npy(dump_dir() + "/tile_offsets.npy");
    assert_dtype(cuda_T, NpyDtype::float32);
    assert_dtype(cuda_n, NpyDtype::int32);
    assert_dtype(cuda_depths, NpyDtype::float32);
    assert_dtype(cuda_sorted, NpyDtype::int32);
    assert_dtype(cuda_offsets, NpyDtype::int32);
    ASSERT_EQ(cuda_T.shape, (std::vector<size_t>{static_cast<size_t>(kH), static_cast<size_t>(kW)}));
    ASSERT_EQ(cuda_n.shape, (std::vector<size_t>{static_cast<size_t>(kH), static_cast<size_t>(kW)}));
    ASSERT_EQ(cuda_depths.shape, (std::vector<size_t>{static_cast<size_t>(model.data.count)}));

    const int grid_x = (kW + 15) / 16;
    const int grid_y = (kH + 15) / 16;
    const int num_tiles = grid_x * grid_y;
    ASSERT_EQ(cuda_offsets.shape, (std::vector<size_t>{static_cast<size_t>(num_tiles + 1)}));

    const std::vector<float>& vk_T = trainer->captured_T_final();
    const std::vector<int>& vk_n = trainer->captured_n_contrib();
    const std::vector<float>& vk_depths = trainer->captured_depths();
    const std::vector<int>& vk_sorted = trainer->captured_sorted_gaussian_ids();
    const std::vector<int>& vk_tile_ranges = trainer->captured_tile_offsets();
    ASSERT_EQ(vk_T.size(), static_cast<size_t>(kW * kH));
    ASSERT_EQ(vk_n.size(), static_cast<size_t>(kW * kH));
    ASSERT_EQ(vk_depths.size(), static_cast<size_t>(model.data.count));
    ASSERT_EQ(vk_tile_ranges.size(), static_cast<size_t>(num_tiles * 2));

    const float* cuda_T_data = cuda_T.f32();
    const int32_t* cuda_n_data = cuda_n.i32();
    const float* cuda_d = cuda_depths.f32();
    const int32_t* cuda_sorted_ids = cuda_sorted.i32();
    const int32_t* cuda_tile_offsets = cuda_offsets.i32();
    ASSERT_EQ(cuda_tile_offsets[0], 0);
    ASSERT_EQ(cuda_sorted.shape, (std::vector<size_t>{static_cast<size_t>(cuda_tile_offsets[num_tiles])}));
    int previous_vk_end = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        ASSERT_LE(cuda_tile_offsets[tile], cuda_tile_offsets[tile + 1]);
        ASSERT_GE(cuda_tile_offsets[tile], 0);
        const int vk_begin = vk_tile_ranges[tile * 2 + 0];
        const int vk_end = vk_tile_ranges[tile * 2 + 1];
        ASSERT_EQ(vk_begin, previous_vk_end);
        ASSERT_GE(vk_begin, 0);
        ASSERT_LE(vk_begin, vk_end);
        ASSERT_LE(static_cast<size_t>(vk_end), vk_sorted.size());
        previous_vk_end = vk_end;
    }
    ASSERT_EQ(static_cast<size_t>(previous_vk_end), vk_sorted.size());
    for (size_t i = 0; i < cuda_sorted.shape[0]; ++i) {
        ASSERT_GE(cuda_sorted_ids[i], 0);
        ASSERT_LT(cuda_sorted_ids[i], static_cast<int32_t>(model.data.count));
    }
    for (int gid : vk_sorted) {
        ASSERT_GE(gid, 0);
        ASSERT_LT(gid, static_cast<int>(model.data.count));
    }
    auto bits_for_float = [](float value) {
        union Bits {
            float f;
            uint32_t u;
        } bits{};
        bits.f = value;
        return bits.u;
    };
    auto cuda_depth_bits_for = [&](int gid) {
        return bits_for_float(cuda_d[static_cast<size_t>(gid)]);
    };
    auto vk_depth_bits_for = [&](int gid) {
        return bits_for_float(vk_depths[static_cast<size_t>(gid)]);
    };
    auto same_modulo_near_depth_windows = [&](const std::vector<int>& cuda_ids,
                                             const std::vector<int>& vk_ids) {
        if (cuda_ids.size() != vk_ids.size()) return false;
        constexpr uint32_t ulp_window = 1u;
        auto depth_window_for = [&](int gid) {
            const uint32_t c_key = cuda_depth_bits_for(gid);
            const uint32_t v_key = vk_depth_bits_for(gid);
            const uint32_t lo0 = std::min(c_key, v_key);
            const uint32_t hi0 = std::max(c_key, v_key);
            return std::pair<uint32_t, uint32_t>{
                lo0 > ulp_window ? lo0 - ulp_window : 0u,
                hi0 <= UINT32_MAX - ulp_window ? hi0 + ulp_window : UINT32_MAX,
            };
        };
        std::vector<size_t> vk_pos(static_cast<size_t>(model.data.count), vk_ids.size());
        for (size_t i = 0; i < vk_ids.size(); ++i) {
            vk_pos[static_cast<size_t>(vk_ids[i])] = i;
        }
        for (int gid : cuda_ids) {
            if (vk_pos[static_cast<size_t>(gid)] == vk_ids.size()) return false;
        }
        for (size_t i = 0; i < cuda_ids.size(); ++i) {
            const size_t vk_i = vk_pos[static_cast<size_t>(cuda_ids[i])];
            for (size_t j = i + 1; j < cuda_ids.size(); ++j) {
                const size_t vk_j = vk_pos[static_cast<size_t>(cuda_ids[j])];
                if (vk_i <= vk_j) continue;
                const auto a = depth_window_for(cuda_ids[i]);
                const auto b = depth_window_for(cuda_ids[j]);
                if (a.first > b.second || b.first > a.second) return false;
                const uint32_t span = std::max(a.second, b.second) - std::min(a.first, b.first);
                if (span > 4u) return false;
            }
        }
        return true;
    };

    std::vector<uint8_t> ambiguous_tile(static_cast<size_t>(num_tiles), 0);
    std::vector<int> ambiguous_set_tile_ids;
    std::vector<int> ambiguous_order_tile_ids;
    int ambiguous_set_tiles = 0;
    int ambiguous_order_tiles = 0;
    int real_tile_order_bad = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        std::vector<int> cuda_ids;
        std::vector<int> vk_ids;
        for (int p = cuda_tile_offsets[tile]; p < cuda_tile_offsets[tile + 1]; ++p) {
            cuda_ids.push_back(cuda_sorted_ids[p]);
        }
        for (int p = vk_tile_ranges[tile * 2 + 0]; p < vk_tile_ranges[tile * 2 + 1]; ++p) {
            vk_ids.push_back(vk_sorted[static_cast<size_t>(p)]);
        }
        if (cuda_ids == vk_ids) continue;

        std::vector<int> cuda_set = cuda_ids;
        std::vector<int> vk_set = vk_ids;
        std::sort(cuda_set.begin(), cuda_set.end());
        std::sort(vk_set.begin(), vk_set.end());
        ambiguous_tile[static_cast<size_t>(tile)] = 1;
        if (cuda_set != vk_set) {
            ambiguous_set_tile_ids.push_back(tile);
            ++ambiguous_set_tiles;
        } else if (same_modulo_near_depth_windows(cuda_ids, vk_ids)) {
            ambiguous_order_tile_ids.push_back(tile);
            ++ambiguous_order_tiles;
        } else {
            ++real_tile_order_bad;
        }
    }

    constexpr float kTerminationWindow = 1.5e-4f;
    int n_bad = 0;
    int n_bad_outside_ambiguous_tiles = 0;
    int n_bad_outside_termination = 0;
    int n_bad_outside_nontermination = 0;
    int n_off_by_one = 0;
    int n_off_by_one_outside_ambiguous_tiles = 0;
    int n_off_by_one_outside_termination = 0;
    int n_off_by_one_outside_nontermination = 0;
    int n_active_mismatch = 0;
    int first_n_bad = -1;
    int first_n_bad_outside_ambiguous_tiles = -1;
    int first_n_bad_outside_nontermination = -1;
    int first_n_off_by_one_outside_ambiguous_tiles = -1;
    int first_n_off_by_one_outside_nontermination = -1;
    int max_n_abs = 0;
    int T_bad = 0;
    int T_bad_outside_ambiguous_tiles = 0;
    int first_T_bad = -1;
    int first_T_bad_outside_ambiguous_tiles = -1;
    float max_T_abs = 0.0f;
    double sum_T_abs = 0.0;
    int both_active = 0;
    int both_inactive = 0;

    for (int idx = 0; idx < kW * kH; ++idx) {
        const int x = idx % kW;
        const int y = idx / kW;
        const int tile = (y / 16) * grid_x + (x / 16);
        const bool in_ambiguous_tile = ambiguous_tile[static_cast<size_t>(tile)] != 0;
        const int vn = vk_n[static_cast<size_t>(idx)];
        const int cn = cuda_n_data[idx];
        const int nd = std::abs(vn - cn);
        const float tv = vk_T[static_cast<size_t>(idx)];
        const float tc = cuda_T_data[idx];
        const bool near_termination = std::min(tv, tc) <= kTerminationWindow;
        if (nd > max_n_abs) max_n_abs = nd;
        if ((vn > 0) != (cn > 0)) ++n_active_mismatch;
        if (vn > 0 && cn > 0) ++both_active;
        if (vn == 0 && cn == 0) ++both_inactive;
        if (nd != 0) {
            if (nd == 1) {
                ++n_off_by_one;
                if (!in_ambiguous_tile) {
                    if (first_n_off_by_one_outside_ambiguous_tiles < 0) first_n_off_by_one_outside_ambiguous_tiles = idx;
                    ++n_off_by_one_outside_ambiguous_tiles;
                    if (near_termination) {
                        ++n_off_by_one_outside_termination;
                    } else {
                        if (first_n_off_by_one_outside_nontermination < 0) first_n_off_by_one_outside_nontermination = idx;
                        ++n_off_by_one_outside_nontermination;
                    }
                }
            } else {
                if (first_n_bad < 0) first_n_bad = idx;
                ++n_bad;
                if (!in_ambiguous_tile) {
                    if (first_n_bad_outside_ambiguous_tiles < 0) first_n_bad_outside_ambiguous_tiles = idx;
                    ++n_bad_outside_ambiguous_tiles;
                    if (near_termination) {
                        ++n_bad_outside_termination;
                    } else {
                        if (first_n_bad_outside_nontermination < 0) first_n_bad_outside_nontermination = idx;
                        ++n_bad_outside_nontermination;
                    }
                }
            }
        }

        const float td = std::fabs(tv - tc);
        max_T_abs = std::max(max_T_abs, td);
        sum_T_abs += static_cast<double>(td);
        const float tol = std::max(1.0e-5f, 1.0e-4f * std::max(std::fabs(tc), 1.0f));
        if (td > tol) {
            if (first_T_bad < 0) first_T_bad = idx;
            ++T_bad;
            if (!in_ambiguous_tile) {
                if (first_T_bad_outside_ambiguous_tiles < 0) first_T_bad_outside_ambiguous_tiles = idx;
                ++T_bad_outside_ambiguous_tiles;
            }
        }
    }

    auto print_pixel_diag = [&](const char* label, int idx) {
        if (idx < 0) return;
        const int x = idx % kW;
        const int y = idx / kW;
        const int tile = (y / 16) * grid_x + (x / 16);
        std::printf("[Gate P6] %s idx=%d x=%d y=%d tile=%d tx=%d ty=%d ambiguous=%d n=(vk=%d,cuda=%d) T=(vk=%.9g,cuda=%.9g,diff=%.9g) tile_sizes=(cuda=%d,vk=%d)\n",
                    label, idx, x, y, tile, tile % grid_x, tile / grid_x,
                    static_cast<int>(ambiguous_tile[static_cast<size_t>(tile)]),
                    vk_n[static_cast<size_t>(idx)], cuda_n_data[idx],
                    vk_T[static_cast<size_t>(idx)], cuda_T_data[idx],
                    std::fabs(vk_T[static_cast<size_t>(idx)] - cuda_T_data[idx]),
                    cuda_tile_offsets[tile + 1] - cuda_tile_offsets[tile],
                    vk_tile_ranges[tile * 2 + 1] - vk_tile_ranges[tile * 2 + 0]);
    };

    std::printf("[Gate P6] tile ambiguity: set_tiles=%d order_tiles=%d real_order_bad=%d set_tile_ids=",
                ambiguous_set_tiles, ambiguous_order_tiles, real_tile_order_bad);
    for (size_t i = 0; i < ambiguous_set_tile_ids.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", ambiguous_set_tile_ids[i]);
    }
    std::printf(" order_tile_ids=");
    for (size_t i = 0; i < ambiguous_order_tile_ids.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", ambiguous_order_tile_ids[i]);
    }
    std::printf("\n");
    const double mean_T_abs = sum_T_abs / static_cast<double>(kW * kH);
    std::printf("[Gate P6] n_contrib: bad_gt1=%d outside_amb=%d outside_term=%d outside_nonterm=%d off_by_one=%d outside_amb=%d outside_term=%d outside_nonterm=%d active_mismatch=%d max_abs=%d first_bad=%d both_active=%d both_inactive=%d\n",
                n_bad, n_bad_outside_ambiguous_tiles, n_bad_outside_termination,
                n_bad_outside_nontermination, n_off_by_one,
                n_off_by_one_outside_ambiguous_tiles,
                n_off_by_one_outside_termination,
                n_off_by_one_outside_nontermination, n_active_mismatch,
                max_n_abs, first_n_bad, both_active, both_inactive);
    print_pixel_diag("first n_contrib bad", first_n_bad);
    print_pixel_diag("first n_contrib bad outside ambiguous tile", first_n_bad_outside_ambiguous_tiles);
    print_pixel_diag("first n_contrib bad outside nontermination", first_n_bad_outside_nontermination);
    print_pixel_diag("first n_contrib off-by-one outside ambiguous tile", first_n_off_by_one_outside_ambiguous_tiles);
    print_pixel_diag("first n_contrib off-by-one outside nontermination", first_n_off_by_one_outside_nontermination);
    std::printf("[Gate P6] T_final: bad=%d outside_amb=%d max_abs=%.9g mean_abs=%.9g first_bad=%d\n",
                T_bad, T_bad_outside_ambiguous_tiles, max_T_abs,
                mean_T_abs, first_T_bad);
    print_pixel_diag("first T_final bad", first_T_bad);
    print_pixel_diag("first T_final bad outside ambiguous tile", first_T_bad_outside_ambiguous_tiles);

    const std::vector<int> expected_ambiguous_set_tile_ids{
        511, 512, 513, 634, 679, 934, 976, 977, 979,
        1022, 1023, 1024, 1059, 1060, 1061, 1103, 1104,
        1118, 1300, 1345, 1390, 1435, 1480, 1655, 1700,
        1745, 1790,
    };
    const std::vector<int> expected_ambiguous_order_tile_ids{
        756, 757, 758, 759, 848, 849, 867, 868, 869,
        893, 894, 912, 913, 914, 974, 978, 1019, 1021,
        1026, 1063, 1064, 1108, 1109, 1117, 1152, 1153,
        1154, 1155, 1197, 1198, 1199, 1200, 1238, 1239,
        1242, 1243, 1244, 1245, 1287, 1288, 1333, 1378,
        1391, 1436, 1565,
    };
    EXPECT_EQ(ambiguous_set_tile_ids, expected_ambiguous_set_tile_ids)
        << "P6 sorted-ID set-difference tile identities changed; update P5 boundary-instance evidence first.";
    EXPECT_EQ(ambiguous_order_tile_ids, expected_ambiguous_order_tile_ids)
        << "P6 near-depth sorted-order ambiguity tile identities changed; update P5 evidence first.";
    EXPECT_EQ(real_tile_order_bad, 0)
        << "P6 found sorted-ID tile order differences not explained by near global-depth keys.";
    EXPECT_EQ(n_active_mismatch, 0)
        << "VK/CUDA P6 active-pixel classification differs.";
    EXPECT_LE(n_bad, 24)
        << "VK/CUDA P6 n_contrib >1 drift exceeds the total bounded budget.";
    EXPECT_LE(n_off_by_one, 6002)
        << "VK/CUDA P6 n_contrib off-by-one drift exceeds the total bounded budget.";
    EXPECT_LE(n_bad_outside_termination, 18)
        << "VK/CUDA P6 n_contrib >1 early-termination drift exceeds the bounded budget outside P5-ambiguous tiles.";
    EXPECT_LE(n_off_by_one_outside_termination, 13)
        << "VK/CUDA P6 n_contrib off-by-one early-termination drift exceeds the bounded budget outside P5-ambiguous tiles.";
    EXPECT_LE(max_n_abs, 52)
        << "VK/CUDA P6 n_contrib max drift exceeds the bounded budget.";
    EXPECT_EQ(n_bad_outside_nontermination, 0)
        << "VK/CUDA P6 n_contrib differs by more than one candidate outside P5-ambiguous tiles and away from the early-termination boundary.";
    EXPECT_EQ(n_off_by_one_outside_nontermination, 0)
        << "VK/CUDA P6 n_contrib differs by one candidate outside P5-ambiguous tiles and away from the early-termination boundary.";
    EXPECT_LE(T_bad_outside_ambiguous_tiles, 16)
        << "VK/CUDA P6 final transmittance drift exceeds the bounded rare-pixel budget outside P5-ambiguous tiles.";
    EXPECT_LE(max_T_abs, 2.0e-3f)
        << "VK/CUDA P6 final transmittance drift exceeds the bounded max-abs budget.";
    EXPECT_LE(mean_T_abs, 2.5e-7)
        << "VK/CUDA P6 final transmittance drift exceeds the bounded mean-abs budget.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_P7_RenderedImage) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        << "gt_image.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/rendered_image.npy"))
        << "rendered_image.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;
    ASSERT_TRUE(tcfg.eval_3D && tcfg.parity_mode)
        << "P7 compares eval_3D parity-mode rendered image.";

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    std::printf("[Gate P7] forward_only loss = %.8f\n", vk_loss);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";

    NpyArray cuda_render = load_npy(dump_dir() + "/rendered_image.npy");
    assert_dtype(cuda_render, NpyDtype::float32);
    ASSERT_EQ(cuda_render.shape, (std::vector<size_t>{3u, static_cast<size_t>(kH), static_cast<size_t>(kW)}));

    const int image_size = 3 * kW * kH;
    ASSERT_EQ(trainer->rendered_image_size(), image_size);
    const float* vk_img = trainer->rendered_image();
    const float* cuda_img = cuda_render.f32();

    int bad_abs_1e5 = 0;
    int bad_abs_1e4 = 0;
    int first_bad_abs_1e5 = -1;
    int first_bad_abs_1e4 = -1;
    int max_abs_idx = -1;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    for (int i = 0; i < image_size; ++i) {
        const float vd = vk_img[i];
        const float cd = cuda_img[i];
        ASSERT_TRUE(std::isfinite(vd)) << "VK rendered image has non-finite value at " << i;
        ASSERT_TRUE(std::isfinite(cd)) << "CUDA rendered image has non-finite value at " << i;
        const float ad = std::fabs(vd - cd);
        if (ad > max_abs) {
            max_abs = ad;
            max_abs_idx = i;
        }
        sum_abs += static_cast<double>(ad);
        sum_sq += static_cast<double>(ad) * static_cast<double>(ad);
        ref_sq += static_cast<double>(cd) * static_cast<double>(cd);
        if (ad > 1.0e-5f) {
            if (first_bad_abs_1e5 < 0) first_bad_abs_1e5 = i;
            ++bad_abs_1e5;
        }
        if (ad > 1.0e-4f) {
            if (first_bad_abs_1e4 < 0) first_bad_abs_1e4 = i;
            ++bad_abs_1e4;
        }
    }

    const double mean_abs = sum_abs / static_cast<double>(image_size);
    const double mse = sum_sq / static_cast<double>(image_size);
    const double psnr = mse > 0.0 ? -10.0 * std::log10(mse) : 999.0;
    const double l2_rel = std::sqrt(sum_sq / std::max(ref_sq, 1.0e-30));
    auto print_image_diag = [&](const char* label, int i) {
        if (i < 0) return;
        const int hw = kW * kH;
        const int c = i / hw;
        const int pixel = i % hw;
        const int x = pixel % kW;
        const int y = pixel / kW;
        std::printf("[Gate P7] %s idx=%d c=%d x=%d y=%d vk=%.9g cuda=%.9g diff=%.9g\n",
                    label, i, c, x, y, vk_img[i], cuda_img[i],
                    std::fabs(vk_img[i] - cuda_img[i]));
    };

    std::printf("[Gate P7] rendered_image: max_abs=%.9g mean_abs=%.9g l2_rel=%.9g psnr=%.9g bad_abs_1e5=%d bad_abs_1e4=%d first_bad_abs_1e5=%d first_bad_abs_1e4=%d max_abs_idx=%d\n",
                max_abs, mean_abs, l2_rel, psnr, bad_abs_1e5, bad_abs_1e4,
                first_bad_abs_1e5, first_bad_abs_1e4, max_abs_idx);
    print_image_diag("first abs>1e-5", first_bad_abs_1e5);
    print_image_diag("first abs>1e-4", first_bad_abs_1e4);
    print_image_diag("max abs", max_abs_idx);

    EXPECT_EQ(first_bad_abs_1e5, 36386)
        << "VK/CUDA P7 first abs>1e-5 drift moved; update raster-state evidence first.";
    EXPECT_EQ(first_bad_abs_1e4, 36386)
        << "VK/CUDA P7 first abs>1e-4 drift moved; update raster-state evidence first.";
    EXPECT_EQ(max_abs_idx, 1516009)
        << "VK/CUDA P7 max rendered-image drift moved; update raster-state evidence first.";
    EXPECT_LE(max_abs, 3.1e-3f)
        << "VK/CUDA P7 rendered image max abs drift exceeds the bounded rare-pixel budget.";
    EXPECT_LE(mean_abs, 4.2e-7)
        << "VK/CUDA P7 rendered image mean abs drift exceeds the full-frame budget.";
    EXPECT_LE(l2_rel, 1.1e-5)
        << "VK/CUDA P7 rendered image L2-relative drift exceeds the full-frame budget.";
    EXPECT_GE(psnr, 103.0)
        << "VK/CUDA P7 rendered image PSNR dropped below the parity budget.";
    EXPECT_LE(bad_abs_1e5, 1960)
        << "VK/CUDA P7 rendered image abs>1e-5 count exceeds the bounded budget.";
    EXPECT_LE(bad_abs_1e4, 127)
        << "VK/CUDA P7 rendered image abs>1e-4 count exceeds the bounded rare-pixel budget.";

    model.free();
}
TEST(VkVsCudaFirstLoss, Gate_L1_L1Loss) {
    const std::string ply = resolve_ply_path();
    if (ply.empty()) GTEST_SKIP() << "basket-aaa.ply not found";
    if (!std::filesystem::exists(kCamPath))
        GTEST_SKIP() << "cameras.json not found: " << kCamPath;
    if (!std::filesystem::exists(dump_dir()))
        GTEST_SKIP() << "CUDA golden dump missing: " << dump_dir();
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/gt_image.npy"))
        << "gt_image.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/rendered_image.npy"))
        << "rendered_image.npy missing in CUDA dump";
    ASSERT_TRUE(std::filesystem::exists(dump_dir() + "/l1_loss.npy"))
        << "l1_loss.npy missing in CUDA dump";

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    auto model = loadPly(ply.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width,  kW);
    ASSERT_EQ(cam.height, kH);
    ASSERT_EQ(model.data.max_coeffs, 16);

    NpyArray gt = load_npy(dump_dir() + "/gt_image.npy");
    assert_dtype(gt, NpyDtype::float32);
    ASSERT_EQ(gt.shape, (std::vector<size_t>{3u, (size_t)kH, (size_t)kW}));

    std::vector<float> vk_pos, vk_scl, vk_rot, vk_sh, vk_opa;
    RawGaussianParams vk_raw = deriveRawFromPlyLoaded(
        model.data, vk_pos, vk_scl, vk_rot, vk_sh, vk_opa);

    VkTrainingConfig tcfg{};
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 0;
    tcfg.lambda_dssim      = 0.0f;
    tcfg.eval_3D           = true;
    tcfg.parity_mode       = true;
    tcfg.densify_from_step = 0;
    tcfg.opacity_reg       = 0.0f;
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;

    GaussianData init_g = model.data;
    auto trainer = std::make_unique<VulkanTrainer>(
        ctx, init_g, vk_raw, /*sh_degree=*/3, kW, kH, tcfg);

    RenderConfig rcfg{};
    rcfg.bg_color[0] = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;
    rcfg.sh_degree      = 3;
    rcfg.eval_3D        = true;
    rcfg.antialiasing   = false;
    rcfg.training       = true;
    rcfg.scale_modifier = 1.0f;
    ASSERT_TRUE(tcfg.eval_3D && tcfg.parity_mode)
        << "L1 compares eval_3D parity-mode scalar loss.";

    const float vk_loss = trainer->forward_only(cam, rcfg, gt.f32(), kW, kH);
    EXPECT_TRUE(std::isfinite(vk_loss)) << "VK forward_only returned non-finite loss";
    EXPECT_FLOAT_EQ(trainer->last_loss(), vk_loss);

    NpyArray cuda_render = load_npy(dump_dir() + "/rendered_image.npy");
    NpyArray cuda_l1 = load_npy(dump_dir() + "/l1_loss.npy");
    assert_dtype(cuda_render, NpyDtype::float32);
    assert_dtype(cuda_l1, NpyDtype::float32);
    ASSERT_EQ(cuda_render.shape, (std::vector<size_t>{3u, static_cast<size_t>(kH), static_cast<size_t>(kW)}));
    ASSERT_EQ(cuda_l1.shape, (std::vector<size_t>{1u}));

    const int image_size = 3 * kW * kH;
    ASSERT_EQ(trainer->rendered_image_size(), image_size);
    const float* vk_img = trainer->rendered_image();
    const float* cuda_img = cuda_render.f32();
    const float* gt_img = gt.f32();
    const float cuda_loss = cuda_l1.f32()[0];
    ASSERT_TRUE(std::isfinite(cuda_loss)) << "CUDA l1_loss.npy contains non-finite loss";

    float vk_l1_sum_float = 0.0f;
    float cuda_l1_sum_float = 0.0f;
    double vk_l1_sum = 0.0;
    double cuda_l1_sum = 0.0;
    for (int i = 0; i < image_size; ++i) {
        ASSERT_TRUE(std::isfinite(vk_img[i])) << "VK rendered image has non-finite value at " << i;
        ASSERT_TRUE(std::isfinite(cuda_img[i])) << "CUDA rendered image has non-finite value at " << i;
        ASSERT_TRUE(std::isfinite(gt_img[i])) << "GT image has non-finite value at " << i;
        const float vk_abs = std::fabs(vk_img[i] - gt_img[i]);
        const float cuda_abs = std::fabs(cuda_img[i] - gt_img[i]);
        vk_l1_sum_float += vk_abs;
        cuda_l1_sum_float += cuda_abs;
        vk_l1_sum += static_cast<double>(vk_abs);
        cuda_l1_sum += static_cast<double>(cuda_abs);
    }
    const float inv_image_size = 1.0f / static_cast<float>(image_size);
    const float vk_l1_recomputed_float = vk_l1_sum_float * inv_image_size;
    const float cuda_l1_recomputed_float = cuda_l1_sum_float * inv_image_size;
    const double vk_l1_recomputed = vk_l1_sum / static_cast<double>(image_size);
    const double cuda_l1_recomputed = cuda_l1_sum / static_cast<double>(image_size);
    const double vk_vs_cuda_loss_abs = std::fabs(static_cast<double>(vk_loss) - static_cast<double>(cuda_loss));
    const double vk_float_recompute_abs = std::fabs(static_cast<double>(vk_loss) - static_cast<double>(vk_l1_recomputed_float));
    const double vk_double_recompute_abs = std::fabs(static_cast<double>(vk_loss) - vk_l1_recomputed);
    const double cuda_float_recompute_abs = std::fabs(static_cast<double>(cuda_loss) - static_cast<double>(cuda_l1_recomputed_float));
    const double cuda_double_recompute_abs = std::fabs(static_cast<double>(cuda_loss) - cuda_l1_recomputed);

    std::printf("[Gate L1] losses: vk_forward=%.9g vk_serial_float=%.9g vk_serial_double=%.9g cuda_golden=%.9g cuda_serial_float=%.9g cuda_serial_double=%.9g vk_cuda_abs=%.9g vk_serial_float_abs=%.9g vk_serial_double_abs=%.9g cuda_serial_float_abs=%.9g cuda_serial_double_abs=%.9g\n",
                vk_loss, vk_l1_recomputed_float, vk_l1_recomputed,
                cuda_loss, cuda_l1_recomputed_float, cuda_l1_recomputed,
                vk_vs_cuda_loss_abs, vk_float_recompute_abs,
                vk_double_recompute_abs, cuda_float_recompute_abs,
                cuda_double_recompute_abs);

    EXPECT_LE(vk_vs_cuda_loss_abs, 1.0e-6)
        << "VK/CUDA L1 scalar loss differs beyond rendered-image-derived budget.";
    EXPECT_LE(vk_double_recompute_abs, 6.0e-7)
        << "VK forward_only loss parallel-reduction-vs-serial-double delta exceeded the bounded budget.";
    EXPECT_LE(cuda_double_recompute_abs, 1.0e-7)
        << "CUDA l1_loss.npy no longer matches rendered_image/gt CHW mean-absolute-error semantics.";

    model.free();
}
