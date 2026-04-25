// test_vk_vs_cuda_first_loss.cpp — VK vs CUDA first-loss parity harness
// (Phase 0.4 of dev_notes/vk_cuda_first_loss_parity_plan.md).
//
// Runs Phase 1 gates (I1-I4, input isomorphism) that compare the exact inputs
// the VK forward path will consume against the CUDA golden dump at
// tests/golden/basketball/cuda_ref/step_0001/.  Phase 2 (forward stage) and
// Phase 3 (loss) gates are GTEST_SKIP() placeholders for later phases.
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
// Phase 2 / Phase 3 stubs — reserved for forward-stage and loss gates.
// ===========================================================================
TEST(VkVsCudaFirstLoss, Gate_P1_Means2D) {
    GTEST_SKIP() << "Phase 2 gate (means2D parity). See "
                    "dev_notes/vk_cuda_first_loss_parity_plan.md";
}
TEST(VkVsCudaFirstLoss, Gate_P2_ConicOpacity) {
    GTEST_SKIP() << "Phase 2 gate (conic+opacity parity), not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_P3_RgbColors) {
    GTEST_SKIP() << "Phase 2 gate (SH-evaluated RGB parity), not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_P4_Radii) {
    GTEST_SKIP() << "Phase 2 gate (radii / AABB parity; blocked on new_aabb "
                    "Phase-2 risk noted in Gate_I4_ConfigFlags), not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_P5_SortedIds) {
    GTEST_SKIP() << "Phase 2 gate (per-tile sorted Gaussian IDs parity), "
                    "not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_P6_TFinalNContrib) {
    GTEST_SKIP() << "Phase 2 gate (T_final + n_contrib parity), not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_P7_RenderedImage) {
    GTEST_SKIP() << "Phase 2 gate (final rendered image parity), not yet implemented";
}
TEST(VkVsCudaFirstLoss, Gate_L1_L1Loss) {
    GTEST_SKIP() << "Phase 3 gate (L1 loss 0.0641618 parity), not yet implemented";
}
