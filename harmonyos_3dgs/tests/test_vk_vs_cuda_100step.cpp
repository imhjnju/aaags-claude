// TEMP DIAGNOSTIC — basketball VK-vs-CUDA 100-step parity
//
// Runs 100 training steps of VulkanTrainer on the COLMAP basketball PLY using
// the SAME initialization, SAME camera, SAME hyperparameters as the CUDA
// reference dumped by tools/dump_basketball_cuda_100step.py. After each step,
// loads the CUDA goldens at tests/golden/basketball_cuda_ref_100step/step_NNNN/
// and reports per-group L2-norm rel_diff, per-element max rel_diff, worst gid,
// and abs_diff for:
//   - gradients: pos, sh_dc, sh_rest, op, sca, rot
//   - post-Adam params: pos, sh, op, sca, rot
//   - loss: scalar rel_diff
//
// Skips automatically if any of:
//   - PLY not present
//   - GT image not present
//   - CUDA goldens missing
//   - No Vulkan compute device
//
// Result is a 100x6 gradient table + 100x5 param table + per-step loss line +
// worst-gid summary at step 100. Lenient EXPECT_LT thresholds so the test passes
// with informational output; tighten later. Trajectory shape is the primary
// signal — see report tables for the sampled-step view.

#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

// Note: STB_IMAGE_IMPLEMENTATION is defined in test_e2e_basketball.cpp; we
// only include the declarations here so we share the same symbols.
#include "extern/stb_image.h"

#include "golden/npy_reader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
constexpr int kW = 720;
constexpr int kH = 960;
constexpr int kSteps = 100;

const std::string kPlyPath =
    "/home/robota/Downloads/basketball/sparse/0/points3D.ply";
const std::string kGtImagePath =
    "/home/robota/Downloads/basketball/images/78899858295079.jpg";

inline std::string golden_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/basketball_cuda_ref_100step";
}
inline std::string step_dir(int step) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "step_%04d", step);
    return golden_dir() + "/" + buf;
}

// ---------------------------------------------------------------------------
// COLMAP PLY loader (binary_little_endian; xyz + nxnynz + rgb)
// ---------------------------------------------------------------------------
struct ColmapPoint {
    float x, y, z;
    unsigned char r, g, b;
};

bool parse_ply_header(std::ifstream& f, int& out_n, size_t& out_offset) {
    std::string line;
    out_n = 0; out_offset = 0;
    if (!std::getline(f, line)) return false;
    if (line.rfind("ply", 0) != 0) return false;
    bool binary_le = false, end = false;
    while (std::getline(f, line)) {
        if (line.rfind("format binary_little_endian", 0) == 0) binary_le = true;
        else if (line.rfind("element vertex ", 0) == 0) out_n = std::stoi(line.substr(15));
        else if (line == "end_header" || line == "end_header\r") { end = true; break; }
    }
    if (!binary_le || !end || out_n <= 0) return false;
    out_offset = static_cast<size_t>(f.tellg());
    return true;
}

std::vector<ColmapPoint> load_colmap_ply(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PLY: " + path);
    int n = 0; size_t off = 0;
    if (!parse_ply_header(f, n, off)) throw std::runtime_error("PLY header parse failed");
    f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    const int stride = 6 * sizeof(float) + 3;
    std::vector<ColmapPoint> pts(static_cast<size_t>(n));
    std::vector<char> buf(static_cast<size_t>(stride));
    for (int i = 0; i < n; ++i) {
        f.read(buf.data(), stride);
        if (f.gcount() != stride) throw std::runtime_error("EOF in PLY data");
        float xyz[3];
        std::memcpy(xyz, buf.data(), 3 * sizeof(float));
        pts[i].x = xyz[0]; pts[i].y = xyz[1]; pts[i].z = xyz[2];
        pts[i].r = (unsigned char)buf[24];
        pts[i].g = (unsigned char)buf[25];
        pts[i].b = (unsigned char)buf[26];
    }
    return pts;
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------
struct GroupStats {
    double l2_diff;
    double l2_ref;
    double l2_rel;        // ||vk - cuda|| / ||cuda||
    float  max_abs_diff;
    float  max_rel_diff;  // |vk - cuda| / max(|cuda|, 1e-8)
    size_t worst_gid_elem;  // flat index of element with max rel diff
    float  vk_at_worst, cuda_at_worst;
};

// TEMP DIAGNOSTIC (Phase 2 of root-cause investigation): bin by per-element
// |cuda_grad| magnitude, compute L2 rel_diff over each bin. If top decile L2
// rel_diff is small and bottom-half is large → entire group rel_diff is
// dominated by atomicAdd noise on near-zero references (option A: atomic-only).
// If top decile L2 rel_diff is also large → real systematic bias (option B).
struct SubsetDiag {
    double l2_rel_top_decile;     // top 10% by |cuda|
    double l2_rel_top_quartile;   // top 25% by |cuda|
    double l2_rel_bottom_half;    // bottom 50% by |cuda|
    double l2_rel_overall;
    double max_abs_cuda;
    double median_abs_cuda;
    double frac_near_zero;        // fraction with |cuda| < 1e-7
    // Absolute-scale checks: if top decile L2 rel_diff is high AND the abs
    // diffs there are at the SAME magnitude as the cuda values, this is a
    // real systematic bias (not cancellation noise).
    double top_decile_mean_abs_diff;
    double top_decile_mean_abs_cuda;
    double top_decile_mean_abs_vk;
};

SubsetDiag compute_subset_diag(const float* vk, const float* cuda, size_t n) {
    SubsetDiag d{};
    if (n == 0) return d;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return std::fabs(cuda[a]) > std::fabs(cuda[b]);  // descending
    });
    auto subset_l2_rel = [&](size_t lo, size_t hi) {
        double sum_d2 = 0.0, sum_r2 = 0.0;
        for (size_t k = lo; k < hi; ++k) {
            size_t i = idx[k];
            double diff = static_cast<double>(vk[i]) - static_cast<double>(cuda[i]);
            sum_d2 += diff * diff;
            sum_r2 += static_cast<double>(cuda[i]) * static_cast<double>(cuda[i]);
        }
        return (sum_r2 > 1e-30) ? std::sqrt(sum_d2 / sum_r2) : 0.0;
    };
    size_t top10 = std::max<size_t>(1, n / 10);
    size_t top25 = std::max<size_t>(1, n / 4);
    size_t halfn = n / 2;
    d.l2_rel_top_decile   = subset_l2_rel(0, top10);
    d.l2_rel_top_quartile = subset_l2_rel(0, top25);
    d.l2_rel_bottom_half  = subset_l2_rel(halfn, n);
    d.l2_rel_overall      = subset_l2_rel(0, n);
    d.max_abs_cuda        = std::fabs(cuda[idx[0]]);
    d.median_abs_cuda     = std::fabs(cuda[idx[n / 2]]);
    size_t near_zero = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::fabs(cuda[i]) < 1e-7f) ++near_zero;
    d.frac_near_zero = double(near_zero) / double(n);

    // Top-decile absolute scale check.
    double sum_abs_d = 0.0, sum_abs_c = 0.0, sum_abs_v = 0.0;
    for (size_t k = 0; k < top10; ++k) {
        size_t i = idx[k];
        sum_abs_d += std::fabs(static_cast<double>(vk[i]) - static_cast<double>(cuda[i]));
        sum_abs_c += std::fabs(static_cast<double>(cuda[i]));
        sum_abs_v += std::fabs(static_cast<double>(vk[i]));
    }
    d.top_decile_mean_abs_diff = sum_abs_d / double(top10);
    d.top_decile_mean_abs_cuda = sum_abs_c / double(top10);
    d.top_decile_mean_abs_vk   = sum_abs_v / double(top10);
    return d;
}

GroupStats compute_stats(const float* vk, const float* cuda, size_t n,
                         int components_per_gid = 1) {
    GroupStats s{};
    double sum_d2 = 0.0, sum_r2 = 0.0;
    s.max_abs_diff = 0.0f;
    s.max_rel_diff = 0.0f;
    s.worst_gid_elem = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = vk[i] - cuda[i];
        float ad = std::fabs(d);
        sum_d2 += static_cast<double>(d) * d;
        sum_r2 += static_cast<double>(cuda[i]) * cuda[i];
        if (ad > s.max_abs_diff) s.max_abs_diff = ad;
        float denom = std::max(std::fabs(cuda[i]), 1e-8f);
        float rd = ad / denom;
        if (rd > s.max_rel_diff) {
            s.max_rel_diff = rd;
            s.worst_gid_elem = i;
            s.vk_at_worst = vk[i];
            s.cuda_at_worst = cuda[i];
        }
    }
    s.l2_diff = std::sqrt(sum_d2);
    s.l2_ref  = std::sqrt(sum_r2);
    s.l2_rel  = s.l2_ref > 1e-30 ? s.l2_diff / s.l2_ref : 0.0;
    (void)components_per_gid;
    return s;
}

// Concatenate VK SH (interleaved [N, K, 3]) into the same layout the CUDA
// dump uses for param_sh.npy: shape (N, K, 3) — features_dc[N,1,3] || features_rest[N,K-1,3].
// VK already stores SH as [N, K, 3] so this is a direct copy.
std::vector<float> vk_sh_to_NKC(const float* vk_sh, int N, int K) {
    std::vector<float> out(static_cast<size_t>(N) * K * 3);
    std::memcpy(out.data(), vk_sh, out.size() * sizeof(float));
    return out;
}

// Build VK gradient slice for SH DC: gather k=0 from interleaved VK [N,K,3].
std::vector<float> vk_grad_sh_dc(const float* vk_grad_sh, int N, int K) {
    std::vector<float> out(static_cast<size_t>(N) * 3);
    for (int i = 0; i < N; ++i) {
        const float* s = vk_grad_sh + static_cast<size_t>(i) * K * 3;
        out[i * 3 + 0] = s[0];
        out[i * 3 + 1] = s[1];
        out[i * 3 + 2] = s[2];
    }
    return out;
}

// Build VK gradient slice for SH REST: gather k=1..K-1.
std::vector<float> vk_grad_sh_rest(const float* vk_grad_sh, int N, int K) {
    if (K <= 1) return {};
    std::vector<float> out(static_cast<size_t>(N) * (K - 1) * 3);
    for (int i = 0; i < N; ++i) {
        const float* s = vk_grad_sh + static_cast<size_t>(i) * K * 3 + 3;
        std::memcpy(&out[static_cast<size_t>(i) * (K - 1) * 3],
                    s, static_cast<size_t>(K - 1) * 3 * sizeof(float));
    }
    return out;
}

}  // namespace

// ===========================================================================
// VkVsCudaBasketball100Step.PerStepParity
// ===========================================================================
TEST(VkVsCudaBasketball100Step, PerStepParity) {
    // ---- Guards ----
    if (!std::filesystem::exists(kPlyPath))
        GTEST_SKIP() << "Basketball PLY not found: " << kPlyPath;
    if (!std::filesystem::exists(kGtImagePath))
        GTEST_SKIP() << "GT image not found: " << kGtImagePath;
    if (!std::filesystem::exists(golden_dir()))
        GTEST_SKIP() << "CUDA 100-step goldens missing: " << golden_dir()
                     << " — generate via tools/dump_basketball_cuda_100step.py";
    for (int s = 1; s <= kSteps; ++s) {
        if (!std::filesystem::exists(step_dir(s) + "/loss.npy")) {
            GTEST_SKIP() << "Missing CUDA golden: " << step_dir(s) << "/loss.npy";
        }
        if (!std::filesystem::exists(step_dir(s) + "/adam_m_pos.npy")) {
            GTEST_SKIP() << "Missing CUDA Adam moment golden: " << step_dir(s) << "/adam_m_pos.npy";
        }
        if (!std::filesystem::exists(step_dir(s) + "/diag_raster_d_conics.npy")) {
            GTEST_SKIP() << "Missing CUDA backward diagnostic golden: " << step_dir(s)
                         << "/diag_raster_d_conics.npy — regenerate via tools/dump_basketball_cuda_100step.py";
        }
    }

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device.";

    // ---- Load GT image as CHW float [0,1] ----
    int loaded_w = 0, loaded_h = 0;
    unsigned char* img_raw = stbi_load(kGtImagePath.c_str(),
                                       &loaded_w, &loaded_h, nullptr, 3);
    ASSERT_NE(img_raw, nullptr) << "Failed to load GT image";
    struct StbGuard { unsigned char* p; ~StbGuard() { stbi_image_free(p); } } _g{img_raw};

    // CUDA dumper resizes to 720x960 if needed (PIL.Image.BILINEAR). For VK
    // we need the same 720x960 input. The GT for cam0 is already 720x960.
    ASSERT_EQ(loaded_w, kW);
    ASSERT_EQ(loaded_h, kH);

    // CHW conversion (CUDA dumper wrote (3,H,W) by transposing HWC).
    std::vector<float> target_chw(static_cast<size_t>(3) * kH * kW);
    const int HW = kW * kH;
    for (int p = 0; p < HW; ++p) {
        target_chw[0 * HW + p] = img_raw[p * 3 + 0] / 255.0f;
        target_chw[1 * HW + p] = img_raw[p * 3 + 1] / 255.0f;
        target_chw[2 * HW + p] = img_raw[p * 3 + 2] / 255.0f;
    }

    // ---- Load PLY + initialise per VK convention ----
    std::vector<ColmapPoint> pts = load_colmap_ply(kPlyPath);
    const int N = static_cast<int>(pts.size());
    std::printf("[100Step] N = %d\n", N);

    const int sh_degree = 3;
    const int max_coeffs = 16;  // (sh_degree+1)^2
    const float C0 = 0.28209479177387814f;

    const float init_raw_scale   = std::log(0.03f);     // ~ -3.5066
    const float init_raw_opacity = std::log(0.1f / 0.9f); // logit(0.1) ~ -2.1972
    const float act_scale   = std::exp(init_raw_scale);
    const float act_opacity = 1.0f / (1.0f + std::exp(-init_raw_opacity));

    std::vector<float> positions   (static_cast<size_t>(N) * 3);
    std::vector<float> scales      (static_cast<size_t>(N) * 3);
    std::vector<float> rotations   (static_cast<size_t>(N) * 4);
    std::vector<float> opacities   (static_cast<size_t>(N));
    std::vector<float> sh_coeffs   (static_cast<size_t>(N) * max_coeffs * 3, 0.0f);
    std::vector<float> act_scales  (static_cast<size_t>(N) * 3);
    std::vector<float> act_opacities(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) {
        positions[i * 3 + 0] = pts[i].x;
        positions[i * 3 + 1] = pts[i].y;
        positions[i * 3 + 2] = pts[i].z;
        scales[i * 3 + 0] = scales[i * 3 + 1] = scales[i * 3 + 2] = init_raw_scale;
        act_scales[i * 3 + 0] = act_scales[i * 3 + 1] = act_scales[i * 3 + 2] = act_scale;
        rotations[i * 4 + 0] = 1.0f;
        rotations[i * 4 + 1] = rotations[i * 4 + 2] = rotations[i * 4 + 3] = 0.0f;
        opacities[i] = init_raw_opacity;
        act_opacities[i] = act_opacity;
        const float r_dc = (pts[i].r / 255.0f - 0.5f) / C0;
        const float g_dc = (pts[i].g / 255.0f - 0.5f) / C0;
        const float b_dc = (pts[i].b / 255.0f - 0.5f) / C0;
        sh_coeffs[i * max_coeffs * 3 + 0] = r_dc;
        sh_coeffs[i * max_coeffs * 3 + 1] = g_dc;
        sh_coeffs[i * max_coeffs * 3 + 2] = b_dc;
    }

    GaussianData init_g{};
    init_g.count      = N;
    init_g.sh_degree  = sh_degree;
    init_g.max_coeffs = max_coeffs;
    init_g.positions  = positions.data();
    init_g.scales     = act_scales.data();
    init_g.rotations  = rotations.data();
    init_g.opacities  = act_opacities.data();
    init_g.sh_coeffs  = sh_coeffs.data();
    init_g.filter_3D  = nullptr;

    RawGaussianParams init_raw{};
    init_raw.count         = N;
    init_raw.sh_degree     = sh_degree;
    init_raw.max_coeffs    = max_coeffs;
    init_raw.raw_positions = positions.data();
    init_raw.raw_scales    = scales.data();
    init_raw.raw_rotations = rotations.data();
    init_raw.raw_sh_coeffs = sh_coeffs.data();
    init_raw.raw_opacities = opacities.data();

    // ---- Camera (matches test_e2e_basketball.cpp:272-329 exactly) ----
    Camera cam{};
    cam.width = kW; cam.height = kH;
    cam.tan_fovx = 0.509851f;
    cam.tan_fovy = 0.678492f;
    cam.cam_pos[0] = -0.42047721f;
    cam.cam_pos[1] =  0.27240201f;
    cam.cam_pos[2] = -0.21650667f;

    cam.view_matrix[ 0] = -0.22993832f; cam.view_matrix[ 1] = -0.38269602f;
    cam.view_matrix[ 2] =  0.89480284f; cam.view_matrix[ 3] =  0.0f;
    cam.view_matrix[ 4] =  0.03527074f; cam.view_matrix[ 5] = -0.92211196f;
    cam.view_matrix[ 6] = -0.38531223f; cam.view_matrix[ 7] =  0.0f;
    cam.view_matrix[ 8] =  0.97256586f; cam.view_matrix[ 9] = -0.05703769f;
    cam.view_matrix[10] =  0.22552685f; cam.view_matrix[11] =  0.0f;
    cam.view_matrix[12] =  0.10427535f; cam.view_matrix[13] =  0.07792116f;
    cam.view_matrix[14] =  0.53003210f; cam.view_matrix[15] =  1.0f;

    cam.viewproj_matrix[ 0] = -0.45099131f; cam.viewproj_matrix[ 1] =  0.06917855f;
    cam.viewproj_matrix[ 2] =  1.90754958f; cam.viewproj_matrix[ 3] =  0.20452127f;
    cam.viewproj_matrix[ 4] = -0.56403945f; cam.viewproj_matrix[ 5] = -1.35906174f;
    cam.viewproj_matrix[ 6] = -0.08406543f; cam.viewproj_matrix[ 7] =  0.11484469f;
    cam.viewproj_matrix[ 8] =  0.89489233f; cam.viewproj_matrix[ 9] = -0.38535077f;
    cam.viewproj_matrix[10] =  0.2255494f;  cam.viewproj_matrix[11] =  0.52008411f;
    cam.viewproj_matrix[12] =  0.89480284f; cam.viewproj_matrix[13] = -0.38531223f;
    cam.viewproj_matrix[14] =  0.22552685f; cam.viewproj_matrix[15] =  0.53003210f;

    // ---- RenderConfig ----
    RenderConfig rcfg{};
    rcfg.sh_degree     = sh_degree;
    rcfg.training      = true;
    rcfg.eval_3D       = false;     // VK backward only supports eval_3D=false
    rcfg.tile_w        = 16;
    rcfg.tile_h        = 16;
    rcfg.antialiasing  = false;
    rcfg.scale_modifier = 1.0f;
    rcfg.bg_color[0]   = rcfg.bg_color[1] = rcfg.bg_color[2] = 0.0f;

    // ---- VkTrainingConfig — match CUDA dumper's Adam config ----
    VkTrainingConfig tcfg{};
    tcfg.max_steps         = kSteps;
    tcfg.pos_lr_init       = 1.6e-4f;
    tcfg.pos_lr_final      = 1.6e-4f;   // CONSTANT pos lr to match CUDA Adam (no schedule)
    tcfg.sh_degree_max     = sh_degree;
    tcfg.sh_degree_warmup  = 0;          // active_sh_degree = sh_degree from step 1
    tcfg.densify_from_step = 0;          // disable
    tcfg.lambda_dssim      = 0.0f;
    tcfg.opacity_reg       = 0.0f;       // CUDA dumper uses pure L1, no reg
    tcfg.scale_reg         = 0.0f;
    tcfg.noise_lr          = 0.0f;       // disable position noise injection
    tcfg.spatial_lr_scale  = 1.0f;
    tcfg.proper_ewa        = false;
    tcfg.eval_3D           = false;
    tcfg.parity_mode       = false;

    // ---- Construct trainer + enable gradient capture ----
    VulkanTrainer trainer(ctx, init_g, init_raw, sh_degree, kW, kH, tcfg);
    trainer.enable_gradient_capture(true);
    trainer.enable_adam_capture(true);
    trainer.enable_backward_diagnostic_capture(true);

    // ---- Per-step parity loop ----
    auto load_npy_safe = [](const std::string& path, size_t expected_numel = 0) {
        NpyArray arr = load_npy(path);
        if (arr.dtype != NpyDtype::float32) {
            throw std::runtime_error("Expected float32 npy: " + path);
        }
        if (expected_numel != 0 && arr.numel() != expected_numel) {
            throw std::runtime_error("Unexpected npy element count: " + path);
        }
        return arr;
    };

    auto print_row = [&](const char* label, const GroupStats& s, size_t n_per_gid) {
        size_t gid = (n_per_gid > 0) ? s.worst_gid_elem / n_per_gid : 0;
        std::printf("    %-10s  l2_rel=%.3e  max_abs=%.3e  max_rel=%.3e  worst_gid=%zu  vk=%g cuda=%g\n",
                    label, s.l2_rel, s.max_abs_diff, s.max_rel_diff, gid,
                    (double)s.vk_at_worst, (double)s.cuda_at_worst);
    };

    std::printf("\n=== [100Step] Live per-step gradient parity (printed every 10 steps + 1,2,5) ===\n");
    std::printf("%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
                "step", "loss_rd", "g_pos", "g_sh_dc", "g_sh_rest", "g_op", "g_sca", "g_rot", "vk_loss");
    auto should_print_live = [](int step) {
        return step == 1 || step == 2 || step == 5 || (step % 10 == 0);
    };

    // TEMP DIAGNOSTIC — step-1 subset bias check
    // Hoisted helpers so step-1 (inside loop) and step-100 (after loop) can
    // both invoke the same magnitude-binned subset diagnostic.
    auto print_subset_diag = [&](const char* label, const SubsetDiag& d) {
        std::printf("    %-12s overall=%.3e top10%%=%.3e top25%%=%.3e "
                    "max|cuda|=%.3e frac<1e-7=%.1f%%  "
                    "[top10%% mean: |diff|=%.3e |cuda|=%.3e |vk|=%.3e ratio=%.3f]\n",
                    label, d.l2_rel_overall, d.l2_rel_top_decile,
                    d.l2_rel_top_quartile,
                    d.max_abs_cuda,
                    d.frac_near_zero * 100.0,
                    d.top_decile_mean_abs_diff,
                    d.top_decile_mean_abs_cuda,
                    d.top_decile_mean_abs_vk,
                    d.top_decile_mean_abs_cuda > 1e-30
                        ? d.top_decile_mean_abs_diff / d.top_decile_mean_abs_cuda
                        : 0.0);
    };

    struct StepRecord {
        float loss_rd;
        GroupStats g_pos, g_sh_dc, g_sh_rest, g_op, g_sca, g_rot;
        GroupStats p_pos, p_sh, p_op, p_sca, p_rot;
        GroupStats m_pos, m_sh_dc, m_sh_rest, m_op, m_sca, m_rot;
        GroupStats v_pos, v_sh_dc, v_sh_rest, v_op, v_sca, v_rot;
        GroupStats rb_m2d, rb_con, rb_op, rb_rgb;
        GroupStats pb_fabc, pb_cov3D, pb_M, pb_scale, pb_R, pb_qn;
        float vk_loss;
        float cuda_loss;
    };
    std::vector<StepRecord> recs(kSteps);

    auto t_run0 = std::chrono::steady_clock::now();
    for (int step = 1; step <= kSteps; ++step) {
        // VK step: returns L1 loss; captures grads + updates raw params.
        const float vk_loss = trainer.step(cam, rcfg, target_chw.data(), kW, kH);
        recs[step - 1].vk_loss = vk_loss;

        // ---- Load CUDA golden for this step ----
        const std::string sd = step_dir(step);
        NpyArray cuda_loss_arr = load_npy_safe(sd + "/loss.npy", 1);
        const float cuda_loss = cuda_loss_arr.f32()[0];
        recs[step - 1].cuda_loss = cuda_loss;
        const float loss_rd = (cuda_loss != 0.0f)
            ? std::fabs(vk_loss - cuda_loss) / std::fabs(cuda_loss)
            : std::fabs(vk_loss - cuda_loss);
        recs[step - 1].loss_rd = loss_rd;

        // ---- Gradients ----
        NpyArray cg_pos    = load_npy_safe(sd + "/grad_pos.npy", (size_t)N * 3);      // (N,3)
        NpyArray cg_sh_dc  = load_npy_safe(sd + "/grad_sh_dc.npy", (size_t)N * 3);    // (N,1,3)
        NpyArray cg_sh_rest= load_npy_safe(sd + "/grad_sh_rest.npy", (size_t)N * (max_coeffs - 1) * 3);  // (N,K-1,3)
        NpyArray cg_op     = load_npy_safe(sd + "/grad_op.npy", (size_t)N);       // (N,1)
        NpyArray cg_sca    = load_npy_safe(sd + "/grad_sca.npy", (size_t)N * 3);      // (N,3)
        NpyArray cg_rot    = load_npy_safe(sd + "/grad_rot.npy", (size_t)N * 4);      // (N,4)

        const std::vector<float>& vg_pos = trainer.captured_grad_positions();
        const std::vector<float>& vg_sca = trainer.captured_grad_scales();
        const std::vector<float>& vg_rot = trainer.captured_grad_rotations();
        const std::vector<float>& vg_sh  = trainer.captured_grad_sh();
        const std::vector<float>& vg_op  = trainer.captured_grad_opacities();

        std::vector<float> vg_sh_dc   = vk_grad_sh_dc(vg_sh.data(), N, max_coeffs);
        std::vector<float> vg_sh_rest = vk_grad_sh_rest(vg_sh.data(), N, max_coeffs);

        recs[step - 1].g_pos     = compute_stats(vg_pos.data(),     cg_pos.f32(),    (size_t)N * 3);
        recs[step - 1].g_sh_dc   = compute_stats(vg_sh_dc.data(),   cg_sh_dc.f32(),  (size_t)N * 3);
        recs[step - 1].g_sh_rest = compute_stats(vg_sh_rest.data(), cg_sh_rest.f32(),(size_t)N * (max_coeffs - 1) * 3);
        recs[step - 1].g_op      = compute_stats(vg_op.data(),      cg_op.f32(),     (size_t)N);
        recs[step - 1].g_sca     = compute_stats(vg_sca.data(),     cg_sca.f32(),    (size_t)N * 3);
        recs[step - 1].g_rot     = compute_stats(vg_rot.data(),     cg_rot.f32(),    (size_t)N * 4);

        NpyArray cd_m2d   = load_npy_safe(sd + "/diag_raster_d_means2D.npy", (size_t)N * 2);
        NpyArray cd_con   = load_npy_safe(sd + "/diag_raster_d_conics.npy",  (size_t)N * 3);
        NpyArray cd_rop   = load_npy_safe(sd + "/diag_raster_d_opacity.npy", (size_t)N);
        NpyArray cd_rgb   = load_npy_safe(sd + "/diag_raster_d_rgb.npy",     (size_t)N * 3);
        NpyArray cd_fabc  = load_npy_safe(sd + "/diag_pre_d_fabc.npy",       (size_t)N * 3);
        NpyArray cd_cov3D = load_npy_safe(sd + "/diag_pre_d_cov3D.npy",      (size_t)N * 6);
        NpyArray cd_M     = load_npy_safe(sd + "/diag_pre_d_M.npy",          (size_t)N * 9);
        NpyArray cd_scale = load_npy_safe(sd + "/diag_pre_d_scale.npy",      (size_t)N * 3);
        NpyArray cd_R     = load_npy_safe(sd + "/diag_pre_d_R.npy",          (size_t)N * 9);
        NpyArray cd_qn    = load_npy_safe(sd + "/diag_pre_d_qn.npy",         (size_t)N * 4);

        recs[step - 1].rb_m2d   = compute_stats(trainer.captured_bwd_d_means2D().data(), cd_m2d.f32(),   (size_t)N * 2);
        recs[step - 1].rb_con   = compute_stats(trainer.captured_bwd_d_conics().data(),  cd_con.f32(),   (size_t)N * 3);
        recs[step - 1].rb_op    = compute_stats(trainer.captured_bwd_d_opacity().data(), cd_rop.f32(),   (size_t)N);
        recs[step - 1].rb_rgb   = compute_stats(trainer.captured_bwd_d_rgb().data(),     cd_rgb.f32(),   (size_t)N * 3);
        recs[step - 1].pb_fabc  = compute_stats(trainer.captured_bwd_d_fabc().data(),    cd_fabc.f32(),  (size_t)N * 3);
        recs[step - 1].pb_cov3D = compute_stats(trainer.captured_bwd_d_cov3D().data(),   cd_cov3D.f32(), (size_t)N * 6);
        recs[step - 1].pb_M     = compute_stats(trainer.captured_bwd_d_M().data(),       cd_M.f32(),     (size_t)N * 9);
        recs[step - 1].pb_scale = compute_stats(trainer.captured_bwd_d_scale().data(),   cd_scale.f32(), (size_t)N * 3);
        recs[step - 1].pb_R     = compute_stats(trainer.captured_bwd_d_R().data(),       cd_R.f32(),     (size_t)N * 9);
        recs[step - 1].pb_qn    = compute_stats(trainer.captured_bwd_d_qn().data(),      cd_qn.f32(),    (size_t)N * 4);

        // ---- Post-Adam params ----
        NpyArray cp_pos = load_npy_safe(sd + "/param_pos.npy", (size_t)N * 3);  // (N,3)
        NpyArray cp_sh  = load_npy_safe(sd + "/param_sh.npy", (size_t)N * max_coeffs * 3);   // (N,K,3)
        NpyArray cp_op  = load_npy_safe(sd + "/param_op.npy", (size_t)N);   // (N,1)
        NpyArray cp_sca = load_npy_safe(sd + "/param_sca.npy", (size_t)N * 3);  // (N,3)
        NpyArray cp_rot = load_npy_safe(sd + "/param_rot.npy", (size_t)N * 4);  // (N,4)

        const RawGaussianParams& vp = trainer.raw_params();
        std::vector<float> v_sh = vk_sh_to_NKC(vp.raw_sh_coeffs, N, max_coeffs);

        recs[step - 1].p_pos = compute_stats(vp.raw_positions, cp_pos.f32(), (size_t)N * 3);
        recs[step - 1].p_sh  = compute_stats(v_sh.data(),      cp_sh.f32(),  (size_t)N * max_coeffs * 3);
        recs[step - 1].p_op  = compute_stats(vp.raw_opacities, cp_op.f32(),  (size_t)N);
        recs[step - 1].p_sca = compute_stats(vp.raw_scales,    cp_sca.f32(), (size_t)N * 3);
        recs[step - 1].p_rot = compute_stats(vp.raw_rotations, cp_rot.f32(), (size_t)N * 4);

        // ---- Post-Adam moments ----
        NpyArray cm_pos     = load_npy_safe(sd + "/adam_m_pos.npy", (size_t)N * 3);
        NpyArray cm_sh_dc   = load_npy_safe(sd + "/adam_m_sh_dc.npy", (size_t)N * 3);
        NpyArray cm_sh_rest = load_npy_safe(sd + "/adam_m_sh_rest.npy", (size_t)N * (max_coeffs - 1) * 3);
        NpyArray cm_op      = load_npy_safe(sd + "/adam_m_op.npy", (size_t)N);
        NpyArray cm_sca     = load_npy_safe(sd + "/adam_m_sca.npy", (size_t)N * 3);
        NpyArray cm_rot     = load_npy_safe(sd + "/adam_m_rot.npy", (size_t)N * 4);
        NpyArray cv_pos     = load_npy_safe(sd + "/adam_v_pos.npy", (size_t)N * 3);
        NpyArray cv_sh_dc   = load_npy_safe(sd + "/adam_v_sh_dc.npy", (size_t)N * 3);
        NpyArray cv_sh_rest = load_npy_safe(sd + "/adam_v_sh_rest.npy", (size_t)N * (max_coeffs - 1) * 3);
        NpyArray cv_op      = load_npy_safe(sd + "/adam_v_op.npy", (size_t)N);
        NpyArray cv_sca     = load_npy_safe(sd + "/adam_v_sca.npy", (size_t)N * 3);
        NpyArray cv_rot     = load_npy_safe(sd + "/adam_v_rot.npy", (size_t)N * 4);

        const auto& vm = trainer.captured_adam_m();
        const auto& vv = trainer.captured_adam_v();
        ASSERT_EQ(vm.size(), 6u);
        ASSERT_EQ(vv.size(), 6u);
        recs[step - 1].m_pos     = compute_stats(vm[0].data(), cm_pos.f32(),     (size_t)N * 3);
        recs[step - 1].m_sh_dc   = compute_stats(vm[1].data(), cm_sh_dc.f32(),   (size_t)N * 3);
        recs[step - 1].m_sh_rest = compute_stats(vm[2].data(), cm_sh_rest.f32(), (size_t)N * (max_coeffs - 1) * 3);
        recs[step - 1].m_op      = compute_stats(vm[3].data(), cm_op.f32(),      (size_t)N);
        recs[step - 1].m_sca     = compute_stats(vm[4].data(), cm_sca.f32(),     (size_t)N * 3);
        recs[step - 1].m_rot     = compute_stats(vm[5].data(), cm_rot.f32(),     (size_t)N * 4);
        recs[step - 1].v_pos     = compute_stats(vv[0].data(), cv_pos.f32(),     (size_t)N * 3);
        recs[step - 1].v_sh_dc   = compute_stats(vv[1].data(), cv_sh_dc.f32(),   (size_t)N * 3);
        recs[step - 1].v_sh_rest = compute_stats(vv[2].data(), cv_sh_rest.f32(), (size_t)N * (max_coeffs - 1) * 3);
        recs[step - 1].v_op      = compute_stats(vv[3].data(), cv_op.f32(),      (size_t)N);
        recs[step - 1].v_sca     = compute_stats(vv[4].data(), cv_sca.f32(),     (size_t)N * 3);
        recs[step - 1].v_rot     = compute_stats(vv[5].data(), cv_rot.f32(),     (size_t)N * 4);

        if (should_print_live(step)) {
            std::printf("%-4d %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e vk=%g cuda=%g\n",
                        step, loss_rd,
                        recs[step - 1].g_pos.l2_rel,
                        recs[step - 1].g_sh_dc.l2_rel,
                        recs[step - 1].g_sh_rest.l2_rel,
                        recs[step - 1].g_op.l2_rel,
                        recs[step - 1].g_sca.l2_rel,
                        recs[step - 1].g_rot.l2_rel,
                        vk_loss, cuda_loss);
            std::fflush(stdout);
        }

        // TEMP DIAGNOSTIC — step-1 subset bias check
        // Run the magnitude-binned subset diagnostic on step-1 raw gradients
        // (pre-Adam). This is the decisive A-vs-B test: if step-1 top-10%
        // L2 rel_diff is already ~3-5%, the bias is per-step chain-rule (A);
        // if step-1 top-10% is bit-exact-ish, the bias is Adam compounding (B).
        // Captures + CUDA goldens are still in scope here; safe to use directly.
        if (step == 1) {
            std::printf("\n=== [100Step] STEP-1 SUBSET DIAGNOSTIC (A-vs-B decisive test) ===\n");
            std::printf("    Pre-Adam gradients only — Adam state is bit-exact-zero at step 1.\n");
            std::printf("    Bin by |cuda_grad| (descending), report L2 rel_diff per subset.\n");
            print_subset_diag("g_pos",     compute_subset_diag(vg_pos.data(),     cg_pos.f32(),    (size_t)N * 3));
            print_subset_diag("g_sh_dc",   compute_subset_diag(vg_sh_dc.data(),   cg_sh_dc.f32(),  (size_t)N * 3));
            print_subset_diag("g_sh_rest", compute_subset_diag(vg_sh_rest.data(), cg_sh_rest.f32(),(size_t)N * (max_coeffs - 1) * 3));
            print_subset_diag("g_op",      compute_subset_diag(vg_op.data(),      cg_op.f32(),     (size_t)N));
            print_subset_diag("g_sca",     compute_subset_diag(vg_sca.data(),     cg_sca.f32(),    (size_t)N * 3));
            print_subset_diag("g_rot",     compute_subset_diag(vg_rot.data(),     cg_rot.f32(),    (size_t)N * 4));
            std::fflush(stdout);
        }
    }
    auto t_run1 = std::chrono::steady_clock::now();
    double vk_runtime_s =
        std::chrono::duration<double>(t_run1 - t_run0).count();
    std::printf("[100Step] VK total runtime: %.2f s (%.1f ms/step avg)\n",
                vk_runtime_s, vk_runtime_s * 1000.0 / kSteps);

    // (Sampled-step tables follow below — full per-step table omitted to keep
    //  output digestible; raw recs[] data is still in memory if extra debug needed.)

    // ---- Sampled-step trajectory-shape table (1, 2, 5, 10, 20, 50, 100) ----
    const int sampled_steps[] = {1, 2, 5, 10, 20, 50, 100};
    std::printf("\n=== [100Step] Sampled trajectory: gradient L2 rel_diff ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "loss_rd", "g_pos", "g_sh_dc", "g_sh_rest",
                "g_op", "g_sca", "g_rot");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s, r.loss_rd,
                    r.g_pos.l2_rel, r.g_sh_dc.l2_rel, r.g_sh_rest.l2_rel,
                    r.g_op.l2_rel, r.g_sca.l2_rel, r.g_rot.l2_rel);
    }

    std::printf("\n=== [100Step] Sampled trajectory: backward chain L2 rel_diff ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "d_con", "d_fabc", "d_cov3D", "d_M", "d_scale", "d_R", "d_qn", "d_m2d");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s, r.rb_con.l2_rel, r.pb_fabc.l2_rel, r.pb_cov3D.l2_rel,
                    r.pb_M.l2_rel, r.pb_scale.l2_rel, r.pb_R.l2_rel,
                    r.pb_qn.l2_rel, r.rb_m2d.l2_rel);
    }

    std::printf("\n=== [100Step] Sampled trajectory: post-Adam param L2 rel_diff ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "p_pos", "p_sh", "p_op", "p_sca", "p_rot");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s, r.p_pos.l2_rel, r.p_sh.l2_rel, r.p_op.l2_rel,
                    r.p_sca.l2_rel, r.p_rot.l2_rel);
    }

    std::printf("\n=== [100Step] Sampled trajectory: Adam m L2 rel_diff ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "m_pos", "m_sh_dc", "m_sh_rest", "m_op", "m_sca", "m_rot");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s, r.m_pos.l2_rel, r.m_sh_dc.l2_rel, r.m_sh_rest.l2_rel,
                    r.m_op.l2_rel, r.m_sca.l2_rel, r.m_rot.l2_rel);
    }

    std::printf("\n=== [100Step] Sampled trajectory: Adam v L2 rel_diff ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "v_pos", "v_sh_dc", "v_sh_rest", "v_op", "v_sca", "v_rot");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s, r.v_pos.l2_rel, r.v_sh_dc.l2_rel, r.v_sh_rest.l2_rel,
                    r.v_op.l2_rel, r.v_sca.l2_rel, r.v_rot.l2_rel);
    }

    std::printf("\n=== [100Step] Sampled trajectory: per-element max rel_diff (worst-gid) ===\n");
    std::printf("%-5s %-11s %-11s %-11s %-11s %-11s %-11s\n",
                "step", "g_pos", "g_sh_dc", "g_sh_rest",
                "g_op", "g_sca", "g_rot");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n",
                    s,
                    r.g_pos.max_rel_diff, r.g_sh_dc.max_rel_diff,
                    r.g_sh_rest.max_rel_diff, r.g_op.max_rel_diff,
                    r.g_sca.max_rel_diff, r.g_rot.max_rel_diff);
    }

    std::printf("\n=== [100Step] Loss absolute trajectory (sampled) ===\n");
    std::printf("%-5s %-12s %-12s %-12s\n", "step", "vk_loss", "cuda_loss", "abs_diff");
    for (int s : sampled_steps) {
        if (s < 1 || s > kSteps) continue;
        const StepRecord& r = recs[s - 1];
        std::printf("%-5d %-12.6f %-12.6f %-12.3e\n",
                    s, (double)r.vk_loss, (double)r.cuda_loss,
                    std::fabs((double)r.vk_loss - (double)r.cuda_loss));
    }

    // Trajectory-shape ratios for each group (step10 -> step100)
    std::printf("\n=== [100Step] Trajectory growth ratio (step100 / step10) ===\n");
    auto growth = [&](double s10, double s100) {
        return (s10 > 1e-30) ? (s100 / s10) : 0.0;
    };
    const StepRecord& r10  = recs[10  - 1];
    const StepRecord& r100 = recs[100 - 1];
    std::printf("  loss_rd      step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.loss_rd, r100.loss_rd, growth(r10.loss_rd, r100.loss_rd));
    std::printf("  g_pos        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_pos.l2_rel, r100.g_pos.l2_rel, growth(r10.g_pos.l2_rel, r100.g_pos.l2_rel));
    std::printf("  g_sh_dc      step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_sh_dc.l2_rel, r100.g_sh_dc.l2_rel, growth(r10.g_sh_dc.l2_rel, r100.g_sh_dc.l2_rel));
    std::printf("  g_sh_rest    step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_sh_rest.l2_rel, r100.g_sh_rest.l2_rel, growth(r10.g_sh_rest.l2_rel, r100.g_sh_rest.l2_rel));
    std::printf("  g_op         step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_op.l2_rel, r100.g_op.l2_rel, growth(r10.g_op.l2_rel, r100.g_op.l2_rel));
    std::printf("  g_sca        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_sca.l2_rel, r100.g_sca.l2_rel, growth(r10.g_sca.l2_rel, r100.g_sca.l2_rel));
    std::printf("  g_rot        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.g_rot.l2_rel, r100.g_rot.l2_rel, growth(r10.g_rot.l2_rel, r100.g_rot.l2_rel));
    std::printf("  p_pos        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.p_pos.l2_rel, r100.p_pos.l2_rel, growth(r10.p_pos.l2_rel, r100.p_pos.l2_rel));
    std::printf("  p_sh         step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.p_sh.l2_rel, r100.p_sh.l2_rel, growth(r10.p_sh.l2_rel, r100.p_sh.l2_rel));
    std::printf("  p_op         step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.p_op.l2_rel, r100.p_op.l2_rel, growth(r10.p_op.l2_rel, r100.p_op.l2_rel));
    std::printf("  p_sca        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.p_sca.l2_rel, r100.p_sca.l2_rel, growth(r10.p_sca.l2_rel, r100.p_sca.l2_rel));
    std::printf("  p_rot        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.p_rot.l2_rel, r100.p_rot.l2_rel, growth(r10.p_rot.l2_rel, r100.p_rot.l2_rel));
    std::printf("  m_sca        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.m_sca.l2_rel, r100.m_sca.l2_rel, growth(r10.m_sca.l2_rel, r100.m_sca.l2_rel));
    std::printf("  m_rot        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.m_rot.l2_rel, r100.m_rot.l2_rel, growth(r10.m_rot.l2_rel, r100.m_rot.l2_rel));
    std::printf("  v_sca        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.v_sca.l2_rel, r100.v_sca.l2_rel, growth(r10.v_sca.l2_rel, r100.v_sca.l2_rel));
    std::printf("  v_rot        step10=%.3e  step100=%.3e  ratio=%.2fx\n",
                r10.v_rot.l2_rel, r100.v_rot.l2_rel, growth(r10.v_rot.l2_rel, r100.v_rot.l2_rel));

    // ---- Detailed step-100 worst-gid summary ----
    std::printf("\n=== [100Step] Step 100 worst-element summary ===\n");
    std::printf("  Gradients:\n");
    print_row("g_pos",     recs[kSteps - 1].g_pos,     3);
    print_row("g_sh_dc",   recs[kSteps - 1].g_sh_dc,   3);
    print_row("g_sh_rest", recs[kSteps - 1].g_sh_rest, (max_coeffs - 1) * 3);
    print_row("g_op",      recs[kSteps - 1].g_op,      1);
    print_row("g_sca",     recs[kSteps - 1].g_sca,     3);
    print_row("g_rot",     recs[kSteps - 1].g_rot,     4);
    std::printf("  Backward chain:\n");
    print_row("rb_con",   recs[kSteps - 1].rb_con,   3);
    print_row("pb_fabc",  recs[kSteps - 1].pb_fabc,  3);
    print_row("pb_cov3D", recs[kSteps - 1].pb_cov3D, 6);
    print_row("pb_M",     recs[kSteps - 1].pb_M,     9);
    print_row("pb_scale", recs[kSteps - 1].pb_scale, 3);
    print_row("pb_R",     recs[kSteps - 1].pb_R,     9);
    print_row("pb_qn",    recs[kSteps - 1].pb_qn,    4);
    std::printf("  Post-Adam params:\n");
    print_row("p_pos",     recs[kSteps - 1].p_pos,     3);
    print_row("p_sh",      recs[kSteps - 1].p_sh,      max_coeffs * 3);
    print_row("p_op",      recs[kSteps - 1].p_op,      1);
    print_row("p_sca",     recs[kSteps - 1].p_sca,     3);
    print_row("p_rot",     recs[kSteps - 1].p_rot,     4);

    // ---- TEMP DIAGNOSTIC (Phase 2 of A-vs-B root-cause investigation) ----
    // Re-run gradients at step 100 through the magnitude-binned subset
    // diagnostic. (print_subset_diag lambda is hoisted above the loop so the
    // step-1 invocation can reuse it.)
    auto run_subset_diag = [&](int step_to_diag) {
        const std::string sd = step_dir(step_to_diag);
        NpyArray cg_pos    = load_npy_safe(sd + "/grad_pos.npy", (size_t)N * 3);
        NpyArray cg_sh_dc  = load_npy_safe(sd + "/grad_sh_dc.npy", (size_t)N * 3);
        NpyArray cg_sh_rest= load_npy_safe(sd + "/grad_sh_rest.npy", (size_t)N * (max_coeffs - 1) * 3);
        NpyArray cg_op     = load_npy_safe(sd + "/grad_op.npy", (size_t)N);
        NpyArray cg_sca    = load_npy_safe(sd + "/grad_sca.npy", (size_t)N * 3);
        NpyArray cg_rot    = load_npy_safe(sd + "/grad_rot.npy", (size_t)N * 4);
        // Note: VK captured grads have been overwritten by later steps; we
        // can ONLY do this at the last step (step 100), since trainer holds
        // captures from the most recent step.call. So step_to_diag MUST be
        // kSteps for the VK side to be valid.
        const std::vector<float>& vg_pos = trainer.captured_grad_positions();
        const std::vector<float>& vg_sca = trainer.captured_grad_scales();
        const std::vector<float>& vg_rot = trainer.captured_grad_rotations();
        const std::vector<float>& vg_sh  = trainer.captured_grad_sh();
        const std::vector<float>& vg_op  = trainer.captured_grad_opacities();
        std::vector<float> vg_sh_dc   = vk_grad_sh_dc(vg_sh.data(), N, max_coeffs);
        std::vector<float> vg_sh_rest = vk_grad_sh_rest(vg_sh.data(), N, max_coeffs);

        std::printf("\n=== [100Step] PHASE-2 SUBSET DIAGNOSTIC at step %d ===\n", step_to_diag);
        std::printf("    Bin by |cuda_grad| (descending), report L2 rel_diff per subset.\n");
        std::printf("    If top10%% bit-exact and bot50%% large -> atomicAdd noise on near-zero refs.\n");
        std::printf("    If top10%% also large -> real systematic bias.\n");
        print_subset_diag("g_pos",     compute_subset_diag(vg_pos.data(),     cg_pos.f32(),    (size_t)N * 3));
        print_subset_diag("g_sh_dc",   compute_subset_diag(vg_sh_dc.data(),   cg_sh_dc.f32(),  (size_t)N * 3));
        print_subset_diag("g_sh_rest", compute_subset_diag(vg_sh_rest.data(), cg_sh_rest.f32(),(size_t)N * (max_coeffs - 1) * 3));
        print_subset_diag("g_op",      compute_subset_diag(vg_op.data(),      cg_op.f32(),     (size_t)N));
        print_subset_diag("g_sca",     compute_subset_diag(vg_sca.data(),     cg_sca.f32(),    (size_t)N * 3));
        print_subset_diag("g_rot",     compute_subset_diag(vg_rot.data(),     cg_rot.f32(),    (size_t)N * 4));
    };
    run_subset_diag(kSteps);  // step 100 — most decisive

    // ---- TEMP DIAGNOSTIC: worst-gid migration step10 -> step100 ----
    auto worst_gid = [](const GroupStats& s, size_t n_per_gid) {
        return n_per_gid > 0 ? s.worst_gid_elem / n_per_gid : 0;
    };
    std::printf("\n=== [100Step] PHASE-4 worst-gid migration step10 vs step100 ===\n");
    std::printf("  %-10s %-10s %-10s %-10s\n", "group", "step10_gid", "step100_gid", "stable?");
    auto report_migration = [&](const char* label,
                                const GroupStats& s10, size_t n10,
                                const GroupStats& s100, size_t n100) {
        size_t g10 = worst_gid(s10, n10);
        size_t g100 = worst_gid(s100, n100);
        const char* tag = (g10 == g100) ? "STABLE" : "MIGRATING";
        std::printf("  %-10s %-10zu %-10zu %s\n", label, g10, g100, tag);
    };
    {
        const StepRecord& r10s = recs[10 - 1];
        const StepRecord& r100s = recs[100 - 1];
        report_migration("g_pos",     r10s.g_pos,     3,
                         r100s.g_pos,     3);
        report_migration("g_sh_dc",   r10s.g_sh_dc,   3,
                         r100s.g_sh_dc,   3);
        report_migration("g_sh_rest", r10s.g_sh_rest, (max_coeffs - 1) * 3,
                         r100s.g_sh_rest, (max_coeffs - 1) * 3);
        report_migration("g_op",      r10s.g_op,      1,
                         r100s.g_op,      1);
        report_migration("g_sca",     r10s.g_sca,     3,
                         r100s.g_sca,     3);
        report_migration("g_rot",     r10s.g_rot,     4,
                         r100s.g_rot,     4);
    }

    // ---- Lenient assertions (informational gate) ----
    // Goal: report numbers, only fail on egregious divergence — at 100 steps
    // we expect loss + grad + param drift to compound, so thresholds are
    // looser than 10-step. Hard ceiling = "things blew up".
    constexpr double kLooseLossRd  = 2e-1;   // 20% loss rel_diff at any step = bad
    constexpr double kLooseGradRel = 5.0;    // 500% grad L2 rel_diff = bad
    constexpr double kLooseParamRel= 1.0;    // 100% param L2 rel_diff = bad

    for (int step = 1; step <= kSteps; ++step) {
        const StepRecord& r = recs[step - 1];
        EXPECT_LT(r.loss_rd, kLooseLossRd)     << "step " << step << " loss";
        EXPECT_LT(r.g_pos.l2_rel,     kLooseGradRel) << "step " << step << " g_pos";
        EXPECT_LT(r.g_sh_dc.l2_rel,   kLooseGradRel) << "step " << step << " g_sh_dc";
        EXPECT_LT(r.g_sh_rest.l2_rel, kLooseGradRel) << "step " << step << " g_sh_rest";
        EXPECT_LT(r.g_op.l2_rel,      kLooseGradRel) << "step " << step << " g_op";
        EXPECT_LT(r.g_sca.l2_rel,     kLooseGradRel) << "step " << step << " g_sca";
        // Step 1 g_rot: CUDA reference is analytically near-zero (~1e-10 L2)
        // because grad is zero at identity rotation + isotropic scale; VK
        // emits ULP-level noise, making the ratio meaningless. Check absolute
        // floor instead. Steps 2+ have meaningful CUDA L2 norms.
        if (step == 1) {
            EXPECT_LT(r.g_rot.max_abs_diff, 1e-6f)
                << "step 1 g_rot abs (CUDA grad is analytically ~0; checking ULP floor)";
        } else {
            EXPECT_LT(r.g_rot.l2_rel, kLooseGradRel) << "step " << step << " g_rot";
        }
        EXPECT_LT(r.p_pos.l2_rel,     kLooseParamRel) << "step " << step << " p_pos";
        EXPECT_LT(r.p_sh.l2_rel,      kLooseParamRel) << "step " << step << " p_sh";
        EXPECT_LT(r.p_op.l2_rel,      kLooseParamRel) << "step " << step << " p_op";
        EXPECT_LT(r.p_sca.l2_rel,     kLooseParamRel) << "step " << step << " p_sca";
        EXPECT_LT(r.p_rot.l2_rel,     kLooseParamRel) << "step " << step << " p_rot";
        EXPECT_LT(r.m_pos.l2_rel,     kLooseGradRel) << "step " << step << " m_pos";
        EXPECT_LT(r.m_sh_dc.l2_rel,   kLooseGradRel) << "step " << step << " m_sh_dc";
        EXPECT_LT(r.m_sh_rest.l2_rel, kLooseGradRel) << "step " << step << " m_sh_rest";
        EXPECT_LT(r.m_op.l2_rel,      kLooseGradRel) << "step " << step << " m_op";
        EXPECT_LT(r.m_sca.l2_rel,     kLooseGradRel) << "step " << step << " m_sca";
        if (step == 1) {
            EXPECT_LT(r.m_rot.max_abs_diff, 1e-6f) << "step 1 m_rot abs";
            EXPECT_LT(r.v_rot.max_abs_diff, 1e-12f) << "step 1 v_rot abs";
        } else {
            EXPECT_LT(r.m_rot.l2_rel, kLooseGradRel) << "step " << step << " m_rot";
        }
        EXPECT_LT(r.v_pos.l2_rel,     kLooseGradRel) << "step " << step << " v_pos";
        EXPECT_LT(r.v_sh_dc.l2_rel,   kLooseGradRel) << "step " << step << " v_sh_dc";
        EXPECT_LT(r.v_sh_rest.l2_rel, kLooseGradRel) << "step " << step << " v_sh_rest";
        EXPECT_LT(r.v_op.l2_rel,      kLooseGradRel) << "step " << step << " v_op";
        EXPECT_LT(r.v_sca.l2_rel,     kLooseGradRel) << "step " << step << " v_sca";
        if (step != 1) {
            EXPECT_LT(r.v_rot.l2_rel, kLooseGradRel) << "step " << step << " v_rot";
        }
    }
}
