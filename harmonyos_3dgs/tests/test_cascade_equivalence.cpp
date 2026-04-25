// test_cascade_equivalence.cpp — Phase 3 of the CUDA↔VK cascade equivalence
// harness.
//
// Loads per-level trace tensors produced by:
//   tools/dump_cuda_cascade_trace.py                → ${DUMP}/cuda/*.npy
//   (future) VK 3-level cascade port with trace    → ${DUMP}/vk/*.npy
//
// Compares: gid sequences bit-exact (equal-depth ties allowed unordered),
// depth/alpha with ε tolerance, per-level.
//
// Until the VK side is implemented, the default test feeds the CUDA trace to
// BOTH sides — a self-equivalence sanity check that validates the comparator
// mechanism itself.  A second test (disabled by default) runs the real
// comparison; it SKIPs if ${DUMP}/vk/ is missing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "golden/npy_reader.h"

namespace {

struct CompareOpts {
    float depth_eps = 1e-5f;
    float alpha_eps = 1e-6f;
    float depth_tie_eps = 1e-7f;  // |d_a - d_b| below this → values considered tied (unordered)
};

struct LevelReport {
    std::string level;
    size_t total_slots = 0;
    size_t nonempty_slots = 0;
    size_t exact_matches = 0;
    size_t tie_matches = 0;
    size_t gid_mismatches = 0;
    size_t depth_mismatches = 0;
    size_t alpha_mismatches = 0;
    std::string first_divergence;

    bool ok() const {
        return gid_mismatches == 0 && depth_mismatches == 0 && alpha_mismatches == 0;
    }
};

// ---- Generic "grouped slot" compare (TAIL uses groups of 64; MID uses 8) ----
// Rationale:
//   TAIL layout [K, MAX_T, 16, 64]  → group = 64 slots (subtile).
//   MID  layout [K, MAX_M, 16, 4, 8] → group = 8 slots ((subtile, quadrant)).
// CUDA side runs `batcherSort<32>` with no secondary key → ties are unordered.
// Both groups sorted by (depth, gid) canonical form before bitwise comparison.
static LevelReport compare_grouped(const char* label,
                                   const NpyArray& a_d, const NpyArray& a_id,
                                   const NpyArray& b_d, const NpyArray& b_id,
                                   size_t slots_per_group,
                                   const CompareOpts& opts) {
    LevelReport r{label};
    assert_dtype(a_d,  NpyDtype::float32);
    assert_dtype(b_d,  NpyDtype::float32);
    assert_dtype(a_id, NpyDtype::int32);
    assert_dtype(b_id, NpyDtype::int32);
    if (a_d.shape != b_d.shape || a_id.shape != b_id.shape) {
        r.first_divergence = "shape mismatch";
        r.gid_mismatches = 1;
        return r;
    }
    const size_t N = a_d.numel();
    const float* ad = const_cast<NpyArray&>(a_d).f32();
    const float* bd = const_cast<NpyArray&>(b_d).f32();
    const int32_t* ai = const_cast<NpyArray&>(a_id).i32();
    const int32_t* bi = const_cast<NpyArray&>(b_id).i32();
    r.total_slots = N;

    const size_t num_groups = N / slots_per_group;
    std::vector<std::pair<float,int32_t>> ga(slots_per_group), gb(slots_per_group);
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t base = g * slots_per_group;
        bool any_valid = false;
        for (size_t s = 0; s < slots_per_group; ++s) {
            ga[s] = {ad[base+s], ai[base+s]};
            gb[s] = {bd[base+s], bi[base+s]};
            if (ga[s].second != -1 || gb[s].second != -1) any_valid = true;
        }
        if (!any_valid) continue;
        ++r.nonempty_slots;

        // Sort both groups by (depth, gid). Tie unordered ⇒ same canonical form.
        auto cmp = [&](const auto& x, const auto& y) {
            if (std::fabs(x.first - y.first) < opts.depth_tie_eps)
                return x.second < y.second;
            return x.first < y.first;
        };
        std::sort(ga.begin(), ga.end(), cmp);
        std::sort(gb.begin(), gb.end(), cmp);

        bool exact = true;
        for (size_t s = 0; s < slots_per_group; ++s) {
            bool d_ok = std::fabs(ga[s].first - gb[s].first) <= opts.depth_eps
                        || (std::isinf(ga[s].first) && std::isinf(gb[s].first));
            bool i_ok = ga[s].second == gb[s].second;
            if (!d_ok) { r.depth_mismatches++; exact = false; }
            if (!i_ok) { r.gid_mismatches++;   exact = false; }
            if (!exact && r.first_divergence.empty()) {
                std::ostringstream os;
                os << label << " group " << g << " slot " << s
                   << ": A(" << ga[s].first << "," << ga[s].second << ") "
                   << "B(" << gb[s].first << "," << gb[s].second << ")";
                r.first_divergence = os.str();
            }
        }
        if (exact) r.exact_matches++;
    }
    return r;
}

// TAIL: group size 64 (one subtile). Shape [K, MAX_T, 16, 64].
LevelReport compare_tail(const NpyArray& a_d, const NpyArray& a_id,
                         const NpyArray& b_d, const NpyArray& b_id,
                         const CompareOpts& opts) {
    return compare_grouped("TAIL", a_d, a_id, b_d, b_id, /*slots_per_group=*/64, opts);
}

// MID: group size 8 (one (subtile, quadrant)). Shape [K, MAX_M, 16, 4, 8].
LevelReport compare_mid(const NpyArray& a_d, const NpyArray& a_id,
                        const NpyArray& b_d, const NpyArray& b_id,
                        const CompareOpts& opts) {
    return compare_grouped("MID", a_d, a_id, b_d, b_id, /*slots_per_group=*/8, opts);
}

// ---- Generic "per-pixel variable-length sequence" compare ---------------
// For HEAD_INS / HEAD_BLEND. Layout [K, 256, MAX_STEP] with a per-pixel
// cursor [K, 256] giving actual step count. gid==-1 marks unused slots.
LevelReport compare_head_stream(
    const char* label,
    const NpyArray& a_d, const NpyArray& a_a, const NpyArray& a_id, const NpyArray& a_cur,
    const NpyArray& b_d, const NpyArray& b_a, const NpyArray& b_id, const NpyArray& b_cur,
    const CompareOpts& opts)
{
    LevelReport r{label};
    if (a_d.shape != b_d.shape || a_cur.shape != b_cur.shape) {
        r.first_divergence = "shape mismatch";
        r.gid_mismatches = 1;
        return r;
    }
    // shape: [K, 256, MAX]
    const size_t K   = a_d.shape[0];
    const size_t PIX = a_d.shape[1];
    const size_t MAX = a_d.shape[2];
    const float* ad  = const_cast<NpyArray&>(a_d).f32();
    const float* bd  = const_cast<NpyArray&>(b_d).f32();
    const float* aa  = const_cast<NpyArray&>(a_a).f32();
    const float* ba  = const_cast<NpyArray&>(b_a).f32();
    const int32_t* ai = const_cast<NpyArray&>(a_id).i32();
    const int32_t* bi = const_cast<NpyArray&>(b_id).i32();
    const uint32_t* ac = const_cast<NpyArray&>(a_cur).u32();
    const uint32_t* bc = const_cast<NpyArray&>(b_cur).u32();

    r.total_slots = K * PIX;
    for (size_t k = 0; k < K; ++k) {
        for (size_t p = 0; p < PIX; ++p) {
            uint32_t na = ac[k * PIX + p];
            uint32_t nb = bc[k * PIX + p];
            if (na == 0 && nb == 0) continue;
            ++r.nonempty_slots;

            if (na != nb) {
                ++r.gid_mismatches;
                if (r.first_divergence.empty()) {
                    std::ostringstream os;
                    os << label << " k=" << k << " p=" << p
                       << " step count A=" << na << " B=" << nb;
                    r.first_divergence = os.str();
                }
                continue;
            }
            // Compare step-by-step. These sequences are ordered (blend order /
            // insertion order), so no reordering.
            const size_t base = (k * PIX + p) * MAX;
            bool exact = true;
            for (uint32_t s = 0; s < na; ++s) {
                bool i_ok = ai[base+s] == bi[base+s];
                bool d_ok = std::fabs(ad[base+s] - bd[base+s]) <= opts.depth_eps;
                bool a_ok = std::fabs(aa[base+s] - ba[base+s]) <= opts.alpha_eps;
                if (!i_ok) { r.gid_mismatches++;   exact = false; }
                if (!d_ok) { r.depth_mismatches++; exact = false; }
                if (!a_ok) { r.alpha_mismatches++; exact = false; }
                if (!exact && r.first_divergence.empty()) {
                    std::ostringstream os;
                    os << label << " k=" << k << " p=" << p << " step=" << s
                       << ": A(" << ai[base+s] << "," << ad[base+s] << "," << aa[base+s] << ") "
                       << "B(" << bi[base+s] << "," << bd[base+s] << "," << ba[base+s] << ")";
                    r.first_divergence = os.str();
                }
            }
            if (exact) r.exact_matches++;
        }
    }
    return r;
}

// ---- Load / skip helpers --------------------------------------------------
bool dir_has_trace(const std::string& d) {
    return std::filesystem::exists(d + "/tail_depths.npy")
        && std::filesystem::exists(d + "/mid_depths.npy")
        && std::filesystem::exists(d + "/head_ins_depth.npy")
        && std::filesystem::exists(d + "/head_blend_depth.npy");
}

}  // namespace

TEST(CascadeEquivalence, SelfCompare_Cuda_vs_Cuda) {
    // Default dump dir: next to the basket dump from Phase 1.
    const std::string dump = std::string(CMAKE_BINARY_DIR) + "/cascade_trace";
    const std::string cuda_dir = dump + "/cuda";
    if (!dir_has_trace(cuda_dir)) {
        GTEST_SKIP() << "CUDA trace not produced — run "
                        "diff-gaussian-rasterization/tools/dump_cuda_cascade_trace.py "
                        "first. Missing: " << cuda_dir;
    }

    auto A_tail_d = load_npy(cuda_dir + "/tail_depths.npy");
    auto A_tail_i = load_npy(cuda_dir + "/tail_ids.npy");
    auto A_mid_d  = load_npy(cuda_dir + "/mid_depths.npy");
    auto A_mid_i  = load_npy(cuda_dir + "/mid_ids.npy");
    auto A_hi_d   = load_npy(cuda_dir + "/head_ins_depth.npy");
    auto A_hi_a   = load_npy(cuda_dir + "/head_ins_alpha.npy");
    auto A_hi_i   = load_npy(cuda_dir + "/head_ins_gid.npy");
    auto A_hi_c   = load_npy(cuda_dir + "/head_ins_cursor.npy");
    auto A_hb_d   = load_npy(cuda_dir + "/head_blend_depth.npy");
    auto A_hb_a   = load_npy(cuda_dir + "/head_blend_alpha.npy");
    auto A_hb_i   = load_npy(cuda_dir + "/head_blend_gid.npy");
    auto A_hb_c   = load_npy(cuda_dir + "/head_blend_cursor.npy");

    CompareOpts opts{};

    // Self-compare: feed CUDA trace as both A and B. Must be exact.
    auto r_tail = compare_tail(A_tail_d, A_tail_i, A_tail_d, A_tail_i, opts);
    auto r_mid  = compare_mid (A_mid_d,  A_mid_i,  A_mid_d,  A_mid_i,  opts);
    auto r_hi   = compare_head_stream("HEAD_INS",
                    A_hi_d, A_hi_a, A_hi_i, A_hi_c,
                    A_hi_d, A_hi_a, A_hi_i, A_hi_c, opts);
    auto r_hb   = compare_head_stream("HEAD_BLEND",
                    A_hb_d, A_hb_a, A_hb_i, A_hb_c,
                    A_hb_d, A_hb_a, A_hb_i, A_hb_c, opts);

    for (const auto* r : {&r_tail, &r_mid, &r_hi, &r_hb}) {
        std::printf("[CascadeEquivalence] %s: slots=%zu non-empty=%zu exact=%zu "
                    "gid_mm=%zu d_mm=%zu a_mm=%zu\n",
                    r->level.c_str(), r->total_slots, r->nonempty_slots,
                    r->exact_matches,
                    r->gid_mismatches, r->depth_mismatches, r->alpha_mismatches);
        if (!r->first_divergence.empty())
            std::printf("    first divergence: %s\n", r->first_divergence.c_str());
    }

    EXPECT_TRUE(r_tail.ok()) << "TAIL self-compare failed: " << r_tail.first_divergence;
    EXPECT_TRUE(r_mid.ok())  << "MID self-compare failed: "  << r_mid.first_divergence;
    EXPECT_TRUE(r_hi.ok())   << "HEAD_INS self-compare failed: " << r_hi.first_divergence;
    EXPECT_TRUE(r_hb.ok())   << "HEAD_BLEND self-compare failed: " << r_hb.first_divergence;
}

TEST(CascadeEquivalence, Vk_vs_Cuda) {
    const std::string dump = std::string(CMAKE_BINARY_DIR) + "/cascade_trace";
    const std::string cuda_dir = dump + "/cuda";
    const std::string vk_dir   = dump + "/vk";
    if (!dir_has_trace(cuda_dir) || !dir_has_trace(vk_dir)) {
        GTEST_SKIP() << "Both " << cuda_dir << " and " << vk_dir
                     << " must be populated. VK side requires the 3-level cascade "
                        "port with trace hooks — expected to be filled in Phase 4.";
    }

    // Phase 4 / Milestone A: the VK shader declares the trace SSBO bindings
    // but does NOT yet emit any trace writes. So vk_dir/*.npy are zero-filled
    // with shapes/dtypes matching the CUDA side. This test is expected to
    // FAIL with a clean "divergence at first CUDA-populated slot vs 0" report
    // until Milestones B..E populate the levels.

    auto A_tail_d = load_npy(cuda_dir + "/tail_depths.npy");
    auto A_tail_i = load_npy(cuda_dir + "/tail_ids.npy");
    auto A_mid_d  = load_npy(cuda_dir + "/mid_depths.npy");
    auto A_mid_i  = load_npy(cuda_dir + "/mid_ids.npy");
    auto A_hi_d   = load_npy(cuda_dir + "/head_ins_depth.npy");
    auto A_hi_a   = load_npy(cuda_dir + "/head_ins_alpha.npy");
    auto A_hi_i   = load_npy(cuda_dir + "/head_ins_gid.npy");
    auto A_hi_c   = load_npy(cuda_dir + "/head_ins_cursor.npy");
    auto A_hb_d   = load_npy(cuda_dir + "/head_blend_depth.npy");
    auto A_hb_a   = load_npy(cuda_dir + "/head_blend_alpha.npy");
    auto A_hb_i   = load_npy(cuda_dir + "/head_blend_gid.npy");
    auto A_hb_c   = load_npy(cuda_dir + "/head_blend_cursor.npy");

    auto B_tail_d = load_npy(vk_dir + "/tail_depths.npy");
    auto B_tail_i = load_npy(vk_dir + "/tail_ids.npy");
    auto B_mid_d  = load_npy(vk_dir + "/mid_depths.npy");
    auto B_mid_i  = load_npy(vk_dir + "/mid_ids.npy");
    auto B_hi_d   = load_npy(vk_dir + "/head_ins_depth.npy");
    auto B_hi_a   = load_npy(vk_dir + "/head_ins_alpha.npy");
    auto B_hi_i   = load_npy(vk_dir + "/head_ins_gid.npy");
    auto B_hi_c   = load_npy(vk_dir + "/head_ins_cursor.npy");
    auto B_hb_d   = load_npy(vk_dir + "/head_blend_depth.npy");
    auto B_hb_a   = load_npy(vk_dir + "/head_blend_alpha.npy");
    auto B_hb_i   = load_npy(vk_dir + "/head_blend_gid.npy");
    auto B_hb_c   = load_npy(vk_dir + "/head_blend_cursor.npy");

    CompareOpts opts{};
    auto r_tail = compare_tail(A_tail_d, A_tail_i, B_tail_d, B_tail_i, opts);
    auto r_mid  = compare_mid (A_mid_d,  A_mid_i,  B_mid_d,  B_mid_i,  opts);
    auto r_hi   = compare_head_stream("HEAD_INS",
                    A_hi_d, A_hi_a, A_hi_i, A_hi_c,
                    B_hi_d, B_hi_a, B_hi_i, B_hi_c, opts);
    auto r_hb   = compare_head_stream("HEAD_BLEND",
                    A_hb_d, A_hb_a, A_hb_i, A_hb_c,
                    B_hb_d, B_hb_a, B_hb_i, B_hb_c, opts);

    for (const auto* r : {&r_tail, &r_mid, &r_hi, &r_hb}) {
        std::printf("[CascadeEquivalence] %s: slots=%zu non-empty=%zu exact=%zu "
                    "gid_mm=%zu d_mm=%zu a_mm=%zu\n",
                    r->level.c_str(), r->total_slots, r->nonempty_slots,
                    r->exact_matches,
                    r->gid_mismatches, r->depth_mismatches, r->alpha_mismatches);
        if (!r->first_divergence.empty())
            std::printf("    first divergence: %s\n", r->first_divergence.c_str());
    }

    EXPECT_TRUE(r_tail.ok()) << "TAIL Vk_vs_Cuda divergence: " << r_tail.first_divergence;
    EXPECT_TRUE(r_mid.ok())  << "MID Vk_vs_Cuda divergence: "  << r_mid.first_divergence;
    EXPECT_TRUE(r_hi.ok())   << "HEAD_INS Vk_vs_Cuda divergence: " << r_hi.first_divergence;
    EXPECT_TRUE(r_hb.ok())   << "HEAD_BLEND Vk_vs_Cuda divergence: " << r_hb.first_divergence;
}
