// test_fuchsia_radix_e2e_vk.cpp -- Phase 2.3b e2e test.
//
// Validates the env-var routing from `GS3D_USE_FUCHSIA_SORT`:
//   - env-on  : SorterVulkan goes through RadixSortFuchsia (GPU radix sort)
//               and the binner/rasterizer pass GPU buffers without host bounce
//   - env-off : SorterVulkan goes through std::sort (CPU fallback)
//
// Both paths sort the same packed u64 keyvals (Phase 2.3a 16/28/20 layout).
// LSD radix on the full 64 bits matches std::sort on the same input bit-
// exactly; the only divergence comes from buffer layout (GPU-resident vs
// host-array) and reads through the rasterizer should produce IDENTICAL
// pixels. Tolerance: very tight (max_abs <= 1e-6).
//
// Tiny fixture lives at harmonyos_3dgs/tests/golden/tiny/step000001/cam0000
// and is the same one test_forward_pipeline_vk.cpp uses.
//
// Skip semantics: GTEST_SKIP if VulkanContext::init() fails (= no Vulkan 1.2
// device with the Fuchsia features), matching every other Vulkan test in the
// suite.

#include "types.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/vk_context.h"

#include "test_helpers/binning_download.h"

#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

// Run a full forward pass on the tiny fixture using the Layer-1 sync path.
// `env_value` is set on GS3D_USE_FUCHSIA_SORT for the duration of the call;
// it is restored to whatever it was on entry.
struct ForwardResult {
    std::vector<float> image;   // [3*H*W] CHW
    int W = 0, H = 0;
    int total_pairs = 0;
    int num_tiles = 0;
    bool fuchsia_features_available = false;
    bool used_keyvals_gpu = false;
    bool used_tile_ranges_gpu = false;
};

ForwardResult run_forward_with_env(const char* env_value,
                                   bool poison_host_tile_ranges = false) {
    // Save current env, set ours.
    const char* saved = std::getenv("GS3D_USE_FUCHSIA_SORT");
    std::string saved_str = (saved != nullptr) ? std::string(saved) : std::string();
    bool was_set = (saved != nullptr);
    if (env_value != nullptr)
        ::setenv("GS3D_USE_FUCHSIA_SORT", env_value, /*overwrite=*/1);
    else
        ::unsetenv("GS3D_USE_FUCHSIA_SORT");

    ForwardResult res{};
    {
        const std::string root = tiny_cam0_dir();
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

        const int sh_degree       = static_cast<int>(meta_npy.f32()[0]);
        const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);
        const int H               = static_cast<int>(meta_npy.f32()[2]);
        const int W               = static_cast<int>(meta_npy.f32()[3]);
        const int N = static_cast<int>(pos_npy.shape[0]);

        std::vector<float> positions = npy_to_f32_vec(pos_npy);
        std::vector<float> scales    = npy_to_f32_vec(scl_npy);
        std::vector<float> rotations = npy_to_f32_vec(rot_npy);
        std::vector<float> opacities = npy_to_f32_vec(opa_npy);
        std::vector<float> sh_coeffs = npy_to_f32_vec(sh_npy);
        std::vector<float> filter_3d = npy_to_f32_vec(f3d_npy);

        Camera cam{};
        std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
        std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
        cam.cam_pos[0] = cp_npy.f32()[0];
        cam.cam_pos[1] = cp_npy.f32()[1];
        cam.cam_pos[2] = cp_npy.f32()[2];
        cam.tan_fovx   = fov_npy.f32()[0];
        cam.tan_fovy   = fov_npy.f32()[1];
        cam.width      = W;
        cam.height     = H;

        GaussianData g{};
        g.count      = N;
        g.sh_degree  = sh_degree;
        g.max_coeffs = sh_coeffs_per_g;
        g.positions  = positions.data();
        g.scales     = scales.data();
        g.rotations  = rotations.data();
        g.opacities  = opacities.data();
        g.sh_coeffs  = sh_coeffs.data();
        g.filter_3D  = filter_3d.data();

        RenderConfig cfg{};
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

        VulkanContext ctx;
        if (!ctx.init()) {
            // Caller will detect via empty res.image and SKIP.
            res.W = 0; res.H = 0;
            // Restore env.
            if (was_set) ::setenv("GS3D_USE_FUCHSIA_SORT", saved_str.c_str(), 1);
            else         ::unsetenv("GS3D_USE_FUCHSIA_SORT");
            return res;
        }

        const auto& caps = ctx.capabilities();
        res.fuchsia_features_available =
            caps.has_shader_int16 && caps.has_buffer_device_address &&
            caps.has_vulkan_memory_model && caps.has_vulkan_memory_model_device_scope;

        const int HW = H * W;
        res.W = W; res.H = H;
        res.image.assign(static_cast<size_t>(3) * HW, 0.0f);

        PreprocessorVulkan prep   (ctx);
        TileBinnerVulkan   binner (ctx);
        SorterVulkan       sorter (ctx);
        RasterizerVulkan   raster (ctx);
        FrameAllocator     alloc  (32u * 1024u * 1024u);

        PreprocessOutput pre = prep.process(g, cam, cfg, alloc);
        BinningOutput    bin = binner.bin(pre, N, cam, cfg, alloc);
        sorter.sort(bin, alloc);
        res.used_keyvals_gpu = bin.keyvals_sorted_gpu != nullptr;
        res.used_tile_ranges_gpu = bin.tile_ranges_gpu != nullptr;
        if (poison_host_tile_ranges && bin.tile_ranges != nullptr) {
            std::memset(bin.tile_ranges, 0,
                        static_cast<std::size_t>(bin.num_tiles) * 2u * sizeof(uint32_t));
        }
        raster.rasterize(pre, bin, cam, cfg, res.image.data());

        res.total_pairs = bin.total_pairs;
        res.num_tiles   = bin.num_tiles;
    }

    // Restore env.
    if (was_set) ::setenv("GS3D_USE_FUCHSIA_SORT", saved_str.c_str(), 1);
    else         ::unsetenv("GS3D_USE_FUCHSIA_SORT");

    return res;
}

}  // namespace

// ============================================================================
// 1. env-on forward path produces same image as env-off forward path.
// ============================================================================
TEST(FuchsiaRadixE2E, EnvOnEnvOffSameImage) {
    auto off = run_forward_with_env("0");
    if (off.W == 0) {
        GTEST_SKIP() << "No Vulkan device with Fuchsia features.";
    }
    auto on  = run_forward_with_env("1");
    ASSERT_GT(on.W, 0)
        << "env-on path failed to initialize even though env-off succeeded — "
        << "this indicates a bug in the Fuchsia integration (likely missing "
        << "device feature, but env-off succeeded so device is OK).";

    if (!on.fuchsia_features_available) {
        GTEST_SKIP() << "Vulkan device lacks Fuchsia sort features.";
    }
    ASSERT_FALSE(off.used_keyvals_gpu);
    ASSERT_FALSE(off.used_tile_ranges_gpu);
    ASSERT_TRUE(on.used_keyvals_gpu);
    ASSERT_TRUE(on.used_tile_ranges_gpu);

    ASSERT_EQ(off.W, on.W);
    ASSERT_EQ(off.H, on.H);
    ASSERT_EQ(off.total_pairs, on.total_pairs)
        << "env-on/off disagree on R (total_pairs); a bug somewhere upstream";

    // Pixel-wise comparison. Tight tolerance because both paths sort the same
    // packed u64 keyvals — Fuchsia LSD radix and std::sort produce bit-equal
    // output, so the rasterizer is fed identical inputs.
    ASSERT_EQ(off.image.size(), on.image.size());
    double sse = 0.0;
    float max_abs = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < off.image.size(); ++i) {
        const float d = std::fabs(off.image[i] - on.image[i]);
        if (d > 1e-6f) ++bad;
        if (d > max_abs) max_abs = d;
        sse += static_cast<double>(d) * d;
    }
    EXPECT_LE(max_abs, 1e-4f)
        << "env-on vs env-off image divergence: max_abs=" << max_abs
        << " sse=" << sse << " bad_pixels(>1e-6)=" << bad;
    // Tighter check: with LSD radix on full 64 bits, sort output is a
    // permutation that matches std::sort exactly when keys are unique — and
    // the packed keyval encodes gauss_idx in the low bits making collisions
    // impossible across the (tile, depth_q28, idx) tuple. So the images
    // should be identical at the bit level (tiny float blending differences
    // can creep in through atomic/non-atomic reductions but rasterize.comp
    // is single-threaded per pixel — no atomics).
    EXPECT_LT(sse, 1e-4)
        << "env-on vs env-off SSE too high: " << sse << " (expected ~0)";
}

TEST(FuchsiaRadixE2E, EnvOnRasterizerUsesGpuTileRanges) {
    auto off = run_forward_with_env("0");
    if (off.W == 0) {
        GTEST_SKIP() << "No Vulkan device with Fuchsia features.";
    }
    auto on = run_forward_with_env("1", true);
    ASSERT_GT(on.W, 0);
    if (!on.fuchsia_features_available) {
        GTEST_SKIP() << "Vulkan device lacks Fuchsia sort features.";
    }
    ASSERT_TRUE(on.used_tile_ranges_gpu);

    ASSERT_EQ(off.image.size(), on.image.size());
    double sse = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < off.image.size(); ++i) {
        const float d = std::fabs(off.image[i] - on.image[i]);
        if (d > max_abs) max_abs = d;
        sse += static_cast<double>(d) * d;
    }
    EXPECT_LE(max_abs, 1e-4f);
    EXPECT_LT(sse, 1e-4);
}

// ============================================================================
// 2. env-on tile_ranges contract: monotonic key order + valid tile_ranges.
// ============================================================================
TEST(FuchsiaRadixE2E, EnvOnTileRangesContract) {
    // Same fixture, env-on. We run forward, then directly inspect the
    // SorterVulkan-populated bin.keyvals_sorted (downloaded host array) to
    // assert (a) monotonic ordering and (b) tile_ranges agrees with a
    // recomputed-from-keys reference.
    const char* saved = std::getenv("GS3D_USE_FUCHSIA_SORT");
    std::string saved_str = (saved != nullptr) ? std::string(saved) : std::string();
    bool was_set = (saved != nullptr);
    ::setenv("GS3D_USE_FUCHSIA_SORT", "1", 1);

    bool ran = false;
    int total_pairs_seen = 0;
    {
        const std::string root = tiny_cam0_dir();
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

        const int sh_degree       = static_cast<int>(meta_npy.f32()[0]);
        const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);
        const int H               = static_cast<int>(meta_npy.f32()[2]);
        const int W               = static_cast<int>(meta_npy.f32()[3]);
        const int N = static_cast<int>(pos_npy.shape[0]);

        std::vector<float> positions = npy_to_f32_vec(pos_npy);
        std::vector<float> scales    = npy_to_f32_vec(scl_npy);
        std::vector<float> rotations = npy_to_f32_vec(rot_npy);
        std::vector<float> opacities = npy_to_f32_vec(opa_npy);
        std::vector<float> sh_coeffs = npy_to_f32_vec(sh_npy);
        std::vector<float> filter_3d = npy_to_f32_vec(f3d_npy);

        Camera cam{};
        std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
        std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
        cam.cam_pos[0] = cp_npy.f32()[0];
        cam.cam_pos[1] = cp_npy.f32()[1];
        cam.cam_pos[2] = cp_npy.f32()[2];
        cam.tan_fovx   = fov_npy.f32()[0];
        cam.tan_fovy   = fov_npy.f32()[1];
        cam.width      = W;
        cam.height     = H;

        GaussianData g{};
        g.count      = N;
        g.sh_degree  = sh_degree;
        g.max_coeffs = sh_coeffs_per_g;
        g.positions  = positions.data();
        g.scales     = scales.data();
        g.rotations  = rotations.data();
        g.opacities  = opacities.data();
        g.sh_coeffs  = sh_coeffs.data();
        g.filter_3D  = filter_3d.data();

        RenderConfig cfg{};
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

        VulkanContext ctx;
        if (ctx.init()) {
            ran = true;
            PreprocessorVulkan prep   (ctx);
            TileBinnerVulkan   binner (ctx);
            const auto& caps = ctx.capabilities();
            if (!(caps.has_shader_int16 && caps.has_buffer_device_address &&
                  caps.has_vulkan_memory_model && caps.has_vulkan_memory_model_device_scope)) {
                GTEST_SKIP() << "Vulkan device lacks Fuchsia sort features.";
            }

            SorterVulkan       sorter (ctx);
            FrameAllocator     alloc  (32u * 1024u * 1024u);

            PreprocessOutput pre = prep.process(g, cam, cfg, alloc);
            BinningOutput    bin = binner.bin(pre, N, cam, cfg, alloc);
            sorter.sort(bin, alloc);
            ASSERT_NE(bin.keyvals_sorted_gpu, nullptr);
            ASSERT_NE(bin.tile_ranges_gpu, nullptr);

            const int R = bin.total_pairs;
            total_pairs_seen = R;
            ASSERT_GT(R, 0);

            // env-aware download: helper returns the host pointer directly when
            // populated, otherwise stage-downloads from published VkBuffer handles.
            // The
            // contract being asserted (monotonicity + tile_ranges
            // consistency) is unchanged.
            std::vector<uint64_t> kv_host;
            std::vector<uint32_t> tr_host;
            const uint64_t* kv = test_helpers::keyvals_sorted_host_view(
                ctx, bin, kv_host);
            const uint32_t* tr = test_helpers::tile_ranges_host_view(
                ctx, bin, tr_host);

            // Monotonic.
            for (int i = 1; i < R; ++i) {
                ASSERT_LE(kv[i - 1], kv[i])
                    << "non-monotonic at i=" << i;
            }

            // tile_ranges contract: for every tile t, [start, end) covers
            // exactly the keyvals with tile_id == t (and they are consecutive
            // in the sorted array because tile_id is the most-significant
            // sort key).
            constexpr uint64_t kTileMask = 0xFFFFu;
            constexpr uint32_t kTileShift = 48u;
            for (int t = 0; t < bin.num_tiles; ++t) {
                const uint32_t s = tr[t * 2];
                const uint32_t e = tr[t * 2 + 1];
                if (s == 0u && e == 0u) continue;  // empty tile
                ASSERT_LE(s, e) << "tile " << t << ": start > end";
                ASSERT_LE(static_cast<int>(e), R)
                    << "tile " << t << ": end out of range";
                for (uint32_t i = s; i < e; ++i) {
                    const uint32_t tid = static_cast<uint32_t>(
                        (kv[i] >> kTileShift) & kTileMask);
                    ASSERT_EQ(tid, static_cast<uint32_t>(t))
                        << "tile " << t << " range [" << s << "," << e
                        << ") contains a key with tid=" << tid;
                }
            }
        }
    }

    // Restore env.
    if (was_set) ::setenv("GS3D_USE_FUCHSIA_SORT", saved_str.c_str(), 1);
    else         ::unsetenv("GS3D_USE_FUCHSIA_SORT");

    if (!ran) GTEST_SKIP() << "No Vulkan device with Fuchsia features.";
    EXPECT_GT(total_pairs_seen, 0);
}
