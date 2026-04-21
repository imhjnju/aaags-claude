// test_rasterizer_backward_vulkan.cpp — SP-3 T22: RasterizerBackwardVulkan
// correctness test against the CPU backward on the real tiny golden fixture.
//
// Scene: N=103 Gaussians, W=64, H=64 (from tests/golden/tiny/step000001/cam0000/).
//
// Flow:
//   1. Load tiny golden fixture (input_*.npy + backward_dL_dout_color.npy).
//   2. CPU forward chain: PreprocessorCPU → TileBinnerCPU → SorterCPU →
//      RasterizerCPU  → populates ForwardCache (T_final, n_contrib, cov2D, …).
//   3. Transpose dL_dout_color from channel-major [3][H][W] to
//      pixel-major [H*W][3] (the CPU backward uses pixel-major indexing).
//      Evidence: npy shape is (3, 64, 64), CUDA dumps in CHW order.
//   4. CPU backward  (RasterizerBackwardCPU::backward) → rgrad_cpu.
//   5. Vulkan backward (RasterizerBackwardVulkan::backward) → rgrad_vk.
//   6. Assert max abs diff < 1e-4 on d_means2D, d_conics, d_opacities_2d, d_rgb.
//
// n_contrib convention: CPU and Vulkan forward pass both store count-based
// n_contrib (number of Gaussians that actually blended). The backward shader
// processes all Gaussians in the tile range and uses alpha/power threshold
// checks to skip non-contributors — matching the CPU backward forward-replay.
//
// GTEST_SKIP if no Vulkan compute device.

#include "types.h"
#include "train_types.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include "vulkan/vk_context.h"
#include "vulkan/rasterizer_backward_vulkan.h"

#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Fixture path (mirrors test_forward_pipeline_vk.cpp)
// ---------------------------------------------------------------------------

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST(RasterizerBackwardVulkan, MatchesCPU_TinyFixture) {
    // ---- 0. Vulkan device check --------------------------------------------
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    // ---- 1. Load golden inputs ---------------------------------------------
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
    auto dL_npy   = load_npy(root + "/backward_dL_dout_color.npy");

    // ---- 2. Parse meta -----------------------------------------------------
    ASSERT_EQ(meta_npy.shape.size(), 1u);
    ASSERT_EQ(meta_npy.shape[0], 4u);
    const int sh_degree       = static_cast<int>(meta_npy.f32()[0]);
    const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);
    const int H               = static_cast<int>(meta_npy.f32()[2]);
    const int W               = static_cast<int>(meta_npy.f32()[3]);
    ASSERT_GT(H, 0);
    ASSERT_GT(W, 0);

    // ---- 3. Shape checks ---------------------------------------------------
    const int N = static_cast<int>(pos_npy.shape[0]);
    ASSERT_GT(N, 0);
    ASSERT_EQ(pos_npy.shape[1], 3u);
    ASSERT_EQ(scl_npy.shape[0], static_cast<size_t>(N));
    ASSERT_EQ(rot_npy.shape[1], 4u);
    ASSERT_EQ(sh_npy.numel(),
              static_cast<size_t>(N) * sh_coeffs_per_g * 3);
    ASSERT_EQ(f3d_npy.numel(), static_cast<size_t>(N));
    ASSERT_EQ(vm_npy.numel(), 16u);
    ASSERT_EQ(pm_npy.numel(), 16u);
    ASSERT_EQ(dL_npy.dtype, NpyDtype::float32);
    // dL_dout_color is channel-major [3][H][W] from the CUDA dump.
    ASSERT_EQ(dL_npy.numel(), static_cast<size_t>(3) * H * W);

    // ---- 4. Copy npy payloads into std::vector (pointers outlive dispatch) -
    std::vector<float> positions = npy_to_f32_vec(pos_npy);
    std::vector<float> scales    = npy_to_f32_vec(scl_npy);
    std::vector<float> rotations = npy_to_f32_vec(rot_npy);
    std::vector<float> opacities = npy_to_f32_vec(opa_npy);
    std::vector<float> sh_coeffs = npy_to_f32_vec(sh_npy);
    std::vector<float> filter_3d = npy_to_f32_vec(f3d_npy);

    // ---- 5. Transpose dL_dout_color: channel-major → pixel-major -----------
    // CUDA dump layout: data[ch * H * W + pix_id] (channel-major CHW)
    // CPU/Vulkan backward expect pixel-major: data[pix * 3 + ch]
    // Evidence: npy shape is (3, 64, 64) — channel first.
    const int HW = H * W;
    std::vector<float> d_image(static_cast<size_t>(HW) * 3);
    {
        const float* src = dL_npy.f32();   // [3][H*W] in memory (C-major)
        for (int pix = 0; pix < HW; ++pix) {
            for (int ch = 0; ch < 3; ++ch) {
                d_image[pix * 3 + ch] = src[ch * HW + pix];
            }
        }
    }

    // ---- 6. Build Camera ---------------------------------------------------
    Camera cam{};
    std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
    std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
    cam.cam_pos[0] = cp_npy.f32()[0];
    cam.cam_pos[1] = cp_npy.f32()[1];
    cam.cam_pos[2] = cp_npy.f32()[2];
    cam.tan_fovx   = fov_npy.f32()[0];
    cam.tan_fovy   = fov_npy.f32()[1];
    cam.width      = static_cast<int>(fov_npy.f32()[2]);
    cam.height     = static_cast<int>(fov_npy.f32()[3]);
    ASSERT_EQ(cam.width,  W);
    ASSERT_EQ(cam.height, H);

    // ---- 7. Build GaussianData + RenderConfig ------------------------------
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

    // ---- 8. CPU forward chain → ForwardCache -------------------------------
    // Use 32 MB arena — N=103, W=64, H=64 is modest.
    FrameAllocator alloc(32u * 1024u * 1024u);

    ForwardCache cache{};

    PreprocessorCPU preprocessor;
    PreprocessOutput pre = preprocessor.process(g, cam, cfg, alloc, &cache);

    TileBinnerCPU binner;
    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);

    SorterCPU sorter;
    sorter.sort(bin, alloc);

    {
        float* buf = alloc.allocate_array<float>(HW * 3);
        std::memset(buf, 0, HW * 3 * sizeof(float));

        RasterizerCPU rast;
        rast.rasterize(pre, bin, cam, cfg, buf, nullptr, &cache, &alloc);
    }

    // Stitch pre/bin pointers into cache (required by the preprocessor backward;
    // the rasterizer backward uses bin directly but cache.pre/bin are needed if
    // upper layers inspect them).
    cache.pre = &pre;
    cache.bin = &bin;

    // ---- 9. CPU backward ---------------------------------------------------
    RasterGradOutput rgrad_cpu{};
    rgrad_cpu.allocate_and_zero(alloc, N);

    RasterizerBackwardCPU bwd_cpu;
    bwd_cpu.backward(pre, bin, cam, cfg, cache, d_image.data(), rgrad_cpu);

    // ---- 10. Vulkan backward -----------------------------------------------
    FrameAllocator vk_alloc(32u * 1024u * 1024u);
    RasterGradOutput rgrad_vk{};

    RasterizerBackwardVulkan bwd_vk(ctx);
    bwd_vk.backward(pre, bin, N, cam, cfg, cache, d_image.data(), rgrad_vk, vk_alloc);

    // ---- 11. Compare -------------------------------------------------------
    const float tol = 1e-4f;

    // dL_dmeans2D [N*2]
    for (int i = 0; i < N * 2; ++i) {
        float diff = std::fabs(rgrad_vk.d_means2D[i] - rgrad_cpu.d_means2D[i]);
        EXPECT_LT(diff, tol)
            << "d_means2D[" << i << "]: vk=" << rgrad_vk.d_means2D[i]
            << " cpu=" << rgrad_cpu.d_means2D[i];
    }

    // dL_dconics [N*3]
    for (int i = 0; i < N * 3; ++i) {
        float diff = std::fabs(rgrad_vk.d_conics[i] - rgrad_cpu.d_conics[i]);
        EXPECT_LT(diff, tol)
            << "d_conics[" << i << "]: vk=" << rgrad_vk.d_conics[i]
            << " cpu=" << rgrad_cpu.d_conics[i];
    }

    // dL_dopacity [N]
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(rgrad_vk.d_opacities_2d[i] - rgrad_cpu.d_opacities_2d[i]);
        EXPECT_LT(diff, tol)
            << "d_opacities_2d[" << i << "]: vk=" << rgrad_vk.d_opacities_2d[i]
            << " cpu=" << rgrad_cpu.d_opacities_2d[i];
    }

    // dL_dcolors [N*3]
    for (int i = 0; i < N * 3; ++i) {
        float diff = std::fabs(rgrad_vk.d_rgb[i] - rgrad_cpu.d_rgb[i]);
        EXPECT_LT(diff, tol)
            << "d_rgb[" << i << "]: vk=" << rgrad_vk.d_rgb[i]
            << " cpu=" << rgrad_cpu.d_rgb[i];
    }
}
