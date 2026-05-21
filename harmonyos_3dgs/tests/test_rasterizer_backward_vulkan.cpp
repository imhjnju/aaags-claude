// test_rasterizer_backward_vulkan.cpp — SP-3 T22: RasterizerBackwardVulkan
// correctness test against the CPU backward on the real tiny golden fixture.
//
// Scene: N=103 Gaussians, W=64, H=64 (from tests/golden/tiny/step000001/cam0000/).
//
// Flow:
//   1. Load tiny golden fixture (input_*.npy + backward_dL_dout_color.npy).
//   2. CPU forward chain: PreprocessorCPU → TileBinnerCPU → SorterCPU →
//      RasterizerCPU  → populates ForwardCache (T_final, n_contrib, cov2D, …).
//   3. Load dL_dout_color directly from npy (CHW [3][H][W], matching our
//      backward pass input format).
//   4. CPU backward  (RasterizerBackwardCPU::backward) → rgrad_cpu.
//   5. Vulkan backward (RasterizerBackwardVulkan::backward) → rgrad_vk.
//   6. Assert max abs diff < 1e-4 on d_means2D, d_conics, d_opacities_2d, d_rgb.
//
// n_contrib convention: CPU and Vulkan forward pass both store CUDA-style
// 1-based position of the last candidate that actually blended. Skipped
// candidates before that position still advance the replay position.
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
#include "vulkan/vk_buffer.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
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

    // Gradient data is CHW [3,H,W] — matches our backward pass input format.
    std::vector<float> d_image(dL_npy.f32(), dL_npy.f32() + dL_npy.numel());

    const int HW = H * W;

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

    // If a GPU loss-gradient buffer is present, it is the authoritative source
    // even when a stale non-null CPU gradient pointer is also passed.
    VulkanBuffer d_image_gpu(ctx,
        static_cast<VkDeviceSize>(d_image.size() * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    d_image_gpu.upload(d_image.data(), d_image.size() * sizeof(float));
    ForwardCache gpu_gradient_cache = cache;
    gpu_gradient_cache.dL_dpixels_gpu = d_image_gpu.handle();
    std::vector<float> stale_cpu_d_image(d_image.size(), 0.0f);
    FrameAllocator gpu_gradient_alloc(32u * 1024u * 1024u);
    RasterGradOutput rgrad_gpu_gradient{};
    bwd_vk.backward(pre, bin, N, cam, cfg, gpu_gradient_cache,
                    stale_cpu_d_image.data(), rgrad_gpu_gradient, gpu_gradient_alloc);

    for (int i = 0; i < N * 2; ++i) {
        float diff = std::fabs(rgrad_gpu_gradient.d_means2D[i] - rgrad_cpu.d_means2D[i]);
        EXPECT_LT(diff, tol)
            << "GPU dL_dpixels precedence d_means2D[" << i << "]: vk="
            << rgrad_gpu_gradient.d_means2D[i] << " cpu=" << rgrad_cpu.d_means2D[i];
    }
    for (int i = 0; i < N * 3; ++i) {
        float diff = std::fabs(rgrad_gpu_gradient.d_conics[i] - rgrad_cpu.d_conics[i]);
        EXPECT_LT(diff, tol)
            << "GPU dL_dpixels precedence d_conics[" << i << "]: vk="
            << rgrad_gpu_gradient.d_conics[i] << " cpu=" << rgrad_cpu.d_conics[i];
    }
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(rgrad_gpu_gradient.d_opacities_2d[i] - rgrad_cpu.d_opacities_2d[i]);
        EXPECT_LT(diff, tol)
            << "GPU dL_dpixels precedence d_opacities_2d[" << i << "]: vk="
            << rgrad_gpu_gradient.d_opacities_2d[i] << " cpu=" << rgrad_cpu.d_opacities_2d[i];
    }
    for (int i = 0; i < N * 3; ++i) {
        float diff = std::fabs(rgrad_gpu_gradient.d_rgb[i] - rgrad_cpu.d_rgb[i]);
        EXPECT_LT(diff, tol)
            << "GPU dL_dpixels precedence d_rgb[" << i << "]: vk="
            << rgrad_gpu_gradient.d_rgb[i] << " cpu=" << rgrad_cpu.d_rgb[i];
    }
}

TEST(RasterizerBackwardVulkan, Eval3DBackwardUsesPreprocessGpuHandles) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

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

    const int N = static_cast<int>(pos_npy.shape[0]);
    const int sh_degree = static_cast<int>(meta_npy.f32()[0]);
    const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);
    const int H = static_cast<int>(meta_npy.f32()[2]);
    const int W = static_cast<int>(meta_npy.f32()[3]);

    std::vector<float> positions = npy_to_f32_vec(pos_npy);
    std::vector<float> scales = npy_to_f32_vec(scl_npy);
    std::vector<float> rotations = npy_to_f32_vec(rot_npy);
    std::vector<float> opacities = npy_to_f32_vec(opa_npy);
    std::vector<float> sh_coeffs = npy_to_f32_vec(sh_npy);
    std::vector<float> filter_3d = npy_to_f32_vec(f3d_npy);
    std::vector<float> d_image(dL_npy.f32(), dL_npy.f32() + dL_npy.numel());

    Camera cam{};
    std::memcpy(cam.view_matrix, vm_npy.f32(), 16 * sizeof(float));
    std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
    cam.cam_pos[0] = cp_npy.f32()[0];
    cam.cam_pos[1] = cp_npy.f32()[1];
    cam.cam_pos[2] = cp_npy.f32()[2];
    cam.tan_fovx = fov_npy.f32()[0];
    cam.tan_fovy = fov_npy.f32()[1];
    cam.width = W;
    cam.height = H;

    GaussianData g{};
    g.count = N;
    g.sh_degree = sh_degree;
    g.max_coeffs = sh_coeffs_per_g;
    g.positions = positions.data();
    g.scales = scales.data();
    g.rotations = rotations.data();
    g.opacities = opacities.data();
    g.sh_coeffs = sh_coeffs.data();
    g.filter_3D = filter_3d.data();

    RenderConfig cfg{};
    cfg.sh_degree = sh_degree;
    cfg.training = true;
    cfg.eval_3D = true;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.scale_modifier = 1.0f;
    cfg.bg_color[0] = 0.25f;
    cfg.bg_color[1] = 0.5f;
    cfg.bg_color[2] = 0.75f;

    FrameAllocator alloc(96u * 1024u * 1024u);
    ForwardCache cache{};
    cache.T_final = alloc.allocate_array<float>(static_cast<size_t>(H) * W);
    cache.n_contrib = alloc.allocate_array<int>(static_cast<size_t>(H) * W);
    cache.retain_gpu_outputs = true;
    PreprocessorVulkan prep(ctx, /*eval_3D=*/true);
    PreprocessOutput pre = prep.process(g, cam, cfg, alloc, &cache);
    ASSERT_NE(pre.rgb_gpu, nullptr);
    ASSERT_NE(pre.conic_opacity_packed_gpu, nullptr);

    TileBinnerVulkan binner(ctx);
    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);
    if (bin.total_pairs == 0) GTEST_SKIP() << "Tiny eval_3D fixture produced no pairs";
    SorterVulkan sorter(ctx);
    sorter.sort(bin, alloc);

    std::vector<float> image(static_cast<size_t>(3) * H * W, 0.0f);
    RasterizerVulkan raster(ctx, /*eval_3D=*/true);
    raster.rasterize(pre, bin, cam, cfg, image.data(), nullptr, &cache, &alloc);

    PreprocessOutput fallback_pre = pre;
    fallback_pre.rgb_gpu = nullptr;
    fallback_pre.conic_opacity_packed_gpu = nullptr;

    FrameAllocator fallback_alloc(32u * 1024u * 1024u);
    RasterGradOutput fallback_grad{};
    RasterizerBackwardVulkan fallback_bwd(ctx);
    fallback_bwd.backward(fallback_pre, bin, N, cam, cfg, cache, d_image.data(), fallback_grad, fallback_alloc);

    std::fill(pre.rgb, pre.rgb + static_cast<size_t>(N) * 3u, 12345.0f);
    std::fill(pre.opacities_2d, pre.opacities_2d + static_cast<size_t>(N), 0.0f);
    std::fill(pre.conics, pre.conics + static_cast<size_t>(N) * 3u, 12345.0f);

    FrameAllocator gpu_alloc(32u * 1024u * 1024u);
    RasterGradOutput gpu_grad{};
    RasterizerBackwardVulkan gpu_bwd(ctx);
    gpu_bwd.backward(pre, bin, N, cam, cfg, cache, d_image.data(), gpu_grad, gpu_alloc);

    const float tol = 1e-4f;
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(gpu_grad.d_opacities_2d[i], fallback_grad.d_opacities_2d[i], tol)
            << "d_opacities_2d[" << i << "]";
    }
    for (int i = 0; i < N * 3; ++i) {
        EXPECT_NEAR(gpu_grad.d_rgb[i], fallback_grad.d_rgb[i], tol)
            << "d_rgb[" << i << "]";
    }
    for (int i = 0; i < N * 16; ++i) {
        EXPECT_NEAR(gpu_grad.d_gauss2screen[i], fallback_grad.d_gauss2screen[i], tol)
            << "d_gauss2screen[" << i << "]";
    }
}

TEST(RasterizerBackwardVulkan, RecordIntoZeroPairsPublishesZeroGradientBuffers) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const int N = 3;
    Camera cam{};
    cam.width = 16;
    cam.height = 16;

    RenderConfig cfg{};
    cfg.eval_3D = true;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    PreprocessOutput pre{};
    pre.num_gaussians = N;
    pre.eval_3D = true;

    BinningOutput bin{};
    bin.total_pairs = 0;
    bin.num_tiles = 1;

    ForwardCache cache{};
    float dL_dpixels[16 * 16 * 3] = {};

    RasterizerBackwardVulkan bwd_vk(ctx);
    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ASSERT_EQ(vkBeginCommandBuffer(cmd, &bi), VK_SUCCESS);
    bwd_vk.backward_record_into(cmd, pre, bin, N, cam, cfg, cache, dL_dpixels);
    ASSERT_EQ(vkEndCommandBuffer(cmd), VK_SUCCESS);
    ctx.freePrimary(cmd);

    EXPECT_NE(bwd_vk.dL_dmeans2D_buf(), VK_NULL_HANDLE);
    EXPECT_NE(bwd_vk.dL_dconics_buf(), VK_NULL_HANDLE);
    EXPECT_NE(bwd_vk.dL_dopacity_buf(), VK_NULL_HANDLE);
    EXPECT_NE(bwd_vk.dL_dcolors_buf(), VK_NULL_HANDLE);
    EXPECT_NE(bwd_vk.dL_dgauss2screen_buf(), VK_NULL_HANDLE);

    std::vector<float> d_means2D;
    std::vector<float> d_conics;
    std::vector<float> d_opacity;
    std::vector<float> d_rgb;
    std::vector<float> d_gauss2screen;
    bwd_vk.download_outputs(N, d_means2D, d_conics, d_opacity, d_rgb, &d_gauss2screen);
    for (float v : d_means2D) EXPECT_EQ(v, 0.0f);
    for (float v : d_conics) EXPECT_EQ(v, 0.0f);
    for (float v : d_opacity) EXPECT_EQ(v, 0.0f);
    for (float v : d_rgb) EXPECT_EQ(v, 0.0f);
    for (float v : d_gauss2screen) EXPECT_EQ(v, 0.0f);
}
