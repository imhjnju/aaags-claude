// test_rasterize_backward_pass_vk.cpp — SP-3 T22: RasterizeBackwardPass smoke
// test.
//
// N=4, W=64, H=64 (16 tiles of 16x16).
// Creates a minimal sorted list (2 pairs in tile 0), sets T_final=0.5 and
// n_contrib=2 for all pixels, dL_dpixels=1.0 everywhere.
// Dispatches the backward pass and checks that dL_dcolors is non-zero and
// finite, confirming the shader ran and produced output.

#include "vulkan/vk_context.h"
#include "vulkan/rasterize_backward_pass.h"
#include "vulkan/vk_buffer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

TEST(RasterizeBackwardPass, TinyFixture) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    // Scene parameters.
    const uint32_t W = 64;
    const uint32_t H = 64;
    const uint32_t N = 4;
    const uint32_t num_tiles_x = (W + 15) / 16;   // 4
    const uint32_t num_tiles_y = (H + 15) / 16;   // 4
    const uint32_t num_tiles   = num_tiles_x * num_tiles_y;  // 16
    const uint32_t HW          = H * W;

    // ---- Input data -------------------------------------------------------

    // tile_ranges: only tile 0 has 2 Gaussians (indices 0..1 in values_sorted).
    std::vector<uint32_t> tile_ranges(num_tiles * 2, 0u);
    tile_ranges[0] = 0u;  // tile 0 start
    tile_ranges[1] = 2u;  // tile 0 end

    // values_sorted: Gaussian indices 1 and 3 (back-to-front ordering).
    std::vector<uint32_t> values_sorted = {1u, 3u};
    const uint32_t R = static_cast<uint32_t>(values_sorted.size());

    // means2D: place Gaussians near center of tile 0.
    // Tile 0 covers pixels (0..15, 0..15). Place at (7, 7) and (5, 5).
    std::vector<float> means2D(N * 2, 0.f);
    means2D[1 * 2 + 0] = 7.f;   means2D[1 * 2 + 1] = 7.f;
    means2D[3 * 2 + 0] = 5.f;   means2D[3 * 2 + 1] = 5.f;

    // conic_opacity: identity-ish conic (a=0.1, b=0, c=0.1), opacity=0.6.
    std::vector<float> conic_opacity(N * 4, 0.f);
    for (uint32_t i = 0; i < N; ++i) {
        conic_opacity[i * 4 + 0] = 0.1f;  // a
        conic_opacity[i * 4 + 1] = 0.0f;  // b
        conic_opacity[i * 4 + 2] = 0.1f;  // c
        conic_opacity[i * 4 + 3] = 0.6f;  // opacity
    }

    // colors: distinct RGB per Gaussian.
    std::vector<float> colors(N * 3, 0.f);
    colors[0 * 3 + 0] = 0.9f; colors[0 * 3 + 1] = 0.1f; colors[0 * 3 + 2] = 0.2f;
    colors[1 * 3 + 0] = 0.3f; colors[1 * 3 + 1] = 0.8f; colors[1 * 3 + 2] = 0.1f;
    colors[2 * 3 + 0] = 0.2f; colors[2 * 3 + 1] = 0.3f; colors[2 * 3 + 2] = 0.9f;
    colors[3 * 3 + 0] = 0.7f; colors[3 * 3 + 1] = 0.2f; colors[3 * 3 + 2] = 0.5f;

    // T_final: set to 0.5 for all pixels.
    std::vector<float> T_final(HW, 0.5f);

    // n_contrib: set to 2 for pixels in tile 0 (0..15 x, 0..15 y), else 0.
    // This tells the shader to process both Gaussians in tile 0's backward.
    std::vector<uint32_t> n_contrib(HW, 0u);
    for (uint32_t py = 0; py < 16u; ++py) {
        for (uint32_t px = 0; px < 16u; ++px) {
            n_contrib[py * W + px] = 2u;
        }
    }

    // dL_dpixels: all 1.0 (pixel-major [H*W][3]).
    std::vector<float> dL_dpixels(HW * 3, 1.0f);

    // ---- Gradient output buffers (zeroed) -----------------------------------
    std::vector<float> zeros_m2d(N * 2, 0.f);
    std::vector<float> zeros_con(N * 3, 0.f);
    std::vector<float> zeros_opa(N, 0.f);
    std::vector<float> zeros_col(N * 3, 0.f);

    // ---- Allocate GPU buffers -----------------------------------------------
    auto tr_buf    = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(num_tiles * 2 * sizeof(uint32_t)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto vs_buf    = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(R * sizeof(uint32_t)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto m2d_buf   = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 2 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto co_buf    = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 4 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto col_buf   = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 3 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tf_buf    = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(HW * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto nc_buf    = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(HW * sizeof(uint32_t)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dlpix_buf = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(HW * 3 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    auto dlm2d_buf = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 2 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dlcon_buf = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 3 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dlopa_buf = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dlcol_buf = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(N * 3 * sizeof(float)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ubo_buf   = std::make_unique<VulkanBuffer>(
        ctx, static_cast<VkDeviceSize>(sizeof(RasterizeBackwardUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // ---- Upload inputs -------------------------------------------------------
    tr_buf   ->upload(tile_ranges.data(),    num_tiles * 2 * sizeof(uint32_t));
    vs_buf   ->upload(values_sorted.data(),  R * sizeof(uint32_t));
    m2d_buf  ->upload(means2D.data(),        N * 2 * sizeof(float));
    co_buf   ->upload(conic_opacity.data(),  N * 4 * sizeof(float));
    col_buf  ->upload(colors.data(),         N * 3 * sizeof(float));
    tf_buf   ->upload(T_final.data(),        HW * sizeof(float));
    nc_buf   ->upload(n_contrib.data(),      HW * sizeof(uint32_t));
    dlpix_buf->upload(dL_dpixels.data(),     HW * 3 * sizeof(float));

    // Zero gradient buffers explicitly (atomicAdd accumulates into them).
    dlm2d_buf->upload(zeros_m2d.data(),  N * 2 * sizeof(float));
    dlcon_buf->upload(zeros_con.data(),  N * 3 * sizeof(float));
    dlopa_buf->upload(zeros_opa.data(),  N * sizeof(float));
    dlcol_buf->upload(zeros_col.data(),  N * 3 * sizeof(float));

    // Upload UBO.
    RasterizeBackwardUBO ubo{};
    ubo.W           = W;
    ubo.H           = H;
    ubo.num_tiles_x = num_tiles_x;
    ubo.bg_color[0] = 0.f;
    ubo.bg_color[1] = 0.f;
    ubo.bg_color[2] = 0.f;
    ubo_buf->upload(&ubo, sizeof(ubo));

    // ---- Create pass and dispatch -------------------------------------------
    RasterizeBackwardPass pass(ctx);

    RasterizeBackwardPass::Buffers rb{};
    rb.tile_ranges   = tr_buf   ->handle();
    rb.values_sorted = vs_buf   ->handle();
    rb.means2D       = m2d_buf  ->handle();
    rb.conic_opacity = co_buf   ->handle();
    rb.colors        = col_buf  ->handle();
    rb.T_final       = tf_buf   ->handle();
    rb.n_contrib     = nc_buf   ->handle();
    rb.dL_dpixels    = dlpix_buf->handle();
    rb.dL_dmeans2D   = dlm2d_buf->handle();
    rb.dL_dconics    = dlcon_buf->handle();
    rb.dL_dopacity   = dlopa_buf->handle();
    rb.dL_dcolors    = dlcol_buf->handle();

    pass.bind_buffers(rb, ubo_buf->handle());
    pass.dispatch_sync(num_tiles_x, num_tiles_y);

    // ---- Download and validate ---------------------------------------------
    std::vector<float> out_dL_dcolors(N * 3, 0.f);
    dlcol_buf->download(out_dL_dcolors.data(), N * 3 * sizeof(float));

    // Gaussians 1 and 3 appear in values_sorted for tile 0.
    // With n_contrib=2 and T_final=0.5, their color gradients should be nonzero.
    // Gaussians 0 and 2 are not referenced — their gradients must stay zero.

    for (int ch = 0; ch < 3; ++ch) {
        EXPECT_EQ(out_dL_dcolors[0 * 3 + ch], 0.0f)
            << "Gaussian 0 not in values_sorted, ch=" << ch;
        EXPECT_EQ(out_dL_dcolors[2 * 3 + ch], 0.0f)
            << "Gaussian 2 not in values_sorted, ch=" << ch;
    }

    // Gaussians 1 and 3 must have non-zero, finite color gradients
    // (pixels in tile 0 have dL_dpix=1.0 and n_contrib=2).
    bool any_nonzero_1 = false;
    bool any_nonzero_3 = false;
    for (int ch = 0; ch < 3; ++ch) {
        float g1 = out_dL_dcolors[1 * 3 + ch];
        float g3 = out_dL_dcolors[3 * 3 + ch];
        EXPECT_TRUE(std::isfinite(g1)) << "dL_dcolors[1][" << ch << "] is not finite";
        EXPECT_TRUE(std::isfinite(g3)) << "dL_dcolors[3][" << ch << "] is not finite";
        if (g1 != 0.f) any_nonzero_1 = true;
        if (g3 != 0.f) any_nonzero_3 = true;
    }
    EXPECT_TRUE(any_nonzero_1) << "dL_dcolors for Gaussian 1 is all-zero";
    EXPECT_TRUE(any_nonzero_3) << "dL_dcolors for Gaussian 3 is all-zero";
}
