// SP-3 T22: RasterizerBackwardVulkan — high-level backward rasterizer adapter.
//
// Uploads all CPU-side inputs to host-visible SSBOs, zero-fills gradient
// output buffers explicitly (VulkanBuffer does not zero on allocation),
// dispatches rasterize_backward.comp via RasterizeBackwardPass, then
// downloads the resulting gradients into the caller-provided rgrad arrays.
//
// Buffer-size contract (matches the shader std430 layouts):
//   tile_ranges    : num_tiles * 2 * sizeof(uint32)
//   values_sorted  : R * sizeof(uint32)
//   means2D        : N * 2 * sizeof(float)
//   conic_opacity  : N * 4 * sizeof(float)  (packed {a,b,c,opacity})
//   colors         : N * 3 * sizeof(float)
//   T_final        : H*W * sizeof(float)
//   n_contrib      : H*W * sizeof(uint32)
//   dL_dpixels     : 3 * H*W * sizeof(float)  CHW channel-first
//   dL_dmeans2D    : N * 2 * sizeof(float)    zeroed
//   dL_dconics     : N * 3 * sizeof(float)    zeroed
//   dL_dopacity    : N * sizeof(float)        zeroed
//   dL_dcolors     : N * 3 * sizeof(float)    zeroed
//   raster_bwd_ubo : sizeof(RasterizeBackwardUBO) = 32  UNIFORM_BUFFER

#include "vulkan/rasterizer_backward_vulkan.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/backward_bindings.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

RasterizerBackwardVulkan::RasterizerBackwardVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    pass_ = std::make_unique<RasterizeBackwardPass>(ctx_);
}

RasterizerBackwardVulkan::~RasterizerBackwardVulkan() = default;

void RasterizerBackwardVulkan::prepare_for_n(int N, int R, int num_tiles, int HW) {
    if (N <= buf_N_ && R <= buf_R_ && num_tiles <= buf_num_tiles_ && HW <= buf_HW_)
        return;

    const int N_new  = std::max(N,  buf_N_);
    const int R_new  = std::max(R,  buf_R_);
    const int T_new  = std::max(num_tiles, buf_num_tiles_);
    const int HW_new = std::max(HW, buf_HW_);

    tr_buf_    = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(T_new) * 2u * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    vs_buf_    = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(R_new) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m2d_buf_   = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 2u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    co_buf_    = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 4u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    col_buf_   = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 3u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    tf_buf_    = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(HW_new) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    nc_buf_    = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(HW_new) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlpix_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(HW_new) * 3u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlm2d_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 2u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlcon_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 3u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlopa_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlcol_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 3u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ubo_buf_   = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(sizeof(RasterizeBackwardUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    buf_N_ = N_new; buf_R_ = R_new;
    buf_num_tiles_ = T_new; buf_HW_ = HW_new;
}

void RasterizerBackwardVulkan::backward(const PreprocessOutput& pre,
                                         const BinningOutput& bin,
                                         int num_gaussians,
                                         const Camera& cam,
                                         const RenderConfig& cfg,
                                         const ForwardCache& cache,
                                         const float* dL_dpixels,
                                         RasterGradOutput& rgrad,
                                         FrameAllocator& alloc) {
    // --- Guard: eval_3D path not implemented -------------------------------
    if (pre.eval_3D)
        throw std::runtime_error(
            "RasterizerBackwardVulkan::backward: eval_3D=true is not supported");

    const int W  = cam.width;
    const int H  = cam.height;
    if (W <= 0 || H <= 0) return;

    // Use the caller-provided N directly. Deriving N from max(values_sorted)+1
    // is UB when Gaussians are culled: a culled Gaussian is absent from
    // values_sorted but the caller still allocates rgrad for all N Gaussians.
    const int N = num_gaussians;

    // Allocate and zero rgrad output arrays. This also gives us the
    // CPU-side destination for the downloaded gradients.
    rgrad.allocate_and_zero(alloc, N);

    if (N == 0 || bin.total_pairs <= 0 || bin.values_sorted == nullptr) {
        // Empty scene: no Gaussians, nothing to back-propagate.
        return;
    }

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + cfg.tile_w - 1) / cfg.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + cfg.tile_h - 1) / cfg.tile_h);

    const int R   = bin.total_pairs;
    const int HW  = H * W;

    // Ensure persistent buffers are large enough for this call.
    prepare_for_n(N, R, static_cast<int>(num_tiles_x * num_tiles_y), HW);

    // -------------------------------------------------------------------
    // Compute upload sizes (in bytes, for current call dimensions).
    // -------------------------------------------------------------------
    const VkDeviceSize bytes_tile_ranges =
        static_cast<VkDeviceSize>(bin.num_tiles) * 2u * sizeof(uint32_t);
    const VkDeviceSize bytes_vs =
        static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_m2d =
        static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_co =
        static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_colors =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal =
        static_cast<VkDeviceSize>(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib =
        static_cast<VkDeviceSize>(HW) * sizeof(uint32_t);
    const VkDeviceSize bytes_dL_dpix =
        static_cast<VkDeviceSize>(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_m2d =
        static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_dL_con =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_opa =
        static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_dL_col =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);

    // -------------------------------------------------------------------
    // Pack conic + opacity into a single interleaved buffer [N*4]:
    //   {a=conic[0], b=conic[1], c=conic[2], opacity=opacities_2d[i]}
    // -------------------------------------------------------------------
    std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N) * 4u);
    for (int i = 0; i < N; ++i) {
        const std::size_t dst = static_cast<std::size_t>(i) * 4u;
        const std::size_t src = static_cast<std::size_t>(i) * 3u;
        conic_opacity_packed[dst + 0] = pre.conics[src + 0];
        conic_opacity_packed[dst + 1] = pre.conics[src + 1];
        conic_opacity_packed[dst + 2] = pre.conics[src + 2];
        conic_opacity_packed[dst + 3] = pre.opacities_2d[i];
    }

    // -------------------------------------------------------------------
    // Upload inputs.
    // -------------------------------------------------------------------
    tr_buf_ ->upload(bin.tile_ranges,
                     static_cast<std::size_t>(bytes_tile_ranges));
    vs_buf_ ->upload(bin.values_sorted,
                     static_cast<std::size_t>(bytes_vs));
    m2d_buf_->upload(pre.means2D,
                     static_cast<std::size_t>(bytes_m2d));
    co_buf_ ->upload(conic_opacity_packed.data(),
                     static_cast<std::size_t>(bytes_co));
    col_buf_->upload(pre.rgb,
                     static_cast<std::size_t>(bytes_colors));
    tf_buf_ ->upload(cache.T_final,
                     static_cast<std::size_t>(bytes_tfinal));
    // n_contrib is int*; we reinterpret as uint32 (same width, counts < 2^31).
    nc_buf_ ->upload(cache.n_contrib,
                     static_cast<std::size_t>(bytes_ncontrib));
    dlpix_buf_->upload(dL_dpixels,
                       static_cast<std::size_t>(bytes_dL_dpix));

    // Zero-fill gradient output buffers (atomicAdd accumulates into them).
    {
        const std::vector<float> zeros_m2d(
            static_cast<std::size_t>(N) * 2u, 0.0f);
        dlm2d_buf_->upload(zeros_m2d.data(),
                           static_cast<std::size_t>(bytes_dL_m2d));

        const std::vector<float> zeros_con(
            static_cast<std::size_t>(N) * 3u, 0.0f);
        dlcon_buf_->upload(zeros_con.data(),
                           static_cast<std::size_t>(bytes_dL_con));

        const std::vector<float> zeros_opa(
            static_cast<std::size_t>(N), 0.0f);
        dlopa_buf_->upload(zeros_opa.data(),
                           static_cast<std::size_t>(bytes_dL_opa));

        const std::vector<float> zeros_col(
            static_cast<std::size_t>(N) * 3u, 0.0f);
        dlcol_buf_->upload(zeros_col.data(),
                           static_cast<std::size_t>(bytes_dL_col));
    }

    // Upload UBO.
    RasterizeBackwardUBO ubo{};
    ubo.W           = static_cast<uint32_t>(W);
    ubo.H           = static_cast<uint32_t>(H);
    ubo.num_tiles_x = num_tiles_x;
    ubo._pad        = 0u;
    ubo.bg_color[0] = cfg.bg_color[0];
    ubo.bg_color[1] = cfg.bg_color[1];
    ubo.bg_color[2] = cfg.bg_color[2];
    ubo._pad2       = 0.f;
    ubo_buf_->upload(&ubo, sizeof(ubo));

    // -------------------------------------------------------------------
    // Bind and dispatch.
    // -------------------------------------------------------------------
    RasterizeBackwardPass::Buffers rb{};
    rb.tile_ranges   = tr_buf_   ->handle();
    rb.values_sorted = vs_buf_   ->handle();
    rb.means2D       = m2d_buf_  ->handle();
    rb.conic_opacity = co_buf_   ->handle();
    rb.colors        = col_buf_  ->handle();
    rb.T_final       = tf_buf_   ->handle();
    rb.n_contrib     = nc_buf_   ->handle();
    rb.dL_dpixels    = dlpix_buf_->handle();
    rb.dL_dmeans2D   = dlm2d_buf_->handle();
    rb.dL_dconics    = dlcon_buf_->handle();
    rb.dL_dopacity   = dlopa_buf_->handle();
    rb.dL_dcolors    = dlcol_buf_->handle();

    pass_->bind_buffers(rb, ubo_buf_->handle());
    pass_->dispatch_sync(num_tiles_x, num_tiles_y);

    // -------------------------------------------------------------------
    // Download gradients into rgrad arrays.
    // -------------------------------------------------------------------
    dlm2d_buf_->download(rgrad.d_means2D,
                         static_cast<std::size_t>(bytes_dL_m2d));
    dlcon_buf_->download(rgrad.d_conics,
                         static_cast<std::size_t>(bytes_dL_con));
    dlopa_buf_->download(rgrad.d_opacities_2d,
                         static_cast<std::size_t>(bytes_dL_opa));
    dlcol_buf_->download(rgrad.d_rgb,
                         static_cast<std::size_t>(bytes_dL_col));
}

void RasterizerBackwardVulkan::download_outputs(
    int N,
    std::vector<float>& d_means2D,
    std::vector<float>& d_conics,
    std::vector<float>& d_opacity,
    std::vector<float>& d_rgb) const
{
    d_means2D.assign(static_cast<std::size_t>(N) * 2u, 0.0f);
    d_conics.assign(static_cast<std::size_t>(N) * 3u, 0.0f);
    d_opacity.assign(static_cast<std::size_t>(N), 0.0f);
    d_rgb.assign(static_cast<std::size_t>(N) * 3u, 0.0f);
    if (N == 0 || !dlm2d_buf_) return;

    dlm2d_buf_->download(d_means2D.data(), d_means2D.size() * sizeof(float));
    dlcon_buf_->download(d_conics.data(),  d_conics.size()  * sizeof(float));
    dlopa_buf_->download(d_opacity.data(), d_opacity.size() * sizeof(float));
    dlcol_buf_->download(d_rgb.data(),     d_rgb.size()     * sizeof(float));
}

void RasterizerBackwardVulkan::backward_record_into(VkCommandBuffer cmd,
                                                     const PreprocessOutput& pre,
                                                     const BinningOutput& bin,
                                                     int num_gaussians,
                                                     const Camera& cam,
                                                     const RenderConfig& cfg,
                                                     const ForwardCache& cache,
                                                     const float* dL_dpixels) {
    // --- Guard: eval_3D path not implemented -------------------------------
    if (pre.eval_3D)
        throw std::runtime_error(
            "RasterizerBackwardVulkan::backward_record_into: eval_3D=true is not supported");

    const int W  = cam.width;
    const int H  = cam.height;
    if (W <= 0 || H <= 0) return;

    const int N = num_gaussians;

    if (N == 0 || bin.total_pairs <= 0 || bin.values_sorted == nullptr) {
        // Empty scene: no Gaussians, nothing to back-propagate.
        return;
    }

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + cfg.tile_w - 1) / cfg.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + cfg.tile_h - 1) / cfg.tile_h);

    const int R   = bin.total_pairs;
    const int HW  = H * W;

    // Ensure persistent buffers are large enough for this call.
    prepare_for_n(N, R, static_cast<int>(num_tiles_x * num_tiles_y), HW);

    // -------------------------------------------------------------------
    // Compute upload sizes (in bytes, for current call dimensions).
    // -------------------------------------------------------------------
    const VkDeviceSize bytes_tile_ranges =
        static_cast<VkDeviceSize>(bin.num_tiles) * 2u * sizeof(uint32_t);
    const VkDeviceSize bytes_vs =
        static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_m2d =
        static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_co =
        static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_colors =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal =
        static_cast<VkDeviceSize>(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib =
        static_cast<VkDeviceSize>(HW) * sizeof(uint32_t);
    const VkDeviceSize bytes_dL_dpix =
        static_cast<VkDeviceSize>(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_m2d =
        static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_dL_con =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_opa =
        static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_dL_col =
        static_cast<VkDeviceSize>(N) * 3u * sizeof(float);

    // -------------------------------------------------------------------
    // Pack conic + opacity into a single interleaved buffer [N*4]:
    //   {a=conic[0], b=conic[1], c=conic[2], opacity=opacities_2d[i]}
    // -------------------------------------------------------------------
    std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N) * 4u);
    for (int i = 0; i < N; ++i) {
        const std::size_t dst = static_cast<std::size_t>(i) * 4u;
        const std::size_t src = static_cast<std::size_t>(i) * 3u;
        conic_opacity_packed[dst + 0] = pre.conics[src + 0];
        conic_opacity_packed[dst + 1] = pre.conics[src + 1];
        conic_opacity_packed[dst + 2] = pre.conics[src + 2];
        conic_opacity_packed[dst + 3] = pre.opacities_2d[i];
    }

    // -------------------------------------------------------------------
    // Upload inputs.
    // -------------------------------------------------------------------
    tr_buf_ ->upload(bin.tile_ranges,
                     static_cast<std::size_t>(bytes_tile_ranges));
    vs_buf_ ->upload(bin.values_sorted,
                     static_cast<std::size_t>(bytes_vs));
    m2d_buf_->upload(pre.means2D,
                     static_cast<std::size_t>(bytes_m2d));
    co_buf_ ->upload(conic_opacity_packed.data(),
                     static_cast<std::size_t>(bytes_co));
    col_buf_->upload(pre.rgb,
                     static_cast<std::size_t>(bytes_colors));
    tf_buf_ ->upload(cache.T_final,
                     static_cast<std::size_t>(bytes_tfinal));
    // n_contrib is int*; we reinterpret as uint32 (same width, counts < 2^31).
    nc_buf_ ->upload(cache.n_contrib,
                     static_cast<std::size_t>(bytes_ncontrib));
    dlpix_buf_->upload(dL_dpixels,
                       static_cast<std::size_t>(bytes_dL_dpix));

    // Zero-fill gradient output buffers (atomicAdd accumulates into them).
    {
        const std::vector<float> zeros_m2d(
            static_cast<std::size_t>(N) * 2u, 0.0f);
        dlm2d_buf_->upload(zeros_m2d.data(),
                           static_cast<std::size_t>(bytes_dL_m2d));

        const std::vector<float> zeros_con(
            static_cast<std::size_t>(N) * 3u, 0.0f);
        dlcon_buf_->upload(zeros_con.data(),
                           static_cast<std::size_t>(bytes_dL_con));

        const std::vector<float> zeros_opa(
            static_cast<std::size_t>(N), 0.0f);
        dlopa_buf_->upload(zeros_opa.data(),
                           static_cast<std::size_t>(bytes_dL_opa));

        const std::vector<float> zeros_col(
            static_cast<std::size_t>(N) * 3u, 0.0f);
        dlcol_buf_->upload(zeros_col.data(),
                           static_cast<std::size_t>(bytes_dL_col));
    }

    // Upload UBO.
    RasterizeBackwardUBO ubo{};
    ubo.W           = static_cast<uint32_t>(W);
    ubo.H           = static_cast<uint32_t>(H);
    ubo.num_tiles_x = num_tiles_x;
    ubo._pad        = 0u;
    ubo.bg_color[0] = cfg.bg_color[0];
    ubo.bg_color[1] = cfg.bg_color[1];
    ubo.bg_color[2] = cfg.bg_color[2];
    ubo._pad2       = 0.f;
    ubo_buf_->upload(&ubo, sizeof(ubo));

    // -------------------------------------------------------------------
    // Bind and record into cmd.
    // -------------------------------------------------------------------
    RasterizeBackwardPass::Buffers rb{};
    rb.tile_ranges   = tr_buf_   ->handle();
    rb.values_sorted = vs_buf_   ->handle();
    rb.means2D       = m2d_buf_  ->handle();
    rb.conic_opacity = co_buf_   ->handle();
    rb.colors        = col_buf_  ->handle();
    rb.T_final       = tf_buf_   ->handle();
    rb.n_contrib     = nc_buf_   ->handle();
    rb.dL_dpixels    = dlpix_buf_->handle();
    rb.dL_dmeans2D   = dlm2d_buf_->handle();
    rb.dL_dconics    = dlcon_buf_->handle();
    rb.dL_dopacity   = dlopa_buf_->handle();
    rb.dL_dcolors    = dlcol_buf_->handle();

    pass_->bind_buffers(rb, ubo_buf_->handle());
    pass_->record(cmd, num_tiles_x, num_tiles_y);
}
