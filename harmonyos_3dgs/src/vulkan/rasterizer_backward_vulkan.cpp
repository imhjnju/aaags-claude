// SP-3 T22: RasterizerBackwardVulkan — high-level backward rasterizer adapter.
//
// Uses GPU-resident forward/preprocess buffers when supplied, uploads CPU
// fallbacks to host-visible SSBOs, zero-fills gradient output buffers explicitly
// (VulkanBuffer does not zero on allocation), dispatches rasterize_backward.comp
// via RasterizeBackwardPass, then downloads gradients for synchronous callers.
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
#include "camera_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool tangent_rot_bwd_enabled() {
    const char* v = std::getenv("GS3D_EVAL3D_TANGENT_ROT_BWD");
    return v && v[0] == '1' && v[1] == '\0';
}

bool tangent_subgroup_bwd_enabled() {
    const char* v = std::getenv("GS3D_EVAL3D_TANGENT_SUBGROUP_BWD");
    return v && v[0] == '1' && v[1] == '\0';
}

bool hot_gid_shard_bwd_enabled() {
    return false;
}

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (!end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) return fallback;
    return static_cast<uint32_t>(parsed);
}

uint32_t floor_power_of_two(uint32_t v) {
    if (v == 0u) return 0u;
    uint32_t p = 1u;
    while (p <= v / 2u) p <<= 1u;
    return p;
}

}  // namespace

RasterizerBackwardVulkan::RasterizerBackwardVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    pass_ = std::make_unique<RasterizeBackwardPass>(ctx_);
    eval3d_pass_ = std::make_unique<RasterizeBackwardEval3DPass>(ctx_);
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
    img_buf_   = std::make_unique<VulkanBuffer>(ctx_,
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
    g2s_buf_   = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 16u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dlg2s_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 16u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    replay_offsets_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        (static_cast<VkDeviceSize>(HW_new) + 1u) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    drot_tangent_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 3u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    hot_lookup_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    hot_lookup_capacity_ = static_cast<size_t>(N_new);
    hot_ubo_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(sizeof(RasterizeBackwardEval3DHotUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    dummy4_buf_ = std::make_unique<VulkanBuffer>(ctx_, 4u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ubo_buf_   = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(sizeof(RasterizeBackwardUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    fused_ubo_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(sizeof(RasterizeBackwardEval3DFusedUBO)),
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
                                         FrameAllocator& alloc,
                                         const float* rendered_image) {
    last_forward_gpu_cache_used_ = false;
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

    if (N == 0 || bin.total_pairs <= 0 || (bin.values_sorted == nullptr && bin.values_sorted_gpu == nullptr)) {
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
    const VkDeviceSize bytes_dL_g2s =
        static_cast<VkDeviceSize>(N) * 16u * sizeof(float);
    const bool has_values_gpu = bin.values_sorted_gpu != nullptr;

    if (pre.eval_3D) {
        const bool has_gauss2screen_gpu = cache.gauss2screen_gpu != nullptr;
        const bool has_rgb_gpu = pre.rgb_gpu != nullptr;
        const bool has_conic_opacity_gpu = pre.conic_opacity_packed_gpu != nullptr;
        if (!has_gauss2screen_gpu && !pre.gauss2screen)
            throw std::runtime_error("RasterizerBackwardVulkan::backward: missing eval_3D gauss2screen");
        if (bin.tile_ranges_gpu == nullptr) {
            tr_buf_->upload(bin.tile_ranges, static_cast<std::size_t>(bytes_tile_ranges));
        }
        if (!has_values_gpu) {
            vs_buf_->upload(bin.values_sorted, static_cast<std::size_t>(bytes_vs));
        }
        VkBuffer gauss2screen_handle = has_gauss2screen_gpu
            ? static_cast<VkBuffer>(cache.gauss2screen_gpu)
            : g2s_buf_->handle();
        if (!has_gauss2screen_gpu) {
            g2s_buf_->upload(pre.gauss2screen, static_cast<std::size_t>(bytes_dL_g2s));
        }
        VkBuffer colors_handle = has_rgb_gpu
            ? static_cast<VkBuffer>(pre.rgb_gpu)
            : col_buf_->handle();
        if (!has_rgb_gpu) {
            col_buf_->upload(pre.rgb, static_cast<std::size_t>(bytes_colors));
        }
        VkBuffer tfinal_handle = cache.T_final_gpu
            ? static_cast<VkBuffer>(cache.T_final_gpu)
            : tf_buf_->handle();
        VkBuffer ncontrib_handle = cache.n_contrib_gpu
            ? static_cast<VkBuffer>(cache.n_contrib_gpu)
            : nc_buf_->handle();
        VkBuffer dlpix_handle = cache.dL_dpixels_gpu
            ? static_cast<VkBuffer>(cache.dL_dpixels_gpu)
            : dlpix_buf_->handle();
        last_forward_gpu_cache_used_ = cache.T_final_gpu != nullptr
            || cache.n_contrib_gpu != nullptr
            || has_gauss2screen_gpu
            || has_rgb_gpu
            || has_conic_opacity_gpu;
        if (!cache.T_final_gpu) {
            tf_buf_->upload(cache.T_final, static_cast<std::size_t>(bytes_tfinal));
        }
        if (!cache.n_contrib_gpu) {
            nc_buf_->upload(cache.n_contrib, static_cast<std::size_t>(bytes_ncontrib));
        }
        if (!cache.dL_dpixels_gpu) {
            if (dL_dpixels == nullptr) {
                throw std::runtime_error("RasterizerBackwardVulkan: missing dL_dpixels");
            }
            dlpix_buf_->upload(dL_dpixels, static_cast<std::size_t>(bytes_dL_dpix));
        }

        VkBuffer conic_opacity_handle = has_conic_opacity_gpu
            ? static_cast<VkBuffer>(pre.conic_opacity_packed_gpu)
            : co_buf_->handle();
        if (!has_conic_opacity_gpu) {
            std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N) * 4u, 0.0f);
            for (int i = 0; i < N; ++i) {
                conic_opacity_packed[static_cast<std::size_t>(i) * 4u + 3u] = pre.opacities_2d[i];
            }
            co_buf_->upload(conic_opacity_packed.data(), static_cast<std::size_t>(bytes_co));
        }

        dlopa_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_opa));
        dlcol_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_col));
        dlg2s_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_g2s));

        const bool has_replay_cpu = cache.replay_order_offsets && cache.replay_order_gids && cache.replay_order_count > 0;
        const bool has_replay_gpu = cache.replay_order_offsets_gpu && cache.replay_order_gids_gpu && cache.replay_order_count > 0;
        const bool use_replay_order = has_replay_cpu || has_replay_gpu;
        VkBuffer replay_offsets_handle = dummy4_buf_->handle();
        VkBuffer replay_gids_handle = dummy4_buf_->handle();
        if (has_replay_cpu) {
            replay_offsets_buf_->upload(cache.replay_order_offsets,
                (static_cast<size_t>(HW) + 1u) * sizeof(uint32_t));
            if (cache.replay_order_count > buf_replay_count_) {
                replay_gids_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                    static_cast<VkDeviceSize>(cache.replay_order_count) * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                buf_replay_count_ = cache.replay_order_count;
            }
            replay_gids_buf_->upload(cache.replay_order_gids,
                static_cast<size_t>(cache.replay_order_count) * sizeof(uint32_t));
            replay_offsets_handle = replay_offsets_buf_->handle();
            replay_gids_handle = replay_gids_buf_->handle();
        } else if (has_replay_gpu) {
            replay_offsets_handle = static_cast<VkBuffer>(cache.replay_order_offsets_gpu);
            replay_gids_handle = static_cast<VkBuffer>(cache.replay_order_gids_gpu);
        }

        RasterizeBackwardUBO ubo{};
        ubo.W = static_cast<uint32_t>(W);
        ubo.H = static_cast<uint32_t>(H);
        ubo.num_tiles_x = num_tiles_x;
        ubo._pad = use_replay_order ? 1u : 0u;
        ubo.bg_color[0] = cfg.bg_color[0];
        ubo.bg_color[1] = cfg.bg_color[1];
        ubo.bg_color[2] = cfg.bg_color[2];
        ubo_buf_->upload(&ubo, sizeof(ubo));

        RasterizeBackwardEval3DPass::Buffers rb{};
        rb.tile_ranges = bin.tile_ranges_gpu
            ? static_cast<VkBuffer>(bin.tile_ranges_gpu)
            : tr_buf_->handle();
        rb.values_sorted = has_values_gpu
            ? static_cast<VkBuffer>(bin.values_sorted_gpu)
            : vs_buf_->handle();
        rb.gauss2screen = gauss2screen_handle;
        rb.conic_opacity = conic_opacity_handle;
        rb.colors = colors_handle;
        rb.T_final = tfinal_handle;
        rb.n_contrib = ncontrib_handle;
        rb.rendered_image = dummy4_buf_->handle();
        rb.dL_dpixels = dlpix_handle;
        rb.dL_dgauss2screen = dlg2s_buf_->handle();
        rb.dL_dopacity = dlopa_buf_->handle();
        rb.dL_dcolors = dlcol_buf_->handle();
        rb.replay_order_offsets = replay_offsets_handle;
        rb.replay_order_gids = replay_gids_handle;
        eval3d_pass_->bind_buffers(rb, ubo_buf_->handle());
        eval3d_pass_->dispatch_sync(num_tiles_x, num_tiles_y, use_replay_order);

        dlopa_buf_->download(rgrad.d_opacities_2d, static_cast<std::size_t>(bytes_dL_opa));
        dlcol_buf_->download(rgrad.d_rgb, static_cast<std::size_t>(bytes_dL_col));
        dlg2s_buf_->download(rgrad.d_gauss2screen, static_cast<std::size_t>(bytes_dL_g2s));
        return;
    }

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
    if (bin.tile_ranges_gpu == nullptr) {
        tr_buf_->upload(bin.tile_ranges,
                        static_cast<std::size_t>(bytes_tile_ranges));
    }
    if (!has_values_gpu) {
        vs_buf_->upload(bin.values_sorted,
                        static_cast<std::size_t>(bytes_vs));
    }
    m2d_buf_->upload(pre.means2D,
                     static_cast<std::size_t>(bytes_m2d));
    co_buf_ ->upload(conic_opacity_packed.data(),
                     static_cast<std::size_t>(bytes_co));
    col_buf_->upload(pre.rgb,
                     static_cast<std::size_t>(bytes_colors));
    VkBuffer tfinal_handle = cache.T_final_gpu
        ? static_cast<VkBuffer>(cache.T_final_gpu)
        : tf_buf_->handle();
    VkBuffer ncontrib_handle = cache.n_contrib_gpu
        ? static_cast<VkBuffer>(cache.n_contrib_gpu)
        : nc_buf_->handle();
    VkBuffer dlpix_handle = cache.dL_dpixels_gpu
        ? static_cast<VkBuffer>(cache.dL_dpixels_gpu)
        : dlpix_buf_->handle();
    last_forward_gpu_cache_used_ = cache.T_final_gpu != nullptr
        || cache.n_contrib_gpu != nullptr;
    if (!cache.T_final_gpu) {
        tf_buf_->upload(cache.T_final,
                        static_cast<std::size_t>(bytes_tfinal));
    }
    if (!cache.n_contrib_gpu) {
        // n_contrib is int*; reinterpret as uint32 (same width, positions < 2^31).
        nc_buf_->upload(cache.n_contrib,
                        static_cast<std::size_t>(bytes_ncontrib));
    }
    if (!cache.dL_dpixels_gpu) {
        if (dL_dpixels == nullptr) {
            throw std::runtime_error("RasterizerBackwardVulkan: missing dL_dpixels");
        }
        dlpix_buf_->upload(dL_dpixels,
                           static_cast<std::size_t>(bytes_dL_dpix));
    }

    // Zero-fill gradient output buffers (atomicAdd accumulates into them).
    dlm2d_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_m2d));
    dlcon_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_con));
    dlopa_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_opa));
    dlcol_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_col));

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
    rb.tile_ranges   = bin.tile_ranges_gpu
        ? static_cast<VkBuffer>(bin.tile_ranges_gpu)
        : tr_buf_->handle();
    rb.values_sorted = has_values_gpu
        ? static_cast<VkBuffer>(bin.values_sorted_gpu)
        : vs_buf_->handle();
    rb.means2D       = m2d_buf_  ->handle();
    rb.conic_opacity = co_buf_   ->handle();
    rb.colors        = col_buf_  ->handle();
    rb.T_final       = tfinal_handle;
    rb.n_contrib     = ncontrib_handle;
    rb.dL_dpixels    = dlpix_handle;
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

void RasterizerBackwardVulkan::backward_record_fused_eval3d_replay_into(
    VkCommandBuffer cmd,
    const PreprocessOutput& pre,
    const BinningOutput& bin,
    int num_gaussians,
    const Camera& cam,
    const RenderConfig& cfg,
    const ForwardCache& cache,
    const float* dL_dpixels,
    VkBuffer positions,
    VkBuffer scales,
    VkBuffer rotations,
    VkBuffer raw_rotations,
    VkBuffer filter_3D,
    VkBuffer d_means3D,
    VkBuffer d_scales,
    VkBuffer d_rotations) {
    last_forward_gpu_cache_used_ = false;
    const int W = cam.width;
    const int H = cam.height;
    if (W <= 0 || H <= 0) return;

    const int N = num_gaussians;
    if (N == 0) return;

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + cfg.tile_w - 1) / cfg.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + cfg.tile_h - 1) / cfg.tile_h);
    const int R = std::max(bin.total_pairs, 1);
    const int HW = H * W;
    prepare_for_n(N, R, static_cast<int>(num_tiles_x * num_tiles_y), HW);

    const bool has_replay_cpu = cache.replay_order_offsets && cache.replay_order_gids && cache.replay_order_count > 0;
    const bool has_replay_gpu = cache.replay_order_offsets_gpu && cache.replay_order_gids_gpu && cache.replay_order_count > 0;
    if (!pre.eval_3D || (!has_replay_cpu && !has_replay_gpu)) {
        throw std::runtime_error("RasterizerBackwardVulkan::backward_record_fused_eval3d_replay_into: missing eval_3D replay order");
    }

    const VkDeviceSize bytes_co = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_colors = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal = static_cast<VkDeviceSize>(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib = static_cast<VkDeviceSize>(HW) * sizeof(uint32_t);
    const VkDeviceSize bytes_dL_dpix = static_cast<VkDeviceSize>(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_opa = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_dL_col = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_drot_tangent = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_dL_g2s = static_cast<VkDeviceSize>(N) * 16u * sizeof(float);

    const bool has_gauss2screen_gpu = cache.gauss2screen_gpu != nullptr;
    const bool has_rgb_gpu = pre.rgb_gpu != nullptr;
    const bool has_conic_opacity_gpu = pre.conic_opacity_packed_gpu != nullptr;
    if (!has_gauss2screen_gpu && !pre.gauss2screen) {
        throw std::runtime_error("RasterizerBackwardVulkan::backward_record_fused_eval3d_replay_into: missing eval_3D gauss2screen");
    }

    VkBuffer gauss2screen_handle = has_gauss2screen_gpu
        ? static_cast<VkBuffer>(cache.gauss2screen_gpu)
        : g2s_buf_->handle();
    if (!has_gauss2screen_gpu) {
        g2s_buf_->upload(pre.gauss2screen, static_cast<std::size_t>(bytes_dL_g2s));
    }
    VkBuffer colors_handle = has_rgb_gpu
        ? static_cast<VkBuffer>(pre.rgb_gpu)
        : col_buf_->handle();
    if (!has_rgb_gpu) {
        col_buf_->upload(pre.rgb, static_cast<std::size_t>(bytes_colors));
    }
    VkBuffer conic_opacity_handle = has_conic_opacity_gpu
        ? static_cast<VkBuffer>(pre.conic_opacity_packed_gpu)
        : co_buf_->handle();
    if (!has_conic_opacity_gpu) {
        std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N) * 4u, 0.0f);
        for (int i = 0; i < N; ++i) {
            conic_opacity_packed[static_cast<std::size_t>(i) * 4u + 3u] = pre.opacities_2d[i];
        }
        co_buf_->upload(conic_opacity_packed.data(), static_cast<std::size_t>(bytes_co));
    }

    VkBuffer tfinal_handle = cache.T_final_gpu
        ? static_cast<VkBuffer>(cache.T_final_gpu)
        : tf_buf_->handle();
    VkBuffer ncontrib_handle = cache.n_contrib_gpu
        ? static_cast<VkBuffer>(cache.n_contrib_gpu)
        : nc_buf_->handle();
    VkBuffer dlpix_handle = cache.dL_dpixels_gpu
        ? static_cast<VkBuffer>(cache.dL_dpixels_gpu)
        : dlpix_buf_->handle();
    last_forward_gpu_cache_used_ = cache.T_final_gpu != nullptr
        || cache.n_contrib_gpu != nullptr
        || has_gauss2screen_gpu
        || has_rgb_gpu
        || has_conic_opacity_gpu;
    if (!cache.T_final_gpu) {
        tf_buf_->upload(cache.T_final, static_cast<std::size_t>(bytes_tfinal));
    }
    if (!cache.n_contrib_gpu) {
        nc_buf_->upload(cache.n_contrib, static_cast<std::size_t>(bytes_ncontrib));
    }
    if (!cache.dL_dpixels_gpu) {
        if (dL_dpixels == nullptr) {
            throw std::runtime_error("RasterizerBackwardVulkan: missing dL_dpixels");
        }
        dlpix_buf_->upload(dL_dpixels, static_cast<std::size_t>(bytes_dL_dpix));
    }

    dlopa_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_opa));
    dlcol_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_col));
    dlg2s_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_g2s));
    const bool use_tangent_subgroup = tangent_subgroup_bwd_enabled();
    const bool use_tangent_rot = tangent_rot_bwd_enabled() || use_tangent_subgroup;
    if (use_tangent_rot) {
        drot_tangent_buf_->zero_fill(static_cast<std::size_t>(bytes_drot_tangent));
    }

    VkBuffer replay_offsets_handle = dummy4_buf_->handle();
    VkBuffer replay_gids_handle = dummy4_buf_->handle();
    if (has_replay_cpu) {
        replay_offsets_buf_->upload(cache.replay_order_offsets,
            (static_cast<size_t>(HW) + 1u) * sizeof(uint32_t));
        if (cache.replay_order_count > buf_replay_count_) {
            replay_gids_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                static_cast<VkDeviceSize>(cache.replay_order_count) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buf_replay_count_ = cache.replay_order_count;
        }
        replay_gids_buf_->upload(cache.replay_order_gids,
            static_cast<size_t>(cache.replay_order_count) * sizeof(uint32_t));
        replay_offsets_handle = replay_offsets_buf_->handle();
        replay_gids_handle = replay_gids_buf_->handle();
    } else {
        replay_offsets_handle = static_cast<VkBuffer>(cache.replay_order_offsets_gpu);
        replay_gids_handle = static_cast<VkBuffer>(cache.replay_order_gids_gpu);
    }

    uint32_t hot_count = 0u;
    uint32_t hot_num_shards = 0u;
    const bool hot_requested = hot_gid_shard_bwd_enabled() && use_tangent_subgroup && has_replay_cpu;
    if (hot_requested) {
        const uint32_t min_hits = env_u32("GS3D_EVAL3D_HOT_GID_MIN_HITS", 256u);
        const uint32_t max_hot_gids = env_u32("GS3D_EVAL3D_HOT_GID_MAX", 8192u);
        hot_num_shards = floor_power_of_two(env_u32("GS3D_EVAL3D_HOT_GID_SHARDS", 64u));
        if (hot_num_shards == 0u) hot_num_shards = 1u;
        const uint32_t scratch_cap_mb = env_u32("GS3D_EVAL3D_HOT_GID_SCRATCH_MB", 256u);

        std::vector<uint32_t> counts(static_cast<size_t>(N), 0u);
        for (size_t i = 0; i < cache.replay_order_count; ++i) {
            const uint32_t gid = cache.replay_order_gids[i];
            if (gid < static_cast<uint32_t>(N)) {
                counts[gid] += 1u;
            }
        }

        std::vector<std::pair<uint32_t, uint32_t>> candidates;
        candidates.reserve(std::min(static_cast<uint32_t>(N), max_hot_gids));
        for (uint32_t gid = 0; gid < static_cast<uint32_t>(N); ++gid) {
            if (counts[gid] >= min_hits) {
                candidates.emplace_back(counts[gid], gid);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        if (candidates.size() > max_hot_gids) {
            candidates.resize(max_hot_gids);
        }

        const uint64_t scratch_bytes64 = static_cast<uint64_t>(candidates.size()) * hot_num_shards * 9u * sizeof(float);
        const uint64_t scratch_cap64 = static_cast<uint64_t>(scratch_cap_mb) * 1024u * 1024u;
        if (!candidates.empty() && scratch_bytes64 <= scratch_cap64) {
            std::vector<uint32_t> hot_lookup(static_cast<size_t>(N), 0xffffffffu);
            std::vector<uint32_t> hot_gids(candidates.size(), 0u);
            uint64_t hot_hits = 0u;
            for (size_t i = 0; i < candidates.size(); ++i) {
                const uint32_t gid = candidates[i].second;
                hot_lookup[gid] = static_cast<uint32_t>(i);
                hot_gids[i] = gid;
                hot_hits += candidates[i].first;
            }
            hot_count = static_cast<uint32_t>(hot_gids.size());
            const size_t scratch_bytes = static_cast<size_t>(scratch_bytes64);
            if (hot_gids_capacity_ < hot_gids.size()) {
                hot_gids_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                    static_cast<VkDeviceSize>(hot_gids.size()) * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                hot_gids_capacity_ = hot_gids.size();
            }
            if (hot_geom_scratch_capacity_ < scratch_bytes) {
                hot_geom_scratch_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                    static_cast<VkDeviceSize>(scratch_bytes),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                hot_geom_scratch_capacity_ = scratch_bytes;
            }
            hot_lookup_buf_->upload(hot_lookup.data(), hot_lookup.size() * sizeof(uint32_t));
            hot_gids_buf_->upload(hot_gids.data(), hot_gids.size() * sizeof(uint32_t));
            hot_geom_scratch_buf_->zero_fill(scratch_bytes);
            RasterizeBackwardEval3DHotUBO hot_ubo{};
            hot_ubo.hot_count = hot_count;
            hot_ubo.num_shards = hot_num_shards;
            hot_ubo.stride = 9u;
            hot_ubo_buf_->upload(&hot_ubo, sizeof(hot_ubo));
            std::printf("[GS3D_HOT_GID_SHARD] N=%d replay=%zu hot=%u min_hits=%u shards=%u hot_hits=%llu coverage=%.4f scratch_mb=%.2f\n",
                N,
                cache.replay_order_count,
                hot_count,
                min_hits,
                hot_num_shards,
                static_cast<unsigned long long>(hot_hits),
                cache.replay_order_count ? static_cast<double>(hot_hits) / static_cast<double>(cache.replay_order_count) : 0.0,
                static_cast<double>(scratch_bytes64) / (1024.0 * 1024.0));
        } else if (!candidates.empty()) {
            std::printf("[GS3D_HOT_GID_SHARD] skipped scratch_mb=%.2f cap_mb=%u hot_candidates=%zu\n",
                static_cast<double>(scratch_bytes64) / (1024.0 * 1024.0),
                scratch_cap_mb,
                candidates.size());
        }
    }

    RasterizeBackwardEval3DFusedUBO ubo{};
    std::memcpy(ubo.view_matrix, cam.view_matrix, 16 * sizeof(float));
    std::memcpy(ubo.viewproj_matrix, cam.viewproj_matrix, 16 * sizeof(float));
    ubo.W = static_cast<uint32_t>(W);
    ubo.H = static_cast<uint32_t>(H);
    ubo.num_tiles_x = num_tiles_x;
    ubo.bg_color[0] = cfg.bg_color[0];
    ubo.bg_color[1] = cfg.bg_color[1];
    ubo.bg_color[2] = cfg.bg_color[2];
    ubo.scale_modifier = cfg.scale_modifier;
    ubo.h_x = static_cast<float>(W) / (2.0f * cam.tan_fovx);
    ubo.h_y = static_cast<float>(H) / (2.0f * cam.tan_fovy);
    const float vp[16] = {
        static_cast<float>(W) * 0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, static_cast<float>(H) * 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        static_cast<float>(W) * 0.5f - 0.5f,
        static_cast<float>(H) * 0.5f - 0.5f,
        0.0f, 1.0f,
    };
    mat4Mul(vp, cam.viewproj_matrix, ubo.world2screen_matrix);
    fused_ubo_buf_->upload(&ubo, sizeof(ubo));

    const bool needs_input_barrier = cache.T_final_gpu || cache.n_contrib_gpu || cache.dL_dpixels_gpu
        || has_gauss2screen_gpu || has_rgb_gpu || has_conic_opacity_gpu || has_replay_gpu;
    if (use_tangent_rot) {
        RasterizeBackwardEval3DPass::FusedReplayTangentBuffers rb{};
        rb.gauss2screen = gauss2screen_handle;
        rb.conic_opacity = conic_opacity_handle;
        rb.colors = colors_handle;
        rb.T_final = tfinal_handle;
        rb.n_contrib = ncontrib_handle;
        rb.dL_dpixels = dlpix_handle;
        rb.dL_dopacity = dlopa_buf_->handle();
        rb.dL_dcolors = dlcol_buf_->handle();
        rb.replay_order_offsets = replay_offsets_handle;
        rb.replay_order_gids = replay_gids_handle;
        rb.positions = positions;
        rb.scales = scales;
        rb.rotations = rotations;
        rb.raw_rotations = raw_rotations;
        rb.filter_3D = filter_3D;
        rb.d_means3D = d_means3D;
        rb.d_scales = d_scales;
        rb.d_rot_tangent = drot_tangent_buf_->handle();
        rb.d_rotations = d_rotations;
        if (hot_count > 0u) {
            RasterizeBackwardEval3DPass::FusedReplayTangentHotBuffers hb{};
            hb.base = rb;
            hb.hot_lookup = hot_lookup_buf_->handle();
            hb.hot_gids = hot_gids_buf_->handle();
            hb.hot_geom_scratch = hot_geom_scratch_buf_->handle();
            eval3d_pass_->bind_fused_replay_tangent_hot_buffers(hb, fused_ubo_buf_->handle(), hot_ubo_buf_->handle());
            insert_compute_barrier(cmd);
            eval3d_pass_->record_fused_replay_tangent_hot(cmd, num_tiles_x, num_tiles_y, static_cast<uint32_t>(N), hot_count);
        } else {
            eval3d_pass_->bind_fused_replay_tangent_buffers(rb, fused_ubo_buf_->handle());
            if (needs_input_barrier) {
                insert_compute_barrier(cmd);
            }
            if (use_tangent_subgroup) {
                eval3d_pass_->record_fused_replay_tangent_subgroup(cmd, num_tiles_x, num_tiles_y, static_cast<uint32_t>(N));
            } else {
                eval3d_pass_->record_fused_replay_tangent(cmd, num_tiles_x, num_tiles_y, static_cast<uint32_t>(N));
            }
        }
    } else {
        RasterizeBackwardEval3DPass::FusedReplayBuffers rb{};
        rb.gauss2screen = gauss2screen_handle;
        rb.conic_opacity = conic_opacity_handle;
        rb.colors = colors_handle;
        rb.T_final = tfinal_handle;
        rb.n_contrib = ncontrib_handle;
        rb.dL_dpixels = dlpix_handle;
        rb.dL_dopacity = dlopa_buf_->handle();
        rb.dL_dcolors = dlcol_buf_->handle();
        rb.replay_order_offsets = replay_offsets_handle;
        rb.replay_order_gids = replay_gids_handle;
        rb.positions = positions;
        rb.scales = scales;
        rb.rotations = rotations;
        rb.raw_rotations = raw_rotations;
        rb.filter_3D = filter_3D;
        rb.d_means3D = d_means3D;
        rb.d_scales = d_scales;
        rb.d_rotations = d_rotations;
        eval3d_pass_->bind_fused_replay_buffers(rb, fused_ubo_buf_->handle());
        if (needs_input_barrier) {
            insert_compute_barrier(cmd);
        }
        eval3d_pass_->record_fused_replay(cmd, num_tiles_x, num_tiles_y);
    }
}

void RasterizerBackwardVulkan::download_outputs(
    int N,
    std::vector<float>& d_means2D,
    std::vector<float>& d_conics,
    std::vector<float>& d_opacity,
    std::vector<float>& d_rgb,
    std::vector<float>* d_gauss2screen) const
{
    d_means2D.assign(static_cast<std::size_t>(N) * 2u, 0.0f);
    d_conics.assign(static_cast<std::size_t>(N) * 3u, 0.0f);
    d_opacity.assign(static_cast<std::size_t>(N), 0.0f);
    d_rgb.assign(static_cast<std::size_t>(N) * 3u, 0.0f);
    if (d_gauss2screen) {
        d_gauss2screen->assign(static_cast<std::size_t>(N) * 16u, 0.0f);
    }
    if (N == 0 || !dlm2d_buf_) return;

    dlm2d_buf_->download(d_means2D.data(), d_means2D.size() * sizeof(float));
    dlcon_buf_->download(d_conics.data(),  d_conics.size()  * sizeof(float));
    dlopa_buf_->download(d_opacity.data(), d_opacity.size() * sizeof(float));
    dlcol_buf_->download(d_rgb.data(),     d_rgb.size()     * sizeof(float));
    if (d_gauss2screen && dlg2s_buf_) {
        dlg2s_buf_->download(d_gauss2screen->data(), d_gauss2screen->size() * sizeof(float));
    }
}

void RasterizerBackwardVulkan::backward_record_into(VkCommandBuffer cmd,
                                                     const PreprocessOutput& pre,
                                                     const BinningOutput& bin,
                                                     int num_gaussians,
                                                     const Camera& cam,
                                                     const RenderConfig& cfg,
                                                     const ForwardCache& cache,
                                                     const float* dL_dpixels,
                                                     const float* rendered_image) {
    last_forward_gpu_cache_used_ = false;
    const int W  = cam.width;
    const int H  = cam.height;
    if (W <= 0 || H <= 0) return;

    const int N = num_gaussians;
    if (N == 0) return;

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + cfg.tile_w - 1) / cfg.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + cfg.tile_h - 1) / cfg.tile_h);

    const int R   = bin.total_pairs;
    const int HW  = H * W;

    if (R <= 0) {
        prepare_for_n(N, 1, static_cast<int>(num_tiles_x * num_tiles_y), HW);
        dlm2d_buf_->zero_fill(static_cast<std::size_t>(N) * 2u * sizeof(float));
        dlcon_buf_->zero_fill(static_cast<std::size_t>(N) * 3u * sizeof(float));
        dlopa_buf_->zero_fill(static_cast<std::size_t>(N) * sizeof(float));
        dlcol_buf_->zero_fill(static_cast<std::size_t>(N) * 3u * sizeof(float));
        dlg2s_buf_->zero_fill(static_cast<std::size_t>(N) * 16u * sizeof(float));
        return;
    }
    if (bin.values_sorted == nullptr && bin.values_sorted_gpu == nullptr) {
        throw std::runtime_error("RasterizerBackwardVulkan::backward_record_into: missing sorted gaussian ids");
    }

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
    const VkDeviceSize bytes_dL_g2s =
        static_cast<VkDeviceSize>(N) * 16u * sizeof(float);
    const bool has_values_gpu = bin.values_sorted_gpu != nullptr;

    if (pre.eval_3D) {
        const bool has_gauss2screen_gpu = cache.gauss2screen_gpu != nullptr;
        const bool has_rgb_gpu = pre.rgb_gpu != nullptr;
        const bool has_conic_opacity_gpu = pre.conic_opacity_packed_gpu != nullptr;
        if (!has_gauss2screen_gpu && !pre.gauss2screen)
            throw std::runtime_error("RasterizerBackwardVulkan::backward_record_into: missing eval_3D gauss2screen");
        if (bin.tile_ranges_gpu == nullptr) {
            tr_buf_->upload(bin.tile_ranges, static_cast<std::size_t>(bytes_tile_ranges));
        }
        if (!has_values_gpu) {
            vs_buf_->upload(bin.values_sorted, static_cast<std::size_t>(bytes_vs));
        }
        VkBuffer gauss2screen_handle = has_gauss2screen_gpu
            ? static_cast<VkBuffer>(cache.gauss2screen_gpu)
            : g2s_buf_->handle();
        if (!has_gauss2screen_gpu) {
            g2s_buf_->upload(pre.gauss2screen, static_cast<std::size_t>(bytes_dL_g2s));
        }
        VkBuffer colors_handle = has_rgb_gpu
            ? static_cast<VkBuffer>(pre.rgb_gpu)
            : col_buf_->handle();
        if (!has_rgb_gpu) {
            col_buf_->upload(pre.rgb, static_cast<std::size_t>(bytes_colors));
        }
        VkBuffer tfinal_handle = cache.T_final_gpu
            ? static_cast<VkBuffer>(cache.T_final_gpu)
            : tf_buf_->handle();
        VkBuffer ncontrib_handle = cache.n_contrib_gpu
            ? static_cast<VkBuffer>(cache.n_contrib_gpu)
            : nc_buf_->handle();
        VkBuffer dlpix_handle = cache.dL_dpixels_gpu
            ? static_cast<VkBuffer>(cache.dL_dpixels_gpu)
            : dlpix_buf_->handle();
        last_forward_gpu_cache_used_ = cache.T_final_gpu != nullptr
            || cache.n_contrib_gpu != nullptr
            || has_gauss2screen_gpu
            || has_rgb_gpu
            || has_conic_opacity_gpu;
        if (!cache.T_final_gpu) {
            tf_buf_->upload(cache.T_final, static_cast<std::size_t>(bytes_tfinal));
        }
        if (!cache.n_contrib_gpu) {
            nc_buf_->upload(cache.n_contrib, static_cast<std::size_t>(bytes_ncontrib));
        }
        if (!cache.dL_dpixels_gpu) {
            if (dL_dpixels == nullptr) {
                throw std::runtime_error("RasterizerBackwardVulkan: missing dL_dpixels");
            }
            dlpix_buf_->upload(dL_dpixels, static_cast<std::size_t>(bytes_dL_dpix));
        }

        VkBuffer conic_opacity_handle = has_conic_opacity_gpu
            ? static_cast<VkBuffer>(pre.conic_opacity_packed_gpu)
            : co_buf_->handle();
        if (!has_conic_opacity_gpu) {
            std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N) * 4u, 0.0f);
            for (int i = 0; i < N; ++i) {
                conic_opacity_packed[static_cast<std::size_t>(i) * 4u + 3u] = pre.opacities_2d[i];
            }
            co_buf_->upload(conic_opacity_packed.data(), static_cast<std::size_t>(bytes_co));
        }

        dlopa_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_opa));
        dlcol_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_col));
        dlg2s_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_g2s));

        const bool has_replay_cpu = cache.replay_order_offsets && cache.replay_order_gids && cache.replay_order_count > 0;
        const bool has_replay_gpu = cache.replay_order_offsets_gpu && cache.replay_order_gids_gpu && cache.replay_order_count > 0;
        const bool use_replay_order = has_replay_cpu || has_replay_gpu;
        VkBuffer replay_offsets_handle = dummy4_buf_->handle();
        VkBuffer replay_gids_handle = dummy4_buf_->handle();
        if (has_replay_cpu) {
            replay_offsets_buf_->upload(cache.replay_order_offsets,
                (static_cast<size_t>(HW) + 1u) * sizeof(uint32_t));
            if (cache.replay_order_count > buf_replay_count_) {
                replay_gids_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                    static_cast<VkDeviceSize>(cache.replay_order_count) * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                buf_replay_count_ = cache.replay_order_count;
            }
            replay_gids_buf_->upload(cache.replay_order_gids,
                static_cast<size_t>(cache.replay_order_count) * sizeof(uint32_t));
            replay_offsets_handle = replay_offsets_buf_->handle();
            replay_gids_handle = replay_gids_buf_->handle();
        } else if (has_replay_gpu) {
            replay_offsets_handle = static_cast<VkBuffer>(cache.replay_order_offsets_gpu);
            replay_gids_handle = static_cast<VkBuffer>(cache.replay_order_gids_gpu);
        }

        RasterizeBackwardUBO ubo{};
        ubo.W = static_cast<uint32_t>(W);
        ubo.H = static_cast<uint32_t>(H);
        ubo.num_tiles_x = num_tiles_x;
        ubo._pad = use_replay_order ? 1u : 0u;
        ubo.bg_color[0] = cfg.bg_color[0];
        ubo.bg_color[1] = cfg.bg_color[1];
        ubo.bg_color[2] = cfg.bg_color[2];
        ubo_buf_->upload(&ubo, sizeof(ubo));

        RasterizeBackwardEval3DPass::Buffers rb{};
        rb.tile_ranges = bin.tile_ranges_gpu
            ? static_cast<VkBuffer>(bin.tile_ranges_gpu)
            : tr_buf_->handle();
        rb.values_sorted = has_values_gpu
            ? static_cast<VkBuffer>(bin.values_sorted_gpu)
            : vs_buf_->handle();
        rb.gauss2screen = gauss2screen_handle;
        rb.conic_opacity = conic_opacity_handle;
        rb.colors = colors_handle;
        rb.T_final = tfinal_handle;
        rb.n_contrib = ncontrib_handle;
        rb.rendered_image = dummy4_buf_->handle();
        rb.dL_dpixels = dlpix_handle;
        rb.dL_dgauss2screen = dlg2s_buf_->handle();
        rb.dL_dopacity = dlopa_buf_->handle();
        rb.dL_dcolors = dlcol_buf_->handle();
        rb.replay_order_offsets = replay_offsets_handle;
        rb.replay_order_gids = replay_gids_handle;
        eval3d_pass_->bind_buffers(rb, ubo_buf_->handle());
        if (bin.tile_ranges_gpu || has_values_gpu || cache.T_final_gpu || cache.n_contrib_gpu
            || cache.dL_dpixels_gpu || has_gauss2screen_gpu || has_rgb_gpu
            || has_conic_opacity_gpu || has_replay_gpu) {
            insert_compute_barrier(cmd);
        }
        eval3d_pass_->record(cmd, num_tiles_x, num_tiles_y, use_replay_order);
        return;
    }

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
    if (bin.tile_ranges_gpu == nullptr) {
        tr_buf_->upload(bin.tile_ranges,
                        static_cast<std::size_t>(bytes_tile_ranges));
    }
    if (!has_values_gpu) {
        vs_buf_->upload(bin.values_sorted,
                        static_cast<std::size_t>(bytes_vs));
    }
    m2d_buf_->upload(pre.means2D,
                     static_cast<std::size_t>(bytes_m2d));
    co_buf_ ->upload(conic_opacity_packed.data(),
                     static_cast<std::size_t>(bytes_co));
    col_buf_->upload(pre.rgb,
                     static_cast<std::size_t>(bytes_colors));
    VkBuffer tfinal_handle = cache.T_final_gpu
        ? static_cast<VkBuffer>(cache.T_final_gpu)
        : tf_buf_->handle();
    VkBuffer ncontrib_handle = cache.n_contrib_gpu
        ? static_cast<VkBuffer>(cache.n_contrib_gpu)
        : nc_buf_->handle();
    VkBuffer dlpix_handle = cache.dL_dpixels_gpu
        ? static_cast<VkBuffer>(cache.dL_dpixels_gpu)
        : dlpix_buf_->handle();
    last_forward_gpu_cache_used_ = cache.T_final_gpu != nullptr
        || cache.n_contrib_gpu != nullptr;
    if (!cache.T_final_gpu) {
        tf_buf_->upload(cache.T_final,
                        static_cast<std::size_t>(bytes_tfinal));
    }
    if (!cache.n_contrib_gpu) {
        // n_contrib is int*; reinterpret as uint32 (same width, positions < 2^31).
        nc_buf_->upload(cache.n_contrib,
                        static_cast<std::size_t>(bytes_ncontrib));
    }
    if (!cache.dL_dpixels_gpu) {
        if (dL_dpixels == nullptr) {
            throw std::runtime_error("RasterizerBackwardVulkan: missing dL_dpixels");
        }
        dlpix_buf_->upload(dL_dpixels,
                           static_cast<std::size_t>(bytes_dL_dpix));
    }

    // Zero-fill gradient output buffers (atomicAdd accumulates into them).
    dlm2d_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_m2d));
    dlcon_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_con));
    dlopa_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_opa));
    dlcol_buf_->zero_fill(static_cast<std::size_t>(bytes_dL_col));

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
    rb.tile_ranges   = bin.tile_ranges_gpu
        ? static_cast<VkBuffer>(bin.tile_ranges_gpu)
        : tr_buf_->handle();
    rb.values_sorted = has_values_gpu
        ? static_cast<VkBuffer>(bin.values_sorted_gpu)
        : vs_buf_->handle();
    rb.means2D       = m2d_buf_  ->handle();
    rb.conic_opacity = co_buf_   ->handle();
    rb.colors        = col_buf_  ->handle();
    rb.T_final       = tfinal_handle;
    rb.n_contrib     = ncontrib_handle;
    rb.dL_dpixels    = dlpix_handle;
    rb.dL_dmeans2D   = dlm2d_buf_->handle();
    rb.dL_dconics    = dlcon_buf_->handle();
    rb.dL_dopacity   = dlopa_buf_->handle();
    rb.dL_dcolors    = dlcol_buf_->handle();

    pass_->bind_buffers(rb, ubo_buf_->handle());
    if (bin.tile_ranges_gpu || has_values_gpu || cache.T_final_gpu || cache.n_contrib_gpu || cache.dL_dpixels_gpu) {
        insert_compute_barrier(cmd);
    }
    pass_->record(cmd, num_tiles_x, num_tiles_y);
}
