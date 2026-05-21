// SP-2 T17: RasterizerVulkan — Rasterizer adapter around rasterize.comp.
//
// Phase 1 sync path: upload all per-frame inputs to host-visible SSBOs,
// dispatch rasterize.comp via RasterizePass, download output image (plus
// optional cache arrays).
//
// Empty-scene fast path: when there are no (Gaussian, tile) pairs — either
// the scene is empty or all Gaussians were culled — write the background
// colour to output_image directly. This matches rasterize.comp's final
// "C += T*bg_color" with T=1, without paying for a GPU round-trip on a trivial
// scene. It mirrors the CPU rasterizer's behavior on empty input.
//
// Buffer-size contract (matches the shader's std430 layouts):
//   - values_sorted       : R * sizeof(uint32)
//   - tile_ranges         : num_tiles * 2 * sizeof(uint32)    (flat pairs)
//   - means2D             : N_eff * 2 * sizeof(float)
//   - conic_opacity_packed: N_eff * 4 * sizeof(float)
//   - rgb                 : N_eff * 3 * sizeof(float)
//   - out_image           : 3 * H * W * sizeof(float)         CHW (channel-first, matches CUDA/PyTorch)
//   - transmittance       : H * W * sizeof(float)
//   - n_contrib           : H * W * sizeof(uint32)            (ForwardCache::n_contrib is int*, same width)
//   - raster_ubo          : sizeof(RasterizeUBO) = 16         UNIFORM_BUFFER
//
// N_eff: we only ever index Gaussians that appear in values_sorted, so
// sizing the per-Gaussian SSBOs to (max referenced index + 1) is sufficient
// and shrinks PCIe traffic on sparse scenes. R <= 256 in Phase 1 smoke tests
// so the linear scan is negligible.

#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/preprocess_bindings.h"   // RasterEval3DUBO
#include "math_utils.h"                   // invertMatrix4x4

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
bool replay_gpu_only_enabled() {
    const char* env = std::getenv("GS3D_EVAL3D_REPLAY_GPU_ONLY");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

}

RasterizerVulkan::RasterizerVulkan(VulkanContext& ctx,
                                   bool eval_3D,
                                   bool eval3d_raw_replay)
    : ctx_(ctx),
      eval_3D_(eval_3D),
      eval3d_raw_replay_(eval3d_raw_replay),
      sort_mode_(rasterize_spec::SORT_MODE_HIERARCHICAL) {
    pass_ = std::make_unique<RasterizePass>(
        ctx_,
        eval_3D ? 1u : 0u,
        /*spec_trace_enabled=*/0u,
        sort_mode_,
        eval3d_raw_replay_ ? 1u : 0u);
}

RasterizerVulkan::RasterizerVulkan(VulkanContext& ctx,
                                   const splatting::SplattingSettings& s,
                                   bool eval3d_raw_replay)
    : ctx_(ctx),
      eval_3D_(s.eval_3D),
      eval3d_raw_replay_(eval3d_raw_replay),
      sort_mode_(static_cast<uint32_t>(s.sort_settings.sort_mode)) {
    splatting::validate_vk_supported(s);
    pass_ = std::make_unique<RasterizePass>(
        ctx_,
        eval_3D_ ? 1u : 0u,
        /*spec_trace_enabled=*/0u,
        sort_mode_,
        eval3d_raw_replay_ ? 1u : 0u);
}

// Out-of-line so unique_ptr<RasterizePass> can see the complete type from
// the header.
RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::rasterize(const PreprocessOutput& preprocess,
                                 const BinningOutput& binning,
                                 const Camera& camera,
                                 const RenderConfig& config,
                                 float* output_image,
                                 float* /*output_depth*/,
                                 ForwardCache* cache,
                                 FrameAllocator* /*allocator*/) {
    const int W = camera.width;
    const int H = camera.height;
    if (W <= 0 || H <= 0) return;

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + config.tile_w - 1) / config.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + config.tile_h - 1) / config.tile_h);

    const int R = binning.total_pairs;
    const int HW = H * W;
    const bool has_values_gpu = binning.values_sorted_gpu != nullptr;
#ifdef GS3D_TESTING
    last_layer1_image_downloaded_ = false;
    last_layer1_cache_downloaded_ = false;
#endif

    // -------------------------------------------------------------------
    // Empty-scene fast path.
    // -------------------------------------------------------------------
    // rasterize.comp's final contribution for every pixel is T*bg_color.
    // With no Gaussians to blend, T stays at 1 and the whole image is
    // bg_color. We short-circuit on the host to avoid a GPU round-trip.
    if (R <= 0 || (binning.values_sorted == nullptr && !has_values_gpu)) {
        for (int ch = 0; ch < 3; ++ch) {
            const float bg = config.bg_color[ch];
            float* plane = output_image + static_cast<std::size_t>(ch) * HW;
            for (int px = 0; px < HW; ++px) plane[px] = bg;
        }
        // T_final = 1 and n_contrib = 0 for every pixel — mirror that into
        // the cache if the caller asked for it.
        if (cache) {
            if (cache->T_final) {
                for (int px = 0; px < HW; ++px) cache->T_final[px] = 1.0f;
            }
            if (cache->n_contrib) {
                for (int px = 0; px < HW; ++px) cache->n_contrib[px] = 0;
            }
            cache->replay_order_offsets_storage.clear();
            cache->replay_order_gids_storage.clear();
            cache->replay_order_offsets = nullptr;
            cache->replay_order_gids = nullptr;
            cache->replay_order_count = 0u;
            cache->replay_order_offsets_gpu = nullptr;
            cache->replay_order_gids_gpu = nullptr;
            cache->rendered_image_gpu = nullptr;
            cache->T_final_gpu = nullptr;
            cache->n_contrib_gpu = nullptr;
            cache->dL_dpixels_gpu = nullptr;
            cache->gpu_resident_outputs = false;
        }
        return;
    }

    // -------------------------------------------------------------------
    // Compute N_eff: 1 + max index referenced in values_sorted.
    // -------------------------------------------------------------------
    // The shader only reads per-Gaussian data at indices appearing in
    // values_sorted, so host values keep uploads tight on sparse scenes. The
    // GPU-resident sort path cannot scan on the host, so it uploads all current
    // preprocess outputs instead.
    uint32_t N_eff = 0u;
    if (binning.values_sorted != nullptr) {
        uint32_t max_gid = 0u;
        for (int i = 0; i < R; ++i) {
            const uint32_t g = binning.values_sorted[i];
            if (g > max_gid) max_gid = g;
        }
        N_eff = max_gid + 1u;
    } else {
        if (preprocess.num_gaussians <= 0) {
            throw std::runtime_error("RasterizerVulkan::rasterize: GPU values_sorted requires preprocess.num_gaussians");
        }
        N_eff = static_cast<uint32_t>(preprocess.num_gaussians);
    }

    const bool has_means2D_gpu = preprocess.means2D_gpu != nullptr;
    const bool has_conic_opacity_gpu = preprocess.conic_opacity_packed_gpu != nullptr;
    const bool has_rgb_gpu = preprocess.rgb_gpu != nullptr;
    const bool has_gauss2screen_gpu = preprocess.gauss2screen_gpu != nullptr
        || (cache && cache->gauss2screen_gpu != nullptr);
    const bool has_cov3D_inv_gpu = preprocess.cov3D_inv_gpu != nullptr;
    const bool has_mean_offset_gpu = preprocess.mean_offset_gpu != nullptr;

    std::vector<float> conic_opacity_packed;
    if (!has_conic_opacity_gpu) {
        conic_opacity_packed.resize(static_cast<std::size_t>(N_eff) * 4u);
        for (uint32_t i = 0; i < N_eff; ++i) {
            const std::size_t dst = static_cast<std::size_t>(i) * 4u;
            const std::size_t src = static_cast<std::size_t>(i) * 3u;
            conic_opacity_packed[dst + 0] = preprocess.conics[src + 0];
            conic_opacity_packed[dst + 1] = preprocess.conics[src + 1];
            conic_opacity_packed[dst + 2] = preprocess.conics[src + 2];
            conic_opacity_packed[dst + 3] = preprocess.opacities_2d[i];
        }
    }

    // -------------------------------------------------------------------
    // Allocate GPU buffers (host-visible coherent; Phase 1).
    // -------------------------------------------------------------------
    const VkDeviceSize bytes_vs       =
        static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_tr       =
        static_cast<VkDeviceSize>(binning.num_tiles) * 2u * sizeof(uint32_t);
    const VkDeviceSize bytes_m2d      =
        static_cast<VkDeviceSize>(N_eff) * 2u * sizeof(float);
    const VkDeviceSize bytes_co       =
        static_cast<VkDeviceSize>(N_eff) * 4u * sizeof(float);
    const VkDeviceSize bytes_rgb      =
        static_cast<VkDeviceSize>(N_eff) * 3u * sizeof(float);
    const VkDeviceSize bytes_img      =
        static_cast<VkDeviceSize>(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal   =
        static_cast<VkDeviceSize>(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib =
        static_cast<VkDeviceSize>(HW) * sizeof(uint32_t);

    const bool gpu_resident_outputs = cache && cache->gpu_resident_outputs;
    const bool retain_gpu_outputs = cache && (cache->gpu_resident_outputs || cache->retain_gpu_outputs);
    auto reuse_or_alloc = [&](std::unique_ptr<VulkanBuffer>& slot,
                              VkDeviceSize bytes,
                              VkBufferUsageFlags usage) {
        if (retain_gpu_outputs && slot && slot->size() >= bytes) {
            return std::move(slot);
        }
        return std::make_unique<VulkanBuffer>(ctx_, bytes, usage);
    };

    std::unique_ptr<VulkanBuffer> vs_buf;
    std::unique_ptr<VulkanBuffer> tr_buf;
    std::unique_ptr<VulkanBuffer> m2d_buf;
    std::unique_ptr<VulkanBuffer> co_buf;
    std::unique_ptr<VulkanBuffer> rgb_buf;
    if (!has_values_gpu) {
        vs_buf = std::make_unique<VulkanBuffer>(
            ctx_, bytes_vs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    if (binning.tile_ranges_gpu == nullptr) {
        tr_buf = std::make_unique<VulkanBuffer>(
            ctx_, bytes_tr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    if (!has_means2D_gpu) {
        m2d_buf = std::make_unique<VulkanBuffer>(
            ctx_, bytes_m2d, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    if (!has_conic_opacity_gpu) {
        co_buf = std::make_unique<VulkanBuffer>(
            ctx_, bytes_co, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    if (!has_rgb_gpu) {
        rgb_buf = std::make_unique<VulkanBuffer>(
            ctx_, bytes_rgb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    auto img_buf = reuse_or_alloc(r_img_, bytes_img, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto t_buf   = reuse_or_alloc(r_tfinal_, bytes_tfinal, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto nc_buf  = reuse_or_alloc(r_ncontrib_, bytes_ncontrib, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ubo_buf = reuse_or_alloc(r_ubo_, static_cast<VkDeviceSize>(sizeof(RasterizeUBO)),
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // -------------------------------------------------------------------
    // Upload inputs.
    // -------------------------------------------------------------------
    if (!has_values_gpu) {
        vs_buf->upload(binning.values_sorted,
                       static_cast<std::size_t>(bytes_vs));
    }
    if (binning.tile_ranges_gpu == nullptr) {
        tr_buf->upload(binning.tile_ranges,
                       static_cast<std::size_t>(bytes_tr));
    }
    if (!has_means2D_gpu) {
        m2d_buf->upload(preprocess.means2D,
                        static_cast<std::size_t>(bytes_m2d));
    }
    if (!has_conic_opacity_gpu) {
        co_buf->upload(conic_opacity_packed.data(),
                       static_cast<std::size_t>(bytes_co));
    }
    if (!has_rgb_gpu) {
        rgb_buf->upload(preprocess.rgb,
                        static_cast<std::size_t>(bytes_rgb));
    }

    RasterizeUBO ubo{};
    ubo.bg_r = config.bg_color[0];
    ubo.bg_g = config.bg_color[1];
    ubo.bg_b = config.bg_color[2];
    ubo_buf->upload(&ubo, sizeof(ubo));

    // -------------------------------------------------------------------
    // eval_3D rasterize buffers (gauss2screen, cov3D_inv, mean_offset,
    // RasterEval3DUBO). Opacity is read from conic_opacity_packed slot 3.
    // -------------------------------------------------------------------
    std::unique_ptr<VulkanBuffer> g2s_buf, r_cov3d_buf, r_mo_buf;
    std::unique_ptr<VulkanBuffer> eval3d_ubo_buf;
    auto dummy4_buf = reuse_or_alloc(r_dummy4_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    if (eval_3D_) {
        if (!has_gauss2screen_gpu) {
            g2s_buf = std::make_unique<VulkanBuffer>(ctx_,
                static_cast<VkDeviceSize>(N_eff) * 16u * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            g2s_buf->upload(preprocess.gauss2screen,
                static_cast<std::size_t>(N_eff) * 16u * sizeof(float));
        }
        if (!has_cov3D_inv_gpu) {
            r_cov3d_buf = std::make_unique<VulkanBuffer>(ctx_,
                static_cast<VkDeviceSize>(N_eff) * 6u * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            r_cov3d_buf->upload(preprocess.cov3D_inv,
                static_cast<std::size_t>(N_eff) * 6u * sizeof(float));
        }
        if (!has_mean_offset_gpu) {
            r_mo_buf = std::make_unique<VulkanBuffer>(ctx_,
                static_cast<VkDeviceSize>(N_eff) * 3u * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            r_mo_buf->upload(preprocess.mean_offset,
                static_cast<std::size_t>(N_eff) * 3u * sizeof(float));
        }
    }
    eval3d_ubo_buf = reuse_or_alloc(r_eval3d_ubo_, sizeof(RasterEval3DUBO),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    RasterEval3DUBO eubo{};
    if (!eval3d_raw_replay_) {
        float inv_vp[16];
        if (invertMatrix4x4(camera.viewproj_matrix, inv_vp))
            std::memcpy(eubo.inverse_vp, inv_vp, sizeof(inv_vp));
        eubo.cam_pos[0] = camera.cam_pos[0];
        eubo.cam_pos[1] = camera.cam_pos[1];
        eubo.cam_pos[2] = camera.cam_pos[2];
        eubo.cam_pos[3] = 0.0f;
        eubo.img_size[0] = static_cast<float>(W);
        eubo.img_size[1] = static_cast<float>(H);
        eubo.img_size[2] = 0.0f;
        eubo.img_size[3] = 0.0f;
        eval3d_ubo_buf->upload(&eubo, sizeof(eubo));
    }
    std::unique_ptr<VulkanBuffer> replay_offsets_buf;
    std::unique_ptr<VulkanBuffer> replay_gids_buf;
    if (!eval3d_raw_replay_) {
        replay_offsets_buf = std::make_unique<VulkanBuffer>(ctx_, 4u,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        replay_gids_buf = std::make_unique<VulkanBuffer>(ctx_, 4u,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    // -------------------------------------------------------------------
    // Bind and dispatch.
    // -------------------------------------------------------------------
    RasterizePass::Buffers rb{};
    rb.values_sorted        = has_values_gpu
        ? static_cast<VkBuffer>(binning.values_sorted_gpu)
        : vs_buf->handle();
    rb.tile_ranges          = binning.tile_ranges_gpu
        ? static_cast<VkBuffer>(binning.tile_ranges_gpu)
        : tr_buf->handle();
    rb.means2D              = has_means2D_gpu
        ? static_cast<VkBuffer>(preprocess.means2D_gpu)
        : m2d_buf->handle();
    rb.conic_opacity_packed = has_conic_opacity_gpu
        ? static_cast<VkBuffer>(preprocess.conic_opacity_packed_gpu)
        : co_buf->handle();
    rb.rgb                  = has_rgb_gpu
        ? static_cast<VkBuffer>(preprocess.rgb_gpu)
        : rgb_buf->handle();
    rb.out_image            = img_buf->handle();
    rb.transmittance        = t_buf  ->handle();
    rb.n_contrib            = nc_buf ->handle();
    rb.raster_ubo           = ubo_buf->handle();
    rb.gauss2screen         = has_gauss2screen_gpu
        ? static_cast<VkBuffer>(preprocess.gauss2screen_gpu ? preprocess.gauss2screen_gpu : cache->gauss2screen_gpu)
        : (g2s_buf ? g2s_buf->handle() : dummy4_buf->handle());
    rb.opacities_2d         = dummy4_buf->handle();
    rb.cov3D_inv            = has_cov3D_inv_gpu
        ? static_cast<VkBuffer>(preprocess.cov3D_inv_gpu)
        : (r_cov3d_buf ? r_cov3d_buf->handle() : dummy4_buf->handle());
    rb.mean_offset          = has_mean_offset_gpu
        ? static_cast<VkBuffer>(preprocess.mean_offset_gpu)
        : (r_mo_buf ? r_mo_buf->handle() : dummy4_buf->handle());
    rb.raster_eval3d_ubo    = eval3d_ubo_buf->handle();
    rb.replay_order_offsets = replay_offsets_buf ? replay_offsets_buf->handle() : dummy4_buf->handle();
    rb.replay_order_gids    = replay_gids_buf ? replay_gids_buf->handle() : dummy4_buf->handle();
    pass_->bind_buffers(rb);
    pass_->dispatch_sync(N_eff,
                         static_cast<uint32_t>(W), static_cast<uint32_t>(H),
                         num_tiles_x, num_tiles_y);

    // -------------------------------------------------------------------
    // Download outputs unless the caller explicitly keeps them GPU-resident.
    // -------------------------------------------------------------------
#ifdef GS3D_TESTING
    last_layer1_image_downloaded_ = false;
    last_layer1_cache_downloaded_ = false;
#endif
    r_W_ = static_cast<uint32_t>(W);
    r_H_ = static_cast<uint32_t>(H);
    if (!gpu_resident_outputs) {
        img_buf->download(output_image, static_cast<std::size_t>(bytes_img));
#ifdef GS3D_TESTING
        last_layer1_image_downloaded_ = true;
#endif
    }

    if (cache) {
        cache->rendered_image_gpu = retain_gpu_outputs ? img_buf->handle() : nullptr;
        cache->T_final_gpu = retain_gpu_outputs ? t_buf->handle() : nullptr;
        cache->n_contrib_gpu = retain_gpu_outputs ? nc_buf->handle() : nullptr;
        cache->dL_dpixels_gpu = nullptr;

        if (cache->T_final && !gpu_resident_outputs) {
            t_buf->download(cache->T_final,
                            static_cast<std::size_t>(bytes_tfinal));
#ifdef GS3D_TESTING
            last_layer1_cache_downloaded_ = true;
#endif
        }
        constexpr uint32_t kReplayScanMaxElements = 256u * 256u * 256u;
        const bool use_gpu_replay_offsets = eval_3D_ && !eval3d_raw_replay_
            && gpu_resident_outputs && replay_gpu_only_enabled()
            && HW > 0u && HW <= kReplayScanMaxElements;
        const bool need_cpu_n_contrib = !gpu_resident_outputs
            || (eval_3D_ && !eval3d_raw_replay_ && !use_gpu_replay_offsets);
        if (cache->n_contrib && need_cpu_n_contrib) {
            // ForwardCache::n_contrib is int*; the shader writes uint32 of the
            // same width so a raw copy is correct — no sign-reinterp happens
            // because counts fit well under 2^31.
            nc_buf->download(cache->n_contrib,
                             static_cast<std::size_t>(bytes_ncontrib));
#ifdef GS3D_TESTING
            last_layer1_cache_downloaded_ = true;
#endif
        }

        cache->replay_order_offsets_storage.clear();
        cache->replay_order_gids_storage.clear();
        cache->replay_order_offsets = nullptr;
        cache->replay_order_gids = nullptr;
        cache->replay_order_count = 0u;
        cache->replay_order_offsets_gpu = nullptr;
        cache->replay_order_gids_gpu = nullptr;

        if (eval_3D_ && !eval3d_raw_replay_ && (cache->n_contrib || use_gpu_replay_offsets)) {
            const size_t HW_size = static_cast<size_t>(HW);
            if (use_gpu_replay_offsets) {
                const uint32_t num_wg1 = (HW + 255u) / 256u;
                const uint32_t num_wg2 = (num_wg1 + 255u) / 256u;
                if (!replay_scan_pass_) {
                    replay_scan_pass_ = std::make_unique<PrefixScanPass>(ctx_);
                }
                if (replay_offsets_gpu_capacity_ < HW_size + 1u) {
                    replay_offsets_gpu_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                        static_cast<VkDeviceSize>(HW_size + 1u) * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    replay_offsets_gpu_capacity_ = HW_size + 1u;
                }
                if (replay_scan_wg_capacity_ < num_wg1) {
                    replay_scan_wg_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                        static_cast<VkDeviceSize>(num_wg1) * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    replay_scan_wg_capacity_ = num_wg1;
                }
                if (replay_scan_wg2_capacity_ < num_wg2) {
                    replay_scan_wg2_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                        static_cast<VkDeviceSize>(num_wg2) * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    replay_scan_wg2_capacity_ = num_wg2;
                }
                replay_scan_pass_->bind_buffers_2level(nc_buf->handle(),
                                                       replay_offsets_gpu_buf_->handle(),
                                                       replay_scan_wg_buf_->handle(),
                                                       replay_scan_wg2_buf_->handle());

                VkCommandBuffer scan_cmd = ctx_.allocatePrimary();
                VkCommandBufferBeginInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(scan_cmd, &bi));
                insert_compute_barrier(scan_cmd);
                replay_scan_pass_->record(scan_cmd, HW);
                VK_CHECK(vkEndCommandBuffer(scan_cmd));
                ctx_.submitAndWait(scan_cmd);
                ctx_.freePrimary(scan_cmd);

                uint32_t last_offset = 0u;
                uint32_t last_count = 0u;
                replay_offsets_gpu_buf_->download(&last_offset,
                    sizeof(uint32_t),
                    static_cast<VkDeviceSize>(HW_size - 1u) * sizeof(uint32_t));
                nc_buf->download(&last_count,
                    sizeof(uint32_t),
                    static_cast<VkDeviceSize>(HW_size - 1u) * sizeof(uint32_t));
                const uint64_t total_replay = static_cast<uint64_t>(last_offset) + last_count;
                if (total_replay > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                    throw std::runtime_error("RasterizerVulkan::rasterize: eval_3D replay order exceeds uint32 capacity");
                }
                cache->replay_order_count = static_cast<size_t>(total_replay);
                const uint32_t total_replay_u32 = static_cast<uint32_t>(total_replay);
                replay_offsets_gpu_buf_->upload(&total_replay_u32,
                    sizeof(uint32_t),
                    static_cast<VkDeviceSize>(HW_size) * sizeof(uint32_t));
            } else {
                cache->replay_order_offsets_storage.assign(HW_size + 1u, 0u);
                cache->replay_order_offsets = cache->replay_order_offsets_storage.data();

                uint64_t total_replay = 0u;
                for (size_t px = 0; px < HW_size; ++px) {
                    const int n_px = cache->n_contrib[px];
                    if (n_px < 0) {
                        throw std::runtime_error("RasterizerVulkan::rasterize: negative n_contrib");
                    }
                    total_replay += static_cast<uint32_t>(n_px);
                    if (total_replay > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                        throw std::runtime_error("RasterizerVulkan::rasterize: eval_3D replay order exceeds uint32 capacity");
                    }
                    cache->replay_order_offsets[px + 1u] = static_cast<uint32_t>(total_replay);
                }

                cache->replay_order_count = static_cast<size_t>(total_replay);
                cache->replay_order_gids_storage.resize(cache->replay_order_count);
                cache->replay_order_gids = cache->replay_order_gids_storage.empty()
                    ? nullptr
                    : cache->replay_order_gids_storage.data();

                if (replay_offsets_gpu_capacity_ < HW_size + 1u) {
                    replay_offsets_gpu_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                        static_cast<VkDeviceSize>(HW_size + 1u) * sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    replay_offsets_gpu_capacity_ = HW_size + 1u;
                }
                replay_offsets_gpu_buf_->upload(cache->replay_order_offsets,
                    (HW_size + 1u) * sizeof(uint32_t));
            }
            if (cache->replay_order_count > replay_gids_gpu_capacity_) {
                replay_gids_gpu_buf_ = std::make_unique<VulkanBuffer>(ctx_,
                    static_cast<VkDeviceSize>(cache->replay_order_count == 0u ? 4u : cache->replay_order_count * sizeof(uint32_t)),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                replay_gids_gpu_capacity_ = cache->replay_order_count;
            } else if (!replay_gids_gpu_buf_) {
                replay_gids_gpu_buf_ = std::make_unique<VulkanBuffer>(ctx_, 4u,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            }

            eubo.img_size[2] = 1.0f;
            eval3d_ubo_buf->upload(&eubo, sizeof(eubo));
            rb.raster_eval3d_ubo = eval3d_ubo_buf->handle();
            rb.replay_order_offsets = replay_offsets_gpu_buf_->handle();
            rb.replay_order_gids = replay_gids_gpu_buf_->handle();
            pass_->bind_buffers(rb);
            pass_->dispatch_sync(N_eff,
                                 static_cast<uint32_t>(W), static_cast<uint32_t>(H),
                                 num_tiles_x, num_tiles_y);
            cache->replay_order_offsets_gpu = replay_offsets_gpu_buf_->handle();
            cache->replay_order_gids_gpu = replay_gids_gpu_buf_->handle();
            if (cache->replay_order_count > 0u) {
                if (use_gpu_replay_offsets) {
                    cache->replay_order_gids_storage.clear();
                    cache->replay_order_gids = nullptr;
                } else {
                    replay_gids_gpu_buf_->download(cache->replay_order_gids,
                        cache->replay_order_count * sizeof(uint32_t));
                }
            }
        }
    }

    if (retain_gpu_outputs) {
        r_img_ = std::move(img_buf);
        r_tfinal_ = std::move(t_buf);
        r_ncontrib_ = std::move(nc_buf);
        r_ubo_ = std::move(ubo_buf);
        r_eval3d_ubo_ = std::move(eval3d_ubo_buf);
        r_dummy4_ = std::move(dummy4_buf);
    } else {
        r_img_.reset();
        r_tfinal_.reset();
        r_ncontrib_.reset();
        r_ubo_.reset();
        r_eval3d_ubo_.reset();
        r_dummy4_.reset();
    }
}

// ---------------------------------------------------------------------------
// Layer-2 record-mode: prepare_record / record / download_* / getters.
// ---------------------------------------------------------------------------
void RasterizerVulkan::prepare_record(uint32_t W, uint32_t H,
                                      uint32_t num_tiles_x,
                                      uint32_t num_tiles_y,
                                      const float bg_color[3],
                                      VkBuffer values_sorted,
                                      VkBuffer tile_ranges,
                                      VkBuffer means2D,
                                      VkBuffer conic_opacity_packed,
                                      VkBuffer rgb,
                                      VkBuffer gauss2screen,
                                      VkBuffer opacities_2d,
                                      VkBuffer cov3D_inv,
                                      VkBuffer mean_offset,
                                      const Camera& cam) {
    if (W == 0u || H == 0u)
        throw std::runtime_error(
            "RasterizerVulkan::prepare_record: W/H must be > 0");

    // Release any previously held buffers up front.
    r_img_.reset();
    r_tfinal_.reset();
    r_ncontrib_.reset();
    r_ubo_.reset();
    r_eval3d_ubo_.reset();
    r_dummy4_.reset();

    const uint32_t HW = H * W;
    const VkDeviceSize bytes_img      =
        static_cast<VkDeviceSize>(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal   =
        static_cast<VkDeviceSize>(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib =
        static_cast<VkDeviceSize>(HW) * sizeof(uint32_t);

    r_img_      = std::make_unique<VulkanBuffer>(
        ctx_, bytes_img,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_tfinal_   = std::make_unique<VulkanBuffer>(
        ctx_, bytes_tfinal,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_ncontrib_ = std::make_unique<VulkanBuffer>(
        ctx_, bytes_ncontrib, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_ubo_      = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(sizeof(RasterizeUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    RasterizeUBO ubo{};
    ubo.bg_r = bg_color[0];
    ubo.bg_g = bg_color[1];
    ubo.bg_b = bg_color[2];
    r_ubo_->upload(&ubo, sizeof(ubo));

    // Build RasterEval3DUBO.
    RasterEval3DUBO eubo{};
    {
        float inv_vp[16];
        if (invertMatrix4x4(cam.viewproj_matrix, inv_vp))
            std::memcpy(eubo.inverse_vp, inv_vp, sizeof(inv_vp));
        eubo.cam_pos[0] = cam.cam_pos[0];
        eubo.cam_pos[1] = cam.cam_pos[1];
        eubo.cam_pos[2] = cam.cam_pos[2];
        eubo.cam_pos[3] = 0.0f;
        eubo.img_size[0] = static_cast<float>(W);
        eubo.img_size[1] = static_cast<float>(H);
        eubo.img_size[2] = 0.0f;
        eubo.img_size[3] = 0.0f;
    }
    r_eval3d_ubo_ = std::make_unique<VulkanBuffer>(
        ctx_, sizeof(RasterEval3DUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    r_eval3d_ubo_->upload(&eubo, sizeof(eubo));

    // Dummy buffer for unbound eval_3D SSBOs when eval_3D=false.
    VkBuffer g2s_h  = gauss2screen;
    VkBuffer opa_h  = opacities_2d;
    VkBuffer cov_h  = cov3D_inv;
    VkBuffer mo_h   = mean_offset;
    if (!eval_3D_ || gauss2screen == VK_NULL_HANDLE) {
        r_dummy4_ = std::make_unique<VulkanBuffer>(
            ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        g2s_h = r_dummy4_->handle();
        opa_h = r_dummy4_->handle();
        cov_h = r_dummy4_->handle();
        mo_h  = r_dummy4_->handle();
    }
    if (!r_dummy4_) {
        r_dummy4_ = std::make_unique<VulkanBuffer>(
            ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    RasterizePass::Buffers rb{};
    rb.values_sorted        = values_sorted;
    rb.tile_ranges          = tile_ranges;
    rb.means2D              = means2D;
    rb.conic_opacity_packed = conic_opacity_packed;
    rb.rgb                  = rgb;
    rb.out_image            = r_img_     ->handle();
    rb.transmittance        = r_tfinal_  ->handle();
    rb.n_contrib            = r_ncontrib_->handle();
    rb.raster_ubo           = r_ubo_     ->handle();
    rb.gauss2screen         = g2s_h;
    rb.opacities_2d         = opa_h;
    rb.cov3D_inv            = cov_h;
    rb.mean_offset          = mo_h;
    rb.raster_eval3d_ubo    = r_eval3d_ubo_->handle();
    rb.replay_order_offsets = r_dummy4_->handle();
    rb.replay_order_gids    = r_dummy4_->handle();
    pass_->bind_buffers(rb);

    r_W_   = W;
    r_H_   = H;
    r_ntx_ = num_tiles_x;
    r_nty_ = num_tiles_y;
}

void RasterizerVulkan::record(VkCommandBuffer cmd,
                              uint32_t N_eff, uint32_t W, uint32_t H,
                              uint32_t num_tiles_x, uint32_t num_tiles_y) {
    if (!r_img_)
        throw std::runtime_error(
            "RasterizerVulkan::record called before prepare_record()");
    pass_->record(cmd, N_eff, W, H, num_tiles_x, num_tiles_y);
}

void RasterizerVulkan::download_image(float* dst, uint32_t W, uint32_t H) const {
    if (!r_img_)
        throw std::runtime_error(
            "RasterizerVulkan::download_image called before prepare_record()");
    if (W != r_W_ || H != r_H_)
        throw std::runtime_error(
            "RasterizerVulkan::download_image called with mismatched dimensions");
    const uint32_t HW = H * W;
    const std::size_t bytes =
        static_cast<std::size_t>(HW) * 3u * sizeof(float);
    r_img_->download(dst, bytes);
}

void RasterizerVulkan::download_cache(float* T_final, int* n_contrib,
                                      uint32_t HW) {
    if (!r_img_)
        throw std::runtime_error(
            "RasterizerVulkan::download_cache called before prepare_record()");
    if (T_final) {
        r_tfinal_->download(T_final,
            static_cast<std::size_t>(HW) * sizeof(float));
    }
    if (n_contrib) {
        r_ncontrib_->download(n_contrib,
            static_cast<std::size_t>(HW) * sizeof(uint32_t));
    }
}

VkBuffer RasterizerVulkan::out_image_buf() const {
    return r_img_ ? r_img_->handle() : VK_NULL_HANDLE;
}
VkBuffer RasterizerVulkan::transmittance_buf() const {
    return r_tfinal_ ? r_tfinal_->handle() : VK_NULL_HANDLE;
}
VkBuffer RasterizerVulkan::n_contrib_buf() const {
    return r_ncontrib_ ? r_ncontrib_->handle() : VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Phase 4 / Milestone A: rasterize_traced.
// Mirrors rasterize() but allocates 17 additional trace SSBOs, dispatches
// the spec_trace_enabled=1 pipeline variant, and downloads every trace
// buffer into a TraceDump. The current rasterize.comp body does NOT yet
// emit trace writes (Milestones B..E will), so downloaded data is all
// zeros and only shape/dtype alignment with the CUDA side is guaranteed.
// ---------------------------------------------------------------------------
RasterizerVulkan::TraceDump RasterizerVulkan::rasterize_traced(
    const PreprocessOutput& preprocess,
    const BinningOutput& binning,
    const Camera& camera,
    const RenderConfig& config,
    const std::vector<uint32_t>& selected_tiles,
    float* output_image,
    ForwardCache* cache) {

    TraceDump dump{};
    if (!eval_3D_) {
        throw std::runtime_error(
            "rasterize_traced requires the rasterizer be constructed with "
            "eval_3D=true (matches CUDA hierarchical_render semantics).");
    }

    const int W = camera.width;
    const int H = camera.height;
    if (W <= 0 || H <= 0) return dump;

    const uint32_t num_tiles_x =
        static_cast<uint32_t>((W + config.tile_w - 1) / config.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((H + config.tile_h - 1) / config.tile_h);
    const uint32_t num_tiles = num_tiles_x * num_tiles_y;
    const uint32_t K = static_cast<uint32_t>(selected_tiles.size());
    if (K == 0u) {
        throw std::runtime_error(
            "rasterize_traced: selected_tiles must be non-empty");
    }

    const int R = binning.total_pairs;
    const int HW = H * W;

    // Lazy-build the trace-enabled pass (kept alive across calls).
    if (!traced_pass_) {
        traced_pass_ = std::make_unique<RasterizePass>(
            ctx_, /*spec_eval_3D=*/1u, /*spec_trace_enabled=*/1u);
    }

    // Empty-scene: short-circuit to background color (mirrors rasterize()).
    if (R <= 0 || binning.values_sorted == nullptr) {
        for (int ch = 0; ch < 3; ++ch) {
            const float bg = config.bg_color[ch];
            float* row = output_image + static_cast<std::size_t>(ch) * HW;
            for (int px = 0; px < HW; ++px) row[px] = bg;
        }
        if (cache) {
            if (cache->T_final)    for (int px = 0; px < HW; ++px) cache->T_final[px]    = 1.0f;
            if (cache->n_contrib)  for (int px = 0; px < HW; ++px) cache->n_contrib[px]  = 0;
        }
        // Produce zero-shape-matched trace buffers.
        dump.K         = K;
        dump.num_tiles = num_tiles;
        dump.slot_lookup.assign(num_tiles, -1);
        for (uint32_t k = 0; k < K; ++k) {
            if (selected_tiles[k] < num_tiles) dump.slot_lookup[selected_tiles[k]] = int32_t(k);
        }
        const size_t n_tail = size_t(K) * 512 * 16 * 64;
        const size_t n_mid  = size_t(K) * 1024 * 16 * 4 * 8;
        const size_t n_head = size_t(K) * 256 * 4096;
        const size_t n_cur  = size_t(K) * 256;
        dump.tail_depths.assign(n_tail, 0.f);
        dump.tail_ids   .assign(n_tail, 0);
        dump.tail_wcur  .assign(K, 0u);
        dump.mid_depths .assign(n_mid, 0.f);
        dump.mid_ids    .assign(n_mid, 0);
        dump.mid_wcur   .assign(K, 0u);
        dump.head_ins_depth   .assign(n_head, 0.f);
        dump.head_ins_alpha   .assign(n_head, 0.f);
        dump.head_ins_gid     .assign(n_head, 0);
        dump.head_ins_cursor  .assign(n_cur, 0u);
        dump.head_blend_depth .assign(n_head, 0.f);
        dump.head_blend_alpha .assign(n_head, 0.f);
        dump.head_blend_T     .assign(n_head, 0.f);
        dump.head_blend_gid   .assign(n_head, 0);
        dump.head_blend_cursor.assign(n_cur, 0u);
        return dump;
    }

    // ---- N_eff + conic_opacity packing (same as rasterize()) -------------
    uint32_t max_gid = 0u;
    for (int i = 0; i < R; ++i) {
        const uint32_t g = binning.values_sorted[i];
        if (g > max_gid) max_gid = g;
    }
    const uint32_t N_eff = max_gid + 1u;

    std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N_eff) * 4u);
    for (uint32_t i = 0; i < N_eff; ++i) {
        const std::size_t dst = static_cast<std::size_t>(i) * 4u;
        const std::size_t src = static_cast<std::size_t>(i) * 3u;
        conic_opacity_packed[dst + 0] = preprocess.conics[src + 0];
        conic_opacity_packed[dst + 1] = preprocess.conics[src + 1];
        conic_opacity_packed[dst + 2] = preprocess.conics[src + 2];
        conic_opacity_packed[dst + 3] = preprocess.opacities_2d[i];
    }

    // ---- Core per-Gaussian + output buffers ------------------------------
    const VkDeviceSize bytes_vs       = VkDeviceSize(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_tr       = VkDeviceSize(binning.num_tiles) * 2u * sizeof(uint32_t);
    const VkDeviceSize bytes_m2d      = VkDeviceSize(N_eff) * 2u * sizeof(float);
    const VkDeviceSize bytes_co       = VkDeviceSize(N_eff) * 4u * sizeof(float);
    const VkDeviceSize bytes_rgb      = VkDeviceSize(N_eff) * 3u * sizeof(float);
    const VkDeviceSize bytes_img      = VkDeviceSize(HW) * 3u * sizeof(float);
    const VkDeviceSize bytes_tfinal   = VkDeviceSize(HW) * sizeof(float);
    const VkDeviceSize bytes_ncontrib = VkDeviceSize(HW) * sizeof(uint32_t);
    const VkDeviceSize bytes_g2s      = VkDeviceSize(N_eff) * 16u * sizeof(float);
    const VkDeviceSize bytes_opa2d    = VkDeviceSize(N_eff) * sizeof(float);
    const VkDeviceSize bytes_cov3i    = VkDeviceSize(N_eff) * 6u * sizeof(float);
    const VkDeviceSize bytes_mo       = VkDeviceSize(N_eff) * 3u * sizeof(float);

    auto vs_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_vs,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tr_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_tr,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto m2d_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_m2d,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto co_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_co,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rgb_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_rgb,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto img_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_img,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto t_buf        = std::make_unique<VulkanBuffer>(ctx_, bytes_tfinal,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto nc_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_ncontrib, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ubo_buf      = std::make_unique<VulkanBuffer>(ctx_, sizeof(RasterizeUBO),     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    auto g2s_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_g2s,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto opa2d_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_opa2d,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cov3i_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_cov3i,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mo_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_mo,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto e3d_ubo_buf  = std::make_unique<VulkanBuffer>(ctx_, sizeof(RasterEval3DUBO),  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    vs_buf ->upload(binning.values_sorted, static_cast<std::size_t>(bytes_vs));
    tr_buf ->upload(binning.tile_ranges,   static_cast<std::size_t>(bytes_tr));
    m2d_buf->upload(preprocess.means2D,    static_cast<std::size_t>(bytes_m2d));
    co_buf ->upload(conic_opacity_packed.data(), static_cast<std::size_t>(bytes_co));
    rgb_buf->upload(preprocess.rgb,        static_cast<std::size_t>(bytes_rgb));
    g2s_buf   ->upload(preprocess.gauss2screen, static_cast<std::size_t>(bytes_g2s));
    opa2d_buf ->upload(preprocess.opacities_2d, static_cast<std::size_t>(bytes_opa2d));
    cov3i_buf ->upload(preprocess.cov3D_inv,    static_cast<std::size_t>(bytes_cov3i));
    mo_buf    ->upload(preprocess.mean_offset,  static_cast<std::size_t>(bytes_mo));

    RasterizeUBO ubo{};
    ubo.bg_r = config.bg_color[0];
    ubo.bg_g = config.bg_color[1];
    ubo.bg_b = config.bg_color[2];
    ubo_buf->upload(&ubo, sizeof(ubo));

    RasterEval3DUBO eubo{};
    {
        float inv_vp[16];
        if (invertMatrix4x4(camera.viewproj_matrix, inv_vp))
            std::memcpy(eubo.inverse_vp, inv_vp, sizeof(inv_vp));
        eubo.cam_pos[0] = camera.cam_pos[0];
        eubo.cam_pos[1] = camera.cam_pos[1];
        eubo.cam_pos[2] = camera.cam_pos[2];
        eubo.cam_pos[3] = 0.0f;
        eubo.img_size[0] = static_cast<float>(W);
        eubo.img_size[1] = static_cast<float>(H);
        eubo.img_size[2] = 0.0f;
        eubo.img_size[3] = 0.0f;
    }
    e3d_ubo_buf->upload(&eubo, sizeof(eubo));

    // ---- Cascade trace SSBOs (bindings 14..30) ---------------------------
    // Sizes follow cascade_trace.h consts; K=selected_tiles.size().
    const size_t n_tail_elems = size_t(K) * 512 * 16 * 64;
    const size_t n_mid_elems  = size_t(K) * 1024 * 16 * 4 * 8;
    const size_t n_head_elems = size_t(K) * 256 * 4096;
    const size_t n_cur_elems  = size_t(K) * 256;

    const VkDeviceSize bytes_meta        = sizeof(TraceMetaUBO);
    const VkDeviceSize bytes_slot_lookup = VkDeviceSize(num_tiles) * sizeof(int32_t);
    const VkDeviceSize bytes_tail_f      = VkDeviceSize(n_tail_elems) * sizeof(float);
    const VkDeviceSize bytes_tail_i      = VkDeviceSize(n_tail_elems) * sizeof(int32_t);
    const VkDeviceSize bytes_tail_wcur   = VkDeviceSize(K) * sizeof(uint32_t);
    const VkDeviceSize bytes_mid_f       = VkDeviceSize(n_mid_elems) * sizeof(float);
    const VkDeviceSize bytes_mid_i       = VkDeviceSize(n_mid_elems) * sizeof(int32_t);
    const VkDeviceSize bytes_mid_wcur    = VkDeviceSize(K) * sizeof(uint32_t);
    const VkDeviceSize bytes_head_f      = VkDeviceSize(n_head_elems) * sizeof(float);
    const VkDeviceSize bytes_head_i      = VkDeviceSize(n_head_elems) * sizeof(int32_t);
    const VkDeviceSize bytes_head_cur    = VkDeviceSize(n_cur_elems) * sizeof(uint32_t);

    auto meta_buf           = std::make_unique<VulkanBuffer>(ctx_, bytes_meta,        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    auto slot_lookup_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_slot_lookup, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tail_depths_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_tail_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tail_ids_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_tail_i,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tail_wcur_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_tail_wcur,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mid_depths_buf     = std::make_unique<VulkanBuffer>(ctx_, bytes_mid_f,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mid_ids_buf        = std::make_unique<VulkanBuffer>(ctx_, bytes_mid_i,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mid_wcur_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_mid_wcur,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hi_depth_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_head_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hi_alpha_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_head_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hi_gid_buf         = std::make_unique<VulkanBuffer>(ctx_, bytes_head_i,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hi_cursor_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_head_cur,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hb_depth_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_head_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hb_alpha_buf       = std::make_unique<VulkanBuffer>(ctx_, bytes_head_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hb_T_buf           = std::make_unique<VulkanBuffer>(ctx_, bytes_head_f,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hb_gid_buf         = std::make_unique<VulkanBuffer>(ctx_, bytes_head_i,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hb_cursor_buf      = std::make_unique<VulkanBuffer>(ctx_, bytes_head_cur,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Upload TraceMetaUBO.
    {
        TraceMetaUBO tm{};
        tm.K = K;
        tm.num_tiles = num_tiles;
        meta_buf->upload(&tm, sizeof(tm));
    }

    // Zero-fill wcur/cursor/depth/alpha/T SSBOs. Init id SSBOs to -1 (CUDA
    // semantics: tail_ids/mid_ids/head_*_gid use -1 as "invalid slot"
    // sentinel; CUDA pre-fills tail/mid with -1 via torch::full(-1) so the
    // VK trace must do the same for unwritten slots to compare cleanly).
    slot_lookup_buf->zero_fill(static_cast<std::size_t>(bytes_slot_lookup));
    tail_depths_buf->zero_fill(static_cast<std::size_t>(bytes_tail_f));
    tail_wcur_buf  ->zero_fill(static_cast<std::size_t>(bytes_tail_wcur));
    mid_depths_buf ->zero_fill(static_cast<std::size_t>(bytes_mid_f));
    mid_wcur_buf   ->zero_fill(static_cast<std::size_t>(bytes_mid_wcur));
    hi_depth_buf   ->zero_fill(static_cast<std::size_t>(bytes_head_f));
    hi_alpha_buf   ->zero_fill(static_cast<std::size_t>(bytes_head_f));
    hi_cursor_buf  ->zero_fill(static_cast<std::size_t>(bytes_head_cur));
    hb_depth_buf   ->zero_fill(static_cast<std::size_t>(bytes_head_f));
    hb_alpha_buf   ->zero_fill(static_cast<std::size_t>(bytes_head_f));
    hb_T_buf       ->zero_fill(static_cast<std::size_t>(bytes_head_f));
    hb_cursor_buf  ->zero_fill(static_cast<std::size_t>(bytes_head_cur));
    // Fill *_gid buffers with -1 (int32). CUDA pre-fills with torch::full(-1);
    // uninitialized slots must report -1 so the comparator doesn't see them
    // as phantom gid=0 entries.
    {
        std::vector<int32_t> neg1_tail(n_tail_elems, -1);
        std::vector<int32_t> neg1_mid (n_mid_elems,  -1);
        std::vector<int32_t> neg1_head(n_head_elems, -1);
        tail_ids_buf->upload(neg1_tail.data(), neg1_tail.size() * sizeof(int32_t));
        mid_ids_buf ->upload(neg1_mid .data(), neg1_mid .size() * sizeof(int32_t));
        hi_gid_buf  ->upload(neg1_head.data(), neg1_head.size() * sizeof(int32_t));
        hb_gid_buf  ->upload(neg1_head.data(), neg1_head.size() * sizeof(int32_t));
    }

    // Populate slot_lookup: -1 for every tile, then 0..K-1 for selected.
    {
        std::vector<int32_t> sl(num_tiles, -1);
        for (uint32_t k = 0; k < K; ++k) {
            if (selected_tiles[k] < num_tiles) sl[selected_tiles[k]] = int32_t(k);
        }
        slot_lookup_buf->upload(sl.data(), sl.size() * sizeof(int32_t));
    }

    auto replay_dummy_buf = std::make_unique<VulkanBuffer>(ctx_, 4u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // ---- Bind + dispatch -------------------------------------------------
    RasterizePass::Buffers rb{};
    rb.values_sorted        = vs_buf   ->handle();
    rb.tile_ranges          = tr_buf   ->handle();
    rb.means2D              = m2d_buf  ->handle();
    rb.conic_opacity_packed = co_buf   ->handle();
    rb.rgb                  = rgb_buf  ->handle();
    rb.out_image            = img_buf  ->handle();
    rb.transmittance        = t_buf    ->handle();
    rb.n_contrib            = nc_buf   ->handle();
    rb.raster_ubo           = ubo_buf  ->handle();
    rb.gauss2screen         = g2s_buf  ->handle();
    rb.opacities_2d         = opa2d_buf->handle();
    rb.cov3D_inv            = cov3i_buf->handle();
    rb.mean_offset          = mo_buf   ->handle();
    rb.raster_eval3d_ubo    = e3d_ubo_buf->handle();
    rb.replay_order_offsets = replay_dummy_buf->handle();
    rb.replay_order_gids    = replay_dummy_buf->handle();
    traced_pass_->bind_buffers(rb);

    RasterizePass::TraceBuffers tb{};
    tb.trace_meta_ubo    = meta_buf       ->handle();
    tb.slot_lookup       = slot_lookup_buf->handle();
    tb.tail_depths       = tail_depths_buf->handle();
    tb.tail_ids          = tail_ids_buf   ->handle();
    tb.tail_wcur         = tail_wcur_buf  ->handle();
    tb.mid_depths        = mid_depths_buf ->handle();
    tb.mid_ids           = mid_ids_buf    ->handle();
    tb.mid_wcur          = mid_wcur_buf   ->handle();
    tb.head_ins_depth    = hi_depth_buf   ->handle();
    tb.head_ins_alpha    = hi_alpha_buf   ->handle();
    tb.head_ins_gid      = hi_gid_buf     ->handle();
    tb.head_ins_cursor   = hi_cursor_buf  ->handle();
    tb.head_blend_depth  = hb_depth_buf   ->handle();
    tb.head_blend_alpha  = hb_alpha_buf   ->handle();
    tb.head_blend_T      = hb_T_buf       ->handle();
    tb.head_blend_gid    = hb_gid_buf     ->handle();
    tb.head_blend_cursor = hb_cursor_buf  ->handle();
    traced_pass_->bind_trace_buffers(tb);

    traced_pass_->dispatch_sync(N_eff,
                                static_cast<uint32_t>(W),
                                static_cast<uint32_t>(H),
                                num_tiles_x, num_tiles_y);

    // ---- Download outputs (image + cache + trace) -----------------------
    img_buf->download(output_image, static_cast<std::size_t>(bytes_img));

    // CHW -> HWC (match rasterize() contract for the image).
    {
        std::vector<float> chw(static_cast<std::size_t>(HW) * 3u);
        std::memcpy(chw.data(), output_image,
                    static_cast<std::size_t>(HW) * 3u * sizeof(float));
        for (int px = 0; px < HW; ++px) {
            for (int ch = 0; ch < 3; ++ch) {
                output_image[static_cast<std::size_t>(px) * 3 + ch] =
                    chw[static_cast<std::size_t>(ch) * HW + px];
            }
        }
    }

    if (cache) {
        if (cache->T_final) {
            t_buf->download(cache->T_final, static_cast<std::size_t>(bytes_tfinal));
        }
        if (cache->n_contrib) {
            nc_buf->download(cache->n_contrib, static_cast<std::size_t>(bytes_ncontrib));
        }
    }

    // Download trace buffers into TraceDump vectors.
    dump.K         = K;
    dump.num_tiles = num_tiles;

    dump.slot_lookup.resize(num_tiles);
    slot_lookup_buf->download(dump.slot_lookup.data(), static_cast<std::size_t>(bytes_slot_lookup));

    dump.tail_depths.resize(n_tail_elems);
    dump.tail_ids   .resize(n_tail_elems);
    dump.tail_wcur  .resize(K);
    tail_depths_buf->download(dump.tail_depths.data(), static_cast<std::size_t>(bytes_tail_f));
    tail_ids_buf   ->download(dump.tail_ids   .data(), static_cast<std::size_t>(bytes_tail_i));
    tail_wcur_buf  ->download(dump.tail_wcur  .data(), static_cast<std::size_t>(bytes_tail_wcur));

    dump.mid_depths.resize(n_mid_elems);
    dump.mid_ids   .resize(n_mid_elems);
    dump.mid_wcur  .resize(K);
    mid_depths_buf->download(dump.mid_depths.data(), static_cast<std::size_t>(bytes_mid_f));
    mid_ids_buf   ->download(dump.mid_ids   .data(), static_cast<std::size_t>(bytes_mid_i));
    mid_wcur_buf  ->download(dump.mid_wcur  .data(), static_cast<std::size_t>(bytes_mid_wcur));

    dump.head_ins_depth  .resize(n_head_elems);
    dump.head_ins_alpha  .resize(n_head_elems);
    dump.head_ins_gid    .resize(n_head_elems);
    dump.head_ins_cursor .resize(n_cur_elems);
    hi_depth_buf ->download(dump.head_ins_depth .data(), static_cast<std::size_t>(bytes_head_f));
    hi_alpha_buf ->download(dump.head_ins_alpha .data(), static_cast<std::size_t>(bytes_head_f));
    hi_gid_buf   ->download(dump.head_ins_gid   .data(), static_cast<std::size_t>(bytes_head_i));
    hi_cursor_buf->download(dump.head_ins_cursor.data(), static_cast<std::size_t>(bytes_head_cur));

    dump.head_blend_depth  .resize(n_head_elems);
    dump.head_blend_alpha  .resize(n_head_elems);
    dump.head_blend_T      .resize(n_head_elems);
    dump.head_blend_gid    .resize(n_head_elems);
    dump.head_blend_cursor .resize(n_cur_elems);
    hb_depth_buf ->download(dump.head_blend_depth .data(), static_cast<std::size_t>(bytes_head_f));
    hb_alpha_buf ->download(dump.head_blend_alpha .data(), static_cast<std::size_t>(bytes_head_f));
    hb_T_buf     ->download(dump.head_blend_T     .data(), static_cast<std::size_t>(bytes_head_f));
    hb_gid_buf   ->download(dump.head_blend_gid   .data(), static_cast<std::size_t>(bytes_head_i));
    hb_cursor_buf->download(dump.head_blend_cursor.data(), static_cast<std::size_t>(bytes_head_cur));

    return dump;
}
