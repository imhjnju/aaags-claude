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
#include "vulkan/preprocess_bindings.h"   // RasterEval3DUBO
#include "math_utils.h"                   // invertMatrix4x4

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

RasterizerVulkan::RasterizerVulkan(VulkanContext& ctx,
                                   bool eval_3D,
                                   bool disable_subtile_resort)
    : ctx_(ctx), eval_3D_(eval_3D), disable_subtile_resort_(disable_subtile_resort) {
    pass_ = std::make_unique<RasterizePass>(
        ctx_,
        eval_3D ? 1u : 0u,
        disable_subtile_resort ? 1u : 0u);
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

    // -------------------------------------------------------------------
    // Empty-scene fast path.
    // -------------------------------------------------------------------
    // rasterize.comp's final contribution for every pixel is T*bg_color.
    // With no Gaussians to blend, T stays at 1 and the whole image is
    // bg_color. We short-circuit on the host to avoid a GPU round-trip.
    if (R <= 0 || binning.values_sorted == nullptr) {
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
        }
        return;
    }

    // -------------------------------------------------------------------
    // Compute N_eff: 1 + max index referenced in values_sorted.
    // -------------------------------------------------------------------
    // The shader only reads per-Gaussian data at indices appearing in
    // values_sorted, so we only need to upload up to max_gid+1. This keeps
    // uploads tight on sparse scenes.
    uint32_t max_gid = 0u;
    for (int i = 0; i < R; ++i) {
        const uint32_t g = binning.values_sorted[i];
        if (g > max_gid) max_gid = g;
    }
    const uint32_t N_eff = max_gid + 1u;

    // -------------------------------------------------------------------
    // Repack per-Gaussian conics (Nx3) + opacities_2d (Nx1) → packed (Nx4).
    // -------------------------------------------------------------------
    // PreprocessOutput stores these separately (conics[N*3] = {a,b,c},
    // opacities_2d[N]). rasterize.comp reads them as one interleaved buffer
    // conic_opacity[N*4] = {a, b, c, opacity}. Matches the layout produced by
    // preprocess.comp (binding CONIC_OPACITY_PACKED).
    std::vector<float> conic_opacity_packed(static_cast<std::size_t>(N_eff) * 4u);
    for (uint32_t i = 0; i < N_eff; ++i) {
        const std::size_t dst = static_cast<std::size_t>(i) * 4u;
        const std::size_t src = static_cast<std::size_t>(i) * 3u;
        conic_opacity_packed[dst + 0] = preprocess.conics[src + 0];
        conic_opacity_packed[dst + 1] = preprocess.conics[src + 1];
        conic_opacity_packed[dst + 2] = preprocess.conics[src + 2];
        conic_opacity_packed[dst + 3] = preprocess.opacities_2d[i];
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

    auto vs_buf  = std::make_unique<VulkanBuffer>(
        ctx_, bytes_vs,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tr_buf  = std::make_unique<VulkanBuffer>(
        ctx_, bytes_tr,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto m2d_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_m2d,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto co_buf  = std::make_unique<VulkanBuffer>(
        ctx_, bytes_co,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rgb_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_rgb,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto img_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_img,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto t_buf   = std::make_unique<VulkanBuffer>(
        ctx_, bytes_tfinal,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto nc_buf  = std::make_unique<VulkanBuffer>(
        ctx_, bytes_ncontrib, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ubo_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(sizeof(RasterizeUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // -------------------------------------------------------------------
    // Upload inputs.
    // -------------------------------------------------------------------
    vs_buf ->upload(binning.values_sorted,
                    static_cast<std::size_t>(bytes_vs));
    tr_buf ->upload(binning.tile_ranges,
                    static_cast<std::size_t>(bytes_tr));
    m2d_buf->upload(preprocess.means2D,
                    static_cast<std::size_t>(bytes_m2d));
    co_buf ->upload(conic_opacity_packed.data(),
                    static_cast<std::size_t>(bytes_co));
    rgb_buf->upload(preprocess.rgb,
                    static_cast<std::size_t>(bytes_rgb));

    RasterizeUBO ubo{};
    ubo.bg_r = config.bg_color[0];
    ubo.bg_g = config.bg_color[1];
    ubo.bg_b = config.bg_color[2];
    ubo_buf->upload(&ubo, sizeof(ubo));

    // -------------------------------------------------------------------
    // eval_3D rasterize buffers (gauss2screen, opacities_2d, cov3D_inv,
    // mean_offset, RasterEval3DUBO).
    // -------------------------------------------------------------------
    std::unique_ptr<VulkanBuffer> g2s_buf, opa2d_buf, r_cov3d_buf, r_mo_buf;
    std::unique_ptr<VulkanBuffer> eval3d_ubo_buf;
    std::unique_ptr<VulkanBuffer> dummy4_buf;

    if (eval_3D_ && preprocess.gauss2screen && preprocess.opacities_2d) {
        g2s_buf = std::make_unique<VulkanBuffer>(ctx_,
            static_cast<VkDeviceSize>(N_eff) * 16u * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        g2s_buf->upload(preprocess.gauss2screen,
            static_cast<std::size_t>(N_eff) * 16u * sizeof(float));
        opa2d_buf = std::make_unique<VulkanBuffer>(ctx_,
            static_cast<VkDeviceSize>(N_eff) * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        opa2d_buf->upload(preprocess.opacities_2d,
            static_cast<std::size_t>(N_eff) * sizeof(float));
        r_cov3d_buf = std::make_unique<VulkanBuffer>(ctx_,
            static_cast<VkDeviceSize>(N_eff) * 6u * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        r_cov3d_buf->upload(preprocess.cov3D_inv,
            static_cast<std::size_t>(N_eff) * 6u * sizeof(float));
        r_mo_buf = std::make_unique<VulkanBuffer>(ctx_,
            static_cast<VkDeviceSize>(N_eff) * 3u * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        r_mo_buf->upload(preprocess.mean_offset,
            static_cast<std::size_t>(N_eff) * 3u * sizeof(float));
    } else {
        dummy4_buf = std::make_unique<VulkanBuffer>(ctx_, 4u,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        g2s_buf   = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        opa2d_buf = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        r_cov3d_buf = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        r_mo_buf  = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    // Build RasterEval3DUBO.
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
    eval3d_ubo_buf = std::make_unique<VulkanBuffer>(ctx_,
        sizeof(RasterEval3DUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    // img_size[2] is an internal replay-sideband switch. The first dispatch
    // renders normally and only populates n_contrib; a second optional dispatch
    // below sets it to 1 and writes exact per-pixel contribution order.
    eubo.img_size[2] = 0.0f;
    eval3d_ubo_buf->upload(&eubo, sizeof(eubo));

    auto replay_offsets_buf = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto replay_gids_buf    = std::make_unique<VulkanBuffer>(ctx_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // -------------------------------------------------------------------
    // Bind and dispatch.
    // -------------------------------------------------------------------
    RasterizePass::Buffers rb{};
    rb.values_sorted        = vs_buf ->handle();
    rb.tile_ranges          = tr_buf ->handle();
    rb.means2D              = m2d_buf->handle();
    rb.conic_opacity_packed = co_buf ->handle();
    rb.rgb                  = rgb_buf->handle();
    rb.out_image            = img_buf->handle();
    rb.transmittance        = t_buf  ->handle();
    rb.n_contrib            = nc_buf ->handle();
    rb.raster_ubo           = ubo_buf->handle();
    rb.gauss2screen         = g2s_buf->handle();
    rb.opacities_2d         = opa2d_buf->handle();
    rb.cov3D_inv            = r_cov3d_buf->handle();
    rb.mean_offset          = r_mo_buf->handle();
    rb.raster_eval3d_ubo    = eval3d_ubo_buf->handle();
    rb.replay_order_offsets = replay_offsets_buf->handle();
    rb.replay_order_gids    = replay_gids_buf->handle();
    pass_->bind_buffers(rb);
    pass_->dispatch_sync(N_eff,
                         static_cast<uint32_t>(W), static_cast<uint32_t>(H),
                         num_tiles_x, num_tiles_y);

    // -------------------------------------------------------------------
    // Download outputs.
    // -------------------------------------------------------------------
    img_buf->download(output_image, static_cast<std::size_t>(bytes_img));

    if (cache) {
        if (cache->T_final) {
            t_buf->download(cache->T_final,
                            static_cast<std::size_t>(bytes_tfinal));
        }
        if (cache->n_contrib) {
            // ForwardCache::n_contrib is int*; the shader writes uint32 of the
            // same width so a raw copy is correct — no sign-reinterp happens
            // because counts fit well under 2^31.
            nc_buf->download(cache->n_contrib,
                             static_cast<std::size_t>(bytes_ncontrib));
        }

        if (eval_3D_ && !disable_subtile_resort_ && cache->n_contrib) {
            // Exact non-parity eval_3D backward needs the real forward HEAD
            // flush order, not raw values_sorted. Build per-pixel offsets from
            // the just-rendered blended counts, rerun the same forward shader in
            // sideband-write mode, and preserve the order in ForwardCache-owned memory.
            // Evidence: spec/README.md §6 + investigations/eval3d_nonparity_backward_replay.md.
            const size_t HW_size = static_cast<size_t>(HW);
            cache->replay_order_offsets_storage.assign(HW_size + 1u, 0u);
            cache->replay_order_gids_storage.clear();
            cache->replay_order_offsets = cache->replay_order_offsets_storage.data();
            cache->replay_order_gids = nullptr;
            cache->replay_order_count = 0u;
            uint64_t total_replay = 0u;
            for (size_t px = 0; px < HW_size; ++px) {
                const int n_px = cache->n_contrib[px];
                if (n_px < 0) throw std::runtime_error("RasterizerVulkan::rasterize: negative n_contrib");
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

            replay_offsets_buf = std::make_unique<VulkanBuffer>(ctx_,
                static_cast<VkDeviceSize>(HW_size + 1u) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            replay_offsets_buf->upload(cache->replay_order_offsets,
                (HW_size + 1u) * sizeof(uint32_t));
            replay_gids_buf = std::make_unique<VulkanBuffer>(ctx_,
                std::max<VkDeviceSize>(4u, static_cast<VkDeviceSize>(total_replay) * sizeof(uint32_t)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

            eubo.img_size[2] = 1.0f;
            eval3d_ubo_buf->upload(&eubo, sizeof(eubo));
            rb.raster_eval3d_ubo = eval3d_ubo_buf->handle();
            rb.replay_order_offsets = replay_offsets_buf->handle();
            rb.replay_order_gids = replay_gids_buf->handle();
            pass_->bind_buffers(rb);
            pass_->dispatch_sync(N_eff,
                                 static_cast<uint32_t>(W), static_cast<uint32_t>(H),
                                 num_tiles_x, num_tiles_y);
            if (total_replay > 0u) {
                replay_gids_buf->download(cache->replay_order_gids,
                    static_cast<size_t>(total_replay) * sizeof(uint32_t));
            }
            // Keep image/T/n from the first pass; the second pass is deterministic
            // and exists only to materialize replay_order_gids.
        }
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

void RasterizerVulkan::download_image(float* dst, uint32_t W, uint32_t H) {
    if (!r_img_)
        throw std::runtime_error(
            "RasterizerVulkan::download_image called before prepare_record()");
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
