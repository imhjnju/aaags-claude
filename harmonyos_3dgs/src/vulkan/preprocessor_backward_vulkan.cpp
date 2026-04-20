// SP-3 T23: PreprocessorBackwardVulkan — high-level backward preprocessor adapter.
//
// Uploads all CPU-side inputs to host-visible SSBOs, zero-fills gradient
// output buffers (preprocess_backward.comp writes directly, not atomicAdd),
// dispatches preprocess_backward.comp via PreprocessBackwardPass, then
// downloads the resulting gradients into the caller-provided grads arrays.
//
// Buffer-size contract (matches the shader std430 layouts):
//   positions     : N * 3 * sizeof(float)
//   radii         : N * sizeof(int)
//   cov3D         : N * 6 * sizeof(float)
//   d_conics      : N * 3 * sizeof(float)
//   d_opacity     : N * sizeof(float)
//   sh_coeffs     : N * max_coeffs * 3 * sizeof(float)
//   scales        : N * 3 * sizeof(float)
//   rotations     : N * 4 * sizeof(float)
//   d_rgb         : N * 3 * sizeof(float)
//   d_means2D     : N * 2 * sizeof(float)
//   d_means3D     : N * 3 * sizeof(float)    zeroed
//   d_sh          : N * max_coeffs * 3 * sizeof(float)  zeroed
//   d_scales      : N * 3 * sizeof(float)    zeroed
//   d_rotations   : N * 4 * sizeof(float)    zeroed
//   preproc_bwd_ubo : sizeof(PreprocessBackwardUBO) = 192  UNIFORM_BUFFER

#include "vulkan/preprocessor_backward_vulkan.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/backward_bindings.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

PreprocessorBackwardVulkan::PreprocessorBackwardVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    pass_ = std::make_unique<PreprocessBackwardPass>(ctx_);
}

PreprocessorBackwardVulkan::~PreprocessorBackwardVulkan() = default;

void PreprocessorBackwardVulkan::prepare_for_n(int N, int K) {
    if (N <= buf_N_ && K <= buf_K_) return;

    const int N_new = std::max(N, buf_N_);
    const int K_new = std::max(K, buf_K_);

    const VkDeviceSize sz_N3 = static_cast<VkDeviceSize>(N_new) * 3u * sizeof(float);
    const VkDeviceSize sz_N4 = static_cast<VkDeviceSize>(N_new) * 4u * sizeof(float);
    const VkDeviceSize sz_N2 = static_cast<VkDeviceSize>(N_new) * 2u * sizeof(float);
    const VkDeviceSize sz_N6 = static_cast<VkDeviceSize>(N_new) * 6u * sizeof(float);
    const VkDeviceSize sz_N  = static_cast<VkDeviceSize>(N_new) * sizeof(float);
    const VkDeviceSize sz_Ni = static_cast<VkDeviceSize>(N_new) * sizeof(int32_t);
    const VkDeviceSize sz_sh = static_cast<VkDeviceSize>(N_new) * static_cast<VkDeviceSize>(K_new) * 3u * sizeof(float);

    pos_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rad_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_Ni, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    cv3_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_N6, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dcon_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dopa_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    sh_buf_        = std::make_unique<VulkanBuffer>(ctx_, sz_sh, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    sc_buf_        = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rot_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    drgb_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dm2d_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    opa_in_buf_    = std::make_unique<VulkanBuffer>(ctx_, sz_N,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_rot_buf_   = std::make_unique<VulkanBuffer>(ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m2d_cache_buf_ = std::make_unique<VulkanBuffer>(ctx_, sz_N2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    pview_in_buf_  = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    cov2d_in_buf_  = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c2ddet_in_buf_ = std::make_unique<VulkanBuffer>(ctx_, sz_N,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    phomw_in_buf_  = std::make_unique<VulkanBuffer>(ctx_, sz_N,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dm3d_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dsh_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_sh, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dsc_buf_       = std::make_unique<VulkanBuffer>(ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    drot_buf_      = std::make_unique<VulkanBuffer>(ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    d_raw_opa_buf_ = std::make_unique<VulkanBuffer>(ctx_, sz_N,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ubo_buf_       = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(sizeof(PreprocessBackwardUBO)),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    buf_N_ = N_new; buf_K_ = K_new;
}

void PreprocessorBackwardVulkan::backward(const GaussianData& g,
                                           int num_gaussians,
                                           const Camera& cam,
                                           const RenderConfig& cfg,
                                           const ForwardCache& cache,
                                           const RasterGradOutput& rgrad,
                                           const RawGaussianParams& raw,
                                           GradientOutput& grads,
                                           FrameAllocator& alloc) {
    // --- Guard: eval_3D path not implemented --------------------------------
    if (cache.pre && cache.pre->eval_3D)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward: eval_3D=true is not supported");

    const int N = num_gaussians;
    const int K = g.max_coeffs;  // (sh_degree+1)^2

    // Allocate and zero grads output arrays.
    grads.allocate_and_zero(alloc, N, K);

    if (N == 0) return;

    prepare_for_n(N, K);

    // Compute UBO parameters
    const float h_x = static_cast<float>(cam.width)  / (2.0f * cam.tan_fovx);
    const float h_y = static_cast<float>(cam.height) / (2.0f * cam.tan_fovy);

    // --- Compute buffer sizes -----------------------------------------------
    const VkDeviceSize bytes_pos     = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_radii   = static_cast<VkDeviceSize>(N) * sizeof(int);
    const VkDeviceSize bytes_cov3D   = static_cast<VkDeviceSize>(N) * 6u * sizeof(float);
    const VkDeviceSize bytes_d_con   = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_opa   = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_sh      = static_cast<VkDeviceSize>(N) * K * 3u * sizeof(float);
    const VkDeviceSize bytes_scales  = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_rot     = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_d_rgb   = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_m2d   = static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_d_m3d   = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sh    = static_cast<VkDeviceSize>(N) * K * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sc    = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_rot   = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_N_float = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_raw_rot    = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_m2d_cache  = static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_p_view     = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_cov2d      = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_c2d_det    = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_phomw      = static_cast<VkDeviceSize>(N) * sizeof(float);

    // Validate that the required cache fields are populated.
    // They are filled by PreprocessorCPU::process() or PreprocessorVulkan::download_cache().
    if (!cache.p_view)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward: cache.p_view is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.cov2D)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward: cache.cov2D is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.cov2D_det)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward: cache.cov2D_det is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.p_hom_w)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward: cache.p_hom_w is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");

    // --- Upload inputs -------------------------------------------------------
    pos_buf_ ->upload(g.positions,                   static_cast<std::size_t>(bytes_pos));
    rad_buf_ ->upload(cache.pre->radii,               static_cast<std::size_t>(bytes_radii));
    cv3_buf_ ->upload(cache.cov3D,                    static_cast<std::size_t>(bytes_cov3D));
    dcon_buf_->upload(rgrad.d_conics,                 static_cast<std::size_t>(bytes_d_con));
    dopa_buf_->upload(rgrad.d_opacities_2d,           static_cast<std::size_t>(bytes_d_opa));  // d_opacity uploaded but not read by shader — SP-3 scope
    sh_buf_  ->upload(g.sh_coeffs,                    static_cast<std::size_t>(bytes_sh));
    sc_buf_  ->upload(g.scales,                       static_cast<std::size_t>(bytes_scales));
    rot_buf_ ->upload(g.rotations,                    static_cast<std::size_t>(bytes_rot));
    drgb_buf_->upload(rgrad.d_rgb,                    static_cast<std::size_t>(bytes_d_rgb));
    dm2d_buf_->upload(rgrad.d_means2D,               static_cast<std::size_t>(bytes_d_m2d));
    opa_in_buf_->upload(g.opacities,                 static_cast<std::size_t>(bytes_N_float));
    raw_rot_buf_->upload(raw.raw_rotations,           static_cast<std::size_t>(bytes_raw_rot));
    m2d_cache_buf_->upload(cache.pre->means2D,        static_cast<std::size_t>(bytes_m2d_cache));
    pview_in_buf_ ->upload(cache.p_view,              static_cast<std::size_t>(bytes_p_view));
    cov2d_in_buf_ ->upload(cache.cov2D,               static_cast<std::size_t>(bytes_cov2d));
    c2ddet_in_buf_->upload(cache.cov2D_det,           static_cast<std::size_t>(bytes_c2d_det));
    phomw_in_buf_ ->upload(cache.p_hom_w,             static_cast<std::size_t>(bytes_phomw));

    // Zero-fill gradient output buffers.
    {
        const std::vector<float> zeros_m3d(static_cast<std::size_t>(N) * 3u, 0.0f);
        dm3d_buf_->upload(zeros_m3d.data(), static_cast<std::size_t>(bytes_d_m3d));

        const std::vector<float> zeros_sh(static_cast<std::size_t>(N) * K * 3u, 0.0f);
        dsh_buf_ ->upload(zeros_sh.data(),  static_cast<std::size_t>(bytes_d_sh));

        const std::vector<float> zeros_sc(static_cast<std::size_t>(N) * 3u, 0.0f);
        dsc_buf_ ->upload(zeros_sc.data(),  static_cast<std::size_t>(bytes_d_sc));

        const std::vector<float> zeros_rot(static_cast<std::size_t>(N) * 4u, 0.0f);
        drot_buf_->upload(zeros_rot.data(), static_cast<std::size_t>(bytes_d_rot));

        const std::vector<float> zeros_opa(static_cast<std::size_t>(N), 0.0f);
        d_raw_opa_buf_->upload(zeros_opa.data(), static_cast<std::size_t>(bytes_N_float));
    }

    // Upload UBO.
    PreprocessBackwardUBO ubo{};
    std::memcpy(ubo.view_matrix, cam.view_matrix,       16 * sizeof(float));
    std::memcpy(ubo.proj_matrix, cam.viewproj_matrix,   16 * sizeof(float));
    ubo.num_gaussians   = static_cast<uint32_t>(N);
    ubo.sh_degree       = static_cast<uint32_t>(cfg.sh_degree);
    ubo.sh_coeffs_per_g = static_cast<uint32_t>(K);
    ubo.scale_modifier  = cfg.scale_modifier;
    ubo.h_x             = h_x;
    ubo.h_y             = h_y;
    ubo.tan_fovx        = cam.tan_fovx;
    ubo.tan_fovy        = cam.tan_fovy;
    ubo.cam_pos[0]      = cam.cam_pos[0];
    ubo.cam_pos[1]      = cam.cam_pos[1];
    ubo.cam_pos[2]      = cam.cam_pos[2];
    ubo.training        = cfg.training ? 1u : 0u;
    ubo.cam_width       = static_cast<uint32_t>(cam.width);
    ubo.cam_height      = static_cast<uint32_t>(cam.height);
    ubo_buf_->upload(&ubo, sizeof(ubo));

    // --- Bind and dispatch --------------------------------------------------
    PreprocessBackwardPass::Buffers pb{};
    pb.positions   = pos_buf_ ->handle();
    pb.radii       = rad_buf_ ->handle();
    pb.cov3D       = cv3_buf_ ->handle();
    pb.d_conics    = dcon_buf_->handle();
    pb.d_opacity   = dopa_buf_->handle();
    pb.sh_coeffs   = sh_buf_  ->handle();
    pb.scales      = sc_buf_  ->handle();
    pb.rotations   = rot_buf_ ->handle();
    pb.d_rgb       = drgb_buf_->handle();
    pb.d_means2D   = dm2d_buf_->handle();
    pb.d_means3D       = dm3d_buf_   ->handle();
    pb.d_sh            = dsh_buf_    ->handle();
    pb.d_scales        = dsc_buf_    ->handle();
    pb.d_rotations     = drot_buf_   ->handle();
    pb.opacities       = opa_in_buf_ ->handle();
    pb.d_raw_opacities = d_raw_opa_buf_->handle();
    pb.raw_rotations      = raw_rot_buf_  ->handle();
    pb.means2D_cache      = m2d_cache_buf_->handle();
    pb.p_view_cache_in    = pview_in_buf_ ->handle();
    pb.cov2D_cache_in     = cov2d_in_buf_ ->handle();
    pb.cov2D_det_cache_in = c2ddet_in_buf_->handle();
    pb.p_hom_w_cache_in   = phomw_in_buf_ ->handle();

    pass_->bind_buffers(pb, ubo_buf_->handle());
    pass_->dispatch_sync(static_cast<uint32_t>(N));

    // --- Download gradients into grads arrays --------------------------------
    dm3d_buf_    ->download(grads.d_raw_positions,  static_cast<std::size_t>(bytes_d_m3d));
    dsh_buf_     ->download(grads.d_raw_sh_coeffs,  static_cast<std::size_t>(bytes_d_sh));
    dsc_buf_     ->download(grads.d_raw_scales,     static_cast<std::size_t>(bytes_d_sc));
    drot_buf_    ->download(grads.d_raw_rotations,  static_cast<std::size_t>(bytes_d_rot));
    d_raw_opa_buf_->download(grads.d_raw_opacities, static_cast<std::size_t>(bytes_N_float));
}

void PreprocessorBackwardVulkan::backward_record_into(VkCommandBuffer cmd,
                                                       const GaussianData& g,
                                                       int num_gaussians,
                                                       const Camera& cam,
                                                       const RenderConfig& cfg,
                                                       const ForwardCache& cache,
                                                       VkBuffer d_conics_gpu,
                                                       VkBuffer d_opacity_gpu,
                                                       VkBuffer d_rgb_gpu,
                                                       VkBuffer d_means2D_gpu,
                                                       const RawGaussianParams& raw) {
    // --- Guard: eval_3D path not implemented --------------------------------
    if (cache.pre && cache.pre->eval_3D)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward_record_into: eval_3D=true is not supported");

    const int N = num_gaussians;
    const int K = g.max_coeffs;  // (sh_degree+1)^2

    if (N == 0) return;

    prepare_for_n(N, K);

    // Compute UBO parameters
    const float h_x = static_cast<float>(cam.width)  / (2.0f * cam.tan_fovx);
    const float h_y = static_cast<float>(cam.height) / (2.0f * cam.tan_fovy);

    // --- Compute buffer sizes -----------------------------------------------
    const VkDeviceSize bytes_pos     = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_radii   = static_cast<VkDeviceSize>(N) * sizeof(int);
    const VkDeviceSize bytes_cov3D   = static_cast<VkDeviceSize>(N) * 6u * sizeof(float);
    const VkDeviceSize bytes_sh      = static_cast<VkDeviceSize>(N) * K * 3u * sizeof(float);
    const VkDeviceSize bytes_scales  = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_rot     = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_d_m3d   = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sh    = static_cast<VkDeviceSize>(N) * K * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sc    = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_rot   = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_N_float = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_raw_rot    = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_m2d_cache  = static_cast<VkDeviceSize>(N) * 2u * sizeof(float);
    const VkDeviceSize bytes_p_view     = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_cov2d      = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_c2d_det    = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_phomw      = static_cast<VkDeviceSize>(N) * sizeof(float);

    // Validate that the required cache fields are populated.
    if (!cache.p_view)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward_record_into: cache.p_view is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.cov2D)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward_record_into: cache.cov2D is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.cov2D_det)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward_record_into: cache.cov2D_det is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");
    if (!cache.p_hom_w)
        throw std::runtime_error(
            "PreprocessorBackwardVulkan::backward_record_into: cache.p_hom_w is null — "
            "call PreprocessorCPU::process() or PreprocessorVulkan::download_cache() first");

    // --- Upload inputs (all except the rgrad GPU inputs) --------------------
    pos_buf_ ->upload(g.positions,                   static_cast<std::size_t>(bytes_pos));
    rad_buf_ ->upload(cache.pre->radii,               static_cast<std::size_t>(bytes_radii));
    cv3_buf_ ->upload(cache.cov3D,                    static_cast<std::size_t>(bytes_cov3D));
    sh_buf_  ->upload(g.sh_coeffs,                    static_cast<std::size_t>(bytes_sh));
    sc_buf_  ->upload(g.scales,                       static_cast<std::size_t>(bytes_scales));
    rot_buf_ ->upload(g.rotations,                    static_cast<std::size_t>(bytes_rot));
    opa_in_buf_->upload(g.opacities,                 static_cast<std::size_t>(bytes_N_float));
    raw_rot_buf_->upload(raw.raw_rotations,           static_cast<std::size_t>(bytes_raw_rot));
    m2d_cache_buf_->upload(cache.pre->means2D,        static_cast<std::size_t>(bytes_m2d_cache));
    pview_in_buf_ ->upload(cache.p_view,              static_cast<std::size_t>(bytes_p_view));
    cov2d_in_buf_ ->upload(cache.cov2D,               static_cast<std::size_t>(bytes_cov2d));
    c2ddet_in_buf_->upload(cache.cov2D_det,           static_cast<std::size_t>(bytes_c2d_det));
    phomw_in_buf_ ->upload(cache.p_hom_w,             static_cast<std::size_t>(bytes_phomw));

    // Zero-fill gradient output buffers.
    {
        const std::vector<float> zeros_m3d(static_cast<std::size_t>(N) * 3u, 0.0f);
        dm3d_buf_->upload(zeros_m3d.data(), static_cast<std::size_t>(bytes_d_m3d));

        const std::vector<float> zeros_sh(static_cast<std::size_t>(N) * K * 3u, 0.0f);
        dsh_buf_ ->upload(zeros_sh.data(),  static_cast<std::size_t>(bytes_d_sh));

        const std::vector<float> zeros_sc(static_cast<std::size_t>(N) * 3u, 0.0f);
        dsc_buf_ ->upload(zeros_sc.data(),  static_cast<std::size_t>(bytes_d_sc));

        const std::vector<float> zeros_rot(static_cast<std::size_t>(N) * 4u, 0.0f);
        drot_buf_->upload(zeros_rot.data(), static_cast<std::size_t>(bytes_d_rot));

        const std::vector<float> zeros_opa(static_cast<std::size_t>(N), 0.0f);
        d_raw_opa_buf_->upload(zeros_opa.data(), static_cast<std::size_t>(bytes_N_float));
    }

    // Upload UBO.
    PreprocessBackwardUBO ubo{};
    std::memcpy(ubo.view_matrix, cam.view_matrix,       16 * sizeof(float));
    std::memcpy(ubo.proj_matrix, cam.viewproj_matrix,   16 * sizeof(float));
    ubo.num_gaussians   = static_cast<uint32_t>(N);
    ubo.sh_degree       = static_cast<uint32_t>(cfg.sh_degree);
    ubo.sh_coeffs_per_g = static_cast<uint32_t>(K);
    ubo.scale_modifier  = cfg.scale_modifier;
    ubo.h_x             = h_x;
    ubo.h_y             = h_y;
    ubo.tan_fovx        = cam.tan_fovx;
    ubo.tan_fovy        = cam.tan_fovy;
    ubo.cam_pos[0]      = cam.cam_pos[0];
    ubo.cam_pos[1]      = cam.cam_pos[1];
    ubo.cam_pos[2]      = cam.cam_pos[2];
    ubo.training        = cfg.training ? 1u : 0u;
    ubo.cam_width       = static_cast<uint32_t>(cam.width);
    ubo.cam_height      = static_cast<uint32_t>(cam.height);
    ubo_buf_->upload(&ubo, sizeof(ubo));

    // --- Bind and record into cmd ------------------------------------------
    // rgrad inputs come directly from GPU buffers (d_conics_gpu, d_opacity_gpu,
    // d_rgb_gpu, d_means2D_gpu) rather than being uploaded from CPU.
    PreprocessBackwardPass::Buffers pb{};
    pb.positions   = pos_buf_ ->handle();
    pb.radii       = rad_buf_ ->handle();
    pb.cov3D       = cv3_buf_ ->handle();
    pb.d_conics    = d_conics_gpu;
    pb.d_opacity   = d_opacity_gpu;
    pb.sh_coeffs   = sh_buf_  ->handle();
    pb.scales      = sc_buf_  ->handle();
    pb.rotations   = rot_buf_ ->handle();
    pb.d_rgb       = d_rgb_gpu;
    pb.d_means2D   = d_means2D_gpu;
    pb.d_means3D       = dm3d_buf_   ->handle();
    pb.d_sh            = dsh_buf_    ->handle();
    pb.d_scales        = dsc_buf_    ->handle();
    pb.d_rotations     = drot_buf_   ->handle();
    pb.opacities       = opa_in_buf_ ->handle();
    pb.d_raw_opacities = d_raw_opa_buf_->handle();
    pb.raw_rotations      = raw_rot_buf_  ->handle();
    pb.means2D_cache      = m2d_cache_buf_->handle();
    pb.p_view_cache_in    = pview_in_buf_ ->handle();
    pb.cov2D_cache_in     = cov2d_in_buf_ ->handle();
    pb.cov2D_det_cache_in = c2ddet_in_buf_->handle();
    pb.p_hom_w_cache_in   = phomw_in_buf_ ->handle();

    pass_->bind_buffers(pb, ubo_buf_->handle());
    pass_->record(cmd, static_cast<uint32_t>(N));
}

void PreprocessorBackwardVulkan::download_grads(int N, int K,
    GradientOutput& grads, FrameAllocator& alloc)
{
    grads.allocate_and_zero(alloc, N, K);
    if (N == 0) return;
    const VkDeviceSize bytes_d_m3d   = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sh    = static_cast<VkDeviceSize>(N) * K * 3u * sizeof(float);
    const VkDeviceSize bytes_d_sc    = static_cast<VkDeviceSize>(N) * 3u * sizeof(float);
    const VkDeviceSize bytes_d_rot   = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    const VkDeviceSize bytes_N_float = static_cast<VkDeviceSize>(N) * sizeof(float);
    dm3d_buf_    ->download(grads.d_raw_positions,
                            static_cast<std::size_t>(bytes_d_m3d));
    dsh_buf_     ->download(grads.d_raw_sh_coeffs,
                            static_cast<std::size_t>(bytes_d_sh));
    dsc_buf_     ->download(grads.d_raw_scales,
                            static_cast<std::size_t>(bytes_d_sc));
    drot_buf_    ->download(grads.d_raw_rotations,
                            static_cast<std::size_t>(bytes_d_rot));
    d_raw_opa_buf_->download(grads.d_raw_opacities,
                             static_cast<std::size_t>(bytes_N_float));
}
