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

    // --- Allocate GPU buffers (host-visible coherent) -----------------------
    auto pos_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_pos,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rad_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_radii, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cv3_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_cov3D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dcon_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_con, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dopa_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_opa, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sh_buf     = std::make_unique<VulkanBuffer>(ctx_, bytes_sh,    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sc_buf     = std::make_unique<VulkanBuffer>(ctx_, bytes_scales,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rot_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_rot,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto drgb_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_rgb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dm2d_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_m2d, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Gradient output buffers — must be zero-filled before dispatch.
    // preprocess_backward.comp writes directly (not atomicAdd), so they must
    // start at zero (culled Gaussians don't write and the buffer retains the value).
    auto dm3d_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_m3d, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dsh_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_d_sh,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dsc_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_d_sc,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto drot_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_d_rot, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    auto opa_in_buf    = std::make_unique<VulkanBuffer>(ctx_, bytes_N_float, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto d_raw_opa_buf = std::make_unique<VulkanBuffer>(ctx_, bytes_N_float, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const VkDeviceSize bytes_raw_rot = static_cast<VkDeviceSize>(N) * 4u * sizeof(float);
    auto raw_rot_buf   = std::make_unique<VulkanBuffer>(ctx_, bytes_raw_rot, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    auto ubo_buf    = std::make_unique<VulkanBuffer>(ctx_,
                          static_cast<VkDeviceSize>(sizeof(PreprocessBackwardUBO)),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // --- Upload inputs -------------------------------------------------------
    pos_buf ->upload(g.positions,                   static_cast<std::size_t>(bytes_pos));
    rad_buf ->upload(cache.pre->radii,               static_cast<std::size_t>(bytes_radii));
    cv3_buf ->upload(cache.cov3D,                    static_cast<std::size_t>(bytes_cov3D));
    dcon_buf->upload(rgrad.d_conics,                 static_cast<std::size_t>(bytes_d_con));
    dopa_buf->upload(rgrad.d_opacities_2d,           static_cast<std::size_t>(bytes_d_opa));  // d_opacity uploaded but not read by shader — SP-3 scope
    sh_buf  ->upload(g.sh_coeffs,                    static_cast<std::size_t>(bytes_sh));
    sc_buf  ->upload(g.scales,                       static_cast<std::size_t>(bytes_scales));
    rot_buf ->upload(g.rotations,                    static_cast<std::size_t>(bytes_rot));
    drgb_buf->upload(rgrad.d_rgb,                    static_cast<std::size_t>(bytes_d_rgb));
    dm2d_buf->upload(rgrad.d_means2D,               static_cast<std::size_t>(bytes_d_m2d));
    opa_in_buf->upload(g.opacities,                 static_cast<std::size_t>(bytes_N_float));
    raw_rot_buf->upload(raw.raw_rotations,           static_cast<std::size_t>(bytes_raw_rot));

    // Zero-fill gradient output buffers.
    {
        const std::vector<float> zeros_m3d(static_cast<std::size_t>(N) * 3u, 0.0f);
        dm3d_buf->upload(zeros_m3d.data(), static_cast<std::size_t>(bytes_d_m3d));

        const std::vector<float> zeros_sh(static_cast<std::size_t>(N) * K * 3u, 0.0f);
        dsh_buf ->upload(zeros_sh.data(),  static_cast<std::size_t>(bytes_d_sh));

        const std::vector<float> zeros_sc(static_cast<std::size_t>(N) * 3u, 0.0f);
        dsc_buf ->upload(zeros_sc.data(),  static_cast<std::size_t>(bytes_d_sc));

        const std::vector<float> zeros_rot(static_cast<std::size_t>(N) * 4u, 0.0f);
        drot_buf->upload(zeros_rot.data(), static_cast<std::size_t>(bytes_d_rot));

        const std::vector<float> zeros_opa(static_cast<std::size_t>(N), 0.0f);
        d_raw_opa_buf->upload(zeros_opa.data(), static_cast<std::size_t>(bytes_N_float));
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
    ubo_buf->upload(&ubo, sizeof(ubo));

    // --- Bind and dispatch --------------------------------------------------
    PreprocessBackwardPass::Buffers pb{};
    pb.positions   = pos_buf ->handle();
    pb.radii       = rad_buf ->handle();
    pb.cov3D       = cv3_buf ->handle();
    pb.d_conics    = dcon_buf->handle();
    pb.d_opacity   = dopa_buf->handle();
    pb.sh_coeffs   = sh_buf  ->handle();
    pb.scales      = sc_buf  ->handle();
    pb.rotations   = rot_buf ->handle();
    pb.d_rgb       = drgb_buf->handle();
    pb.d_means2D   = dm2d_buf->handle();
    pb.d_means3D       = dm3d_buf   ->handle();
    pb.d_sh            = dsh_buf    ->handle();
    pb.d_scales        = dsc_buf    ->handle();
    pb.d_rotations     = drot_buf   ->handle();
    pb.opacities       = opa_in_buf ->handle();
    pb.d_raw_opacities = d_raw_opa_buf->handle();
    pb.raw_rotations   = raw_rot_buf->handle();

    pass_->bind_buffers(pb, ubo_buf->handle());
    pass_->dispatch_sync(static_cast<uint32_t>(N));

    // --- Download gradients into grads arrays --------------------------------
    dm3d_buf    ->download(grads.d_raw_positions,  static_cast<std::size_t>(bytes_d_m3d));
    dsh_buf     ->download(grads.d_raw_sh_coeffs,  static_cast<std::size_t>(bytes_d_sh));
    dsc_buf     ->download(grads.d_raw_scales,     static_cast<std::size_t>(bytes_d_sc));
    drot_buf    ->download(grads.d_raw_rotations,  static_cast<std::size_t>(bytes_d_rot));
    d_raw_opa_buf->download(grads.d_raw_opacities, static_cast<std::size_t>(bytes_N_float));
}
