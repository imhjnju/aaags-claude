// SP-2 Task 6: PreprocessorVulkan::process() implementation.
//
// Layer-1 (sync) adapter:
//   1. Allocate host-visible Vulkan buffers for all 12 SSBO slots + CameraUBO.
//   2. Upload inputs (positions/scales/rotations/opacities/sh/filter_3D).
//   3. Build std140 CameraUBO (spec §4.6).
//   4. Bind buffers to PreprocessPass, dispatch_sync().
//   5. Download outputs; deinterleave packed {conic.xyz, opacity} into the
//      separate `conics` (N*3) and `opacities_2d` (N) fields of
//      PreprocessOutput (types.h).
//
// SP-2 constraints enforced as hard errors (spec §4.4):
//   - eval_3D=true not supported: spec_eval_3D=0 baked into the pipeline.
//   - tile 16x16 only: shader's getRect assumes 16x16 tiles.
//   - antialiasing flag not supported: shader path is the fixed +0.3 dilation
//     variant without h_conv_scaling.
//
// Buffer allocation: fresh per process() call. Phase-1 simplicity — no caching
// or reuse across calls. Future phases may hoist allocation for throughput.

#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/preprocess_pass.h"
#include "vulkan/preprocess_bindings.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_camera_ubo.h"
#include "math_utils.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

PreprocessorVulkan::PreprocessorVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    // SP-2: spec_training=1 (match CPU training path, no upper SH-RGB clamp),
    //       spec_eval_3D=0 (2D anti-aliasing path only).
    // TODO(SP-2 T7+): RenderConfig::training is a per-process() flag on the
    // CPU side but a pipeline-time specialization constant here. If mixed
    // training/inference in the same session is ever required, we'll need
    // either two pre-built PreprocessPass instances (one per mode) or a
    // pipeline rebuild. For SP-2 bring-up tests we match the CPU training
    // path; tests/callers must pass cfg.training=true for bitwise parity.
    pass_ = std::make_unique<PreprocessPass>(ctx_,
                                             /*spec_training=*/1u,
                                             /*spec_eval_3D=*/0u);
}

// Out-of-line destructor so std::unique_ptr<PreprocessPass> can see the
// complete type (header uses a forward declaration only).
PreprocessorVulkan::~PreprocessorVulkan() = default;

PreprocessOutput PreprocessorVulkan::process(const GaussianData& g,
                                             const Camera& cam,
                                             const RenderConfig& cfg,
                                             FrameAllocator& alloc,
                                             ForwardCache* cache) {
    // --- SP-2 hard errors (spec §4.4) ----------------------------------------
    if (cfg.eval_3D)
        throw std::runtime_error(
            "PreprocessorVulkan: eval_3D=true not supported in SP-2");
    if (cfg.tile_w != 16 || cfg.tile_h != 16)
        throw std::runtime_error(
            "PreprocessorVulkan: only 16x16 tiles supported");
    if (cfg.antialiasing)
        throw std::runtime_error(
            "PreprocessorVulkan: antialiasing flag not supported");

    const int N = g.count;
    const int M = g.max_coeffs;  // (sh_degree+1)^2

    // --- 1. Allocate host-visible buffers ------------------------------------
    // Inputs
    auto pos_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 3 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto scl_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 3 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rot_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 4 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto op_buf  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sh_buf  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * M * 3 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto f3_buf  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Outputs
    auto m2d_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 2 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dep_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Packed {conic.a, conic.b, conic.c, opacity_2d}: 4 floats per Gaussian.
    auto cop_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 4 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rgb_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 3 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // radii/tiles_touched: shader declares `int` (binding 10/11). Host-side
    // PreprocessOutput uses `int*`. Buffer sized in 32-bit ints.
    auto rad_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(int32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto tt_buf  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(int32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rf_buf  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cam_buf = std::make_unique<VulkanBuffer>(
        ctx_, sizeof(CameraUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    // ForwardCache output buffers (bindings 14..16).
    // Always allocated so the shader descriptor set is always fully bound.
    cov3d_buf_   = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 6 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    p_view_buf_  = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * 3 * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    p_hom_w_buf_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // --- 2. Upload inputs ----------------------------------------------------
    pos_buf->upload(g.positions, static_cast<std::size_t>(N) * 3 * sizeof(float));
    scl_buf->upload(g.scales,    static_cast<std::size_t>(N) * 3 * sizeof(float));
    rot_buf->upload(g.rotations, static_cast<std::size_t>(N) * 4 * sizeof(float));
    op_buf ->upload(g.opacities, static_cast<std::size_t>(N) * sizeof(float));
    sh_buf ->upload(g.sh_coeffs, static_cast<std::size_t>(N) * M * 3 * sizeof(float));

    // filter_3D is optional. When nullptr, upload N zeros so the shader's
    // binding-5 SSBO is valid (preprocess.comp always reads filter_3D[i]).
    if (g.filter_3D != nullptr) {
        f3_buf->upload(g.filter_3D, static_cast<std::size_t>(N) * sizeof(float));
    } else {
        std::vector<float> zeros(static_cast<std::size_t>(N), 0.0f);
        f3_buf->upload(zeros.data(), static_cast<std::size_t>(N) * sizeof(float));
    }

    // --- 3. Build CameraUBO (std140, 224 bytes per spec §4.6) ----------------
    CameraUBO c{};
    // Column-major 4x4 matches CUDA/GLM and Vulkan std140 mat4 layout.
    std::memcpy(c.viewmatrix, cam.view_matrix,     16 * sizeof(float));
    std::memcpy(c.projmatrix, cam.viewproj_matrix, 16 * sizeof(float));

    // inv_viewprojmatrix: used by shaders that convert pixel rays back to
    // world space (not currently read by preprocess.comp in SP-2, but the UBO
    // layout requires a valid matrix in every slot). Fall back to identity if
    // the viewproj is singular to avoid producing NaNs.
    float inv_vp[16];
    if (!invertMatrix4x4(cam.viewproj_matrix, inv_vp)) {
        // Identity as a safe fallback — preprocess.comp doesn't read this
        // field; downstream passes will detect the singular case explicitly.
        std::memset(inv_vp, 0, sizeof(inv_vp));
        inv_vp[0] = inv_vp[5] = inv_vp[10] = inv_vp[15] = 1.0f;
    }
    std::memcpy(c.inv_viewprojmatrix, inv_vp, 16 * sizeof(float));

    c.campos_pad[0] = cam.cam_pos[0];
    c.campos_pad[1] = cam.cam_pos[1];
    c.campos_pad[2] = cam.cam_pos[2];
    c.campos_pad[3] = 0.0f;

    c.fov_size[0] = cam.tan_fovx;
    c.fov_size[1] = cam.tan_fovy;
    c.fov_size[2] = static_cast<float>(cam.width);
    c.fov_size[3] = static_cast<float>(cam.height);

    cam_buf->upload(&c, sizeof(CameraUBO));

    // --- 4. Bind + dispatch --------------------------------------------------
    PreprocessPass::Buffers b{};
    b.positions             = pos_buf->handle();
    b.scales                = scl_buf->handle();
    b.rotations             = rot_buf->handle();
    b.opacities             = op_buf ->handle();
    b.sh                    = sh_buf ->handle();
    b.filter_3D             = f3_buf ->handle();
    b.means2D               = m2d_buf->handle();
    b.depths                = dep_buf->handle();
    b.conic_opacity_packed  = cop_buf->handle();
    b.rgb                   = rgb_buf->handle();
    b.radii                 = rad_buf->handle();
    b.tiles_touched         = tt_buf ->handle();
    b.camera_ubo            = cam_buf->handle();
    b.radius_f              = rf_buf ->handle();
    b.cov3D_cache           = cov3d_buf_  ->handle();
    b.p_view_cache          = p_view_buf_ ->handle();
    b.p_hom_w_cache         = p_hom_w_buf_->handle();
    pass_->bind_buffers(b);

    PreprocessPushConstants pc{};
    pc.num_gaussians   = static_cast<uint32_t>(N);
    pc.sh_degree       = static_cast<uint32_t>(cfg.sh_degree);
    pc.sh_coeffs_per_g = static_cast<uint32_t>(M);
    pc.num_tiles_x     = static_cast<uint32_t>((cam.width  + 15) / 16);
    pc.num_tiles_y     = static_cast<uint32_t>((cam.height + 15) / 16);
    pc.scale_modifier  = cfg.scale_modifier;

    pass_->dispatch_sync(pc);

    // --- 5. Download outputs + deinterleave ----------------------------------
    PreprocessOutput out{};
    out.means2D       = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 2);
    out.depths        = alloc.allocate_array<float>(static_cast<std::size_t>(N));
    out.conics        = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 3);
    out.opacities_2d  = alloc.allocate_array<float>(static_cast<std::size_t>(N));
    out.rgb           = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 3);
    out.radii         = alloc.allocate_array<int>  (static_cast<std::size_t>(N));
    out.tiles_touched = alloc.allocate_array<int>  (static_cast<std::size_t>(N));
    out.gauss2screen  = nullptr;   // eval_3D=false
    out.cov3D_inv     = nullptr;
    out.mean_offset   = nullptr;
    out.eval_3D       = false;
    out.radius_f      = alloc.allocate_array<float>(static_cast<std::size_t>(N));

    m2d_buf->download(out.means2D,       static_cast<std::size_t>(N) * 2 * sizeof(float));
    dep_buf->download(out.depths,        static_cast<std::size_t>(N) * sizeof(float));
    rgb_buf->download(out.rgb,           static_cast<std::size_t>(N) * 3 * sizeof(float));
    rad_buf->download(out.radii,         static_cast<std::size_t>(N) * sizeof(int32_t));
    tt_buf ->download(out.tiles_touched, static_cast<std::size_t>(N) * sizeof(int32_t));
    rf_buf ->download(out.radius_f,      static_cast<std::size_t>(N) * sizeof(float));

    // Deinterleave packed {conic.a, conic.b, conic.c, opacity} (stride-4 per
    // Gaussian) into the CPU-reference layout: conics[N*3] and opacities_2d[N].
    std::vector<float> packed(static_cast<std::size_t>(N) * 4);
    cop_buf->download(packed.data(), static_cast<std::size_t>(N) * 4 * sizeof(float));
    for (int i = 0; i < N; ++i) {
        out.conics[i * 3 + 0] = packed[i * 4 + 0];
        out.conics[i * 3 + 1] = packed[i * 4 + 1];
        out.conics[i * 3 + 2] = packed[i * 4 + 2];
        out.opacities_2d[i]   = packed[i * 4 + 3];
    }

    // Populate ForwardCache if requested.
    if (cache) {
        download_cache(N, *cache, alloc);
    }

    return out;
}

// ---------------------------------------------------------------------------
// download_cache(): download ForwardCache fields populated during process().
// Must be called after process(); throws if process() hasn't run yet.
// ---------------------------------------------------------------------------
void PreprocessorVulkan::download_cache(int N, ForwardCache& cache,
                                        FrameAllocator& alloc) {
    if (!cov3d_buf_ || !p_view_buf_ || !p_hom_w_buf_)
        throw std::runtime_error(
            "PreprocessorVulkan::download_cache: process() not yet called");
    cache.cov3D   = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 6);
    cache.p_view  = alloc.allocate_array<float>(static_cast<std::size_t>(N) * 3);
    cache.p_hom_w = alloc.allocate_array<float>(static_cast<std::size_t>(N));
    cov3d_buf_  ->download(cache.cov3D,   static_cast<std::size_t>(N) * 6 * sizeof(float));
    p_view_buf_ ->download(cache.p_view,  static_cast<std::size_t>(N) * 3 * sizeof(float));
    p_hom_w_buf_->download(cache.p_hom_w, static_cast<std::size_t>(N) * sizeof(float));
}

// ---------------------------------------------------------------------------
// Layer-2 record path: prepare_record() + record() + buffer getters.
//
// Shape mirrors the first 4 steps of process(): alloc + upload + CameraUBO +
// bind. The resulting GPU buffers stay owned by `record_bufs_` until the next
// prepare_record() or destruction, so downstream adapters (TileBinnerVulkan,
// SorterVulkan, RasterizerVulkan) can read the handles via the getters.
// ---------------------------------------------------------------------------
namespace {
// Fixed indices into PreprocessorVulkan::record_bufs_. Matches preprocess_bind::
// for readability — but CAMERA_UBO (binding 12) sits at index 12 in the vector,
// all other bindings are SSBOs and their index == binding.
enum RecBufIdx : size_t {
    kPositions = 0,
    kScales,
    kRotations,
    kOpacities,
    kSH,
    kFilter3D,
    kMeans2D,
    kDepths,
    kConicOpacityPacked,
    kRGB,
    kRadii,
    kTilesTouched,
    kCameraUBO,
    kRadiusF,    // binding 13: float eigenvalue radius (scatter fix)
    kRecBufCount,
};
}  // namespace

void PreprocessorVulkan::prepare_record(const GaussianData& g,
                                        const Camera& cam,
                                        const RenderConfig& cfg) {
    // --- SP-2 hard errors (spec §4.4, same set as process()) -----------------
    if (cfg.eval_3D)
        throw std::runtime_error(
            "PreprocessorVulkan::prepare_record: eval_3D=true not supported in SP-2");
    if (cfg.tile_w != 16 || cfg.tile_h != 16)
        throw std::runtime_error(
            "PreprocessorVulkan::prepare_record: only 16x16 tiles supported");
    if (cfg.antialiasing)
        throw std::runtime_error(
            "PreprocessorVulkan::prepare_record: antialiasing flag not supported");

    const int N = g.count;
    const int M = g.max_coeffs;
    if (N <= 0)
        throw std::runtime_error(
            "PreprocessorVulkan::prepare_record: count must be > 0");

    // Release any previously held buffers up front so memory usage doesn't
    // double-balloon while we allocate the new set.
    record_bufs_.clear();
    record_bufs_.resize(kRecBufCount);

    auto mk_ssbo = [&](VkDeviceSize bytes) {
        return std::make_unique<VulkanBuffer>(
            ctx_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    };

    // --- 1. Allocate host-visible buffers ------------------------------------
    // Inputs
    record_bufs_[kPositions] =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 3 * sizeof(float));
    record_bufs_[kScales]    =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 3 * sizeof(float));
    record_bufs_[kRotations] =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 4 * sizeof(float));
    record_bufs_[kOpacities] =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(float));
    record_bufs_[kSH]        =
        mk_ssbo(static_cast<VkDeviceSize>(N) * M * 3 * sizeof(float));
    record_bufs_[kFilter3D]  =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(float));
    // Outputs
    record_bufs_[kMeans2D]             =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 2 * sizeof(float));
    record_bufs_[kDepths]              =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(float));
    record_bufs_[kConicOpacityPacked]  =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 4 * sizeof(float));
    record_bufs_[kRGB]                 =
        mk_ssbo(static_cast<VkDeviceSize>(N) * 3 * sizeof(float));
    record_bufs_[kRadii]               =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(int32_t));
    record_bufs_[kTilesTouched]        =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(int32_t));
    record_bufs_[kCameraUBO]           = std::make_unique<VulkanBuffer>(
        ctx_, sizeof(CameraUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    record_bufs_[kRadiusF]             =
        mk_ssbo(static_cast<VkDeviceSize>(N) * sizeof(float));

    // --- 2. Upload inputs ----------------------------------------------------
    record_bufs_[kPositions]->upload(g.positions,
        static_cast<std::size_t>(N) * 3 * sizeof(float));
    record_bufs_[kScales]->upload(g.scales,
        static_cast<std::size_t>(N) * 3 * sizeof(float));
    record_bufs_[kRotations]->upload(g.rotations,
        static_cast<std::size_t>(N) * 4 * sizeof(float));
    record_bufs_[kOpacities]->upload(g.opacities,
        static_cast<std::size_t>(N) * sizeof(float));
    record_bufs_[kSH]->upload(g.sh_coeffs,
        static_cast<std::size_t>(N) * M * 3 * sizeof(float));

    if (g.filter_3D != nullptr) {
        record_bufs_[kFilter3D]->upload(g.filter_3D,
            static_cast<std::size_t>(N) * sizeof(float));
    } else {
        std::vector<float> zeros(static_cast<std::size_t>(N), 0.0f);
        record_bufs_[kFilter3D]->upload(zeros.data(),
            static_cast<std::size_t>(N) * sizeof(float));
    }

    // --- 3. Build CameraUBO (same as process()) ------------------------------
    CameraUBO c{};
    std::memcpy(c.viewmatrix, cam.view_matrix,     16 * sizeof(float));
    std::memcpy(c.projmatrix, cam.viewproj_matrix, 16 * sizeof(float));
    float inv_vp[16];
    if (!invertMatrix4x4(cam.viewproj_matrix, inv_vp)) {
        std::memset(inv_vp, 0, sizeof(inv_vp));
        inv_vp[0] = inv_vp[5] = inv_vp[10] = inv_vp[15] = 1.0f;
    }
    std::memcpy(c.inv_viewprojmatrix, inv_vp, 16 * sizeof(float));
    c.campos_pad[0] = cam.cam_pos[0];
    c.campos_pad[1] = cam.cam_pos[1];
    c.campos_pad[2] = cam.cam_pos[2];
    c.campos_pad[3] = 0.0f;
    c.fov_size[0] = cam.tan_fovx;
    c.fov_size[1] = cam.tan_fovy;
    c.fov_size[2] = static_cast<float>(cam.width);
    c.fov_size[3] = static_cast<float>(cam.height);
    record_bufs_[kCameraUBO]->upload(&c, sizeof(CameraUBO));

    // --- 4. Bind to PreprocessPass ------------------------------------------
    PreprocessPass::Buffers b{};
    b.positions            = record_bufs_[kPositions]           ->handle();
    b.scales               = record_bufs_[kScales]              ->handle();
    b.rotations            = record_bufs_[kRotations]           ->handle();
    b.opacities            = record_bufs_[kOpacities]           ->handle();
    b.sh                   = record_bufs_[kSH]                  ->handle();
    b.filter_3D            = record_bufs_[kFilter3D]            ->handle();
    b.means2D              = record_bufs_[kMeans2D]             ->handle();
    b.depths               = record_bufs_[kDepths]              ->handle();
    b.conic_opacity_packed = record_bufs_[kConicOpacityPacked]  ->handle();
    b.rgb                  = record_bufs_[kRGB]                 ->handle();
    b.radii                = record_bufs_[kRadii]               ->handle();
    b.tiles_touched        = record_bufs_[kTilesTouched]        ->handle();
    b.camera_ubo           = record_bufs_[kCameraUBO]           ->handle();
    b.radius_f             = record_bufs_[kRadiusF]             ->handle();
    pass_->bind_buffers(b);
}

void PreprocessorVulkan::record(VkCommandBuffer cmd,
                                uint32_t num_gaussians, uint32_t sh_degree,
                                uint32_t sh_coeffs_per_g, uint32_t num_tiles_x,
                                uint32_t num_tiles_y, float scale_modifier) {
    if (record_bufs_.empty())
        throw std::runtime_error(
            "PreprocessorVulkan::record called before prepare_record()");

    PreprocessPushConstants pc{};
    pc.num_gaussians   = num_gaussians;
    pc.sh_degree       = sh_degree;
    pc.sh_coeffs_per_g = sh_coeffs_per_g;
    pc.num_tiles_x     = num_tiles_x;
    pc.num_tiles_y     = num_tiles_y;
    pc.scale_modifier  = scale_modifier;
    pass_->record(cmd, pc);
}

// Handle getters: return VK_NULL_HANDLE if prepare_record() hasn't run yet,
// otherwise the persistent VkBuffer handle from the last prepare_record().
VkBuffer PreprocessorVulkan::means2D_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kMeans2D]->handle();
}
VkBuffer PreprocessorVulkan::depths_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kDepths]->handle();
}
VkBuffer PreprocessorVulkan::conic_opacity_packed_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kConicOpacityPacked]->handle();
}
VkBuffer PreprocessorVulkan::rgb_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kRGB]->handle();
}
VkBuffer PreprocessorVulkan::radii_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kRadii]->handle();
}
VkBuffer PreprocessorVulkan::tiles_touched_buffer() const {
    return record_bufs_.empty() ? VK_NULL_HANDLE
                                : record_bufs_[kTilesTouched]->handle();
}
VkBuffer PreprocessorVulkan::radius_f_buffer() const {
    if (record_bufs_.size() <= kRadiusF) return VK_NULL_HANDLE;
    return record_bufs_[kRadiusF] ? record_bufs_[kRadiusF]->handle() : VK_NULL_HANDLE;
}
