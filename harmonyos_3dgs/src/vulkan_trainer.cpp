#include "vulkan_trainer.h"
#include "vulkan/vk_pipeline.h"
#include "dssim.h"
#include "train_utils.h"
#include "mcmc_densification.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// SH DC/REST de/interleave helpers
//
// Storage layout convention (matches Python `raw_sh.reshape(N, max_coeffs*3)`,
// see tools/dump_tiny_reference.py:100): the unified raw_sh_coeffs_ buffer is
// `[N, K, 3]` flat — per-Gaussian blocks of K*3 floats with channel innermost.
// The DC of Gaussian i lives at index `i*K*3 + 0..2`; band-1 at `i*K*3 + 3..11`.
//
// The Adam optimizer splits these into two GPU buffers with different LRs:
//   group 1 (DC):   sz=N*3,         lr=2.5e-3
//   group 2 (REST): sz=N*(K-1)*3,   lr=1.25e-4
//
// Per the 3DGS reference convention (gaussian_renderer:scene/gaussian_model.py
// `features_dc` [N,1,3] and `features_rest` [N,K-1,3] are SEPARATE tensors
// registered as separate optimizer groups), the DC GPU buffer must hold
// "DC of all N Gaussians, contiguous" — NOT the first N*3 floats of the
// interleaved buffer. The previous code took the first N*3 floats verbatim,
// which only happens to be all-DC when K==1; for K=16 it picks up G[0]'s
// full 48 floats and G[1]'s first 12 floats, leaving G[2..N-1] DCs in the
// REST buffer (wrong LR). Evidence: TempDiag3StepSHCompare showed G[2,DC,*]
// receiving lr=1.25e-4 instead of lr=2.5e-3 — diff -2.375e-3 = -(2.5e-3-1.25e-4)
// per channel per step.
//
// Reference: Python autograd uses ONE group for raw_sh with lr=2.5e-3
// (tools/dump_tiny_reference.py:281); the original 3DGS PyTorch trainer uses
// TWO groups with the (DC, REST) split using separate features_dc/features_rest
// tensors (no interleave) with lrs 2.5e-3 / 1.25e-4.
// ---------------------------------------------------------------------------
namespace {

using StageClock = std::chrono::steady_clock;

bool training_stage_timers_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("GS3D_TRAIN_STAGE_TIMING");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
}

bool gpu_l1_loss_enabled() {
    const char* v = std::getenv("GS3D_TRAIN_GPU_L1_LOSS");
    return v && v[0] == '1' && v[1] == '\0';
}

bool gpu_dssim_loss_enabled() {
    const char* dssim = std::getenv("GS3D_TRAIN_GPU_DSSIM_LOSS");
    if (dssim && dssim[0] == '1' && dssim[1] == '\0') return true;
    return gpu_l1_loss_enabled();
}

bool forward_gpu_cache_enabled() {
    const char* v = std::getenv("GS3D_REUSE_FORWARD_OUTPUTS");
    return v && v[0] == '1' && v[1] == '\0';
}

bool gpu_grad_adam_enabled() {
    const char* v = std::getenv("GS3D_TRAIN_GPU_GRAD_ADAM");
    return v && v[0] == '1' && v[1] == '\0';
}

bool gpu_raw_activation_enabled() {
    const char* v = std::getenv("GS3D_TRAIN_GPU_RAW_ACTIVATE");
    return v && v[0] == '1' && v[1] == '\0';
}

bool compact_eval3d_tiles_enabled() {
    const char* v = std::getenv("GS3D_EVAL3D_COMPACT_TILES");
    return v && v[0] == '1' && v[1] == '\0';
}

bool split_backward_timing_enabled() {
    const char* v = std::getenv("GS3D_SPLIT_BACKWARD_TIMING");
    return v && v[0] == '1' && v[1] == '\0';
}

double stage_elapsed_ms(StageClock::time_point start) {
    return std::chrono::duration<double, std::milli>(StageClock::now() - start).count();
}

// Gather DC slices (k=0) of all N Gaussians from interleaved [N, K, 3] -> [N, 3].
inline void sh_gather_dc(const float* src_interleaved, int N, int K, float* dst_dc) {
    for (int i = 0; i < N; ++i) {
        const float* s = src_interleaved + static_cast<size_t>(i) * K * 3;
        float* d = dst_dc + static_cast<size_t>(i) * 3;
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
    }
}

// Gather REST slices (k=1..K-1) from interleaved [N, K, 3] -> [N, K-1, 3].
inline void sh_gather_rest(const float* src_interleaved, int N, int K, float* dst_rest) {
    if (K <= 1) return;
    const size_t rest_stride = static_cast<size_t>(K - 1) * 3;
    for (int i = 0; i < N; ++i) {
        const float* s = src_interleaved + static_cast<size_t>(i) * K * 3 + 3;  // skip DC
        float* d = dst_rest + static_cast<size_t>(i) * rest_stride;
        std::memcpy(d, s, rest_stride * sizeof(float));
    }
}

// Scatter DC [N, 3] back into interleaved [N, K, 3] (writes only k=0 slot).
inline void sh_scatter_dc(const float* src_dc, int N, int K, float* dst_interleaved) {
    for (int i = 0; i < N; ++i) {
        const float* s = src_dc + static_cast<size_t>(i) * 3;
        float* d = dst_interleaved + static_cast<size_t>(i) * K * 3;
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
    }
}

// Scatter REST [N, K-1, 3] back into interleaved [N, K, 3] (writes k=1..K-1).
inline void sh_scatter_rest(const float* src_rest, int N, int K, float* dst_interleaved) {
    if (K <= 1) return;
    const size_t rest_stride = static_cast<size_t>(K - 1) * 3;
    for (int i = 0; i < N; ++i) {
        const float* s = src_rest + static_cast<size_t>(i) * rest_stride;
        float* d = dst_interleaved + static_cast<size_t>(i) * K * 3 + 3;
        std::memcpy(d, s, rest_stride * sizeof(float));
    }
}

void zero_mcmc_adam_state(VulkanAdam& adam,
                          int max_coeffs,
                          const mcmc::DensifyResult& result) {
    std::vector<int> reset_indices = result.modified_source_indices;
    reset_indices.insert(reset_indices.end(),
                         result.replaced_destination_indices.begin(),
                         result.replaced_destination_indices.end());
    std::sort(reset_indices.begin(), reset_indices.end());
    reset_indices.erase(std::unique(reset_indices.begin(), reset_indices.end()),
                        reset_indices.end());
    if (reset_indices.empty()) return;

    const int rest_coeffs = std::max(max_coeffs - 1, 0);
    const int strides[6] = {3, 3, rest_coeffs * 3, 1, 3, 4};
    for (int group = 0; group < 6; ++group) {
        if (strides[group] == 0) continue;
        std::vector<uint32_t> float_indices;
        float_indices.reserve(reset_indices.size() * static_cast<size_t>(strides[group]));
        for (int gi : reset_indices) {
            for (int k = 0; k < strides[group]; ++k) {
                float_indices.push_back(static_cast<uint32_t>(gi * strides[group] + k));
            }
        }
        adam.zero_moment_floats(group, float_indices);
    }
}

inline float logit_clamped(float p) {
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

VulkanTrainer::VulkanTrainer(VulkanContext& ctx,
                             const GaussianData& init_g,
                             const RawGaussianParams& init_raw,
                             int sh_degree,
                             int cam_width,
                             int cam_height,
                             const VkTrainingConfig& tcfg)
    : ctx_(ctx)
    , N_(init_g.count)
    , max_coeffs_(init_g.max_coeffs)
    , alloc_(64u * 1024u * 1024u)
    , vulkan_adam_(ctx_, 0.9f, 0.999f, 1e-15f)
    , tcfg_(tcfg)
    , active_sh_degree_(tcfg.sh_degree_warmup > 0 ? 0 : tcfg.sh_degree_max)
    , preprocessor_(ctx, /*eval_3D=*/tcfg.eval_3D, tcfg.proper_ewa)
    , binner_(ctx)
    , sorter_(ctx)
    , rasterizer_(ctx, /*eval_3D=*/tcfg.eval_3D, /*eval3d_raw_replay=*/tcfg.parity_mode)
    , rasterizer_bwd_(ctx)
    , preprocessor_bwd_(ctx, tcfg.proper_ewa)
    , l1_loss_pass_(ctx)
    , sh_grad_split_pass_(ctx)
    , raw_activation_pass_(ctx)
{
    // 1. Copy raw parameters into owned vectors.
    raw_positions_.assign(init_raw.raw_positions, init_raw.raw_positions + N_ * 3);
    raw_scales_.assign   (init_raw.raw_scales,    init_raw.raw_scales    + N_ * 3);
    raw_rotations_.assign(init_raw.raw_rotations, init_raw.raw_rotations + N_ * 4);
    raw_sh_coeffs_.assign(init_raw.raw_sh_coeffs, init_raw.raw_sh_coeffs + N_ * max_coeffs_ * 3);
    raw_opacities_.assign(init_raw.raw_opacities, init_raw.raw_opacities + N_);

    // 2. Set non-owning view into the owned vectors.
    raw_view_.count         = N_;
    raw_view_.sh_degree     = sh_degree;
    raw_view_.max_coeffs    = max_coeffs_;
    raw_view_.raw_positions = raw_positions_.data();
    raw_view_.raw_scales    = raw_scales_.data();
    raw_view_.raw_rotations = raw_rotations_.data();
    raw_view_.raw_sh_coeffs = raw_sh_coeffs_.data();
    raw_view_.raw_opacities = raw_opacities_.data();

    // 3. Resize activated-value vectors.
    act_positions_.resize(static_cast<size_t>(N_) * 3);
    act_scales_.resize   (static_cast<size_t>(N_) * 3);
    act_rotations_.resize(static_cast<size_t>(N_) * 4);
    act_opacities_.resize(static_cast<size_t>(N_));
    act_sh_coeffs_.resize(static_cast<size_t>(N_) * max_coeffs_ * 3);
    if (init_g.filter_3D) {
        act_filter_3D_.assign(init_g.filter_3D, init_g.filter_3D + static_cast<size_t>(N_));
    } else {
        act_filter_3D_.assign(static_cast<size_t>(N_), 0.0f);
    }

    // 4. Set up GaussianData non-owning view.
    g_.count      = N_;
    g_.sh_degree  = sh_degree;
    g_.max_coeffs = max_coeffs_;
    g_.positions  = act_positions_.data();
    g_.scales     = act_scales_.data();
    g_.rotations  = act_rotations_.data();
    g_.opacities  = act_opacities_.data();
    g_.sh_coeffs  = act_sh_coeffs_.data();
    g_.filter_3D  = act_filter_3D_.data();

    // 5. Register 6 parameter groups with VulkanAdam.
    //    lr values match Python reference training.py defaults.
    //
    //    Group 0: positions          lr=1.6e-4,  n=N*3
    //    Group 1: sh_coeffs DC       lr=2.5e-3,  n=N*3
    //    Group 2: sh_coeffs rest     lr=1.25e-4, n=N*(max_coeffs-1)*3
    //    Group 3: opacities          lr=0.05,    n=N
    //    Group 4: scales             lr=0.005,   n=N*3
    //    Group 5: rotations          lr=0.001,   n=N*4

    const int rest_coeffs = max_coeffs_ - 1;

    struct GroupSpec { uint32_t n; float lr; };
    const GroupSpec specs[6] = {
        { static_cast<uint32_t>(N_ * 3),                     group_lrs_[0] },  // 0: positions
        { static_cast<uint32_t>(N_ * 3),                     group_lrs_[1] },  // 1: sh DC
        { static_cast<uint32_t>(N_ * rest_coeffs * 3),       group_lrs_[2] },  // 2: sh rest
        { static_cast<uint32_t>(N_),                         group_lrs_[3] },  // 3: opacities
        { static_cast<uint32_t>(N_ * 3),                     group_lrs_[4] },  // 4: scales
        { static_cast<uint32_t>(N_ * 4),                     group_lrs_[5] },  // 5: rotations
    };

    for (int i = 0; i < 6; ++i) {
        vulkan_adam_.add_group(specs[i].n, specs[i].lr);
    }

    // 6. Allocate persistent GPU raw param buffers and upload initial values.
    //    Group 1 (sh DC) covers first N*3 floats of raw_sh_coeffs_.
    //    Group 2 (sh rest) covers the remaining N*(max_coeffs-1)*3 floats.
    //    These are separate VulkanBuffer allocations (not offsets into a shared buffer).
    const size_t sz_N3  = static_cast<size_t>(N_) * 3 * sizeof(float);
    const size_t sz_N4  = static_cast<size_t>(N_) * 4 * sizeof(float);
    const size_t sz_N   = static_cast<size_t>(N_) * sizeof(float);
    const size_t sz_sh_dc   = static_cast<size_t>(N_) * 3 * sizeof(float);
    // Guard: VkBuffer size must be > 0. When sh_degree==0, rest_coeffs==0 and
    // sz_sh_rest would be 0. Use 4 bytes minimum; the group 2 dispatch is guarded
    // in step() by the `max_coeffs_ > 1` check so the padding byte is never read.
    const size_t sz_sh_rest = std::max(
        static_cast<size_t>(N_) * rest_coeffs * 3 * sizeof(float),
        static_cast<size_t>(4));

    raw_param_gpu_bufs_[0] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_param_gpu_bufs_[0]->upload(raw_positions_.data(), sz_N3);

    raw_param_gpu_bufs_[1] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_dc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    {
        // Gather DC of all N Gaussians into a contiguous [N,3] scratch before upload.
        std::vector<float> sh_dc_scratch(static_cast<size_t>(N_) * 3);
        sh_gather_dc(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_dc_scratch.data());
        raw_param_gpu_bufs_[1]->upload(sh_dc_scratch.data(), sz_sh_dc);
    }

    raw_param_gpu_bufs_[2] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (max_coeffs_ > 1) {
        // Gather REST (k=1..K-1) of all N Gaussians into [N, K-1, 3] scratch.
        std::vector<float> sh_rest_scratch(
            static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3);
        sh_gather_rest(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_rest_scratch.data());
        raw_param_gpu_bufs_[2]->upload(sh_rest_scratch.data(),
            static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3 * sizeof(float));
    }

    raw_param_gpu_bufs_[3] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_param_gpu_bufs_[3]->upload(raw_opacities_.data(), sz_N);

    raw_param_gpu_bufs_[4] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_param_gpu_bufs_[4]->upload(raw_scales_.data(), sz_N3);

    raw_param_gpu_bufs_[5] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_param_gpu_bufs_[5]->upload(raw_rotations_.data(), sz_N4);

    // 7. Allocate zero-initialized gradient GPU buffers (one per group).
    grad_positions_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_sh_dc_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_dc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_sh_rest_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_opacities_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_scales_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_rotations_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    active_scales_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_rotations_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_opacities_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_sh_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<size_t>(N_) * max_coeffs_ * 3 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    filter_3D_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    filter_3D_gpu_->upload(act_filter_3D_.data(), sz_N);

    // 8. Allocate per-frame output buffers (H*W*3).
    const size_t HW3 = static_cast<size_t>(cam_height) * cam_width * 3;
    image_.resize(HW3, 0.0f);
    dL_dpixels_.resize(HW3, 0.0f);

    // 9. Initialize accumulated gradient norms (for densification).
    grad_means2D_accum_.assign(static_cast<size_t>(N_), 0.0f);
}

void VulkanTrainer::ensure_rendered_image_downloaded() const {
    if (!image_gpu_pending_download_) return;
    rasterizer_.download_image(image_.data(), image_W_, image_H_);
    image_gpu_pending_download_ = false;
}

void VulkanTrainer::enable_backward_diagnostic_capture(bool enable) {
    capture_backward_diagnostics_ = enable;
    preprocessor_bwd_.enable_debug_capture(enable);
    if (!enable) {
        captured_bwd_d_means2D_.clear();
        captured_bwd_d_conics_.clear();
        captured_bwd_d_opacity_.clear();
        captured_bwd_d_rgb_.clear();
        captured_bwd_d_gauss2screen_.clear();
        captured_bwd_d_fabc_.clear();
        captured_bwd_d_cov3D_.clear();
        captured_bwd_d_M_.clear();
        captured_bwd_d_scale_.clear();
        captured_bwd_d_R_.clear();
        captured_bwd_d_qn_.clear();
    }
}

// ---------------------------------------------------------------------------
// activate_params — raw → activated
// ---------------------------------------------------------------------------

void VulkanTrainer::activate_params() {
    // Positions: identity (no activation)
    std::memcpy(act_positions_.data(), raw_positions_.data(),
                static_cast<size_t>(N_) * 3 * sizeof(float));

    // Scales: exp(raw)
    for (int i = 0; i < N_ * 3; ++i) {
        act_scales_[i] = std::exp(raw_scales_[i]);
    }

    // Rotations: normalize quaternion (r, x, y, z)
    for (int i = 0; i < N_; ++i) {
        float r = raw_rotations_[i * 4 + 0];
        float x = raw_rotations_[i * 4 + 1];
        float y = raw_rotations_[i * 4 + 2];
        float z = raw_rotations_[i * 4 + 3];
        float len = std::sqrt(r * r + x * x + y * y + z * z);
        if (len < 1e-12f) len = 1e-12f;
        act_rotations_[i * 4 + 0] = r / len;
        act_rotations_[i * 4 + 1] = x / len;
        act_rotations_[i * 4 + 2] = y / len;
        act_rotations_[i * 4 + 3] = z / len;
    }

    // Opacities: sigmoid(raw)
    for (int i = 0; i < N_; ++i) {
        act_opacities_[i] = 1.0f / (1.0f + std::exp(-raw_opacities_[i]));
    }

    // SH coefficients: identity (no activation)
    std::memcpy(act_sh_coeffs_.data(), raw_sh_coeffs_.data(),
                static_cast<size_t>(N_) * max_coeffs_ * 3 * sizeof(float));

    if (active_scales_gpu_) {
        active_scales_gpu_->upload(act_scales_.data(), static_cast<size_t>(N_) * 3 * sizeof(float));
        active_rotations_gpu_->upload(act_rotations_.data(), static_cast<size_t>(N_) * 4 * sizeof(float));
        active_opacities_gpu_->upload(act_opacities_.data(), static_cast<size_t>(N_) * sizeof(float));
        active_sh_gpu_->upload(act_sh_coeffs_.data(), static_cast<size_t>(N_) * max_coeffs_ * 3 * sizeof(float));
        filter_3D_gpu_->upload(act_filter_3D_.data(), static_cast<size_t>(N_) * sizeof(float));
    }
    raw_cpu_dirty_ = false;
}

bool VulkanTrainer::can_use_gpu_raw_activation() const {
    return gpu_raw_activation_enabled()
        && gpu_grad_adam_enabled()
        && tcfg_.cap_max > 0
        && tcfg_.noise_lr == 0.0f
        && tcfg_.opacity_reg == 0.0f
        && tcfg_.scale_reg == 0.0f
        && !capture_grads_
        && !capture_backward_diagnostics_;
}

void VulkanTrainer::activate_params_gpu() {
    raw_activation_pass_.bind_buffers(raw_param_gpu_bufs_[4]->handle(),
                                      raw_param_gpu_bufs_[5]->handle(),
                                      raw_param_gpu_bufs_[3]->handle(),
                                      raw_param_gpu_bufs_[1]->handle(),
                                      raw_param_gpu_bufs_[2]->handle(),
                                      active_scales_gpu_->handle(),
                                      active_rotations_gpu_->handle(),
                                      active_opacities_gpu_->handle(),
                                      active_sh_gpu_->handle());
    raw_activation_pass_.dispatch_sync(static_cast<uint32_t>(N_), static_cast<uint32_t>(max_coeffs_));
    last_gpu_raw_activation_used_ = true;
}

void VulkanTrainer::materialize_raw_params() const {
    if (!raw_cpu_dirty_) return;
    raw_param_gpu_bufs_[0]->download(raw_positions_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    {
        std::vector<float> sh_dc_scratch(static_cast<size_t>(N_) * 3);
        raw_param_gpu_bufs_[1]->download(sh_dc_scratch.data(),
                                         static_cast<size_t>(N_) * 3 * sizeof(float));
        sh_scatter_dc(sh_dc_scratch.data(), N_, max_coeffs_, raw_sh_coeffs_.data());
    }
    if (max_coeffs_ > 1) {
        std::vector<float> sh_rest_scratch(
            static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3);
        raw_param_gpu_bufs_[2]->download(sh_rest_scratch.data(),
                                         static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3 * sizeof(float));
        sh_scatter_rest(sh_rest_scratch.data(), N_, max_coeffs_, raw_sh_coeffs_.data());
    }
    raw_param_gpu_bufs_[3]->download(raw_opacities_.data(),
                                     static_cast<size_t>(N_) * sizeof(float));
    raw_param_gpu_bufs_[4]->download(raw_scales_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    raw_param_gpu_bufs_[5]->download(raw_rotations_.data(),
                                     static_cast<size_t>(N_) * 4 * sizeof(float));
    raw_view_.count = N_;
    raw_view_.max_coeffs = max_coeffs_;
    raw_view_.raw_positions = raw_positions_.data();
    raw_view_.raw_scales = raw_scales_.data();
    raw_view_.raw_rotations = raw_rotations_.data();
    raw_view_.raw_sh_coeffs = raw_sh_coeffs_.data();
    raw_view_.raw_opacities = raw_opacities_.data();
    raw_cpu_dirty_ = false;
}

const RawGaussianParams& VulkanTrainer::raw_params() const {
    materialize_raw_params();
    return raw_view_;
}

// ---------------------------------------------------------------------------
// reset_for_oracle — inject params + zero Adam state (for oracle per-step testing)
// ---------------------------------------------------------------------------

void VulkanTrainer::reset_for_oracle(const RawGaussianParams& new_raw)
{
    // Replace owned param vectors with new_raw data.
    raw_positions_.assign(new_raw.raw_positions,
                          new_raw.raw_positions + static_cast<size_t>(N_) * 3);
    raw_scales_.assign   (new_raw.raw_scales,
                          new_raw.raw_scales    + static_cast<size_t>(N_) * 3);
    raw_rotations_.assign(new_raw.raw_rotations,
                          new_raw.raw_rotations + static_cast<size_t>(N_) * 4);
    raw_sh_coeffs_.assign(new_raw.raw_sh_coeffs,
                          new_raw.raw_sh_coeffs + static_cast<size_t>(N_) * max_coeffs_ * 3);
    raw_opacities_.assign(new_raw.raw_opacities,
                          new_raw.raw_opacities + static_cast<size_t>(N_));

    // Re-upload to GPU raw param buffers (same layout as constructor).
    const size_t sz_N3 = static_cast<size_t>(N_) * 3 * sizeof(float);
    const size_t sz_N4 = static_cast<size_t>(N_) * 4 * sizeof(float);
    const size_t sz_N  = static_cast<size_t>(N_) * sizeof(float);

    raw_param_gpu_bufs_[0]->upload(raw_positions_.data(), sz_N3);    // positions
    {
        // SH DC: gather k=0 from interleaved [N, K, 3] (see sh_gather_dc note above).
        std::vector<float> sh_dc_scratch(static_cast<size_t>(N_) * 3);
        sh_gather_dc(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_dc_scratch.data());
        raw_param_gpu_bufs_[1]->upload(sh_dc_scratch.data(), sz_N3);
    }
    if (max_coeffs_ > 1) {
        const size_t sz_rest = static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3 * sizeof(float);
        std::vector<float> sh_rest_scratch(
            static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3);
        sh_gather_rest(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_rest_scratch.data());
        raw_param_gpu_bufs_[2]->upload(sh_rest_scratch.data(), sz_rest);
    }
    raw_param_gpu_bufs_[3]->upload(raw_opacities_.data(), sz_N);     // opacities
    raw_param_gpu_bufs_[4]->upload(raw_scales_.data(), sz_N3);       // scales
    raw_param_gpu_bufs_[5]->upload(raw_rotations_.data(), sz_N4);    // rotations

    // Zero all Adam moment buffers and reset step counter.
    vulkan_adam_.zero_moments();
    step_count_ = 0;
    last_loss_  = 0.0f;

    // Re-activate (raw → activated values for the next forward pass).
    activate_params();
}

#ifdef GS3D_TESTING
mcmc::DensifyResult VulkanTrainer::apply_mcmc_densification_for_test(
    float opacity_thresh,
    int cap_max,
    const mcmc::DensifySamplePlan& plan)
{
    OwnedRawParams raw_owned;
    raw_owned.sh_degree = raw_view_.sh_degree;
    raw_owned.max_coeffs = max_coeffs_;
    raw_owned.from_raw(raw_view_);
    raw_owned.filter_3D = act_filter_3D_;

    const int old_N = N_;
    mcmc::DensifyResult result = mcmc::densify_with_samples(
        raw_owned, opacity_thresh, cap_max, plan);

    N_ = result.final_count;
    raw_positions_.assign(raw_owned.positions.begin(), raw_owned.positions.end());
    raw_scales_.assign(raw_owned.scales.begin(), raw_owned.scales.end());
    raw_rotations_.assign(raw_owned.rotations.begin(), raw_owned.rotations.end());
    raw_sh_coeffs_.assign(raw_owned.sh_coeffs.begin(), raw_owned.sh_coeffs.end());
    raw_opacities_.assign(raw_owned.opacities.begin(), raw_owned.opacities.end());
    act_filter_3D_.assign(raw_owned.filter_3D.begin(), raw_owned.filter_3D.end());

    reallocate_for_n(N_, old_N);
    zero_mcmc_adam_state(vulkan_adam_, max_coeffs_, result);
    grad_means2D_accum_.resize(static_cast<size_t>(N_), 0.0f);
    activate_params();
    return result;
}
#endif

float VulkanTrainer::run_gpu_l1_loss(const float* target,
                                      int W,
                                      int H,
                                      ForwardCache& out_cache) {
    if (out_cache.rendered_image_gpu == nullptr) {
        throw std::runtime_error("VulkanTrainer::run_gpu_l1_loss: missing rendered_image_gpu");
    }
    const size_t num_elements = static_cast<size_t>(W) * static_cast<size_t>(H) * 3u;
    const size_t partial_count = (num_elements + 255u) / 256u;
    const VkDeviceSize bytes_elements = static_cast<VkDeviceSize>(num_elements) * sizeof(float);
    const VkDeviceSize bytes_partials = static_cast<VkDeviceSize>(partial_count == 0u ? 1u : partial_count) * sizeof(float);

    if (num_elements > gpu_l1_elements_capacity_) {
        gpu_l1_target_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_l1_dlpix_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_l1_elements_capacity_ = num_elements;
    }
    if (partial_count > gpu_l1_partials_capacity_) {
        gpu_l1_partials_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_partials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_l1_partials_.resize(partial_count);
        gpu_l1_partials_capacity_ = partial_count;
    }

    gpu_l1_target_buf_->upload(target, static_cast<std::size_t>(bytes_elements));
    l1_loss_pass_.bind_buffers(static_cast<VkBuffer>(out_cache.rendered_image_gpu),
                               gpu_l1_target_buf_->handle(),
                               gpu_l1_dlpix_buf_->handle(),
                               gpu_l1_partials_buf_->handle());
    l1_loss_pass_.dispatch_sync(static_cast<uint32_t>(num_elements));
    gpu_l1_partials_buf_->download(gpu_l1_partials_.data(), partial_count * sizeof(float));

    float loss = 0.0f;
    for (size_t i = 0; i < partial_count; ++i) loss += gpu_l1_partials_[i];
    out_cache.dL_dpixels_gpu = gpu_l1_dlpix_buf_->handle();
    return loss;
}

float VulkanTrainer::run_gpu_dssim_loss(const float* target,
                                         int W,
                                         int H,
                                         ForwardCache& out_cache) {
    if (out_cache.rendered_image_gpu == nullptr) {
        throw std::runtime_error("VulkanTrainer::run_gpu_dssim_loss: missing rendered_image_gpu");
    }
    const size_t num_elements = static_cast<size_t>(W) * static_cast<size_t>(H) * 3u;
    const size_t partial_count = (num_elements + 255u) / 256u;
    const VkDeviceSize bytes_elements = static_cast<VkDeviceSize>(num_elements) * sizeof(float);
    const VkDeviceSize bytes_partials = static_cast<VkDeviceSize>(partial_count == 0u ? 1u : partial_count) * sizeof(float);

    if (num_elements > gpu_dssim_elements_capacity_) {
        gpu_dssim_target_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_dlpix_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_alpha_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_beta_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_gamma_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_elements_capacity_ = num_elements;
    }
    if (partial_count > gpu_dssim_partials_capacity_) {
        gpu_dssim_partials_buf_ = std::make_unique<VulkanBuffer>(ctx_, bytes_partials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gpu_dssim_partials_.resize(partial_count);
        gpu_dssim_partials_capacity_ = partial_count;
    }

    if (!dssim_loss_pass_) {
        dssim_loss_pass_ = std::make_unique<DssimLossPass>(ctx_);
    }
    gpu_dssim_target_buf_->upload(target, static_cast<std::size_t>(bytes_elements));
    dssim_loss_pass_->bind_buffers(static_cast<VkBuffer>(out_cache.rendered_image_gpu),
                                   gpu_dssim_target_buf_->handle(),
                                   gpu_dssim_dlpix_buf_->handle(),
                                   gpu_dssim_partials_buf_->handle(),
                                   gpu_dssim_alpha_buf_->handle(),
                                   gpu_dssim_beta_buf_->handle(),
                                   gpu_dssim_gamma_buf_->handle());
    dssim_loss_pass_->dispatch_sync(static_cast<uint32_t>(W), static_cast<uint32_t>(H), tcfg_.lambda_dssim);
    gpu_dssim_partials_buf_->download(gpu_dssim_partials_.data(), partial_count * sizeof(float));

    float loss = tcfg_.lambda_dssim;
    for (size_t i = 0; i < partial_count; ++i) loss += gpu_dssim_partials_[i];
    out_cache.dL_dpixels_gpu = gpu_dssim_dlpix_buf_->handle();
    return loss;
}

// ---------------------------------------------------------------------------
// run_forward_and_loss — shared forward+loss prefix used by step() and
// forward_only(). Resets the frame arena, activates params, runs preprocess →
// bin → sort → rasterize, optionally captures intermediates, then computes the
// combined L1 + DSSIM loss + dL/dpixel into last_loss_ / dL_dpixels_.
//
// Does NOT touch backward/Adam/densification/step_count_. Returns last_loss_.
// ---------------------------------------------------------------------------

float VulkanTrainer::run_forward_and_loss(const Camera& cam,
                                          const RenderConfig& cfg,
                                          const float* target,
                                          int W,
                                          int H,
                                          PreprocessOutput& out_pre,
                                          BinningOutput& out_bin,
                                          ForwardCache& out_cache)
{
    const bool timing = training_stage_timers_enabled();
    auto t_stage = timing ? StageClock::now() : StageClock::time_point{};
    if (timing) {
        stage_timing_arena_activate_ms_ = 0.0;
        stage_timing_preprocess_process_ms_ = 0.0;
        stage_timing_preprocess_cache_refresh_ms_ = 0.0;
        stage_timing_bin_ms_ = 0.0;
        stage_timing_sort_ms_ = 0.0;
        stage_timing_raster_ms_ = 0.0;
        stage_timing_capture_ms_ = 0.0;
        stage_timing_loss_ms_ = 0.0;
    }

    alloc_.reset();
    // Grow CPU arena to hold binner+sorter R-pair CPU arrays (R × 32 bytes)
    // plus cache and gradient arrays (N × ~300 bytes). Use last step's actual R,
    // scaled by N growth ratio with 4× headroom for densification.
    // First step has no observed R yet, so use bounded tile-fanout headroom.
    {
        const size_t cur_N = static_cast<size_t>(N_);
        const size_t tile_w = static_cast<size_t>(std::max(1, cfg.tile_w));
        const size_t tile_h = static_cast<size_t>(std::max(1, cfg.tile_h));
        const size_t tiles_x = (static_cast<size_t>(W) + tile_w - 1u) / tile_w;
        const size_t tiles_y = (static_cast<size_t>(H) + tile_h - 1u) / tile_h;
        const size_t first_frame_fanout = std::max<size_t>(50u, std::min<size_t>(512u, tiles_x * tiles_y));
        const size_t est_R = (last_bin_N_ > 0)
            ? last_bin_R_ * cur_N * 4u / last_bin_N_
            : cur_N * first_frame_fanout;
        const size_t needed = (6u << 20) + cur_N * 300u + est_R * 32u;
        alloc_.grow(needed);
    }
    const bool use_gpu_raw_activation = can_use_gpu_raw_activation();
    last_gpu_raw_activation_used_ = false;
    if (use_gpu_raw_activation) {
        activate_params_gpu();
    } else {
        materialize_raw_params();
        activate_params();
    }
    if (timing) stage_timing_arena_activate_ms_ = stage_elapsed_ms(t_stage);

    // Pre-allocate ForwardCache fields that the backward passes need.
    // T_final and n_contrib are written by RasterizerVulkan::rasterize() only
    // if the pointers are non-null, so we must allocate them here.
    // cov3D, p_view, p_hom_w, cov2D, cov2D_det are allocated inside
    // PreprocessorVulkan::download_cache().
    const int HW_pixels = H * W;
    image_W_ = static_cast<uint32_t>(W);
    image_H_ = static_cast<uint32_t>(H);
    image_gpu_pending_download_ = false;
    out_cache = ForwardCache{};
    out_cache.T_final   = alloc_.allocate_array<float>(static_cast<size_t>(HW_pixels));
    out_cache.n_contrib = alloc_.allocate_array<int>  (static_cast<size_t>(HW_pixels));
    last_gpu_l1_used_ = false;
    last_gpu_dssim_used_ = false;
    last_forward_gpu_cache_used_ = false;
    last_forward_gpu_resident_outputs_ = false;
    last_forward_cpu_image_downloaded_ = false;
    last_forward_cpu_cache_downloaded_ = false;
    const bool use_gpu_l1 = gpu_l1_fast_path_allowed_
        && gpu_l1_loss_enabled()
        && tcfg_.lambda_dssim == 0.0f
        && !capture_intermediates_;
    const bool use_gpu_dssim = gpu_l1_fast_path_allowed_
        && gpu_dssim_loss_enabled()
        && tcfg_.lambda_dssim != 0.0f
        && !capture_intermediates_;
    const bool use_gpu_loss = use_gpu_l1 || use_gpu_dssim;
    const bool reuse_forward_outputs = forward_gpu_cache_allowed_
        && forward_gpu_cache_enabled()
        && !capture_intermediates_;
    const bool use_gpu_resident_outputs = use_gpu_loss && reuse_forward_outputs && N_ > 0;
    out_cache.gpu_resident_outputs = use_gpu_resident_outputs;
    out_cache.retain_gpu_outputs = use_gpu_loss || reuse_forward_outputs;
    std::memset(out_cache.T_final,   0, static_cast<size_t>(HW_pixels) * sizeof(float));
    std::memset(out_cache.n_contrib, 0, static_cast<size_t>(HW_pixels) * sizeof(int));

    // Build active config: override sh_degree with the scheduled value.
    RenderConfig active_cfg = cfg;
    active_cfg.sh_degree = active_sh_degree_;
    active_cfg.eval_3D_parity_mode = tcfg_.eval_3D && tcfg_.parity_mode;
    active_cfg.compact_eval3D_tiles = tcfg_.eval_3D && !tcfg_.parity_mode && compact_eval3d_tiles_enabled();

    // 1. Forward preprocess (populates cache: cov3D, p_view, p_hom_w, cov2D, cov2D_det)
    if (timing) t_stage = StageClock::now();
    if (use_gpu_raw_activation) {
        out_pre = preprocessor_.process_gpu_inputs(N_, max_coeffs_,
            raw_param_gpu_bufs_[0]->handle(),
            active_scales_gpu_->handle(),
            active_rotations_gpu_->handle(),
            active_opacities_gpu_->handle(),
            active_sh_gpu_->handle(),
            filter_3D_gpu_->handle(),
            cam, active_cfg, alloc_, &out_cache);
    } else {
        out_pre = preprocessor_.process(g_, cam, active_cfg, alloc_, &out_cache);
    }
    if (timing) stage_timing_preprocess_process_ms_ = stage_elapsed_ms(t_stage);
    if (timing) t_stage = StageClock::now();
    if (N_ > 0) {
        preprocessor_.download_cache(N_, out_cache, alloc_);
    }
    if (timing) stage_timing_preprocess_cache_refresh_ms_ = stage_elapsed_ms(t_stage);
    out_cache.pre = &out_pre;

    // 2. Tile binning
    if (timing) t_stage = StageClock::now();
    out_bin = binner_.bin(out_pre, N_, cam, active_cfg, alloc_);
    if (timing) stage_timing_bin_ms_ = stage_elapsed_ms(t_stage);
    out_cache.bin = &out_bin;
    // Record actual R for next step's arena estimate.
    last_bin_R_ = static_cast<size_t>(out_bin.total_pairs);
    last_bin_N_ = static_cast<size_t>(N_);

    // 3. Sort
    if (timing) t_stage = StageClock::now();
    sorter_.sort(out_bin, alloc_);
    if (timing) stage_timing_sort_ms_ = stage_elapsed_ms(t_stage);

    // 4. Rasterize (populates cache.T_final and cache.n_contrib since they are non-null)
    if (timing) t_stage = StageClock::now();
    rasterizer_.rasterize(out_pre, out_bin, cam, active_cfg, image_.data(),
                          /*depth=*/nullptr, &out_cache, &alloc_);
    image_gpu_pending_download_ = out_cache.gpu_resident_outputs;
    last_forward_gpu_resident_outputs_ = out_cache.gpu_resident_outputs;
#ifdef GS3D_TESTING
    last_forward_cpu_image_downloaded_ = rasterizer_.last_layer1_image_downloaded_for_test();
    last_forward_cpu_cache_downloaded_ = rasterizer_.last_layer1_cache_downloaded_for_test();
#endif
    if (timing) stage_timing_raster_ms_ = stage_elapsed_ms(t_stage);
    last_replay_order_count_ = out_cache.replay_order_count;

    // 4b. Optional intermediate capture — for VK-vs-CUDA parity tests only.
    //     All source pointers (pre.*, bin.*, cache.T_final/n_contrib) reference
    //     CPU-side FrameAllocator memory that the subsequent backward pass is
    //     free to overwrite, so we snapshot into owned std::vector storage here.
    //     Zero-overhead when capture_intermediates_ is false.
    if (timing) t_stage = StageClock::now();
    if (capture_intermediates_) {
        const size_t n = static_cast<size_t>(N_);
        const size_t HW = static_cast<size_t>(HW_pixels);
        const size_t R  = static_cast<size_t>(out_bin.total_pairs);
        const size_t NT = static_cast<size_t>(out_bin.num_tiles);

        captured_means2D_.assign(out_pre.means2D, out_pre.means2D + n * 2);
        // Pack (a, b, c, opacity) per Gaussian into [N*4] — match CUDA layout.
        captured_conic_opacity_.resize(n * 4);
        for (size_t i = 0; i < n; ++i) {
            captured_conic_opacity_[i * 4 + 0] = out_pre.conics[i * 3 + 0];
            captured_conic_opacity_[i * 4 + 1] = out_pre.conics[i * 3 + 1];
            captured_conic_opacity_[i * 4 + 2] = out_pre.conics[i * 3 + 2];
            captured_conic_opacity_[i * 4 + 3] = out_pre.opacities_2d[i];
        }
        captured_rgb_.assign(out_pre.rgb, out_pre.rgb + n * 3);
        captured_radii_.assign(out_pre.radii, out_pre.radii + n);
        captured_tiles_touched_.assign(out_pre.tiles_touched, out_pre.tiles_touched + n);
        captured_depths_.assign(out_pre.depths, out_pre.depths + n);
        if (out_pre.radius_f) {
            captured_radius_f_.assign(out_pre.radius_f, out_pre.radius_f + n * 2);
        } else {
            captured_radius_f_.clear();
        }
        if (out_pre.gauss2screen) {
            captured_gauss2screen_.assign(out_pre.gauss2screen, out_pre.gauss2screen + n * 16);
        } else {
            captured_gauss2screen_.clear();
        }

        if (out_bin.values_sorted && R > 0) {
            captured_sorted_gaussian_ids_.resize(R);
            for (size_t i = 0; i < R; ++i) {
                captured_sorted_gaussian_ids_[i] =
                    static_cast<int>(out_bin.values_sorted[i]);
            }
        } else {
            captured_sorted_gaussian_ids_.clear();
        }

        if (out_bin.tile_ranges && NT > 0) {
            captured_tile_offsets_.resize(NT * 2);
            for (size_t i = 0; i < NT * 2; ++i) {
                captured_tile_offsets_[i] = static_cast<int>(out_bin.tile_ranges[i]);
            }
        } else {
            captured_tile_offsets_.clear();
        }

        if (out_cache.T_final) {
            captured_T_final_.assign(out_cache.T_final, out_cache.T_final + HW);
        } else {
            captured_T_final_.clear();
        }
        if (out_cache.n_contrib) {
            captured_n_contrib_.assign(out_cache.n_contrib, out_cache.n_contrib + HW);
        } else {
            captured_n_contrib_.clear();
        }
    }
    if (timing) stage_timing_capture_ms_ = stage_elapsed_ms(t_stage);

    // 5. Combined L1 + DSSIM loss + gradient.
    if (timing) t_stage = StageClock::now();
    if (use_gpu_l1 && out_cache.rendered_image_gpu) {
        last_loss_ = run_gpu_l1_loss(target, W, H, out_cache);
        last_gpu_l1_used_ = true;
    } else if (use_gpu_dssim && out_cache.rendered_image_gpu) {
        last_loss_ = run_gpu_dssim_loss(target, W, H, out_cache);
        last_gpu_dssim_used_ = true;
    } else {
        last_loss_ = compute_combined_loss_gradient(
            image_.data(), target, dL_dpixels_.data(), W, H, tcfg_.lambda_dssim);
    }
    if (timing) stage_timing_loss_ms_ = stage_elapsed_ms(t_stage);

    return last_loss_;
}

// ---------------------------------------------------------------------------
// step
// ---------------------------------------------------------------------------

float VulkanTrainer::step(const Camera& cam,
                          const RenderConfig& cfg,
                          const float* target,
                          int W,
                          int H,
                          bool apply_update)
{
    const bool timing = training_stage_timers_enabled();
    const int timing_step = step_count_ + 1;
    auto t_step = timing ? StageClock::now() : StageClock::time_point{};
    auto t_stage = t_step;
    double forward_total_ms = 0.0;
    double backward_gpu_ms = 0.0;
    double raster_backward_gpu_ms = 0.0;
    double preprocess_backward_gpu_ms = 0.0;
    double backward_download_ms = 0.0;
    double regularize_ms = 0.0;
    double grad_capture_ms = 0.0;
    double grad_upload_ms = 0.0;
    double adam_gpu_ms = 0.0;
    double adam_capture_ms = 0.0;
    double raw_download_ms = 0.0;
    double noise_ms = 0.0;
    double grad_accum_ms = 0.0;
    double densify_ms = 0.0;
    double opacity_reset_ms = 0.0;

    // Steps 1-5: forward + loss. Out-params alias FrameAllocator memory which
    // remains valid until the next alloc_.reset() — the backward path below
    // reads them in-place.
    PreprocessOutput pre{};
    BinningOutput    bin{};
    ForwardCache     cache{};
    if (timing) t_stage = StageClock::now();
    {
        gpu_l1_fast_path_allowed_ = true;
        forward_gpu_cache_allowed_ = true;
        struct FastPathReset {
            VulkanTrainer* trainer;
            ~FastPathReset() {
                trainer->forward_gpu_cache_allowed_ = false;
                trainer->gpu_l1_fast_path_allowed_ = false;
            }
        } fast_path_reset{this};
        run_forward_and_loss(cam, cfg, target, W, H, pre, bin, cache);
    }
    if (timing) forward_total_ms = stage_elapsed_ms(t_stage);

    auto emit_stage_timing = [&]() {
        if (!timing) return;
        const double total_ms = stage_elapsed_ms(t_step);
        if (!stage_timing_header_printed_) {
            std::fprintf(stderr,
                "[GS3D_STAGE_TIMING] step,N,R,apply_update,total_ms,forward_total_ms,arena_activate_ms,preprocess_process_ms,preprocess_cache_refresh_ms,bin_ms,sort_ms,raster_ms,capture_ms,loss_ms,backward_gpu_ms,backward_download_ms,regularize_ms,grad_capture_ms,grad_upload_ms,adam_gpu_ms,adam_capture_ms,raw_download_ms,noise_ms,grad_accum_ms,densify_ms,opacity_reset_ms\n");
            stage_timing_header_printed_ = true;
        }
        std::fprintf(stderr,
            "[GS3D_STAGE_TIMING] %d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            timing_step, N_, bin.total_pairs, apply_update ? 1 : 0,
            total_ms, forward_total_ms, stage_timing_arena_activate_ms_,
            stage_timing_preprocess_process_ms_, stage_timing_preprocess_cache_refresh_ms_,
            stage_timing_bin_ms_, stage_timing_sort_ms_, stage_timing_raster_ms_,
            stage_timing_capture_ms_, stage_timing_loss_ms_, backward_gpu_ms,
            backward_download_ms, regularize_ms, grad_capture_ms, grad_upload_ms,
            adam_gpu_ms, adam_capture_ms, raw_download_ms, noise_ms, grad_accum_ms,
            densify_ms, opacity_reset_ms);
        std::fflush(stderr);
    };

    // Re-derive active_cfg locally (run_forward_and_loss applies the same
    // sh_degree override but the result is needed for the backward record).
    RenderConfig active_cfg = cfg;
    active_cfg.sh_degree = active_sh_degree_;
    active_cfg.eval_3D_parity_mode = tcfg_.eval_3D && tcfg_.parity_mode;

    last_gpu_grad_adam_used_ = false;
    const bool use_gpu_grad_adam = gpu_grad_adam_enabled()
        && N_ > 0
        && tcfg_.cap_max > 0
        && tcfg_.opacity_reg == 0.0f
        && tcfg_.scale_reg == 0.0f
        && !capture_grads_
        && !capture_backward_diagnostics_;

    // 6+7. Rasterize backward + preprocess backward chained into one CB.
    //      rasterize_bwd writes dL_d* to GPU; preprocess_bwd reads them directly.
    //      Eliminates rgrad CPU round-trip + merges 2 submit+wait into 1.
    GradientOutput grads{};
    if (timing) t_stage = StageClock::now();
    {
        PreprocessBackwardGpuInputs pb_gpu_inputs{};
        pb_gpu_inputs.positions = raw_param_gpu_bufs_[0]->handle();
        pb_gpu_inputs.radii = pre.radii_gpu ? static_cast<VkBuffer>(pre.radii_gpu) : VK_NULL_HANDLE;
        pb_gpu_inputs.sh_coeffs = active_sh_gpu_->handle();
        pb_gpu_inputs.scales = active_scales_gpu_->handle();
        pb_gpu_inputs.rotations = active_rotations_gpu_->handle();
        pb_gpu_inputs.opacities = active_opacities_gpu_->handle();
        pb_gpu_inputs.raw_rotations = raw_param_gpu_bufs_[5]->handle();
        pb_gpu_inputs.filter_3D = filter_3D_gpu_->handle();

        if (split_backward_timing_enabled()) {
            auto t_split = StageClock::now();
            VkCommandBuffer rb_cmd = ctx_.allocatePrimary();
            VkCommandBufferBeginInfo rb_bi{};
            rb_bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            rb_bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(rb_cmd, &rb_bi);
            rasterizer_bwd_.backward_record_into(rb_cmd, pre, bin, N_, cam, active_cfg,
                                                  cache, dL_dpixels_.data(), image_.data());
            vkEndCommandBuffer(rb_cmd);
            ctx_.submitAndWait(rb_cmd);
            ctx_.freePrimary(rb_cmd);
            raster_backward_gpu_ms = stage_elapsed_ms(t_split);
#ifdef GS3D_TESTING
            last_forward_gpu_cache_used_ = rasterizer_bwd_.last_forward_gpu_cache_used_for_test();
#endif

            t_split = StageClock::now();
            VkCommandBuffer pb_cmd = ctx_.allocatePrimary();
            VkCommandBufferBeginInfo pb_bi{};
            pb_bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            pb_bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(pb_cmd, &pb_bi);
            insert_compute_barrier(pb_cmd);
            preprocessor_bwd_.backward_record_into(pb_cmd, g_, N_, cam, active_cfg, cache,
                rasterizer_bwd_.dL_dconics_buf(),
                rasterizer_bwd_.dL_dopacity_buf(),
                rasterizer_bwd_.dL_dcolors_buf(),
                rasterizer_bwd_.dL_dmeans2D_buf(),
                raw_view_,
                rasterizer_bwd_.dL_dgauss2screen_buf(),
                &pb_gpu_inputs);
            vkEndCommandBuffer(pb_cmd);
            ctx_.submitAndWait(pb_cmd);
            ctx_.freePrimary(pb_cmd);
            preprocess_backward_gpu_ms = stage_elapsed_ms(t_split);
        } else {
            VkCommandBuffer bwd_cmd = ctx_.allocatePrimary();
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(bwd_cmd, &bi);

            rasterizer_bwd_.backward_record_into(bwd_cmd, pre, bin, N_, cam, active_cfg,
                                                  cache, dL_dpixels_.data(), image_.data());
#ifdef GS3D_TESTING
            last_forward_gpu_cache_used_ = rasterizer_bwd_.last_forward_gpu_cache_used_for_test();
#endif
            insert_compute_barrier(bwd_cmd);
            preprocessor_bwd_.backward_record_into(bwd_cmd, g_, N_, cam, active_cfg, cache,
                rasterizer_bwd_.dL_dconics_buf(),
                rasterizer_bwd_.dL_dopacity_buf(),
                rasterizer_bwd_.dL_dcolors_buf(),
                rasterizer_bwd_.dL_dmeans2D_buf(),
                raw_view_,
                rasterizer_bwd_.dL_dgauss2screen_buf(),
                &pb_gpu_inputs);

            vkEndCommandBuffer(bwd_cmd);
            ctx_.submitAndWait(bwd_cmd);
            ctx_.freePrimary(bwd_cmd);
        }
    }
    if (timing) {
        backward_gpu_ms = stage_elapsed_ms(t_stage);
        if (split_backward_timing_enabled()) {
            static bool split_timing_header_printed = false;
            if (!split_timing_header_printed) {
                std::fprintf(stderr,
                    "[GS3D_BACKWARD_SPLIT] step,N,R,raster_bwd_ms,preprocess_bwd_ms,total_bwd_ms\n");
                split_timing_header_printed = true;
            }
            std::fprintf(stderr,
                "[GS3D_BACKWARD_SPLIT] %d,%d,%d,%.3f,%.3f,%.3f\n",
                timing_step, N_, bin.total_pairs, raster_backward_gpu_ms,
                preprocess_backward_gpu_ms, backward_gpu_ms);
            std::fflush(stderr);
        }
    }
    if (capture_backward_diagnostics_) {
        rasterizer_bwd_.download_outputs(N_,
            captured_bwd_d_means2D_,
            captured_bwd_d_conics_,
            captured_bwd_d_opacity_,
            captured_bwd_d_rgb_,
            &captured_bwd_d_gauss2screen_);
        preprocessor_bwd_.download_debug_buffers(N_,
            captured_bwd_d_fabc_,
            captured_bwd_d_cov3D_,
            captured_bwd_d_M_,
            captured_bwd_d_scale_,
            captured_bwd_d_R_,
            captured_bwd_d_qn_);
    }
    if (timing) t_stage = StageClock::now();
    if (!use_gpu_grad_adam) {
        preprocessor_bwd_.download_grads(N_, max_coeffs_, grads, alloc_);
    }
    if (timing) backward_download_ms = stage_elapsed_ms(t_stage);

    if (!use_gpu_grad_adam) {
        // SP-6 T2: Regularization gradient injection.
        // After preprocessor_bwd_ writes raw-param gradients, add L1 regularization
        // gradient corrections before the GPU upload.
        //
        // Opacity reg: loss += opacity_reg * mean(|sigmoid(raw_op)|)
        //   => dL/d_raw_op[i] += (opacity_reg / N) * sig * (1 - sig)
        //      where sig = sigmoid(raw_op[i]) = act_opacities_[i]
        //
        // Scale reg: loss += scale_reg * mean(|exp(raw_sc)|) over [N,3]
        //   => dL/d_raw_sc[i,k] += (scale_reg / (3*N)) * exp(raw_sc[i,k])
        //      where exp(raw_sc[i,k]) = act_scales_[i*3+k]
        //
        // Evidence: test RegularizationGrad.ScaleGradAtRawLogScale asserts the [N,3]
        // mean denominator; test RegularizationGrad.OpacityGradAtRawZero asserts plain
        // sigmoid derivative (not op_sigmoid which is for noise injection).
        if (timing) t_stage = StageClock::now();
        if (N_ > 0 && (tcfg_.opacity_reg > 0.f || tcfg_.scale_reg > 0.f)) {
            const float inv_N = 1.0f / static_cast<float>(N_);
            for (int i = 0; i < N_; ++i) {
                if (tcfg_.opacity_reg > 0.f) {
                    const float sig = act_opacities_[static_cast<size_t>(i)];
                    grads.d_raw_opacities[static_cast<size_t>(i)] += (tcfg_.opacity_reg * inv_N) * sig * (1.f - sig);
                }
                if (tcfg_.scale_reg > 0.f) {
                    const float coeff = (tcfg_.scale_reg * inv_N) / 3.0f;
                    for (int k = 0; k < 3; ++k) {
                        const size_t idx = static_cast<size_t>(i) * 3 + k;
                        grads.d_raw_scales[idx] += coeff * act_scales_[idx];
                    }
                }
            }
        }
        if (timing) regularize_ms = stage_elapsed_ms(t_stage);

        // Gradient capture — optional, for testing. Copies grads after regularization
        // corrections, before GPU Adam upload. Matches Python autograd capture point.
        if (timing) t_stage = StageClock::now();
        if (capture_grads_) {
            const size_t n = static_cast<size_t>(N_);
            const size_t mc3 = static_cast<size_t>(max_coeffs_) * 3;
            captured_grad_positions_.assign(grads.d_raw_positions,
                                            grads.d_raw_positions + n * 3);
            captured_grad_scales_.assign(grads.d_raw_scales,
                                         grads.d_raw_scales + n * 3);
            captured_grad_rotations_.assign(grads.d_raw_rotations,
                                            grads.d_raw_rotations + n * 4);
            captured_grad_sh_.assign(grads.d_raw_sh_coeffs,
                                     grads.d_raw_sh_coeffs + n * mc3);
            captured_grad_opacities_.assign(grads.d_raw_opacities,
                                            grads.d_raw_opacities + n);
        }
        if (timing) grad_capture_ms = stage_elapsed_ms(t_stage);

        // 8. Upload CPU gradients to gradient GPU buffers.
        //    SH: DC (first N*3 floats) and rest (remaining N*(max_coeffs-1)*3 floats)
        //    are uploaded to separate buffers to match the separate raw param GPU bufs.
        if (timing) t_stage = StageClock::now();
        grad_positions_gpu_->upload(grads.d_raw_positions,
                                    static_cast<size_t>(N_) * 3 * sizeof(float));
        {
            // Same DC/REST gather as raw params: gradient layout matches raw_sh
            // layout [N, K, 3] (preprocess_backward.comp:600-602 writes `d_sh[i*K*3+0..2]`
            // for DC). Without this gather, group 1 (lr=2.5e-3) would receive G[0]'s
            // 48 gradient floats and G[1]'s first 12 — wrong indices.
            std::vector<float> dsh_dc_scratch(static_cast<size_t>(N_) * 3);
            sh_gather_dc(grads.d_raw_sh_coeffs, N_, max_coeffs_, dsh_dc_scratch.data());
            grad_sh_dc_gpu_->upload(dsh_dc_scratch.data(),
                                    static_cast<size_t>(N_) * 3 * sizeof(float));
        }
        if (max_coeffs_ > 1) {
            std::vector<float> dsh_rest_scratch(
                static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3);
            sh_gather_rest(grads.d_raw_sh_coeffs, N_, max_coeffs_, dsh_rest_scratch.data());
            grad_sh_rest_gpu_->upload(dsh_rest_scratch.data(),
                                      static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3 * sizeof(float));
        }
        grad_opacities_gpu_->upload(grads.d_raw_opacities,
                                    static_cast<size_t>(N_) * sizeof(float));
        grad_scales_gpu_->upload(grads.d_raw_scales,
                                 static_cast<size_t>(N_) * 3 * sizeof(float));
        grad_rotations_gpu_->upload(grads.d_raw_rotations,
                                    static_cast<size_t>(N_) * 4 * sizeof(float));
        if (timing) grad_upload_ms = stage_elapsed_ms(t_stage);
    }

    // 9. GPU Adam step — increment step counter first (1-indexed for bias correction).
    ++step_count_;
    const uint32_t s = static_cast<uint32_t>(step_count_);

    float pos_lr = spatial_lr_schedule(tcfg_.pos_lr_init, tcfg_.pos_lr_final,
                                       tcfg_.spatial_lr_scale,
                                       step_count_, tcfg_.max_steps);

    if (tcfg_.sh_degree_warmup > 0)
        active_sh_degree_ = std::min(step_count_ / tcfg_.sh_degree_warmup,
                                     tcfg_.sh_degree_max);
    else
        active_sh_degree_ = tcfg_.sh_degree_max;

    if (!apply_update) {
        captured_adam_m_.clear();
        captured_adam_v_.clear();
        emit_stage_timing();
        return last_loss_;
    }

    // GPU Adam — chain all 6 groups into one CB (6 submit+wait → 1).
    if (timing) t_stage = StageClock::now();
    {
        VkCommandBuffer adam_cmd = ctx_.allocatePrimary();
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(adam_cmd, &bi);

        VkBuffer grad_pos = grad_positions_gpu_->handle();
        VkBuffer grad_sh_dc = grad_sh_dc_gpu_->handle();
        VkBuffer grad_sh_rest = grad_sh_rest_gpu_->handle();
        VkBuffer grad_opa = grad_opacities_gpu_->handle();
        VkBuffer grad_sc = grad_scales_gpu_->handle();
        VkBuffer grad_rot = grad_rotations_gpu_->handle();
        if (use_gpu_grad_adam) {
            grad_pos = preprocessor_bwd_.d_raw_positions_buf();
            grad_opa = preprocessor_bwd_.d_raw_opacities_buf();
            grad_sc = preprocessor_bwd_.d_raw_scales_buf();
            grad_rot = preprocessor_bwd_.d_raw_rotations_buf();
            sh_grad_split_pass_.bind_buffers(preprocessor_bwd_.d_raw_sh_buf(), grad_sh_dc, grad_sh_rest);
            sh_grad_split_pass_.record(adam_cmd, static_cast<uint32_t>(N_), static_cast<uint32_t>(max_coeffs_));
            insert_compute_barrier(adam_cmd);
            last_gpu_grad_adam_used_ = true;
        }

        vulkan_adam_.step_group_record(adam_cmd, 0,
            raw_param_gpu_bufs_[0]->handle(), grad_pos, pos_lr, s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 1,
            raw_param_gpu_bufs_[1]->handle(), grad_sh_dc, group_lrs_[1], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 2,
            raw_param_gpu_bufs_[2]->handle(), grad_sh_rest, group_lrs_[2], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 3,
            raw_param_gpu_bufs_[3]->handle(), grad_opa, group_lrs_[3], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 4,
            raw_param_gpu_bufs_[4]->handle(), grad_sc, group_lrs_[4], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 5,
            raw_param_gpu_bufs_[5]->handle(), grad_rot, group_lrs_[5], s);

        vkEndCommandBuffer(adam_cmd);
        ctx_.submitAndWait(adam_cmd);
        ctx_.freePrimary(adam_cmd);
    }
    if (timing) adam_gpu_ms = stage_elapsed_ms(t_stage);

    if (timing) t_stage = StageClock::now();
    if (capture_adam_) {
        captured_adam_m_.resize(6);
        captured_adam_v_.resize(6);
        for (int i = 0; i < 6; ++i) {
            vulkan_adam_.download_moments(i, captured_adam_m_[i], captured_adam_v_[i]);
        }
    }
    if (timing) adam_capture_ms = stage_elapsed_ms(t_stage);

    const bool densification_enabled = (tcfg_.densify_from_step > 0);
    const bool should_densify =
        densification_enabled &&
        (step_count_ >= tcfg_.densify_from_step) &&
        (step_count_ <= tcfg_.densify_until_step) &&
        (step_count_ % tcfg_.densify_interval == 0);
    const bool should_reset_opacity =
        densification_enabled && tcfg_.opacity_reset_interval > 0 &&
        step_count_ > 0 && step_count_ <= tcfg_.densify_until_step &&
        (step_count_ % tcfg_.opacity_reset_interval == 0);
    const bool skip_raw_download = can_use_gpu_raw_activation() && !should_densify && !should_reset_opacity;

    if (timing) t_stage = StageClock::now();
    raw_cpu_dirty_ = true;
    if (!skip_raw_download) {
        materialize_raw_params();
    }
    if (timing) raw_download_ms = stage_elapsed_ms(t_stage);

    if (timing) t_stage = StageClock::now();
    inject_position_noise(pos_lr);
    if (timing) noise_ms = stage_elapsed_ms(t_stage);

    if (timing) t_stage = StageClock::now();
    if (tcfg_.cap_max <= 0) {
        for (int i = 0; i < N_; ++i) {
            const float* dp = grads.d_raw_positions + i * 3;
            const float norm = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
            grad_means2D_accum_[i] += norm;
        }
    }
    if (timing) grad_accum_ms = stage_elapsed_ms(t_stage);

    if (timing) t_stage = StageClock::now();
    if (should_densify) {
        OwnedRawParams raw_owned;
        raw_owned.sh_degree  = raw_view_.sh_degree;
        raw_owned.max_coeffs = max_coeffs_;
        raw_owned.from_raw(raw_view_);
        raw_owned.filter_3D = act_filter_3D_;

        const int old_N = N_;
        mcmc::DensifyResult mcmc_result{N_, {}, {}};

        if (tcfg_.cap_max > 0) {
            mcmc_result = mcmc::densify_ex(
                raw_owned, tcfg_.opacity_thresh, tcfg_.cap_max,
                static_cast<uint32_t>(step_count_));
            N_ = mcmc_result.final_count;
        } else {
            float scene_extent = 1.0f;
            if (N_ > 0) {
                float pos_min[3] = { raw_positions_[0], raw_positions_[1], raw_positions_[2] };
                float pos_max[3] = { raw_positions_[0], raw_positions_[1], raw_positions_[2] };
                for (int i = 1; i < N_; ++i) {
                    for (int k = 0; k < 3; ++k) {
                        const float v = raw_positions_[static_cast<size_t>(i) * 3 + k];
                        if (v < pos_min[k]) pos_min[k] = v;
                        if (v > pos_max[k]) pos_max[k] = v;
                    }
                }
                scene_extent = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    scene_extent = std::max(scene_extent, pos_max[k] - pos_min[k]);
                }
                if (scene_extent < 1e-6f) scene_extent = 1.0f;
            }
            N_ = densify_and_prune(
                raw_owned, grad_means2D_accum_.data(), step_count_, tcfg_, scene_extent);
        }

        raw_positions_.assign(raw_owned.positions.begin(), raw_owned.positions.end());
        raw_scales_.assign(raw_owned.scales.begin(), raw_owned.scales.end());
        raw_rotations_.assign(raw_owned.rotations.begin(), raw_owned.rotations.end());
        raw_sh_coeffs_.assign(raw_owned.sh_coeffs.begin(), raw_owned.sh_coeffs.end());
        raw_opacities_.assign(raw_owned.opacities.begin(), raw_owned.opacities.end());
        act_filter_3D_.assign(raw_owned.filter_3D.begin(), raw_owned.filter_3D.end());

        if (tcfg_.cap_max > 0) {
            reallocate_for_n(N_, old_N);
            zero_mcmc_adam_state(vulkan_adam_, max_coeffs_, mcmc_result);
            grad_means2D_accum_.resize(static_cast<size_t>(N_), 0.0f);
        } else {
            reallocate_for_n(N_);
            grad_means2D_accum_.assign(static_cast<size_t>(N_), 0.0f);
        }
    }
    if (timing) densify_ms = stage_elapsed_ms(t_stage);

    if (timing) t_stage = StageClock::now();
    if (should_reset_opacity) {
        raw_opacities_.assign(static_cast<size_t>(N_), logit_clamped(0.01f));
        raw_view_.raw_opacities = raw_opacities_.data();
        if (N_ > 0) {
            raw_param_gpu_bufs_[3]->upload(raw_opacities_.data(),
                                           static_cast<size_t>(N_) * sizeof(float));
            std::vector<uint32_t> opacity_indices(static_cast<size_t>(N_));
            std::iota(opacity_indices.begin(), opacity_indices.end(), 0u);
            vulkan_adam_.zero_moment_floats(3, opacity_indices);
        }
    }
    if (timing) opacity_reset_ms = stage_elapsed_ms(t_stage);

    emit_stage_timing();
    return last_loss_;
}

// ---------------------------------------------------------------------------
// forward_only — forward + loss only, no backward / Adam / densification
// ---------------------------------------------------------------------------
// Used by VK-vs-CUDA parity harnesses on eval_3D=true paths where
// RasterizerBackwardVulkan currently throws. The same accessors that step()
// populates are valid after this call (rendered_image(), captured_*(),
// last_loss()). Does NOT advance step_count_ — call sites that need parity
// runs to be repeatable can compare against a fixed CUDA golden.

float VulkanTrainer::forward_only(const Camera& cam,
                                  const RenderConfig& cfg,
                                  const float* target,
                                  int W,
                                  int H)
{
    PreprocessOutput pre{};
    BinningOutput    bin{};
    ForwardCache     cache{};
    return run_forward_and_loss(cam, cfg, target, W, H, pre, bin, cache);
}

// ---------------------------------------------------------------------------
// inject_position_noise — covariance-scaled noise for near-dead Gaussians
// ---------------------------------------------------------------------------
// Matches train.py:141-148:
//   noise = randn_like(xyz) * sigmoid(-100*(opacity - 0.995)) * noise_lr * xyz_lr
//   noise = Sigma @ noise   where Sigma = L @ L^T
//   xyz += noise
//
// Evidence: op_sigmoid(1 - opacity) = sigmoid(-100*(opacity - 0.005))
// when opacity >= ~0.01 (active) the factor is < 1e-6 (no noise injected);
// when opacity < ~0.005 (dead) the factor approaches 0.62 (noise injected).
void VulkanTrainer::inject_position_noise(float pos_lr) {
    if (tcfg_.noise_lr == 0.f || N_ == 0) return;

    std::mt19937 rng(static_cast<uint32_t>(step_count_));
    std::normal_distribution<float> normal(0.f, 1.f);

    bool changed = false;
    for (int i = 0; i < N_; ++i) {
        const float* act_sc  = act_scales_.data()    + static_cast<size_t>(i) * 3;
        const float* act_rot = act_rotations_.data() + static_cast<size_t>(i) * 4;
        const float  opacity = act_opacities_[static_cast<size_t>(i)];

        // Soft-step gate: near zero for active Gaussians (opacity close to 1),
        // approaches 1 for dead Gaussians (opacity close to 0).
        const float opacity_factor = op_sigmoid(1.f - opacity);
        if (opacity_factor < 1e-6f) continue;

        // Build L = R @ diag(act_scale), then Sigma = L @ L^T.
        float L[3][3];
        build_L(act_rot, act_sc, L);

        float Sigma[3][3] = {};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k)
                    Sigma[r][c] += L[r][k] * L[c][k];

        // Draw isotropic noise then scale by Sigma, opacity_factor, noise_lr, pos_lr.
        const float scalar = opacity_factor * tcfg_.noise_lr * pos_lr;
        float eta[3];
        for (int k = 0; k < 3; ++k) eta[k] = scalar * normal(rng);

        float* pos = raw_positions_.data() + static_cast<size_t>(i) * 3;
        for (int r = 0; r < 3; ++r) {
            float delta = 0.f;
            for (int c = 0; c < 3; ++c) delta += Sigma[r][c] * eta[c];
            pos[r] += delta;
            changed = true;
        }
    }
    if (changed && raw_param_gpu_bufs_[0]) {
        raw_param_gpu_bufs_[0]->upload(raw_positions_.data(), static_cast<size_t>(N_) * 3 * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// reallocate_for_n — re-create GPU buffers and Adam groups after N changes
// ---------------------------------------------------------------------------

void VulkanTrainer::reallocate_for_n(int new_N, int old_N)
{
    N_ = new_N;
    const int rest_coeffs = max_coeffs_ - 1;

    // Re-size activated-value vectors.
    act_positions_.resize(static_cast<size_t>(N_) * 3);
    act_scales_.resize   (static_cast<size_t>(N_) * 3);
    act_rotations_.resize(static_cast<size_t>(N_) * 4);
    act_opacities_.resize(static_cast<size_t>(N_));
    act_sh_coeffs_.resize(static_cast<size_t>(N_) * max_coeffs_ * 3);
    act_filter_3D_.resize(static_cast<size_t>(N_), 0.0f);

    // Update GaussianData view pointers.
    g_.count      = N_;
    g_.positions  = act_positions_.data();
    g_.scales     = act_scales_.data();
    g_.rotations  = act_rotations_.data();
    g_.opacities  = act_opacities_.data();
    g_.sh_coeffs  = act_sh_coeffs_.data();
    g_.filter_3D  = act_filter_3D_.data();

    // Update RawGaussianParams view pointers.
    raw_view_.count         = N_;
    raw_view_.raw_positions = raw_positions_.data();
    raw_view_.raw_scales    = raw_scales_.data();
    raw_view_.raw_rotations = raw_rotations_.data();
    raw_view_.raw_sh_coeffs = raw_sh_coeffs_.data();
    raw_view_.raw_opacities = raw_opacities_.data();

    // Buffer sizes.
    // Guard: VkBuffer size must be > 0 (VUID-VkBufferCreateInfo-size-00912).
    // When N_==0 (all Gaussians pruned), logical sizes are 0; clamp to 4 bytes.
    // upload() calls are skipped when N_==0 — no data to transfer.
    const size_t sz_N3 = std::max(
        static_cast<size_t>(N_) * 3 * sizeof(float), static_cast<size_t>(4));
    const size_t sz_N4 = std::max(
        static_cast<size_t>(N_) * 4 * sizeof(float), static_cast<size_t>(4));
    const size_t sz_N  = std::max(
        static_cast<size_t>(N_) * sizeof(float),     static_cast<size_t>(4));
    const size_t sz_sh_dc = std::max(
        static_cast<size_t>(N_) * 3 * sizeof(float), static_cast<size_t>(4));
    const size_t sz_sh_rest = std::max(
        static_cast<size_t>(N_) * rest_coeffs * 3 * sizeof(float),
        static_cast<size_t>(4));
    const size_t sz_sh_full = std::max(
        static_cast<size_t>(N_) * max_coeffs_ * 3 * sizeof(float),
        static_cast<size_t>(4));

    // Re-allocate raw param GPU buffers and upload current CPU values (skip when N_==0).
    raw_param_gpu_bufs_[0] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0)
        raw_param_gpu_bufs_[0]->upload(raw_positions_.data(),
                                       static_cast<size_t>(N_) * 3 * sizeof(float));

    raw_param_gpu_bufs_[1] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_dc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0) {
        // SH DC gather: see helper note at top of file.
        std::vector<float> sh_dc_scratch(static_cast<size_t>(N_) * 3);
        sh_gather_dc(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_dc_scratch.data());
        raw_param_gpu_bufs_[1]->upload(sh_dc_scratch.data(),
                                       static_cast<size_t>(N_) * 3 * sizeof(float));
    }

    raw_param_gpu_bufs_[2] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0 && max_coeffs_ > 1) {
        std::vector<float> sh_rest_scratch(
            static_cast<size_t>(N_) * rest_coeffs * 3);
        sh_gather_rest(raw_sh_coeffs_.data(), N_, max_coeffs_, sh_rest_scratch.data());
        raw_param_gpu_bufs_[2]->upload(sh_rest_scratch.data(),
            static_cast<size_t>(N_) * rest_coeffs * 3 * sizeof(float));
    }

    raw_param_gpu_bufs_[3] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0)
        raw_param_gpu_bufs_[3]->upload(raw_opacities_.data(),
                                       static_cast<size_t>(N_) * sizeof(float));

    raw_param_gpu_bufs_[4] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0)
        raw_param_gpu_bufs_[4]->upload(raw_scales_.data(),
                                       static_cast<size_t>(N_) * 3 * sizeof(float));

    raw_param_gpu_bufs_[5] = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0)
        raw_param_gpu_bufs_[5]->upload(raw_rotations_.data(),
                                       static_cast<size_t>(N_) * 4 * sizeof(float));

    // Re-allocate gradient GPU buffers.
    grad_positions_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_sh_dc_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_dc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_sh_rest_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_opacities_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_scales_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    grad_rotations_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    active_scales_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_rotations_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_opacities_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    active_sh_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_full, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    filter_3D_gpu_ = std::make_unique<VulkanBuffer>(
        ctx_, sz_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0) {
        filter_3D_gpu_->upload(act_filter_3D_.data(), static_cast<size_t>(N_) * sizeof(float));
    }
    raw_cpu_dirty_ = false;

    struct GroupSpec { uint32_t n; float lr; };
    const GroupSpec specs[6] = {
        { static_cast<uint32_t>(N_ * 3),                                   group_lrs_[0] },
        { static_cast<uint32_t>(N_ * 3),                                   group_lrs_[1] },
        { static_cast<uint32_t>(N_ * std::max(rest_coeffs, 0) * 3),         group_lrs_[2] },
        { static_cast<uint32_t>(N_),                                       group_lrs_[3] },
        { static_cast<uint32_t>(N_ * 3),                                   group_lrs_[4] },
        { static_cast<uint32_t>(N_ * 4),                                   group_lrs_[5] },
    };

    if (old_N < 0) {
        vulkan_adam_.reset_groups();
        for (int i = 0; i < 6; ++i) {
            vulkan_adam_.add_group(specs[i].n, specs[i].lr);
        }
    } else if (new_N > old_N) {
        const int delta = new_N - old_N;
        const uint32_t added[6] = {
            static_cast<uint32_t>(delta * 3),
            static_cast<uint32_t>(delta * 3),
            static_cast<uint32_t>(delta * std::max(rest_coeffs, 0) * 3),
            static_cast<uint32_t>(delta),
            static_cast<uint32_t>(delta * 3),
            static_cast<uint32_t>(delta * 4),
        };
        for (int i = 0; i < 6; ++i) {
            vulkan_adam_.extend_group(i, added[i]);
        }
    } else if (new_N < old_N) {
        for (int i = 0; i < 6; ++i) {
            vulkan_adam_.shrink_group(i, specs[i].n);
        }
    }
}
