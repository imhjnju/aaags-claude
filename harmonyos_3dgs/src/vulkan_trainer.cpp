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
    , rasterizer_(ctx, /*eval_3D=*/tcfg.eval_3D, /*disable_subtile_resort=*/tcfg.parity_mode)
    , rasterizer_bwd_(ctx)
    , preprocessor_bwd_(ctx, tcfg.proper_ewa)
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

    // 8. Allocate per-frame output buffers (H*W*3).
    const size_t HW3 = static_cast<size_t>(cam_height) * cam_width * 3;
    image_.resize(HW3, 0.0f);
    dL_dpixels_.resize(HW3, 0.0f);

    // 9. Initialize accumulated gradient norms (for densification).
    grad_means2D_accum_.assign(static_cast<size_t>(N_), 0.0f);
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
    alloc_.reset();
    // Grow CPU arena to hold binner+sorter R-pair CPU arrays (R × 32 bytes)
    // plus cache and gradient arrays (N × ~300 bytes). Use last step's actual R,
    // scaled by N growth ratio with 4× headroom for densification.
    // First step has no observed R yet, so cover the worst-case tile fanout.
    {
        const size_t cur_N = static_cast<size_t>(N_);
        const size_t tile_w = static_cast<size_t>(std::max(1, cfg.tile_w));
        const size_t tile_h = static_cast<size_t>(std::max(1, cfg.tile_h));
        const size_t tiles_x = (static_cast<size_t>(W) + tile_w - 1u) / tile_w;
        const size_t tiles_y = (static_cast<size_t>(H) + tile_h - 1u) / tile_h;
        const size_t max_tiles_per_gaussian = std::max<size_t>(50u, tiles_x * tiles_y);
        const size_t est_R = (last_bin_N_ > 0)
            ? last_bin_R_ * cur_N * 4u / last_bin_N_
            : cur_N * max_tiles_per_gaussian;
        const size_t needed = (6u << 20) + cur_N * 300u + est_R * 32u;
        alloc_.grow(needed);
    }
    activate_params();

    // Pre-allocate ForwardCache fields that the backward passes need.
    // T_final and n_contrib are written by RasterizerVulkan::rasterize() only
    // if the pointers are non-null, so we must allocate them here.
    // cov3D, p_view, p_hom_w, cov2D, cov2D_det are allocated inside
    // PreprocessorVulkan::download_cache().
    const int HW_pixels = H * W;
    out_cache = ForwardCache{};
    out_cache.T_final   = alloc_.allocate_array<float>(static_cast<size_t>(HW_pixels));
    out_cache.n_contrib = alloc_.allocate_array<int>  (static_cast<size_t>(HW_pixels));
    std::memset(out_cache.T_final,   0, static_cast<size_t>(HW_pixels) * sizeof(float));
    std::memset(out_cache.n_contrib, 0, static_cast<size_t>(HW_pixels) * sizeof(int));

    // Build active config: override sh_degree with the scheduled value.
    RenderConfig active_cfg = cfg;
    active_cfg.sh_degree = active_sh_degree_;
    active_cfg.eval_3D_parity_mode = tcfg_.eval_3D && tcfg_.parity_mode;

    // 1. Forward preprocess (populates cache: cov3D, p_view, p_hom_w, cov2D, cov2D_det)
    out_pre = preprocessor_.process(g_, cam, active_cfg, alloc_, &out_cache);
    preprocessor_.download_cache(N_, out_cache, alloc_);
    out_cache.pre = &out_pre;

    // 2. Tile binning
    out_bin = binner_.bin(out_pre, N_, cam, active_cfg, alloc_);
    out_cache.bin = &out_bin;
    // Record actual R for next step's arena estimate.
    last_bin_R_ = static_cast<size_t>(out_bin.total_pairs);
    last_bin_N_ = static_cast<size_t>(N_);

    // 3. Sort
    sorter_.sort(out_bin, alloc_);

    // 4. Rasterize (populates cache.T_final and cache.n_contrib since they are non-null)
    rasterizer_.rasterize(out_pre, out_bin, cam, active_cfg, image_.data(),
                          /*depth=*/nullptr, &out_cache, &alloc_);
    last_replay_order_count_ = out_cache.replay_order_count;

    // 4b. Optional intermediate capture — for VK-vs-CUDA parity tests only.
    //     All source pointers (pre.*, bin.*, cache.T_final/n_contrib) reference
    //     CPU-side FrameAllocator memory that the subsequent backward pass is
    //     free to overwrite, so we snapshot into owned std::vector storage here.
    //     Zero-overhead when capture_intermediates_ is false.
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

    // 5. Combined L1 + DSSIM loss + gradient.
    //    lambda_dssim=0 disables the O(W*H*WINDOW^2) SSIM computation (use for large images).
    last_loss_ = compute_combined_loss_gradient(
        image_.data(), target, dL_dpixels_.data(), W, H, tcfg_.lambda_dssim);

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
    // Steps 1-5: forward + loss. Out-params alias FrameAllocator memory which
    // remains valid until the next alloc_.reset() — the backward path below
    // reads them in-place.
    PreprocessOutput pre{};
    BinningOutput    bin{};
    ForwardCache     cache{};
    run_forward_and_loss(cam, cfg, target, W, H, pre, bin, cache);

    // Re-derive active_cfg locally (run_forward_and_loss applies the same
    // sh_degree override but the result is needed for the backward record).
    RenderConfig active_cfg = cfg;
    active_cfg.sh_degree = active_sh_degree_;
    active_cfg.eval_3D_parity_mode = tcfg_.eval_3D && tcfg_.parity_mode;

    // 6+7. Rasterize backward + preprocess backward chained into one CB.
    //      rasterize_bwd writes dL_d* to GPU; preprocess_bwd reads them directly.
    //      Eliminates rgrad CPU round-trip + merges 2 submit+wait into 1.
    GradientOutput grads{};
    {
        VkCommandBuffer bwd_cmd = ctx_.allocatePrimary();
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(bwd_cmd, &bi);

        rasterizer_bwd_.backward_record_into(bwd_cmd, pre, bin, N_, cam, active_cfg,
                                              cache, dL_dpixels_.data(), image_.data());
        insert_compute_barrier(bwd_cmd);
        preprocessor_bwd_.backward_record_into(bwd_cmd, g_, N_, cam, active_cfg, cache,
            rasterizer_bwd_.dL_dconics_buf(),
            rasterizer_bwd_.dL_dopacity_buf(),
            rasterizer_bwd_.dL_dcolors_buf(),
            rasterizer_bwd_.dL_dmeans2D_buf(),
            raw_view_,
            rasterizer_bwd_.dL_dgauss2screen_buf());

        vkEndCommandBuffer(bwd_cmd);
        ctx_.submitAndWait(bwd_cmd);
        ctx_.freePrimary(bwd_cmd);
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
    preprocessor_bwd_.download_grads(N_, max_coeffs_, grads, alloc_);

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
    // Evidence: AAA-Gaussians/train.py loss lines + spec/README.md §7.
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

    // Gradient capture — optional, for testing. Copies grads after regularization
    // corrections, before GPU Adam upload. Matches Python autograd capture point.
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

    // 8. Upload CPU gradients to gradient GPU buffers.
    //    SH: DC (first N*3 floats) and rest (remaining N*(max_coeffs-1)*3 floats)
    //    are uploaded to separate buffers to match the separate raw param GPU bufs.
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
        return last_loss_;
    }

    // GPU Adam — chain all 6 groups into one CB (6 submit+wait → 1).
    {
        VkCommandBuffer adam_cmd = ctx_.allocatePrimary();
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(adam_cmd, &bi);

        vulkan_adam_.step_group_record(adam_cmd, 0,
            raw_param_gpu_bufs_[0]->handle(), grad_positions_gpu_->handle(), pos_lr, s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 1,
            raw_param_gpu_bufs_[1]->handle(), grad_sh_dc_gpu_->handle(), group_lrs_[1], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 2,
            raw_param_gpu_bufs_[2]->handle(), grad_sh_rest_gpu_->handle(), group_lrs_[2], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 3,
            raw_param_gpu_bufs_[3]->handle(), grad_opacities_gpu_->handle(), group_lrs_[3], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 4,
            raw_param_gpu_bufs_[4]->handle(), grad_scales_gpu_->handle(), group_lrs_[4], s);
        insert_compute_barrier(adam_cmd);
        vulkan_adam_.step_group_record(adam_cmd, 5,
            raw_param_gpu_bufs_[5]->handle(), grad_rotations_gpu_->handle(), group_lrs_[5], s);

        vkEndCommandBuffer(adam_cmd);
        ctx_.submitAndWait(adam_cmd);
        ctx_.freePrimary(adam_cmd);
    }

    if (capture_adam_) {
        captured_adam_m_.resize(6);
        captured_adam_v_.resize(6);
        for (int i = 0; i < 6; ++i) {
            vulkan_adam_.download_moments(i, captured_adam_m_[i], captured_adam_v_[i]);
        }
    }

    // 10. Download updated raw params from GPU back to CPU vectors,
    //     so activate_params() on the next step sees the Adam-updated values.
    raw_param_gpu_bufs_[0]->download(raw_positions_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    // SH: download DC [N,3] and REST [N,K-1,3] from their separate GPU buffers,
    // then SCATTER back into the interleaved [N, K, 3] raw_sh_coeffs_ layout.
    // Inverse of the gather done at upload (see sh_gather_dc note above).
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

    // 10b. Position noise injection — SP-6 Gap 2.
    //      Near-dead Gaussians (opacity < ~0.995) receive covariance-scaled N(0,1)
    //      noise proportional to pos_lr. Matches train.py:141-148.
    //      Must run after raw_positions_ and act_* are both on CPU (activate_params()
    //      called at step start, download complete above). act_opacities_ reflect
    //      the *previous* step's values, which is correct — same as Python reference.
    inject_position_noise(pos_lr);

    if (tcfg_.cap_max <= 0) {
        for (int i = 0; i < N_; ++i) {
            const float* dp = grads.d_raw_positions + i * 3;
            const float norm = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
            grad_means2D_accum_[i] += norm;
        }
    }

    const bool densification_enabled = (tcfg_.densify_from_step > 0);
    const bool should_densify =
        densification_enabled &&
        (step_count_ >= tcfg_.densify_from_step) &&
        (step_count_ <= tcfg_.densify_until_step) &&
        (step_count_ % tcfg_.densify_interval == 0);

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

    if (densification_enabled && tcfg_.opacity_reset_interval > 0 &&
        step_count_ > 0 && step_count_ <= tcfg_.densify_until_step &&
        (step_count_ % tcfg_.opacity_reset_interval == 0)) {
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
        }
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
