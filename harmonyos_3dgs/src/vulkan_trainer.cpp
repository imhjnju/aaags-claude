#include "vulkan_trainer.h"
#include "dssim.h"
#include "train_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>

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
    , active_sh_degree_(0)
    , preprocessor_(ctx)
    , binner_(ctx)
    , sorter_(ctx)
    , rasterizer_(ctx)
    , rasterizer_bwd_(ctx)
    , preprocessor_bwd_(ctx)
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

    // 4. Set up GaussianData non-owning view.
    g_.count      = N_;
    g_.sh_degree  = sh_degree;
    g_.max_coeffs = max_coeffs_;
    g_.positions  = act_positions_.data();
    g_.scales     = act_scales_.data();
    g_.rotations  = act_rotations_.data();
    g_.opacities  = act_opacities_.data();
    g_.sh_coeffs  = act_sh_coeffs_.data();
    g_.filter_3D  = nullptr;   // not used in training path

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
    raw_param_gpu_bufs_[1]->upload(raw_sh_coeffs_.data(), sz_sh_dc);

    raw_param_gpu_bufs_[2] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    raw_param_gpu_bufs_[2]->upload(
        raw_sh_coeffs_.data() + static_cast<size_t>(N_) * 3, sz_sh_rest);

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
// step
// ---------------------------------------------------------------------------

float VulkanTrainer::step(const Camera& cam,
                          const RenderConfig& cfg,
                          const float* target,
                          int W,
                          int H)
{
    alloc_.reset();
    activate_params();

    // Pre-allocate ForwardCache fields that the backward passes need.
    // T_final and n_contrib are written by RasterizerVulkan::rasterize() only
    // if the pointers are non-null, so we must allocate them here.
    // cov3D, p_view, p_hom_w, cov2D, cov2D_det are allocated inside
    // PreprocessorVulkan::download_cache().
    const int HW_pixels = H * W;
    ForwardCache cache{};
    cache.T_final  = alloc_.allocate_array<float>(static_cast<size_t>(HW_pixels));
    cache.n_contrib = alloc_.allocate_array<int>  (static_cast<size_t>(HW_pixels));
    std::memset(cache.T_final,  0, static_cast<size_t>(HW_pixels) * sizeof(float));
    std::memset(cache.n_contrib, 0, static_cast<size_t>(HW_pixels) * sizeof(int));

    // Build active config: override sh_degree with the scheduled value.
    RenderConfig active_cfg = cfg;
    active_cfg.sh_degree = active_sh_degree_;

    // 1. Forward preprocess (populates cache: cov3D, p_view, p_hom_w, cov2D, cov2D_det)
    PreprocessOutput pre = preprocessor_.process(g_, cam, active_cfg, alloc_, &cache);
    preprocessor_.download_cache(N_, cache, alloc_);
    cache.pre = &pre;

    // 2. Tile binning
    BinningOutput bin = binner_.bin(pre, N_, cam, active_cfg, alloc_);
    cache.bin = &bin;

    // 3. Sort
    sorter_.sort(bin, alloc_);

    // 4. Rasterize (populates cache.T_final and cache.n_contrib since they are non-null)
    rasterizer_.rasterize(pre, bin, cam, active_cfg, image_.data(),
                          /*depth=*/nullptr, &cache, &alloc_);

    // 5. Combined L1 + DSSIM loss + gradient.
    //    lambda_dssim=0 disables the O(W*H*WINDOW^2) SSIM computation (use for large images).
    last_loss_ = compute_combined_loss_gradient(
        image_.data(), target, dL_dpixels_.data(), W, H, tcfg_.lambda_dssim);

    // 6. Rasterizer backward
    RasterGradOutput rgrad{};
    rasterizer_bwd_.backward(pre, bin, N_, cam, active_cfg, cache,
                             dL_dpixels_.data(), rgrad, alloc_);

    // 7. Preprocessor backward
    GradientOutput grads{};
    preprocessor_bwd_.backward(g_, N_, cam, active_cfg, cache, rgrad,
                               raw_view_, grads, alloc_);

    // SP-6 T2: Regularization gradient injection.
    // After preprocessor_bwd_ writes raw-param gradients, add L1 regularization
    // gradient corrections before the GPU upload.
    //
    // Opacity reg: loss += opacity_reg * mean(|sigmoid(raw_op)|)
    //   => dL/d_raw_op[i] += (opacity_reg / N) * sig * (1 - sig)
    //      where sig = sigmoid(raw_op[i]) = act_opacities_[i]
    //
    // Scale reg: loss += scale_reg * mean(|exp(raw_sc)|)  (per-component)
    //   => dL/d_raw_sc[i,k] += (scale_reg / N) * exp(raw_sc[i,k])
    //      where exp(raw_sc[i,k]) = act_scales_[i*3+k]
    //
    // Evidence: test RegularizationGrad.ScaleGradAtRawLogScale asserts per-component
    // formula (no norm division), test RegularizationGrad.OpacityGradAtRawZero asserts
    // plain sigmoid derivative (not op_sigmoid which is for noise injection).
    if (N_ > 0 && (tcfg_.opacity_reg > 0.f || tcfg_.scale_reg > 0.f)) {
        const float inv_N = 1.0f / static_cast<float>(N_);
        for (int i = 0; i < N_; ++i) {
            if (tcfg_.opacity_reg > 0.f) {
                const float sig = act_opacities_[static_cast<size_t>(i)];
                grads.d_raw_opacities[i] += (tcfg_.opacity_reg * inv_N) * sig * (1.f - sig);
            }
            if (tcfg_.scale_reg > 0.f) {
                const float coeff = tcfg_.scale_reg * inv_N;
                for (int k = 0; k < 3; ++k) {
                    const size_t idx = static_cast<size_t>(i) * 3 + k;
                    grads.d_raw_scales[idx] += coeff * act_scales_[idx];
                }
            }
        }
    }

    // 8. Upload CPU gradients to gradient GPU buffers.
    //    SH: DC (first N*3 floats) and rest (remaining N*(max_coeffs-1)*3 floats)
    //    are uploaded to separate buffers to match the separate raw param GPU bufs.
    grad_positions_gpu_->upload(grads.d_raw_positions,
                                static_cast<size_t>(N_) * 3 * sizeof(float));
    grad_sh_dc_gpu_->upload(grads.d_raw_sh_coeffs,
                            static_cast<size_t>(N_) * 3 * sizeof(float));
    if (max_coeffs_ > 1) {
        grad_sh_rest_gpu_->upload(grads.d_raw_sh_coeffs + static_cast<size_t>(N_) * 3,
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

    float pos_lr = lr_schedule(tcfg_.pos_lr_init, tcfg_.pos_lr_final,
                               step_count_, tcfg_.max_steps);

    if (tcfg_.sh_degree_warmup > 0)
        active_sh_degree_ = std::min(step_count_ / tcfg_.sh_degree_warmup,
                                     tcfg_.sh_degree_max);
    else
        active_sh_degree_ = tcfg_.sh_degree_max;

    vulkan_adam_.step_group(0, raw_param_gpu_bufs_[0]->handle(),
                            grad_positions_gpu_->handle(), pos_lr, s);
    vulkan_adam_.step_group(1, raw_param_gpu_bufs_[1]->handle(),
                            grad_sh_dc_gpu_->handle(),    group_lrs_[1], s);
    vulkan_adam_.step_group(2, raw_param_gpu_bufs_[2]->handle(),
                            grad_sh_rest_gpu_->handle(),  group_lrs_[2], s);
    vulkan_adam_.step_group(3, raw_param_gpu_bufs_[3]->handle(),
                            grad_opacities_gpu_->handle(), group_lrs_[3], s);
    vulkan_adam_.step_group(4, raw_param_gpu_bufs_[4]->handle(),
                            grad_scales_gpu_->handle(),   group_lrs_[4], s);
    vulkan_adam_.step_group(5, raw_param_gpu_bufs_[5]->handle(),
                            grad_rotations_gpu_->handle(), group_lrs_[5], s);

    // 10. Download updated raw params from GPU back to CPU vectors,
    //     so activate_params() on the next step sees the Adam-updated values.
    raw_param_gpu_bufs_[0]->download(raw_positions_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    // SH: download DC block to [0..N*3) and rest block to [N*3..end) of raw_sh_coeffs_.
    raw_param_gpu_bufs_[1]->download(raw_sh_coeffs_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    if (max_coeffs_ > 1) {
        raw_param_gpu_bufs_[2]->download(raw_sh_coeffs_.data() + static_cast<size_t>(N_) * 3,
                                         static_cast<size_t>(N_) * (max_coeffs_ - 1) * 3 * sizeof(float));
    }
    raw_param_gpu_bufs_[3]->download(raw_opacities_.data(),
                                     static_cast<size_t>(N_) * sizeof(float));
    raw_param_gpu_bufs_[4]->download(raw_scales_.data(),
                                     static_cast<size_t>(N_) * 3 * sizeof(float));
    raw_param_gpu_bufs_[5]->download(raw_rotations_.data(),
                                     static_cast<size_t>(N_) * 4 * sizeof(float));

    // 11. Accumulate per-Gaussian gradient norm from raw position gradients
    //     (proxy for |dL/dmeans2D|).  grads.d_raw_positions is valid on CPU
    //     because preprocessor_bwd_ writes it there.
    for (int i = 0; i < N_; ++i) {
        const float* dp = grads.d_raw_positions + i * 3;
        const float norm = std::sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
        grad_means2D_accum_[i] += norm;
    }

    // 12. Densification step — triggered at configured intervals.
    const bool should_densify =
        (tcfg_.densify_from_step > 0) &&
        (step_count_ >= tcfg_.densify_from_step) &&
        (step_count_ <= tcfg_.densify_until_step) &&
        (step_count_ % tcfg_.densify_interval == 0);

    if (should_densify) {
        // Compute scene_extent: max axis range of position values.
        float pos_min[3] = { raw_positions_[0], raw_positions_[1], raw_positions_[2] };
        float pos_max[3] = { raw_positions_[0], raw_positions_[1], raw_positions_[2] };
        for (int i = 1; i < N_; ++i) {
            for (int k = 0; k < 3; ++k) {
                const float v = raw_positions_[static_cast<size_t>(i) * 3 + k];
                if (v < pos_min[k]) pos_min[k] = v;
                if (v > pos_max[k]) pos_max[k] = v;
            }
        }
        float scene_extent = 0.0f;
        for (int k = 0; k < 3; ++k) {
            scene_extent = std::max(scene_extent, pos_max[k] - pos_min[k]);
        }
        if (scene_extent < 1e-6f) scene_extent = 1.0f;

        // Pack current raw params into OwnedRawParams for densify_and_prune.
        OwnedRawParams raw_owned;
        raw_owned.sh_degree  = raw_view_.sh_degree;
        raw_owned.max_coeffs = max_coeffs_;
        raw_owned.from_raw(raw_view_);

        const int new_N = densify_and_prune(
            raw_owned, grad_means2D_accum_.data(), step_count_, tcfg_, scene_extent);

        // Update N_ and re-allocate GPU buffers / Adam groups.
        N_ = new_N;
        raw_positions_.assign(raw_owned.positions.begin(), raw_owned.positions.end());
        raw_scales_.assign   (raw_owned.scales.begin(),    raw_owned.scales.end());
        raw_rotations_.assign(raw_owned.rotations.begin(), raw_owned.rotations.end());
        raw_sh_coeffs_.assign(raw_owned.sh_coeffs.begin(), raw_owned.sh_coeffs.end());
        raw_opacities_.assign(raw_owned.opacities.begin(), raw_owned.opacities.end());

        reallocate_for_n(N_);

        // Reset accumulated gradient norms.
        grad_means2D_accum_.assign(static_cast<size_t>(N_), 0.0f);
    }

    return last_loss_;
}

// ---------------------------------------------------------------------------
// reallocate_for_n — re-create GPU buffers and Adam groups after N changes
// ---------------------------------------------------------------------------

void VulkanTrainer::reallocate_for_n(int new_N)
{
    N_ = new_N;
    const int rest_coeffs = max_coeffs_ - 1;

    // Re-size activated-value vectors.
    act_positions_.resize(static_cast<size_t>(N_) * 3);
    act_scales_.resize   (static_cast<size_t>(N_) * 3);
    act_rotations_.resize(static_cast<size_t>(N_) * 4);
    act_opacities_.resize(static_cast<size_t>(N_));
    act_sh_coeffs_.resize(static_cast<size_t>(N_) * max_coeffs_ * 3);

    // Update GaussianData view pointers.
    g_.count      = N_;
    g_.positions  = act_positions_.data();
    g_.scales     = act_scales_.data();
    g_.rotations  = act_rotations_.data();
    g_.opacities  = act_opacities_.data();
    g_.sh_coeffs  = act_sh_coeffs_.data();

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
    if (N_ > 0)
        raw_param_gpu_bufs_[1]->upload(raw_sh_coeffs_.data(),
                                       static_cast<size_t>(N_) * 3 * sizeof(float));

    raw_param_gpu_bufs_[2] = std::make_unique<VulkanBuffer>(
        ctx_, sz_sh_rest, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (N_ > 0 && max_coeffs_ > 1)
        raw_param_gpu_bufs_[2]->upload(
            raw_sh_coeffs_.data() + static_cast<size_t>(N_) * 3,
            static_cast<size_t>(N_) * rest_coeffs * 3 * sizeof(float));

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

    // Re-initialize VulkanAdam groups (reset clears existing m/v state).
    vulkan_adam_.reset_groups();

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
}
