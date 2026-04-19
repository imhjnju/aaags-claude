#include "vulkan_trainer.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

VulkanTrainer::VulkanTrainer(VulkanContext& ctx,
                             const GaussianData& init_g,
                             const RawGaussianParams& init_raw,
                             int sh_degree,
                             int cam_width,
                             int cam_height)
    : ctx_(ctx)
    , N_(init_g.count)
    , max_coeffs_(init_g.max_coeffs)
    , alloc_(64u * 1024u * 1024u)
    , adam_(0.9f, 0.999f, 1e-15f)
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

    // 5. Set up Adam with 6 parameter groups and persistent m/v storage.
    //    lr values match Python reference training.py defaults.
    //
    //    Group 0: positions          lr=1.6e-4,  n=N*3
    //    Group 1: sh_coeffs DC       lr=2.5e-3,  n=N*3
    //    Group 2: sh_coeffs rest     lr=1.25e-4, n=N*(max_coeffs-1)*3
    //    Group 3: opacities          lr=0.05,    n=N
    //    Group 4: scales             lr=0.005,   n=N*3
    //    Group 5: rotations          lr=0.001,   n=N*4

    const int rest_coeffs = max_coeffs_ - 1;

    struct GroupSpec { int n; float lr; };
    const GroupSpec specs[6] = {
        { N_ * 3,                     1.6e-4f  },  // 0: positions
        { N_ * 3,                     2.5e-3f  },  // 1: sh DC
        { N_ * rest_coeffs * 3,       1.25e-4f },  // 2: sh rest
        { N_,                         0.05f    },  // 3: opacities
        { N_ * 3,                     0.005f   },  // 4: scales
        { N_ * 4,                     0.001f   },  // 5: rotations
    };

    // Param pointers for each group (grad is nullptr — filled each step by set_grad).
    float* param_ptrs[6] = {
        raw_positions_.data(),
        raw_sh_coeffs_.data(),                          // DC: first N*3 floats
        raw_sh_coeffs_.data() + static_cast<size_t>(N_) * 3,  // rest: remaining
        raw_opacities_.data(),
        raw_scales_.data(),
        raw_rotations_.data(),
    };

    m_storage_.resize(6);
    v_storage_.resize(6);

    for (int i = 0; i < 6; ++i) {
        m_storage_[i].assign(static_cast<size_t>(specs[i].n), 0.0f);
        v_storage_[i].assign(static_cast<size_t>(specs[i].n), 0.0f);

        AdamGroup ag{};
        ag.params = param_ptrs[i];
        ag.grad   = nullptr;   // will be set each step
        ag.m      = m_storage_[i].data();
        ag.v      = v_storage_[i].data();
        ag.n      = specs[i].n;
        ag.lr     = specs[i].lr;
        adam_.add_group(ag);
    }

    // 6. Allocate per-frame output buffers (H*W*3).
    const size_t HW3 = static_cast<size_t>(cam_height) * cam_width * 3;
    image_.resize(HW3, 0.0f);
    dL_dpixels_.resize(HW3, 0.0f);
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

    // 1. Forward preprocess (populates cache: cov3D, p_view, p_hom_w, cov2D, cov2D_det)
    PreprocessOutput pre = preprocessor_.process(g_, cam, cfg, alloc_, &cache);
    preprocessor_.download_cache(N_, cache, alloc_);
    cache.pre = &pre;

    // 2. Tile binning
    BinningOutput bin = binner_.bin(pre, N_, cam, cfg, alloc_);
    cache.bin = &bin;

    // 3. Sort
    sorter_.sort(bin, alloc_);

    // 4. Rasterize (populates cache.T_final and cache.n_contrib since they are non-null)
    rasterizer_.rasterize(pre, bin, cam, cfg, image_.data(),
                          /*depth=*/nullptr, &cache, &alloc_);

    // 5. L1 loss + gradient: dL/dpixel = sign(rendered - target) / (W*H*3)
    const int HW3 = H * W * 3;
    float loss = 0.0f;
    const float scale = 1.0f / static_cast<float>(HW3);
    for (int k = 0; k < HW3; ++k) {
        float diff = image_[k] - target[k];
        loss += std::fabs(diff);
        // sign: +1 if diff > 0, -1 if diff < 0, 0 if equal
        dL_dpixels_[k] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) * scale;
    }
    last_loss_ = loss * scale;

    // 6. Rasterizer backward
    RasterGradOutput rgrad{};
    rasterizer_bwd_.backward(pre, bin, N_, cam, cfg, cache,
                             dL_dpixels_.data(), rgrad, alloc_);

    // 7. Preprocessor backward
    GradientOutput grads{};
    preprocessor_bwd_.backward(g_, N_, cam, cfg, cache, rgrad,
                               raw_view_, grads, alloc_);

    // 8. Adam step — update raw parameters using fresh gradients.
    //    re-wire grad pointers each step (grads arrays are arena-allocated,
    //    so pointers change each call).
    adam_.set_grad(0, grads.d_raw_positions);
    adam_.set_grad(1, grads.d_raw_sh_coeffs);                                          // DC (first N*3)
    adam_.set_grad(2, grads.d_raw_sh_coeffs + static_cast<size_t>(N_) * 3);           // rest
    adam_.set_grad(3, grads.d_raw_opacities);
    adam_.set_grad(4, grads.d_raw_scales);
    adam_.set_grad(5, grads.d_raw_rotations);
    adam_.step();

    return last_loss_;
}
