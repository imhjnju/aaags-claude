#include "trainer.h"
#include "loss.h"

Trainer::Trainer(size_t arena_size)
    : allocator_(arena_size) {}

Trainer::StepResult Trainer::step(RawGaussianParams& params, const Camera& camera,
                                   const float* gt_image, const RenderConfig& cfg,
                                   const TrainConfig& train_cfg, int iteration) {
    allocator_.reset();
    int N = params.count;
    int mc3 = params.max_coeffs * 3;
    int num_pixels = camera.width * camera.height;

    // Resize activation buffers if needed
    if (N != last_count_) {
        act_positions_.resize(N * 3);
        act_scales_.resize(N * 3);
        act_rotations_.resize(N * 4);
        act_sh_.resize(N * mc3);
        act_opacities_.resize(N);
        last_count_ = N;
    }

    // Activate raw -> GaussianData
    GaussianData g;
    g.count = N;
    g.sh_degree = params.sh_degree;
    g.max_coeffs = params.max_coeffs;
    g.positions = act_positions_.data();
    g.scales = act_scales_.data();
    g.rotations = act_rotations_.data();
    g.sh_coeffs = act_sh_.data();
    g.opacities = act_opacities_.data();
    g.filter_3D = nullptr;
    params.activate(g);

    // Forward
    ForwardCache cache{};
    PreprocessOutput pre = preprocessor_.process(g, camera, cfg, allocator_, &cache);
    BinningOutput bin = binner_.bin(pre, N, camera, cfg, allocator_);
    if (bin.total_pairs > 0) sorter_.sort(bin, allocator_);
    cache.pre = &pre;
    cache.bin = &bin;

    float* rendered = allocator_.allocate_array<float>(num_pixels * 3);
    rasterizer_.rasterize(pre, bin, camera, cfg, rendered, nullptr, &cache, &allocator_);
    last_rendered_ = rendered;

    // Loss
    float* d_image = allocator_.allocate_array<float>(num_pixels * 3);
    float loss = l1_loss(rendered, gt_image, camera.height, camera.width, d_image);

    // Backward
    RasterGradOutput rgrad;
    rgrad.allocate_and_zero(allocator_, N);
    rasterizer_bw_.backward(pre, bin, camera, cfg, cache, d_image, rgrad);

    GradientOutput grads;
    grads.allocate_and_zero(allocator_, N, params.max_coeffs);
    preprocessor_bw_.backward(g, camera, cfg, cache, rgrad, params, grads);

    // Optimize
    if (train_cfg.use_adam)
        adam_optimizer_.step(params, grads, train_cfg, iteration);
    else
        sgd_optimizer_.step(params, grads, train_cfg, iteration);

    return {loss, iteration, N};
}

Trainer::StepResult Trainer::step_with_densify(
        OwnedRawParams& owned, const Camera& camera,
        const float* gt_image, const RenderConfig& cfg,
        const TrainConfig& train_cfg, const DensifyConfig& dcfg,
        float scene_extent, int iteration) {

    RawGaussianParams params = owned.as_raw();

    // Normal forward + backward + optimize step
    StepResult result = step(params, camera, gt_image, cfg, train_cfg, iteration);

    // Accumulate 2D position gradient stats (need rgrad from the step)
    // Re-extract d_means2D from the last rasterizer backward output.
    // Since step() used the allocator, we need to access the rgrad before reset.
    // Actually the allocator was reset at start of step(). The rgrad data is gone.
    // We need to re-run backward or save the rgrad. For simplicity, re-compute:

    // For the accumulation, we use the analytical gradient magnitude as proxy.
    // Official 3DGS uses d_means2D from rasterizer backward. We approximate by
    // using the position gradient norm (which includes the projection Jacobian).
    // This is a valid proxy since large position gradients imply large screen movement.
    int N = params.count;
    std::vector<float> d_means2D_approx(N * 2, 0.0f);
    // Use raw position gradients projected to approximate screen gradients
    // For a more accurate implementation, save d_means2D during step().
    // For now, use the norm of 3D position gradient as a proxy.
    // This is intentionally simplified; full implementation should save rgrad.d_means2D.
    // TODO: Save d_means2D from step() for exact accumulation.

    // Actually, let's just re-access the allocator's last rgrad if possible.
    // Better approach: refactor step to expose rgrad. For now, run a minimal version:

    // Simplest correct approach: after step(), rerun just forward to get d_means2D.
    // But that's wasteful. Instead, let's modify the flow to save d_means2D.

    // PRAGMATIC FIX: Use the position gradient magnitude as the accumulation signal.
    // This is what some implementations do (e.g., gaussian-splatting-cuda).
    // The position gradient magnitude correlates strongly with the 2D gradient norm.
    float* d_pos = nullptr;
    // The grads were written to params during step(), but the gradient output
    // was on the arena allocator which was reset. We need to save it.
    // For now, use a heuristic based on the loss value.

    // CORRECT APPROACH: Save d_means2D in the step function.
    // Let's refactor step() to optionally return rgrad data.

    // For THIS implementation: re-run the forward+backward with a fresh allocator
    // allocation just to get d_means2D. This is correct but doubles compute cost
    // when densification is active.
    {
        allocator_.reset();
        int mc3 = params.max_coeffs * 3;
        int num_pixels = camera.width * camera.height;

        GaussianData g;
        g.count = N; g.sh_degree = params.sh_degree; g.max_coeffs = params.max_coeffs;
        g.positions = act_positions_.data(); g.scales = act_scales_.data();
        g.rotations = act_rotations_.data(); g.sh_coeffs = act_sh_.data();
        g.opacities = act_opacities_.data(); g.filter_3D = nullptr;
        params.activate(g);

        ForwardCache cache{};
        PreprocessOutput pre = preprocessor_.process(g, camera, cfg, allocator_, &cache);
        BinningOutput bin = binner_.bin(pre, N, camera, cfg, allocator_);
        if (bin.total_pairs > 0) sorter_.sort(bin, allocator_);
        cache.pre = &pre; cache.bin = &bin;

        float* rendered2 = allocator_.allocate_array<float>(num_pixels * 3);
        rasterizer_.rasterize(pre, bin, camera, cfg, rendered2, nullptr, &cache, &allocator_);

        float* d_image2 = allocator_.allocate_array<float>(num_pixels * 3);
        l1_loss(rendered2, gt_image, camera.height, camera.width, d_image2);

        RasterGradOutput rgrad;
        rgrad.allocate_and_zero(allocator_, N);
        rasterizer_bw_.backward(pre, bin, camera, cfg, cache, d_image2, rgrad);

        density_ctrl_.accumulate_stats(rgrad.d_means2D, pre.radii, N);
    }

    // Densification (conditional on iteration range and interval)
    if (dcfg.enabled &&
        iteration >= dcfg.densify_from_iter &&
        iteration < dcfg.densify_until_iter &&
        iteration % dcfg.densify_interval == 0) {

        int old_N = owned.count();
        int new_N = density_ctrl_.densify_and_prune(owned, dcfg, scene_extent);

        if (new_N != old_N) {
            // Update params pointer and reset optimizer
            params = owned.as_raw();
            adam_optimizer_.reset(new_N, params.max_coeffs);
        }

        result.n_gaussians = new_N;
    }

    // Opacity reset
    if (dcfg.enabled &&
        dcfg.opacity_reset_interval > 0 &&
        iteration > 0 &&
        iteration % dcfg.opacity_reset_interval == 0) {
        DensityController::reset_opacity(owned);
    }

    // Sync back owned → params pointers (in case densification changed them)
    params = owned.as_raw();

    return result;
}
