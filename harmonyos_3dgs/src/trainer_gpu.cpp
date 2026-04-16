// trainer_gpu.cpp -- GPU training step implementation.
// Uses CPU for forward pass (to fill ForwardCache) and GPU for backward passes
// where the compute is most intensive.

#ifdef ENABLE_OPENCL

#include "trainer_gpu.h"
#include "loss.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"

TrainerGPU::TrainerGPU(OpenCLContext& ctx, size_t arena_size)
    : ctx_(ctx), allocator_(arena_size),
      preprocessor_(ctx), binner_(ctx), sorter_(ctx),
      rasterizer_(ctx), rasterizer_bw_(ctx), preprocessor_bw_(ctx) {}

TrainerGPU::StepResult TrainerGPU::step(RawGaussianParams& params, const Camera& camera,
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

    // Forward pass on CPU (fills ForwardCache which backward needs)
    PreprocessorCPU cpu_preprocessor;
    TileBinnerCPU cpu_binner;
    SorterCPU cpu_sorter;
    RasterizerCPU cpu_rasterizer;

    ForwardCache cache{};
    PreprocessOutput pre = cpu_preprocessor.process(g, camera, cfg, allocator_, &cache);
    BinningOutput bin = cpu_binner.bin(pre, N, camera, cfg, allocator_);
    if (bin.total_pairs > 0) cpu_sorter.sort(bin, allocator_);
    cache.pre = &pre;
    cache.bin = &bin;

    float* rendered = allocator_.allocate_array<float>(num_pixels * 3);
    cpu_rasterizer.rasterize(pre, bin, camera, cfg, rendered, nullptr, &cache, &allocator_);
    last_rendered_ = rendered;

    // Loss
    float* d_image = allocator_.allocate_array<float>(num_pixels * 3);
    float loss = l1_loss(rendered, gt_image, camera.height, camera.width, d_image);

    // GPU Backward: rasterizer
    RasterGradOutput rgrad;
    rgrad.allocate_and_zero(allocator_, N);

    // The GPU backward needs device_data pointers, but since we ran CPU forward,
    // we don't have them. So we use the CPU backward for the rasterizer too,
    // or upload the data. For correctness and simplicity, upload CPU data to GPU
    // by creating temporary device buffers.
    //
    // Actually, the GPU rasterizer backward reads means2D, conics, rgb, opacities_2d
    // from device buffers (pre.device_data) and values_sorted, tile_ranges from
    // (bin.device_data). Since we ran CPU forward, device_data is null.
    //
    // For the GPU backward to work with CPU forward data, we need to upload them.
    // We'll create temporary PreprocessDeviceBuffers and BinningDeviceBuffers.

    // Create temporary device buffers for the rasterizer backward
    PreprocessDeviceBuffers temp_pre_dev{};
    temp_pre_dev.means2D      = ctx_.createBuffer(N * 2 * sizeof(float), CL_MEM_READ_ONLY);
    temp_pre_dev.conics       = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    temp_pre_dev.rgb          = ctx_.createBuffer(N * 3 * sizeof(float), CL_MEM_READ_ONLY);
    temp_pre_dev.opacities_2d = ctx_.createBuffer(N * sizeof(float),     CL_MEM_READ_ONLY);

    ctx_.writeBuffer(temp_pre_dev.means2D,      pre.means2D,      N * 2 * sizeof(float));
    ctx_.writeBuffer(temp_pre_dev.conics,       pre.conics,       N * 3 * sizeof(float));
    ctx_.writeBuffer(temp_pre_dev.rgb,          pre.rgb,          N * 3 * sizeof(float));
    ctx_.writeBuffer(temp_pre_dev.opacities_2d, pre.opacities_2d, N * sizeof(float));

    BinningDeviceBuffers temp_bin_dev{};
    temp_bin_dev.values_sorted = ctx_.createBuffer(bin.total_pairs * sizeof(uint32_t), CL_MEM_READ_ONLY);
    temp_bin_dev.tile_ranges   = ctx_.createBuffer(bin.num_tiles * 2 * sizeof(uint32_t), CL_MEM_READ_ONLY);

    ctx_.writeBuffer(temp_bin_dev.values_sorted, bin.values_sorted, bin.total_pairs * sizeof(uint32_t));
    ctx_.writeBuffer(temp_bin_dev.tile_ranges,   bin.tile_ranges,   bin.num_tiles * 2 * sizeof(uint32_t));

    // Temporarily set device_data pointers
    PreprocessOutput pre_gpu = pre;
    pre_gpu.device_data = &temp_pre_dev;
    BinningOutput bin_gpu = bin;
    bin_gpu.device_data = &temp_bin_dev;

    rasterizer_bw_.backward(pre_gpu, bin_gpu, camera, cfg, cache, d_image, N, rgrad);

    // GPU Backward: preprocessor
    GradientOutput grads;
    grads.allocate_and_zero(allocator_, N, params.max_coeffs);
    preprocessor_bw_.backward(g, camera, cfg, cache, rgrad, params, grads);

    // Release temporary device buffers
    auto& cl = ctx_.cl();
    cl.clReleaseMemObject(temp_pre_dev.means2D);
    cl.clReleaseMemObject(temp_pre_dev.conics);
    cl.clReleaseMemObject(temp_pre_dev.rgb);
    cl.clReleaseMemObject(temp_pre_dev.opacities_2d);
    cl.clReleaseMemObject(temp_bin_dev.values_sorted);
    cl.clReleaseMemObject(temp_bin_dev.tile_ranges);

    // Optimize (CPU)
    optimizer_.step(params, grads, train_cfg, iteration);

    return {loss, iteration};
}

#endif // ENABLE_OPENCL
