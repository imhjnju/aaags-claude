// trainer_gpu.h -- GPU training step using OpenCL backward passes.
// Same structure as the CPU Trainer but uses GPU implementations for
// forward preprocessing, rasterization, and backward passes.

#pragma once

#ifdef ENABLE_OPENCL

#include "types.h"
#include "train_types.h"
#include "gpu/opencl_context.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/sorter_gpu.h"
#include "gpu/rasterizer_gpu.h"
#include "gpu/rasterizer_backward_gpu.h"
#include "gpu/preprocessor_backward_gpu.h"
#include "optimizer.h"
#include <vector>

class TrainerGPU {
public:
    explicit TrainerGPU(OpenCLContext& ctx, size_t arena_size = 2UL * 1024 * 1024 * 1024);

    struct StepResult { float loss; int iteration; };

    StepResult step(RawGaussianParams& params, const Camera& camera,
                    const float* gt_image, const RenderConfig& cfg,
                    const TrainConfig& train_cfg, int iteration);

    const float* last_rendered() const { return last_rendered_; }

private:
    OpenCLContext& ctx_;
    FrameAllocator allocator_;

    // GPU pipeline stages
    PreprocessorGPU preprocessor_;
    TileBinnerGPU binner_;
    SorterGPU sorter_;
    RasterizerGPU rasterizer_;
    RasterizerBackwardGPU rasterizer_bw_;
    PreprocessorBackwardGPU preprocessor_bw_;
    SGDOptimizer optimizer_;

    // Activation buffers
    std::vector<float> act_positions_, act_scales_, act_sh_, act_opacities_;
    std::vector<float> act_rotations_;
    int last_count_ = 0;
    const float* last_rendered_ = nullptr;
};

#endif // ENABLE_OPENCL
