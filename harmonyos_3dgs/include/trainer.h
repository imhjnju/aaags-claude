#pragma once
#include "types.h"
#include "train_types.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include "cpu/preprocessor_backward_cpu.h"
#include "optimizer.h"
#include "density_controller.h"
#include <vector>

class Trainer {
public:
    explicit Trainer(size_t arena_size = 2UL * 1024 * 1024 * 1024);

    struct StepResult { float loss; int iteration; int n_gaussians; };

    // Standard step (non-owning params, no densification)
    StepResult step(RawGaussianParams& params, const Camera& camera,
                    const float* gt_image, const RenderConfig& cfg,
                    const TrainConfig& train_cfg, int iteration);

    // Step with densification (owning params, may change N)
    StepResult step_with_densify(OwnedRawParams& params, const Camera& camera,
                                  const float* gt_image, const RenderConfig& cfg,
                                  const TrainConfig& train_cfg,
                                  const DensifyConfig& densify_cfg,
                                  float scene_extent, int iteration);

    const float* last_rendered() const { return last_rendered_; }

private:
    FrameAllocator allocator_;
    PreprocessorCPU preprocessor_;
    TileBinnerCPU binner_;
    SorterCPU sorter_;
    RasterizerCPU rasterizer_;
    RasterizerBackwardCPU rasterizer_bw_;
    PreprocessorBackwardCPU preprocessor_bw_;
    AdamOptimizer adam_optimizer_;
    SGDOptimizer sgd_optimizer_;
    DensityController density_ctrl_;

    std::vector<float> act_positions_, act_scales_, act_sh_, act_opacities_;
    std::vector<float> act_rotations_;
    int last_count_ = 0;
    const float* last_rendered_ = nullptr;
};
