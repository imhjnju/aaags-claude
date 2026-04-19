#pragma once

#include "types.h"
#include "train_types.h"
#include "cpu_adam.h"
#include "vulkan/vk_context.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/rasterizer_backward_vulkan.h"
#include "vulkan/preprocessor_backward_vulkan.h"

#include <vector>
#include <memory>

// VulkanTrainer wires the full forward→backward→Adam pipeline for one
// training step. The optimizer is CpuAdam (temporary CPU-side; a Vulkan
// compute Adam will replace it in a future sprint). The optimizer call site
// is isolated to step() and parameterized through the Adam groups stored in
// the constructor, so swapping to a GPU Adam requires changes only at those
// sites.
class VulkanTrainer {
public:
    // init_g: initial Gaussian parameters (activated values)
    // init_raw: initial raw (pre-activation) parameters — must be same count as init_g
    VulkanTrainer(VulkanContext& ctx,
                  const GaussianData& init_g,
                  const RawGaussianParams& init_raw,
                  int sh_degree,
                  int cam_width,
                  int cam_height);

    // One training step. Returns L1 loss value.
    float step(const Camera& cam,
               const RenderConfig& cfg,
               const float* target_image,   // [H*W*3] pixel-major
               int target_W,
               int target_H);

    int step_count() const { return adam_.step_count(); }
    float last_loss() const { return last_loss_; }

    // Access current raw parameters (for inspection/checkpointing).
    const RawGaussianParams& raw_params() const { return raw_view_; }

private:
    void activate_params();   // raw_ → g_ (exp/sigmoid/normalize)

    VulkanContext&             ctx_;
    int                        N_;
    int                        max_coeffs_;

    // Raw parameters — Adam updates these in-place.
    // Stored as vectors so they own the memory.
    std::vector<float> raw_positions_;    // [N*3]
    std::vector<float> raw_scales_;       // [N*3]
    std::vector<float> raw_rotations_;   // [N*4]
    std::vector<float> raw_sh_coeffs_;   // [N*max_coeffs*3]
    std::vector<float> raw_opacities_;   // [N]

    // Non-owning view into the above vectors.
    RawGaussianParams raw_view_;

    // Adam moment storage (one pair of m/v vectors per parameter group).
    std::vector<std::vector<float>> m_storage_;  // 6 groups
    std::vector<std::vector<float>> v_storage_;  // 6 groups

    // Activated parameters (rebuilt each step from raw_).
    std::vector<float> act_positions_;   // [N*3]   (= raw, no activation)
    std::vector<float> act_scales_;      // [N*3]   exp(raw_scales)
    std::vector<float> act_rotations_;  // [N*4]   normalized quaternion
    std::vector<float> act_opacities_;  // [N]     sigmoid(raw_opacities)
    std::vector<float> act_sh_coeffs_;  // [N*max_coeffs*3] (= raw, no activation)
    GaussianData       g_;               // non-owning view into act_* buffers

    // Per-frame output buffers.
    std::vector<float> image_;           // [H*W*3]
    std::vector<float> dL_dpixels_;     // [H*W*3]

    FrameAllocator      alloc_;
    CpuAdam             adam_;
    float               last_loss_ = 0.0f;

    // Vulkan pipeline components.
    PreprocessorVulkan        preprocessor_;
    TileBinnerVulkan          binner_;
    SorterVulkan              sorter_;
    RasterizerVulkan          rasterizer_;
    RasterizerBackwardVulkan  rasterizer_bwd_;
    PreprocessorBackwardVulkan preprocessor_bwd_;
};
