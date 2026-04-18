// preprocessor_vk.h -- Vulkan preprocessor, grown in TDD slices.
//
// Slice coverage so far:
//   2a: world->view + near-plane cull  -> Output{depths, active}
//   2b: + viewproj + NDC + pixel pos   -> Output{..., means2D, p_hom_w}
//   2c: + 3D cov, 2D cov, conic        -> Output{..., conics, opacities_2d}
//
// Output semantics:
//   depths, p_hom_w                : always written (even for culled).
//   active                          : 1 iff Gaussian survives all checks
//                                     currently in the shader.
//   means2D, conics, opacities_2d  : valid iff active[i]==1; else zero.
//
// The process() call takes the standard GaussianData + Camera + RenderConfig
// to match the abstract Preprocessor interface we will eventually conform to.

#pragma once

#include "types.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class PreprocessorVK {
public:
    struct Output {
        std::vector<float>    depths;         // Slice 2a  [N]
        std::vector<uint32_t> active;         // Slice 2a  [N]
        std::vector<float>    means2D;        // Slice 2b  [N * 2]
        std::vector<float>    p_hom_w;        // Slice 2b  [N]
        std::vector<float>    conics;         // Slice 2c  [N * 3]
        std::vector<float>    opacities_2d;   // Slice 2c  [N]
    };

    PreprocessorVK(VulkanContext& ctx, const std::string& shader_dir);
    ~PreprocessorVK();

    PreprocessorVK(const PreprocessorVK&)            = delete;
    PreprocessorVK& operator=(const PreprocessorVK&) = delete;

    /// Run the preprocessor over the Gaussians in `g` using camera `cam` and
    /// scale modifier from `cfg`. Reads: positions, scales, rotations,
    /// opacities. (SH coefficients, filter_3D, eval_3D are handled by later
    /// slices.)
    Output process(const GaussianData& g, const Camera& cam,
                   const RenderConfig& cfg);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
};
