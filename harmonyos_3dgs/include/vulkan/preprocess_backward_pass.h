// preprocess_backward_pass.h — SP-3 T23: thin owner of the
// preprocess_backward.comp compute pipeline.
//
// One workgroup = 256 Gaussians. Dispatch: ceil(N/256) × 1 × 1.
// Each invocation computes gradients for one Gaussian (if active, i.e. radii>0).
//
// Bindings match preprocess_backward_bind:: in backward_bindings.h:
//   0..9   read-only SSBOs  (positions, radii, cov3D, d_conics, d_opacity,
//                             sh_coeffs, scales, rotations, d_rgb, d_means2D)
//   10..13 write-only SSBOs (d_means3D, d_sh, d_scales, d_rotations)
//   14     UBO              (PreprocessBackwardUBO)
//
// The caller is responsible for zeroing the gradient SSBOs before calling
// bind_buffers() / dispatch_sync().
#pragma once

#include "vulkan/vk_context.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/backward_bindings.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

class PreprocessBackwardPass {
public:
    /// All VkBuffers wired to bindings 0..14. Names mirror preprocess_backward_bind::.
    struct Buffers {
        VkBuffer positions;   // RO float[N*3]
        VkBuffer radii;       // RO int[N]
        VkBuffer cov3D;       // RO float[N*6]
        VkBuffer d_conics;    // RO float[N*3]
        VkBuffer d_opacity;   // RO float[N]
        VkBuffer sh_coeffs;   // RO float[N*max_coeffs*3]
        VkBuffer scales;      // RO float[N*3]
        VkBuffer rotations;   // RO float[N*4]
        VkBuffer d_rgb;       // RO float[N*3]
        VkBuffer d_means2D;   // RO float[N*2]
        VkBuffer d_means3D;   // RW float[N*3]  zero-filled by caller
        VkBuffer d_sh;        // RW float[N*max_coeffs*3]  zero-filled by caller
        VkBuffer d_scales;    // RW float[N*3]  zero-filled by caller
        VkBuffer d_rotations; // RW float[N*4]  zero-filled by caller
    };

    explicit PreprocessBackwardPass(VulkanContext& ctx);
    ~PreprocessBackwardPass();  // out-of-line: unique_ptr to incomplete type

    PreprocessBackwardPass(const PreprocessBackwardPass&)            = delete;
    PreprocessBackwardPass& operator=(const PreprocessBackwardPass&) = delete;

    /// Wire the descriptor set to these buffers. Safe to call once per scene.
    /// ubo must point to an uploaded PreprocessBackwardUBO (192 bytes, std140).
    void bind_buffers(const Buffers& b, VkBuffer ubo);

    /// Synchronous dispatch: ceil(num_gaussians/256) × 1 × 1 workgroups.
    /// bind_buffers() must have been called first.
    void dispatch_sync(uint32_t num_gaussians);

private:
    VulkanContext& ctx_;
    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
};
