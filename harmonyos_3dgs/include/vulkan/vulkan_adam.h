#pragma once
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

// AdamStepUBO matches the shader's std140 layout (32 bytes).
// std140 rules: float = 4B, uint = 4B, vec4-alignment at struct start.
// Layout: beta1(4) + beta2(4) + eps(4) + lr(4) = 16B,
//         n(4) + step(4) + _pad0(4) + _pad1(4) = 16B. Total = 32B.
struct alignas(16) AdamStepUBO {
    float    beta1  = 0.9f;
    float    beta2  = 0.999f;
    float    eps    = 1e-15f;  // matches Python reference training.py (intentional — not 1e-8)
    float    lr     = 1e-3f;
    uint32_t n      = 0;
    uint32_t step   = 1;
    float    _pad0  = 0.f;
    float    _pad1  = 0.f;
};
static_assert(sizeof(AdamStepUBO) == 32, "AdamStepUBO must be 32 bytes");

// VulkanAdam: GPU Adam optimizer for one or more parameter groups.
// Each group has its own m/v state GPU buffers (allocated once at add_group()).
// step_group() dispatches adam_step.comp with the provided params/grad buffers.
// Thread-safety: step_group() is NOT thread-safe (single shared descriptor set,
// no mutex). Callers must serialize all step_group() calls externally.
class VulkanAdam {
public:
    VulkanAdam(VulkanContext& ctx,
               float beta1 = 0.9f,
               float beta2 = 0.999f,
               float eps   = 1e-15f);
    ~VulkanAdam() = default;
    VulkanAdam(const VulkanAdam&)            = delete;
    VulkanAdam& operator=(const VulkanAdam&) = delete;

    // Add a parameter group. Returns its index (0-based).
    // Allocates m/v GPU buffers of `n` floats, zero-initialized.
    int add_group(uint32_t n, float lr);

    // Remove all groups (frees m/v GPU buffers). Call before re-adding groups
    // after Gaussian count changes (e.g. after densification).
    void reset_groups();

    // Dispatch one Adam step for group `idx`.
    // params_buf: GPU buffer of grp.n floats — updated in-place.
    // grad_buf:   GPU buffer of grp.n floats — gradient (read-only).
    // step:       1-indexed step count (for bias correction).
    // lr:         learning rate; use the group's default lr if negative.
    // Element count is taken from grp.n (set at add_group()) — not caller-supplied.
    void step_group(int idx, VkBuffer params_buf, VkBuffer grad_buf,
                    float lr, uint32_t step);

    int group_count() const { return static_cast<int>(groups_.size()); }

private:
    VulkanContext& ctx_;
    float beta1_, beta2_, eps_;

    struct Group {
        std::unique_ptr<VulkanBuffer> m_buf;
        std::unique_ptr<VulkanBuffer> v_buf;
        uint32_t n;
        float    lr;
    };
    std::vector<Group> groups_;

    std::unique_ptr<VulkanShader>          shader_;
    std::unique_ptr<VulkanComputePipeline> pipeline_;
    VkDescriptorSet                        descriptor_set_ = VK_NULL_HANDLE;
    std::unique_ptr<VulkanBuffer>          ubo_buf_;  // 32-byte UBO
};
