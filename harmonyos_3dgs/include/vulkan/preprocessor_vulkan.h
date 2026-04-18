// SP-2: Vulkan forward pipeline — PreprocessorVulkan adapter.
//
// Hard-error constraints (see dev_notes/master_plan/sp2_vulkan_forward_plan.md):
//   - No silent fallbacks: every Vulkan failure must throw or abort.
//   - No CPU shadow path: this class is Vulkan-only; do not mirror CPU work.
//   - Deterministic layout: buffers and bindings must match preprocess.comp
//     exactly — any divergence is a bug.
//   - Ownership: this adapter owns its PreprocessPass and GPU buffers; the
//     VulkanContext is held by reference and must outlive the adapter.
//
// Two usage patterns are supported:
//   1) Layer-1 process() — sync, self-contained. Allocates/uploads/dispatches/
//      downloads within one call.
//   2) Layer-2 prepare_record() + record() — chained pipeline (T19). The
//      adapter allocates persistent GPU buffers, uploads inputs, and binds
//      to PreprocessPass in prepare_record(). Then record() writes the
//      dispatch onto a caller-owned VkCommandBuffer. Buffer getters expose
//      handles so TileBinner/Sorter/Rasterizer adapters can consume them
//      without a host round-trip. Buffers persist until the next
//      prepare_record() call or destruction.

#pragma once

#include "preprocessor.h"
#include "vulkan/vk_context.h"

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

class PreprocessPass;
class VulkanBuffer;

class PreprocessorVulkan : public Preprocessor {
public:
    explicit PreprocessorVulkan(VulkanContext& ctx);
    // Defined out-of-line in preprocessor_vulkan.cpp because
    // std::unique_ptr<PreprocessPass> requires PreprocessPass to be complete
    // at the destruction point.
    ~PreprocessorVulkan() override;

    PreprocessOutput process(const GaussianData& g, const Camera& cam,
                             const RenderConfig& cfg, FrameAllocator& alloc,
                             ForwardCache* cache = nullptr) override;

    // Layer 2: allocate persistent GPU buffers, upload inputs, and bind to
    // the PreprocessPass. Must be called before record(). Buffers live until
    // the next prepare_record() call or destruction. Hard-errors on SP-2
    // unsupported configs (eval_3D, non-16 tiles, antialiasing).
    void prepare_record(const GaussianData& g,
                        const Camera& cam,
                        const RenderConfig& cfg);

    // Layer 2: record dispatch into external command buffer. prepare_record()
    // must have been called first. The caller is responsible for inserting
    // a compute barrier AFTER this returns (use insert_compute_barrier).
    // Parameters are used to rebuild the push-constant block at record time.
    void record(VkCommandBuffer cmd, uint32_t num_gaussians, uint32_t sh_degree,
                uint32_t sh_coeffs_per_g, uint32_t num_tiles_x,
                uint32_t num_tiles_y, float scale_modifier);

    // Handles returned here are owned by *this; valid only between a call to
    // prepare_record() and the next prepare_record() / destruction. Do not
    // call vkDestroyBuffer on them. Returns VK_NULL_HANDLE before the first
    // prepare_record() call.
    VkBuffer means2D_buffer() const;
    VkBuffer depths_buffer() const;
    VkBuffer conic_opacity_packed_buffer() const;
    VkBuffer rgb_buffer() const;
    VkBuffer radii_buffer() const;
    VkBuffer tiles_touched_buffer() const;

private:
    VulkanContext& ctx_;
    std::unique_ptr<PreprocessPass> pass_;

    // Layer-2 persistent buffers. Filled by prepare_record(); released and
    // re-allocated on the next prepare_record(). Index into this vector:
    //   0 positions, 1 scales, 2 rotations, 3 opacities, 4 sh, 5 filter_3D
    //   6 means2D,   7 depths, 8 conic_opacity_packed,
    //   9 rgb,      10 radii, 11 tiles_touched, 12 camera_ubo
    // The mapping mirrors preprocess_bind:: for readability at call sites.
    std::vector<std::unique_ptr<VulkanBuffer>> record_bufs_;
};
