#include "vulkan/preprocess_pass.h"
#include "vulkan/vk_camera_ubo.h"

// xxd-embedded SPIR-V for preprocess.comp (produced by the build via
// gs3d_embed_spirv). Provides: unsigned char preprocess_spv[] and
// unsigned int preprocess_spv_len.
#include "preprocess_spv.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
// Workgroup size for preprocess.comp (spec §4.8.1: local_size_x = 256).
constexpr uint32_t kPreprocessLocalSize = 256;
}  // namespace

PreprocessPass::PreprocessPass(VulkanContext& ctx,
                               uint32_t spec_training,
                               uint32_t spec_eval_3D)
    : ctx_(ctx) {
    // --- 1. Load SPIR-V module from embedded bytes ------------------------
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(preprocess_spv),
        static_cast<std::size_t>(preprocess_spv_len));

    // --- 2. Specialization constants (spec §4.5) --------------------------
    // constant_id 0 = spec_training, constant_id 1 = spec_eval_3D.
    // Both are uint32 values embedded into a contiguous data blob.
    struct SpecData {
        uint32_t training;
        uint32_t eval_3D;
    };
    // Keep SpecData, entries, and spec_info alive across vkCreateComputePipelines.
    // We hold them on the stack during the VulkanComputePipeline constructor
    // call; Vulkan copies specialization data at pipeline creation so local
    // lifetimes are sufficient.
    SpecData spec_data{spec_training, spec_eval_3D};
    std::array<VkSpecializationMapEntry, 2> spec_entries{{
        {preprocess_spec::TRAINING,
         static_cast<uint32_t>(offsetof(SpecData, training)),
         sizeof(uint32_t)},
        {preprocess_spec::EVAL_3D,
         static_cast<uint32_t>(offsetof(SpecData, eval_3D)),
         sizeof(uint32_t)},
    }};
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries   = spec_entries.data();
    spec_info.dataSize      = sizeof(SpecData);
    spec_info.pData         = &spec_data;

    // --- 3. Descriptor layout: 13 SSBOs + 1 UBO (binding 12) + 1 SSBO (binding 13)
    std::vector<VkDescriptorType> binding_types(14,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[preprocess_bind::CAMERA_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(PreprocessPushConstants),
        /*max_descriptor_sets=*/4,
        /*spec_info=*/&spec_info);

    // --- 4. Allocate the descriptor set once ------------------------------
    // The pool was created without VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
    // and sized for only 4 sets, so re-allocating in bind_buffers() would
    // exhaust the pool on the 5th frame. Instead we allocate once here and
    // let bind_buffers() update the set in-place per frame.
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void PreprocessPass::bind_buffers(const Buffers& b) {
    // Update the 12 SSBO bindings in-place on the pre-allocated descriptor
    // set. Ordering mirrors preprocess_bind::.
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::POSITIONS,            b.positions);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::SCALES,               b.scales);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::ROTATIONS,            b.rotations);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::OPACITIES,            b.opacities);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::SH,                   b.sh);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::FILTER_3D,            b.filter_3D);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::MEANS2D,              b.means2D);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::DEPTHS,               b.depths);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::CONIC_OPACITY_PACKED, b.conic_opacity_packed);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::RGB,                  b.rgb);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::RADII,                b.radii);
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::TILES_TOUCHED,        b.tiles_touched);
    // UBO binding 12.
    pipeline_->update_ubo(descriptor_set_,
                          preprocess_bind::CAMERA_UBO,
                          b.camera_ubo,
                          sizeof(CameraUBO));
    // SSBO binding 13: float radius_f output.
    pipeline_->update_ssbo(descriptor_set_, preprocess_bind::RADIUS_F, b.radius_f);
}

void PreprocessPass::dispatch_sync(const PreprocessPushConstants& pc) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PreprocessPass::dispatch_sync called before bind_buffers()");
    const uint32_t groups =
        (pc.num_gaussians + kPreprocessLocalSize - 1) / kPreprocessLocalSize;
    pipeline_->dispatch_sync(descriptor_set_,
                             groups, 1, 1,
                             &pc, sizeof(pc));
}

void PreprocessPass::record(VkCommandBuffer cmd,
                            const PreprocessPushConstants& pc) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PreprocessPass::record called before bind_buffers()");
    const uint32_t groups =
        (pc.num_gaussians + kPreprocessLocalSize - 1) / kPreprocessLocalSize;
    pipeline_->record(cmd, descriptor_set_,
                      groups, 1, 1,
                      &pc, sizeof(pc));
}
