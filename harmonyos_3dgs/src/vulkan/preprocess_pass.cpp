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

    // --- 3. Descriptor layout: 12 SSBOs + 1 UBO (binding 12) --------------
    std::vector<VkDescriptorType> binding_types(13,
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
}

void PreprocessPass::bind_buffers(const Buffers& b) {
    // Assemble SSBO list in binding order 0..11 (mirrors preprocess_bind::).
    std::vector<VkBuffer> ssbos{
        b.positions,               // 0: POSITIONS
        b.scales,                  // 1: SCALES
        b.rotations,               // 2: ROTATIONS
        b.opacities,               // 3: OPACITIES
        b.sh,                      // 4: SH
        b.filter_3D,               // 5: FILTER_3D
        b.means2D,                 // 6: MEANS2D
        b.depths,                  // 7: DEPTHS
        b.conic_opacity_packed,    // 8: CONIC_OPACITY_PACKED
        b.rgb,                     // 9: RGB
        b.radii,                   // 10: RADII
        b.tiles_touched,           // 11: TILES_TOUCHED
    };
    descriptor_set_ = pipeline_->allocateDescriptorSet(ssbos);
    // UBO binding 12 is populated separately (allocateDescriptorSet leaves
    // UBO bindings unwritten by design).
    pipeline_->update_ubo(descriptor_set_,
                          preprocess_bind::CAMERA_UBO,
                          b.camera_ubo,
                          sizeof(CameraUBO));
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
