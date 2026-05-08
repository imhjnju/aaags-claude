#include "vulkan/preprocess_backward_eval3d_pass.h"

#include "preprocess_backward_eval3d_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

PreprocessBackwardEval3DPass::PreprocessBackwardEval3DPass(VulkanContext& ctx,
                                                             uint32_t spec_proper_ewa)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(preprocess_backward_eval3d_spv),
        static_cast<std::size_t>(preprocess_backward_eval3d_spv_len));

    VkSpecializationMapEntry spec_entry{};
    spec_entry.constantID = 0u;
    spec_entry.offset = 0u;
    spec_entry.size = sizeof(uint32_t);
    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 1u;
    spec_info.pMapEntries = &spec_entry;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &spec_proper_ewa;

    std::vector<VkDescriptorType> binding_types(
        preprocess_backward_eval3d_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[preprocess_backward_eval3d_bind::PREPROCESS_BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u, &spec_info);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

PreprocessBackwardEval3DPass::~PreprocessBackwardEval3DPass() = default;

void PreprocessBackwardEval3DPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::POSITIONS,       b.positions);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::RADII,           b.radii);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::SH_COEFFS,       b.sh_coeffs);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::SCALES,          b.scales);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::ROTATIONS,       b.rotations);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_RGB,           b.d_rgb);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_OPACITY,       b.d_opacity);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_GAUSS2SCREEN,  b.d_gauss2screen);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_MEANS3D,       b.d_means3D);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_SH,            b.d_sh);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_SCALES,        b.d_scales);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_ROTATIONS,     b.d_rotations);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::OPACITIES,       b.opacities);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::D_RAW_OPACITIES, b.d_raw_opacities);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::RAW_ROTATIONS,   b.raw_rotations);
    pipeline_->update_ssbo(descriptor_set_, preprocess_backward_eval3d_bind::FILTER_3D,        b.filter_3D);
    pipeline_->update_ubo(descriptor_set_,
                          preprocess_backward_eval3d_bind::PREPROCESS_BACKWARD_UBO,
                          ubo,
                          sizeof(PreprocessBackwardUBO));
}

void PreprocessBackwardEval3DPass::dispatch_sync(uint32_t num_gaussians) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error("PreprocessBackwardEval3DPass::dispatch_sync called before bind_buffers()");
    if (num_gaussians == 0u) return;
    uint32_t groups = (num_gaussians + 255u) / 256u;
    pipeline_->dispatch_sync(descriptor_set_, groups, 1u, 1u, nullptr, 0u);
}

void PreprocessBackwardEval3DPass::record(VkCommandBuffer cmd, uint32_t num_gaussians) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error("PreprocessBackwardEval3DPass::record called before bind_buffers()");
    if (num_gaussians == 0u) return;
    uint32_t groups = (num_gaussians + 255u) / 256u;
    pipeline_->record(cmd, descriptor_set_, groups, 1u, 1u, nullptr, 0u);
}
