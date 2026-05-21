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

    struct SpecData {
        uint32_t proper_ewa;
        uint32_t skip_geometry;
    };
    VkSpecializationMapEntry spec_entries[2]{};
    spec_entries[0].constantID = 0u;
    spec_entries[0].offset = offsetof(SpecData, proper_ewa);
    spec_entries[0].size = sizeof(uint32_t);
    spec_entries[1].constantID = 1u;
    spec_entries[1].offset = offsetof(SpecData, skip_geometry);
    spec_entries[1].size = sizeof(uint32_t);
    SpecData normal_spec{spec_proper_ewa, 0u};
    SpecData skip_spec{spec_proper_ewa, 1u};
    VkSpecializationInfo normal_spec_info{};
    normal_spec_info.mapEntryCount = 2u;
    normal_spec_info.pMapEntries = spec_entries;
    normal_spec_info.dataSize = sizeof(SpecData);
    normal_spec_info.pData = &normal_spec;
    VkSpecializationInfo skip_spec_info{};
    skip_spec_info.mapEntryCount = 2u;
    skip_spec_info.pMapEntries = spec_entries;
    skip_spec_info.dataSize = sizeof(SpecData);
    skip_spec_info.pData = &skip_spec;

    std::vector<VkDescriptorType> binding_types(
        preprocess_backward_eval3d_bind::BINDING_COUNT,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[preprocess_backward_eval3d_bind::PREPROCESS_BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u, &normal_spec_info);
    skip_geometry_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_, binding_types, 0u, 4u, &skip_spec_info);
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
    skip_geometry_descriptor_set_ = skip_geometry_pipeline_->allocate_empty_descriptor_set();
}

PreprocessBackwardEval3DPass::~PreprocessBackwardEval3DPass() = default;

void PreprocessBackwardEval3DPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    auto bind = [&](VulkanComputePipeline& pipeline, VkDescriptorSet set) {
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::POSITIONS,       b.positions);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::RADII,           b.radii);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::SH_COEFFS,       b.sh_coeffs);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::SCALES,          b.scales);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::ROTATIONS,       b.rotations);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_RGB,           b.d_rgb);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_OPACITY,       b.d_opacity);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_GAUSS2SCREEN,  b.d_gauss2screen);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_MEANS3D,       b.d_means3D);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_SH,            b.d_sh);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_SCALES,        b.d_scales);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_ROTATIONS,     b.d_rotations);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::OPACITIES,       b.opacities);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::D_RAW_OPACITIES, b.d_raw_opacities);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::RAW_ROTATIONS,   b.raw_rotations);
        pipeline.update_ssbo(set, preprocess_backward_eval3d_bind::FILTER_3D,        b.filter_3D);
        pipeline.update_ubo(set,
                            preprocess_backward_eval3d_bind::PREPROCESS_BACKWARD_UBO,
                            ubo,
                            sizeof(PreprocessBackwardUBO));
    };
    bind(*pipeline_, descriptor_set_);
    bind(*skip_geometry_pipeline_, skip_geometry_descriptor_set_);
}

void PreprocessBackwardEval3DPass::dispatch_sync(uint32_t num_gaussians, bool skip_geometry) {
    VkDescriptorSet set = skip_geometry ? skip_geometry_descriptor_set_ : descriptor_set_;
    VulkanComputePipeline* pipeline = skip_geometry ? skip_geometry_pipeline_.get() : pipeline_.get();
    if (set == VK_NULL_HANDLE)
        throw std::runtime_error("PreprocessBackwardEval3DPass::dispatch_sync called before bind_buffers()");
    if (num_gaussians == 0u) return;
    uint32_t groups = (num_gaussians + 255u) / 256u;
    pipeline->dispatch_sync(set, groups, 1u, 1u, nullptr, 0u);
}

void PreprocessBackwardEval3DPass::record(VkCommandBuffer cmd, uint32_t num_gaussians, bool skip_geometry) {
    VkDescriptorSet set = skip_geometry ? skip_geometry_descriptor_set_ : descriptor_set_;
    VulkanComputePipeline* pipeline = skip_geometry ? skip_geometry_pipeline_.get() : pipeline_.get();
    if (set == VK_NULL_HANDLE)
        throw std::runtime_error("PreprocessBackwardEval3DPass::record called before bind_buffers()");
    if (num_gaussians == 0u) return;
    uint32_t groups = (num_gaussians + 255u) / 256u;
    pipeline->record(cmd, set, groups, 1u, 1u, nullptr, 0u);
}
