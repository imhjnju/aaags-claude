// SP-3 T23: PreprocessBackwardPass — wraps preprocess_backward.comp.
//
// Bindings (mirroring preprocess_backward_bind::):
//   0..9  read-only SSBOs: positions, radii, cov3D, d_conics, d_opacity,
//                           sh_coeffs, scales, rotations, d_rgb, d_means2D
//   10..13 write-only SSBOs: d_means3D, d_sh, d_scales, d_rotations
//   14    uniform buffer:   PreprocessBackwardUBO (192 bytes)
//
// Dispatch: one 256-thread workgroup per Gaussian block.
// Grid = ceil(N/256) × 1 × 1.

#include "vulkan/preprocess_backward_pass.h"

// xxd-embedded SPIR-V (CMake produces preprocess_backward_spv.h in build dir).
#include "preprocess_backward_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

PreprocessBackwardPass::PreprocessBackwardPass(VulkanContext& ctx)
    : ctx_(ctx) {
    // --- 1. Load SPIR-V from embedded bytes --------------------------------
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(preprocess_backward_spv),
        static_cast<std::size_t>(preprocess_backward_spv_len));

    // --- 2. Descriptor layout: 14 SSBOs (bindings 0..13) + 1 UBO (binding 14) ---
    // 15 bindings total; binding 14 is UNIFORM_BUFFER.
    std::vector<VkDescriptorType> binding_types(15,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    binding_types[preprocess_backward_bind::PREPROCESS_BACKWARD_UBO] =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/0u,
        /*max_descriptor_sets=*/4u);

    // --- 3. Allocate the descriptor set once (update in-place per dispatch) ---
    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

PreprocessBackwardPass::~PreprocessBackwardPass() = default;

void PreprocessBackwardPass::bind_buffers(const Buffers& b, VkBuffer ubo) {
    // SSBOs 0..13.
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::POSITIONS,   b.positions);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::RADII,       b.radii);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::COV3D,       b.cov3D);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_CONICS,    b.d_conics);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_OPACITY,   b.d_opacity);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::SH_COEFFS,   b.sh_coeffs);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::SCALES,      b.scales);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::ROTATIONS,   b.rotations);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_RGB,       b.d_rgb);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_MEANS2D,   b.d_means2D);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_MEANS3D,   b.d_means3D);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_SH,        b.d_sh);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_SCALES,    b.d_scales);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_ROTATIONS, b.d_rotations);

    // UBO binding 14 (PreprocessBackwardUBO, 192 bytes).
    pipeline_->update_ubo(descriptor_set_,
                          preprocess_backward_bind::PREPROCESS_BACKWARD_UBO,
                          ubo,
                          sizeof(PreprocessBackwardUBO));
}

void PreprocessBackwardPass::dispatch_sync(uint32_t num_gaussians) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PreprocessBackwardPass::dispatch_sync called before bind_buffers()");
    if (num_gaussians == 0u) return;

    // ceil(N / 256) workgroups
    const uint32_t num_wg = (num_gaussians + 255u) / 256u;

    pipeline_->dispatch_sync(descriptor_set_,
                             num_wg, 1u, 1u,
                             nullptr, 0u);
}
