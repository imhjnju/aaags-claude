// SP-3 T23 / SP-4 T3/T4: PreprocessBackwardPass — wraps preprocess_backward.comp.
//
// Bindings (mirroring preprocess_backward_bind::):
//   0..9  read-only SSBOs: positions, radii, cov3D, d_conics, d_opacity,
//                           sh_coeffs, scales, rotations, d_rgb, d_means2D
//   10..13 write-only SSBOs: d_means3D, d_sh, d_scales, d_rotations
//   14    read-only SSBO:   opacities (activated sigmoid values)
//   15    write-only SSBO:  d_raw_opacities
//   16    uniform buffer:   PreprocessBackwardUBO (192 bytes)
//   17    read-only SSBO:   raw_rotations (unnormalized quaternions)
//   18    read-only SSBO:   means2D_cache (pixel-space means2D from forward)
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

    // --- 2. Descriptor layout: 18 SSBOs (bindings 0..15, 17, 18) + 1 UBO (binding 16) ---
    // 19 bindings total; binding 16 is UNIFORM_BUFFER, bindings 17 and 18 are STORAGE_BUFFER.
    std::vector<VkDescriptorType> binding_types(19,
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
                           preprocess_backward_bind::D_ROTATIONS,     b.d_rotations);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::OPACITIES,       b.opacities);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::D_RAW_OPACITIES, b.d_raw_opacities);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::RAW_ROTATIONS,   b.raw_rotations);
    pipeline_->update_ssbo(descriptor_set_,
                           preprocess_backward_bind::MEANS2D_CACHE,   b.means2D_cache);

    // UBO binding 16 (PreprocessBackwardUBO, 192 bytes).
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
