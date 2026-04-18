// SP-2 T10: PrefixScanPass — 3-phase Blelloch exclusive scan on a uint[] SSBO.
//
// Orchestrates prefix_sum.comp (spec §4.8.2) per its documented protocol:
//   num_wg = (N + 255) / 256
//   phase 0   dispatch(num_wg, 1, 1)   push = { N,      0, 0, 0 }
//   barrier
//   phase 1   dispatch(1, 1, 1)        push = { num_wg, 1, 0, 0 }
//   barrier
//   phase 2   dispatch(num_wg, 1, 1)   push = { N,      2, 0, 0 }
//
// Phase 1 must see every phase-0 write to workgroup_sums; phase 2 reads the
// scanned workgroup_sums AND writes back into output_array. Both transitions
// are compute→compute RAW on the same SSBOs, so insert_compute_barrier() is
// the right primitive. The barrier between phases is part of the scan
// algorithm, not caller business — PrefixScanPass::record() inserts them
// unconditionally and leaves PRE-scan and POST-scan synchronization to the
// caller.

#include "vulkan/tile_binner_passes.h"

// xxd-embedded SPIR-V (CMake build dir produces prefix_sum_spv.h). Provides:
//   unsigned char prefix_sum_spv[], unsigned int prefix_sum_spv_len.
#include "prefix_sum_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
// prefix_sum.comp uses local_size_x=256 — keep in one place to avoid drift
// between shader and host dispatch arithmetic.
constexpr uint32_t kPrefixSumLocalSize = 256;
}  // namespace

PrefixScanPass::PrefixScanPass(VulkanContext& ctx)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(prefix_sum_spv),
        static_cast<std::size_t>(prefix_sum_spv_len));

    // 3 SSBOs: input_array, output_array, workgroup_sums. All
    // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER — no UBOs in this pipeline.
    std::vector<VkDescriptorType> binding_types(3,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/sizeof(PrefixSumPushConstants),
        /*max_descriptor_sets=*/4);

    descriptor_set_ = pipeline_->allocate_empty_descriptor_set();
}

void PrefixScanPass::bind_buffers(VkBuffer input_array,
                                  VkBuffer output_array,
                                  VkBuffer workgroup_sums) {
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::INPUT_ARRAY,    input_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::OUTPUT_ARRAY,   output_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::WORKGROUP_SUMS, workgroup_sums);
}

void PrefixScanPass::dispatch_phase(VkCommandBuffer cmd,
                                    uint32_t num_elements,
                                    uint32_t phase,
                                    uint32_t num_wgs) {
    PrefixSumPushConstants pc{};
    pc.num_elements = num_elements;
    pc.phase        = phase;
    pc.stride       = 0u;
    pc._pad         = 0u;
    pipeline_->record(cmd, descriptor_set_,
                      num_wgs, 1u, 1u,
                      &pc, sizeof(pc));
}

void PrefixScanPass::record(VkCommandBuffer cmd, uint32_t num_elements) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PrefixScanPass::record called before bind_buffers()");
    if (num_elements == 0u)
        return;  // nothing to scan; leave output untouched.

    const uint32_t num_wgs =
        (num_elements + kPrefixSumLocalSize - 1u) / kPrefixSumLocalSize;

    // Phase 1's single workgroup constraint: each phase-0 WG contributes one
    // slot into workgroup_sums, and phase 1 scans that in a single 256-thread
    // WG. So num_wgs must be <= 256 (i.e. N <= 65536). Enforce here to fail
    // fast with a clear message instead of garbage results.
    if (num_wgs > kPrefixSumLocalSize)
        throw std::runtime_error(
            "PrefixScanPass: num_elements exceeds single-level scan capacity "
            "(N > 256*256 = 65536). Multi-level recursion not implemented.");

    // Phase 0: per-workgroup local scan.
    dispatch_phase(cmd, num_elements, /*phase=*/0u, num_wgs);
    insert_compute_barrier(cmd);

    // Phase 1: scan workgroup_sums in a single workgroup.
    dispatch_phase(cmd, num_wgs, /*phase=*/1u, /*num_wgs=*/1u);
    insert_compute_barrier(cmd);

    // Phase 2: add workgroup offsets back onto output_array.
    dispatch_phase(cmd, num_elements, /*phase=*/2u, num_wgs);
}

void PrefixScanPass::scan_sync(uint32_t num_elements) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PrefixScanPass::scan_sync called before bind_buffers()");
    if (num_elements == 0u) return;

    // Allocate-record-submit-wait, single-submit style (matches
    // VulkanComputePipeline::dispatch_sync's shape; we can't reuse that
    // helper directly because it only does ONE dispatch, not a chain).
    VkCommandBuffer cmd = ctx_.allocatePrimary();

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    record(cmd, num_elements);

    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}
