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
    // Second DS for 2-level support; allocated here so it's always available.
    descriptor_set2_ = pipeline_->allocate_empty_descriptor_set();
}

void PrefixScanPass::bind_buffers(VkBuffer input_array,
                                  VkBuffer output_array,
                                  VkBuffer workgroup_sums) {
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::INPUT_ARRAY,    input_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::OUTPUT_ARRAY,   output_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::WORKGROUP_SUMS, workgroup_sums);
    has_2level_ = false;
}

void PrefixScanPass::bind_buffers_2level(VkBuffer input_array,
                                         VkBuffer output_array,
                                         VkBuffer workgroup_sums,
                                         VkBuffer workgroup_sums2) {
    // DS1: {input, output, wg_sums1}  — used for phase 0 on N and phase 2 on N.
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::INPUT_ARRAY,    input_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::OUTPUT_ARRAY,   output_array);
    pipeline_->update_ssbo(descriptor_set_, prefix_sum_bind::WORKGROUP_SUMS, workgroup_sums);
    // DS2: {wg_sums1, wg_sums1, wg_sums2}  — used for the mid-level 3-phase scan.
    // Binding wg_sums1 as both INPUT and OUTPUT is safe: the shader copies to
    // shared memory before writing, so there is no intra-invocation aliasing.
    pipeline_->update_ssbo(descriptor_set2_, prefix_sum_bind::INPUT_ARRAY,    workgroup_sums);
    pipeline_->update_ssbo(descriptor_set2_, prefix_sum_bind::OUTPUT_ARRAY,   workgroup_sums);
    pipeline_->update_ssbo(descriptor_set2_, prefix_sum_bind::WORKGROUP_SUMS, workgroup_sums2);
    has_2level_ = true;
}

void PrefixScanPass::dispatch_phase(VkCommandBuffer cmd,
                                    uint32_t num_elements,
                                    uint32_t phase,
                                    uint32_t num_wgs,
                                    VkDescriptorSet ds) {
    if (ds == VK_NULL_HANDLE) ds = descriptor_set_;
    PrefixSumPushConstants pc{};
    pc.num_elements = num_elements;
    pc.phase        = phase;
    pc.stride       = 0u;
    pc._pad         = 0u;
    pipeline_->record(cmd, ds,
                      num_wgs, 1u, 1u,
                      &pc, sizeof(pc));
}

void PrefixScanPass::record(VkCommandBuffer cmd, uint32_t num_elements) {
    if (descriptor_set_ == VK_NULL_HANDLE)
        throw std::runtime_error(
            "PrefixScanPass::record called before bind_buffers()");
    if (num_elements == 0u)
        return;  // nothing to scan; leave output untouched.

    const uint32_t num_wg1 =
        (num_elements + kPrefixSumLocalSize - 1u) / kPrefixSumLocalSize;

    if (num_wg1 <= kPrefixSumLocalSize) {
        // ---------------------------------------------------------------
        // Single-level path (N ≤ 65536): original 3-phase protocol.
        // ---------------------------------------------------------------
        dispatch_phase(cmd, num_elements, /*phase=*/0u, num_wg1);
        insert_compute_barrier(cmd);
        dispatch_phase(cmd, num_wg1, /*phase=*/1u, /*num_wgs=*/1u);
        insert_compute_barrier(cmd);
        dispatch_phase(cmd, num_elements, /*phase=*/2u, num_wg1);
    } else {
        // ---------------------------------------------------------------
        // Two-level path (65536 < N ≤ 256³ ≈ 16M):
        //   DS1 = {input, output, wg_sums1}
        //   DS2 = {wg_sums1, wg_sums1, wg_sums2}  (set by bind_buffers_2level)
        //
        //   Step 1 (DS1 ph0, num_wg1 WGs): local scan of input[N] → output[N],
        //          produce wg_sums1[num_wg1] = per-WG totals.
        //   Step 2 (DS2 ph0, num_wg2 WGs): local scan of wg_sums1 in-place,
        //          produce wg_sums2[num_wg2] = level-2 WG totals.
        //   Step 3 (DS2 ph1, 1 WG):        scan wg_sums2 in a single WG.
        //   Step 4 (DS2 ph2, num_wg2 WGs): add wg_sums2 back to wg_sums1
        //          → wg_sums1 now holds the correct exclusive prefix of level-1 totals.
        //   Step 5 (DS1 ph2, num_wg1 WGs): add wg_sums1 back to output[N].
        // ---------------------------------------------------------------
        if (!has_2level_)
            throw std::runtime_error(
                "PrefixScanPass: N > 65536 requires bind_buffers_2level() "
                "before record(). Call bind_buffers_2level with wg_sums2.");

        const uint32_t num_wg2 =
            (num_wg1 + kPrefixSumLocalSize - 1u) / kPrefixSumLocalSize;
        if (num_wg2 > kPrefixSumLocalSize)
            throw std::runtime_error(
                "PrefixScanPass: N exceeds two-level scan capacity "
                "(N > 256^3 ≈ 16M). Not implemented.");

        // Step 1
        dispatch_phase(cmd, num_elements, /*phase=*/0u, num_wg1, descriptor_set_);
        insert_compute_barrier(cmd);
        // Step 2
        dispatch_phase(cmd, num_wg1, /*phase=*/0u, num_wg2, descriptor_set2_);
        insert_compute_barrier(cmd);
        // Step 3
        dispatch_phase(cmd, num_wg2, /*phase=*/1u, /*num_wgs=*/1u, descriptor_set2_);
        insert_compute_barrier(cmd);
        // Step 4
        dispatch_phase(cmd, num_wg1, /*phase=*/2u, num_wg2, descriptor_set2_);
        insert_compute_barrier(cmd);
        // Step 5
        dispatch_phase(cmd, num_elements, /*phase=*/2u, num_wg1, descriptor_set_);
    }
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
