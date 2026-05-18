// SP-2 T14: RadixSortPass — 16-pass LSD 4-bit radix sort over uint64 keys.
//
// Each pass = count (per-WG 16-bucket histogram) → exclusive scan (over 16
// histogram entries) → scatter (stable per-bucket positioning). Two
// key/value buffer pairs (A, B) are ping-ponged:
//
//   initial: in=A, out=B
//   pass 0:  count reads A, writes hist_count; scan writes hist_scan;
//            scatter reads A (+hist_scan as bucket_offsets) and writes B;
//            swap → in=B, out=A
//   pass 1:  count reads B, scatter reads B, writes A; swap → in=A, out=B
//   ...
//   after 16 (even) passes: data is back in A (the original keys_in_buf).
//
// Single-workgroup variant (spec §4.8.4/5, SP-2 Phase 1): dispatch exactly
// one WG of size 256 — so num_elements must fit in that WG (<= 256). The
// tiny fixture has R = 103 which fits. Multi-WG stable radix is deferred.
//
// The 16-element histogram scan uses PrefixScanPass (T10). We rebind its
// descriptor set per pass because the buffer pair feeding the scan does not
// change — PrefixScanPass::scan_sync handles the sync internally, and
// radix_sort_count / radix_sort_scatter each have their own submit+wait via
// VulkanComputePipeline::dispatch_sync. That gives us the required ordering
// (count → scan → scatter) without needing a manual barrier: each
// dispatch_sync does a vkQueueWaitIdle before returning.

#include "vulkan/sort_passes.h"

// xxd-embedded SPIR-V (CMake build dir produces these headers).
#include "radix_sort_count_spv.h"
#include "radix_sort_scatter_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kRadixLocalSize = 256;
constexpr uint32_t kRadixBuckets   = 16;
constexpr uint32_t kRadixPasses    = 16;

uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}
}  // namespace

RadixSortPass::RadixSortPass(VulkanContext& ctx)
    : ctx_(ctx) {
    // Pool sizing: one DS for the sync path + kRadixPasses DSes for the
    // record path. Record-path DSes are pre-allocated here so we can update
    // each exactly once (in sort_record()) without violating Vulkan's
    // "no update between bind and submit" rule for the reused DSes.
    constexpr uint32_t kDsPerPipeline = 1u + kRadixPasses;  // 17

    // --- count pipeline: 2 SSBOs (keys_in, histograms) -----------------
    count_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(radix_sort_count_spv),
        static_cast<std::size_t>(radix_sort_count_spv_len));
    {
        std::vector<VkDescriptorType> binding_types(2,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        count_pipeline_ = std::make_unique<VulkanComputePipeline>(
            ctx_,
            *count_shader_,
            binding_types,
            /*push_constant_bytes=*/sizeof(RadixSortPushConstants),
            /*max_descriptor_sets=*/kDsPerPipeline);
        count_ds_ = count_pipeline_->allocate_empty_descriptor_set();
        count_ds_per_pass_.resize(kRadixPasses);
        for (uint32_t p = 0; p < kRadixPasses; ++p) {
            count_ds_per_pass_[p] =
                count_pipeline_->allocate_empty_descriptor_set();
        }
    }

    // --- scatter pipeline: 5 SSBOs ------------------------------------
    scatter_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(radix_sort_scatter_spv),
        static_cast<std::size_t>(radix_sort_scatter_spv_len));
    {
        std::vector<VkDescriptorType> binding_types(5,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        scatter_pipeline_ = std::make_unique<VulkanComputePipeline>(
            ctx_,
            *scatter_shader_,
            binding_types,
            /*push_constant_bytes=*/sizeof(RadixSortPushConstants),
            /*max_descriptor_sets=*/kDsPerPipeline);
        scatter_ds_ = scatter_pipeline_->allocate_empty_descriptor_set();
        scatter_ds_per_pass_.resize(kRadixPasses);
        for (uint32_t p = 0; p < kRadixPasses; ++p) {
            scatter_ds_per_pass_[p] =
                scatter_pipeline_->allocate_empty_descriptor_set();
        }
    }

    // --- inner prefix scan (16-element exclusive) ---------------------
    scan_pass_ = std::make_unique<PrefixScanPass>(ctx_);
}

void RadixSortPass::sort_sync(VkBuffer keys_in_buf,  VkBuffer values_in_buf,
                              VkBuffer keys_out_buf, VkBuffer values_out_buf,
                              VkBuffer hist_count_buf, VkBuffer hist_scan_buf,
                              VkBuffer wg_sums_buf, VkBuffer wg_sums2_buf,
                              uint32_t num_elements) {
    if (num_elements == 0u) return;

    VkCommandBuffer cmd = ctx_.allocatePrimary();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    sort_record(cmd,
                keys_in_buf, values_in_buf,
                keys_out_buf, values_out_buf,
                hist_count_buf, hist_scan_buf,
                wg_sums_buf, wg_sums2_buf,
                num_elements);
    insert_compute_barrier(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);

    // After 16 even passes, sorted data is in keys_in_buf / values_in_buf.
}

// Layer-2 record-path variant of sort_sync(). Identical ping-pong structure,
// but each inner dispatch is recorded onto the caller-owned cmd buffer via
// VulkanComputePipeline::record(), and we insert a compute-to-compute barrier
// between each (count → scan → scatter) transition inside the same pass as
// well as between the final scatter of pass i and the count of pass i+1.
//
// Vulkan constraint: updating a descriptor set between its binding in a cmd
// buffer and submission is undefined. We sidestep this by pre-allocating one
// (count_ds, scatter_ds) pair per pass and updating each exactly once here,
// BEFORE the corresponding record() call. The inner PrefixScanPass's DS is
// updated once at the top with hist_count_buf/hist_scan_buf/wg_sums_buf
// (which don't vary across passes) so its 3 per-pass dispatches are safe.
void RadixSortPass::sort_record(VkCommandBuffer cmd,
                                VkBuffer keys_in_buf,  VkBuffer values_in_buf,
                                VkBuffer keys_out_buf, VkBuffer values_out_buf,
                                VkBuffer hist_count_buf, VkBuffer hist_scan_buf,
                                VkBuffer wg_sums_buf, VkBuffer wg_sums2_buf,
                                uint32_t num_elements) {
    if (num_elements == 0u) return;

    const uint32_t num_wgs = ceil_div_u32(num_elements, kRadixLocalSize);
    const uint32_t hist_entries = num_wgs * kRadixBuckets;

    // Bind the inner scan's DS exactly once — the histogram buffers are
    // constant across all 16 passes (each pass's count writes them fresh,
    // each pass's scatter consumes them fresh, sequenced by barriers).
    scan_pass_->bind_buffers_2level(hist_count_buf, hist_scan_buf, wg_sums_buf, wg_sums2_buf);

    // Pre-populate per-pass descriptor sets. Each pass's cur_keys_in /
    // cur_vals_in / cur_keys_out / cur_vals_out is determined by ping-pong
    // parity: pass 0 reads A writes B; pass 1 reads B writes A; ...
    for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
        const bool read_A = (pass % 2u == 0u);
        VkBuffer cur_keys_in  = read_A ? keys_in_buf  : keys_out_buf;
        VkBuffer cur_vals_in  = read_A ? values_in_buf : values_out_buf;
        VkBuffer cur_keys_out = read_A ? keys_out_buf : keys_in_buf;
        VkBuffer cur_vals_out = read_A ? values_out_buf : values_in_buf;

        count_pipeline_->update_ssbo(count_ds_per_pass_[pass],
                                     radix_count_bind::KEYS_IN,
                                     cur_keys_in);
        count_pipeline_->update_ssbo(count_ds_per_pass_[pass],
                                     radix_count_bind::HISTOGRAMS,
                                     hist_count_buf);

        scatter_pipeline_->update_ssbo(scatter_ds_per_pass_[pass],
                                       radix_scatter_bind::KEYS_IN,
                                       cur_keys_in);
        scatter_pipeline_->update_ssbo(scatter_ds_per_pass_[pass],
                                       radix_scatter_bind::VALUES_IN,
                                       cur_vals_in);
        scatter_pipeline_->update_ssbo(scatter_ds_per_pass_[pass],
                                       radix_scatter_bind::BUCKET_OFFSETS,
                                       hist_scan_buf);
        scatter_pipeline_->update_ssbo(scatter_ds_per_pass_[pass],
                                       radix_scatter_bind::KEYS_OUT,
                                       cur_keys_out);
        scatter_pipeline_->update_ssbo(scatter_ds_per_pass_[pass],
                                       radix_scatter_bind::VALUES_OUT,
                                       cur_vals_out);
    }

    // Record all 16 passes of (count → scan → scatter), with compute→compute
    // barriers between dispatches. Descriptor sets are now static for the
    // remainder of the command buffer's life, satisfying Vulkan's constraint.
    for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
        const uint32_t current_bit = pass * 4u;

        RadixSortPushConstants pc{};
        pc.num_elements = num_elements;
        pc.current_bit  = current_bit;
        pc.num_workgroups = num_wgs;
        pc._pad1        = 0u;

        // ---------- Step 1: count ----------
        count_pipeline_->record(cmd, count_ds_per_pass_[pass],
                                /*gx=*/num_wgs, 1u, 1u,
                                &pc, sizeof(pc));
        insert_compute_barrier(cmd);

        // ---------- Step 2: exclusive scan over bucket-major histogram entries ----
        scan_pass_->record(cmd, hist_entries);
        insert_compute_barrier(cmd);

        // ---------- Step 3: scatter ----------
        scatter_pipeline_->record(cmd, scatter_ds_per_pass_[pass],
                                  /*gx=*/num_wgs, 1u, 1u,
                                  &pc, sizeof(pc));
        // Barrier between passes: next pass's count reads what this scatter
        // wrote. Skip the trailing barrier after the final scatter.
        if (pass + 1u < kRadixPasses) {
            insert_compute_barrier(cmd);
        }
    }
    // After 16 even swaps, sorted data is in keys_in_buf / values_in_buf.
}
