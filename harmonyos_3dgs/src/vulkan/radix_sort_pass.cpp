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
// Shaders declare local_size_x=256. Single-WG variant: one dispatch group.
constexpr uint32_t kRadixLocalSize = 256;
// 4-bit radix => 16 buckets per pass. Histogram is 16 uint32s.
constexpr uint32_t kRadixBuckets   = 16;
// 64-bit keys / 4 bits per pass = 16 passes.
constexpr uint32_t kRadixPasses    = 16;
}  // namespace

RadixSortPass::RadixSortPass(VulkanContext& ctx)
    : ctx_(ctx) {
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
            /*max_descriptor_sets=*/4);
        count_ds_ = count_pipeline_->allocate_empty_descriptor_set();
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
            /*max_descriptor_sets=*/4);
        scatter_ds_ = scatter_pipeline_->allocate_empty_descriptor_set();
    }

    // --- inner prefix scan (16-element exclusive) ---------------------
    scan_pass_ = std::make_unique<PrefixScanPass>(ctx_);
}

void RadixSortPass::sort_sync(VkBuffer keys_in_buf,  VkBuffer values_in_buf,
                              VkBuffer keys_out_buf, VkBuffer values_out_buf,
                              VkBuffer hist_count_buf, VkBuffer hist_scan_buf,
                              VkBuffer wg_sums_buf,
                              uint32_t num_elements) {
    if (num_elements == 0u) return;
    if (num_elements > kRadixLocalSize)
        throw std::runtime_error(
            "RadixSortPass::sort_sync: num_elements > 256 — SP-2 Phase 1 "
            "single-workgroup radix does not support larger inputs.");

    // Ping-pong state. After each pass we swap so the "in" of the next pass
    // is what the scatter just wrote.
    VkBuffer cur_keys_in  = keys_in_buf;
    VkBuffer cur_vals_in  = values_in_buf;
    VkBuffer cur_keys_out = keys_out_buf;
    VkBuffer cur_vals_out = values_out_buf;

    for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
        const uint32_t current_bit = pass * 4u;

        // ---------- Step 1: count ----------
        count_pipeline_->update_ssbo(count_ds_,
                                     radix_count_bind::KEYS_IN,
                                     cur_keys_in);
        count_pipeline_->update_ssbo(count_ds_,
                                     radix_count_bind::HISTOGRAMS,
                                     hist_count_buf);
        {
            RadixSortPushConstants pc{};
            pc.num_elements = num_elements;
            pc.current_bit  = current_bit;
            pc._pad0        = 0u;
            pc._pad1        = 0u;
            // Single WG: ceil(num_elements / 256) == 1 for num_elements <= 256.
            count_pipeline_->dispatch_sync(count_ds_,
                                           /*gx=*/1u, 1u, 1u,
                                           &pc, sizeof(pc));
        }

        // ---------- Step 2: exclusive scan over 16 histogram entries ----
        // prefix_sum.comp does NOT scan in-place (phase 0 reads input_array,
        // writes output_array), so we use two separate buffers: hist_count_buf
        // (input) → hist_scan_buf (output). The scatter shader reads the scan
        // result at binding 2 as bucket_offsets.
        scan_pass_->bind_buffers(hist_count_buf, hist_scan_buf, wg_sums_buf);
        scan_pass_->scan_sync(kRadixBuckets);

        // ---------- Step 3: scatter ----------
        scatter_pipeline_->update_ssbo(scatter_ds_,
                                       radix_scatter_bind::KEYS_IN,
                                       cur_keys_in);
        scatter_pipeline_->update_ssbo(scatter_ds_,
                                       radix_scatter_bind::VALUES_IN,
                                       cur_vals_in);
        scatter_pipeline_->update_ssbo(scatter_ds_,
                                       radix_scatter_bind::BUCKET_OFFSETS,
                                       hist_scan_buf);
        scatter_pipeline_->update_ssbo(scatter_ds_,
                                       radix_scatter_bind::KEYS_OUT,
                                       cur_keys_out);
        scatter_pipeline_->update_ssbo(scatter_ds_,
                                       radix_scatter_bind::VALUES_OUT,
                                       cur_vals_out);
        {
            RadixSortPushConstants pc{};
            pc.num_elements = num_elements;
            pc.current_bit  = current_bit;
            pc._pad0        = 0u;
            pc._pad1        = 0u;
            scatter_pipeline_->dispatch_sync(scatter_ds_,
                                             /*gx=*/1u, 1u, 1u,
                                             &pc, sizeof(pc));
        }

        // ---------- Ping-pong for next pass ----------
        std::swap(cur_keys_in,  cur_keys_out);
        std::swap(cur_vals_in,  cur_vals_out);
    }

    // After 16 (even) swaps, cur_keys_in == keys_in_buf, cur_keys_out ==
    // keys_out_buf. The last scatter wrote into cur_keys_out **before** the
    // final swap, i.e. into keys_in_buf after the swap: so the sorted data is
    // in keys_in_buf. This matches the API contract.
    //
    // NOTE: the API contract assumes an even number of passes. Changing
    // kRadixPasses to an odd number would invert the final output buffer.
}
