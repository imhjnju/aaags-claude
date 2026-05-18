// radix_sort_fuchsia.h -- Phase 2.2 wrapper around the vendored Fuchsia
// vk_radix_sort static library (`third_party/fuchsia_vk_radix_sort`).
//
// Wraps `radix_sort_vk_t` in a RAII C++ object that:
//   - Picks the right per-vendor target descriptor for the device behind a
//     given VulkanContext (auto-detect; throws if no target matches).
//   - Calls `radix_sort_vk_create` at construction (compiles 8 compute
//     pipelines — expensive, do once per process).
//   - Calls `radix_sort_vk_destroy` at destruction.
//   - Exposes `record(cmd, keyvals_in, keyvals_scratch, internal_scratch,
//     count, key_bits)` to append the sort to a caller-owned command buffer.
//     The caller is responsible for begin/end + submit + sync.
//   - Exposes `memory_requirements(count)` so the caller can size and align
//     its keyval / internal buffers correctly.
//   - Exposes `sorted_buffer_after_sort(...)` so the caller knows whether the
//     sorted output ended up in `keyvals_in` or `keyvals_scratch` (Fuchsia
//     ping-pongs every pass; even passes ⇒ even buf, odd ⇒ odd).
//
// IMPORTANT: VulkanContext must already have requested Vulkan 1.2 + the four
// required core features (shaderInt64, bufferDeviceAddress, vulkanMemoryModel,
// vulkanMemoryModelDeviceScope). VulkanContext::init() does this on Phase 2.2;
// constructing this wrapper on a context that did not enable them is a bug.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

class VulkanContext;

// Forward-declare the opaque library types so callers don't have to include
// the vendored header.
struct radix_sort_vk;
struct radix_sort_vk_target;

class RadixSortFuchsia {
public:
    /// `max_keyvals` is the upper bound on count for any sort recorded against
    /// this instance (used to size the library's internal scratch). Pick the
    /// largest sort count you expect; sorting fewer elements is fine.
    /// Throws std::runtime_error if no Fuchsia target matches the device, or
    /// if `radix_sort_vk_create` returns NULL.
    RadixSortFuchsia(VulkanContext& ctx, uint32_t max_keyvals,
                     uint32_t keyval_dwords = 2);
    ~RadixSortFuchsia();

    RadixSortFuchsia(const RadixSortFuchsia&)            = delete;
    RadixSortFuchsia& operator=(const RadixSortFuchsia&) = delete;
    RadixSortFuchsia(RadixSortFuchsia&&)                 = delete;
    RadixSortFuchsia& operator=(RadixSortFuchsia&&)      = delete;

    /// Aggregate of `radix_sort_vk_memory_requirements_t`. Sizes are in bytes;
    /// alignments are powers of two and must be respected when binding the
    /// VkBuffer to its VkDeviceMemory (over-allocate-and-bind-at-aligned-offset
    /// pattern, mirrored from `spike/fuchsia_radix_sort/bench_radix.cpp:67-73`).
    struct MemoryRequirements {
        VkDeviceSize keyval_size;          // bytes per single keyval (8 for u64)
        VkDeviceSize keyvals_size;         // bytes per even/odd keyval buffer
        VkDeviceSize keyvals_alignment;    // power of two
        VkDeviceSize internal_size;        // bytes for library-internal buffer
        VkDeviceSize internal_alignment;   // power of two
        VkDeviceSize indirect_size;        // bytes for optional indirect buffer
        VkDeviceSize indirect_alignment;   // power of two
    };

    /// Query buffer-size + alignment requirements for sorting `count` keyvals.
    MemoryRequirements memory_requirements(uint32_t count) const;

    /// Append the sort dispatches into `cmd`. Caller owns the command buffer
    /// and is responsible for vkBeginCommandBuffer / vkEndCommandBuffer /
    /// queue submit / fences. Caller also owns sync barriers around the sort:
    /// Fuchsia internally inserts COMPUTE→COMPUTE barriers between its own
    /// passes, but the caller must barrier on `keyvals_in` / `keyvals_scratch`
    /// / `internal_scratch` from any prior write (e.g. TRANSFER_WRITE on a
    /// staging upload) and after the sort before any subsequent read.
    ///
    /// `count` must be <= max_keyvals from the constructor and < 2^30.
    /// `key_bits` is the number of most-significant bits to sort on (1..64 for u64).
    /// `keyvals_in` and `keyvals_scratch` must be sized at
    /// `memory_requirements(count).keyvals_size` and aligned to
    /// `keyvals_alignment`. `internal_scratch` likewise for `internal_*`.
    /// All three buffers must have been created with
    /// VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    /// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT (and TRANSFER_DST_BIT on
    /// `internal_scratch` is recommended even though only direct mode needs
    /// no fill).
    void record(VkCommandBuffer cmd,
                VkBuffer keyvals_in,
                VkBuffer keyvals_scratch,
                VkBuffer internal_scratch,
                uint32_t count,
                uint32_t key_bits);

    /// After a sort with the given `count` and `key_bits`, returns whichever
    /// of (`keyvals_in`, `keyvals_scratch`) holds the sorted output. Fuchsia
    /// ping-pongs every internal pass; the parity of the pass count
    /// determines this. The library returns this info via the
    /// `keyvals_sorted` out-parameter on `radix_sort_vk_sort` — we mirror that
    /// query without re-recording the sort.
    ///
    /// NOTE: this re-runs `radix_sort_vk_sort` with a NULL command buffer to
    /// query the result; if Fuchsia changes its API to require a non-NULL cb,
    /// we will mirror the parity logic explicitly. Today (4165014f) the
    /// ping-pong info is plumbed alongside the dispatches; we capture it
    /// from the same record() call.
    VkBuffer sorted_buffer_after_sort(VkBuffer keyvals_in,
                                      VkBuffer keyvals_scratch,
                                      uint32_t count,
                                      uint32_t key_bits) const;

    /// Auto-detect helper exposed for testing. Returns NULL if no vendored
    /// target matches the device; the constructor throws in that case.
    static const radix_sort_vk_target* pick_target(VulkanContext& ctx);

private:
    VulkanContext&                      ctx_;
    radix_sort_vk*                      rs_           = nullptr;
    const radix_sort_vk_target*         target_       = nullptr;
    uint32_t                            max_keyvals_  = 0;
    uint32_t                            keyval_dwords_ = 2;
    // The result-buffer parity is stable for a given (count, key_bits) pair
    // because Fuchsia's pass count is a deterministic function of key_bits and
    // the keyval radix. We capture it on the most recent `record()` call.
    VkBuffer                            last_sorted_  = VK_NULL_HANDLE;
};
