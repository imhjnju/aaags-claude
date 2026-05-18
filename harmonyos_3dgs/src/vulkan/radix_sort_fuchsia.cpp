// radix_sort_fuchsia.cpp -- Phase 2.2 wrapper. See header for contract.

#include "vulkan/radix_sort_fuchsia.h"

#include "vulkan/vk_context.h"

#include "radix_sort/platforms/vk/radix_sort_vk.h"

#include <stdexcept>
#include <string>

// `radix_sort_vk_t` is a typedef of `struct radix_sort_vk` (per the vendored
// header), so the forward decl in our header (`struct radix_sort_vk;`)
// resolves to the same type. No reinterpret_cast needed between them. Same
// for `radix_sort_vk_target_t` / `struct radix_sort_vk_target`.

const radix_sort_vk_target* RadixSortFuchsia::pick_target(VulkanContext& ctx) {
    // Auto-detect via the device's VkPhysicalDeviceProperties + a fixed
    // keyval_dwords=2 (i.e. 64-bit keyvals — the only width we care about for
    // 3DGS, where each keyval is (tile_id_high32 | depth_low32) in the
    // production sort path).
    //
    // Auto-target lives here (not in vk_context.cpp) because it's specific to
    // the Fuchsia library: it inspects vendor/device IDs against the SPIR-V
    // blobs we vendor. VulkanContext is generic infra — it should not know
    // about Fuchsia targets.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice(), &props);
    return radix_sort_vk_target_auto_detect(&props, /*keyval_dwords=*/2);
}

namespace {
constexpr uint32_t kMaxFuchsiaKeyvals = 1u << 30;

void validate_count(uint32_t count, uint32_t max_keyvals, const char* where) {
    if (count > max_keyvals) {
        throw std::runtime_error(std::string(where) + ": count exceeds max_keyvals");
    }
    if (count >= kMaxFuchsiaKeyvals) {
        throw std::runtime_error(std::string(where) + ": count must be < 1<<30");
    }
}
}

RadixSortFuchsia::RadixSortFuchsia(VulkanContext& ctx, uint32_t max_keyvals,
                                   uint32_t keyval_dwords)
    : ctx_(ctx), max_keyvals_(max_keyvals), keyval_dwords_(keyval_dwords) {
    if (max_keyvals_ == 0u || max_keyvals_ >= kMaxFuchsiaKeyvals) {
        throw std::runtime_error("RadixSortFuchsia: max_keyvals must be in [1, 1<<30)");
    }
    if (keyval_dwords_ == 0u || keyval_dwords_ > 4u) {
        throw std::runtime_error("RadixSortFuchsia: keyval_dwords must be 1, 2, 3, or 4");
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
    target_ = radix_sort_vk_target_auto_detect(&props, keyval_dwords_);
    if (target_ == nullptr) {
        throw std::runtime_error(
            std::string("RadixSortFuchsia: no vendored target matches device '") +
            props.deviceName + "' (vendorID=0x" +
            std::to_string(props.vendorID) + " deviceID=0x" +
            std::to_string(props.deviceID) +
            "). Vendored targets: nvidia/sm35, arm/bifrost8.");
    }

    rs_ = radix_sort_vk_create(ctx_.device(),
                               /*ac=*/nullptr,
                               /*pc=*/VK_NULL_HANDLE,
                               target_);
    if (rs_ == nullptr) {
        throw std::runtime_error(
            "RadixSortFuchsia: radix_sort_vk_create returned NULL "
            "(target was selected but pipeline creation failed — likely "
            "missing required device feature; check VulkanContext logs)");
    }
}

RadixSortFuchsia::~RadixSortFuchsia() {
    if (rs_ != nullptr) {
        radix_sort_vk_destroy(rs_, ctx_.device(), /*ac=*/nullptr);
        rs_ = nullptr;
    }
}

RadixSortFuchsia::MemoryRequirements
RadixSortFuchsia::memory_requirements(uint32_t count) const {
    validate_count(count, max_keyvals_, "RadixSortFuchsia::memory_requirements");

    radix_sort_vk_memory_requirements_t mr{};
    radix_sort_vk_get_memory_requirements(rs_, count, &mr);
    MemoryRequirements out{};
    out.keyval_size        = mr.keyval_size;
    out.keyvals_size       = mr.keyvals_size;
    out.keyvals_alignment  = mr.keyvals_alignment;
    out.internal_size      = mr.internal_size;
    out.internal_alignment = mr.internal_alignment;
    out.indirect_size      = mr.indirect_size;
    out.indirect_alignment = mr.indirect_alignment;
    return out;
}

void RadixSortFuchsia::record(VkCommandBuffer cmd,
                              VkBuffer keyvals_in,
                              VkBuffer keyvals_scratch,
                              VkBuffer internal_scratch,
                              uint32_t count,
                              uint32_t key_bits) {
    if (count == 0 || key_bits == 0) {
        // Library no-ops in this case (per radix_sort_vk.h: count<=1 or
        // key_bits==0 returns immediately). With zero passes the input is
        // also the output.
        last_sorted_ = keyvals_in;
        return;
    }
    validate_count(count, max_keyvals_, "RadixSortFuchsia::record");
    const uint32_t max_key_bits = keyval_dwords_ * 32u;
    if (key_bits > max_key_bits) {
        throw std::runtime_error("RadixSortFuchsia::record: key_bits exceeds record width");
    }

    radix_sort_vk_memory_requirements_t mr{};
    radix_sort_vk_get_memory_requirements(rs_, count, &mr);

    radix_sort_vk_sort_info_t info{};
    info.ext      = nullptr;
    info.key_bits = key_bits;
    info.count    = count;
    info.keyvals_even.buffer = keyvals_in;
    info.keyvals_even.offset = 0;
    info.keyvals_even.range  = mr.keyvals_size;
    info.keyvals_odd.buffer  = keyvals_scratch;
    info.keyvals_odd.offset  = 0;
    info.keyvals_odd.range   = mr.keyvals_size;
    info.internal.buffer     = internal_scratch;
    info.internal.offset     = 0;
    info.internal.range      = mr.internal_size;

    VkDescriptorBufferInfo sorted_out{};
    radix_sort_vk_sort(rs_, &info, ctx_.device(), cmd, &sorted_out);
    last_sorted_ = sorted_out.buffer;
}

VkBuffer RadixSortFuchsia::sorted_buffer_after_sort(VkBuffer keyvals_in,
                                                    VkBuffer keyvals_scratch,
                                                    uint32_t count,
                                                    uint32_t key_bits) const {
    // Fuchsia plumbs the result-buffer parity out through the `keyvals_sorted`
    // parameter on radix_sort_vk_sort; we capture it during record() rather
    // than re-computing pass parity here.
    if (last_sorted_ != VK_NULL_HANDLE) {
        return last_sorted_;
    }
    (void)keyvals_scratch;
    (void)count;
    (void)key_bits;
    return keyvals_in;
}
