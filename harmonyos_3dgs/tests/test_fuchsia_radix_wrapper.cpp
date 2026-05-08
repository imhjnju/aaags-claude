// test_fuchsia_radix_wrapper.cpp -- Phase 2.2 TDD gate for RadixSortFuchsia.
//
// What we test (per Plan §6.2):
//   1. Construct + destruct succeed against a real device — pipeline compile
//      round-trips, RAII destroy releases all library state.
//   2. Sort 1M random u64 keyvals through RadixSortFuchsia::record() driven
//      by a custom command buffer, assert monotonic AND bit-exact equality
//      with std::sort on the same input. (LSD radix over the full 64 bits is
//      stable in the trivial sense that equal keyvals are bit-equal.)
//   3. memory_requirements() returns positive sizes and power-of-two
//      alignments at several counts that span our production workload
//      (1024, 100k, 1<<20).
//
// We allocate raw VkBuffers + VkDeviceMemory directly (bypassing the
// host-visible-coherent VulkanBuffer abstraction) because Fuchsia requires
// DEVICE_LOCAL + SHADER_DEVICE_ADDRESS_BIT, which VulkanBuffer does not
// expose. Pattern mirrors `spike/fuchsia_radix_sort/bench_radix.cpp:51-89`.
//
// Skip semantics: GTEST_SKIP if VulkanContext::init() returns false (no
// device with Vulkan-1.2 + the four required core features). With Phase 2.2
// VulkanContext changes, that means GTEST_SKIP if the first 234 tests would
// also have skipped. Anything past init succeeding is expected to pass.

#include "vulkan/vk_context.h"
#include "vulkan/radix_sort_fuchsia.h"

#include <gtest/gtest.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ----- Raw VkBuffer + VkDeviceMemory helpers (mirrors the spike). ---------

struct RawBuf {
    VkDevice       dev = VK_NULL_HANDLE;
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    void*          mapped = nullptr;

    ~RawBuf() {
        if (dev != VK_NULL_HANDLE) {
            if (mapped) vkUnmapMemory(dev, mem);
            if (buf != VK_NULL_HANDLE) vkDestroyBuffer(dev, buf, nullptr);
            if (mem != VK_NULL_HANDLE) vkFreeMemory(dev, mem, nullptr);
        }
    }
};

uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t type_bits,
                       VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

bool make_raw_buffer(VulkanContext& ctx, VkDeviceSize size,
                     VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags mem_flags,
                     RawBuf& out) {
    out.dev  = ctx.device();
    out.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device(), &bci, nullptr, &out.buf) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), out.buf, &req);

    VkMemoryAllocateFlagsInfo flags_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    uint32_t mem_idx = find_mem_type(ctx.physicalDevice(),
                                     req.memoryTypeBits, mem_flags);
    if (mem_idx == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = (flags_info.flags ? &flags_info : nullptr);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = mem_idx;
    if (vkAllocateMemory(ctx.device(), &mai, nullptr, &out.mem) != VK_SUCCESS)
        return false;
    if (vkBindBufferMemory(ctx.device(), out.buf, out.mem, 0) != VK_SUCCESS)
        return false;
    if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(ctx.device(), out.mem, 0, VK_WHOLE_SIZE, 0,
                        &out.mapped) != VK_SUCCESS)
            return false;
    }
    return true;
}

// True iff x is a power of two (and nonzero).
bool is_pow2(VkDeviceSize x) { return x != 0 && (x & (x - 1)) == 0; }

}  // namespace

// =============================================================================
// 1. Construct + destruct the wrapper, no validation errors expected.
// =============================================================================
TEST(FuchsiaRadixWrapper, ConstructDestruct) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "skip: no Vulkan 1.2 device with Fuchsia features";
    }
    {
        RadixSortFuchsia rs(ctx, /*max_keyvals=*/1u << 20);
        SUCCEED() << "RadixSortFuchsia constructed on '"
                  << ctx.deviceName() << "' (max_keyvals=" << (1u<<20) << ")";
    }
    // RAII destroy on scope exit — vkDeviceWaitIdle runs on context release.
}

// =============================================================================
// 3. memory_requirements sanity at several counts.
// =============================================================================
TEST(FuchsiaRadixWrapper, MemoryRequirementsSanity) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "skip: no Vulkan 1.2 device with Fuchsia features";
    }
    RadixSortFuchsia rs(ctx, /*max_keyvals=*/1u << 20);

    for (uint32_t count : {1024u, 100000u, 1u << 20}) {
        auto mr = rs.memory_requirements(count);
        EXPECT_EQ(mr.keyval_size, 8u)
            << "expected 8-byte keyvals (keyval_dwords=2 in pick_target)";
        EXPECT_GT(mr.keyvals_size, 0u)
            << "count=" << count << ": keyvals_size must be positive";
        EXPECT_GE(mr.keyvals_size,
                  static_cast<VkDeviceSize>(count) * mr.keyval_size)
            << "count=" << count
            << ": keyvals_size must hold at least count*keyval_size";
        EXPECT_GT(mr.internal_size, 0u)
            << "count=" << count << ": internal_size must be positive";
        EXPECT_TRUE(is_pow2(mr.keyvals_alignment))
            << "count=" << count
            << ": keyvals_alignment must be power of two ("
            << mr.keyvals_alignment << ")";
        EXPECT_TRUE(is_pow2(mr.internal_alignment))
            << "count=" << count
            << ": internal_alignment must be power of two ("
            << mr.internal_alignment << ")";
    }
}

// =============================================================================
// 2. Sort 1M random u64 keyvals; assert monotonic + bit-exact match vs std::sort.
// =============================================================================
TEST(FuchsiaRadixWrapper, Sort1MKeyvalsMonotonicAndBitExact) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "skip: no Vulkan 1.2 device with Fuchsia features";
    }

    constexpr uint32_t KEY_COUNT = 1u << 20;       // 1,048,576
    constexpr uint32_t KEY_BITS  = 64;             // sort full keyvals
    constexpr uint64_t RNG_SEED  = 0xA1B2C3D4E5F60718ULL;  // matches spike

    RadixSortFuchsia rs(ctx, KEY_COUNT);
    auto mr = rs.memory_requirements(KEY_COUNT);

    // -- Allocate the three sort buffers (DEVICE_LOCAL + DEVICE_ADDRESS) ----
    const VkBufferUsageFlags kv_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags int_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    RawBuf kv_in, kv_scratch, internal;
    ASSERT_TRUE(make_raw_buffer(ctx, mr.keyvals_size, kv_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, kv_in));
    ASSERT_TRUE(make_raw_buffer(ctx, mr.keyvals_size, kv_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                kv_scratch));
    ASSERT_TRUE(make_raw_buffer(ctx, mr.internal_size, int_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                internal));

    // Host-visible staging for upload + download.
    const VkDeviceSize host_bytes = (VkDeviceSize)KEY_COUNT * 8u;
    RawBuf staging_up, staging_dn;
    ASSERT_TRUE(make_raw_buffer(ctx, host_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                staging_up));
    ASSERT_TRUE(make_raw_buffer(ctx, host_bytes,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                staging_dn));

    // -- Generate input + std::sort reference -------------------------------
    std::mt19937_64 rng(RNG_SEED);
    std::vector<uint64_t> input(KEY_COUNT);
    for (auto& x : input) x = rng();
    std::vector<uint64_t> reference = input;
    std::sort(reference.begin(), reference.end());

    std::memcpy(staging_up.mapped, input.data(), host_bytes);

    // -- Record + submit ----------------------------------------------------
    VkCommandBuffer cmd = ctx.allocatePrimary();
    ASSERT_NE(cmd, VK_NULL_HANDLE);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ASSERT_EQ(vkBeginCommandBuffer(cmd, &bi), VK_SUCCESS);

    // staging_up -> kv_in
    VkBufferCopy copy_up{0, 0, host_bytes};
    vkCmdCopyBuffer(cmd, staging_up.buf, kv_in.buf, 1, &copy_up);

    // Barrier: TRANSFER_WRITE -> COMPUTE_READ on kv_in (the sort reads it
    // first thing). The library's own internal barriers handle inter-pass
    // dependencies.
    VkBufferMemoryBarrier mb_pre{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    mb_pre.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb_pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT;
    mb_pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb_pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb_pre.buffer = kv_in.buf;
    mb_pre.offset = 0;
    mb_pre.size   = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &mb_pre, 0, nullptr);

    rs.record(cmd, kv_in.buf, kv_scratch.buf, internal.buf,
              KEY_COUNT, KEY_BITS);

    // Find the sorted output buffer (Fuchsia ping-pongs between kv_in and
    // kv_scratch). For a 64-bit sort the pass count is even ⇒ kv_in, but we
    // honor the wrapper API instead of hardcoding parity.
    VkBuffer sorted_buf =
        rs.sorted_buffer_after_sort(kv_in.buf, kv_scratch.buf,
                                    KEY_COUNT, KEY_BITS);
    ASSERT_TRUE(sorted_buf == kv_in.buf || sorted_buf == kv_scratch.buf);

    // Barrier: COMPUTE_WRITE -> TRANSFER_READ on the sorted buffer.
    VkBufferMemoryBarrier mb_post{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    mb_post.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
    mb_post.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    mb_post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb_post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb_post.buffer = sorted_buf;
    mb_post.offset = 0;
    mb_post.size   = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &mb_post, 0, nullptr);

    VkBufferCopy copy_dn{0, 0, host_bytes};
    vkCmdCopyBuffer(cmd, sorted_buf, staging_dn.buf, 1, &copy_dn);

    ASSERT_EQ(vkEndCommandBuffer(cmd), VK_SUCCESS);
    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);

    // -- Read back + verify -------------------------------------------------
    std::vector<uint64_t> got(KEY_COUNT);
    std::memcpy(got.data(), staging_dn.mapped, host_bytes);

    // Monotonic.
    bool monotonic = true;
    size_t bad_idx = 0;
    for (size_t i = 1; i < got.size(); ++i) {
        if (got[i] < got[i-1]) { monotonic = false; bad_idx = i; break; }
    }
    ASSERT_TRUE(monotonic)
        << "non-monotonic at i=" << bad_idx
        << ": got[i-1]=" << got[bad_idx-1]
        << " got[i]=" << got[bad_idx];

    // Bit-exact match with std::sort. LSD radix on full 64-bit keyvals
    // produces a unique sorted permutation when keys are unique-with-prob-1
    // (mt19937_64 over 1M draws → collision prob ~5.7e-8).
    EXPECT_EQ(got, reference)
        << "Fuchsia sort disagrees with std::sort on 1M random u64 keyvals";
}
