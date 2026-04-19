// SP-2 T14: SorterVulkan — drives RadixSortPass + TileRangePass to produce
// sorted (key, value) pairs and per-tile ranges from the unsorted pairs that
// TileBinnerVulkan produced.
//
// Phase-1 shape (mirrors TileBinnerVulkan::bin()):
//   1. Allocate fresh host-visible SSBOs for this sort call:
//        - two key/value ping-pong pairs (A, B), each sized R elements
//        - hist_count[16], hist_scan[16], wg_sums[16]      (inner scan)
//        - tile_ranges[num_tiles*2]                        (range output)
//   2. Upload binning.keys_unsorted / values_unsorted to A.
//   3. Run RadixSortPass: 16 passes ⇒ sorted result returns to A.
//   4. Zero-initialise tile_ranges (precondition of tile_range.comp).
//   5. Run TileRangePass over the sorted keys in A.
//   6. Download sorted keys/values from A and tile_ranges into
//      FrameAllocator-backed output arrays on BinningOutput.
//
// Empty-scene and R==0 fast paths match TileBinnerCPU: leave the sorted
// pointers null; allocate a zero-filled tile_ranges so downstream code can
// iterate tiles unconditionally.

#include "vulkan/sorter_vulkan.h"
#include "vulkan/vk_buffer.h"
#include "types.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// CPU fallback sort — used when R > kRadixLocalSize (256).
// The Phase-1 radix sort is single-workgroup (256 threads), so it cannot
// handle more than 256 elements. For large scenes (e.g. basketball 2892 pts)
// total_pairs easily exceeds 256. This fallback uses std::sort on the CPU
// and is functionally identical to SorterCPU::sort().
// ---------------------------------------------------------------------------
static void sort_cpu_fallback(BinningOutput& bin, FrameAllocator& alloc)
{
    const int R         = bin.total_pairs;
    const int num_tiles = bin.num_tiles;

    // Allocate output arrays from the frame allocator (same lifetime as Vulkan path).
    bin.keys_sorted   = alloc.allocate_array<uint64_t>(static_cast<size_t>(R));
    bin.values_sorted = alloc.allocate_array<uint32_t>(static_cast<size_t>(R));
    bin.tile_ranges   = alloc.allocate_array<uint32_t>(
                            static_cast<size_t>(num_tiles) * 2u);
    std::memset(bin.tile_ranges, 0,
                static_cast<size_t>(num_tiles) * 2u * sizeof(uint32_t));

    // Stable-sort an index array by key, then scatter.
    uint32_t* indices = alloc.allocate_array<uint32_t>(static_cast<size_t>(R));
    std::iota(indices, indices + R, 0u);
    std::stable_sort(indices, indices + R, [&](uint32_t a, uint32_t b) {
        return bin.keys_unsorted[a] < bin.keys_unsorted[b];
    });
    for (int i = 0; i < R; ++i) {
        bin.keys_sorted[i]   = bin.keys_unsorted[indices[i]];
        bin.values_sorted[i] = bin.values_unsorted[indices[i]];
    }

    // Compute tile ranges from sorted keys (upper 32 bits = tile index).
    for (int i = 0; i < R; ++i) {
        const uint32_t cur_tile = static_cast<uint32_t>(bin.keys_sorted[i] >> 32);
        if (i == 0) {
            bin.tile_ranges[cur_tile * 2u] = 0u;
        } else {
            const uint32_t prev_tile = static_cast<uint32_t>(bin.keys_sorted[i - 1] >> 32);
            if (cur_tile != prev_tile) {
                bin.tile_ranges[prev_tile * 2u + 1u] = static_cast<uint32_t>(i);
                bin.tile_ranges[cur_tile  * 2u]      = static_cast<uint32_t>(i);
            }
        }
        if (i == R - 1) {
            bin.tile_ranges[cur_tile * 2u + 1u] = static_cast<uint32_t>(R);
        }
    }
}

SorterVulkan::SorterVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    sort_pass_  = std::make_unique<RadixSortPass>(ctx_);
    range_pass_ = std::make_unique<TileRangePass>(ctx_);
}

// Out-of-line dtor so unique_ptr<RadixSortPass/TileRangePass> can see the
// complete types from sort_passes.h at destructor-emit time.
SorterVulkan::~SorterVulkan() = default;

void SorterVulkan::sort(BinningOutput& bin, FrameAllocator& alloc) {
    const int R         = bin.total_pairs;
    const int num_tiles = bin.num_tiles;

    // ---- Empty-scene fast path -----------------------------------------
    // R == 0 means tile-binner saw no Gaussian-tile overlaps. The forward
    // pipeline still needs a tile_ranges array of the correct length, fully
    // zeroed (every tile is empty). Match SorterCPU: allocate + zero.
    if (R == 0) {
        bin.keys_sorted   = nullptr;
        bin.values_sorted = nullptr;
        if (num_tiles > 0) {
            bin.tile_ranges = alloc.allocate_array<uint32_t>(
                static_cast<size_t>(num_tiles) * 2u);
            std::memset(bin.tile_ranges, 0,
                        static_cast<size_t>(num_tiles) * 2u * sizeof(uint32_t));
        } else {
            bin.tile_ranges = nullptr;
        }
        return;
    }

    if (R < 0)
        throw std::runtime_error("SorterVulkan::sort: negative total_pairs");
    if (num_tiles <= 0)
        throw std::runtime_error(
            "SorterVulkan::sort: num_tiles must be > 0 when R > 0");
    if (!bin.keys_unsorted || !bin.values_unsorted)
        throw std::runtime_error(
            "SorterVulkan::sort: keys_unsorted/values_unsorted is null");

    // ---- Large-input CPU fallback -----------------------------------------
    // Phase-1 radix sort is single-workgroup (256 threads) and cannot handle
    // R > 256. For large scenes (e.g. basketball, thousands of (tile,Gaussian)
    // pairs) fall back to CPU std::stable_sort which is always correct.
    constexpr int kRadixLocalSize = 256;
    if (R > kRadixLocalSize) {
        sort_cpu_fallback(bin, alloc);
        return;
    }

    const uint32_t R_u = static_cast<uint32_t>(R);

    // -------------------------------------------------------------------
    // 1. Allocate GPU buffers (host-visible, Phase 1 style).
    // -------------------------------------------------------------------
    const VkDeviceSize bytes_keys    = static_cast<VkDeviceSize>(R_u) * sizeof(uint64_t);
    const VkDeviceSize bytes_vals    = static_cast<VkDeviceSize>(R_u) * sizeof(uint32_t);
    const VkDeviceSize bytes_hist    = 16u * sizeof(uint32_t);
    const VkDeviceSize bytes_ranges  =
        static_cast<VkDeviceSize>(num_tiles) * 2u * sizeof(uint32_t);

    auto keys_a    = std::make_unique<VulkanBuffer>(ctx_, bytes_keys,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto vals_a    = std::make_unique<VulkanBuffer>(ctx_, bytes_vals,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto keys_b    = std::make_unique<VulkanBuffer>(ctx_, bytes_keys,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto vals_b    = std::make_unique<VulkanBuffer>(ctx_, bytes_vals,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hist_cnt  = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto hist_scn  = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // workgroup_sums for the 16-element inner scan: 1 uint is enough (num_wg=1
    // when N=16, since ceil(16/256)=1). Allocate 16 to be safely over-sized.
    auto wg_sums   = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ranges_b  = std::make_unique<VulkanBuffer>(ctx_, bytes_ranges,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // -------------------------------------------------------------------
    // 2. Upload unsorted keys/values to A.
    // -------------------------------------------------------------------
    keys_a->upload(bin.keys_unsorted,   static_cast<std::size_t>(bytes_keys));
    vals_a->upload(bin.values_unsorted, static_cast<std::size_t>(bytes_vals));

    // -------------------------------------------------------------------
    // 3. 16-pass LSD radix sort. After 16 (even) passes the sorted data
    //    ends up in keys_a / vals_a (the "in" side).
    // -------------------------------------------------------------------
    sort_pass_->sort_sync(keys_a->handle(),   vals_a->handle(),
                          keys_b->handle(),   vals_b->handle(),
                          hist_cnt->handle(), hist_scn->handle(),
                          wg_sums->handle(),
                          R_u);

    // -------------------------------------------------------------------
    // 4. Zero-initialise tile_ranges (tile_range.comp precondition).
    //    Tiles with no Gaussians are never written by the shader and must
    //    read back as [0, 0).
    // -------------------------------------------------------------------
    {
        std::vector<uint32_t> zeros(static_cast<size_t>(num_tiles) * 2u, 0u);
        ranges_b->upload(zeros.data(),
                         zeros.size() * sizeof(uint32_t));
    }

    // -------------------------------------------------------------------
    // 5. TileRangePass over sorted keys (in keys_a).
    // -------------------------------------------------------------------
    range_pass_->bind_buffers(keys_a->handle(), ranges_b->handle());
    range_pass_->dispatch_sync(R_u, static_cast<uint32_t>(num_tiles));

    // -------------------------------------------------------------------
    // 6. Download to FrameAllocator-backed output arrays.
    // -------------------------------------------------------------------
    bin.keys_sorted   = alloc.allocate_array<uint64_t>(R_u);
    bin.values_sorted = alloc.allocate_array<uint32_t>(R_u);
    bin.tile_ranges   = alloc.allocate_array<uint32_t>(
        static_cast<size_t>(num_tiles) * 2u);

    keys_a->download(bin.keys_sorted,   static_cast<std::size_t>(bytes_keys));
    vals_a->download(bin.values_sorted, static_cast<std::size_t>(bytes_vals));
    ranges_b->download(bin.tile_ranges, static_cast<std::size_t>(bytes_ranges));
}

// ---------------------------------------------------------------------------
// Layer-2 record-mode.
// ---------------------------------------------------------------------------
void SorterVulkan::prepare_record(uint32_t R, uint32_t num_tiles,
                                  VkBuffer keys_unsorted,
                                  VkBuffer values_unsorted) {
    if (R == 0u)
        throw std::runtime_error(
            "SorterVulkan::prepare_record: R must be > 0");
    if (num_tiles == 0u)
        throw std::runtime_error(
            "SorterVulkan::prepare_record: num_tiles must be > 0");
    if (keys_unsorted == VK_NULL_HANDLE || values_unsorted == VK_NULL_HANDLE)
        throw std::runtime_error(
            "SorterVulkan::prepare_record: keys/values_unsorted is null");

    // Release old ones.
    r_keys_b_.reset();
    r_vals_b_.reset();
    r_hist_cnt_.reset();
    r_hist_scn_.reset();
    r_wg_sums_.reset();
    r_ranges_.reset();

    const VkDeviceSize bytes_keys   = static_cast<VkDeviceSize>(R) * sizeof(uint64_t);
    const VkDeviceSize bytes_vals   = static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_hist   = 16u * sizeof(uint32_t);
    const VkDeviceSize bytes_ranges =
        static_cast<VkDeviceSize>(num_tiles) * 2u * sizeof(uint32_t);

    r_keys_b_   = std::make_unique<VulkanBuffer>(ctx_, bytes_keys,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_vals_b_   = std::make_unique<VulkanBuffer>(ctx_, bytes_vals,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_hist_cnt_ = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_hist_scn_ = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // workgroup sums for the 16-element inner scan — 1 uint suffices; size 16
    // to match the other histogram buffers (over-allocation is negligible).
    r_wg_sums_  = std::make_unique<VulkanBuffer>(ctx_, bytes_hist,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_ranges_   = std::make_unique<VulkanBuffer>(ctx_, bytes_ranges,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // tile_range.comp precondition: tile_ranges must be zero on dispatch so
    // empty tiles read back as [0, 0). Uploading zeros satisfies this now —
    // the command buffer will execute after prepare_record() returns, so the
    // GPU-visible contents at dispatch time are our zeros.
    std::vector<uint32_t> zeros(static_cast<size_t>(num_tiles) * 2u, 0u);
    r_ranges_->upload(zeros.data(), zeros.size() * sizeof(uint32_t));

    // Stash the external "A" handles. No ownership transfer — caller keeps
    // the binner's buffers alive across record + submit.
    r_keys_a_ = keys_unsorted;
    r_vals_a_ = values_unsorted;

    // Bind TileRangePass descriptor sets here (pure DS mutation). record()
    // is "pure" — only GPU command recording. All vkUpdateDescriptorSets
    // calls must happen before record() is invoked.
    range_pass_->bind_buffers(r_keys_a_, r_ranges_->handle());
}

void SorterVulkan::record(VkCommandBuffer cmd,
                          uint32_t R, uint32_t num_tiles) {
    if (!r_keys_b_)
        throw std::runtime_error(
            "SorterVulkan::record called before prepare_record()");

    // 16-pass LSD radix sort — sorted data lands back in the A side.
    sort_pass_->sort_record(cmd,
        r_keys_a_,           r_vals_a_,
        r_keys_b_->handle(), r_vals_b_->handle(),
        r_hist_cnt_->handle(), r_hist_scn_->handle(),
        r_wg_sums_->handle(),
        R);

    // Barrier: tile_range.comp reads keys_sorted (which is r_keys_a_).
    insert_compute_barrier(cmd);

    // Tile-range sweep over sorted keys (A side).
    // bind_buffers() was already called in prepare_record() — record() is pure.
    range_pass_->dispatch_record(cmd, R, num_tiles);
}

VkBuffer SorterVulkan::keys_sorted_buf()   const { return r_keys_a_; }
VkBuffer SorterVulkan::values_sorted_buf() const { return r_vals_a_; }
VkBuffer SorterVulkan::tile_ranges_buf()   const {
    return r_ranges_ ? r_ranges_->handle() : VK_NULL_HANDLE;
}
