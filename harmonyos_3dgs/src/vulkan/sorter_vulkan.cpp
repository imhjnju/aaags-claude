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
#include "vulkan/vk_aligned_buffer.h"
#include "vulkan/radix_sort_fuchsia.h"
#include "types.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

static uint64_t make_packed_keyval(uint64_t legacy_key, uint32_t gauss_idx)
{
    const uint32_t tile_id = static_cast<uint32_t>(legacy_key >> 32);
    const uint32_t depth_bits = static_cast<uint32_t>(legacy_key & 0xFFFFFFFFu);
    return keyval_pack::pack(tile_id, keyval_pack::quantize_depth(depth_bits), gauss_idx);
}

static void populate_packed_keyvals(BinningOutput& bin, FrameAllocator& alloc)
{
    const int R = bin.total_pairs;
    if (R <= 0 || !bin.keys_sorted || !bin.values_sorted) {
        bin.keyvals_sorted = nullptr;
        return;
    }
    bin.keyvals_sorted = alloc.allocate_array<uint64_t>(static_cast<size_t>(R));
    for (int i = 0; i < R; ++i) {
        bin.keyvals_sorted[i] = make_packed_keyval(bin.keys_sorted[i], bin.values_sorted[i]);
    }
}

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
    populate_packed_keyvals(bin, alloc);

    // Compute tile ranges from sorted keys (upper 32 bits = tile index).
    // Guard: INVALID sentinel keys have tile_id=0xFFFFFFFF (>= num_tiles);
    // skip them to avoid OOB writes. Matches tile_range.comp line 45.
    // INVALID keys sort to END (highest key), so valid entries are at [0, first_invalid).
    const uint32_t num_tiles_u = static_cast<uint32_t>(num_tiles);
    for (int i = 0; i < R; ++i) {
        const uint32_t cur_tile = static_cast<uint32_t>(bin.keys_sorted[i] >> 32);
        if (cur_tile >= num_tiles_u) break; // rest are INVALID sentinels, all >= num_tiles

        // Detect left boundary (start of new tile run).
        const bool is_start = (i == 0) ||
            (static_cast<uint32_t>(bin.keys_sorted[i - 1] >> 32) != cur_tile);
        // Detect right boundary (end of tile run).
        const bool is_end = (i == R - 1) ||
            (static_cast<uint32_t>(bin.keys_sorted[i + 1] >> 32) != cur_tile);

        if (is_start) bin.tile_ranges[cur_tile * 2u]      = static_cast<uint32_t>(i);
        if (is_end)   bin.tile_ranges[cur_tile * 2u + 1u] = static_cast<uint32_t>(i + 1);
    }
}

SorterVulkan::SorterVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    sort_pass_  = std::make_unique<RadixSortPass>(ctx_);
    range_pass_ = std::make_unique<TileRangePass>(ctx_);
    pair_pack_pass_ = std::make_unique<CanonicalSortPairPackPass>(ctx_);
    pair_extract_pass_ = std::make_unique<CanonicalSortPairExtractPass>(ctx_);
}

// Out-of-line dtor so unique_ptr<RadixSortPass/TileRangePass> can see the
// complete types from sort_passes.h at destructor-emit time.
SorterVulkan::~SorterVulkan() = default;

bool SorterVulkan::fuchsia_sort_enabled() {
    const char* env = std::getenv("GS3D_USE_FUCHSIA_SORT");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool SorterVulkan::sort_via_fuchsia_sortpairs(BinningOutput& bin, FrameAllocator& alloc) {
    const uint32_t R = static_cast<uint32_t>(bin.total_pairs);
    const uint32_t num_tiles = static_cast<uint32_t>(bin.num_tiles);
    constexpr uint32_t kRecordDwords = 3u;
    const bool compact_key48 = num_tiles <= 65536u;
    const uint32_t key_bits = compact_key48 ? 48u : 64u;
    const VkDeviceSize bytes_keys = static_cast<VkDeviceSize>(R) * sizeof(uint64_t);
    const VkDeviceSize bytes_values = static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_ranges = static_cast<VkDeviceSize>(num_tiles) * 2u * sizeof(uint32_t);

    try {
        if (!fuchsia_pairs_ || R > f_record_capacity_) {
            fuchsia_pairs_ = std::make_unique<RadixSortFuchsia>(ctx_, R, kRecordDwords);
            f_record_capacity_ = R;
            f_record_bytes_capacity_ = 0;
            f_internal_bytes_capacity_ = 0;
        }

        const auto mr = fuchsia_pairs_->memory_requirements(R);
        const VkBufferUsageFlags record_usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!f_records_even_ || mr.keyvals_size > f_record_bytes_capacity_) {
            f_records_even_ = std::make_unique<VulkanAlignedBuffer>(
                ctx_, mr.keyvals_size, mr.keyvals_alignment, record_usage);
            f_records_odd_ = std::make_unique<VulkanAlignedBuffer>(
                ctx_, mr.keyvals_size, mr.keyvals_alignment, record_usage);
            f_record_bytes_capacity_ = mr.keyvals_size;
        }
        if (!f_internal_scratch_ || mr.internal_size > f_internal_bytes_capacity_) {
            f_internal_scratch_ = std::make_unique<VulkanAlignedBuffer>(
                ctx_, mr.internal_size, mr.internal_alignment,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            f_internal_bytes_capacity_ = mr.internal_size;
        }
        if (!f_keys_sorted_ || R > f_output_capacity_) {
            f_keys_sorted_ = std::make_unique<VulkanBuffer>(
                ctx_, bytes_keys, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            f_values_sorted_ = std::make_unique<VulkanBuffer>(
                ctx_, bytes_values, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            f_output_capacity_ = R;
        }
        if (!f_ranges_ || num_tiles > f_range_capacity_) {
            f_ranges_ = std::make_unique<VulkanBuffer>(
                ctx_, bytes_ranges, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            f_range_capacity_ = num_tiles;
        }
        f_ranges_->zero_fill(static_cast<std::size_t>(bytes_ranges));

        std::unique_ptr<VulkanBuffer> keys_in;
        std::unique_ptr<VulkanBuffer> values_in;
        VkBuffer keys_in_handle = VK_NULL_HANDLE;
        VkBuffer values_in_handle = VK_NULL_HANDLE;
        if (bin.keys_unsorted_gpu && bin.values_unsorted_gpu) {
            keys_in_handle = static_cast<VkBuffer>(bin.keys_unsorted_gpu);
            values_in_handle = static_cast<VkBuffer>(bin.values_unsorted_gpu);
        } else {
            if (!bin.keys_unsorted || !bin.values_unsorted) {
                throw std::runtime_error("SorterVulkan::sort: canonical SortPairs inputs are null");
            }
            keys_in = std::make_unique<VulkanBuffer>(
                ctx_, bytes_keys, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            values_in = std::make_unique<VulkanBuffer>(
                ctx_, bytes_values, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            keys_in->upload(bin.keys_unsorted, static_cast<std::size_t>(bytes_keys));
            values_in->upload(bin.values_unsorted, static_cast<std::size_t>(bytes_values));
            keys_in_handle = keys_in->handle();
            values_in_handle = values_in->handle();
        }

        VkCommandBuffer cmd = ctx_.allocatePrimary();
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        pair_pack_pass_->bind_buffers(keys_in_handle, values_in_handle, f_records_even_->handle());
        pair_pack_pass_->dispatch_record(cmd, R, compact_key48);
        insert_compute_barrier(cmd);

        fuchsia_pairs_->record(cmd,
                               f_records_even_->handle(),
                               f_records_odd_->handle(),
                               f_internal_scratch_->handle(),
                               R,
                               key_bits);
        insert_compute_barrier(cmd);

        VkBuffer sorted_records = fuchsia_pairs_->sorted_buffer_after_sort(
            f_records_even_->handle(), f_records_odd_->handle(), R, key_bits);
        pair_extract_pass_->bind_buffers(sorted_records,
                                         f_keys_sorted_->handle(),
                                         f_values_sorted_->handle(),
                                         f_ranges_->handle());
        pair_extract_pass_->dispatch_record(cmd, R, num_tiles, compact_key48);

        VK_CHECK(vkEndCommandBuffer(cmd));
        ctx_.submitAndWait(cmd);
        ctx_.freePrimary(cmd);

        bin.values_sorted_gpu = reinterpret_cast<void*>(f_values_sorted_->handle());
        bin.tile_ranges_gpu = reinterpret_cast<void*>(f_ranges_->handle());
        bin.keyvals_sorted_gpu = nullptr;

        if (host_mirror_enabled_) {
            bin.keys_sorted = alloc.allocate_array<uint64_t>(R);
            bin.values_sorted = alloc.allocate_array<uint32_t>(R);
            bin.tile_ranges = alloc.allocate_array<uint32_t>(static_cast<size_t>(num_tiles) * 2u);
            f_keys_sorted_->download(bin.keys_sorted, static_cast<std::size_t>(bytes_keys));
            f_values_sorted_->download(bin.values_sorted, static_cast<std::size_t>(bytes_values));
            f_ranges_->download(bin.tile_ranges, static_cast<std::size_t>(bytes_ranges));
            populate_packed_keyvals(bin, alloc);
        } else {
            bin.keys_sorted = nullptr;
            bin.values_sorted = nullptr;
            bin.tile_ranges = nullptr;
            bin.keyvals_sorted = nullptr;
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

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

    const bool has_host_input = bin.keys_unsorted && bin.values_unsorted;
    const bool has_gpu_input = bin.keys_unsorted_gpu && bin.values_unsorted_gpu;
    if (!has_host_input && !has_gpu_input)
        throw std::runtime_error(
            "SorterVulkan::sort: canonical SortPairs inputs are null");

    const uint32_t R_u = static_cast<uint32_t>(R);

    if (sort_via_fuchsia_sortpairs(bin, alloc)) {
        return;
    }
    if (!has_host_input)
        throw std::runtime_error(
            "SorterVulkan::sort: CPU fallback requires host keys_unsorted/values_unsorted");

    // -------------------------------------------------------------------
    // 1. Allocate GPU buffers (host-visible, Phase 1 style).
    // -------------------------------------------------------------------
    constexpr uint32_t kRadixLocalSize = 256u;
    constexpr uint32_t kRadixBuckets = 16u;
    const uint32_t sort_wgs = (R_u + kRadixLocalSize - 1u) / kRadixLocalSize;
    const uint32_t hist_entries = sort_wgs * kRadixBuckets;
    const uint32_t scan_wgs = (hist_entries + kRadixLocalSize - 1u) / kRadixLocalSize;
    const uint32_t scan_wgs2 = (scan_wgs + kRadixLocalSize - 1u) / kRadixLocalSize;

    const VkDeviceSize bytes_keys    = static_cast<VkDeviceSize>(R_u) * sizeof(uint64_t);
    const VkDeviceSize bytes_vals    = static_cast<VkDeviceSize>(R_u) * sizeof(uint32_t);
    const VkDeviceSize bytes_hist    = static_cast<VkDeviceSize>(hist_entries) * sizeof(uint32_t);
    const VkDeviceSize bytes_wg_sums = static_cast<VkDeviceSize>(std::max(scan_wgs, 1u)) * sizeof(uint32_t);
    const VkDeviceSize bytes_wg_sums2 = static_cast<VkDeviceSize>(std::max(scan_wgs2, 1u)) * sizeof(uint32_t);
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
    auto wg_sums   = std::make_unique<VulkanBuffer>(ctx_, bytes_wg_sums,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto wg_sums2  = std::make_unique<VulkanBuffer>(ctx_, bytes_wg_sums2,
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
                          wg_sums->handle(),  wg_sums2->handle(),
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
    populate_packed_keyvals(bin, alloc);
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
    r_wg_sums2_.reset();
    r_ranges_.reset();

    constexpr uint32_t kRadixLocalSize = 256u;
    constexpr uint32_t kRadixBuckets = 16u;
    const uint32_t sort_wgs = (R + kRadixLocalSize - 1u) / kRadixLocalSize;
    const uint32_t hist_entries = sort_wgs * kRadixBuckets;
    const uint32_t scan_wgs = (hist_entries + kRadixLocalSize - 1u) / kRadixLocalSize;
    const uint32_t scan_wgs2 = (scan_wgs + kRadixLocalSize - 1u) / kRadixLocalSize;

    const VkDeviceSize bytes_keys   = static_cast<VkDeviceSize>(R) * sizeof(uint64_t);
    const VkDeviceSize bytes_vals   = static_cast<VkDeviceSize>(R) * sizeof(uint32_t);
    const VkDeviceSize bytes_hist   = static_cast<VkDeviceSize>(hist_entries) * sizeof(uint32_t);
    const VkDeviceSize bytes_wg_sums = static_cast<VkDeviceSize>(std::max(scan_wgs, 1u)) * sizeof(uint32_t);
    const VkDeviceSize bytes_wg_sums2 = static_cast<VkDeviceSize>(std::max(scan_wgs2, 1u)) * sizeof(uint32_t);
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
    r_wg_sums_  = std::make_unique<VulkanBuffer>(ctx_, bytes_wg_sums,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_wg_sums2_ = std::make_unique<VulkanBuffer>(ctx_, bytes_wg_sums2,
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
        r_wg_sums_->handle(), r_wg_sums2_->handle(),
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
