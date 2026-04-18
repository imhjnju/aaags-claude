// SP-2 T10: TileBinnerVulkan — scan + scatter half of the tile-binner.
//
// Phase-1 (sync) pipeline, mirroring PreprocessorVulkan::process()'s shape:
//   1. Upload per-Gaussian host arrays to fresh host-visible SSBOs.
//   2. Run PrefixScanPass over tiles_touched → point_offsets (exclusive).
//   3. Download point_offsets[N-1] + tiles_touched[N-1] to recover total
//      pairs R (single short round-trip — the alternative is a 1-byte
//      host-readback of a uint, which is not a measurable cost on the
//      Phase-1 host-visible path).
//   4. Allocate keys_unsorted[R] (uint64) and values_unsorted[R] (uint32)
//      SSBOs, run ScatterPass.
//   5. Download keys_unsorted / values_unsorted into allocator-backed arrays
//      in BinningOutput. Leave sorted / tile_ranges null for SorterVulkan.
//
// SP-2 constraints: tile_w=tile_h=16 (SP-2 spec §4.4). We don't re-check here
// because PreprocessorVulkan::process() has already thrown on any non-16 tile
// in the same frame, but we DO use cfg.tile_w/h to compute the grid so that
// the scatter shader's grid arithmetic matches whatever the caller passed.
//
// A note on `preprocess.tiles_touched`: the CPU-facing PreprocessOutput holds
// it as `int*` (non-negative tile counts). The GLSL shader reads it as `int`
// at binding 4 of scatter, and as `uint` at binding 0 of prefix_sum. The two
// interpretations are bit-identical for non-negative values, so we upload the
// host int* buffer verbatim.

#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_buffer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

TileBinnerVulkan::TileBinnerVulkan(VulkanContext& ctx)
    : ctx_(ctx) {
    scan_pass_    = std::make_unique<PrefixScanPass>(ctx_);
    scatter_pass_ = std::make_unique<ScatterPass>(ctx_);
}

// Out-of-line so unique_ptr<PrefixScanPass/ScatterPass> can see the complete
// types from tile_binner_passes.h (header uses forward-declaration style via
// the include).
TileBinnerVulkan::~TileBinnerVulkan() = default;

BinningOutput TileBinnerVulkan::bin(const PreprocessOutput& pre,
                                    int num_gaussians,
                                    const Camera& cam,
                                    const RenderConfig& cfg,
                                    FrameAllocator& alloc) {
    BinningOutput out{};

    const int N = num_gaussians;
    const uint32_t num_tiles_x =
        static_cast<uint32_t>((cam.width  + cfg.tile_w - 1) / cfg.tile_w);
    const uint32_t num_tiles_y =
        static_cast<uint32_t>((cam.height + cfg.tile_h - 1) / cfg.tile_h);
    out.num_tiles = static_cast<int>(num_tiles_x * num_tiles_y);

    // Empty-scene fast path: match TileBinnerCPU's handling (zero pairs,
    // tile_ranges is zero-initialized and allocated for downstream).
    if (N <= 0) {
        out.total_pairs     = 0;
        out.keys_unsorted   = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted     = nullptr;
        out.values_sorted   = nullptr;
        out.tile_ranges     = nullptr;  // left to SorterVulkan (T14)
        return out;
    }

    // -------------------------------------------------------------------
    // 1. Allocate + upload input SSBOs.
    // -------------------------------------------------------------------
    // tiles_touched: host int* (non-negative), used by prefix scan as uint[]
    // AND by scatter as int[]. Bit layout is identical, so one buffer works.
    const VkDeviceSize bytes_int_N    = static_cast<VkDeviceSize>(N) * sizeof(int32_t);
    const VkDeviceSize bytes_uint_N   = static_cast<VkDeviceSize>(N) * sizeof(uint32_t);
    const VkDeviceSize bytes_float_N  = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_float_N2 = static_cast<VkDeviceSize>(N) * 2 * sizeof(float);

    auto tt_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_int_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto po_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_uint_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // workgroup_sums[ceil(N/256)], min 1. Phase-1 scan writes one slot per
    // phase-0 workgroup; phase-1 scans it in a single 256-thread WG in-place.
    const uint32_t num_wgs =
        (static_cast<uint32_t>(N) + 255u) / 256u;
    const uint32_t wg_sums_count = num_wgs == 0u ? 1u : num_wgs;
    auto ws_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(wg_sums_count) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    tt_buf->upload(pre.tiles_touched, static_cast<std::size_t>(bytes_int_N));

    // -------------------------------------------------------------------
    // 2. Exclusive prefix scan: tiles_touched -> point_offsets.
    // -------------------------------------------------------------------
    scan_pass_->bind_buffers(tt_buf->handle(),
                             po_buf->handle(),
                             ws_buf->handle());
    scan_pass_->scan_sync(static_cast<uint32_t>(N));

    // -------------------------------------------------------------------
    // 3. Compute R (total pairs) = point_offsets[N-1] + tiles_touched[N-1].
    // -------------------------------------------------------------------
    uint32_t last_offset = 0u;
    uint32_t last_count  = 0u;
    po_buf->download(&last_offset,
                     sizeof(uint32_t),
                     /*offset=*/static_cast<VkDeviceSize>(N - 1) * sizeof(uint32_t));
    // tiles_touched is stored as int32 but non-negative; read as uint32.
    {
        int32_t tmp = 0;
        tt_buf->download(&tmp,
                         sizeof(int32_t),
                         static_cast<VkDeviceSize>(N - 1) * sizeof(int32_t));
        if (tmp < 0)
            throw std::runtime_error(
                "TileBinnerVulkan: tiles_touched[N-1] is negative — preprocess bug?");
        last_count = static_cast<uint32_t>(tmp);
    }
    const uint32_t R = last_offset + last_count;
    out.total_pairs  = static_cast<int>(R);

    if (R == 0u) {
        // No Gaussian touches any tile — all culled. Nothing for scatter to
        // write. Return with null pair arrays; downstream sorter (T14) will
        // handle the zero-pair case.
        out.keys_unsorted   = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted     = nullptr;
        out.values_sorted   = nullptr;
        out.tile_ranges     = nullptr;
        return out;
    }

    // -------------------------------------------------------------------
    // 4. Allocate scatter input SSBOs (means2D, depths, radii) + output
    //    SSBOs (keys_unsorted[R], values_unsorted[R]), upload, dispatch.
    // -------------------------------------------------------------------
    auto m2d_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_float_N2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dep_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_float_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rad_buf = std::make_unique<VulkanBuffer>(
        ctx_, bytes_int_N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    auto keys_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(R) * sizeof(uint64_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto vals_buf = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(R) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    m2d_buf->upload(pre.means2D, static_cast<std::size_t>(bytes_float_N2));
    dep_buf->upload(pre.depths,  static_cast<std::size_t>(bytes_float_N));
    rad_buf->upload(pre.radii,   static_cast<std::size_t>(bytes_int_N));

    ScatterPass::Buffers sb{};
    sb.means2D         = m2d_buf->handle();
    sb.depths          = dep_buf->handle();
    sb.radii           = rad_buf->handle();
    sb.point_offsets   = po_buf ->handle();
    sb.tiles_touched   = tt_buf ->handle();
    sb.keys_unsorted   = keys_buf->handle();
    sb.values_unsorted = vals_buf->handle();
    scatter_pass_->bind_buffers(sb);
    scatter_pass_->dispatch_sync(static_cast<uint32_t>(N),
                                 num_tiles_x,
                                 num_tiles_y);

    // -------------------------------------------------------------------
    // 5. Download pairs into FrameAllocator-backed output.
    // -------------------------------------------------------------------
    out.keys_unsorted   = alloc.allocate_array<uint64_t>(R);
    out.values_unsorted = alloc.allocate_array<uint32_t>(R);
    out.keys_sorted     = nullptr;   // SorterVulkan (T14)
    out.values_sorted   = nullptr;
    out.tile_ranges     = nullptr;

    keys_buf->download(out.keys_unsorted,
                       static_cast<std::size_t>(R) * sizeof(uint64_t));
    vals_buf->download(out.values_unsorted,
                       static_cast<std::size_t>(R) * sizeof(uint32_t));

    return out;
}
