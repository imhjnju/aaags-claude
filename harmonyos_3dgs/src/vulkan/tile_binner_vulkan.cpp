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

#include <algorithm>
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

// Allocate/grow the 7 scan-phase buffers (all N-or-wg-sized).
// Scatter output buffers (bin_keys_buf_, bin_vals_buf_) are NOT allocated here;
// prepare_for_scatter() handles them after the scan gives us actual R.
void TileBinnerVulkan::prepare_for_bin(uint32_t N, uint32_t num_wgs) {
    if (N <= bin_N_ && num_wgs <= bin_wg_) return;

    const uint32_t N_new  = std::max(N,       bin_N_);
    const uint32_t wg_new = std::max(num_wgs, bin_wg_);

    bin_tt_buf_  = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(int32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_po_buf_  = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_ws_buf_  = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(wg_new) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_m2d_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * 2u * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_dep_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_rad_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(int32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_rf_buf_  = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(N_new) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    bin_N_ = N_new; bin_wg_ = wg_new;
}

// Grow scatter output buffers to hold at least R pairs.
// Called after the scan gives us actual R — never uses the N*num_tiles overestimate.
void TileBinnerVulkan::prepare_for_scatter(uint64_t R) {
    if (R <= bin_R_max_) return;
    bin_R_max_ = R;  // grow-only; R is already the new max
    bin_keys_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(R) * sizeof(uint64_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bin_vals_buf_ = std::make_unique<VulkanBuffer>(ctx_,
        static_cast<VkDeviceSize>(R) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

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
    // 1. Ensure persistent SSBOs are large enough, then upload inputs.
    // -------------------------------------------------------------------
    // tiles_touched: host int* (non-negative), used by prefix scan as uint[]
    // AND by scatter as int[]. Bit layout is identical, so one buffer works.
    const VkDeviceSize bytes_int_N    = static_cast<VkDeviceSize>(N) * sizeof(int32_t);
    const VkDeviceSize bytes_float_N  = static_cast<VkDeviceSize>(N) * sizeof(float);
    const VkDeviceSize bytes_float_N2 = static_cast<VkDeviceSize>(N) * 2 * sizeof(float);

    // workgroup_sums[ceil(N/256)], min 1.
    const uint32_t num_wgs = (static_cast<uint32_t>(N) + 255u) / 256u;
    // Allocate scan-phase buffers (N-sized). Scatter buffers are deferred to
    // prepare_for_scatter() after we know actual R from the scan.
    prepare_for_bin(static_cast<uint32_t>(N), std::max(num_wgs, 1u));

    bin_tt_buf_->upload(pre.tiles_touched, static_cast<std::size_t>(bytes_int_N));

    // -------------------------------------------------------------------
    // 2. Exclusive prefix scan: tiles_touched -> point_offsets.
    // -------------------------------------------------------------------
    scan_pass_->bind_buffers(bin_tt_buf_->handle(),
                             bin_po_buf_->handle(),
                             bin_ws_buf_->handle());
    scan_pass_->scan_sync(static_cast<uint32_t>(N));

    // -------------------------------------------------------------------
    // 3. Compute R (total pairs) = point_offsets[N-1] + tiles_touched[N-1].
    // -------------------------------------------------------------------
    uint32_t last_offset = 0u;
    uint32_t last_count  = 0u;
    bin_po_buf_->download(&last_offset,
                          sizeof(uint32_t),
                          /*offset=*/static_cast<VkDeviceSize>(N - 1) * sizeof(uint32_t));
    // tiles_touched is stored as int32 but non-negative; read as uint32.
    {
        int32_t tmp = 0;
        bin_tt_buf_->download(&tmp,
                              sizeof(int32_t),
                              static_cast<VkDeviceSize>(N - 1) * sizeof(int32_t));
        if (tmp < 0)
            throw std::runtime_error(
                "TileBinnerVulkan: tiles_touched[N-1] is negative — preprocess bug?");
        last_count = static_cast<uint32_t>(tmp);
    }
    const uint32_t R = last_offset + last_count;
    out.total_pairs  = static_cast<int>(R);

    // Allocate/grow scatter output GPU buffers to exactly R entries.
    // This runs AFTER the scan — actual R is known, so no over-allocation.
    if (R > 0u)
        prepare_for_scatter(static_cast<uint64_t>(R));

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
    // 4. Upload scatter inputs into persistent SSBOs, dispatch.
    //    keys_unsorted[R_max] / values_unsorted[R_max] are pre-allocated
    //    (R_max >= R by construction); we only download R valid elements.
    // -------------------------------------------------------------------
    bin_m2d_buf_->upload(pre.means2D, static_cast<std::size_t>(bytes_float_N2));
    bin_dep_buf_->upload(pre.depths,  static_cast<std::size_t>(bytes_float_N));
    bin_rad_buf_->upload(pre.radii,   static_cast<std::size_t>(bytes_int_N));
    if (pre.radius_f != nullptr) {
        bin_rf_buf_->upload(pre.radius_f, static_cast<std::size_t>(bytes_float_N));
    } else {
        std::vector<float> rf_host(static_cast<std::size_t>(N));
        for (int k = 0; k < N; ++k)
            rf_host[static_cast<std::size_t>(k)] = static_cast<float>(pre.radii[k]);
        bin_rf_buf_->upload(rf_host.data(), static_cast<std::size_t>(bytes_float_N));
    }

    ScatterPass::Buffers sb{};
    sb.means2D         = bin_m2d_buf_->handle();
    sb.depths          = bin_dep_buf_->handle();
    sb.radii           = bin_rad_buf_->handle();
    sb.point_offsets   = bin_po_buf_ ->handle();
    sb.tiles_touched   = bin_tt_buf_ ->handle();
    sb.keys_unsorted   = bin_keys_buf_->handle();
    sb.values_unsorted = bin_vals_buf_->handle();
    sb.radius_f        = bin_rf_buf_ ->handle();
    scatter_pass_->bind_buffers(sb);
    scatter_pass_->dispatch_sync(static_cast<uint32_t>(N),
                                 num_tiles_x,
                                 num_tiles_y);

    // -------------------------------------------------------------------
    // 5. Download R valid pairs into FrameAllocator-backed output.
    // -------------------------------------------------------------------
    out.keys_unsorted   = alloc.allocate_array<uint64_t>(R);
    out.values_unsorted = alloc.allocate_array<uint32_t>(R);
    out.keys_sorted     = nullptr;   // SorterVulkan (T14)
    out.values_sorted   = nullptr;
    out.tile_ranges     = nullptr;

    bin_keys_buf_->download(out.keys_unsorted,
                            static_cast<std::size_t>(R) * sizeof(uint64_t));
    bin_vals_buf_->download(out.values_unsorted,
                            static_cast<std::size_t>(R) * sizeof(uint32_t));

    return out;
}

// ---------------------------------------------------------------------------
// Layer-2 record-mode: prepare_record() + record() + buffer getters.
// ---------------------------------------------------------------------------
void TileBinnerVulkan::prepare_record(uint32_t N, uint32_t R_max,
                                      uint32_t num_tiles_x,
                                      uint32_t num_tiles_y,
                                      VkBuffer tiles_touched,
                                      VkBuffer means2D,
                                      VkBuffer depths,
                                      VkBuffer radii,
                                      VkBuffer radius_f) {
    if (N == 0u)
        throw std::runtime_error(
            "TileBinnerVulkan::prepare_record: N must be > 0");
    if (R_max == 0u)
        throw std::runtime_error(
            "TileBinnerVulkan::prepare_record: R_max must be > 0");

    // Release old buffers first.
    r_po_buf_.reset();
    r_ws_buf_.reset();
    r_keys_buf_.reset();
    r_vals_buf_.reset();

    // point_offsets[N]: exclusive scan output.
    r_po_buf_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(N) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // workgroup_sums: one uint per phase-0 workgroup, min 1.
    const uint32_t num_wgs = (N + 255u) / 256u;
    const uint32_t wg_sums_count = num_wgs == 0u ? 1u : num_wgs;
    r_ws_buf_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(wg_sums_count) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // keys_unsorted[R_max] + values_unsorted[R_max].
    r_keys_buf_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(R_max) * sizeof(uint64_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_vals_buf_ = std::make_unique<VulkanBuffer>(
        ctx_, static_cast<VkDeviceSize>(R_max) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Bind scan (input=tiles_touched, output=point_offsets, wg_sums).
    scan_pass_->bind_buffers(tiles_touched,
                             r_po_buf_->handle(),
                             r_ws_buf_->handle());

    // Bind scatter (inputs + outputs).
    ScatterPass::Buffers sb{};
    sb.means2D         = means2D;
    sb.depths          = depths;
    sb.radii           = radii;
    sb.point_offsets   = r_po_buf_->handle();
    sb.tiles_touched   = tiles_touched;
    sb.keys_unsorted   = r_keys_buf_->handle();
    sb.values_unsorted = r_vals_buf_->handle();
    sb.radius_f        = radius_f;
    scatter_pass_->bind_buffers(sb);

    // num_tiles_x/_y are used only at record() time (push constant) — nothing
    // else to stash here.
    (void)num_tiles_x;
    (void)num_tiles_y;
}

void TileBinnerVulkan::record(VkCommandBuffer cmd,
                              uint32_t N, uint32_t num_tiles_x,
                              uint32_t num_tiles_y) {
    if (!r_po_buf_)
        throw std::runtime_error(
            "TileBinnerVulkan::record called before prepare_record()");

    // Scan writes point_offsets; scatter reads point_offsets (+tiles_touched,
    // which is unmodified). Barrier between the two is a RAW compute→compute
    // dependency.
    scan_pass_->record(cmd, N);
    insert_compute_barrier(cmd);
    scatter_pass_->record(cmd, N, num_tiles_x, num_tiles_y);
}

VkBuffer TileBinnerVulkan::keys_unsorted_buf() const {
    return r_keys_buf_ ? r_keys_buf_->handle() : VK_NULL_HANDLE;
}
VkBuffer TileBinnerVulkan::values_unsorted_buf() const {
    return r_vals_buf_ ? r_vals_buf_->handle() : VK_NULL_HANDLE;
}
