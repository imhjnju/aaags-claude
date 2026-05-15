// Single source of truth for SP-2 forward pipeline binding indices,
// push constants, and specialization constants. Mirrored exactly in
// each shader's `layout(binding=...)` declarations.
#pragma once
#include <cstdint>

// --- preprocess.comp bindings (spec §4.8.1) ---
namespace preprocess_bind {
constexpr uint32_t POSITIONS           = 0;
constexpr uint32_t SCALES              = 1;
constexpr uint32_t ROTATIONS           = 2;
constexpr uint32_t OPACITIES           = 3;
constexpr uint32_t SH                  = 4;
constexpr uint32_t FILTER_3D           = 5;
constexpr uint32_t MEANS2D             = 6;
constexpr uint32_t DEPTHS              = 7;
constexpr uint32_t CONIC_OPACITY_PACKED = 8;
constexpr uint32_t RGB                 = 9;
constexpr uint32_t RADII               = 10;
constexpr uint32_t TILES_TOUCHED       = 11;
constexpr uint32_t CAMERA_UBO          = 12;
constexpr uint32_t RADIUS_F            = 13;  // float eigenvalue radius (scatter fix)
constexpr uint32_t COV3D_CACHE         = 14;  // [N*6] 3D covariance cache for backward
constexpr uint32_t P_VIEW_CACHE        = 15;  // [N*3] view-space position cache for backward
constexpr uint32_t P_HOM_W_CACHE       = 16;  // [N]   clip-space w cache for backward
constexpr uint32_t COV2D_CACHE         = 17;  // WO float[N*3]  (fa, fb, fc) dilated cov2D
constexpr uint32_t COV2D_DET_CACHE     = 18;  // WO float[N]    det = fa*fc - fb*fb
constexpr uint32_t GAUSS2SCREEN = 19; // WO float[N*16] gauss2screen row-major (eval_3D only)
constexpr uint32_t COV3D_INV    = 20; // WO float[N*6]  inverse 3D covariance upper tri (eval_3D only)
constexpr uint32_t MEAN_OFFSET  = 21; // WO float[N*3]  world-space (pos - cam_pos) (eval_3D only)
}  // namespace preprocess_bind

// Push constants (32 bytes, spec §4.8.1 + opt-in perf flag)
struct PreprocessPushConstants {
    uint32_t num_gaussians;
    uint32_t sh_degree;
    uint32_t sh_coeffs_per_g;  // == (sh_degree+1)^2; set via helper to avoid drift with sh_degree
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    float    scale_modifier;
    uint32_t compact_eval3D_tiles;
    uint32_t _pad;
};
static_assert(sizeof(PreprocessPushConstants) == 32,
              "PreprocessPushConstants must be exactly 32 bytes");

// Specialization constant IDs (spec §4.5)
namespace preprocess_spec {
constexpr uint32_t TRAINING    = 0;  // spec_training: 1=training (no clamp), 0=inference (clamp)
constexpr uint32_t EVAL_3D     = 1;  // spec_eval_3D: 0=2D conic path, 1=3D evaluation path
constexpr uint32_t PROPER_EWA  = 2;  // spec_proper_ewa: 1=AAA proper_ewa_scaling+tight_opacity_bounding+rect_bounding
}  // namespace preprocess_spec

// --- prefix_sum.comp bindings (spec §4.8.2) ---
namespace prefix_sum_bind {
constexpr uint32_t INPUT_ARRAY     = 0;
constexpr uint32_t OUTPUT_ARRAY    = 1;
constexpr uint32_t WORKGROUP_SUMS  = 2;
}
struct PrefixSumPushConstants {
    uint32_t num_elements;
    uint32_t phase;       // 0=local scan, 1=workgroup-sum scan, 2=add-back
    uint32_t stride;
    uint32_t _pad;
};
static_assert(sizeof(PrefixSumPushConstants) == 16, "PrefixSumPushConstants must be 16 bytes per spec §4.8.2");

// --- scatter.comp bindings (spec §4.8.3) ---
namespace scatter_bind {
constexpr uint32_t MEANS2D         = 0;
constexpr uint32_t DEPTHS          = 1;
constexpr uint32_t RADII           = 2;
constexpr uint32_t POINT_OFFSETS   = 3;
constexpr uint32_t TILES_TOUCHED   = 4;
constexpr uint32_t KEYS_UNSORTED   = 5;
constexpr uint32_t VALUES_UNSORTED = 6;
constexpr uint32_t RADIUS_F        = 7;  // float eigenvalue radius (read-only)
constexpr uint32_t COV3D_INV    = 8;  // RO float[N*6]  inverse 3D covariance (eval_3D per-tile depth)
constexpr uint32_t MEAN_OFFSET  = 9;  // RO float[N*3]  world-space (pos - cam_pos)
constexpr uint32_t SCATTER_UBO  = 10; // UB ScatterUBO  (inverse_vp + eval_3D flag)
constexpr uint32_t GAUSS2SCREEN = 11; // RO float[N*16] gauss2screen row-major (eval_3D sort key)
constexpr uint32_t CONIC_OPACITY_PACKED = 12; // RO float[N*4]  {a,b,c,opacity} per Gaussian (eval_3D tile culling)
constexpr uint32_t KEYVALS_UNSORTED = 13; // WO uint64[R] packed keyvals for Fuchsia radix input
}
struct ScatterPushConstants {
    uint32_t num_gaussians;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t eval_3D;   // 1 = use depthAlongRay per-tile, 0 = use view-space z
    uint32_t compact_eval3D_tiles;
    uint32_t _pad;
};
static_assert(sizeof(ScatterPushConstants) == 32, "ScatterPushConstants must be 32 bytes");
struct alignas(16) ScatterUBO {
    float inverse_vp[16];   // 64B: inverse viewproj matrix (column-major)
    float cam_pos[4];       // 16B: xyz=cam_pos, w=0
    float img_size[4];      // 16B: x=width, y=height, z=0, w=0
};
static_assert(sizeof(ScatterUBO) == 96, "ScatterUBO must be 96 bytes (std140)");

// --- radix_sort_count.comp bindings (spec §4.8.4) ---
namespace radix_count_bind {
constexpr uint32_t KEYS_IN   = 0;
constexpr uint32_t HISTOGRAMS = 1;
}
// --- radix_sort_scatter.comp bindings (spec §4.8.5) ---
namespace radix_scatter_bind {
constexpr uint32_t KEYS_IN        = 0;
constexpr uint32_t VALUES_IN      = 1;
constexpr uint32_t BUCKET_OFFSETS = 2;
constexpr uint32_t KEYS_OUT       = 3;
constexpr uint32_t VALUES_OUT     = 4;
}
struct RadixSortPushConstants {
    uint32_t num_elements;
    uint32_t current_bit;   // 0, 4, 8, ..., 60
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(RadixSortPushConstants) == 16, "RadixSortPushConstants must be 16 bytes per spec §4.8.4/5");

// --- tile_range.comp bindings (spec §4.8.6) ---
namespace tile_range_bind {
constexpr uint32_t KEYS_SORTED = 0;
constexpr uint32_t TILE_RANGES = 1;
}
struct TileRangePushConstants {
    uint32_t num_elements;
    uint32_t num_tiles;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(TileRangePushConstants) == 16, "TileRangePushConstants must be 16 bytes per spec §4.8.6");

namespace packed_keyval_extract_bind {
constexpr uint32_t KEYVALS_SORTED = 0;
constexpr uint32_t VALUES_SORTED  = 1;
constexpr uint32_t TILE_RANGES    = 2;
}
struct PackedKeyvalExtractPushConstants {
    uint32_t num_elements;
    uint32_t num_tiles;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(PackedKeyvalExtractPushConstants) == 16,
              "PackedKeyvalExtractPushConstants must be 16 bytes");

// --- rasterize.comp bindings (spec §4.8.7) ---
namespace rasterize_bind {
constexpr uint32_t VALUES_SORTED      = 0;
constexpr uint32_t TILE_RANGES        = 1;
constexpr uint32_t MEANS2D            = 2;
constexpr uint32_t CONIC_OPACITY_PACKED = 3;
constexpr uint32_t RGB                = 4;
constexpr uint32_t OUT_IMAGE          = 5;
constexpr uint32_t TRANSMITTANCE      = 6;
constexpr uint32_t N_CONTRIB          = 7;
constexpr uint32_t RASTER_UBO         = 8;
constexpr uint32_t GAUSS2SCREEN      = 9;  // RO float[N*16] gauss2screen row-major
constexpr uint32_t OPACITIES_2D      = 10; // RO float[N]    pre-dilated opacity (eval_3D)
constexpr uint32_t COV3D_INV         = 11; // RO float[N*6]  inverse 3D covariance
constexpr uint32_t MEAN_OFFSET       = 12; // RO float[N*3]  world-space offset
constexpr uint32_t RASTER_EVAL3D_UBO = 13; // UB  RasterEval3DUBO (inverse_vp + cam info)
constexpr uint32_t REPLAY_ORDER_OFFSETS = 14; // RO uint[H*W+1] exact eval_3D replay offsets
constexpr uint32_t REPLAY_ORDER_GIDS    = 15; // WO uint[sum(n_contrib)] exact replay Gaussian IDs
}
struct RasterizePushConstants {
    uint32_t num_gaussians;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
};
static_assert(sizeof(RasterizePushConstants) == 20, "RasterizePushConstants must be 20 bytes per spec §4.8.7");

// --- RasterizeUBO (std140, 16 bytes) — background colour for rasterize.comp ---
struct alignas(16) RasterizeUBO {
    float bg_r = 0.f;
    float bg_g = 0.f;
    float bg_b = 0.f;
    float _pad = 0.f;
};
static_assert(sizeof(RasterizeUBO) == 16, "RasterizeUBO must be 16 bytes (std140)");

struct alignas(16) RasterEval3DUBO {
    float inverse_vp[16];   // 64B: inverse viewproj matrix
    float cam_pos[4];       // 16B: xyz=cam_pos, w=0
    float img_size[4];      // 16B: x=width, y=height, z=0, w=0
};
static_assert(sizeof(RasterEval3DUBO) == 96, "RasterEval3DUBO must be 96 bytes (std140)");

namespace rasterize_spec {
constexpr uint32_t EVAL_3D            = 0;  // 1 = eval_3D path, 0 = 2D conic path
constexpr uint32_t TRACE_ENABLED      = 1;  // 1 = emit cascade trace (Phase 4 VK port)
constexpr uint32_t SORT_MODE          = 2;  // CUDA SortMode enum (Y1 routing — see below)
constexpr uint32_t EVAL3D_RAW_REPLAY  = 3;  // 1 = eval_3D raw replay for training parity

// Sort-mode values that the Y1 routing currently understands. Other CUDA
// SortMode values (PER_PIXEL_FULL=1, PER_PIXEL_KBUFFER=2) are NOT implemented
// in the VK shader and validate_vk_supported() rejects them at config load.
constexpr uint32_t SORT_MODE_GLOBAL       = 0u;  // existing HEAD_W=8 sub-tile-sort path
constexpr uint32_t SORT_MODE_HIERARCHICAL = 3u;  // 3-stage cascade (TAIL/MID/HEAD)
}  // namespace rasterize_spec

// --- rasterize.comp CASCADE TRACE bindings (Phase 4 / Milestone A) -----------
// Trace bindings are shifted after replay-order bindings 14..15.
namespace rasterize_trace_bind {
constexpr uint32_t TRACE_META         = 16;  // UBO TraceMetaUBO { K, num_tiles, ... }
constexpr uint32_t SLOT_LOOKUP        = 17;  // SSBO int[num_tiles]
constexpr uint32_t TAIL_DEPTHS        = 18;  // SSBO float[K*512*16*64]
constexpr uint32_t TAIL_IDS           = 19;  // SSBO int  [K*512*16*64]
constexpr uint32_t TAIL_WCUR          = 20;  // SSBO uint [K]
constexpr uint32_t MID_DEPTHS         = 21;  // SSBO float[K*1024*16*4*8]
constexpr uint32_t MID_IDS            = 22;  // SSBO int  [K*1024*16*4*8]
constexpr uint32_t MID_WCUR           = 23;  // SSBO uint [K]
constexpr uint32_t HEAD_INS_DEPTH     = 24;  // SSBO float[K*256*4096]
constexpr uint32_t HEAD_INS_ALPHA     = 25;  // SSBO float[K*256*4096]
constexpr uint32_t HEAD_INS_GID       = 26;  // SSBO int  [K*256*4096]
constexpr uint32_t HEAD_INS_CURSOR    = 27;  // SSBO uint [K*256]
constexpr uint32_t HEAD_BLEND_DEPTH   = 28;  // SSBO float[K*256*4096]
constexpr uint32_t HEAD_BLEND_ALPHA   = 29;  // SSBO float[K*256*4096]
constexpr uint32_t HEAD_BLEND_T       = 30;  // SSBO float[K*256*4096]
constexpr uint32_t HEAD_BLEND_GID     = 31;  // SSBO int  [K*256*4096]
constexpr uint32_t HEAD_BLEND_CURSOR  = 32;  // SSBO uint [K*256]
constexpr uint32_t BINDING_COUNT      = 33;  // total bindings (0..32)
}  // namespace rasterize_trace_bind

// TraceMetaUBO (std140) — 32 bytes, matches the shader's TraceMetaUBO block.
struct alignas(16) TraceMetaUBO {
    uint32_t K = 0;
    uint32_t num_tiles = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _reserved[4] = {0, 0, 0, 0};
};
static_assert(sizeof(TraceMetaUBO) == 32, "TraceMetaUBO must be 32 bytes (std140)");

// Oversized trace buffer element counts (must match CascadeTraceConsts on the
// CUDA side — see AAA-Gaussians/submodules/diff-gaussian-rasterization/
// cuda_rasterizer/stopthepop/cascade_trace.h).
namespace rasterize_trace_consts {
constexpr uint32_t MAX_TAIL_SNAPSHOTS = 512;
constexpr uint32_t MAX_MID_SNAPSHOTS  = 1024;
constexpr uint32_t MAX_HEAD_INS       = 4096;
constexpr uint32_t MAX_HEAD_BLEND     = 4096;
constexpr uint32_t SUBTILES_PER_TILE  = 16;
constexpr uint32_t QUADRANTS_PER_SUB  = 4;
constexpr uint32_t TAIL_SLOTS         = 64;
constexpr uint32_t MID_SLOTS          = 8;
constexpr uint32_t PIXELS_PER_TILE    = 256;
}  // namespace rasterize_trace_consts
