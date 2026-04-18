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
}  // namespace preprocess_bind

// Push constants (24 bytes, spec §4.8.1)
struct PreprocessPushConstants {
    uint32_t num_gaussians;
    uint32_t sh_degree;
    uint32_t sh_coeffs_per_g;  // == (sh_degree+1)^2; set via helper to avoid drift with sh_degree
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    float    scale_modifier;
};
static_assert(sizeof(PreprocessPushConstants) == 24,
              "PreprocessPushConstants must be exactly 24 bytes per spec §4.8.1");

// Specialization constant IDs (spec §4.5)
namespace preprocess_spec {
constexpr uint32_t TRAINING = 0;  // spec_training: 1=training (no clamp), 0=inference (clamp)
constexpr uint32_t EVAL_3D  = 1;  // spec_eval_3D: always 0 in SP-2
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
}
struct ScatterPushConstants {
    uint32_t num_gaussians;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t _pad;  // align to 24 bytes
};
static_assert(sizeof(ScatterPushConstants) == 24, "ScatterPushConstants must be 24 bytes per spec §4.8.3");

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
}
struct RasterizePushConstants {
    uint32_t num_gaussians;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
};
static_assert(sizeof(RasterizePushConstants) == 20, "RasterizePushConstants must be 20 bytes per spec §4.8.7");
