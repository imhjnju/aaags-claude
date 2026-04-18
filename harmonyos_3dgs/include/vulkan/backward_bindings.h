// backward_bindings.h — Single source of truth for rasterize_backward.comp
// binding indices. Mirrored exactly in the shader's layout declarations.
//
// Binding layout (13 total: 12 SSBOs + 1 UBO):
//   0..7   read-only input SSBOs  (tile_ranges, values_sorted, means2D,
//                                   conic_opacity_packed, colors, T_final,
//                                   n_contrib, dL_dpixels)
//   8..11  read-write output SSBOs (dL_dmeans2D, dL_dconics, dL_dopacity,
//                                   dL_dcolors)
//   12     uniform buffer          (RasterizeBackwardUBO)
#pragma once
#include <cstdint>

namespace rasterize_backward_bind {
constexpr uint32_t TILE_RANGES   = 0;
constexpr uint32_t VALUES_SORTED = 1;
constexpr uint32_t MEANS2D       = 2;
constexpr uint32_t CONIC_OPACITY = 3;   // packed N*4 float (a, b, c, opacity)
constexpr uint32_t COLORS        = 4;   // rgb N*3
constexpr uint32_t T_FINAL       = 5;
constexpr uint32_t N_CONTRIB     = 6;
constexpr uint32_t DL_DPIXELS    = 7;
constexpr uint32_t DL_DMEANS2D   = 8;   // RW float atomicAdd
constexpr uint32_t DL_DCONICS    = 9;   // RW float atomicAdd (a,b,c)
constexpr uint32_t DL_DOPACITY   = 10;  // RW float atomicAdd
constexpr uint32_t DL_DCOLORS    = 11;  // RW float atomicAdd (N*3)
constexpr uint32_t BACKWARD_UBO  = 12;
}  // namespace rasterize_backward_bind

// RasterizeBackwardUBO (std140, 32 bytes).
// W, H, num_tiles_x are provided via UBO (no push constants needed).
// bg_color is needed for the background gradient term.
struct alignas(16) RasterizeBackwardUBO {
    uint32_t W            = 0;
    uint32_t H            = 0;
    uint32_t num_tiles_x  = 0;
    uint32_t _pad         = 0;
    float    bg_color[3]  = {};
    float    _pad2        = 0.f;
};
static_assert(sizeof(RasterizeBackwardUBO) == 32,
              "RasterizeBackwardUBO must be 32 bytes (std140)");
