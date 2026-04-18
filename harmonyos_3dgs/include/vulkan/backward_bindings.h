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

// ---- preprocess_backward.comp binding layout (18 total: 17 SSBOs + 1 UBO) ----
//
// Binding layout:
//   0..9   read-only input SSBOs
//   10..13 read-write output SSBOs
//   14..15 new: opacities input + d_raw_opacities output
//   16     uniform buffer (PreprocessBackwardUBO)
//   17     read-only SSBO: raw_rotations (unnormalized quaternions)
namespace preprocess_backward_bind {
constexpr uint32_t POSITIONS        = 0;   // RO float[N*3]          world-space xyz
constexpr uint32_t RADII            = 1;   // RO int[N]              int-ceiled radius; 0=culled
constexpr uint32_t COV3D            = 2;   // RO float[N*6]          from ForwardCache.cov3D
constexpr uint32_t D_CONICS         = 3;   // RO float[N*3]          from rasterizer backward (a,b,c)
constexpr uint32_t D_OPACITY        = 4;   // RO float[N]            from rasterizer backward
constexpr uint32_t SH_COEFFS        = 5;   // RO float[N*max_coeffs*3]
constexpr uint32_t SCALES           = 6;   // RO float[N*3]          exp-activated
constexpr uint32_t ROTATIONS        = 7;   // RO float[N*4]          normalized quaternion (r,x,y,z)
constexpr uint32_t D_RGB            = 8;   // RO float[N*3]          from rasterizer backward
constexpr uint32_t D_MEANS2D        = 9;   // RO float[N*2]          from rasterizer backward
constexpr uint32_t D_MEANS3D        = 10;  // RW float[N*3]          output (write, not atomic)
constexpr uint32_t D_SH             = 11;  // RW float[N*max_coeffs*3] output
constexpr uint32_t D_SCALES         = 12;  // RW float[N*3]          output
constexpr uint32_t D_ROTATIONS      = 13;  // RW float[N*4]          output
constexpr uint32_t OPACITIES        = 14;  // RO float[N]            activated sigmoid values
constexpr uint32_t D_RAW_OPACITIES  = 15;  // RW float[N]            output
constexpr uint32_t PREPROCESS_BACKWARD_UBO = 16;  // UBO PreprocessBackwardUBO (192 bytes)
constexpr uint32_t RAW_ROTATIONS    = 17;  // RO float[N*4]          unnormalized quaternions
}  // namespace preprocess_backward_bind

// PreprocessBackwardUBO (std140).
// Layout (192 bytes):
//   float view_matrix[16]    → 64 bytes (offset 0)
//   float proj_matrix[16]    → 64 bytes (offset 64)
//   uint32_t num_gaussians   → 4 bytes  (offset 128)
//   uint32_t sh_degree       → 4 bytes  (offset 132)
//   uint32_t sh_coeffs_per_g → 4 bytes  (offset 136)
//   float scale_modifier     → 4 bytes  (offset 140)
//   float h_x                → 4 bytes  (offset 144)
//   float h_y                → 4 bytes  (offset 148)
//   float tan_fovx           → 4 bytes  (offset 152)
//   float tan_fovy           → 4 bytes  (offset 156)
//   float cam_pos[3]         → 12 bytes (offset 160)
//   float _pad               → 4 bytes  (offset 172)
//   uint32_t training        → 4 bytes  (offset 176)
//   uint32_t cam_width       → 4 bytes  (offset 180)  pixel width  (previously _pad2[0])
//   uint32_t cam_height      → 4 bytes  (offset 184)  pixel height (previously _pad2[1])
//   uint32_t _pad2           → 4 bytes  (offset 188)
// Total: 192 bytes
struct alignas(16) PreprocessBackwardUBO {
    float    view_matrix[16]  = {};  // column-major 4x4
    float    proj_matrix[16]  = {};  // column-major 4x4 (viewproj)
    uint32_t num_gaussians    = 0;
    uint32_t sh_degree        = 0;
    uint32_t sh_coeffs_per_g  = 0;  // (sh_degree+1)^2
    float    scale_modifier   = 1.f;
    float    h_x              = 0.f;  // W / (2 * tan_fovx)
    float    h_y              = 0.f;  // H / (2 * tan_fovy)
    float    tan_fovx         = 0.f;
    float    tan_fovy         = 0.f;
    float    cam_pos[3]       = {};
    float    _pad             = 0.f;
    uint32_t training         = 1;   // 1=training (no clamp mask), 0=inference
    uint32_t cam_width        = 0;   // pixel width  — used by shader for ndc2Pix backward
    uint32_t cam_height       = 0;   // pixel height — used by shader for ndc2Pix backward
    uint32_t _pad2            = 0;
};
static_assert(sizeof(PreprocessBackwardUBO) == 192,
              "PreprocessBackwardUBO must be 192 bytes (std140)");
