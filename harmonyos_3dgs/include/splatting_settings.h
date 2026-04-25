// splatting_settings.h — Defensive configuration mirror of CUDA's
// SplattingSettings (AAA-Gaussians/submodules/diff-gaussian-rasterization/
// cuda_rasterizer/rasterizer.h:60-180).
//
// Intent (Path A — defensive config unification):
//   * VK reads the SAME `configs/aaa.json` as CUDA so both backends agree by
//     construction, not by accident.
//   * VK currently HARD-CODES many of these flags in shaders. This header
//     plumbs the JSON values; `validate_vk_supported()` then ASSERTS that
//     every JSON value matches the value VK has hard-coded. Any mismatch is
//     a configuration error and throws — VK refuses configs it does not
//     implement rather than silently rendering with the wrong behaviour.
//
// This file is intentionally JSON-free so it can live in `gs3d_vk_core`
// without dragging nlohmann into the public API. The loader lives in
// `splatting_settings.cpp` (linked alongside this header).
//
// Field name/type/default convention mirrors CUDA exactly so a `nlohmann::json`
// blob round-trips between the two struct definitions.

#pragma once

#include <string>

namespace splatting {

enum SortMode {
    GLOBAL            = 0,
    PER_PIXEL_FULL    = 1,
    PER_PIXEL_KBUFFER = 2,
    HIERARCHICAL      = 3,
};

enum GlobalSortOrder {
    VIEWSPACE_Z           = 0,
    DISTANCE              = 1,
    PER_TILE_DEPTH_CENTER = 2,
    PER_TILE_DEPTH_MAXPOS = 3,
};

struct SortQueueSizes {
    int tile_4x4  = 64;
    int tile_2x2  = 8;
    int per_pixel = 4;
};

struct SortSettings {
    SortMode        sort_mode  = SortMode::GLOBAL;
    GlobalSortOrder sort_order = GlobalSortOrder::VIEWSPACE_Z;
    SortQueueSizes  queue_sizes;
};

struct CullingSettings {
    bool rect_bounding           = false;
    bool tight_opacity_bounding  = false;
    bool tile_based_culling      = false;
    bool hierarchical_4x4_culling = false;
};

struct SplattingSettings {
    SortSettings    sort_settings;
    CullingSettings culling_settings;
    bool            load_balancing      = false;
    bool            proper_ewa_scaling  = false;
    bool            eval_3D             = false;
    bool            near_clipping       = false;
    bool            new_aabb            = true;
};

// Hard-coded values currently baked into the VK shaders. validate_vk_supported()
// compares the JSON-loaded SplattingSettings against these.
namespace vk_hardcoded {
constexpr int  QUEUE_SIZE_PER_PIXEL = 4;   // rasterize.comp:724  HEAD_W=4
constexpr int  QUEUE_SIZE_TILE_2X2  = 8;   // rasterize.comp s_mid_depth[16*4*8]
constexpr int  QUEUE_SIZE_TILE_4X4  = 64;  // rasterize.comp:237  TAIL_SLOTS=64
// Y1: Two sort_mode values are now supported:
//   GLOBAL       — routes to the existing HEAD_W=8 sub-tile-sort path
//   HIERARCHICAL — routes to the 3-stage cascade (TAIL/MID/HEAD)
// PER_PIXEL_FULL / PER_PIXEL_KBUFFER are still rejected — no VK path exists.
constexpr GlobalSortOrder SORT_ORDER = PER_TILE_DEPTH_MAXPOS;   // matches scatter.comp eval_3D path
}  // namespace vk_hardcoded

// Load `configs/aaa.json` (or any file with the same schema) into out.
// Throws std::runtime_error on file/parse error or schema mismatch.
void load_from_json(const std::string& path, SplattingSettings& out);

// Validate that `s` matches what the current VK port implements. Throws
// std::runtime_error with a descriptive message on the first mismatch.
//
// Validation policy (defensive — VK refuses configs it has not ported):
//   * sort_settings.sort_mode  MUST be GLOBAL or HIERARCHICAL (Y1: GLOBAL
//     routes to the existing HEAD_W=8 fallback path; HIERARCHICAL routes to
//     the cascade). PER_PIXEL_FULL / PER_PIXEL_KBUFFER are NOT implemented.
//   * sort_settings.sort_order MUST be PER_TILE_DEPTH_MAXPOS.
//   * sort_settings.queue_sizes MUST equal {4, 8, 64}.
//   * new_aabb MUST be true (compute_aabb_screen path is dead code in VK).
//   * culling_settings.* and load_balancing currently must all be true (the
//     VK shaders do not gate these — they apply unconditionally — so any
//     `false` would render with the wrong behaviour silently).
//   * eval_3D and near_clipping are RUNTIME-honoured by VK and do NOT
//     trigger an assertion; they are passed to the pipeline as flags.
void validate_vk_supported(const SplattingSettings& s);

}  // namespace splatting
