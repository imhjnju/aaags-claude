# Raster Shader Variant Contract

## Evidence

- Current merge `rasterize.comp` binding 0 is legacy `uint values_sorted[]`; the shader reads sorted Gaussian IDs directly while binding 1 carries `uint tile_ranges[]`.
- `include/vulkan/preprocess_bindings.h` defines the current rasterizer ABI as `VALUES_SORTED`, `TILE_RANGES`, `MEANS2D`, `CONIC_OPACITY_PACKED`, `RGB`, outputs, `RasterizeUBO`, and eval_3D extras. The push-constant ABI is unchanged at 20 bytes.
- The CUDA reference emits key/value pairs, sorts by tile/depth, then derives tile ranges from sorted keys while rasterization consumes sorted Gaussian IDs. This matches the current Vulkan legacy ABI.
- Profiling's packed-keyval shader ABI binds `uint64_t keyvals_sorted[]` at binding 0 and extracts `gauss_idx` from the low 20 bits. That ABI is incompatible with current `rasterize.comp` and must not be bound into the legacy values slot.
- The current sorter env-on path can sort packed keyvals with Fuchsia, but reconstructs legacy `keys_sorted`, `values_sorted`, and `tile_ranges` for the current rasterizer.

## Decision

Keep current `rasterize.comp` as the default production shader and preserve specialization IDs `EVAL_3D=0`, `TRACE_ENABLED=1`, `SORT_MODE=2`, `EVAL3D_RAW_REPLAY=3`.

Add profiling raster shaders only as explicitly selected variants. Do not route the default path to a packed-keyval shader until the shader, tile-range extraction, rasterizer forward consumer, and backward replay contracts are updated together.
