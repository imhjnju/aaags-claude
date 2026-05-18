#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

// Arena-style frame allocator. Avoids per-frame malloc/free overhead.
class FrameAllocator {
public:
    explicit FrameAllocator(size_t capacity);
    ~FrameAllocator();

    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    void* allocate(size_t bytes, size_t alignment = 64);

    template<typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T)));
    }

    void reset();
    // Grow the buffer to at least new_capacity bytes. No-op if already large enough.
    // Safe to call after reset() (offset must be 0). Invalidates all prior pointers.
    void grow(size_t new_capacity);
    size_t used() const;
    size_t capacity() const;

private:
    char* buffer_;
    size_t capacity_;
    size_t offset_;
};

// SoA layout for Gaussian model data (pre-activated values)
struct GaussianData {
    int count;              // N: number of Gaussians
    int sh_degree;          // 0-3
    int max_coeffs;         // (sh_degree+1)^2

    float* positions;       // [N * 3] world-space xyz
    float* sh_coeffs;       // [N * max_coeffs * 3] interleaved RGB per basis
    float* scales;          // [N * 3] exp()-activated
    float* rotations;       // [N * 4] normalized quaternion (r, x, y, z)
    float* opacities;       // [N] sigmoid()-activated
    float* filter_3D;       // [N] AAA-Gaussians 3D mip filter (nullptr if not present)
};

// Camera parameters. All matrices are column-major (matching CUDA/GLM).
struct Camera {
    float view_matrix[16];       // 4x4 world->camera, column-major
    float viewproj_matrix[16];   // 4x4 world->clip, column-major
    float cam_pos[3];            // World-space camera position
    float tan_fovx, tan_fovy;   // tan(fov/2)
    int width, height;           // Image resolution
};

// Render configuration
struct RenderConfig {
    float bg_color[3] = {0, 0, 0};
    float scale_modifier = 1.0f;
    bool antialiasing = false;
    bool eval_3D = false;        // AAA-Gaussians: full 3D Gaussian evaluation
    bool eval_3D_parity_mode = false;  // Match CUDA eval_3D sort_mode=GLOBAL/no tile culling
    bool compact_eval3D_tiles = false;
    bool training = false;       // Training mode: skip upper color clamp in SH eval
    int sh_degree = 3;
    int tile_w = 16;
    int tile_h = 16;
};

// Preprocessor output (per-Gaussian arrays, size N)
struct PreprocessOutput {
    int num_gaussians = 0;
    float* means2D;          // [N * 2]
    float* depths;           // [N]
    float* conics;           // [N * 3]
    float* opacities_2d;     // [N]
    float* rgb;              // [N * 3]
    int* radii;              // [N]
    int* tiles_touched;      // [N]
    int num_tile_pairs = -1;  // sum(tiles_touched), valid sideband when non-negative
    float* radius_f = nullptr;  // [N*2] (extent_x, extent_y) per Gaussian (nullptr for CPU paths)
    float* gauss2screen;     // [N * 16] AAA-Gaussians: 4x4 matrix per Gaussian (nullptr if not eval_3D)
    float* cov3D_inv;       // [N * 6] inverse 3D covariance upper triangle (eval_3D only, nullptr otherwise)
    float* mean_offset;     // [N * 3] world-space (pos - cam_pos) per Gaussian (eval_3D only)
    bool eval_3D = false;    // Whether eval_3D mode was used
    void* device_data = nullptr;  // GPU: opaque handle to device buffers
    void* means2D_gpu = nullptr;              // VkBuffer, optional [N*2], valid until producer's next process/destruction
    void* depths_gpu = nullptr;               // VkBuffer, optional [N], valid until producer's next process/destruction
    void* conic_opacity_packed_gpu = nullptr; // VkBuffer, optional [N*4], valid until producer's next process/destruction
    void* rgb_gpu = nullptr;                  // VkBuffer, optional [N*3], valid until producer's next process/destruction
    void* radii_gpu = nullptr;                // VkBuffer, optional [N], valid until producer's next process/destruction
    void* tiles_touched_gpu = nullptr;        // VkBuffer, optional [N], valid until producer's next process/destruction
    void* radius_f_gpu = nullptr;             // VkBuffer, optional [N*2], valid until producer's next process/destruction
    void* gauss2screen_gpu = nullptr;         // VkBuffer, optional [N*16], valid until producer's next process/destruction
    void* cov3D_inv_gpu = nullptr;            // VkBuffer, optional [N*6], valid until producer's next process/destruction
    void* mean_offset_gpu = nullptr;          // VkBuffer, optional [N*3], valid until producer's next process/destruction
};

// Tile binner output
struct BinningOutput {
    int total_pairs;
    uint64_t* keys_unsorted;
    uint32_t* values_unsorted;
    uint64_t* keys_sorted;
    uint32_t* values_sorted;
    uint64_t* keyvals_unsorted = nullptr;
    uint64_t* keyvals_sorted = nullptr;
    int num_tiles;
    uint32_t max_value_exclusive = 0;  // optional upper bound for values_unsorted Gaussian ids
    uint32_t* tile_ranges;    // [num_tiles * 2] (start, end)
    void* device_data = nullptr;  // GPU: opaque handle to device buffers
    void* keys_unsorted_gpu = nullptr;     // VkBuffer, optional GPU-resident CUDA-style keys
    void* values_unsorted_gpu = nullptr;   // VkBuffer, optional GPU-resident gaussian ids
    void* keyvals_unsorted_gpu = nullptr;  // VkBuffer, optional GPU-resident packed keyvals
    void* keyvals_sorted_gpu = nullptr;    // VkBuffer, optional GPU-resident packed keyvals
    void* values_sorted_gpu = nullptr;     // VkBuffer, optional GPU-resident sorted gaussian ids
    void* tile_ranges_gpu = nullptr;       // VkBuffer, optional GPU-resident tile ranges
};

namespace keyval_pack {
    constexpr uint32_t TILE_BITS  = 16;
    constexpr uint32_t DEPTH_BITS = 28;
    constexpr uint32_t IDX_BITS   = 20;
    constexpr uint32_t IDX_SHIFT  = 0;
    constexpr uint32_t DEPTH_SHIFT = IDX_BITS;
    constexpr uint32_t TILE_SHIFT  = IDX_BITS + DEPTH_BITS;
    constexpr uint32_t IDX_MASK    = (1u << IDX_BITS) - 1u;
    constexpr uint32_t DEPTH_MASK  = (1u << DEPTH_BITS) - 1u;
    constexpr uint32_t TILE_MASK   = (1u << TILE_BITS) - 1u;
    constexpr uint32_t MAX_TILES   = 1u << TILE_BITS;
    constexpr uint32_t MAX_GAUSS   = 1u << IDX_BITS;

    inline uint64_t pack(uint32_t tile_id, uint32_t depth_q28, uint32_t gauss_idx) {
        return (static_cast<uint64_t>(tile_id & TILE_MASK) << TILE_SHIFT)
             | (static_cast<uint64_t>(depth_q28 & DEPTH_MASK) << DEPTH_SHIFT)
             | (static_cast<uint64_t>(gauss_idx & IDX_MASK) << IDX_SHIFT);
    }
    inline uint32_t tile_of(uint64_t kv) { return static_cast<uint32_t>(kv >> TILE_SHIFT) & TILE_MASK; }
    inline uint32_t depth_of(uint64_t kv) { return static_cast<uint32_t>(kv >> DEPTH_SHIFT) & DEPTH_MASK; }
    inline uint32_t idx_of(uint64_t kv) { return static_cast<uint32_t>(kv) & IDX_MASK; }
    inline uint32_t quantize_depth(uint32_t depth_bits_u32) { return depth_bits_u32 >> 4; }
    constexpr uint64_t INVALID = 0xFFFFFFFFFFFFFFFFULL;
}

// Cached intermediate values from the forward pass, needed for backward.
struct ForwardCache {
    float* T_final;       // [H*W] per-pixel final transmittance
    int*   n_contrib;     // [H*W] per-pixel last contributing candidate position / eval_3D blended count
    std::vector<uint32_t> replay_order_offsets_storage;
    std::vector<uint32_t> replay_order_gids_storage;
    uint32_t* replay_order_offsets = nullptr;  // [H*W+1]
    uint32_t* replay_order_gids = nullptr;     // [replay_order_count]
    size_t replay_order_count = 0;
    void* replay_order_offsets_gpu = nullptr;  // VkBuffer, optional [H*W+1]
    void* replay_order_gids_gpu = nullptr;     // VkBuffer, optional [replay_order_count]
    bool gpu_resident_outputs = false;
    bool retain_gpu_outputs = false;
    void* rendered_image_gpu = nullptr;        // VkBuffer, optional [3*H*W]
    void* T_final_gpu = nullptr;               // VkBuffer, optional [H*W]
    void* n_contrib_gpu = nullptr;             // VkBuffer, optional [H*W]
    void* dL_dpixels_gpu = nullptr;            // VkBuffer, optional [3*H*W]
    void* gauss2screen_gpu = nullptr;          // VkBuffer, optional [N*16], producer-owned
    float* cov2D;         // [N*3] filtered 2D covariance
    float* cov2D_det;     // [N] determinant of filtered cov2D
    float* cov3D;         // [N*6] 3D covariance upper triangle
    float* p_view;        // [N*3] view-space position
    float* p_hom_w;       // [N] clip-space w component
    PreprocessOutput* pre;
    BinningOutput*    bin;
};
