#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>

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
    bool training = false;       // Training mode: skip upper color clamp in SH eval
    int sh_degree = 3;
    int tile_w = 16;
    int tile_h = 16;
};

// Preprocessor output (per-Gaussian arrays, size N)
struct PreprocessOutput {
    float* means2D;          // [N * 2]
    float* depths;           // [N]
    float* conics;           // [N * 3]
    float* opacities_2d;     // [N]
    float* rgb;              // [N * 3]
    int* radii;              // [N]
    int* tiles_touched;      // [N]
    float* radius_f = nullptr;  // [N*2] (extent_x, extent_y) per Gaussian (nullptr for CPU paths)
    float* gauss2screen;     // [N * 16] AAA-Gaussians: 4x4 matrix per Gaussian (nullptr if not eval_3D)
    float* cov3D_inv;       // [N * 6] inverse 3D covariance upper triangle (eval_3D only, nullptr otherwise)
    float* mean_offset;     // [N * 3] world-space (pos - cam_pos) per Gaussian (eval_3D only)
    bool eval_3D = false;    // Whether eval_3D mode was used
    void* device_data = nullptr;  // GPU: opaque handle to device buffers
};

// Tile binner output
struct BinningOutput {
    int total_pairs;
    uint64_t* keys_unsorted;
    uint32_t* values_unsorted;
    uint64_t* keys_sorted;
    uint32_t* values_sorted;
    int num_tiles;
    uint32_t* tile_ranges;    // [num_tiles * 2] (start, end)
    void* device_data = nullptr;  // GPU: opaque handle to device buffers
};

// Cached intermediate values from the forward pass, needed for backward.
struct ForwardCache {
    float* T_final;       // [H*W] per-pixel final transmittance
    int*   n_contrib;     // [H*W] per-pixel contributing Gaussian count
    float* cov2D;         // [N*3] filtered 2D covariance
    float* cov2D_det;     // [N] determinant of filtered cov2D
    float* cov3D;         // [N*6] 3D covariance upper triangle
    float* p_view;        // [N*3] view-space position
    float* p_hom_w;       // [N] clip-space w component
    PreprocessOutput* pre;
    BinningOutput*    bin;
};
