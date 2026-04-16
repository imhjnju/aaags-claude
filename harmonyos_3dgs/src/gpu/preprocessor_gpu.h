// preprocessor_gpu.h -- OpenCL GPU preprocessor for 3DGS
// Implements the Preprocessor interface using an OpenCL kernel.

#pragma once

#include "preprocessor.h"
#include "gpu/opencl_context.h"

/// Device-side output buffers (cl_mem handles).
/// Passed via PreprocessOutput::device_data so downstream GPU stages
/// can consume them without readback.
struct PreprocessDeviceBuffers {
    cl_mem means2D;       // [N*2]
    cl_mem depths;        // [N]
    cl_mem conics;        // [N*3]
    cl_mem rgb;           // [N*3]
    cl_mem opacities_2d;  // [N]
    cl_mem radii;         // [N]
    cl_mem tiles_touched; // [N]
    cl_mem gauss2screen;  // [N*16] AAA-Gaussians (nullptr if not eval_3D)
    cl_mem cov3D_inv;     // [N*6] inverse 3D covariance (eval_3D only)
    cl_mem mean_offset;   // [N*3] world-space (pos - cam_pos) (eval_3D only)
};

class PreprocessorGPU : public Preprocessor {
public:
    explicit PreprocessorGPU(OpenCLContext& ctx);
    ~PreprocessorGPU() override;

    // Not copyable/movable (owns CL resources)
    PreprocessorGPU(const PreprocessorGPU&) = delete;
    PreprocessorGPU& operator=(const PreprocessorGPU&) = delete;

    PreprocessOutput process(const GaussianData& g, const Camera& cam,
                             const RenderConfig& cfg, FrameAllocator& alloc,
                             ForwardCache* cache = nullptr) override;

private:
    OpenCLContext& ctx_;
    cl_kernel kernel_ = nullptr;

    // Persistent model data buffers (uploaded once, reused across frames)
    cl_mem d_positions_  = nullptr;
    cl_mem d_sh_coeffs_  = nullptr;
    cl_mem d_scales_     = nullptr;
    cl_mem d_rotations_  = nullptr;
    cl_mem d_opacities_  = nullptr;
    cl_mem d_filter_3D_  = nullptr;  // AAA-Gaussians
    int last_count_ = 0;  // track if model changed

    // Camera uniform buffers (small, updated each frame)
    cl_mem d_view_matrix_ = nullptr;
    cl_mem d_viewproj_    = nullptr;
    cl_mem d_cam_pos_     = nullptr;

    // Per-frame output buffers (reallocated if N changes)
    PreprocessDeviceBuffers dev_bufs_{};
    int last_output_count_ = 0;

    void uploadModel(const GaussianData& g);
    void allocOutputBuffers(int N);
    void releaseModelBuffers();
    void releaseOutputBuffers();
    void releaseCameraBuffers();
};
