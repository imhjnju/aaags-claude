#pragma once
#include <cmath>
#include <algorithm>

// 4x3 transform: p' = M * [p, 1], column-major matrix
void transformPoint4x3(const float* p, const float* m, float out[3]);

// 4x4 transform: returns [x, y, z, w], column-major matrix
void transformPoint4x4(const float* p, const float* m, float out[4]);

// NDC [-1,1] to pixel [0, S-1]
inline float ndc2Pix(float v, int S) {
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

// Compute tile bounding rect for Gaussian at pixel pos p with given radius
void getRect(const float* p, int max_radius,
             int grid_x, int grid_y, int tile_w, int tile_h,
             int rect_min[2], int rect_max[2]);

// 3D covariance from scale + rotation quaternion
// Quaternion order: rot[0]=r(real), rot[1]=x, rot[2]=y, rot[3]=z
// Output: 6 floats upper triangle [0,0],[0,1],[0,2],[1,1],[1,2],[2,2]
void computeCov3D(const float scale[3], float scale_modifier,
                  const float rot[4], float cov3D[6]);

// Project 3D covariance to 2D screen-space
// Output: 3 floats (a, b, c) where cov2D = [[a, b], [b, c]]
void computeCov2D(const float mean3d[3], const float cov3D[6],
                  const float* view_matrix,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  float cov2D[3]);

// Frustum test: returns true if point is in front of camera (p_view.z > 0.2)
bool inFrustum(const float point[3], const float* view_matrix,
               const float* viewproj_matrix, float p_view[3]);

// --- AAA-Gaussians 3D evaluation functions ---

// Quaternion to 3x3 rotation matrix (row-major R[row][col])
void quat2mat(const float rot[4], float R[3][3]);

// Compute gauss2screen 4x4 matrix for 3D Gaussian evaluation.
// Returns dilation_factor for opacity adjustment.
// gauss2screen and gauss2view are stored row-major.
float computeGauss2Screen(
    const float mean3D[3], const float scale[3], const float rot[4],
    float scale_modifier, const float cam_pos[3],
    const float* viewproj_matrix, const float* view_matrix,
    float focal, float kernel_size, float filter_3d,
    int W, int H,
    float gauss2screen[16], float gauss2view[16]);

// Compute view-space AABB for 3D Gaussian bounding.
// Returns false if Gaussian should be culled.
bool computeAABBView(
    const float gauss2view[16], const float mean3D_view[3],
    float focal_x, float focal_y, float W, float H, float cutoff,
    float mean2D[2], float extent2D[2]);

// Compute screen-space AABB (Hahlbohm et al.) - more robust than view-space AABB.
// gauss2screen is row-major 4x4. Returns false if Gaussian should be culled.
bool computeAABBScreen(
    const float gauss2screen[16], float cutoff,
    float mean2D[2], float extent2D[2]);

// 3D frustum culling: returns max contribution of Gaussian within screen bounds.
float maxContribGaussianFrustum3D(
    float screen_min_x, float screen_min_y,
    float screen_max_x, float screen_max_y,
    const float g2s[16], float& max_pos_depth);

// Per-pixel ray evaluation: returns Mahalanobis distance squared
float maxContribRayPixel(const float plane_a[4], const float plane_b[4], float max_pos[4]);

// Compute inverse 3D covariance from dilated scale and rotation.
// Output: 6 floats upper triangle [0,0],[0,1],[0,2],[1,1],[1,2],[2,2]
void computeCov3DInv(const float scale_dilated[3], float scale_modifier,
                     const float rot[4], float cov3D_inv[6]);

// Depth along ray: t = (mu^T Sigma^{-1} d) / (d^T Sigma^{-1} d)
// cov3D_inv: 6-float symmetric upper triangle.
// mean_offset: world-space (gaussian_pos - cam_pos).
// viewdir: normalized world-space direction.
float depthAlongRay(const float cov3D_inv[6], const float mean_offset[3],
                    const float viewdir[3]);

// Pixel to world-space ray direction (normalized).
void pixelToWorldDir(float px, float py, int width, int height,
                     const float inverse_vp[16], const float cam_pos[3],
                     float dir_out[3]);

// Invert a 4x4 column-major matrix. Returns false if singular.
bool invertMatrix4x4(const float m[16], float inv[16]);
