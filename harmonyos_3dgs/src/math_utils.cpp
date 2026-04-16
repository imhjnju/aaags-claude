#include "math_utils.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <utility>

void transformPoint4x3(const float* p, const float* m, float out[3]) {
    out[0] = m[0] * p[0] + m[4] * p[1] + m[8]  * p[2] + m[12];
    out[1] = m[1] * p[0] + m[5] * p[1] + m[9]  * p[2] + m[13];
    out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
}

void transformPoint4x4(const float* p, const float* m, float out[4]) {
    out[0] = m[0] * p[0] + m[4] * p[1] + m[8]  * p[2] + m[12];
    out[1] = m[1] * p[0] + m[5] * p[1] + m[9]  * p[2] + m[13];
    out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    out[3] = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
}

void getRect(const float* p, int max_radius,
             int grid_x, int grid_y, int tile_w, int tile_h,
             int rect_min[2], int rect_max[2]) {
    rect_min[0] = std::min(grid_x, std::max(0, (int)((p[0] - max_radius) / tile_w)));
    rect_min[1] = std::min(grid_y, std::max(0, (int)((p[1] - max_radius) / tile_h)));
    rect_max[0] = std::min(grid_x, std::max(0, (int)((p[0] + max_radius + tile_w - 1) / tile_w)));
    rect_max[1] = std::min(grid_y, std::max(0, (int)((p[1] + max_radius + tile_h - 1) / tile_h)));
}

void computeCov3D(const float scale[3], float mod, const float rot[4], float cov3D[6]) {
    // Scale matrix S = diag(mod*scale)
    float sx = mod * scale[0], sy = mod * scale[1], sz = mod * scale[2];

    // Quaternion to rotation matrix R (rot[0]=r, rot[1]=x, rot[2]=y, rot[3]=z)
    float r = rot[0], x = rot[1], y = rot[2], z = rot[3];

    // Rotation matrix R[row][col] — must match CUDA/GLM mathematical convention.
    // CUDA uses glm::mat3 which fills COLUMNS, so the 9 values in the constructor
    // become col0, col1, col2. When we store as C [row][col], we must transpose
    // the off-diagonal elements to get R_math(row,col) correct.
    float R[3][3] = {
        {1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z),       2.f * (x * z - r * y)},
        {2.f * (x * y - r * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x)},
        {2.f * (x * z + r * y),       2.f * (y * z - r * x),       1.f - 2.f * (x * x + y * y)}
    };

    // M = S * R (S is diagonal, so M[i][j] = S[i] * R[i][j])
    float M[3][3];
    for (int i = 0; i < 3; i++) {
        float s = (i == 0) ? sx : ((i == 1) ? sy : sz);
        for (int j = 0; j < 3; j++)
            M[i][j] = s * R[i][j];
    }

    // Sigma = M^T * M (symmetric), store upper triangle
    cov3D[0] = M[0][0] * M[0][0] + M[1][0] * M[1][0] + M[2][0] * M[2][0]; // [0,0]
    cov3D[1] = M[0][0] * M[0][1] + M[1][0] * M[1][1] + M[2][0] * M[2][1]; // [0,1]
    cov3D[2] = M[0][0] * M[0][2] + M[1][0] * M[1][2] + M[2][0] * M[2][2]; // [0,2]
    cov3D[3] = M[0][1] * M[0][1] + M[1][1] * M[1][1] + M[2][1] * M[2][1]; // [1,1]
    cov3D[4] = M[0][1] * M[0][2] + M[1][1] * M[1][2] + M[2][1] * M[2][2]; // [1,2]
    cov3D[5] = M[0][2] * M[0][2] + M[1][2] * M[1][2] + M[2][2] * M[2][2]; // [2,2]
}

void computeCov2D(const float mean3d[3], const float cov3D[6],
                  const float* vm,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  float cov2D[3]) {
    // Transform point to view space
    float t[3];
    transformPoint4x3(mean3d, vm, t);

    // Clamp to 1.3x frustum boundary
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t[0] / t[2];
    float tytz = t[1] / t[2];
    t[0] = std::min(limx, std::max(-limx, txtz)) * t[2];
    t[1] = std::min(limy, std::max(-limy, tytz)) * t[2];

    // Jacobian of perspective projection (stored as [col][row], column-major like GLM)
    // GLM mat3(a,b,c, d,e,f, g,h,i) fills: col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
    // CUDA: J = mat3(focal_x/z, 0, -(fx*x)/z², 0, focal_y/z, -(fy*y)/z², 0, 0, 0)
    // So: col0 = (focal_x/z, 0, -(fx*x)/z²)
    //     col1 = (0, focal_y/z, -(fy*y)/z²)
    //     col2 = (0, 0, 0)
    float J[3][3] = {
        {focal_x / t[2], 0.0f, -(focal_x * t[0]) / (t[2] * t[2])},                          // col 0
        {0.0f, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])},                           // col 1
        {0.0f, 0.0f, 0.0f}                                                                    // col 2
    };

    // Extract 3x3 rotation W from view_matrix (column-major)
    // GLM: W = mat3(vm[0],vm[4],vm[8], vm[1],vm[5],vm[9], vm[2],vm[6],vm[10])
    // col0 = (vm[0],vm[4],vm[8]), col1 = (vm[1],vm[5],vm[9]), col2 = (vm[2],vm[6],vm[10])
    float W[3][3] = {
        {vm[0], vm[4], vm[8]},     // col 0
        {vm[1], vm[5], vm[9]},     // col 1
        {vm[2], vm[6], vm[10]}     // col 2
    };

    // T = W * J (matrix multiply, both stored as [col][row])
    float T[3][3]; // T[col][row]
    for (int col = 0; col < 3; col++)
        for (int row = 0; row < 3; row++) {
            T[col][row] = 0;
            for (int k = 0; k < 3; k++)
                T[col][row] += W[k][row] * J[col][k];
        }

    // Build 3D covariance matrix Vrk from upper triangle
    float Vrk[3][3] = {
        {cov3D[0], cov3D[1], cov3D[2]},
        {cov3D[1], cov3D[3], cov3D[4]},
        {cov3D[2], cov3D[4], cov3D[5]}
    };

    // cov = transpose(T) * transpose(Vrk) * T
    // Since Vrk is symmetric, transpose(Vrk) = Vrk
    // First compute: tmp = Vrk * T (tmp[col][row])
    float tmp[3][3];
    for (int col = 0; col < 3; col++)
        for (int row = 0; row < 3; row++) {
            tmp[col][row] = 0;
            for (int k = 0; k < 3; k++)
                tmp[col][row] += Vrk[row][k] * T[col][k];
        }

    // Then: result = transpose(T) * tmp
    // transpose(T)[col][row] = T[row][col]
    // result[col][row] = sum_k T[row][k] * tmp[col][k]
    float result[3][3];
    for (int col = 0; col < 3; col++)
        for (int row = 0; row < 3; row++) {
            result[col][row] = 0;
            for (int k = 0; k < 3; k++)
                result[col][row] += T[row][k] * tmp[col][k];
        }

    // Output upper triangle of 2x2 submatrix
    cov2D[0] = result[0][0]; // a
    cov2D[1] = result[1][0]; // b (= result[0][1] by symmetry)
    cov2D[2] = result[1][1]; // c
}

bool inFrustum(const float point[3], const float* view_matrix,
               const float* viewproj_matrix, float p_view[3]) {
    transformPoint4x3(point, view_matrix, p_view);
    return p_view[2] > 0.2f;
}

// =============================================================================
// AAA-Gaussians 3D evaluation functions
// =============================================================================

static inline float sq(float x) { return x * x; }

void quat2mat(const float rot[4], float R[3][3]) {
    float r = rot[0], x = rot[1], y = rot[2], z = rot[3];
    R[0][0] = 1.f - 2.f*(y*y + z*z); R[0][1] = 2.f*(x*y + r*z);       R[0][2] = 2.f*(x*z - r*y);
    R[1][0] = 2.f*(x*y - r*z);       R[1][1] = 1.f - 2.f*(x*x + z*z); R[1][2] = 2.f*(y*z + r*x);
    R[2][0] = 2.f*(x*z + r*y);       R[2][1] = 2.f*(y*z - r*x);       R[2][2] = 1.f - 2.f*(x*x + y*y);
}

// 4x4 column-major matrix multiply: out = A * B
static void mat4Mul(const float A[16], const float B[16], float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            out[col*4 + row] = sum;
        }
}

// Helper: dot product of two float4 stored as arrays
static inline float dot4(const float a[4], const float b[4]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

// Helper: access row i of a row-major 4x4 matrix
static inline const float* row4(const float m[16], int i) { return m + i*4; }

// max_contrib_ray: find maximum Gaussian contribution along ray defined by
// intersection of two planes. Returns the squared Mahalanobis distance.
static float maxContribRay(const float plane_a[4], const float plane_b[4], float max_pos[4]) {
    // d = cross(plane_a.xyz, plane_b.xyz)
    float dx = plane_a[1]*plane_b[2] - plane_a[2]*plane_b[1];
    float dy = plane_a[2]*plane_b[0] - plane_a[0]*plane_b[2];
    float dz = plane_a[0]*plane_b[1] - plane_a[1]*plane_b[0];

    // m = plane_a.w * plane_b.xyz - plane_a.xyz * plane_b.w
    float mx = plane_a[3]*plane_b[0] - plane_a[0]*plane_b[3];
    float my = plane_a[3]*plane_b[1] - plane_a[1]*plane_b[3];
    float mz = plane_a[3]*plane_b[2] - plane_a[2]*plane_b[3];

    float dd = dx*dx + dy*dy + dz*dz;
    float m_div_dd[3] = {mx/dd, my/dd, mz/dd};

    // max_pos = cross(d, m/dd), with w=1
    max_pos[0] = dy*m_div_dd[2] - dz*m_div_dd[1];
    max_pos[1] = dz*m_div_dd[0] - dx*m_div_dd[2];
    max_pos[2] = dx*m_div_dd[1] - dy*m_div_dd[0];
    max_pos[3] = 1.0f;

    return mx*m_div_dd[0] + my*m_div_dd[1] + mz*m_div_dd[2];
}

// max_contrib_plane: find max contribution on a single plane
static float maxContribPlane(const float plane[4], const float g2s[16], float max_pos_screen[4]) {
    float norm = plane[3] / (plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]);
    float max_pos_gauss[4] = {-plane[0]*norm, -plane[1]*norm, -plane[2]*norm, -plane[3]*norm};

    // max_pos_screen = transpose(g2s) * max_pos_gauss
    // g2s is row-major, so transpose(g2s) acts as column-major multiply
    for (int i = 0; i < 4; i++) {
        max_pos_screen[i] = 0;
        for (int j = 0; j < 4; j++)
            max_pos_screen[i] += g2s[j*4+i] * max_pos_gauss[j];
    }
    return -max_pos_gauss[3];
}

// Check if position is in range for dimension DIM
static inline bool inScreenRange(const float pos_screen[4], const float range_from[3],
                                  const float extent[3], int dim, float& d) {
    d = pos_screen[dim] - range_from[dim] * pos_screen[3];
    return d > 0.0f && d < extent[dim] * pos_screen[3];
}
static inline bool inScreenRange(const float pos_screen[4], const float range_from[3],
                                  const float extent[3], int dim) {
    float d;
    return inScreenRange(pos_screen, range_from, extent, dim, d);
}

// maxContribRay with g2s transform to screen space
static float maxContribRayScreen(const float plane_a[4], const float plane_b[4],
                                  const float g2s[16], float max_pos_screen[4]) {
    float max_pos_gauss[4];
    float contrib = maxContribRay(plane_a, plane_b, max_pos_gauss);
    // max_pos_screen = transpose(g2s) * max_pos_gauss
    for (int i = 0; i < 4; i++) {
        max_pos_screen[i] = 0;
        for (int j = 0; j < 4; j++)
            max_pos_screen[i] += g2s[j*4+i] * max_pos_gauss[j];
    }
    return contrib;
}

static constexpr float Z_NEAR = -1.0f;
static constexpr float Z_FAR = 1.0f;

float maxContribGaussianFrustum3D(
    float screen_min_x, float screen_min_y,
    float screen_max_x, float screen_max_y,
    const float g2s[16], float& max_pos_depth)
{
    float max_contrib = 1e30f; // FLT_MAX equivalent
    float max_pos_screen[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    float RANGE_FROM[3] = {screen_min_x, screen_min_y, Z_NEAR};
    float EXTENT[3] = {screen_max_x - screen_min_x, screen_max_y - screen_min_y, Z_FAR - Z_NEAR};

    // Mean in screen space: last column of g2s (row-major row 0-3, col 3)
    float mean_screen[4] = {g2s[0*4+3], g2s[1*4+3], g2s[2*4+3], g2s[3*4+3]};

    float d[3];
    bool between_x = inScreenRange(mean_screen, RANGE_FROM, EXTENT, 0, d[0]);
    bool between_y = inScreenRange(mean_screen, RANGE_FROM, EXTENT, 1, d[1]);
    bool between_z = inScreenRange(mean_screen, RANGE_FROM, EXTENT, 2, d[2]);

    if (between_x && between_y && between_z) {
        for (int i = 0; i < 4; i++) max_pos_screen[i] = mean_screen[i];
        max_contrib = 0.0f;
    } else {
        float dx_sign = (d[0] - (EXTENT[0]*0.5f * mean_screen[3])) >= 0 ? EXTENT[0]*0.5f : -EXTENT[0]*0.5f;
        float dy_sign = (d[1] - (EXTENT[1]*0.5f * mean_screen[3])) >= 0 ? EXTENT[1]*0.5f : -EXTENT[1]*0.5f;

        // closer_plane_x = g2s[row0] - g2s[row3] * (RANGE_FROM.x + EXTENT.x/2 + dx)
        float cx = RANGE_FROM[0] + EXTENT[0]*0.5f + dx_sign;
        float cy = RANGE_FROM[1] + EXTENT[1]*0.5f + dy_sign;

        float closer_plane_x[4], closer_plane_y[4];
        for (int i = 0; i < 4; i++) {
            closer_plane_x[i] = g2s[0*4+i] - g2s[3*4+i] * cx;
            closer_plane_y[i] = g2s[1*4+i] - g2s[3*4+i] * cy;
        }

        float pos_screen[4];
        float contrib;

        // Test plane_x alone
        contrib = maxContribPlane(closer_plane_x, g2s, pos_screen);
        if (contrib < max_contrib && inScreenRange(pos_screen, RANGE_FROM, EXTENT, 1) &&
            inScreenRange(pos_screen, RANGE_FROM, EXTENT, 2)) {
            max_contrib = contrib;
            for (int i = 0; i < 4; i++) max_pos_screen[i] = pos_screen[i];
        }

        // Test plane_y alone
        contrib = maxContribPlane(closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && inScreenRange(pos_screen, RANGE_FROM, EXTENT, 0) &&
            inScreenRange(pos_screen, RANGE_FROM, EXTENT, 2)) {
            max_contrib = contrib;
            for (int i = 0; i < 4; i++) max_pos_screen[i] = pos_screen[i];
        }

        // Test ray: closer_plane_x ∩ closer_plane_y
        contrib = maxContribRayScreen(closer_plane_x, closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && inScreenRange(pos_screen, RANGE_FROM, EXTENT, 2)) {
            max_contrib = contrib;
            for (int i = 0; i < 4; i++) max_pos_screen[i] = pos_screen[i];
        }

        // Test ray: closer_plane_x ∩ other_plane_y
        float other_plane_y[4];
        float ocy = RANGE_FROM[1] + EXTENT[1]*0.5f - dy_sign;
        for (int i = 0; i < 4; i++)
            other_plane_y[i] = g2s[1*4+i] - g2s[3*4+i] * ocy;
        contrib = maxContribRayScreen(closer_plane_x, other_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && inScreenRange(pos_screen, RANGE_FROM, EXTENT, 2)) {
            max_contrib = contrib;
            for (int i = 0; i < 4; i++) max_pos_screen[i] = pos_screen[i];
        }

        // Test ray: other_plane_x ∩ closer_plane_y
        float other_plane_x[4];
        float ocx = RANGE_FROM[0] + EXTENT[0]*0.5f - dx_sign;
        for (int i = 0; i < 4; i++)
            other_plane_x[i] = g2s[0*4+i] - g2s[3*4+i] * ocx;
        contrib = maxContribRayScreen(other_plane_x, closer_plane_y, g2s, pos_screen);
        if (contrib < max_contrib && inScreenRange(pos_screen, RANGE_FROM, EXTENT, 2)) {
            max_contrib = contrib;
            for (int i = 0; i < 4; i++) max_pos_screen[i] = pos_screen[i];
        }
    }

    max_pos_depth = max_pos_screen[2] / max_pos_screen[3];
    return 0.5f * max_contrib;
}

bool computeAABBScreen(
    const float g2s[16], float cutoff,
    float mean2D[2], float extent2D[2])
{
    // Screen-space bounding (Hahlbohm et al.)
    // g2s is row-major: row i = g2s[i*4 .. i*4+3]
    // In CUDA code, T[i] (column of transposed mat) = row i of our matrix
    float t[4] = {cutoff, cutoff, cutoff, -1.0f};

    // dot(t, T[3] * T[3]) -- element-wise product of row3 with itself, dotted with t
    const float* r0 = g2s;       // row 0
    const float* r1 = g2s + 4;   // row 1
    const float* r2 = g2s + 8;   // row 2
    const float* r3 = g2s + 12;  // row 3

    float s = 0;
    for (int k = 0; k < 4; k++) s += t[k] * r3[k] * r3[k];
    if (s >= 0.0f) return false;

    float f[4];
    for (int k = 0; k < 4; k++) f[k] = t[k] / s;

    // p = (dot(f, r0*r3), dot(f, r1*r3), dot(f, r2*r3))
    float p[3] = {0, 0, 0};
    for (int k = 0; k < 4; k++) {
        p[0] += f[k] * r0[k] * r3[k];
        p[1] += f[k] * r1[k] * r3[k];
        p[2] += f[k] * r2[k] * r3[k];
    }

    // h = p^2 - (dot(f, ri*ri))
    float h[3] = {p[0]*p[0], p[1]*p[1], p[2]*p[2]};
    for (int k = 0; k < 4; k++) {
        h[0] -= f[k] * r0[k] * r0[k];
        h[1] -= f[k] * r1[k] * r1[k];
        h[2] -= f[k] * r2[k] * r2[k];
    }

    float extent_z = std::sqrt(std::max(0.0f, h[2]));
    float z_lo = p[2] - extent_z, z_hi = p[2] + extent_z;
    if (z_lo < -1.0f || z_hi > 1.0f) return false;

    mean2D[0] = p[0];
    mean2D[1] = p[1];
    extent2D[0] = std::sqrt(std::max(0.0f, h[0]));
    extent2D[1] = std::sqrt(std::max(0.0f, h[1]));
    return true;
}

static float normalizeAngle(float theta) {
    theta = std::fmod(theta, 2.0f * (float)M_PI);
    if (theta > (float)M_PI) theta -= 2.0f * (float)M_PI;
    if (theta <= -(float)M_PI) theta += 2.0f * (float)M_PI;
    return theta;
}

bool computeAABBView(
    const float gauss2view[16], const float mean3D_view[3],
    float focal_x, float focal_y, float W, float H, float cutoff,
    float mean2D[2], float extent2D[2])
{
    float t[4] = {cutoff, cutoff, cutoff, -1.0f};
    float vlen = std::sqrt(mean3D_view[0]*mean3D_view[0] +
                           mean3D_view[1]*mean3D_view[1] +
                           mean3D_view[2]*mean3D_view[2]);
    float viewdir[3] = {mean3D_view[0]/vlen, mean3D_view[1]/vlen, mean3D_view[2]/vlen};

    // gauss2view is row-major: T[row][col] = gauss2view[row*4+col]
    // We need T[axis] and T[2] as vec4 rows
    auto computeTheta = [&](int axis) -> std::pair<float, float> {
        float theta_mu = std::atan2(axis == 0 ? viewdir[0] : viewdir[1], viewdir[2]);

        // T[axis] row and T[2] row
        const float* Ta = gauss2view + axis * 4;
        const float* T2 = gauss2view + 2 * 4;

        // dot(t, T[axis] * T[axis])
        float squared_axis = 0, squared_z = 0, mid_val = 0;
        for (int i = 0; i < 4; i++) {
            squared_axis += t[i] * Ta[i] * Ta[i];
            squared_z += t[i] * T2[i] * T2[i];
            mid_val += t[i] * Ta[i] * T2[i];
        }

        float inside_sqrt = mid_val * mid_val - squared_z * squared_axis;

        float result_lo = -((float)M_PI / 2.0f - 1e-5f);
        float result_hi = (float)M_PI / 2.0f - 1e-5f;

        if (inside_sqrt > 0.0f) {
            float sqrt_val = std::sqrt(inside_sqrt);
            float theta_axis_0 = std::atan2(-(mid_val + sqrt_val), -squared_z);
            float theta_axis_1 = std::atan2(-(mid_val - sqrt_val), -squared_z);

            // Rotate to get correct order
            while (theta_axis_0 > theta_mu) theta_axis_0 -= (float)M_PI;
            while (theta_axis_0 < (theta_mu - (float)M_PI)) theta_axis_0 += (float)M_PI;
            while (theta_axis_1 < theta_mu) theta_axis_1 += (float)M_PI;
            while (theta_axis_1 > (theta_mu + (float)M_PI)) theta_axis_1 -= (float)M_PI;

            float norm0 = normalizeAngle(theta_axis_0);
            float norm1 = normalizeAngle(theta_axis_1);
            if (theta_mu < 0.0f && std::fabs(norm0) < std::fabs(norm1)) {
                theta_axis_0 += 2.0f * (float)M_PI;
                theta_axis_1 += 2.0f * (float)M_PI;
            } else if (theta_mu > 0.0f && std::fabs(norm1) < std::fabs(norm0)) {
                theta_axis_0 -= 2.0f * (float)M_PI;
                theta_axis_1 -= 2.0f * (float)M_PI;
            }

            result_lo = std::max(result_lo, theta_axis_0);
            result_hi = std::min(result_hi, theta_axis_1);
        }

        return {std::tan(result_lo), std::tan(result_hi)};
    };

    auto [bx_lo, bx_hi] = computeTheta(0);
    auto [by_lo, by_hi] = computeTheta(1);

    float bounds_x_lo = W / 2.0f + focal_x * bx_lo;
    float bounds_x_hi = W / 2.0f + focal_x * bx_hi;
    float bounds_y_lo = H / 2.0f + focal_y * by_lo;
    float bounds_y_hi = H / 2.0f + focal_y * by_hi;

    mean2D[0] = (bounds_x_hi + bounds_x_lo) / 2.0f;
    mean2D[1] = (bounds_y_hi + bounds_y_lo) / 2.0f;
    extent2D[0] = (bounds_x_hi - bounds_x_lo) / 2.0f;
    extent2D[1] = (bounds_y_hi - bounds_y_lo) / 2.0f;
    return true;
}

float computeGauss2Screen(
    const float mean3D[3], const float scale[3], const float rot[4],
    float scale_modifier, const float cam_pos[3],
    const float* viewproj_matrix, const float* view_matrix,
    float focal, float kernel_size, float filter_3d,
    int W, int H,
    float gauss2screen_out[16], float gauss2view_out[16])
{
    float R[3][3];
    quat2mat(rot, R);

    // Compute dilated scale with mip filter
    float scale_dilated[3] = {scale[0], scale[1], scale[2]};
    float dilation_factor = 1.0f;

    if (kernel_size > 0.0f) {
        // viewray_world = normalize(mean3D - cam_pos)
        float vr[3] = {mean3D[0]-cam_pos[0], mean3D[1]-cam_pos[1], mean3D[2]-cam_pos[2]};
        float vr_len = std::sqrt(vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]);
        vr[0]/=vr_len; vr[1]/=vr_len; vr[2]/=vr_len;

        // viewray_gauss = R * viewray_world (R is row-major, so this is R * v)
        float vrg[3];
        for (int i = 0; i < 3; i++)
            vrg[i] = R[i][0]*vr[0] + R[i][1]*vr[1] + R[i][2]*vr[2];
        float r_sq[3] = {vrg[0]*vrg[0], vrg[1]*vrg[1], vrg[2]*vrg[2]};

        // mean3D in view space
        float mean_view[3];
        transformPoint4x3(mean3D, view_matrix, mean_view);
        float scale_mip = sq(mean_view[2] / focal) * kernel_size;
        scale_mip = std::max(sq(filter_3d), scale_mip);

        scale_dilated[0] = std::sqrt(scale[0]*scale[0] + scale_mip);
        scale_dilated[1] = std::sqrt(scale[1]*scale[1] + scale_mip);
        scale_dilated[2] = std::sqrt(scale[2]*scale[2] + scale_mip);

        // Compute dilation factor
        float s[3] = {scale[0]*scale[0], scale[1]*scale[1], scale[2]*scale[2]};
        float s_dil[3] = {scale_dilated[0]*scale_dilated[0],
                          scale_dilated[1]*scale_dilated[1],
                          scale_dilated[2]*scale_dilated[2]};

        float det_mul = r_sq[0]*(s[1]*s[2]) + r_sq[1]*(s[2]*s[0]) + r_sq[2]*(s[0]*s[1]);
        float det_mul_dil = r_sq[0]*(s_dil[1]*s_dil[2]) + r_sq[1]*(s_dil[2]*s_dil[0]) + r_sq[2]*(s_dil[0]*s_dil[1]);
        dilation_factor = std::sqrt(det_mul / det_mul_dil);
    }

    // Build L = transpose(S * R), where S = diag(scale_dilated) * sqrt(scale_modifier)
    float sm = std::sqrt(scale_modifier);
    float sd[3] = {scale_dilated[0]*sm, scale_dilated[1]*sm, scale_dilated[2]*sm};

    // S*R [i][j] = sd[i] * R[i][j]
    // L = transpose(S*R): L[j][i] = sd[i] * R[i][j]
    // L is stored as L[row][col], where L[row][col] = sd[col] * R[col][row]
    float L[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            L[i][j] = sd[j] * R[j][i];

    // gauss2world (column-major 4x4):
    // col0 = (L[0][0], L[1][0], L[2][0], 0)
    // col1 = (L[0][1], L[1][1], L[2][1], 0)
    // col2 = (L[0][2], L[1][2], L[2][2], 0)
    // col3 = (mean3D[0], mean3D[1], mean3D[2], 1)
    float g2w[16];
    for (int col = 0; col < 3; col++) {
        g2w[col*4+0] = L[0][col];
        g2w[col*4+1] = L[1][col];
        g2w[col*4+2] = L[2][col];
        g2w[col*4+3] = 0.0f;
    }
    g2w[12] = mean3D[0]; g2w[13] = mean3D[1]; g2w[14] = mean3D[2]; g2w[15] = 1.0f;

    // viewport matrix (column-major):
    // Maps NDC to pixel coords: x_pixel = W/2 * x_ndc + W/2 - 0.5
    float vp[16];
    std::memset(vp, 0, sizeof(vp));
    vp[0]  = W / 2.0f;   // col0 row0
    vp[5]  = H / 2.0f;   // col1 row1
    vp[10] = 1.0f;       // col2 row2
    vp[15] = 1.0f;       // col3 row3
    vp[12] = W / 2.0f - 0.5f; // col3 row0
    vp[13] = H / 2.0f - 0.5f; // col3 row1

    // world2screen = viewport * viewproj
    float w2s[16];
    mat4Mul(vp, viewproj_matrix, w2s);

    // gauss2screen_colmajor = world2screen * gauss2world
    float g2s_cm[16];
    mat4Mul(w2s, g2w, g2s_cm);

    // Store as row-major (transpose of column-major) -- matching AAA-Gaussians convention
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            gauss2screen_out[row*4+col] = g2s_cm[col*4+row];

    // gauss2view_colmajor = view_matrix * gauss2world
    float g2v_cm[16];
    mat4Mul(view_matrix, g2w, g2v_cm);

    // Store as row-major
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            gauss2view_out[row*4+col] = g2v_cm[col*4+row];

    return dilation_factor;
}

// maxContribRay for per-pixel evaluation (used by rasterizer)
// This is exposed for direct use:
float maxContribRayPixel(const float plane_a[4], const float plane_b[4], float max_pos[4]) {
    return maxContribRay(plane_a, plane_b, max_pos);
}

void computeCov3DInv(const float scale_dilated[3], float scale_modifier,
                     const float rot[4], float cov3D_inv[6]) {
    float R[3][3];
    quat2mat(rot, R);

    float sm = std::sqrt(scale_modifier);
    float inv_sd2[3];
    for (int j = 0; j < 3; j++) {
        float s = scale_dilated[j] * sm;
        inv_sd2[j] = 1.0f / (s * s);
    }

    // cov3D_inv = R * diag(1/sd^2) * R^T
    // R is row-major: R[row][col]
    // element (r,c) = sum_k R[k][r] * inv_sd2[k] * R[k][c]
    int idx = 0;
    for (int r = 0; r < 3; r++) {
        for (int c = r; c < 3; c++) {
            float val = 0;
            for (int k = 0; k < 3; k++)
                val += R[k][r] * inv_sd2[k] * R[k][c];
            cov3D_inv[idx++] = val;
        }
    }
}

float depthAlongRay(const float cov3D_inv[6], const float mean_offset[3],
                    const float viewdir[3]) {
    // Sigma_inv * viewdir (symmetric matrix)
    float Sv[3];
    Sv[0] = cov3D_inv[0]*viewdir[0] + cov3D_inv[1]*viewdir[1] + cov3D_inv[2]*viewdir[2];
    Sv[1] = cov3D_inv[1]*viewdir[0] + cov3D_inv[3]*viewdir[1] + cov3D_inv[4]*viewdir[2];
    Sv[2] = cov3D_inv[2]*viewdir[0] + cov3D_inv[4]*viewdir[1] + cov3D_inv[5]*viewdir[2];

    float num = mean_offset[0]*Sv[0] + mean_offset[1]*Sv[1] + mean_offset[2]*Sv[2];
    float den = viewdir[0]*Sv[0] + viewdir[1]*Sv[1] + viewdir[2]*Sv[2];

    if (std::fabs(den) < 1e-10f)
        return std::sqrt(mean_offset[0]*mean_offset[0] + mean_offset[1]*mean_offset[1] + mean_offset[2]*mean_offset[2]);
    return num / den;
}

void pixelToWorldDir(float px, float py, int width, int height,
                     const float inverse_vp[16], const float cam_pos[3],
                     float dir_out[3]) {
    // Pixel to NDC
    float ndc_x = (2.0f * px + 1.0f) / (float)width - 1.0f;
    float ndc_y = (2.0f * py + 1.0f) / (float)height - 1.0f;

    // NDC to world via inverse VP (column-major)
    float p[4] = {ndc_x, ndc_y, 0.0f, 1.0f};
    float wp[4];
    for (int i = 0; i < 4; i++)
        wp[i] = inverse_vp[i]*p[0] + inverse_vp[4+i]*p[1] + inverse_vp[8+i]*p[2] + inverse_vp[12+i]*p[3];
    float w_inv = 1.0f / (wp[3] + 1e-10f);

    dir_out[0] = wp[0]*w_inv - cam_pos[0];
    dir_out[1] = wp[1]*w_inv - cam_pos[1];
    dir_out[2] = wp[2]*w_inv - cam_pos[2];
    float len = std::sqrt(dir_out[0]*dir_out[0] + dir_out[1]*dir_out[1] + dir_out[2]*dir_out[2]);
    if (len > 1e-10f) { dir_out[0]/=len; dir_out[1]/=len; dir_out[2]/=len; }
}

bool invertMatrix4x4(const float m[16], float inv[16]) {
    // Column-major 4x4 inverse using cofactors
    // m[col*4+row]
    float a00=m[0], a10=m[1], a20=m[2], a30=m[3];
    float a01=m[4], a11=m[5], a21=m[6], a31=m[7];
    float a02=m[8], a12=m[9], a22=m[10], a32=m[11];
    float a03=m[12], a13=m[13], a23=m[14], a33=m[15];

    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;

    float det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
    if (std::fabs(det) < 1e-12f) return false;
    float inv_det = 1.0f / det;

    inv[0]  = ( a11*b11 - a12*b10 + a13*b09) * inv_det;
    inv[1]  = (-a10*b11 + a12*b08 - a13*b07) * inv_det;
    inv[2]  = ( a10*b10 - a11*b08 + a13*b06) * inv_det;
    inv[3]  = (-a10*b09 + a11*b07 - a12*b06) * inv_det;
    inv[4]  = (-a01*b11 + a02*b10 - a03*b09) * inv_det;
    inv[5]  = ( a00*b11 - a02*b08 + a03*b07) * inv_det;
    inv[6]  = (-a00*b10 + a01*b08 - a03*b06) * inv_det;
    inv[7]  = ( a00*b09 - a01*b07 + a02*b06) * inv_det;
    inv[8]  = ( a31*b05 - a32*b04 + a33*b03) * inv_det;
    inv[9]  = (-a30*b05 + a32*b02 - a33*b01) * inv_det;
    inv[10] = ( a30*b04 - a31*b02 + a33*b00) * inv_det;
    inv[11] = (-a30*b03 + a31*b01 - a32*b00) * inv_det;
    inv[12] = (-a21*b05 + a22*b04 - a23*b03) * inv_det;
    inv[13] = ( a20*b05 - a22*b02 + a23*b01) * inv_det;
    inv[14] = (-a20*b04 + a21*b02 - a23*b00) * inv_det;
    inv[15] = ( a20*b03 - a21*b01 + a22*b00) * inv_det;

    return true;
}
