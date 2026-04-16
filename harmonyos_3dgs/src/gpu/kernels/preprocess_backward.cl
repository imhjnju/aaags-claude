// preprocess_backward.cl -- Backward pass for 3DGS preprocessing
// Each work-item handles ONE Gaussian (embarrassingly parallel).
// Implements 4 gradient chains:
//   Chain 1: d_conics -> d_cov2D -> d_cov3D -> d_scales, d_rotations
//   Chain 2: d_rgb -> d_sh_coeffs
//   Chain 3: d_opacity_2d -> d_raw_opacity
//   Chain 4: d_means2D -> d_positions

// SH constants
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

__kernel void preprocess_backward(
    // Model data (activated values)
    __global const float* restrict positions,    // [N*3]
    __global const float* restrict sh_coeffs,    // [N*max_coeffs*3]
    __global const float* restrict scales,       // [N*3]
    __global const float* restrict rotations,    // [N*4] normalized quaternion
    __global const float* restrict opacities,    // [N] sigmoid-activated
    // Camera
    __constant const float* restrict view_matrix,  // [16]
    __constant const float* restrict viewproj,     // [16]
    __constant const float* restrict cam_pos_buf,  // [3]
    // Dimensions and config
    const int N,
    const int width,
    const int height,
    const float tan_fovx,
    const float tan_fovy,
    const float scale_modifier,
    const int sh_degree,
    const int max_coeffs,
    // Forward cache
    __global const float* restrict cov2D,         // [N*3] filtered 2D covariance
    __global const float* restrict cov2D_det,     // [N] determinant
    __global const float* restrict p_view,        // [N*3] view-space position
    __global const float* restrict p_hom_w,       // [N] clip-space w
    __global const float* restrict means2D_fwd,   // [N*2] forward means2D
    __global const int*   restrict radii,         // [N]
    // Rasterizer gradients (input)
    __global const float* restrict d_means2D,    // [N*2]
    __global const float* restrict d_conics,     // [N*3]
    __global const float* restrict d_rgb,        // [N*3]
    __global const float* restrict d_opacities_2d,// [N]
    // Raw params (for quaternion normalization backward)
    __global const float* restrict raw_rotations, // [N*4]
    // Output gradients
    __global float* restrict d_raw_positions,    // [N*3]
    __global float* restrict d_raw_scales,       // [N*3]
    __global float* restrict d_raw_rotations,    // [N*4]
    __global float* restrict d_raw_sh_coeffs,    // [N*max_coeffs*3]
    __global float* restrict d_raw_opacities     // [N]
)
{
    int i = get_global_id(0);
    if (i >= N) return;

    // Skip invisible Gaussians
    if (radii[i] <= 0) {
        // Zero outputs for this Gaussian
        d_raw_positions[i*3] = 0; d_raw_positions[i*3+1] = 0; d_raw_positions[i*3+2] = 0;
        d_raw_scales[i*3] = 0; d_raw_scales[i*3+1] = 0; d_raw_scales[i*3+2] = 0;
        d_raw_rotations[i*4] = 0; d_raw_rotations[i*4+1] = 0;
        d_raw_rotations[i*4+2] = 0; d_raw_rotations[i*4+3] = 0;
        for (int k = 0; k < max_coeffs * 3; k++)
            d_raw_sh_coeffs[i * max_coeffs * 3 + k] = 0;
        d_raw_opacities[i] = 0;
        return;
    }

    float cam_pos[3] = { cam_pos_buf[0], cam_pos_buf[1], cam_pos_buf[2] };

    // ===================================================================
    // Chain 2: d_rgb -> d_sh_coeffs
    // ===================================================================
    {
        float pos[3] = { positions[i*3], positions[i*3+1], positions[i*3+2] };
        float ddx = pos[0] - cam_pos[0];
        float ddy = pos[1] - cam_pos[1];
        float ddz = pos[2] - cam_pos[2];
        float len = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        if (len < 1e-10f) len = 1e-10f;
        ddx /= len; ddy /= len; ddz /= len;

        float x = ddx, y = ddy, z = ddz;

        // Forward SH to determine clamp mask
        int mc3 = max_coeffs * 3;
        float rgb_raw[3];
        for (int ch = 0; ch < 3; ch++)
            rgb_raw[ch] = SH_C0 * sh_coeffs[i*mc3 + 0*3 + ch];

        if (sh_degree > 0) {
            for (int ch = 0; ch < 3; ch++) {
                rgb_raw[ch] += -SH_C1 * y * sh_coeffs[i*mc3 + 1*3 + ch]
                             +  SH_C1 * z * sh_coeffs[i*mc3 + 2*3 + ch]
                             + -SH_C1 * x * sh_coeffs[i*mc3 + 3*3 + ch];
            }
            if (sh_degree > 1) {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                for (int ch = 0; ch < 3; ch++) {
                    rgb_raw[ch] += SH_C2_0 * xy * sh_coeffs[i*mc3 + 4*3 + ch]
                                 + SH_C2_1 * yz * sh_coeffs[i*mc3 + 5*3 + ch]
                                 + SH_C2_2 * (2.0f*zz - xx - yy) * sh_coeffs[i*mc3 + 6*3 + ch]
                                 + SH_C2_3 * xz * sh_coeffs[i*mc3 + 7*3 + ch]
                                 + SH_C2_4 * (xx - yy) * sh_coeffs[i*mc3 + 8*3 + ch];
                }
                if (sh_degree > 2) {
                    for (int ch = 0; ch < 3; ch++) {
                        rgb_raw[ch] += SH_C3_0 * y * (3.0f*xx - yy) * sh_coeffs[i*mc3 + 9*3 + ch]
                                     + SH_C3_1 * xy * z * sh_coeffs[i*mc3 + 10*3 + ch]
                                     + SH_C3_2 * y * (4.0f*zz - xx - yy) * sh_coeffs[i*mc3 + 11*3 + ch]
                                     + SH_C3_3 * z * (2.0f*zz - 3.0f*xx - 3.0f*yy) * sh_coeffs[i*mc3 + 12*3 + ch]
                                     + SH_C3_4 * x * (4.0f*zz - xx - yy) * sh_coeffs[i*mc3 + 13*3 + ch]
                                     + SH_C3_5 * z * (xx - yy) * sh_coeffs[i*mc3 + 14*3 + ch]
                                     + SH_C3_6 * x * (xx - 3.0f*yy) * sh_coeffs[i*mc3 + 15*3 + ch];
                    }
                }
            }
        }

        float mask[3];
        for (int ch = 0; ch < 3; ch++)
            mask[ch] = (rgb_raw[ch] + 0.5f >= 0.0f) ? 1.0f : 0.0f;

        float d_rgb_eff[3];
        for (int ch = 0; ch < 3; ch++)
            d_rgb_eff[ch] = d_rgb[i*3 + ch] * mask[ch];

        // Zero all SH grads first
        for (int k = 0; k < mc3; k++)
            d_raw_sh_coeffs[i*mc3 + k] = 0;

        // Degree 0
        for (int ch = 0; ch < 3; ch++)
            d_raw_sh_coeffs[i*mc3 + 0*3 + ch] = SH_C0 * d_rgb_eff[ch];

        if (sh_degree > 0) {
            float basis1[3] = { -SH_C1 * y, SH_C1 * z, -SH_C1 * x };
            for (int k = 0; k < 3; k++)
                for (int ch = 0; ch < 3; ch++)
                    d_raw_sh_coeffs[i*mc3 + (1+k)*3 + ch] = basis1[k] * d_rgb_eff[ch];

            if (sh_degree > 1) {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                float basis2[5] = {
                    SH_C2_0 * xy, SH_C2_1 * yz,
                    SH_C2_2 * (2.0f*zz - xx - yy),
                    SH_C2_3 * xz, SH_C2_4 * (xx - yy)
                };
                for (int k = 0; k < 5; k++)
                    for (int ch = 0; ch < 3; ch++)
                        d_raw_sh_coeffs[i*mc3 + (4+k)*3 + ch] = basis2[k] * d_rgb_eff[ch];

                if (sh_degree > 2) {
                    float basis3[7] = {
                        SH_C3_0 * y * (3.0f*xx - yy),
                        SH_C3_1 * xy * z,
                        SH_C3_2 * y * (4.0f*zz - xx - yy),
                        SH_C3_3 * z * (2.0f*zz - 3.0f*xx - 3.0f*yy),
                        SH_C3_4 * x * (4.0f*zz - xx - yy),
                        SH_C3_5 * z * (xx - yy),
                        SH_C3_6 * x * (xx - 3.0f*yy)
                    };
                    for (int k = 0; k < 7; k++)
                        for (int ch = 0; ch < 3; ch++)
                            d_raw_sh_coeffs[i*mc3 + (9+k)*3 + ch] = basis3[k] * d_rgb_eff[ch];
                }
            }
        }
    }

    // ===================================================================
    // Chain 3: d_opacity_2d -> d_raw_opacity (sigmoid backward)
    // ===================================================================
    {
        float sigma = opacities[i];
        d_raw_opacities[i] = d_opacities_2d[i] * sigma * (1.0f - sigma);
    }

    // ===================================================================
    // Chain 1: d_conics -> d_cov2D -> d_cov3D -> d_scales, d_rotations
    // ===================================================================
    {
        float a = cov2D[i*3];
        float b = cov2D[i*3+1];
        float c = cov2D[i*3+2];
        float det = cov2D_det[i];

        if (fabs(det) < 1e-10f) {
            d_raw_scales[i*3] = 0; d_raw_scales[i*3+1] = 0; d_raw_scales[i*3+2] = 0;
            d_raw_rotations[i*4] = 0; d_raw_rotations[i*4+1] = 0;
            d_raw_rotations[i*4+2] = 0; d_raw_rotations[i*4+3] = 0;
            d_raw_positions[i*3] = 0; d_raw_positions[i*3+1] = 0; d_raw_positions[i*3+2] = 0;
            return;
        }

        float inv_det = 1.0f / det;
        float dc0 = d_conics[i*3];
        float dc1 = d_conics[i*3+1];
        float dc2 = d_conics[i*3+2];

        float inv_det2 = inv_det * inv_det;
        float d_a = dc0 * (-c*c * inv_det2)
                  + dc1 * (b*c * inv_det2)
                  + dc2 * (-b*b * inv_det2);
        float d_b = dc0 * (2.0f*b*c * inv_det2)
                  + dc1 * (-(a*c + b*b) * inv_det2)
                  + dc2 * (2.0f*a*b * inv_det2);
        float d_c_val = dc0 * (-b*b * inv_det2)
                  + dc1 * (a*b * inv_det2)
                  + dc2 * (-a*a * inv_det2);

        // --- Step 1b: d_cov2D -> d_cov3D ---
        float focal_x = (float)width / (2.0f * tan_fovx);
        float focal_y = (float)height / (2.0f * tan_fovy);

        float t[3];
        t[0] = p_view[i*3];
        t[1] = p_view[i*3+1];
        t[2] = p_view[i*3+2];

        float limx = 1.3f * tan_fovx;
        float limy = 1.3f * tan_fovy;
        float txtz = t[0] / t[2];
        float tytz = t[1] / t[2];
        t[0] = fmin(limx, fmax(-limx, txtz)) * t[2];
        t[1] = fmin(limy, fmax(-limy, tytz)) * t[2];

        float J[3][3] = {
            {focal_x / t[2], 0.0f, -(focal_x * t[0]) / (t[2] * t[2])},
            {0.0f, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])},
            {0.0f, 0.0f, 0.0f}
        };

        float W[3][3] = {
            {view_matrix[0], view_matrix[4], view_matrix[8]},
            {view_matrix[1], view_matrix[5], view_matrix[9]},
            {view_matrix[2], view_matrix[6], view_matrix[10]}
        };

        float T_mat[3][3];
        for (int col = 0; col < 3; col++)
            for (int row = 0; row < 3; row++) {
                T_mat[col][row] = 0;
                for (int k = 0; k < 3; k++)
                    T_mat[col][row] += W[k][row] * J[col][k];
            }

        float d_Vrk[3][3];
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++) {
                d_Vrk[k][j] = d_a * T_mat[0][k] * T_mat[0][j]
                             + d_b * T_mat[0][k] * T_mat[1][j]
                             + d_c_val * T_mat[1][k] * T_mat[1][j];
            }

        float d_cov3D[6];
        d_cov3D[0] = d_Vrk[0][0];
        d_cov3D[1] = d_Vrk[0][1] + d_Vrk[1][0];
        d_cov3D[2] = d_Vrk[0][2] + d_Vrk[2][0];
        d_cov3D[3] = d_Vrk[1][1];
        d_cov3D[4] = d_Vrk[1][2] + d_Vrk[2][1];
        d_cov3D[5] = d_Vrk[2][2];

        // --- Step 1c: d_cov3D -> d_M -> d_raw_scales, d_raw_rotations ---
        float sx = scale_modifier * scales[i*3];
        float sy = scale_modifier * scales[i*3+1];
        float sz = scale_modifier * scales[i*3+2];

        float r_q = rotations[i*4], x_q = rotations[i*4+1];
        float y_q = rotations[i*4+2], z_q = rotations[i*4+3];

        float R[3][3] = {
            {1.f - 2.f*(y_q*y_q + z_q*z_q), 2.f*(x_q*y_q + r_q*z_q),       2.f*(x_q*z_q - r_q*y_q)},
            {2.f*(x_q*y_q - r_q*z_q),       1.f - 2.f*(x_q*x_q + z_q*z_q), 2.f*(y_q*z_q + r_q*x_q)},
            {2.f*(x_q*z_q + r_q*y_q),       2.f*(y_q*z_q - r_q*x_q),       1.f - 2.f*(x_q*x_q + y_q*y_q)}
        };

        float s[3] = {sx, sy, sz};
        float M[3][3];
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                M[ii][jj] = s[ii] * R[ii][jj];

        float d_M[3][3];
        for (int k = 0; k < 3; k++) {
            d_M[k][0] = 2.0f*d_cov3D[0]*M[k][0] + d_cov3D[1]*M[k][1] + d_cov3D[2]*M[k][2];
            d_M[k][1] = d_cov3D[1]*M[k][0] + 2.0f*d_cov3D[3]*M[k][1] + d_cov3D[4]*M[k][2];
            d_M[k][2] = d_cov3D[2]*M[k][0] + d_cov3D[4]*M[k][1] + 2.0f*d_cov3D[5]*M[k][2];
        }

        float d_scale[3];
        float d_R[3][3];
        for (int k = 0; k < 3; k++) {
            d_scale[k] = 0.0f;
            for (int j = 0; j < 3; j++) {
                d_scale[k] += d_M[k][j] * R[k][j];
                d_R[k][j] = d_M[k][j] * s[k];
            }
        }

        for (int k = 0; k < 3; k++)
            d_raw_scales[i*3+k] = d_scale[k] * scale_modifier * scales[i*3+k];

        // d_R -> d_normalized_quat
        float d_qn[4] = {0, 0, 0, 0};

        d_qn[2] += d_R[0][0] * (-4.0f*y_q);
        d_qn[3] += d_R[0][0] * (-4.0f*z_q);

        d_qn[0] += d_R[0][1] * 2.0f*z_q;
        d_qn[1] += d_R[0][1] * 2.0f*y_q;
        d_qn[2] += d_R[0][1] * 2.0f*x_q;
        d_qn[3] += d_R[0][1] * 2.0f*r_q;

        d_qn[0] += d_R[0][2] * (-2.0f*y_q);
        d_qn[1] += d_R[0][2] * 2.0f*z_q;
        d_qn[2] += d_R[0][2] * (-2.0f*r_q);
        d_qn[3] += d_R[0][2] * 2.0f*x_q;

        d_qn[0] += d_R[1][0] * (-2.0f*z_q);
        d_qn[1] += d_R[1][0] * 2.0f*y_q;
        d_qn[2] += d_R[1][0] * 2.0f*x_q;
        d_qn[3] += d_R[1][0] * (-2.0f*r_q);

        d_qn[1] += d_R[1][1] * (-4.0f*x_q);
        d_qn[3] += d_R[1][1] * (-4.0f*z_q);

        d_qn[0] += d_R[1][2] * 2.0f*x_q;
        d_qn[1] += d_R[1][2] * 2.0f*r_q;
        d_qn[2] += d_R[1][2] * 2.0f*z_q;
        d_qn[3] += d_R[1][2] * 2.0f*y_q;

        d_qn[0] += d_R[2][0] * 2.0f*y_q;
        d_qn[1] += d_R[2][0] * 2.0f*z_q;
        d_qn[2] += d_R[2][0] * 2.0f*r_q;
        d_qn[3] += d_R[2][0] * 2.0f*x_q;

        d_qn[0] += d_R[2][1] * (-2.0f*x_q);
        d_qn[1] += d_R[2][1] * (-2.0f*r_q);
        d_qn[2] += d_R[2][1] * 2.0f*z_q;
        d_qn[3] += d_R[2][1] * 2.0f*y_q;

        d_qn[1] += d_R[2][2] * (-4.0f*x_q);
        d_qn[2] += d_R[2][2] * (-4.0f*y_q);

        // d_normalized_quat -> d_raw_quat
        float q_raw[4] = { raw_rotations[i*4], raw_rotations[i*4+1],
                           raw_rotations[i*4+2], raw_rotations[i*4+3] };
        float q_raw_len = sqrt(q_raw[0]*q_raw[0] + q_raw[1]*q_raw[1] +
                               q_raw[2]*q_raw[2] + q_raw[3]*q_raw[3]);
        if (q_raw_len < 1e-12f) q_raw_len = 1e-12f;
        float inv_len = 1.0f / q_raw_len;

        float dot_dqn_qn = d_qn[0]*r_q + d_qn[1]*x_q + d_qn[2]*y_q + d_qn[3]*z_q;

        d_raw_rotations[i*4+0] = (d_qn[0] - r_q * dot_dqn_qn) * inv_len;
        d_raw_rotations[i*4+1] = (d_qn[1] - x_q * dot_dqn_qn) * inv_len;
        d_raw_rotations[i*4+2] = (d_qn[2] - y_q * dot_dqn_qn) * inv_len;
        d_raw_rotations[i*4+3] = (d_qn[3] - z_q * dot_dqn_qn) * inv_len;

        // ===================================================================
        // Chain 4: d_means2D -> d_raw_positions
        // ===================================================================
        float d_pixel_x = d_means2D[i*2];
        float d_pixel_y = d_means2D[i*2+1];

        float d_ndc_x = d_pixel_x * (float)width * 0.5f;
        float d_ndc_y = d_pixel_y * (float)height * 0.5f;

        float w_hom = p_hom_w[i] + 1e-7f;
        float inv_w = 1.0f / w_hom;

        float pixel_x_cached = means2D_fwd[i*2];
        float pixel_y_cached = means2D_fwd[i*2+1];
        float ndc_x = (2.0f * pixel_x_cached + 1.0f) / (float)width - 1.0f;
        float ndc_y = (2.0f * pixel_y_cached + 1.0f) / (float)height - 1.0f;

        float d_p_hom[4];
        d_p_hom[0] = d_ndc_x * inv_w;
        d_p_hom[1] = d_ndc_y * inv_w;
        d_p_hom[2] = 0.0f;
        d_p_hom[3] = -(ndc_x * d_ndc_x + ndc_y * d_ndc_y) * inv_w;

        for (int dim = 0; dim < 3; dim++) {
            d_raw_positions[i*3 + dim] =
                viewproj[dim*4+0] * d_p_hom[0] +
                viewproj[dim*4+1] * d_p_hom[1] +
                viewproj[dim*4+2] * d_p_hom[2] +
                viewproj[dim*4+3] * d_p_hom[3];
        }
    }
}
