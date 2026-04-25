#include "cpu/preprocessor_backward_cpu.h"
#include "math_utils.h"
#include "sh_eval.h"
#include <cmath>
#include <algorithm>
#include <cstring>

void computeColorFromSH_backward(int degree, int max_coeffs,
                                   const float* sh_coeffs,
                                   const float* pos, const float* cam_pos,
                                   const float* d_rgb, float* d_sh_coeffs,
                                   bool training, float* d_pos) {
    // Zero output
    std::memset(d_sh_coeffs, 0, max_coeffs * 3 * sizeof(float));

    // Compute normalized view direction (same as forward)
    float dx = pos[0] - cam_pos[0];
    float dy = pos[1] - cam_pos[1];
    float dz = pos[2] - cam_pos[2];
    float len = std::sqrt(dx*dx + dy*dy + dz*dz);
    dx /= len; dy /= len; dz /= len;

    float x = dx, y = dy, z = dz;

    // Recompute forward SH value per channel to determine clamp mask
    float rgb_raw[3];
    for (int ch = 0; ch < 3; ch++)
        rgb_raw[ch] = SH_C0 * sh_coeffs[0 * 3 + ch];

    if (degree > 0) {
        for (int ch = 0; ch < 3; ch++) {
            rgb_raw[ch] += -SH_C1 * y * sh_coeffs[1 * 3 + ch]
                         +  SH_C1 * z * sh_coeffs[2 * 3 + ch]
                         + -SH_C1 * x * sh_coeffs[3 * 3 + ch];
        }
        if (degree > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            for (int ch = 0; ch < 3; ch++) {
                rgb_raw[ch] += SH_C2[0] * xy * sh_coeffs[4 * 3 + ch]
                             + SH_C2[1] * yz * sh_coeffs[5 * 3 + ch]
                             + SH_C2[2] * (2.0f * zz - xx - yy) * sh_coeffs[6 * 3 + ch]
                             + SH_C2[3] * xz * sh_coeffs[7 * 3 + ch]
                             + SH_C2[4] * (xx - yy) * sh_coeffs[8 * 3 + ch];
            }
            if (degree > 2) {
                for (int ch = 0; ch < 3; ch++) {
                    rgb_raw[ch] += SH_C3[0] * y * (3.0f*xx - yy) * sh_coeffs[9 * 3 + ch]
                                 + SH_C3[1] * xy * z * sh_coeffs[10 * 3 + ch]
                                 + SH_C3[2] * y * (4.0f*zz - xx - yy) * sh_coeffs[11 * 3 + ch]
                                 + SH_C3[3] * z * (2.0f*zz - 3.0f*xx - 3.0f*yy) * sh_coeffs[12 * 3 + ch]
                                 + SH_C3[4] * x * (4.0f*zz - xx - yy) * sh_coeffs[13 * 3 + ch]
                                 + SH_C3[5] * z * (xx - yy) * sh_coeffs[14 * 3 + ch]
                                 + SH_C3[6] * x * (xx - 3.0f*yy) * sh_coeffs[15 * 3 + ch];
                }
            }
        }
    }

    // Clamp mask: gradient passes through only if rgb_raw + 0.5 >= 0
    // (i.e., the lower clamp max(0, ...) was NOT active)
    float mask[3];
    for (int ch = 0; ch < 3; ch++) {
        mask[ch] = (rgb_raw[ch] + 0.5f >= 0.0f) ? 1.0f : 0.0f;
    }

    // Effective d_rgb after clamp mask
    float d_rgb_eff[3];
    for (int ch = 0; ch < 3; ch++) {
        d_rgb_eff[ch] = d_rgb[ch] * mask[ch];
    }

    // Compute SH basis values and accumulate gradients
    // Degree 0: basis = SH_C0
    for (int ch = 0; ch < 3; ch++)
        d_sh_coeffs[0 * 3 + ch] = SH_C0 * d_rgb_eff[ch];

    if (degree > 0) {
        // Degree 1
        float basis1[3] = { -SH_C1 * y, SH_C1 * z, -SH_C1 * x };
        for (int k = 0; k < 3; k++) {
            for (int ch = 0; ch < 3; ch++) {
                d_sh_coeffs[(1 + k) * 3 + ch] = basis1[k] * d_rgb_eff[ch];
            }
        }

        if (degree > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            float basis2[5] = {
                SH_C2[0] * xy,
                SH_C2[1] * yz,
                SH_C2[2] * (2.0f * zz - xx - yy),
                SH_C2[3] * xz,
                SH_C2[4] * (xx - yy)
            };
            for (int k = 0; k < 5; k++) {
                for (int ch = 0; ch < 3; ch++) {
                    d_sh_coeffs[(4 + k) * 3 + ch] = basis2[k] * d_rgb_eff[ch];
                }
            }

            if (degree > 2) {
                float basis3[7] = {
                    SH_C3[0] * y * (3.0f*xx - yy),
                    SH_C3[1] * xy * z,
                    SH_C3[2] * y * (4.0f*zz - xx - yy),
                    SH_C3[3] * z * (2.0f*zz - 3.0f*xx - 3.0f*yy),
                    SH_C3[4] * x * (4.0f*zz - xx - yy),
                    SH_C3[5] * z * (xx - yy),
                    SH_C3[6] * x * (xx - 3.0f*yy)
                };
                for (int k = 0; k < 7; k++) {
                    for (int ch = 0; ch < 3; ch++) {
                        d_sh_coeffs[(9 + k) * 3 + ch] = basis3[k] * d_rgb_eff[ch];
                    }
                }
            }
        }
    }

    // Position gradient through SH view-direction dependency (degree >= 1)
    if (d_pos && degree > 0) {
        // d_color/d_pos = d_color/d_dir * d_dir/d_pos
        //
        // d_dir/d_pos: direction = normalize(pos - cam_pos)
        //   d_dir_i/d_pos_j = (delta_ij - dir_i*dir_j) / |pos - cam_pos|
        //
        // d_color/d_dir: partial derivatives of SH wrt x,y,z (direction components)

        float d_color_d_dir[3][3] = {};  // [dir_component][rgb_channel]

        // Degree 1: color += -SH_C1*y*sh[1] + SH_C1*z*sh[2] - SH_C1*x*sh[3]
        //   d/dx = -SH_C1 * sh[3]
        //   d/dy = -SH_C1 * sh[1]
        //   d/dz =  SH_C1 * sh[2]
        for (int ch = 0; ch < 3; ch++) {
            d_color_d_dir[0][ch] += -SH_C1 * sh_coeffs[3*3 + ch];  // d/dx
            d_color_d_dir[1][ch] += -SH_C1 * sh_coeffs[1*3 + ch];  // d/dy
            d_color_d_dir[2][ch] +=  SH_C1 * sh_coeffs[2*3 + ch];  // d/dz
        }

        if (degree > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            // Degree 2 derivatives w.r.t. x,y,z:
            // basis2[0] = C2[0]*xy  -> d/dx = C2[0]*y, d/dy = C2[0]*x
            // basis2[1] = C2[1]*yz  -> d/dy = C2[1]*z, d/dz = C2[1]*y
            // basis2[2] = C2[2]*(2zz-xx-yy) -> d/dx = -2*C2[2]*x, d/dy = -2*C2[2]*y, d/dz = 4*C2[2]*z
            // basis2[3] = C2[3]*xz  -> d/dx = C2[3]*z, d/dz = C2[3]*x
            // basis2[4] = C2[4]*(xx-yy) -> d/dx = 2*C2[4]*x, d/dy = -2*C2[4]*y
            for (int ch = 0; ch < 3; ch++) {
                float s4 = sh_coeffs[4*3+ch], s5 = sh_coeffs[5*3+ch];
                float s6 = sh_coeffs[6*3+ch], s7 = sh_coeffs[7*3+ch], s8 = sh_coeffs[8*3+ch];
                d_color_d_dir[0][ch] += SH_C2[0]*y*s4 - 2.0f*SH_C2[2]*x*s6 + SH_C2[3]*z*s7 + 2.0f*SH_C2[4]*x*s8;
                d_color_d_dir[1][ch] += SH_C2[0]*x*s4 + SH_C2[1]*z*s5 - 2.0f*SH_C2[2]*y*s6 - 2.0f*SH_C2[4]*y*s8;
                d_color_d_dir[2][ch] += SH_C2[1]*y*s5 + 4.0f*SH_C2[2]*z*s6 + SH_C2[3]*x*s7;
            }

            if (degree > 2) {
                // Degree 3 derivatives w.r.t. x,y,z:
                for (int ch = 0; ch < 3; ch++) {
                    float s9  = sh_coeffs[9*3+ch],  s10 = sh_coeffs[10*3+ch];
                    float s11 = sh_coeffs[11*3+ch], s12 = sh_coeffs[12*3+ch];
                    float s13 = sh_coeffs[13*3+ch], s14 = sh_coeffs[14*3+ch], s15 = sh_coeffs[15*3+ch];
                    // d/dx:
                    d_color_d_dir[0][ch] +=
                        SH_C3[0]*6.0f*x*y*s9 + SH_C3[1]*y*z*s10 + SH_C3[2]*(-2.0f*x*y)*s11
                        + SH_C3[3]*(-6.0f*x*z)*s12 + SH_C3[4]*(4.0f*zz - 3.0f*xx - yy)*s13
                        + SH_C3[5]*2.0f*x*z*s14 + SH_C3[6]*(3.0f*xx - 3.0f*yy)*s15;
                    // d/dy:
                    d_color_d_dir[1][ch] +=
                        SH_C3[0]*(3.0f*xx - 3.0f*yy)*s9 + SH_C3[1]*x*z*s10
                        + SH_C3[2]*(4.0f*zz - xx - 3.0f*yy)*s11 + SH_C3[3]*(-6.0f*y*z)*s12
                        + SH_C3[4]*(-2.0f*x*y)*s13 - SH_C3[5]*2.0f*y*z*s14 + SH_C3[6]*(-6.0f*x*y)*s15;
                    // d/dz:
                    d_color_d_dir[2][ch] +=
                        SH_C3[1]*x*y*s10 + SH_C3[2]*8.0f*y*z*s11
                        + SH_C3[3]*(6.0f*zz - 3.0f*xx - 3.0f*yy)*s12
                        + SH_C3[4]*8.0f*x*z*s13 + SH_C3[5]*(xx - yy)*s14;
                }
            }
        }

        // Apply clamp mask (same mask as for d_sh_coeffs)
        float mask[3];
        for (int ch = 0; ch < 3; ch++)
            mask[ch] = (rgb_raw[ch] + 0.5f >= 0.0f) ? 1.0f : 0.0f;

        // Combine: d_loss/d_dir_k = sum_ch d_rgb[ch] * mask[ch] * d_color_d_dir[k][ch]
        float d_dir[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
            for (int ch = 0; ch < 3; ch++)
                d_dir[k] += d_rgb_eff[ch] * d_color_d_dir[k][ch];

        // Jacobian of normalize: d_dir/d_pos = (I - d*d^T) / len
        // d_pos[i] = sum_k d_dir[k] * (delta_ik - dir_i*dir_k) / len
        float inv_len = 1.0f / len;
        float dot_d_dir = d_dir[0]*x + d_dir[1]*y + d_dir[2]*z;
        d_pos[0] += (d_dir[0] - x * dot_d_dir) * inv_len;
        d_pos[1] += (d_dir[1] - y * dot_d_dir) * inv_len;
        d_pos[2] += (d_dir[2] - z * dot_d_dir) * inv_len;
    }
}

void PreprocessorBackwardCPU::backward(const GaussianData& g, const Camera& cam,
                                        const RenderConfig& cfg, const ForwardCache& cache,
                                        const RasterGradOutput& rgrad, const RawGaussianParams& raw,
                                        GradientOutput& grads) {
    const int N = g.count;
    float focal_x = cam.width / (2.0f * cam.tan_fovx);
    float focal_y = cam.height / (2.0f * cam.tan_fovy);

    for (int i = 0; i < N; i++) {
        // Skip invisible Gaussians
        if (cache.pre->radii[i] <= 0)
            continue;

        // Chain 2: d_rgb -> d_sh_coeffs + d_position (SH view-direction dependency)
        computeColorFromSH_backward(
            g.sh_degree, g.max_coeffs,
            &g.sh_coeffs[i * g.max_coeffs * 3],
            &g.positions[i * 3],
            cam.cam_pos,
            &rgrad.d_rgb[i * 3],
            &grads.d_raw_sh_coeffs[i * g.max_coeffs * 3],
            true,  // training mode
            &grads.d_raw_positions[i * 3]  // accumulate SH->position gradient
        );

        // Chain 3: d_opacity_2d -> d_raw_opacity (sigmoid backward + h_conv_scaling)
        // Forward (matching VK preprocess.comp and the unconditional proper_ewa_scaling
        // in CUDA forward_common.h dilateCov2D):
        //   opa_2d = sigma(raw) * h_conv,  h_conv = sqrt(max(0.000025, det_orig / det_dilated))
        // cache.cov2D stores POST-dilation (a = a_raw+0.3, c = c_raw+0.3, b unchanged).
        float sigma = g.opacities[i];  // already-activated sigmoid value
        float a_post = cache.cov2D[i*3];
        float b_post = cache.cov2D[i*3+1];
        float c_post = cache.cov2D[i*3+2];
        float det_dilated_bwd = cache.cov2D_det[i];
        float a_raw_cov = a_post - 0.3f;
        float c_raw_cov = c_post - 0.3f;
        float det_orig_bwd = a_raw_cov * c_raw_cov - b_post * b_post;
        float h_conv_bwd = 1.0f;
        float d_det_orig = 0.0f;
        float d_det_dilated = 0.0f;
        if (det_dilated_bwd != 0.0f) {
            float r_bwd = det_orig_bwd / det_dilated_bwd;
            bool clamped = r_bwd <= 0.000025f;
            float r_eff = clamped ? 0.000025f : r_bwd;
            h_conv_bwd = std::sqrt(r_eff);
            if (!clamped && h_conv_bwd > 0.0f) {
                // d_h = rgrad.d_opacities_2d[i] * sigma
                // d_r = d_h * 0.5 / h_conv
                // d_det_orig    = d_r / det_dilated
                // d_det_dilated = d_r * (-det_orig / det_dilated^2)
                float d_h = rgrad.d_opacities_2d[i] * sigma;
                float d_r = d_h * 0.5f / h_conv_bwd;
                d_det_orig    = d_r / det_dilated_bwd;
                d_det_dilated = d_r * (-det_orig_bwd / (det_dilated_bwd * det_dilated_bwd));
            }
        }
        // d_raw_opacity includes the h_conv factor since opa_2d = sigma * h_conv.
        grads.d_raw_opacities[i] = rgrad.d_opacities_2d[i] * sigma * (1.0f - sigma) * h_conv_bwd;

        // ===================================================================
        // Chain 1: d_conics -> d_cov2D -> d_cov3D -> d_M -> d_raw_scales, d_raw_rotations
        // ===================================================================

        // --- Step 1a: d_conics -> d_cov2D ---
        // Forward: conic = {c/det, -b/det, a/det}
        //   where a = cov2D[0] (filtered), b = cov2D[1], c = cov2D[2] (filtered)
        //   det = a*c - b*b
        float a = cache.cov2D[i*3];
        float b = cache.cov2D[i*3+1];
        float c = cache.cov2D[i*3+2];
        float det = cache.cov2D_det[i];
        if (det == 0.0f) continue;
        float inv_det = 1.0f / det;

        // conic[0] = c/det,  conic[1] = -b/det,  conic[2] = a/det
        // The 2x2 inverse matrix is:
        //   C_inv = [[c/det, -b/det], [-b/det, a/det]]
        // which maps to conic[0]=C_inv[0][0], conic[1]=C_inv[0][1]=C_inv[1][0], conic[2]=C_inv[1][1]
        //
        // d_conic has 3 components corresponding to these.
        // Using d(C_inv) = -C_inv * dC * C_inv for symmetric 2x2:
        //
        // The Jacobian of the 2x2 symmetric matrix inverse:
        // Let C_inv = [[A,B],[B,D]] = [[c/det, -b/det],[-b/det, a/det]]
        // dA/da = -A^2, dA/db = 2*A*B, dA/dc = -B^2
        // dB/da = -A*B, dB/db = -(A*D + B^2), dB/dc = -B*D   (but with factor for off-diag doubling... actually no)
        // dD/da = -B^2, dD/db = 2*B*D, dD/dc = -D^2
        //
        // Wait -- more carefully. C = [[a,b],[b,c]]. C_inv = [[c,-b],[-b,a]] / det.
        // d(c/det)/da = d/da [c/(ac-b^2)] = -c^2/(ac-b^2)^2 = -(c/det)^2 = -A^2
        // d(c/det)/db = d/db [c/(ac-b^2)] = 2bc/(ac-b^2)^2 = 2*A*(-B) ... let me just compute directly.

        float dc0 = rgrad.d_conics[i*3];    // d_loss/d(conic[0]) = d_loss/d(c*inv_det)
        float dc1 = rgrad.d_conics[i*3+1];  // d_loss/d(conic[1]) = d_loss/d(-b*inv_det)
        float dc2 = rgrad.d_conics[i*3+2];  // d_loss/d(conic[2]) = d_loss/d(a*inv_det)

        // Using the identity: for symmetric C with inverse stored as (c/det, -b/det, a/det):
        // d_a = sum of d_conic[k] * d(conic[k])/da for k=0,1,2
        //
        // conic[0] = c * inv_det, where det = a*c - b^2
        //   d(conic[0])/da = c * (-c/det^2) = -c^2/det^2
        //   d(conic[0])/db = c * (2b/det^2) = 2bc/det^2
        //   d(conic[0])/dc = 1/det + c*(-a/det^2) = 1/det - ac/det^2 = (det - ac)/det^2 = -b^2/det^2... wait
        //     d(conic[0])/dc = d/dc [c/(ac-b^2)] = (ac-b^2 - c*a)/(ac-b^2)^2 = -b^2/det^2
        //     Hmm that's not right either. Let me redo:
        //     conic[0] = c/det = c/(ac-b^2)
        //     d/dc = [(ac-b^2) - c*a] / (ac-b^2)^2 = [ac-b^2-ac]/det^2 = -b^2/det^2

        // conic[1] = -b * inv_det = -b/(ac-b^2)
        //   d/da = -b * (-c/det^2) = bc/det^2
        //   d/db = [-det - (-b)*(-2b)] / det^2 = [-det - 2b^2]/det^2 = -(det+2b^2)/det^2 = -(ac+b^2)/det^2
        //     Actually: d/db [-b/(ac-b^2)] = [-(ac-b^2) - (-b)(-2b)] / (ac-b^2)^2
        //                                  = [-ac+b^2 - 2b^2] / det^2 = -(ac+b^2)/det^2
        //   d/dc = -b*(-a/det^2) = ab/det^2

        // conic[2] = a * inv_det = a/(ac-b^2)
        //   d/da = [(ac-b^2) - a*c] / det^2 = -b^2/det^2
        //   d/db = a*(2b/det^2) = 2ab/det^2
        //   d/dc = a*(-a/det^2) = -a^2/det^2

        float inv_det2 = inv_det * inv_det;
        float d_a = dc0 * (-c*c * inv_det2)
                  + dc1 * (b*c * inv_det2)
                  + dc2 * (-b*b * inv_det2);

        float d_b = dc0 * (2.0f*b*c * inv_det2)
                  + dc1 * (-(a*c + b*b) * inv_det2)
                  + dc2 * (2.0f*a*b * inv_det2);

        float d_c = dc0 * (-b*b * inv_det2)
                  + dc1 * (a*b * inv_det2)
                  + dc2 * (-a*a * inv_det2);

        // Add h_conv_scaling chain contributions (computed above in Chain 3).
        //   d_det_orig    / d_a = c_raw = c - 0.3        d_det_dilated / d_a = c (post-dilation)
        //   d_det_orig    / d_b = -2b                     d_det_dilated / d_b = -2b
        //   d_det_orig    / d_c = a_raw = a - 0.3        d_det_dilated / d_c = a
        d_a += d_det_orig * c_raw_cov + d_det_dilated * c_post;
        d_b += (d_det_orig + d_det_dilated) * (-2.0f * b_post);
        d_c += d_det_orig * a_raw_cov + d_det_dilated * a_post;

        // The +0.3 low-pass filter: a_filtered = a_raw + 0.3, c_filtered = c_raw + 0.3
        // Gradient passes through unchanged (constant offset).
        // d_a_raw = d_a, d_b_raw = d_b, d_c_raw = d_c

        // --- Step 1b: d_cov2D -> d_cov3D ---
        // Forward: cov2D = T^T * cov3D * T
        // where T[col][row] is computed as T = W * J (both column-major [col][row])
        // Recompute T from cached p_view
        float t[3];
        t[0] = cache.p_view[i*3];
        t[1] = cache.p_view[i*3+1];
        t[2] = cache.p_view[i*3+2];

        // Clamp to frustum (same as forward)
        float limx = 1.3f * cam.tan_fovx;
        float limy = 1.3f * cam.tan_fovy;
        float txtz = t[0] / t[2];
        float tytz = t[1] / t[2];
        t[0] = std::min(limx, std::max(-limx, txtz)) * t[2];
        t[1] = std::min(limy, std::max(-limy, tytz)) * t[2];

        // Jacobian J[col][row] (column-major)
        float J[3][3] = {
            {focal_x / t[2], 0.0f, -(focal_x * t[0]) / (t[2] * t[2])},
            {0.0f, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])},
            {0.0f, 0.0f, 0.0f}
        };

        // View rotation W[col][row] (column-major)
        const float* vm = cam.view_matrix;
        float W[3][3] = {
            {vm[0], vm[4], vm[8]},
            {vm[1], vm[5], vm[9]},
            {vm[2], vm[6], vm[10]}
        };

        // T = W * J: T[col][row] = sum_k W[k][row] * J[col][k]
        float T[3][3];
        for (int col = 0; col < 3; col++)
            for (int row = 0; row < 3; row++) {
                T[col][row] = 0;
                for (int k = 0; k < 3; k++)
                    T[col][row] += W[k][row] * J[col][k];
            }

        // Forward: result[c][r] = sum_{k,j} T[r][k] * Vrk[k][j] * T[c][j]
        // cov2D[0] = result[0][0], cov2D[1] = result[1][0], cov2D[2] = result[1][1]
        //
        // Backward: d_Vrk[k][j] = sum over outputs of d_output * d(output)/d(Vrk[k][j])
        // d(result[c][r])/d(Vrk[k][j]) = T[r][k] * T[c][j]
        //
        // d_Vrk[k][j] = d_a * T[0][k]*T[0][j]      (from result[0][0])
        //             + d_b * T[0][k]*T[1][j]        (from result[1][0])
        //             + d_c * T[1][k]*T[1][j]        (from result[1][1])

        float d_Vrk[3][3];
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++) {
                d_Vrk[k][j] = d_a * T[0][k] * T[0][j]
                             + d_b * T[0][k] * T[1][j]
                             + d_c * T[1][k] * T[1][j];
            }

        // Map d_Vrk (3x3 symmetric) to d_cov3D upper-triangle (6 elements)
        // cov3D indices: [0]=(0,0), [1]=(0,1), [2]=(0,2), [3]=(1,1), [4]=(1,2), [5]=(2,2)
        // For off-diagonal elements, the gradient accumulates from both (i,j) and (j,i)
        float d_cov3D[6];
        d_cov3D[0] = d_Vrk[0][0];
        d_cov3D[1] = d_Vrk[0][1] + d_Vrk[1][0];  // Vrk[0][1] and Vrk[1][0] both use cov3D[1]
        d_cov3D[2] = d_Vrk[0][2] + d_Vrk[2][0];
        d_cov3D[3] = d_Vrk[1][1];
        d_cov3D[4] = d_Vrk[1][2] + d_Vrk[2][1];
        d_cov3D[5] = d_Vrk[2][2];

        // --- Step 1c: d_cov3D -> d_M -> d_raw_scales, d_raw_rotations ---
        // Forward: Sigma = M^T * M, stored as upper triangle:
        //   cov3D[0] = sum_k M[k][0]^2          (0,0)
        //   cov3D[1] = sum_k M[k][0]*M[k][1]    (0,1)
        //   cov3D[2] = sum_k M[k][0]*M[k][2]    (0,2)
        //   cov3D[3] = sum_k M[k][1]^2          (1,1)
        //   cov3D[4] = sum_k M[k][1]*M[k][2]    (1,2)
        //   cov3D[5] = sum_k M[k][2]^2          (2,2)
        //
        // dL/dM[k][0] = 2*d_cov3D[0]*M[k][0] + d_cov3D[1]*M[k][1] + d_cov3D[2]*M[k][2]
        // dL/dM[k][1] = d_cov3D[1]*M[k][0] + 2*d_cov3D[3]*M[k][1] + d_cov3D[4]*M[k][2]
        // dL/dM[k][2] = d_cov3D[2]*M[k][0] + d_cov3D[4]*M[k][1] + 2*d_cov3D[5]*M[k][2]

        // Recompute M from scales and rotations
        float sx = cfg.scale_modifier * g.scales[i*3];
        float sy = cfg.scale_modifier * g.scales[i*3+1];
        float sz = cfg.scale_modifier * g.scales[i*3+2];

        float r_q = g.rotations[i*4], x_q = g.rotations[i*4+1];
        float y_q = g.rotations[i*4+2], z_q = g.rotations[i*4+3];

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

        // Compute d_M directly from upper-triangle d_cov3D
        float d_M[3][3];
        for (int k = 0; k < 3; k++) {
            d_M[k][0] = 2.0f*d_cov3D[0]*M[k][0] + d_cov3D[1]*M[k][1] + d_cov3D[2]*M[k][2];
            d_M[k][1] = d_cov3D[1]*M[k][0] + 2.0f*d_cov3D[3]*M[k][1] + d_cov3D[4]*M[k][2];
            d_M[k][2] = d_cov3D[2]*M[k][0] + d_cov3D[4]*M[k][1] + 2.0f*d_cov3D[5]*M[k][2];
        }

        // d_scale[k] and d_R[k][j]
        // M[k][j] = s[k] * R[k][j]
        // d_scale[k] = sum_j d_M[k][j] * R[k][j]
        // d_R[k][j] = d_M[k][j] * s[k]
        float d_scale[3];
        float d_R[3][3];
        for (int k = 0; k < 3; k++) {
            d_scale[k] = 0.0f;
            for (int j = 0; j < 3; j++) {
                d_scale[k] += d_M[k][j] * R[k][j];
                d_R[k][j] = d_M[k][j] * s[k];
            }
        }

        // d_raw_scale[k] = d_scale[k] * scale_modifier * scale[k]
        //   since scale_in_M = scale_modifier * exp(raw_scale)
        //   d/d(raw_scale) = scale_modifier * exp(raw_scale) = scale_modifier * scale
        for (int k = 0; k < 3; k++) {
            grads.d_raw_scales[i*3+k] = d_scale[k] * cfg.scale_modifier * g.scales[i*3+k];
        }

        // d_R -> d_normalized_quat
        // R[row][col] depends on normalized quaternion (r,x,y,z):
        //   R[0][0] = 1 - 2(y^2+z^2)    R[0][1] = 2(xy+rz)       R[0][2] = 2(xz-ry)
        //   R[1][0] = 2(xy-rz)           R[1][1] = 1-2(x^2+z^2)   R[1][2] = 2(yz+rx)
        //   R[2][0] = 2(xz+ry)           R[2][1] = 2(yz-rx)       R[2][2] = 1-2(x^2+y^2)
        //
        // dR/dr: R[0][1]->2z, R[0][2]->-2y, R[1][0]->-2z, R[1][2]->2x, R[2][0]->2y, R[2][1]->-2x
        // dR/dx: R[0][1]->2y, R[0][2]->2z, R[1][0]->2y, R[1][1]->-4x, R[1][2]->2r (wait: d/dx of 2(yz+rx) = 2r)
        //        R[2][0]->2z, R[2][1]->-2r (d/dx of 2(yz-rx) = -2r), R[2][2]->-4x
        // Let me be systematic.

        float d_qn[4] = {0, 0, 0, 0}; // d_loss/d(normalized quaternion r,x,y,z)

        // dR[0][0]/dr = 0, /dx = 0, /dy = -4y, /dz = -4z
        d_qn[2] += d_R[0][0] * (-4.0f*y_q);
        d_qn[3] += d_R[0][0] * (-4.0f*z_q);

        // dR[0][1]/dr = 2z, /dx = 2y, /dy = 2x, /dz = 2r
        d_qn[0] += d_R[0][1] * 2.0f*z_q;
        d_qn[1] += d_R[0][1] * 2.0f*y_q;
        d_qn[2] += d_R[0][1] * 2.0f*x_q;
        d_qn[3] += d_R[0][1] * 2.0f*r_q;

        // dR[0][2]/dr = -2y, /dx = 2z, /dy = -2r, /dz = 2x
        d_qn[0] += d_R[0][2] * (-2.0f*y_q);
        d_qn[1] += d_R[0][2] * 2.0f*z_q;
        d_qn[2] += d_R[0][2] * (-2.0f*r_q);
        d_qn[3] += d_R[0][2] * 2.0f*x_q;

        // dR[1][0]/dr = -2z, /dx = 2y, /dy = 2x, /dz = -2r
        d_qn[0] += d_R[1][0] * (-2.0f*z_q);
        d_qn[1] += d_R[1][0] * 2.0f*y_q;
        d_qn[2] += d_R[1][0] * 2.0f*x_q;
        d_qn[3] += d_R[1][0] * (-2.0f*r_q);

        // dR[1][1]/dr = 0, /dx = -4x, /dy = 0, /dz = -4z
        d_qn[1] += d_R[1][1] * (-4.0f*x_q);
        d_qn[3] += d_R[1][1] * (-4.0f*z_q);

        // dR[1][2]/dr = 2x, /dx = 2r, /dy = 2z, /dz = 2y
        d_qn[0] += d_R[1][2] * 2.0f*x_q;
        d_qn[1] += d_R[1][2] * 2.0f*r_q;
        d_qn[2] += d_R[1][2] * 2.0f*z_q;
        d_qn[3] += d_R[1][2] * 2.0f*y_q;

        // dR[2][0]/dr = 2y, /dx = 2z, /dy = 2r, /dz = 2x
        d_qn[0] += d_R[2][0] * 2.0f*y_q;
        d_qn[1] += d_R[2][0] * 2.0f*z_q;
        d_qn[2] += d_R[2][0] * 2.0f*r_q;
        d_qn[3] += d_R[2][0] * 2.0f*x_q;

        // dR[2][1]/dr = -2x, /dx = -2r, /dy = 2z, /dz = 2y
        d_qn[0] += d_R[2][1] * (-2.0f*x_q);
        d_qn[1] += d_R[2][1] * (-2.0f*r_q);
        d_qn[2] += d_R[2][1] * 2.0f*z_q;
        d_qn[3] += d_R[2][1] * 2.0f*y_q;

        // dR[2][2]/dr = 0, /dx = -4x, /dy = -4y, /dz = 0
        d_qn[1] += d_R[2][2] * (-4.0f*x_q);
        d_qn[2] += d_R[2][2] * (-4.0f*y_q);

        // d_normalized_quat -> d_raw_quat through quaternion normalization
        // q_norm = q_raw / |q_raw|
        // d_raw[k] = (d_norm[k] - q_norm[k] * dot(d_norm, q_norm)) / |q_raw|
        const float* q_raw = &raw.raw_rotations[i*4];
        float q_raw_len = std::sqrt(q_raw[0]*q_raw[0] + q_raw[1]*q_raw[1] +
                                     q_raw[2]*q_raw[2] + q_raw[3]*q_raw[3]);
        if (q_raw_len < 1e-12f) q_raw_len = 1e-12f;
        float inv_len = 1.0f / q_raw_len;

        float dot_dqn_qn = d_qn[0]*r_q + d_qn[1]*x_q + d_qn[2]*y_q + d_qn[3]*z_q;

        grads.d_raw_rotations[i*4+0] = (d_qn[0] - r_q * dot_dqn_qn) * inv_len;
        grads.d_raw_rotations[i*4+1] = (d_qn[1] - x_q * dot_dqn_qn) * inv_len;
        grads.d_raw_rotations[i*4+2] = (d_qn[2] - y_q * dot_dqn_qn) * inv_len;
        grads.d_raw_rotations[i*4+3] = (d_qn[3] - z_q * dot_dqn_qn) * inv_len;

        // ===================================================================
        // Path A: d_cov2D -> d_T -> d_J -> d_p_view -> d_position
        //   (position gradient contribution through the covariance path)
        // ===================================================================
        // We already have d_a, d_b, d_c (d_cov2D), and T, J, W, t from Step 1b.
        //
        // Forward: result[col][row] = sum_{k,j} T[row][k] * Vrk[k][j] * T[col][j]
        //   cov2D[0]=a=result[0][0], cov2D[1]=b=result[1][0], cov2D[2]=c=result[1][1]
        //
        // d_T[col][q]:
        //   VT[col][row] = sum_k Vrk[row][k] * T[col][k]  (= tmp from forward)
        //   d_T[0][q] = 2*d_a*VT[0][q] + d_b*VT[1][q]
        //   d_T[1][q] = d_b*VT[0][q] + 2*d_c*VT[1][q]

        // Compute VT = Vrk * T (reusing same formula as forward tmp)
        float Vrk[3][3] = {
            {cache.cov3D[i*6], cache.cov3D[i*6+1], cache.cov3D[i*6+2]},
            {cache.cov3D[i*6+1], cache.cov3D[i*6+3], cache.cov3D[i*6+4]},
            {cache.cov3D[i*6+2], cache.cov3D[i*6+4], cache.cov3D[i*6+5]}
        };

        float VT[3][3] = {};
        for (int col = 0; col < 3; col++)
            for (int row = 0; row < 3; row++)
                for (int k = 0; k < 3; k++)
                    VT[col][row] += Vrk[row][k] * T[col][k];

        float d_T_cov[2][3] = {};  // d_T[col][row], only col 0,1 matter
        for (int q = 0; q < 3; q++) {
            d_T_cov[0][q] = 2.0f * d_a * VT[0][q] + d_b * VT[1][q];
            d_T_cov[1][q] = d_b * VT[0][q] + 2.0f * d_c * VT[1][q];
        }

        // Backward through T = W * J:
        //   T[col][row] = sum_k W[k][row] * J[col][k]
        //   d_J[col][k] = sum_row d_T[col][row] * W[k][row]
        float d_J[3][3] = {};
        for (int col = 0; col < 2; col++)
            for (int k = 0; k < 3; k++)
                for (int row = 0; row < 3; row++)
                    d_J[col][k] += d_T_cov[col][row] * W[k][row];

        // Backward through J -> p_view (clamped t):
        //   J[0][0] = focal_x / t[2]
        //   J[0][2] = -focal_x * t[0] / (t[2]^2)
        //   J[1][1] = focal_y / t[2]
        //   J[1][2] = -focal_y * t[1] / (t[2]^2)
        //
        // d_t[0] = d_J[0][2] * (-focal_x / t[2]^2)
        // d_t[1] = d_J[1][2] * (-focal_y / t[2]^2)
        // d_t[2] = d_J[0][0] * (-focal_x / t[2]^2)
        //        + d_J[1][1] * (-focal_y / t[2]^2)
        //        + d_J[0][2] * (2*focal_x*t[0] / t[2]^3)
        //        + d_J[1][2] * (2*focal_y*t[1] / t[2]^3)

        float tz2 = t[2] * t[2];
        float tz3 = tz2 * t[2];
        float d_t_cov[3];
        d_t_cov[0] = d_J[0][2] * (-focal_x / tz2);
        d_t_cov[1] = d_J[1][2] * (-focal_y / tz2);
        d_t_cov[2] = d_J[0][0] * (-focal_x / tz2)
                    + d_J[1][1] * (-focal_y / tz2)
                    + d_J[0][2] * (2.0f * focal_x * t[0] / tz3)
                    + d_J[1][2] * (2.0f * focal_y * t[1] / tz3);

        // Backward through frustum clamping:
        //   If |txtz| >= limx, t[0] was clamped -> d_p_view[0] = 0
        //   If |tytz| >= limy, t[1] was clamped -> d_p_view[1] = 0
        if (std::abs(txtz) >= limx) d_t_cov[0] = 0.0f;
        if (std::abs(tytz) >= limy) d_t_cov[1] = 0.0f;

        // Backward through transformPoint4x3:
        //   p_view[row] = sum_col vm[col*4+row] * pos[col] + vm[12+row]
        //   d_pos[col] = sum_row d_p_view[row] * vm[col*4 + row]
        for (int col = 0; col < 3; col++)
            for (int row = 0; row < 3; row++)
                grads.d_raw_positions[i*3 + col] += d_t_cov[row] * vm[col*4 + row];

        // ===================================================================
        // Chain 4: d_means2D -> d_raw_positions (position gradient via projection)
        // ===================================================================
        // Forward:
        //   p_hom = viewproj * [pos, 1]
        //   w = p_hom[3] + 1e-7f
        //   p_ndc = p_hom[0..2] / w
        //   pixel_x = ndc2Pix(p_ndc[0], width)  = ((p_ndc[0]+1)*W - 1) * 0.5
        //   pixel_y = ndc2Pix(p_ndc[1], height)  = ((p_ndc[1]+1)*H - 1) * 0.5

        float d_pixel_x = rgrad.d_means2D[i*2];
        float d_pixel_y = rgrad.d_means2D[i*2+1];

        // ndc2Pix backward: pixel = ((ndc+1)*S - 1) * 0.5
        //   d_ndc = d_pixel * S * 0.5
        float d_ndc_x = d_pixel_x * cam.width * 0.5f;
        float d_ndc_y = d_pixel_y * cam.height * 0.5f;

        // Perspective divide backward: ndc = p_hom.xyz / w, where w = p_hom[3] + 1e-7f
        float w_hom = cache.p_hom_w[i] + 1e-7f;  // p_hom[3] + epsilon (as in forward)
        float inv_w = 1.0f / w_hom;

        // Recover ndc from cached means2D via inverse ndc2Pix:
        //   ndc = (2*pixel + 1) / S - 1
        float pixel_x_cached = cache.pre->means2D[i*2];
        float pixel_y_cached = cache.pre->means2D[i*2+1];
        float ndc_x = (2.0f * pixel_x_cached + 1.0f) / cam.width - 1.0f;
        float ndc_y = (2.0f * pixel_y_cached + 1.0f) / cam.height - 1.0f;

        float d_p_hom[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        d_p_hom[0] = d_ndc_x * inv_w;
        d_p_hom[1] = d_ndc_y * inv_w;
        // d_p_hom[2] = 0 (depth ndc not used in means2D)
        d_p_hom[3] = -(ndc_x * d_ndc_x + ndc_y * d_ndc_y) * inv_w;

        // viewproj backward: p_hom[j] = M[j]*pos[0] + M[4+j]*pos[1] + M[8+j]*pos[2] + M[12+j]
        // d_pos[dim] = sum_j M[dim*4 + j] * d_p_hom[j]
        const float* VP = cam.viewproj_matrix;
        for (int dim = 0; dim < 3; dim++) {
            grads.d_raw_positions[i*3 + dim] +=
                VP[dim*4+0] * d_p_hom[0] +
                VP[dim*4+1] * d_p_hom[1] +
                VP[dim*4+2] * d_p_hom[2] +
                VP[dim*4+3] * d_p_hom[3];
        }
    }
}
