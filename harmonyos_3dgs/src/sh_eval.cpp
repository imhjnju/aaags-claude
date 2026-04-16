#include "sh_eval.h"
#include <cmath>
#include <algorithm>

const float SH_C0 = 0.28209479177387814f;
const float SH_C1 = 0.4886025119029199f;
const float SH_C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
const float SH_C3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

void computeColorFromSH(int degree, int max_coeffs,
                         const float* sh_coeffs,
                         const float pos[3],
                         const float cam_pos[3],
                         float rgb_out[3],
                         bool training) {
    // Compute normalized view direction
    float dx = pos[0] - cam_pos[0];
    float dy = pos[1] - cam_pos[1];
    float dz = pos[2] - cam_pos[2];
    float len = std::sqrt(dx*dx + dy*dy + dz*dz);
    dx /= len; dy /= len; dz /= len;

    float x = dx, y = dy, z = dz;

    // Access: sh[k][ch] = sh_coeffs[k * 3 + ch]
    // Degree 0
    for (int ch = 0; ch < 3; ch++)
        rgb_out[ch] = SH_C0 * sh_coeffs[0 * 3 + ch];

    if (degree > 0) {
        // Degree 1
        for (int ch = 0; ch < 3; ch++) {
            rgb_out[ch] += -SH_C1 * y * sh_coeffs[1 * 3 + ch]
                         +  SH_C1 * z * sh_coeffs[2 * 3 + ch]
                         + -SH_C1 * x * sh_coeffs[3 * 3 + ch];
        }

        if (degree > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            // Degree 2
            for (int ch = 0; ch < 3; ch++) {
                rgb_out[ch] += SH_C2[0] * xy * sh_coeffs[4 * 3 + ch]
                             + SH_C2[1] * yz * sh_coeffs[5 * 3 + ch]
                             + SH_C2[2] * (2.0f * zz - xx - yy) * sh_coeffs[6 * 3 + ch]
                             + SH_C2[3] * xz * sh_coeffs[7 * 3 + ch]
                             + SH_C2[4] * (xx - yy) * sh_coeffs[8 * 3 + ch];
            }

            if (degree > 2) {
                // Degree 3
                for (int ch = 0; ch < 3; ch++) {
                    rgb_out[ch] += SH_C3[0] * y * (3.0f*xx - yy) * sh_coeffs[9 * 3 + ch]
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

    // Add 0.5 DC offset, clamp lower bound to 0.
    // Upper clamp (to 1.0) is only applied in inference mode — during training
    // it kills gradients for super-bright Gaussians, preventing the optimizer
    // from reducing them.
    for (int ch = 0; ch < 3; ch++) {
        rgb_out[ch] = std::max(0.0f, rgb_out[ch] + 0.5f);
        if (!training) {
            rgb_out[ch] = std::min(1.0f, rgb_out[ch]);
        }
    }
}
