#pragma once

// SH constants matching CUDA auxiliary.h
extern const float SH_C0;
extern const float SH_C1;
extern const float SH_C2[5];
extern const float SH_C3[7];

// Evaluate SH coefficients to RGB color for a given view direction.
// sh_coeffs: pointer to this Gaussian's SH data [max_coeffs * 3], interleaved RGB per basis
// Access pattern: sh[k][ch] = sh_coeffs[k * 3 + ch]
// pos: Gaussian world position [3]
// cam_pos: camera world position [3]
// degree: active SH degree 0-3
// max_coeffs: total number of SH basis functions in the data ((degree+1)^2 for the model)
// rgb_out: output RGB color [3], clamped >= 0
void computeColorFromSH(int degree, int max_coeffs,
                         const float* sh_coeffs,
                         const float pos[3],
                         const float cam_pos[3],
                         float rgb_out[3],
                         bool training = false);
