#pragma once

// relocation.h — MCMC relocation kernel (Eq. 9 of "3D Gaussian Splatting as
// Markov Chain Monte Carlo"). Port of AAA-Gaussians
// submodules/diff-gaussian-rasterization/cuda_rasterizer/utils.cu::compute_relocation.
//
// Sequential float math, deterministic. Used by mcmc_densification.cpp to
// compute updated opacity/scale for clones during MCMC relocation.

namespace mcmc {

// Maximum supported number of clones (matches Python reloc_utils.py N_MAX=51).
constexpr int N_MAX = 51;

// Returns pointer to a [N_MAX*N_MAX] flattened table of binomial coefficients
// math.comb(n, k) for n in [0, N_MAX), k in [0, n] (zeros elsewhere). Row-major:
// element (n, k) at index n*N_MAX + k. Lazily initialised on first call;
// thread-safe via Meyers' singleton.
const float* binomial_table();

// Eq. 9 of "3DGS as Markov Chain Monte Carlo".
//   opacity_old:  activation-space opacity, in (0, 1).
//   scale_old:    activation-space scale (already exp'd, NOT log).
//   N:            number of clones; clamped to [1, N_MAX-1].
// Outputs:
//   opacity_new:  activation-space opacity = 1 - (1-opacity_old)^(1/N).
//   scale_new:    activation-space scale = coeff * scale_old, axis-wise (so
//                 anisotropy is preserved).
//
// Mirrors compute_relocation kernel in utils.cu — sequential float math,
// outer i in 1..N, inner k in 0..i-1.
void compute_relocation(
    float opacity_old,
    const float scale_old[3],
    int N,
    float& opacity_new,
    float scale_new[3]);

}  // namespace mcmc
