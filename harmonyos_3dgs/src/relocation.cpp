// relocation.cpp — MCMC relocation kernel (Eq. 9 of "3DGS as MCMC").
//
// Port of AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/
// utils.cu::compute_relocation. Sequential float math, identical loop nesting,
// to match CUDA output bit-for-bit modulo ULP from differing rounding modes.

#include "relocation.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace mcmc {

namespace {

// Build the binomial table once. Use double-precision factorial-like recurrence
// then cast to float at storage — same as the Python reference (math.comb is
// integer arithmetic, then converted to float64). Only entries n < N_MAX,
// k <= n are filled; the rest stay 0.0f.
struct BinomialTable {
    std::array<float, static_cast<size_t>(N_MAX) * N_MAX> data{};
    BinomialTable() {
        // math.comb(n, k) is exact integer; for N_MAX=51 the largest value is
        // C(50, 25) ~= 1.26e14 which fits in double exactly and rounds cleanly
        // to float (the float port of utils.cu uses float storage too).
        for (int n = 0; n < N_MAX; ++n) {
            // Pascal's triangle row n built incrementally in double precision.
            // c(n, k) = c(n, k-1) * (n - k + 1) / k.
            double c = 1.0;
            data[static_cast<size_t>(n) * N_MAX + 0] = static_cast<float>(c);
            for (int k = 1; k <= n; ++k) {
                c = c * static_cast<double>(n - k + 1) / static_cast<double>(k);
                data[static_cast<size_t>(n) * N_MAX + k] = static_cast<float>(c);
            }
        }
    }
};

}  // namespace

const float* binomial_table() {
    static const BinomialTable t;
    return t.data.data();
}

void compute_relocation(
    float opacity_old,
    const float scale_old[3],
    int N,
    float& opacity_new,
    float scale_new[3])
{
    // Match Python: max(1, min(N, N_MAX - 1)).
    if (N < 1) N = 1;
    if (N > N_MAX - 1) N = N_MAX - 1;

    const float* binoms = binomial_table();

    // New opacity: 1 - (1 - opacity_old)^(1/N).
    opacity_new = 1.0f - std::pow(1.0f - opacity_old, 1.0f / static_cast<float>(N));

    // Denom sum from Eq. 9. Mirrors utils.cu loop nesting exactly:
    //   outer i in 1..N, inner k in 0..i-1.
    float denom_sum = 0.0f;
    for (int i = 1; i <= N; ++i) {
        for (int k = 0; k < i; ++k) {
            const float bin_coeff = binoms[static_cast<size_t>(i - 1) * N_MAX + k];
            const float sign      = (k % 2 == 0) ? 1.0f : -1.0f;
            const float inv_sqrt  = 1.0f / std::sqrt(static_cast<float>(k + 1));
            const float pow_term  = std::pow(opacity_new, static_cast<float>(k + 1));
            const float term      = sign * inv_sqrt * pow_term;
            denom_sum += bin_coeff * term;
        }
    }

    const float coeff = opacity_old / denom_sum;
    scale_new[0] = coeff * scale_old[0];
    scale_new[1] = coeff * scale_old[1];
    scale_new[2] = coeff * scale_old[2];
}

}  // namespace mcmc
