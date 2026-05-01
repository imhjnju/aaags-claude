#include <gtest/gtest.h>
#include "dssim.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>

namespace {

constexpr int kWindow = 11;
constexpr float kSigma = 1.5f;
constexpr float kC1 = 0.01f * 0.01f;
constexpr float kC2 = 0.03f * 0.03f;

std::vector<float> make_gaussian_kernel_2d() {
    std::vector<float> kernel(kWindow * kWindow);
    float sum = 0.f;
    const int half = kWindow / 2;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            const float dist2 = static_cast<float>(dx * dx + dy * dy);
            const float value = std::exp(-dist2 / (2.f * kSigma * kSigma));
            kernel[(dy + half) * kWindow + (dx + half)] = value;
            sum += value;
        }
    }
    for (float& v : kernel) v /= sum;
    return kernel;
}

float slow_reference_combined_loss(const std::vector<float>& rendered,
                                   const std::vector<float>& target,
                                   int W, int H,
                                   float lambda_dssim) {
    const int HW = W * H;
    const int N = HW * 3;
    float l1 = 0.f;
    for (int i = 0; i < N; ++i) l1 += std::fabs(rendered[i] - target[i]);
    l1 /= static_cast<float>(N);

    if (lambda_dssim == 0.f) return l1;

    const auto kernel = make_gaussian_kernel_2d();
    const int half = kWindow / 2;
    float ssim_sum = 0.f;

    for (int ch = 0; ch < 3; ++ch) {
        const int channel = ch * HW;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float mu1 = 0.f, mu2 = 0.f;
                float ex2 = 0.f, ey2 = 0.f, exy = 0.f;
                for (int dy = -half; dy <= half; ++dy) {
                    const int yy = std::clamp(y + dy, 0, H - 1);
                    for (int dx = -half; dx <= half; ++dx) {
                        const int xx = std::clamp(x + dx, 0, W - 1);
                        const float w = kernel[(dy + half) * kWindow + (dx + half)];
                        const int idx = channel + yy * W + xx;
                        const float r = rendered[idx];
                        const float t = target[idx];
                        mu1 += w * r;
                        mu2 += w * t;
                        ex2 += w * r * r;
                        ey2 += w * t * t;
                        exy += w * r * t;
                    }
                }
                const float s1sq = ex2 - mu1 * mu1;
                const float s2sq = ey2 - mu2 * mu2;
                const float s12 = exy - mu1 * mu2;
                const float A = 2.f * mu1 * mu2 + kC1;
                const float B = 2.f * s12 + kC2;
                const float D = mu1 * mu1 + mu2 * mu2 + kC1;
                const float E = s1sq + s2sq + kC2;
                ssim_sum += A * B / (D * E);
            }
        }
    }

    const float ssim_mean = ssim_sum / static_cast<float>(N);
    return (1.f - lambda_dssim) * l1 + lambda_dssim * (1.f - ssim_mean);
}

}  // namespace

// T-dssim1: Constant image — rendered == target.
// SSIM = 1, DSSIM = 0, combined loss ≈ 0, gradient ≈ 0.
TEST(DSSIM, ConstantImage) {
    const int W = 8, H = 8;
    const int N = W * H * 3;

    std::vector<float> rendered(N, 0.5f);
    std::vector<float> target(N, 0.5f);
    std::vector<float> grad(N, 0.f);

    float loss = compute_combined_loss_gradient(
        rendered.data(), target.data(), grad.data(), W, H, 0.2f);

    // L1 part: all diffs are 0 → loss_l1 = 0, so combined loss = 0
    EXPECT_NEAR(loss, 0.f, 1e-5f) << "Loss should be ~0 for identical images";

    // Gradient should be ~0 everywhere
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(grad[i], 0.f, 1e-5f)
            << "Gradient at index " << i << " should be ~0 for identical images";
    }
}

// T-dssim2: Optimized separable DSSIM must preserve direct clamp-window semantics.
TEST(DSSIM, SlowReferenceSlidingWindowEquivalence) {
    const int W = 9, H = 8;
    const int N = W * H * 3;

    std::vector<float> rendered(N), target(N);
    unsigned seed = 91u;
    auto lcg = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return 0.1f + 0.8f * (static_cast<float>(seed >> 16) / 65535.f);
    };
    for (int i = 0; i < N; ++i) { rendered[i] = lcg(); target[i] = lcg(); }

    const float lambda_dssim = 0.2f;
    std::vector<float> grad_optimized(N, 0.f);
    const float optimized_loss = compute_combined_loss_gradient(
        rendered.data(), target.data(), grad_optimized.data(), W, H, lambda_dssim);
    const float reference_loss = slow_reference_combined_loss(
        rendered, target, W, H, lambda_dssim);

    EXPECT_NEAR(optimized_loss, reference_loss, 5e-5f);

    const float eps_fd = 1e-3f;
    const float tol = 2e-3f;
    const int test_indices[] = {
        0,
        W * H,
        W * H + 2 * W + 3,
        2 * W * H + (H - 1) * W + (W - 1),
    };

    for (int idx : test_indices) {
        std::vector<float> rp(rendered), rm(rendered);
        rp[idx] += eps_fd;
        rm[idx] -= eps_fd;
        const float lp = slow_reference_combined_loss(rp, target, W, H, lambda_dssim);
        const float lm = slow_reference_combined_loss(rm, target, W, H, lambda_dssim);
        const float fd = (lp - lm) / (2.f * eps_fd);
        EXPECT_NEAR(grad_optimized[idx], fd, tol)
            << "Slow-reference gradient mismatch at index " << idx
            << " optimized=" << grad_optimized[idx]
            << " fd=" << fd;
    }
}

// T-dssim3: Verify DSSIM gradient direction and magnitude against numerical central difference.
//
// The implementation uses the correct analytical sliding-window gradient:
//   d(SSIM_mean)/d(x[px,py,ch]) = (1/(W*H*3)) *
//     sum over ALL windows (cx,cy) that contain (px,py) of d(SSIM(cx,cy))/d(x[px,py,ch])
TEST(DSSIM, GradientFiniteDifferenceCheck) {
    const int W = 8, H = 8;
    const int N = W * H * 3;

    // Reproducible pseudo-random image in [0.2, 0.8] to avoid flat regions.
    std::vector<float> rendered(N), target(N);
    unsigned seed = 42u;
    auto lcg = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return 0.2f + 0.6f * (static_cast<float>(seed >> 16) / 65535.f);
    };
    for (int i = 0; i < N; ++i) { rendered[i] = lcg(); target[i] = lcg(); }

    // Compute analytical gradient via compute_combined_loss_gradient.
    std::vector<float> grad_analytical(N, 0.f);
    float loss0 = compute_combined_loss_gradient(
        rendered.data(), target.data(), grad_analytical.data(), W, H, 0.2f);

    // For a handful of representative pixels, check gradient sign and approximate magnitude.
    // eps_fd for the outer FD must be large enough to avoid floating-point noise but
    // small enough for the linearization to hold.
    const float eps_fd = 1e-3f;
    // Tolerance accounts for outer FD discretization error O(eps_fd^2) only.
    // Analytical gradient matches FD to machine precision for the SSIM component.
    const float tol = 1e-3f;

    // Test pixels: corner, center, last corner — boundary and interior cases.
    int test_pixels[] = {0, W/2*W + W/2, (H-1)*W + (W-1)};  // (0,0), (H/2,W/2), (H-1,W-1)
    for (int ch = 0; ch < 3; ++ch) {
        for (int tp : test_pixels) {
            int idx = tp * 3 + ch;

            std::vector<float> r_plus(rendered), r_minus(rendered);
            std::vector<float> g_dummy(N);

            r_plus[idx]  = rendered[idx] + eps_fd;
            r_minus[idx] = rendered[idx] - eps_fd;

            float loss_plus = compute_combined_loss_gradient(
                r_plus.data(), target.data(), g_dummy.data(), W, H, 0.2f);
            float loss_minus = compute_combined_loss_gradient(
                r_minus.data(), target.data(), g_dummy.data(), W, H, 0.2f);

            float fd_grad = (loss_plus - loss_minus) / (2.f * eps_fd);

            // Check sign agreement (gradient direction must be correct).
            if (std::fabs(fd_grad) > 1e-4f) {
                EXPECT_EQ(std::signbit(grad_analytical[idx]), std::signbit(fd_grad))
                    << "Sign mismatch at pixel=" << tp << " ch=" << ch
                    << "  analytical=" << grad_analytical[idx]
                    << "  FD=" << fd_grad;
            }

            // Check magnitude within tolerance.
            EXPECT_NEAR(grad_analytical[idx], fd_grad, tol)
                << "Gradient magnitude mismatch at pixel=" << tp << " ch=" << ch
                << "  analytical=" << grad_analytical[idx]
                << "  FD=" << fd_grad;
        }
    }

    (void)loss0;  // suppress unused-variable warning
}

// T-dssim4: Interior-pixel FD check on a 32x32 image.
// All pixels in [11,20]x[11,20] are fully interior — no clamp-to-edge padding
// affects their 11x11 windows. This exercises gradient accumulation without
// any boundary weight duplication.
TEST(DSSIM, GradientInteriorPixelCheck) {
    const int W = 32, H = 32;
    const int N = W * H * 3;

    std::vector<float> rendered(N), target(N);
    unsigned seed = 137u;
    auto lcg = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return 0.2f + 0.6f * (static_cast<float>(seed >> 16) / 65535.f);
    };
    for (int i = 0; i < N; ++i) { rendered[i] = lcg(); target[i] = lcg(); }

    std::vector<float> grad_analytical(N, 0.f);
    compute_combined_loss_gradient(rendered.data(), target.data(),
                                   grad_analytical.data(), W, H, 0.2f);

    const float eps_fd = 1e-3f;
    const float tol    = 1e-3f;

    // Test a fully-interior pixel at (row=15, col=15): window [10,20]x[10,20], all valid.
    int interior_pixel = 15 * W + 15;
    for (int ch = 0; ch < 3; ++ch) {
        int idx = interior_pixel * 3 + ch;
        std::vector<float> rp(rendered), rm(rendered);
        std::vector<float> gd(N);
        rp[idx] = rendered[idx] + eps_fd;
        rm[idx] = rendered[idx] - eps_fd;
        float lp = compute_combined_loss_gradient(rp.data(), target.data(), gd.data(), W, H, 0.2f);
        float lm = compute_combined_loss_gradient(rm.data(), target.data(), gd.data(), W, H, 0.2f);
        float fd = (lp - lm) / (2.f * eps_fd);

        if (std::fabs(fd) > 1e-4f) {
            EXPECT_EQ(std::signbit(grad_analytical[idx]), std::signbit(fd))
                << "Sign mismatch at interior pixel ch=" << ch;
        }
        EXPECT_NEAR(grad_analytical[idx], fd, tol)
            << "Interior gradient mismatch at ch=" << ch;
    }
}
