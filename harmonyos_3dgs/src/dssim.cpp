#include "dssim.h"
#include <cmath>
#include <vector>
#include <algorithm>

static const int WINDOW = 11;
static const float SIGMA = 1.5f;
static const float C1 = 0.01f * 0.01f;
static const float C2 = 0.03f * 0.03f;

// Build 1D Gaussian kernel (length WINDOW, normalized to sum=1).
static std::vector<float> make_gaussian_kernel() {
    std::vector<float> k(WINDOW);
    float sum = 0.f;
    int half = WINDOW / 2;
    for (int i = 0; i < WINDOW; ++i) {
        float x = static_cast<float>(i - half);
        k[i] = std::exp(-x * x / (2.f * SIGMA * SIGMA));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

// Build 2D Gaussian kernel (WINDOW x WINDOW), normalized to sum=1.
static std::vector<float> make_gaussian_kernel_2d() {
    auto k1 = make_gaussian_kernel();
    std::vector<float> k2(WINDOW * WINDOW);
    float sum = 0.f;
    for (int r = 0; r < WINDOW; ++r) {
        for (int c = 0; c < WINDOW; ++c) {
            k2[r * WINDOW + c] = k1[r] * k1[c];
            sum += k2[r * WINDOW + c];
        }
    }
    for (auto& v : k2) v /= sum;
    return k2;
}

// Compute SSIM for the 11x11 window centered at (cx, cy), channel ch.
// rendered, target: [3*H*W] CHW channel-first layout (channel ch at ch*H*W + pixel).
// Uses clamp-to-edge padding at image boundaries.
static float ssim_window(const float* rendered, const float* target,
                         int W, int H, int cx, int cy, int ch,
                         const std::vector<float>& kernel) {
    int half = WINDOW / 2;
    float mu1 = 0.f, mu2 = 0.f;

    // Pass 1: compute means
    for (int dr = -half; dr <= half; ++dr) {
        for (int dc = -half; dc <= half; ++dc) {
            int r = std::clamp(cy + dr, 0, H - 1);
            int c = std::clamp(cx + dc, 0, W - 1);
            int ki = (dr + half) * WINDOW + (dc + half);
            float w = kernel[ki];
            float x = rendered[ch * H * W + r * W + c];
            float y = target  [ch * H * W + r * W + c];
            mu1 += w * x;
            mu2 += w * y;
        }
    }

    // Pass 2: compute variances and covariance
    float s1sq = 0.f, s2sq = 0.f, s12 = 0.f;
    for (int dr = -half; dr <= half; ++dr) {
        for (int dc = -half; dc <= half; ++dc) {
            int r = std::clamp(cy + dr, 0, H - 1);
            int c = std::clamp(cx + dc, 0, W - 1);
            int ki = (dr + half) * WINDOW + (dc + half);
            float w = kernel[ki];
            float x = rendered[ch * H * W + r * W + c];
            float y = target  [ch * H * W + r * W + c];
            s1sq += w * (x - mu1) * (x - mu1);
            s2sq += w * (y - mu2) * (y - mu2);
            s12  += w * (x - mu1) * (y - mu2);
        }
    }

    float num   = (2.f * mu1 * mu2 + C1) * (2.f * s12 + C2);
    float denom = (mu1 * mu1 + mu2 * mu2 + C1) * (s1sq + s2sq + C2);
    return num / denom;
}

float compute_combined_loss_gradient(
    const float* rendered,
    const float* target,
    float* dL_dpixels,
    int W, int H,
    float lambda_dssim)
{
    const int N = W * H * 3;
    const float scale_l1 = 1.0f / static_cast<float>(N);
    const float w_l1   = 1.0f - lambda_dssim;
    const float w_ssim = lambda_dssim;

    // --- L1 part ---
    float loss_l1 = 0.f;
    for (int i = 0; i < N; ++i) {
        float d = rendered[i] - target[i];
        loss_l1 += std::fabs(d);
        float sign = (d > 0.f) ? 1.f : (d < 0.f ? -1.f : 0.f);
        dL_dpixels[i] = w_l1 * sign * scale_l1;
    }
    loss_l1 *= scale_l1;

    // --- SSIM part ---
    // Analytical sliding-window gradient.
    //
    // SSIM_mean = (1 / (W*H*3)) * sum_{cx,cy,ch} SSIM(cx,cy,ch)
    //
    // d(SSIM_mean)/d(x[px,py,ch]) = (1/(W*H*3)) *
    //   sum over ALL windows (cx,cy) that contain (px,py) of d(SSIM(cx,cy,ch))/d(x[px,py,ch])
    //
    // For window (cx,cy), pixel at kernel offset (dr=py-cy, dc=px-cx), weight w_k:
    //   d(SSIM(cx,cy))/d(x_k) = w_k * [
    //     (2*mu2*B + 2*A*(y_k - mu2)) / (D*E)
    //     - ssim_val * (2*mu1/D + 2*(x_k - mu1)/E)
    //   ]
    // where A=2*mu1*mu2+C1, B=2*s12+C2, D=mu1²+mu2²+C1, E=s1sq+s2sq+C2.
    // Evidence: derived analytically from SSIM = A*B/(D*E) via product/quotient rule.
    auto kernel = make_gaussian_kernel_2d();
    const float norm_ssim = 1.0f / static_cast<float>(W * H * 3);
    int half = WINDOW / 2;

    // Compute mean SSIM for the loss value.
    float ssim_sum = 0.f;
    for (int cy = 0; cy < H; ++cy) {
        for (int cx = 0; cx < W; ++cx) {
            for (int ch = 0; ch < 3; ++ch) {
                ssim_sum += ssim_window(rendered, target, W, H, cx, cy, ch, kernel);
            }
        }
    }
    float ssim_mean = ssim_sum * norm_ssim;
    float loss_ssim = 1.0f - ssim_mean;

    // Analytical gradient: for each window, accumulate contributions into all pixels in the window.
    for (int cy = 0; cy < H; ++cy) {
        for (int cx = 0; cx < W; ++cx) {
            for (int ch = 0; ch < 3; ++ch) {
                // Compute window statistics (same as ssim_window but inline for gradient).
                float mu1 = 0.f, mu2 = 0.f;
                for (int dr = -half; dr <= half; ++dr) {
                    for (int dc = -half; dc <= half; ++dc) {
                        int r = std::clamp(cy + dr, 0, H - 1);
                        int c = std::clamp(cx + dc, 0, W - 1);
                        float w = kernel[(dr + half) * WINDOW + (dc + half)];
                        mu1 += w * rendered[ch * H * W + r * W + c];
                        mu2 += w * target  [ch * H * W + r * W + c];
                    }
                }
                float s1sq = 0.f, s2sq = 0.f, s12 = 0.f;
                for (int dr = -half; dr <= half; ++dr) {
                    for (int dc = -half; dc <= half; ++dc) {
                        int r = std::clamp(cy + dr, 0, H - 1);
                        int c = std::clamp(cx + dc, 0, W - 1);
                        float w = kernel[(dr + half) * WINDOW + (dc + half)];
                        float x = rendered[ch * H * W + r * W + c];
                        float y = target  [ch * H * W + r * W + c];
                        s1sq += w * (x - mu1) * (x - mu1);
                        s2sq += w * (y - mu2) * (y - mu2);
                        s12  += w * (x - mu1) * (y - mu2);
                    }
                }
                float A = 2.f * mu1 * mu2 + C1;
                float B = 2.f * s12 + C2;
                float D = mu1 * mu1 + mu2 * mu2 + C1;
                float E = s1sq + s2sq + C2;
                float ssim_val = A * B / (D * E);

                // Accumulate analytical gradient contribution from this window into each pixel.
                for (int dr = -half; dr <= half; ++dr) {
                    for (int dc = -half; dc <= half; ++dc) {
                        int r = std::clamp(cy + dr, 0, H - 1);
                        int c = std::clamp(cx + dc, 0, W - 1);
                        float w_k = kernel[(dr + half) * WINDOW + (dc + half)];
                        float x_k = rendered[ch * H * W + r * W + c];
                        float y_k = target  [ch * H * W + r * W + c];

                        float d_ssim_dxk = w_k * (
                            (2.f * mu2 * B + 2.f * A * (y_k - mu2)) / (D * E)
                            - ssim_val * (2.f * mu1 / D + 2.f * (x_k - mu1) / E)
                        );
                        // d(DSSIM_mean)/d(x_k) = -d(SSIM_mean)/d(x_k) = -norm_ssim * d(SSIM(cx,cy))/d(x_k)
                        dL_dpixels[ch * H * W + r * W + c] += -w_ssim * norm_ssim * d_ssim_dxk;
                    }
                }
            }
        }
    }

    return w_l1 * loss_l1 + w_ssim * loss_ssim;
}
