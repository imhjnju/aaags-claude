#include "dssim.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

static const int WINDOW = 11;
static const float SIGMA = 1.5f;
static const float C1 = 0.01f * 0.01f;
static const float C2 = 0.03f * 0.03f;

static const std::vector<float>& gaussian_kernel() {
    static const std::vector<float> k = [] {
        std::vector<float> out(WINDOW);
        float sum = 0.f;
        const int half = WINDOW / 2;
        for (int i = 0; i < WINDOW; ++i) {
            const float x = static_cast<float>(i - half);
            out[i] = std::exp(-x * x / (2.f * SIGMA * SIGMA));
            sum += out[i];
        }
        for (float& v : out) v /= sum;
        return out;
    }();
    return k;
}

static int worker_count(int n) {
    const unsigned hw = std::thread::hardware_concurrency();
    const int max_workers = hw > 0 ? static_cast<int>(hw) : 1;
    if (n < 64) return 1;
    return std::max(1, std::min(max_workers, n));
}

template <typename Fn>
static void parallel_chunks(int n, Fn fn) {
    const int workers = worker_count(n);
    if (workers == 1) {
        fn(0, n, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers));
    for (int t = 0; t < workers; ++t) {
        const int begin = (n * t) / workers;
        const int end = (n * (t + 1)) / workers;
        threads.emplace_back([=, &fn] { fn(begin, end, t); });
    }
    for (auto& thread : threads) thread.join();
}

static void blur_clamp(const float* in, float* out, float* scratch,
                       int W, int H, const std::vector<float>& k) {
    const int half = WINDOW / 2;
    parallel_chunks(H, [&](int y0, int y1, int) {
        for (int y = y0; y < y1; ++y) {
            const int row = y * W;
            for (int x = 0; x < W; ++x) {
                float sum = 0.f;
                for (int dx = -half; dx <= half; ++dx) {
                    const int xx = std::clamp(x + dx, 0, W - 1);
                    sum += k[dx + half] * in[row + xx];
                }
                scratch[row + x] = sum;
            }
        }
    });

    parallel_chunks(H, [&](int y0, int y1, int) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.f;
                for (int dy = -half; dy <= half; ++dy) {
                    const int yy = std::clamp(y + dy, 0, H - 1);
                    sum += k[dy + half] * scratch[yy * W + x];
                }
                out[y * W + x] = sum;
            }
        }
    });
}

static void transpose_blur_clamp(const float* in, float* out, float* scratch,
                                 int W, int H, const std::vector<float>& k) {
    const int HW = W * H;
    const int half = WINDOW / 2;
    std::fill(scratch, scratch + HW, 0.f);
    std::fill(out, out + HW, 0.f);

    parallel_chunks(W, [&](int x0, int x1, int) {
        for (int x = x0; x < x1; ++x) {
            for (int y = 0; y < H; ++y) {
                const float g = in[y * W + x];
                for (int dy = -half; dy <= half; ++dy) {
                    const int yy = std::clamp(y + dy, 0, H - 1);
                    scratch[yy * W + x] += k[dy + half] * g;
                }
            }
        }
    });

    parallel_chunks(H, [&](int y0, int y1, int) {
        for (int y = y0; y < y1; ++y) {
            const int row = y * W;
            for (int x = 0; x < W; ++x) {
                const float g = scratch[row + x];
                for (int dx = -half; dx <= half; ++dx) {
                    const int xx = std::clamp(x + dx, 0, W - 1);
                    out[row + xx] += k[dx + half] * g;
                }
            }
        }
    });
}

float compute_combined_loss_gradient(
    const float* rendered,
    const float* target,
    float* dL_dpixels,
    int W, int H,
    float lambda_dssim)
{
    const int HW = W * H;
    const int N = HW * 3;
    const float scale_l1 = 1.0f / static_cast<float>(N);
    const float w_l1 = 1.0f - lambda_dssim;
    const float w_ssim = lambda_dssim;

    const int workers = worker_count(N);
    std::vector<float> l1_partials(static_cast<size_t>(workers), 0.f);
    parallel_chunks(N, [&](int i0, int i1, int tid) {
        float local = 0.f;
        for (int i = i0; i < i1; ++i) {
            const float d = rendered[i] - target[i];
            local += std::fabs(d);
            const float sign = (d > 0.f) ? 1.f : (d < 0.f ? -1.f : 0.f);
            dL_dpixels[i] = w_l1 * sign * scale_l1;
        }
        l1_partials[static_cast<size_t>(tid)] = local;
    });
    float loss_l1 = 0.f;
    for (float v : l1_partials) loss_l1 += v;
    loss_l1 *= scale_l1;

    if (lambda_dssim == 0.0f) {
        return loss_l1;
    }

    const auto& k = gaussian_kernel();
    const float norm_ssim = 1.0f / static_cast<float>(N);
    float ssim_sum = 0.f;

    std::vector<float> x2(HW), y2(HW), xy(HW);
    std::vector<float> mu1(HW), mu2(HW), ex2(HW), ey2(HW), exy(HW);
    std::vector<float> alpha(HW), beta(HW), gamma(HW);
    std::vector<float> adj_alpha(HW), adj_beta(HW), adj_gamma(HW);
    std::vector<float> scratch(HW);
    std::vector<float> ssim_partials(static_cast<size_t>(worker_count(HW)), 0.f);

    for (int ch = 0; ch < 3; ++ch) {
        const float* x = rendered + ch * HW;
        const float* y = target + ch * HW;
        parallel_chunks(HW, [&](int i0, int i1, int) {
            for (int i = i0; i < i1; ++i) {
                x2[i] = x[i] * x[i];
                y2[i] = y[i] * y[i];
                xy[i] = x[i] * y[i];
            }
        });

        blur_clamp(x, mu1.data(), scratch.data(), W, H, k);
        blur_clamp(y, mu2.data(), scratch.data(), W, H, k);
        blur_clamp(x2.data(), ex2.data(), scratch.data(), W, H, k);
        blur_clamp(y2.data(), ey2.data(), scratch.data(), W, H, k);
        blur_clamp(xy.data(), exy.data(), scratch.data(), W, H, k);

        std::fill(ssim_partials.begin(), ssim_partials.end(), 0.f);
        parallel_chunks(HW, [&](int i0, int i1, int tid) {
            float local = 0.f;
            for (int i = i0; i < i1; ++i) {
                const float s1sq = ex2[i] - mu1[i] * mu1[i];
                const float s2sq = ey2[i] - mu2[i] * mu2[i];
                const float s12 = exy[i] - mu1[i] * mu2[i];
                const float A = 2.f * mu1[i] * mu2[i] + C1;
                const float B = 2.f * s12 + C2;
                const float D = mu1[i] * mu1[i] + mu2[i] * mu2[i] + C1;
                const float E = s1sq + s2sq + C2;
                const float ssim_val = A * B / (D * E);
                local += ssim_val;

                const float inv_de = 1.f / (D * E);
                beta[i] = 2.f * A * inv_de;
                gamma[i] = -2.f * ssim_val / E;
                alpha[i] = 2.f * mu2[i] * B * inv_de
                         - 2.f * ssim_val * mu1[i] / D
                         - beta[i] * mu2[i]
                         - gamma[i] * mu1[i];
            }
            ssim_partials[static_cast<size_t>(tid)] = local;
        });
        for (float v : ssim_partials) ssim_sum += v;

        transpose_blur_clamp(alpha.data(), adj_alpha.data(), scratch.data(), W, H, k);
        transpose_blur_clamp(beta.data(), adj_beta.data(), scratch.data(), W, H, k);
        transpose_blur_clamp(gamma.data(), adj_gamma.data(), scratch.data(), W, H, k);

        float* grad = dL_dpixels + ch * HW;
        parallel_chunks(HW, [&](int i0, int i1, int) {
            for (int i = i0; i < i1; ++i) {
                const float d_ssim_dx = adj_alpha[i] + y[i] * adj_beta[i] + x[i] * adj_gamma[i];
                grad[i] += -w_ssim * norm_ssim * d_ssim_dx;
            }
        });
    }

    const float ssim_mean = ssim_sum * norm_ssim;
    const float loss_ssim = 1.0f - ssim_mean;
    return w_l1 * loss_l1 + w_ssim * loss_ssim;
}
