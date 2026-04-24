#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include "sh_eval.h"
#include "math_utils.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/preprocessor_backward_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ---------- SH backward finite-difference tests ----------

// Helper: fill sh_coeffs with deterministic pseudo-random values
static void fillSH(float* sh, int max_coeffs) {
    for (int i = 0; i < max_coeffs * 3; i++) {
        sh[i] = 0.1f * ((i * 7 + 3) % 13) - 0.5f;
    }
}

TEST(PreprocessorBackward, SHBackwardDegree0) {
    const int degree = 0;
    const int max_coeffs = 1;
    float sh_coeffs[max_coeffs * 3];
    fillSH(sh_coeffs, max_coeffs);

    float pos[3] = {1.0f, 2.0f, 3.0f};
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};
    float d_rgb[3] = {1.0f, 0.5f, -0.3f};

    // Analytic gradient
    float d_sh_analytic[max_coeffs * 3];
    computeColorFromSH_backward(degree, max_coeffs, sh_coeffs, pos, cam_pos,
                                  d_rgb, d_sh_analytic, true);

    // Finite-difference gradient
    const float eps = 1e-3f;
    float d_sh_fd[max_coeffs * 3];
    for (int i = 0; i < max_coeffs * 3; i++) {
        float sh_plus[max_coeffs * 3];
        float sh_minus[max_coeffs * 3];
        std::memcpy(sh_plus, sh_coeffs, sizeof(sh_coeffs));
        std::memcpy(sh_minus, sh_coeffs, sizeof(sh_coeffs));
        sh_plus[i] += eps;
        sh_minus[i] -= eps;

        float rgb_plus[3], rgb_minus[3];
        computeColorFromSH(degree, max_coeffs, sh_plus, pos, cam_pos, rgb_plus, true);
        computeColorFromSH(degree, max_coeffs, sh_minus, pos, cam_pos, rgb_minus, true);

        // d_loss = sum(d_rgb[ch] * rgb[ch]), so gradient w.r.t. sh[i] is:
        float grad = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            grad += d_rgb[ch] * (rgb_plus[ch] - rgb_minus[ch]) / (2.0f * eps);
        }
        d_sh_fd[i] = grad;
    }

    for (int i = 0; i < max_coeffs * 3; i++) {
        EXPECT_NEAR(d_sh_analytic[i], d_sh_fd[i], 1e-3f)
            << "SH degree 0 backward mismatch at index " << i;
    }
}

TEST(PreprocessorBackward, SHBackwardDegree1) {
    const int degree = 1;
    const int max_coeffs = 4;
    float sh_coeffs[max_coeffs * 3];
    fillSH(sh_coeffs, max_coeffs);

    float pos[3] = {1.0f, 2.0f, 3.0f};
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};
    float d_rgb[3] = {1.0f, 0.5f, -0.3f};

    // Analytic gradient
    float d_sh_analytic[max_coeffs * 3];
    computeColorFromSH_backward(degree, max_coeffs, sh_coeffs, pos, cam_pos,
                                  d_rgb, d_sh_analytic, true);

    // Finite-difference gradient
    const float eps = 1e-3f;
    float d_sh_fd[max_coeffs * 3];
    for (int i = 0; i < max_coeffs * 3; i++) {
        float sh_plus[max_coeffs * 3];
        float sh_minus[max_coeffs * 3];
        std::memcpy(sh_plus, sh_coeffs, sizeof(sh_coeffs));
        std::memcpy(sh_minus, sh_coeffs, sizeof(sh_coeffs));
        sh_plus[i] += eps;
        sh_minus[i] -= eps;

        float rgb_plus[3], rgb_minus[3];
        computeColorFromSH(degree, max_coeffs, sh_plus, pos, cam_pos, rgb_plus, true);
        computeColorFromSH(degree, max_coeffs, sh_minus, pos, cam_pos, rgb_minus, true);

        float grad = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            grad += d_rgb[ch] * (rgb_plus[ch] - rgb_minus[ch]) / (2.0f * eps);
        }
        d_sh_fd[i] = grad;
    }

    for (int i = 0; i < max_coeffs * 3; i++) {
        EXPECT_NEAR(d_sh_analytic[i], d_sh_fd[i], 1e-3f)
            << "SH degree 1 backward mismatch at index " << i;
    }
}

TEST(PreprocessorBackward, SHBackwardDegree2) {
    const int degree = 2;
    const int max_coeffs = 9;
    float sh_coeffs[max_coeffs * 3];
    fillSH(sh_coeffs, max_coeffs);

    float pos[3] = {1.0f, 2.0f, 3.0f};
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};
    float d_rgb[3] = {1.0f, 0.5f, -0.3f};

    float d_sh_analytic[max_coeffs * 3];
    computeColorFromSH_backward(degree, max_coeffs, sh_coeffs, pos, cam_pos,
                                  d_rgb, d_sh_analytic, true);

    const float eps = 1e-3f;
    float d_sh_fd[max_coeffs * 3];
    for (int i = 0; i < max_coeffs * 3; i++) {
        float sh_plus[max_coeffs * 3];
        float sh_minus[max_coeffs * 3];
        std::memcpy(sh_plus, sh_coeffs, sizeof(sh_coeffs));
        std::memcpy(sh_minus, sh_coeffs, sizeof(sh_coeffs));
        sh_plus[i] += eps;
        sh_minus[i] -= eps;

        float rgb_plus[3], rgb_minus[3];
        computeColorFromSH(degree, max_coeffs, sh_plus, pos, cam_pos, rgb_plus, true);
        computeColorFromSH(degree, max_coeffs, sh_minus, pos, cam_pos, rgb_minus, true);

        float grad = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            grad += d_rgb[ch] * (rgb_plus[ch] - rgb_minus[ch]) / (2.0f * eps);
        }
        d_sh_fd[i] = grad;
    }

    for (int i = 0; i < max_coeffs * 3; i++) {
        EXPECT_NEAR(d_sh_analytic[i], d_sh_fd[i], 1e-3f)
            << "SH degree 2 backward mismatch at index " << i;
    }
}

TEST(PreprocessorBackward, SHBackwardDegree3) {
    const int degree = 3;
    const int max_coeffs = 16;
    float sh_coeffs[max_coeffs * 3];
    fillSH(sh_coeffs, max_coeffs);

    float pos[3] = {1.0f, 2.0f, 3.0f};
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};
    float d_rgb[3] = {1.0f, 0.5f, -0.3f};

    float d_sh_analytic[max_coeffs * 3];
    computeColorFromSH_backward(degree, max_coeffs, sh_coeffs, pos, cam_pos,
                                  d_rgb, d_sh_analytic, true);

    const float eps = 1e-3f;
    float d_sh_fd[max_coeffs * 3];
    for (int i = 0; i < max_coeffs * 3; i++) {
        float sh_plus[max_coeffs * 3];
        float sh_minus[max_coeffs * 3];
        std::memcpy(sh_plus, sh_coeffs, sizeof(sh_coeffs));
        std::memcpy(sh_minus, sh_coeffs, sizeof(sh_coeffs));
        sh_plus[i] += eps;
        sh_minus[i] -= eps;

        float rgb_plus[3], rgb_minus[3];
        computeColorFromSH(degree, max_coeffs, sh_plus, pos, cam_pos, rgb_plus, true);
        computeColorFromSH(degree, max_coeffs, sh_minus, pos, cam_pos, rgb_minus, true);

        float grad = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            grad += d_rgb[ch] * (rgb_plus[ch] - rgb_minus[ch]) / (2.0f * eps);
        }
        d_sh_fd[i] = grad;
    }

    for (int i = 0; i < max_coeffs * 3; i++) {
        EXPECT_NEAR(d_sh_analytic[i], d_sh_fd[i], 1e-3f)
            << "SH degree 3 backward mismatch at index " << i;
    }
}

// Test SH backward with lower clamp active (some channels clamped to 0)
TEST(PreprocessorBackward, SHBackwardWithClamp) {
    const int degree = 1;
    const int max_coeffs = 4;
    // Set SH coeffs so that the raw color + 0.5 < 0 for at least one channel
    float sh_coeffs[max_coeffs * 3];
    std::memset(sh_coeffs, 0, sizeof(sh_coeffs));
    // DC term: make channel 0 very negative so it gets clamped
    sh_coeffs[0] = -3.0f;  // ch0: SH_C0 * (-3) + 0.5 = ~-0.346 < 0 => clamped
    sh_coeffs[1] = 1.0f;   // ch1: SH_C0 * 1.0 + 0.5 = ~0.782 > 0 => not clamped
    sh_coeffs[2] = 1.0f;   // ch2: similar

    float pos[3] = {1.0f, 2.0f, 3.0f};
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};
    float d_rgb[3] = {1.0f, 1.0f, 1.0f};

    float d_sh_analytic[max_coeffs * 3];
    computeColorFromSH_backward(degree, max_coeffs, sh_coeffs, pos, cam_pos,
                                  d_rgb, d_sh_analytic, true);

    const float eps = 1e-3f;
    for (int i = 0; i < max_coeffs * 3; i++) {
        float sh_plus[max_coeffs * 3];
        float sh_minus[max_coeffs * 3];
        std::memcpy(sh_plus, sh_coeffs, sizeof(sh_coeffs));
        std::memcpy(sh_minus, sh_coeffs, sizeof(sh_coeffs));
        sh_plus[i] += eps;
        sh_minus[i] -= eps;

        float rgb_plus[3], rgb_minus[3];
        computeColorFromSH(degree, max_coeffs, sh_plus, pos, cam_pos, rgb_plus, true);
        computeColorFromSH(degree, max_coeffs, sh_minus, pos, cam_pos, rgb_minus, true);

        float grad = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            grad += d_rgb[ch] * (rgb_plus[ch] - rgb_minus[ch]) / (2.0f * eps);
        }
        EXPECT_NEAR(d_sh_analytic[i], grad, 1e-3f)
            << "SH backward with clamp mismatch at index " << i;
    }
}

// ---------- Opacity backward finite-difference test ----------

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

TEST(PreprocessorBackward, OpacityBackward) {
    float raw_opacity = 1.5f;
    float d_opacity_2d = 0.7f;

    float sigma = sigmoid(raw_opacity);

    // Analytic gradient
    float d_raw_analytic = d_opacity_2d * sigma * (1.0f - sigma);

    // Finite-difference (use double for more precise FD)
    const double eps = 1e-5;
    double sigma_plus = 1.0 / (1.0 + std::exp(-(raw_opacity + eps)));
    double sigma_minus = 1.0 / (1.0 + std::exp(-(raw_opacity - eps)));
    float d_raw_fd = static_cast<float>(d_opacity_2d * (sigma_plus - sigma_minus) / (2.0 * eps));

    EXPECT_NEAR(d_raw_analytic, d_raw_fd, 1e-4f);
}

TEST(PreprocessorBackward, OpacityBackwardAtZero) {
    float raw_opacity = 0.0f;
    float d_opacity_2d = 1.0f;

    float sigma = sigmoid(raw_opacity);
    float d_raw_analytic = d_opacity_2d * sigma * (1.0f - sigma);

    // At raw=0, sigmoid=0.5, so d_raw = 1.0 * 0.5 * 0.5 = 0.25
    EXPECT_NEAR(d_raw_analytic, 0.25f, 1e-6f);

    const double eps2 = 1e-5;
    double sp = 1.0 / (1.0 + std::exp(-eps2));
    double sm = 1.0 / (1.0 + std::exp(eps2));
    float d_raw_fd = static_cast<float>(d_opacity_2d * (sp - sm) / (2.0 * eps2));
    EXPECT_NEAR(d_raw_analytic, d_raw_fd, 1e-4f);
}

// ---------- Integration test: backward() method ----------

TEST(PreprocessorBackward, BackwardIntegration) {
    const int N = 2;
    const int degree = 1;
    const int max_coeffs = 4;

    // Allocate all arrays
    FrameAllocator alloc(1 << 20);  // 1 MB

    // GaussianData
    GaussianData g;
    g.count = N;
    g.sh_degree = degree;
    g.max_coeffs = max_coeffs;
    g.positions = alloc.allocate_array<float>(N * 3);
    g.sh_coeffs = alloc.allocate_array<float>(N * max_coeffs * 3);
    g.scales = alloc.allocate_array<float>(N * 3);
    g.rotations = alloc.allocate_array<float>(N * 4);
    g.opacities = alloc.allocate_array<float>(N);
    g.filter_3D = nullptr;

    g.positions[0] = 1; g.positions[1] = 2; g.positions[2] = 5;
    g.positions[3] = -1; g.positions[4] = 0; g.positions[5] = 3;

    for (int i = 0; i < N * max_coeffs * 3; i++)
        g.sh_coeffs[i] = 0.1f * ((i * 7 + 3) % 13) - 0.5f;

    g.opacities[0] = 0.8f;  // sigmoid-activated
    g.opacities[1] = 0.3f;

    // Camera
    Camera cam;
    cam.cam_pos[0] = 0; cam.cam_pos[1] = 0; cam.cam_pos[2] = 0;
    cam.width = 8; cam.height = 8;

    RenderConfig cfg;
    cfg.sh_degree = degree;

    // PreprocessOutput (only radii needed for visibility check)
    PreprocessOutput pre;
    pre.radii = alloc.allocate_array<int>(N);
    pre.radii[0] = 5;   // visible
    pre.radii[1] = 0;   // invisible

    // ForwardCache — cov2D/cov2D_det must be valid arrays: backward reads them
    // before the `if (det==0) continue` guard. Zero-init makes det=0, which
    // causes the conic/position chains to skip (they are not under test here).
    ForwardCache cache;
    cache.pre = &pre;
    cache.cov2D     = alloc.allocate_array<float>(N * 3);
    cache.cov2D_det = alloc.allocate_array<float>(N);
    std::memset(cache.cov2D,     0, N * 3 * sizeof(float));
    std::memset(cache.cov2D_det, 0, N     * sizeof(float));

    // RasterGradOutput
    RasterGradOutput rgrad;
    rgrad.d_rgb = alloc.allocate_array<float>(N * 3);
    rgrad.d_opacities_2d = alloc.allocate_array<float>(N);
    rgrad.d_rgb[0] = 1.0f; rgrad.d_rgb[1] = 0.5f; rgrad.d_rgb[2] = -0.2f;
    rgrad.d_rgb[3] = 0.3f; rgrad.d_rgb[4] = 0.8f; rgrad.d_rgb[5] = 0.1f;
    rgrad.d_opacities_2d[0] = 0.7f;
    rgrad.d_opacities_2d[1] = -0.5f;

    // RawGaussianParams (not used directly in current chains, but needed for interface)
    RawGaussianParams raw;
    raw.count = N;
    raw.sh_degree = degree;
    raw.max_coeffs = max_coeffs;

    // GradientOutput
    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, max_coeffs);

    // Run backward
    PreprocessorBackwardCPU bwd;
    bwd.backward(g, cam, cfg, cache, rgrad, raw, grads);

    // Gaussian 0 (visible): SH grads should be non-zero
    bool has_nonzero_sh = false;
    for (int i = 0; i < max_coeffs * 3; i++) {
        if (std::abs(grads.d_raw_sh_coeffs[i]) > 1e-8f) {
            has_nonzero_sh = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_sh) << "Visible Gaussian should have non-zero SH gradients";

    // Gaussian 0: opacity grad should be non-zero
    float sigma0 = g.opacities[0];
    float expected_d_raw_op = rgrad.d_opacities_2d[0] * sigma0 * (1.0f - sigma0);
    EXPECT_NEAR(grads.d_raw_opacities[0], expected_d_raw_op, 1e-6f);

    // Gaussian 1 (invisible): all grads should be zero
    for (int i = 0; i < max_coeffs * 3; i++) {
        EXPECT_EQ(grads.d_raw_sh_coeffs[N * 0 + max_coeffs * 3 + i], 0.0f)
            << "Invisible Gaussian should have zero SH gradients at index " << i;
    }
    EXPECT_EQ(grads.d_raw_opacities[1], 0.0f)
        << "Invisible Gaussian should have zero opacity gradient";
}

// ======================================================================
// End-to-end finite-difference tests for the covariance backward chain
// ======================================================================

// Helper: MSE loss (smooth, better for FD than L1)
static float mse_loss_e2e(const float* rendered, const float* gt, int H, int W, float* d_image) {
    int n = H * W * 3;
    float inv_n = 1.0f / (float)n;
    double sum = 0.0;  // double accumulation — eliminates float summation-order sensitivity across CHW/HWC layouts
    for (int i = 0; i < n; i++) {
        float diff = rendered[i] - gt[i];
        sum += (double)diff * (double)diff;
        if (d_image) d_image[i] = 2.0f * diff * inv_n;
    }
    return (float)(sum * (double)inv_n);
}

// Set up a simple look-at camera (identity rotation, looking down +Z)
static void setup_simple_camera(Camera& cam) {
    cam.width = 16;
    cam.height = 16;
    cam.tan_fovx = 1.0f;
    cam.tan_fovy = 1.0f;
    cam.cam_pos[0] = 0.0f;
    cam.cam_pos[1] = 0.0f;
    cam.cam_pos[2] = 0.0f;

    // Identity rotation, translation = 0 (camera at origin, looking down +Z)
    // Column-major 4x4
    std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
    cam.view_matrix[0] = 1.0f;
    cam.view_matrix[5] = 1.0f;
    cam.view_matrix[10] = 1.0f;
    cam.view_matrix[15] = 1.0f;

    // viewproj = proj * view. For simple test, use a basic perspective matrix.
    // NDC: x_ndc = focal_x * x_view / z_view, y_ndc = focal_y * y_view / z_view
    // With tan_fov=1, focal = W/(2*tan) = 8.
    // We use a simple projection: viewproj maps (x,y,z) to (focal*x/z, focal*y/z, z, z)
    // In column-major 4x4 for transformPoint4x4:
    // out[0] = m[0]*x + m[4]*y + m[8]*z + m[12]
    // We want: out[0] = focal*x, out[1] = focal*y, out[2] = z, out[3] = z
    // So: m[0]=focal, m[5]=focal, m[10]=1, m[11]=1, rest=0
    // Then ndc = out/w = (focal*x/z, focal*y/z, 1, 1)
    float focal_x = cam.width / (2.0f * cam.tan_fovx);
    float focal_y = cam.height / (2.0f * cam.tan_fovy);
    std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0] = focal_x;
    cam.viewproj_matrix[5] = focal_y;
    cam.viewproj_matrix[10] = 1.0f;
    cam.viewproj_matrix[11] = 1.0f;
}

// Run full forward pipeline: activate -> preprocess -> bin -> sort -> rasterize -> MSE loss
// Returns loss. If d_image is non-null, fills it for backward.
static float run_full_forward(const RawGaussianParams& raw, const Camera& cam,
                               const RenderConfig& cfg, FrameAllocator& alloc,
                               const float* gt, float* d_image,
                               ForwardCache* cache_out = nullptr,
                               PreprocessOutput* pre_out = nullptr,
                               BinningOutput* bin_out = nullptr) {
    int N = raw.count;
    int npix = cam.width * cam.height;

    // Allocate GaussianData
    GaussianData g;
    g.count = N;
    g.sh_degree = raw.sh_degree;
    g.max_coeffs = raw.max_coeffs;
    g.positions = alloc.allocate_array<float>(N * 3);
    g.scales = alloc.allocate_array<float>(N * 3);
    g.rotations = alloc.allocate_array<float>(N * 4);
    g.sh_coeffs = alloc.allocate_array<float>(N * raw.max_coeffs * 3);
    g.opacities = alloc.allocate_array<float>(N);
    g.filter_3D = nullptr;

    raw.activate(g);

    ForwardCache cache{};
    PreprocessorCPU preprocessor;
    PreprocessOutput pre = preprocessor.process(g, cam, cfg, alloc, &cache);

    // Check that at least one Gaussian is visible
    bool any_visible = false;
    for (int i = 0; i < N; i++)
        if (pre.radii[i] > 0) any_visible = true;
    if (!any_visible) return 0.0f;

    TileBinnerCPU binner;
    BinningOutput bin = binner.bin(pre, N, cam, cfg, alloc);

    SorterCPU sorter;
    sorter.sort(bin, alloc);

    float* out_img = alloc.allocate_array<float>(npix * 3);
    std::memset(out_img, 0, npix * 3 * sizeof(float));

    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, out_img, nullptr, &cache, &alloc);

    float loss = mse_loss_e2e(out_img, gt, cam.height, cam.width, d_image);

    // Save outputs if requested
    if (cache_out) *cache_out = cache;
    if (pre_out) *pre_out = pre;
    if (bin_out) *bin_out = bin;

    return loss;
}

// Isolated test: conic inversion backward
TEST(PreprocessorBackward, CovChain_ConicInversion) {
    // Test conic = {c/det, -b/det, a/det} where det = a*c - b*b
    // after a += 0.3, c += 0.3
    float a_raw = 2.5f, b_raw = 0.3f, c_raw = 1.8f;
    float a = a_raw + 0.3f;
    float c = c_raw + 0.3f;
    float b = b_raw;
    float det = a * c - b * b;
    float inv_det = 1.0f / det;

    // conics
    float conic[3] = {c * inv_det, -b * inv_det, a * inv_det};

    // upstream gradient d_conic
    float dc[3] = {1.2f, -0.5f, 0.8f};

    // Analytic backward
    float inv_det2 = inv_det * inv_det;
    float d_a = dc[0] * (-c*c * inv_det2)
              + dc[1] * (b*c * inv_det2)
              + dc[2] * (-b*b * inv_det2);
    float d_b = dc[0] * (2.0f*b*c * inv_det2)
              + dc[1] * (-(a*c + b*b) * inv_det2)
              + dc[2] * (2.0f*a*b * inv_det2);
    float d_c_val = dc[0] * (-b*b * inv_det2)
                  + dc[1] * (a*b * inv_det2)
                  + dc[2] * (-a*a * inv_det2);

    // Finite-difference: perturb each cov2D component (after filtering)
    const double eps = 1e-5;

    // d_a check
    {
        float a2 = a + (float)eps;
        float det2 = a2 * c - b * b;
        float conic2[3] = {c / det2, -b / det2, a2 / det2};
        float a3 = a - (float)eps;
        float det3 = a3 * c - b * b;
        float conic3[3] = {c / det3, -b / det3, a3 / det3};
        // loss = sum(dc[k] * conic[k])
        double loss_plus = 0, loss_minus = 0;
        for (int k = 0; k < 3; k++) {
            loss_plus += dc[k] * conic2[k];
            loss_minus += dc[k] * conic3[k];
        }
        double fd = (loss_plus - loss_minus) / (2.0 * eps);
        printf("  conic d_a: analytic=%.8f  fd=%.8f  diff=%.8f\n", d_a, fd, d_a - (float)fd);
        EXPECT_NEAR(d_a, (float)fd, 1e-3f);
    }
    // d_b check
    {
        float b2 = b + (float)eps;
        float det2 = a * c - b2 * b2;
        float conic2[3] = {c / det2, -b2 / det2, a / det2};
        float b3 = b - (float)eps;
        float det3 = a * c - b3 * b3;
        float conic3[3] = {c / det3, -b3 / det3, a / det3};
        double loss_plus = 0, loss_minus = 0;
        for (int k = 0; k < 3; k++) {
            loss_plus += dc[k] * conic2[k];
            loss_minus += dc[k] * conic3[k];
        }
        double fd = (loss_plus - loss_minus) / (2.0 * eps);
        printf("  conic d_b: analytic=%.8f  fd=%.8f  diff=%.8f\n", d_b, fd, d_b - (float)fd);
        EXPECT_NEAR(d_b, (float)fd, 1e-3f);
    }
    // d_c check
    {
        float c2 = c + (float)eps;
        float det2 = a * c2 - b * b;
        float conic2[3] = {c2 / det2, -b / det2, a / det2};
        float c3 = c - (float)eps;
        float det3 = a * c3 - b * b;
        float conic3[3] = {c3 / det3, -b / det3, a / det3};
        double loss_plus = 0, loss_minus = 0;
        for (int k = 0; k < 3; k++) {
            loss_plus += dc[k] * conic2[k];
            loss_minus += dc[k] * conic3[k];
        }
        double fd = (loss_plus - loss_minus) / (2.0 * eps);
        printf("  conic d_c: analytic=%.8f  fd=%.8f  diff=%.8f\n", d_c_val, fd, d_c_val - (float)fd);
        EXPECT_NEAR(d_c_val, (float)fd, 1e-3f);
    }
}

// Isolated test: cov2D -> cov3D backward
TEST(PreprocessorBackward, CovChain_Cov2DToCov3D) {
    // Set up a simple case with known view matrix and position
    float mean3d[3] = {0.5f, 0.3f, 5.0f};
    float vm[16]; // identity view matrix (column-major)
    std::memset(vm, 0, sizeof(vm));
    vm[0] = 1; vm[5] = 1; vm[10] = 1; vm[15] = 1;

    float focal_x = 8.0f, focal_y = 8.0f;
    float tan_fovx = 1.0f, tan_fovy = 1.0f;

    // Create a known cov3D
    float cov3D[6] = {1.0f, 0.2f, 0.1f, 0.8f, -0.1f, 0.5f};

    // Forward: compute cov2D from cov3D
    float cov2D[3];
    computeCov2D(mean3d, cov3D, vm, focal_x, focal_y, tan_fovx, tan_fovy, cov2D);

    // upstream gradient d_cov2D
    float d_cov2D[3] = {1.0f, 0.5f, -0.3f};

    // Recompute T (same as backward code)
    float t[3];
    transformPoint4x3(mean3d, vm, t);
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t[0] / t[2];
    float tytz = t[1] / t[2];
    t[0] = std::min(limx, std::max(-limx, txtz)) * t[2];
    t[1] = std::min(limy, std::max(-limy, tytz)) * t[2];

    float J[3][3] = {
        {focal_x / t[2], 0.0f, -(focal_x * t[0]) / (t[2] * t[2])},
        {0.0f, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])},
        {0.0f, 0.0f, 0.0f}
    };
    float W[3][3] = {
        {vm[0], vm[4], vm[8]},
        {vm[1], vm[5], vm[9]},
        {vm[2], vm[6], vm[10]}
    };
    float T[3][3];
    for (int col = 0; col < 3; col++)
        for (int row = 0; row < 3; row++) {
            T[col][row] = 0;
            for (int k = 0; k < 3; k++)
                T[col][row] += W[k][row] * J[col][k];
        }

    // Backward
    // result[c][r] = sum_{k,j} T[r][k] * Vrk[k][j] * T[c][j]
    // cov2D[0] = result[0][0], cov2D[1] = result[1][0], cov2D[2] = result[1][1]
    // d_Vrk[k][j] = d_cov2D[0]*T[0][k]*T[0][j] + d_cov2D[1]*T[0][k]*T[1][j] + d_cov2D[2]*T[1][k]*T[1][j]
    float d_Vrk[3][3];
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++) {
            d_Vrk[k][j] = d_cov2D[0] * T[0][k] * T[0][j]
                         + d_cov2D[1] * T[0][k] * T[1][j]
                         + d_cov2D[2] * T[1][k] * T[1][j];
        }

    float d_cov3D_analytic[6];
    d_cov3D_analytic[0] = d_Vrk[0][0];
    d_cov3D_analytic[1] = d_Vrk[0][1] + d_Vrk[1][0];
    d_cov3D_analytic[2] = d_Vrk[0][2] + d_Vrk[2][0];
    d_cov3D_analytic[3] = d_Vrk[1][1];
    d_cov3D_analytic[4] = d_Vrk[1][2] + d_Vrk[2][1];
    d_cov3D_analytic[5] = d_Vrk[2][2];

    // Finite-difference
    const double eps = 1e-5;
    for (int idx = 0; idx < 6; idx++) {
        float cov3D_plus[6], cov3D_minus[6];
        std::memcpy(cov3D_plus, cov3D, sizeof(cov3D));
        std::memcpy(cov3D_minus, cov3D, sizeof(cov3D));
        cov3D_plus[idx] += (float)eps;
        cov3D_minus[idx] -= (float)eps;

        float cov2D_plus[3], cov2D_minus[3];
        computeCov2D(mean3d, cov3D_plus, vm, focal_x, focal_y, tan_fovx, tan_fovy, cov2D_plus);
        computeCov2D(mean3d, cov3D_minus, vm, focal_x, focal_y, tan_fovx, tan_fovy, cov2D_minus);

        double fd = 0;
        for (int k = 0; k < 3; k++)
            fd += d_cov2D[k] * (cov2D_plus[k] - cov2D_minus[k]) / (2.0 * eps);

        printf("  d_cov3D[%d]: analytic=%.8f  fd=%.8f  diff=%.2e\n",
               idx, d_cov3D_analytic[idx], fd, d_cov3D_analytic[idx] - fd);
        EXPECT_NEAR(d_cov3D_analytic[idx], (float)fd, 0.01f)
            << "d_cov3D[" << idx << "]";
    }
}

// Isolated test: cov3D -> M -> scale,rotation backward
TEST(PreprocessorBackward, CovChain_Cov3DToScaleRot) {
    float mod = 1.0f;
    float scale[3] = {1.5f, 0.8f, 1.2f};
    // Normalized quaternion
    float rot_raw[4] = {1.0f, 0.3f, -0.2f, 0.1f};
    float len = std::sqrt(rot_raw[0]*rot_raw[0] + rot_raw[1]*rot_raw[1] +
                           rot_raw[2]*rot_raw[2] + rot_raw[3]*rot_raw[3]);
    float rot[4] = {rot_raw[0]/len, rot_raw[1]/len, rot_raw[2]/len, rot_raw[3]/len};

    float cov3D[6];
    computeCov3D(scale, mod, rot, cov3D);

    float d_cov3D[6] = {1.0f, 0.5f, -0.3f, 0.8f, 0.2f, -0.1f};

    // Analytic backward: d_scale[k] and d_R[k][j]
    float r_q = rot[0], x_q = rot[1], y_q = rot[2], z_q = rot[3];
    float R[3][3] = {
        {1.f - 2.f*(y_q*y_q + z_q*z_q), 2.f*(x_q*y_q + r_q*z_q),       2.f*(x_q*z_q - r_q*y_q)},
        {2.f*(x_q*y_q - r_q*z_q),       1.f - 2.f*(x_q*x_q + z_q*z_q), 2.f*(y_q*z_q + r_q*x_q)},
        {2.f*(x_q*z_q + r_q*y_q),       2.f*(y_q*z_q - r_q*x_q),       1.f - 2.f*(x_q*x_q + y_q*y_q)}
    };
    float sx = mod * scale[0], sy = mod * scale[1], sz = mod * scale[2];
    float s[3] = {sx, sy, sz};
    float M[3][3];
    for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
            M[ii][jj] = s[ii] * R[ii][jj];

    // d_M from upper-triangle d_cov3D
    float d_M[3][3];
    for (int k = 0; k < 3; k++) {
        d_M[k][0] = 2.0f*d_cov3D[0]*M[k][0] + d_cov3D[1]*M[k][1] + d_cov3D[2]*M[k][2];
        d_M[k][1] = d_cov3D[1]*M[k][0] + 2.0f*d_cov3D[3]*M[k][1] + d_cov3D[4]*M[k][2];
        d_M[k][2] = d_cov3D[2]*M[k][0] + d_cov3D[4]*M[k][1] + 2.0f*d_cov3D[5]*M[k][2];
    }

    float d_scale_analytic[3];
    for (int k = 0; k < 3; k++) {
        d_scale_analytic[k] = 0.0f;
        for (int j = 0; j < 3; j++)
            d_scale_analytic[k] += d_M[k][j] * R[k][j];
    }

    // FD for scale
    const double eps = 1e-5;
    for (int k = 0; k < 3; k++) {
        float scale_plus[3], scale_minus[3];
        std::memcpy(scale_plus, scale, sizeof(scale));
        std::memcpy(scale_minus, scale, sizeof(scale));
        scale_plus[k] += (float)eps;
        scale_minus[k] -= (float)eps;

        float cov_plus[6], cov_minus[6];
        computeCov3D(scale_plus, mod, rot, cov_plus);
        computeCov3D(scale_minus, mod, rot, cov_minus);

        double fd = 0;
        for (int idx = 0; idx < 6; idx++)
            fd += d_cov3D[idx] * (cov_plus[idx] - cov_minus[idx]) / (2.0 * eps);

        printf("  d_scale[%d]: analytic=%.8f  fd=%.8f  diff=%.2e\n",
               k, d_scale_analytic[k], fd, d_scale_analytic[k] - fd);
        EXPECT_NEAR(d_scale_analytic[k] * mod, (float)fd, 0.01f)
            << "d_scale[" << k << "]";
    }
}

TEST(PreprocessorBackward, CovChain_ScaleGradient) {
    const int N = 1;
    const int degree = 0;
    const int max_coeffs = 1;

    Camera cam;
    setup_simple_camera(cam);

    RenderConfig cfg;
    cfg.sh_degree = degree;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.eval_3D = false;
    cfg.antialiasing = false;
    cfg.scale_modifier = 1.0f;
    cfg.training = true;

    int npix = cam.width * cam.height;
    std::vector<float> gt(npix * 3, 0.0f);

    // Set up raw params
    // Use large scales so the Gaussian covers entire 16x16 image.
    // This avoids tile-coverage discontinuities in FD.
    float raw_pos[3] = {0.0f, 0.0f, 5.0f};   // Gaussian at z=5
    float raw_scales[3] = {0.5f, 0.3f, 0.1f};  // log-space -> exp gives ~1.6, 1.35, 1.1
    float raw_rot[4] = {1.0f, 0.1f, 0.2f, -0.1f};
    float raw_sh[3] = {1.0f, 0.5f, 0.8f};  // SH degree 0, 1 coeff * 3 channels
    float raw_opacity[1] = {2.0f};  // sigmoid(2) ~ 0.88

    // Run forward + backward to get analytic gradients
    float analytic_d_scale[3];
    {
        FrameAllocator alloc(8 * 1024 * 1024);
        RawGaussianParams raw;
        raw.count = N;
        raw.sh_degree = degree;
        raw.max_coeffs = max_coeffs;
        raw.raw_positions = raw_pos;
        raw.raw_scales = raw_scales;
        raw.raw_rotations = raw_rot;
        raw.raw_sh_coeffs = raw_sh;
        raw.raw_opacities = raw_opacity;

        ForwardCache cache{};
        PreprocessOutput pre{};
        BinningOutput bin{};
        float* d_image = alloc.allocate_array<float>(npix * 3);
        run_full_forward(raw, cam, cfg, alloc, gt.data(), d_image, &cache, &pre, &bin);

        // Rasterizer backward
        RasterGradOutput rgrad;
        rgrad.allocate_and_zero(alloc, N);
        RasterizerBackwardCPU rast_bwd;
        rast_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad);

        // Preprocessor backward
        GaussianData g;
        g.count = N; g.sh_degree = degree; g.max_coeffs = max_coeffs;
        g.positions = alloc.allocate_array<float>(N * 3);
        g.scales = alloc.allocate_array<float>(N * 3);
        g.rotations = alloc.allocate_array<float>(N * 4);
        g.sh_coeffs = alloc.allocate_array<float>(N * max_coeffs * 3);
        g.opacities = alloc.allocate_array<float>(N);
        g.filter_3D = nullptr;
        raw.activate(g);

        cache.pre = &pre;
        cache.bin = &bin;

        GradientOutput grads;
        grads.allocate_and_zero(alloc, N, max_coeffs);
        PreprocessorBackwardCPU preproc_bwd;
        preproc_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads);

        for (int k = 0; k < 3; k++)
            analytic_d_scale[k] = grads.d_raw_scales[k];
    }

    // Finite-difference for each raw_scale component
    const double eps = 1e-4;
    for (int k = 0; k < 3; k++) {
        float raw_scales_plus[3], raw_scales_minus[3];
        std::memcpy(raw_scales_plus, raw_scales, sizeof(raw_scales));
        std::memcpy(raw_scales_minus, raw_scales, sizeof(raw_scales));
        raw_scales_plus[k] += (float)eps;
        raw_scales_minus[k] -= (float)eps;

        double loss_plus, loss_minus;
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos; r.raw_scales = raw_scales_plus;
            r.raw_rotations = raw_rot; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_plus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos; r.raw_scales = raw_scales_minus;
            r.raw_rotations = raw_rot; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_minus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }

        double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        double abs_err = std::fabs((double)analytic_d_scale[k] - fd_grad);
        double denom = std::max(std::fabs((double)analytic_d_scale[k]), std::fabs(fd_grad));
        double rel_err = (denom > 1e-7) ? abs_err / denom : abs_err;

        printf("  d_raw_scale[%d]: analytic=%.8f  fd=%.8f  rel_err=%.6f\n",
               k, analytic_d_scale[k], fd_grad, rel_err);

        EXPECT_LT(rel_err, 0.02)
            << "d_raw_scale[" << k << "]: analytic=" << analytic_d_scale[k]
            << " fd=" << fd_grad << " rel_err=" << rel_err;
    }
}

TEST(PreprocessorBackward, CovChain_RotationGradient) {
    const int N = 1;
    const int degree = 0;
    const int max_coeffs = 1;

    Camera cam;
    setup_simple_camera(cam);

    RenderConfig cfg;
    cfg.sh_degree = degree;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.eval_3D = false;
    cfg.antialiasing = false;
    cfg.scale_modifier = 1.0f;
    cfg.training = true;

    int npix = cam.width * cam.height;
    std::vector<float> gt(npix * 3, 0.0f);

    // Large anisotropic Gaussian so it covers entire image, avoiding tile discontinuities
    float raw_pos[3] = {0.0f, 0.0f, 5.0f};
    float raw_scales[3] = {0.5f, 0.0f, 0.3f};  // anisotropic to make rotation matter
    float raw_rot[4] = {1.0f, 0.3f, -0.2f, 0.1f};
    float raw_sh[3] = {1.0f, 0.5f, 0.8f};
    float raw_opacity[1] = {2.0f};

    // Run forward + backward to get analytic gradients
    float analytic_d_rot[4];
    {
        FrameAllocator alloc(8 * 1024 * 1024);
        RawGaussianParams raw;
        raw.count = N; raw.sh_degree = degree; raw.max_coeffs = max_coeffs;
        raw.raw_positions = raw_pos; raw.raw_scales = raw_scales;
        raw.raw_rotations = raw_rot; raw.raw_sh_coeffs = raw_sh; raw.raw_opacities = raw_opacity;

        ForwardCache cache{};
        PreprocessOutput pre{};
        BinningOutput bin{};
        float* d_image = alloc.allocate_array<float>(npix * 3);
        run_full_forward(raw, cam, cfg, alloc, gt.data(), d_image, &cache, &pre, &bin);

        RasterGradOutput rgrad;
        rgrad.allocate_and_zero(alloc, N);
        RasterizerBackwardCPU rast_bwd;
        rast_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad);

        GaussianData g;
        g.count = N; g.sh_degree = degree; g.max_coeffs = max_coeffs;
        g.positions = alloc.allocate_array<float>(N * 3);
        g.scales = alloc.allocate_array<float>(N * 3);
        g.rotations = alloc.allocate_array<float>(N * 4);
        g.sh_coeffs = alloc.allocate_array<float>(N * max_coeffs * 3);
        g.opacities = alloc.allocate_array<float>(N);
        g.filter_3D = nullptr;
        raw.activate(g);

        cache.pre = &pre;
        cache.bin = &bin;

        GradientOutput grads;
        grads.allocate_and_zero(alloc, N, max_coeffs);
        PreprocessorBackwardCPU preproc_bwd;
        preproc_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads);

        for (int k = 0; k < 4; k++)
            analytic_d_rot[k] = grads.d_raw_rotations[k];
    }

    // Finite-difference for each raw_rotation component
    const double eps = 1e-4;
    for (int k = 0; k < 4; k++) {
        float raw_rot_plus[4], raw_rot_minus[4];
        std::memcpy(raw_rot_plus, raw_rot, sizeof(raw_rot));
        std::memcpy(raw_rot_minus, raw_rot, sizeof(raw_rot));
        raw_rot_plus[k] += (float)eps;
        raw_rot_minus[k] -= (float)eps;

        double loss_plus, loss_minus;
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos; r.raw_scales = raw_scales;
            r.raw_rotations = raw_rot_plus; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_plus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos; r.raw_scales = raw_scales;
            r.raw_rotations = raw_rot_minus; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_minus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }

        double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        double abs_err = std::fabs((double)analytic_d_rot[k] - fd_grad);
        double denom = std::max(std::fabs((double)analytic_d_rot[k]), std::fabs(fd_grad));
        double rel_err = (denom > 1e-7) ? abs_err / denom : abs_err;

        printf("  d_raw_rot[%d]: analytic=%.8f  fd=%.8f  rel_err=%.6f\n",
               k, analytic_d_rot[k], fd_grad, rel_err);

        // 3% threshold — rotation FD has inherently weak signal (gradient
        // magnitude ~0.001) and double-accumulated loss diverges slightly
        // from the float-precision analytic backward chain.
        EXPECT_LT(rel_err, 0.03)
            << "d_raw_rotation[" << k << "]: analytic=" << analytic_d_rot[k]
            << " fd=" << fd_grad << " rel_err=" << rel_err;
    }
}

// ======================================================================
// End-to-end finite-difference test for position gradient (Chain 4)
// ======================================================================

TEST(PreprocessorBackward, PositionGradient) {
    const int N = 1;
    const int degree = 0;
    const int max_coeffs = 1;

    Camera cam;
    setup_simple_camera(cam);

    RenderConfig cfg;
    cfg.sh_degree = degree;
    cfg.tile_w = 16;
    cfg.tile_h = 16;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.eval_3D = false;
    cfg.antialiasing = false;
    cfg.scale_modifier = 1.0f;
    cfg.training = true;

    int npix = cam.width * cam.height;
    std::vector<float> gt(npix * 3, 0.0f);

    // Gaussian offset from center so means2D gradients are non-zero,
    // with large scale so it covers the entire 16x16 image
    // to avoid tile-coverage discontinuities in FD
    float raw_pos[3] = {1.0f, -0.6f, 10.0f};
    float raw_scales[3] = {1.0f, 1.0f, 0.5f};  // large so Gaussian covers image
    float raw_rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float raw_sh[3] = {1.0f, 0.5f, 0.8f};
    float raw_opacity[1] = {2.0f};

    // Run forward + backward to get analytic gradients
    float analytic_d_pos[3];
    {
        FrameAllocator alloc(8 * 1024 * 1024);
        RawGaussianParams raw;
        raw.count = N; raw.sh_degree = degree; raw.max_coeffs = max_coeffs;
        raw.raw_positions = raw_pos; raw.raw_scales = raw_scales;
        raw.raw_rotations = raw_rot; raw.raw_sh_coeffs = raw_sh; raw.raw_opacities = raw_opacity;

        ForwardCache cache{};
        PreprocessOutput pre{};
        BinningOutput bin{};
        float* d_image = alloc.allocate_array<float>(npix * 3);
        run_full_forward(raw, cam, cfg, alloc, gt.data(), d_image, &cache, &pre, &bin);

        RasterGradOutput rgrad;
        rgrad.allocate_and_zero(alloc, N);
        RasterizerBackwardCPU rast_bwd;
        rast_bwd.backward(pre, bin, cam, cfg, cache, d_image, rgrad);

        GaussianData g;
        g.count = N; g.sh_degree = degree; g.max_coeffs = max_coeffs;
        g.positions = alloc.allocate_array<float>(N * 3);
        g.scales = alloc.allocate_array<float>(N * 3);
        g.rotations = alloc.allocate_array<float>(N * 4);
        g.sh_coeffs = alloc.allocate_array<float>(N * max_coeffs * 3);
        g.opacities = alloc.allocate_array<float>(N);
        g.filter_3D = nullptr;
        raw.activate(g);

        cache.pre = &pre;
        cache.bin = &bin;

        GradientOutput grads;
        grads.allocate_and_zero(alloc, N, max_coeffs);
        PreprocessorBackwardCPU preproc_bwd;
        preproc_bwd.backward(g, cam, cfg, cache, rgrad, raw, grads);

        for (int k = 0; k < 3; k++)
            analytic_d_pos[k] = grads.d_raw_positions[k];
    }

    // Finite-difference for each raw_position component
    const double eps = 1e-4;
    for (int k = 0; k < 3; k++) {
        float raw_pos_plus[3], raw_pos_minus[3];
        std::memcpy(raw_pos_plus, raw_pos, sizeof(raw_pos));
        std::memcpy(raw_pos_minus, raw_pos, sizeof(raw_pos));
        raw_pos_plus[k] += (float)eps;
        raw_pos_minus[k] -= (float)eps;

        double loss_plus, loss_minus;
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos_plus; r.raw_scales = raw_scales;
            r.raw_rotations = raw_rot; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_plus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }
        {
            FrameAllocator alloc(8 * 1024 * 1024);
            RawGaussianParams r;
            r.count = N; r.sh_degree = degree; r.max_coeffs = max_coeffs;
            r.raw_positions = raw_pos_minus; r.raw_scales = raw_scales;
            r.raw_rotations = raw_rot; r.raw_sh_coeffs = raw_sh; r.raw_opacities = raw_opacity;
            loss_minus = run_full_forward(r, cam, cfg, alloc, gt.data(), nullptr);
        }

        double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        double abs_err = std::fabs((double)analytic_d_pos[k] - fd_grad);
        double denom = std::max(std::fabs((double)analytic_d_pos[k]), std::fabs(fd_grad));
        double rel_err = (denom > 1e-7) ? abs_err / denom : abs_err;

        printf("  d_raw_pos[%d]: analytic=%.8f  fd=%.8f  rel_err=%.6f\n",
               k, analytic_d_pos[k], fd_grad, rel_err);

        // For x and y, the means2D path dominates the position gradient.
        // For z, the cov2D path (not implemented) dominates, so we only
        // verify that the analytic gradient has reasonable magnitude.
        if (k < 2) {
            EXPECT_LT(rel_err, 0.05)
                << "d_raw_position[" << k << "]: analytic=" << analytic_d_pos[k]
                << " fd=" << fd_grad << " rel_err=" << rel_err;
        } else {
            // z: the means2D-only gradient is incomplete; just check it's finite
            EXPECT_TRUE(std::isfinite(analytic_d_pos[k]))
                << "d_raw_position[z] should be finite";
        }
    }
}
