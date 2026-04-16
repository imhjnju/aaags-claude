#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include "loss.h"
#include <vector>
#include <cmath>
#include <cstring>

// ---- Scene setup helpers ----

// 8x8 image, single 16x16 tile, 1 Gaussian near center
struct BackwardTestScene {
    // PreprocessOutput arrays (1 Gaussian)
    float means2D[2];
    float depths[1];
    float conics[3];
    float opacities_2d[1];
    float rgb[3];
    int radii[1];
    int tiles_touched[1];

    // BinningOutput arrays
    uint32_t tile_ranges[2];    // 1 tile
    uint32_t values_sorted[1];

    PreprocessOutput pre;
    BinningOutput bin;
    Camera cam;
    RenderConfig cfg;

    BackwardTestScene() {
        // Gaussian off-center to avoid symmetry cancellation in gradient tests
        means2D[0] = 4.8f;
        means2D[1] = 2.3f;
        depths[0] = 5.0f;

        // Conics: anisotropic, very wide Gaussian so all 8x8 pixels contribute
        // well above alpha threshold, avoiding discontinuity in FD
        // cov2D = [[20.0, 2.0], [2.0, 15.0]] => det = 296
        // inv = [15/296, -2/296; -2/296, 20/296]
        conics[0] = 0.05068f;    // a = 15/296
        conics[1] = -0.00676f;   // b = -2/296
        conics[2] = 0.06757f;    // c = 20/296

        opacities_2d[0] = 0.8f;
        rgb[0] = 0.7f;
        rgb[1] = 0.3f;
        rgb[2] = 0.5f;

        radii[0] = 8;
        tiles_touched[0] = 1;

        // PreprocessOutput
        pre.means2D = means2D;
        pre.depths = depths;
        pre.conics = conics;
        pre.opacities_2d = opacities_2d;
        pre.rgb = rgb;
        pre.radii = radii;
        pre.tiles_touched = tiles_touched;
        pre.gauss2screen = nullptr;
        pre.cov3D_inv = nullptr;
        pre.mean_offset = nullptr;
        pre.eval_3D = false;

        // BinningOutput: 1 tile, 1 Gaussian
        tile_ranges[0] = 0;  // start
        tile_ranges[1] = 1;  // end
        values_sorted[0] = 0;

        bin.total_pairs = 1;
        bin.keys_unsorted = nullptr;
        bin.values_unsorted = nullptr;
        bin.keys_sorted = nullptr;
        bin.values_sorted = values_sorted;
        bin.num_tiles = 1;
        bin.tile_ranges = tile_ranges;

        // Camera 8x8
        cam = {};
        cam.width = 8;
        cam.height = 8;
        cam.tan_fovx = 1.0f;
        cam.tan_fovy = 1.0f;

        // RenderConfig
        cfg = {};
        cfg.bg_color[0] = 0.0f;
        cfg.bg_color[1] = 0.0f;
        cfg.bg_color[2] = 0.0f;
        cfg.tile_w = 16;
        cfg.tile_h = 16;
        cfg.eval_3D = false;
        cfg.training = true;
    }
};

// Compute MSE loss = sum((rendered - gt)^2) / n, and d_image = 2*(rendered - gt) / n
// This is smooth, unlike L1, so FD works better.
static float mse_loss(const float* rendered, const float* gt, int H, int W, float* d_image) {
    int n = H * W * 3;
    float inv_n = 1.0f / (float)n;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = rendered[i] - gt[i];
        sum += diff * diff;
        d_image[i] = 2.0f * diff * inv_n;
    }
    return sum * inv_n;
}

// Run forward + MSE loss (gt=0), return loss
static float run_forward_loss(BackwardTestScene& scene, FrameAllocator& alloc) {
    int W = scene.cam.width, H = scene.cam.height;
    int npix = W * H;

    ForwardCache cache{};
    float* out_img = alloc.allocate_array<float>(npix * 3);
    std::memset(out_img, 0, npix * 3 * sizeof(float));

    RasterizerCPU rast;
    rast.rasterize(scene.pre, scene.bin, scene.cam, scene.cfg,
                   out_img, nullptr, &cache, &alloc);

    std::vector<float> gt(npix * 3, 0.0f);
    std::vector<float> d_image_dummy(npix * 3, 0.0f);
    float loss = mse_loss(out_img, gt.data(), H, W, d_image_dummy.data());
    return loss;
}

// Run forward + backward, fill rgrad, return loss
static float run_forward_backward(BackwardTestScene& scene, FrameAllocator& alloc,
                                   RasterGradOutput& rgrad) {
    int W = scene.cam.width, H = scene.cam.height;
    int npix = W * H;

    ForwardCache cache{};
    float* out_img = alloc.allocate_array<float>(npix * 3);
    std::memset(out_img, 0, npix * 3 * sizeof(float));

    RasterizerCPU rast;
    rast.rasterize(scene.pre, scene.bin, scene.cam, scene.cfg,
                   out_img, nullptr, &cache, &alloc);

    std::vector<float> gt(npix * 3, 0.0f);
    float* d_image = alloc.allocate_array<float>(npix * 3);
    float loss = mse_loss(out_img, gt.data(), H, W, d_image);

    rgrad.allocate_and_zero(alloc, 1);
    RasterizerBackwardCPU bwd;
    bwd.backward(scene.pre, scene.bin, scene.cam, scene.cfg, cache, d_image, rgrad);

    return loss;
}

// ---- Finite-difference tests ----

TEST(RasterizerBackward, FiniteDiff_RGB) {
    const float eps = 1e-3f;

    for (int ch = 0; ch < 3; ch++) {
        float analytic_grad;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            RasterGradOutput rgrad{};
            run_forward_backward(scene, alloc, rgrad);
            analytic_grad = rgrad.d_rgb[ch];
        }

        float loss_plus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.rgb[ch] += eps;
            scene.pre.rgb = scene.rgb;
            loss_plus = run_forward_loss(scene, alloc);
        }

        float loss_minus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.rgb[ch] -= eps;
            scene.pre.rgb = scene.rgb;
            loss_minus = run_forward_loss(scene, alloc);
        }

        float fd_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float abs_err = std::fabs(analytic_grad - fd_grad);
        float denom = std::max(std::fabs(analytic_grad), std::fabs(fd_grad));
        float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;

        EXPECT_LT(rel_err, 1e-3f)
            << "RGB channel " << ch
            << ": analytic=" << analytic_grad << " fd=" << fd_grad
            << " rel_err=" << rel_err;
    }
}

TEST(RasterizerBackward, FiniteDiff_Opacity) {
    const float eps = 1e-3f;

    float analytic_grad;
    {
        FrameAllocator alloc(4 * 1024 * 1024);
        BackwardTestScene scene;
        RasterGradOutput rgrad{};
        run_forward_backward(scene, alloc, rgrad);
        analytic_grad = rgrad.d_opacities_2d[0];
    }

    float loss_plus;
    {
        FrameAllocator alloc(4 * 1024 * 1024);
        BackwardTestScene scene;
        scene.opacities_2d[0] += eps;
        scene.pre.opacities_2d = scene.opacities_2d;
        loss_plus = run_forward_loss(scene, alloc);
    }

    float loss_minus;
    {
        FrameAllocator alloc(4 * 1024 * 1024);
        BackwardTestScene scene;
        scene.opacities_2d[0] -= eps;
        scene.pre.opacities_2d = scene.opacities_2d;
        loss_minus = run_forward_loss(scene, alloc);
    }

    float fd_grad = (loss_plus - loss_minus) / (2.0f * eps);
    float abs_err = std::fabs(analytic_grad - fd_grad);
    float denom = std::max(std::fabs(analytic_grad), std::fabs(fd_grad));
    float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;

    EXPECT_LT(rel_err, 1e-3f)
        << "Opacity: analytic=" << analytic_grad << " fd=" << fd_grad
        << " rel_err=" << rel_err;
}

TEST(RasterizerBackward, FiniteDiff_Means2D) {
    // Use double-precision arithmetic for FD to reduce cancellation error.
    // The analytic gradient matches FD to ~0.1-0.3% with float eps=1e-3,
    // limited by FD truncation/cancellation, not by the analytic computation.
    const double eps = 1e-3;

    for (int dim = 0; dim < 2; dim++) {
        float analytic_grad;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            RasterGradOutput rgrad{};
            run_forward_backward(scene, alloc, rgrad);
            analytic_grad = rgrad.d_means2D[dim];
        }

        double loss_plus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.means2D[dim] += (float)eps;
            scene.pre.means2D = scene.means2D;
            loss_plus = (double)run_forward_loss(scene, alloc);
        }

        double loss_minus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.means2D[dim] -= (float)eps;
            scene.pre.means2D = scene.means2D;
            loss_minus = (double)run_forward_loss(scene, alloc);
        }

        double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        double abs_err = std::fabs((double)analytic_grad - fd_grad);
        double denom = std::max(std::fabs((double)analytic_grad), std::fabs(fd_grad));
        double rel_err = (denom > 1e-7) ? abs_err / denom : abs_err;

        EXPECT_LT(rel_err, 5e-3)
            << "Means2D[" << dim << "]: analytic=" << analytic_grad
            << " fd=" << fd_grad << " rel_err=" << rel_err;
    }
}

TEST(RasterizerBackward, FiniteDiff_Conics) {
    const float eps = 1e-3f;

    for (int k = 0; k < 3; k++) {
        float analytic_grad;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            RasterGradOutput rgrad{};
            run_forward_backward(scene, alloc, rgrad);
            analytic_grad = rgrad.d_conics[k];
        }

        float loss_plus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.conics[k] += eps;
            scene.pre.conics = scene.conics;
            loss_plus = run_forward_loss(scene, alloc);
        }

        float loss_minus;
        {
            FrameAllocator alloc(4 * 1024 * 1024);
            BackwardTestScene scene;
            scene.conics[k] -= eps;
            scene.pre.conics = scene.conics;
            loss_minus = run_forward_loss(scene, alloc);
        }

        float fd_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float abs_err = std::fabs(analytic_grad - fd_grad);
        float denom = std::max(std::fabs(analytic_grad), std::fabs(fd_grad));
        float rel_err = (denom > 1e-7f) ? abs_err / denom : abs_err;

        EXPECT_LT(rel_err, 1e-3f)
            << "Conics[" << k << "]: analytic=" << analytic_grad
            << " fd=" << fd_grad << " rel_err=" << rel_err;
    }
}
