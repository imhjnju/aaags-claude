#include <gtest/gtest.h>
#include "trainer.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>

// Helper: create a camera looking down +Z with identity view matrix
// (reused from test_forward_cache.cpp)
static Camera makeTrainerTestCamera(int w, int h, float fov_deg = 60.0f) {
    Camera cam{};
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    memcpy(cam.view_matrix, identity, sizeof(identity));
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)h / (float)w;
    cam.width = w; cam.height = h;
    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0;
    // OpenGL-style perspective: col-major
    float n = 0.01f, f = 100.0f;
    float r = cam.tan_fovx * n, t = cam.tan_fovy * n;
    memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0]  = n / r;
    cam.viewproj_matrix[5]  = n / t;
    cam.viewproj_matrix[10] = f / (f - n);
    cam.viewproj_matrix[11] = 1.0f;
    cam.viewproj_matrix[14] = -(f * n) / (f - n);
    return cam;
}

TEST(TrainerE2E, LossDecreases_SingleGaussian) {
    // Strategy: render a target image from a Gaussian with known SH,
    // then train from different SH starting point. Since the Gaussian shape
    // is fixed, training only SH coefficients should converge exactly.
    const int W = 16, H = 16;

    Camera cam = makeTrainerTestCamera(W, H);

    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = 0;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // First: render a ground truth image from a known Gaussian
    // SH_C0 ~ 0.282, so sh=2.0 -> rgb ~ 0.564
    {
        GaussianData g;
        g.count = 1; g.sh_degree = 0; g.max_coeffs = 1;
        float gp[3] = {0,0,5}, gs_[3] = {0.3f,0.3f,0.3f}, gr[4] = {1,0,0,0};
        float gsh[3] = {2.0f, 2.0f, 2.0f}, gop[1] = {0.9f};
        g.positions = gp; g.scales = gs_; g.rotations = gr;
        g.sh_coeffs = gsh; g.opacities = gop; g.filter_3D = nullptr;

        FrameAllocator tmp_alloc(64 * 1024 * 1024);
        PreprocessorCPU pp; TileBinnerCPU bn; SorterCPU sr; RasterizerCPU rs;
        auto pre = pp.process(g, cam, cfg, tmp_alloc);
        auto bin = bn.bin(pre, 1, cam, cfg, tmp_alloc);
        if (bin.total_pairs > 0) sr.sort(bin, tmp_alloc);
        // Verify Gaussian is visible
        ASSERT_GT(pre.radii[0], 0) << "Target Gaussian was culled";

        std::vector<float> gt(W*H*3, 0.0f);
        rs.rasterize(pre, bin, cam, cfg, gt.data());

        // Verify target has non-zero content
        float gt_sum = 0;
        for (auto v : gt) gt_sum += v;
        ASSERT_GT(gt_sum, 0.0f) << "Target image is all black";
        printf("  Target image sum = %.4f\n", gt_sum);

        // Now train: same Gaussian shape but start with wrong SH
        RawGaussianParams raw;
        raw.count = 1; raw.sh_degree = 0; raw.max_coeffs = 1;
        float pos[3] = {0.0f, 0.0f, 5.0f};
        float sc[3]  = {logf(0.3f), logf(0.3f), logf(0.3f)};
        float rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float sh[3]  = {0.5f, 0.5f, 0.5f};   // wrong SH (target is 2.0)
        // inverse-sigmoid(0.9) ~ 2.197
        float op[1]  = {2.197f};
        raw.raw_positions = pos;
        raw.raw_scales = sc;
        raw.raw_rotations = rot;
        raw.raw_sh_coeffs = sh;
        raw.raw_opacities = op;

        TrainConfig tcfg;
        tcfg.lr_position_init  = 1e-6f;   // near-frozen (avoids log(0) NaN)
        tcfg.lr_position_final = 1e-6f;
        tcfg.lr_feature        = 10.0f;   // learn SH color
        tcfg.lr_opacity        = 1e-4f;   // near-frozen
        tcfg.lr_scaling        = 1e-4f;   // near-frozen
        tcfg.lr_rotation       = 1e-4f;   // near-frozen
        tcfg.max_steps         = 500;

        Trainer trainer(256 * 1024 * 1024);

        float first_loss = -1, last_loss = -1;
        for (int iter = 0; iter < 500; iter++) {
            auto result = trainer.step(raw, cam, gt.data(), cfg, tcfg, iter);
            if (iter == 0) first_loss = result.loss;
            last_loss = result.loss;
            if (iter < 5 || iter % 100 == 0 || iter == 499) {
                printf("  iter %3d: loss = %.6f  sh=[%.4f,%.4f,%.4f]\n",
                       iter, result.loss, sh[0], sh[1], sh[2]);
            }
        }

        printf("  first_loss=%.6f  last_loss=%.6f  ratio=%.3f\n",
               first_loss, last_loss, last_loss / first_loss);

        EXPECT_LT(last_loss, first_loss * 0.5f)
            << "Loss should decrease by at least 50% after 500 iterations";
    }
}
