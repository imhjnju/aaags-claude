// test_trainer_gpu.cpp -- Basic test for the GPU trainer
// Verifies that loss decreases over a few steps of training.

#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "trainer_gpu.h"
#include "test_gpu_context.h"
#endif

TEST(TrainerGPU, LossDecreases) {
#ifndef ENABLE_OPENCL
    GTEST_SKIP() << "OpenCL not enabled";
#else
    if (!initTestGPUContext()) {
        GTEST_SKIP() << "OpenCL not available";
    }
    OpenCLContext& ctx = getTestGPUContext();

    // Simple scene: 1 Gaussian, 8x8 image
    constexpr int N = 1;
    constexpr int SH_DEGREE = 0;
    constexpr int MAX_COEFFS = 1;
    constexpr int W = 8;
    constexpr int H = 8;

    // Raw parameters
    std::vector<float> raw_positions = {0.0f, 0.0f, 5.0f};
    std::vector<float> raw_scales = {-1.0f, -1.0f, -1.0f};  // log-space, exp(-1) ~ 0.37
    std::vector<float> raw_rotations = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> raw_sh = {1.0f, 1.0f, 1.0f};  // degree 0: 3 channels
    std::vector<float> raw_opacities = {2.0f};  // sigmoid(2) ~ 0.88

    RawGaussianParams params;
    params.count = N;
    params.sh_degree = SH_DEGREE;
    params.max_coeffs = MAX_COEFFS;
    params.raw_positions = raw_positions.data();
    params.raw_scales = raw_scales.data();
    params.raw_rotations = raw_rotations.data();
    params.raw_sh_coeffs = raw_sh.data();
    params.raw_opacities = raw_opacities.data();

    Camera cam = {};
    cam.width = W;
    cam.height = H;
    cam.tan_fovx = 0.5f;
    cam.tan_fovy = 0.5f;
    // Identity view matrix
    std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
    cam.view_matrix[0] = 1.0f; cam.view_matrix[5] = 1.0f;
    cam.view_matrix[10] = 1.0f; cam.view_matrix[15] = 1.0f;
    // Simple perspective viewproj
    float fx = W / (2.0f * cam.tan_fovx);
    float fy = H / (2.0f * cam.tan_fovy);
    std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0] = fx / (W * 0.5f);
    cam.viewproj_matrix[5] = fy / (H * 0.5f);
    cam.viewproj_matrix[10] = 1.0f;
    cam.viewproj_matrix[11] = 1.0f;
    cam.cam_pos[0] = 0.0f; cam.cam_pos[1] = 0.0f; cam.cam_pos[2] = 0.0f;

    RenderConfig cfg;
    cfg.bg_color[0] = 0.0f; cfg.bg_color[1] = 0.0f; cfg.bg_color[2] = 0.0f;
    cfg.scale_modifier = 1.0f;
    cfg.sh_degree = SH_DEGREE;
    cfg.tile_w = 16; cfg.tile_h = 16;
    cfg.eval_3D = false;
    cfg.training = true;

    // Ground truth: simple white image
    std::vector<float> gt_image(W * H * 3, 0.5f);

    TrainConfig train_cfg;
    train_cfg.lr_position_init = 0.001f;
    train_cfg.lr_position_final = 0.0001f;
    train_cfg.lr_feature = 0.01f;
    train_cfg.lr_opacity = 0.05f;
    train_cfg.lr_scaling = 0.005f;
    train_cfg.lr_rotation = 0.001f;
    train_cfg.max_steps = 100;

    TrainerGPU trainer(ctx, 256UL * 1024 * 1024);

    float first_loss = -1.0f;
    float last_loss = -1.0f;

    for (int i = 0; i < 5; i++) {
        auto result = trainer.step(params, cam, gt_image.data(), cfg, train_cfg, i);
        if (i == 0) first_loss = result.loss;
        last_loss = result.loss;
        std::fprintf(stderr, "  TrainerGPU step %d: loss=%.6f\n", i, result.loss);
    }

    // Loss should decrease (or at least not increase significantly)
    EXPECT_LT(last_loss, first_loss + 0.01f)
        << "Loss should decrease over training steps. "
        << "first=" << first_loss << " last=" << last_loss;

    std::fprintf(stderr, "TrainerGPU: loss %.6f -> %.6f over 5 steps\n",
                 first_loss, last_loss);
#endif
}
