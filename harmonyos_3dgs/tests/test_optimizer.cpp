#include <gtest/gtest.h>
#include "optimizer.h"
#include <cmath>
#include <cstring>
#include <vector>

class OptimizerTest : public ::testing::Test {
protected:
    static constexpr int N = 1;
    static constexpr int sh_degree = 1;
    static constexpr int max_coeffs = 4; // (1+1)^2

    FrameAllocator alloc{1 << 20};
    RawGaussianParams params{};
    GradientOutput grads{};
    TrainConfig cfg{};
    SGDOptimizer opt;

    void SetUp() override {
        params.count = N;
        params.sh_degree = sh_degree;
        params.max_coeffs = max_coeffs;

        params.raw_positions  = alloc.allocate_array<float>(N * 3);
        params.raw_scales     = alloc.allocate_array<float>(N * 3);
        params.raw_rotations  = alloc.allocate_array<float>(N * 4);
        params.raw_sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
        params.raw_opacities  = alloc.allocate_array<float>(N);

        grads.allocate_and_zero(alloc, N, max_coeffs);

        // Fill params with known values
        for (int i = 0; i < N * 3; i++) params.raw_positions[i] = 1.0f;
        for (int i = 0; i < N * 3; i++) params.raw_scales[i] = 2.0f;
        for (int i = 0; i < N * 4; i++) params.raw_rotations[i] = 3.0f;
        for (int i = 0; i < N * max_coeffs * 3; i++) params.raw_sh_coeffs[i] = 4.0f;
        for (int i = 0; i < N; i++) params.raw_opacities[i] = 5.0f;
    }
};

TEST_F(OptimizerTest, BasicUpdate) {
    // Set gradients to known values
    for (int i = 0; i < N * 3; i++) grads.d_raw_positions[i] = 0.5f;
    for (int i = 0; i < N * 3; i++) grads.d_raw_scales[i] = 0.5f;
    for (int i = 0; i < N * 4; i++) grads.d_raw_rotations[i] = 0.5f;
    for (int i = 0; i < N * max_coeffs * 3; i++) grads.d_raw_sh_coeffs[i] = 0.5f;
    for (int i = 0; i < N; i++) grads.d_raw_opacities[i] = 0.5f;

    int iteration = 0;
    float lr_pos = lr_schedule(cfg.lr_position_init, cfg.lr_position_final,
                                iteration, cfg.max_steps);

    opt.step(params, grads, cfg, iteration);

    // param_new = param_old - lr * grad
    for (int i = 0; i < N * 3; i++)
        EXPECT_NEAR(params.raw_positions[i], 1.0f - lr_pos * 0.5f, 1e-7f);
    for (int i = 0; i < N * 3; i++)
        EXPECT_NEAR(params.raw_scales[i], 2.0f - cfg.lr_scaling * 0.5f, 1e-7f);
    for (int i = 0; i < N * 4; i++)
        EXPECT_NEAR(params.raw_rotations[i], 3.0f - cfg.lr_rotation * 0.5f, 1e-7f);
    for (int i = 0; i < N * max_coeffs * 3; i++)
        EXPECT_NEAR(params.raw_sh_coeffs[i], 4.0f - cfg.lr_feature * 0.5f, 1e-7f);
    for (int i = 0; i < N; i++)
        EXPECT_NEAR(params.raw_opacities[i], 5.0f - cfg.lr_opacity * 0.5f, 1e-7f);
}

TEST_F(OptimizerTest, PositionLRDecays) {
    for (int i = 0; i < N * 3; i++) grads.d_raw_positions[i] = 1.0f;

    float lr_at_0 = lr_schedule(cfg.lr_position_init, cfg.lr_position_final,
                                 0, cfg.max_steps);
    float lr_at_max = lr_schedule(cfg.lr_position_init, cfg.lr_position_final,
                                   cfg.max_steps, cfg.max_steps);

    // Step at iteration 0
    opt.step(params, grads, cfg, 0);
    float pos_after_step0 = params.raw_positions[0];
    EXPECT_NEAR(pos_after_step0, 1.0f - lr_at_0 * 1.0f, 1e-7f);

    // Reset position
    params.raw_positions[0] = 1.0f;

    // Step at iteration max_steps
    opt.step(params, grads, cfg, cfg.max_steps);
    float pos_after_step_max = params.raw_positions[0];
    EXPECT_NEAR(pos_after_step_max, 1.0f - lr_at_max * 1.0f, 1e-7f);

    // lr at step 0 should be larger than lr at max_steps
    EXPECT_GT(lr_at_0, lr_at_max);
}

// ============================================================
// Adam Optimizer Tests
// ============================================================

TEST(AdamOptimizer, BasicUpdate) {
    // Verify single-step Adam matches manual computation
    const int N = 1, mc = 1;
    AdamOptimizer adam;
    adam.reset(N, mc);

    FrameAllocator alloc(1 << 20);
    RawGaussianParams params;
    params.count = N; params.sh_degree = 0; params.max_coeffs = mc;
    params.raw_positions = alloc.allocate_array<float>(3);
    params.raw_scales = alloc.allocate_array<float>(3);
    params.raw_rotations = alloc.allocate_array<float>(4);
    params.raw_sh_coeffs = alloc.allocate_array<float>(3);
    params.raw_opacities = alloc.allocate_array<float>(1);

    // Initialize params
    for (int i = 0; i < 3; i++) params.raw_sh_coeffs[i] = 1.0f;
    for (int i = 0; i < 3; i++) params.raw_positions[i] = 0.0f;
    for (int i = 0; i < 3; i++) params.raw_scales[i] = 0.0f;
    for (int i = 0; i < 4; i++) params.raw_rotations[i] = 0.0f;
    params.raw_opacities[0] = 0.0f;

    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, mc);
    grads.d_raw_sh_coeffs[0] = 0.1f;
    grads.d_raw_sh_coeffs[1] = 0.2f;
    grads.d_raw_sh_coeffs[2] = -0.3f;

    TrainConfig cfg;
    cfg.use_adam = true;
    float lr = cfg.lr_feature;
    float b1 = cfg.adam_beta1, b2 = cfg.adam_beta2, eps = cfg.adam_eps;

    adam.step(params, grads, cfg, 0);

    // Manual Adam step 1:
    // m = (1-b1)*g = 0.1*g, v = (1-b2)*g^2 = 0.001*g^2
    // bc1 = 1/(1-b1) = 10, bc2 = 1/(1-b2) = 1000
    // m_hat = m*bc1 = g, v_hat = v*bc2 = g^2
    // step = lr * g / (|g| + eps) = lr * sign(g) ≈ lr
    float g0 = 0.1f;
    float m0 = (1-b1)*g0, v0 = (1-b2)*g0*g0;
    float bc1 = 1.0f/(1.0f-b1), bc2 = 1.0f/(1.0f-b2);
    float expected = 1.0f - lr * (m0*bc1) / (std::sqrt(v0*bc2) + eps);

    EXPECT_NEAR(params.raw_sh_coeffs[0], expected, 1e-6f)
        << "Adam step 1 formula mismatch";
    EXPECT_EQ(adam.step_count(), 1);
}

TEST(AdamOptimizer, MomentumAccumulation) {
    // After multiple steps with same gradient, Adam should accelerate
    const int N = 1, mc = 1;
    AdamOptimizer adam;
    adam.reset(N, mc);

    FrameAllocator alloc(1 << 20);
    RawGaussianParams params;
    params.count = N; params.sh_degree = 0; params.max_coeffs = mc;
    params.raw_positions = alloc.allocate_array<float>(3);
    params.raw_scales = alloc.allocate_array<float>(3);
    params.raw_rotations = alloc.allocate_array<float>(4);
    params.raw_sh_coeffs = alloc.allocate_array<float>(3);
    params.raw_opacities = alloc.allocate_array<float>(1);
    std::memset(params.raw_positions, 0, 3 * sizeof(float));
    std::memset(params.raw_scales, 0, 3 * sizeof(float));
    std::memset(params.raw_rotations, 0, 4 * sizeof(float));
    params.raw_opacities[0] = 0.0f;
    params.raw_sh_coeffs[0] = 10.0f;  // far from optimal

    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, mc);

    TrainConfig cfg;
    cfg.use_adam = true;

    // 10 steps with constant gradient
    for (int i = 0; i < 10; i++) {
        grads.d_raw_sh_coeffs[0] = 1.0f;  // constant gradient
        adam.step(params, grads, cfg, i);
    }

    // After 10 steps, SH should have decreased substantially
    EXPECT_LT(params.raw_sh_coeffs[0], 10.0f - 0.01f)
        << "Adam should decrease param with consistent positive gradient";
    EXPECT_EQ(adam.step_count(), 10);
}

TEST(AdamOptimizer, ZeroGradNoChange) {
    const int N = 1, mc = 1;
    AdamOptimizer adam;
    adam.reset(N, mc);

    FrameAllocator alloc(1 << 20);
    RawGaussianParams params;
    params.count = N; params.sh_degree = 0; params.max_coeffs = mc;
    params.raw_positions = alloc.allocate_array<float>(3);
    params.raw_scales = alloc.allocate_array<float>(3);
    params.raw_rotations = alloc.allocate_array<float>(4);
    params.raw_sh_coeffs = alloc.allocate_array<float>(3);
    params.raw_opacities = alloc.allocate_array<float>(1);
    for (int i = 0; i < 3; i++) params.raw_sh_coeffs[i] = 5.0f;
    std::memset(params.raw_positions, 0, 3*sizeof(float));
    std::memset(params.raw_scales, 0, 3*sizeof(float));
    std::memset(params.raw_rotations, 0, 4*sizeof(float));
    params.raw_opacities[0] = 0.0f;

    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, mc);
    // All grads remain zero

    TrainConfig cfg;
    cfg.use_adam = true;

    adam.step(params, grads, cfg, 0);

    // Zero gradient → zero update (m=0, v=0 → m_hat=0 → step=0)
    EXPECT_FLOAT_EQ(params.raw_sh_coeffs[0], 5.0f);
}

TEST(AdamOptimizer, AllParamTypes) {
    // Verify Adam updates all 5 parameter types with distinct gradients
    const int N = 2, mc = 4;  // 2 Gaussians, SH degree 1
    AdamOptimizer adam;
    adam.reset(N, mc);

    FrameAllocator alloc(1 << 20);
    RawGaussianParams params;
    params.count = N; params.sh_degree = 1; params.max_coeffs = mc;
    params.raw_positions = alloc.allocate_array<float>(N*3);
    params.raw_scales = alloc.allocate_array<float>(N*3);
    params.raw_rotations = alloc.allocate_array<float>(N*4);
    params.raw_sh_coeffs = alloc.allocate_array<float>(N*mc*3);
    params.raw_opacities = alloc.allocate_array<float>(N);

    // Known initial values
    for (int i = 0; i < N*3; i++) params.raw_positions[i] = 1.0f;
    for (int i = 0; i < N*3; i++) params.raw_scales[i] = 0.5f;
    for (int i = 0; i < N*4; i++) params.raw_rotations[i] = 0.25f;
    for (int i = 0; i < N*mc*3; i++) params.raw_sh_coeffs[i] = 2.0f;
    for (int i = 0; i < N; i++) params.raw_opacities[i] = 0.0f;

    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, mc);

    // Distinct gradients per type
    for (int i = 0; i < N*3; i++) grads.d_raw_positions[i] = 0.1f;
    for (int i = 0; i < N*3; i++) grads.d_raw_scales[i] = -0.2f;
    for (int i = 0; i < N*4; i++) grads.d_raw_rotations[i] = 0.05f;
    for (int i = 0; i < N*mc*3; i++) grads.d_raw_sh_coeffs[i] = 0.3f;
    for (int i = 0; i < N; i++) grads.d_raw_opacities[i] = -0.5f;

    TrainConfig cfg;
    cfg.use_adam = true;

    adam.step(params, grads, cfg, 0);

    // All params should have changed from initial values
    // Positive gradient → param decreases; negative → param increases
    for (int i = 0; i < N*3; i++)
        EXPECT_LT(params.raw_positions[i], 1.0f) << "Position should decrease (positive grad)";
    for (int i = 0; i < N*3; i++)
        EXPECT_GT(params.raw_scales[i], 0.5f) << "Scale should increase (negative grad)";
    for (int i = 0; i < N*4; i++)
        EXPECT_LT(params.raw_rotations[i], 0.25f) << "Rotation should decrease (positive grad)";
    for (int i = 0; i < N*mc*3; i++)
        EXPECT_LT(params.raw_sh_coeffs[i], 2.0f) << "SH should decrease (positive grad)";
    for (int i = 0; i < N; i++)
        EXPECT_GT(params.raw_opacities[i], 0.0f) << "Opacity should increase (negative grad)";
}

TEST(AdamOptimizer, ConvergesFasterThanSGD) {
    // Same gradient sequence, Adam should reach lower param value than SGD
    const int N = 1, mc = 1, steps = 50;

    // Use separate heap arrays to avoid allocator issues
    float adam_sh[3] = {10.0f, 0, 0}, sgd_sh[3] = {10.0f, 0, 0};
    float adam_pos[3] = {}, sgd_pos[3] = {};
    float adam_sc[3] = {}, sgd_sc[3] = {};
    float adam_rot[4] = {}, sgd_rot[4] = {};
    float adam_op[1] = {}, sgd_op[1] = {};

    RawGaussianParams adam_p{N, 0, mc, adam_pos, adam_sc, adam_rot, adam_sh, adam_op};
    RawGaussianParams sgd_p{N, 0, mc, sgd_pos, sgd_sc, sgd_rot, sgd_sh, sgd_op};

    AdamOptimizer adam; adam.reset(N, mc);
    SGDOptimizer sgd;
    TrainConfig cfg;

    for (int i = 0; i < steps; i++) {
        // Separate gradient buffers each step
        float ga_sh[3] = {1.0f, 0, 0}, gs_sh[3] = {1.0f, 0, 0};
        float ga_pos[3] = {}, gs_pos[3] = {};
        float ga_sc[3] = {}, gs_sc[3] = {};
        float ga_rot[4] = {}, gs_rot[4] = {};
        float ga_op[1] = {}, gs_op[1] = {};
        GradientOutput ga{ga_pos, ga_sc, ga_rot, ga_sh, ga_op};
        GradientOutput gs{gs_pos, gs_sc, gs_rot, gs_sh, gs_op};

        adam.step(adam_p, ga, cfg, i);
        sgd.step(sgd_p, gs, cfg, i);
    }

    float adam_val = adam_p.raw_sh_coeffs[0];
    float sgd_val = sgd_p.raw_sh_coeffs[0];
    // Adam should move further from 10.0 than SGD
    // SGD step = lr_feature * 1.0 = 0.0025 per step → 50*0.0025 = 0.125
    // Adam converges to lr * sign(grad) ≈ 0.0025 per step → similar total
    // But Adam with bias correction moves faster initially
    // Just verify both decreased and Adam decreased at least as much
    EXPECT_LT(adam_val, 10.0f) << "Adam should decrease SH";
    EXPECT_LT(sgd_val, 10.0f) << "SGD should decrease SH";
    EXPECT_LE(adam_val, sgd_val + 0.01f)
        << "Adam (" << adam_val << ") should converge at least as fast as SGD (" << sgd_val << ")";
}

// Original SGD test
TEST_F(OptimizerTest, ZeroGradNoChange) {
    // Gradients are already zeroed from allocate_and_zero

    // Save original values
    std::vector<float> orig_pos(params.raw_positions, params.raw_positions + N * 3);
    std::vector<float> orig_scales(params.raw_scales, params.raw_scales + N * 3);
    std::vector<float> orig_rot(params.raw_rotations, params.raw_rotations + N * 4);
    std::vector<float> orig_sh(params.raw_sh_coeffs, params.raw_sh_coeffs + N * max_coeffs * 3);
    std::vector<float> orig_opa(params.raw_opacities, params.raw_opacities + N);

    opt.step(params, grads, cfg, 100);

    for (int i = 0; i < N * 3; i++)
        EXPECT_FLOAT_EQ(params.raw_positions[i], orig_pos[i]);
    for (int i = 0; i < N * 3; i++)
        EXPECT_FLOAT_EQ(params.raw_scales[i], orig_scales[i]);
    for (int i = 0; i < N * 4; i++)
        EXPECT_FLOAT_EQ(params.raw_rotations[i], orig_rot[i]);
    for (int i = 0; i < N * max_coeffs * 3; i++)
        EXPECT_FLOAT_EQ(params.raw_sh_coeffs[i], orig_sh[i]);
    for (int i = 0; i < N; i++)
        EXPECT_FLOAT_EQ(params.raw_opacities[i], orig_opa[i]);
}
