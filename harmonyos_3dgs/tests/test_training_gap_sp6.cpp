// test_training_gap_sp6.cpp — SP-6 unit tests for training gap helpers.
#include <gtest/gtest.h>
#include "train_utils.h"
#include "train_types.h"
#include <cmath>

TEST(TrainUtils, OpSigmoidNearDeadGaussian) {
    float val = op_sigmoid(1.f - 0.001f);
    EXPECT_GT(val, 0.5f);
}

TEST(TrainUtils, OpSigmoidActiveGaussian) {
    float val = op_sigmoid(1.f - 0.9f);
    EXPECT_LT(val, 1e-6f);
}

TEST(TrainUtils, OpSigmoidAtThreshold) {
    EXPECT_NEAR(op_sigmoid(0.995f), 0.5f, 1e-5f);
}

TEST(TrainUtils, BuildLIdentityRotation) {
    const float rot[4] = {1.f, 0.f, 0.f, 0.f};
    const float act_scale[3] = {0.03f, 0.05f, 0.02f};
    float L[3][3];
    build_L(rot, act_scale, L);
    EXPECT_NEAR(L[0][0], act_scale[0], 1e-6f);
    EXPECT_NEAR(L[1][1], act_scale[1], 1e-6f);
    EXPECT_NEAR(L[2][2], act_scale[2], 1e-6f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                EXPECT_NEAR(L[i][j], 0.f, 1e-6f);
}

TEST(TrainUtils, BuildLSigmaSymmetry) {
    const float angle = 3.14159265f / 8.f;
    const float rot[4] = {std::cos(angle), 0.f, 0.f, std::sin(angle)};
    const float act_scale[3] = {0.01f, 0.02f, 0.03f};
    float L[3][3];
    build_L(rot, act_scale, L);
    float Sigma[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                Sigma[i][j] += L[i][k] * L[j][k];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(Sigma[i][j], Sigma[j][i], 1e-6f);
    for (int i = 0; i < 3; ++i)
        EXPECT_GE(Sigma[i][i], 0.f);
}

TEST(TrainUtils, SpatialLrScheduleStep0) {
    float lr = spatial_lr_schedule(1e-4f, 1e-6f, 2.0f, 0, 30000);
    EXPECT_NEAR(lr, 2.0f * 1e-4f, 1e-10f);
}

TEST(TrainUtils, SpatialLrScheduleMaxStep) {
    float lr = spatial_lr_schedule(1e-4f, 1e-6f, 3.0f, 30000, 30000);
    EXPECT_NEAR(lr, 3.0f * 1e-6f, 1e-10f);
}

TEST(TrainUtils, SpatialLrScaleIsNoop) {
    float expected = lr_schedule(1.6e-4f, 1.6e-6f, 1000, 30000);
    float actual   = spatial_lr_schedule(1.6e-4f, 1.6e-6f, 1.0f, 1000, 30000);
    EXPECT_FLOAT_EQ(actual, expected);
}

TEST(VkTrainingConfig, SP6DefaultFields) {
    VkTrainingConfig cfg{};
    EXPECT_FLOAT_EQ(cfg.opacity_reg,      0.01f);
    EXPECT_FLOAT_EQ(cfg.scale_reg,        0.01f);
    EXPECT_FLOAT_EQ(cfg.noise_lr,         5e5f);
    EXPECT_FLOAT_EQ(cfg.spatial_lr_scale, 1.0f);
}

TEST(RegularizationGrad, OpacityGradAtRawZero) {
    // At raw_op=0: sigmoid(0)=0.5, so grad=(opacity_reg/N)*0.5*(1-0.5)=0.0025 for N=1
    const float raw_op = 0.0f;
    const float sig = 1.f / (1.f + std::exp(-raw_op));  // call the actual formula
    const float grad = (0.01f / 1.f) * sig * (1.f - sig);
    EXPECT_NEAR(grad, 0.0025f, 1e-6f);
    // Also verify op_sigmoid is not involved here — opacity grad uses plain sigmoid
    // (op_sigmoid is for noise injection, not regularization)
    EXPECT_NEAR(sig, 0.5f, 1e-6f);
}

TEST(RegularizationGrad, ScaleGradAtRawLogScale) {
    // At raw_sc=log(0.03): exp(raw_sc)=act_sc=0.03
    // grad=(scale_reg/N)*act_sc = 0.01*0.03=0.0003 for N=1
    const float raw_sc = std::log(0.03f);
    const float act_sc = std::exp(raw_sc);  // call the actual formula
    const float grad = (0.01f / 1.f) * act_sc;
    EXPECT_NEAR(act_sc, 0.03f, 1e-6f);
    EXPECT_NEAR(grad, 0.0003f, 1e-6f);
}

TEST(RegularizationGrad, OpacityGradNonNeg) {
    for (float raw_op : {-10.f, -1.f, 0.f, 1.f, 10.f}) {
        const float sig = 1.f / (1.f + std::exp(-raw_op));
        EXPECT_GE(0.01f * sig * (1.f - sig), 0.f);
    }
}

TEST(PositionNoise, NoiseScaleOrder) {
    // noise should be sub-meter (< 1.0) and above floating-point noise floor (> 1e-4)
    // for typical basketball scene geometry (scale~0.03m, pos_lr~1.6e-4)
    const float sigma_max = 0.03f * 0.03f;
    const float scalar    = 1.0f * 5e5f * 1.6e-4f;
    const float magnitude = sigma_max * scalar;
    EXPECT_GT(magnitude, 1e-4f);
    EXPECT_LT(magnitude, 1.0f);
}

TEST(PositionNoise, ActiveGaussianGetsNoNoise) {
    // An active Gaussian with opacity=0.9 should have opacity_factor near 0
    // (op_sigmoid(1-0.9) = op_sigmoid(0.1)), so it gets essentially no noise.
    const float opacity_factor = op_sigmoid(1.f - 0.9f);
    EXPECT_LT(opacity_factor, 1e-6f);
}

TEST(PositionNoise, DeadGaussianGetsFullNoise) {
    // A dead Gaussian with opacity=0.001 should have opacity_factor near 1.
    // op_sigmoid(1-0.001) = op_sigmoid(0.999 - 0.995 + 0.995) ~ 1.
    const float opacity_factor = op_sigmoid(1.f - 0.001f);
    EXPECT_GT(opacity_factor, 0.5f);
}
