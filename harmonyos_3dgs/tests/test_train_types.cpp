#include <gtest/gtest.h>
#include "types.h"
#include "train_types.h"
#include <cmath>
#include <cstring>

// --- ForwardCache exists ---
TEST(ForwardCache, LayoutMatchesSpec) {
    ForwardCache fc{};
    EXPECT_EQ(fc.T_final,   nullptr);
    EXPECT_EQ(fc.n_contrib, nullptr);
    EXPECT_EQ(fc.cov2D,     nullptr);
    EXPECT_EQ(fc.cov2D_det, nullptr);
    EXPECT_EQ(fc.cov3D,     nullptr);
    EXPECT_EQ(fc.p_view,    nullptr);
    EXPECT_EQ(fc.p_hom_w,   nullptr);
    EXPECT_EQ(fc.pre,       nullptr);
    EXPECT_EQ(fc.bin,       nullptr);
}

// --- RawGaussianParams::activate ---
class ActivateTest : public ::testing::Test {
protected:
    static constexpr int N = 4;
    static constexpr int sh_degree = 1;
    static constexpr int max_coeffs = 4; // (1+1)^2

    FrameAllocator alloc{1 << 20};
    GaussianData g{};
    RawGaussianParams raw{};

    void SetUp() override {
        raw.count = N;
        raw.sh_degree = sh_degree;
        raw.max_coeffs = max_coeffs;

        raw.raw_positions  = alloc.allocate_array<float>(N * 3);
        raw.raw_scales     = alloc.allocate_array<float>(N * 3);
        raw.raw_rotations  = alloc.allocate_array<float>(N * 4);
        raw.raw_sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
        raw.raw_opacities  = alloc.allocate_array<float>(N);

        g.count = N;
        g.sh_degree = sh_degree;
        g.max_coeffs = max_coeffs;
        g.positions  = alloc.allocate_array<float>(N * 3);
        g.scales     = alloc.allocate_array<float>(N * 3);
        g.rotations  = alloc.allocate_array<float>(N * 4);
        g.sh_coeffs  = alloc.allocate_array<float>(N * max_coeffs * 3);
        g.opacities  = alloc.allocate_array<float>(N);
        g.filter_3D  = nullptr;

        // Fill raw data with known values
        for (int i = 0; i < N * 3; i++) raw.raw_positions[i] = float(i) * 0.1f;
        for (int i = 0; i < N * 3; i++) raw.raw_scales[i] = 1.0f;  // exp(1) ≈ 2.718
        for (int i = 0; i < N * max_coeffs * 3; i++) raw.raw_sh_coeffs[i] = float(i) * 0.01f;
        for (int i = 0; i < N; i++) raw.raw_opacities[i] = 0.0f;   // sigmoid(0) = 0.5

        // Set rotations to (3, 0, 0, 0) -> normalized = (1, 0, 0, 0)
        for (int i = 0; i < N; i++) {
            raw.raw_rotations[i * 4 + 0] = 3.0f;
            raw.raw_rotations[i * 4 + 1] = 0.0f;
            raw.raw_rotations[i * 4 + 2] = 0.0f;
            raw.raw_rotations[i * 4 + 3] = 0.0f;
        }
    }
};

TEST_F(ActivateTest, PositionsCopiedIdentity) {
    raw.activate(g);
    for (int i = 0; i < N * 3; i++) {
        EXPECT_FLOAT_EQ(g.positions[i], raw.raw_positions[i]);
    }
}

TEST_F(ActivateTest, ScalesAreExp) {
    raw.activate(g);
    for (int i = 0; i < N * 3; i++) {
        EXPECT_NEAR(g.scales[i], std::exp(1.0f), 1e-5f);
    }
}

TEST_F(ActivateTest, RotationsAreNormalized) {
    // Set a non-trivial quaternion: (3, 4, 0, 0) -> norm=5 -> (0.6, 0.8, 0, 0)
    raw.raw_rotations[0] = 3.0f;
    raw.raw_rotations[1] = 4.0f;
    raw.raw_rotations[2] = 0.0f;
    raw.raw_rotations[3] = 0.0f;

    raw.activate(g);
    EXPECT_NEAR(g.rotations[0], 0.6f, 1e-5f);
    EXPECT_NEAR(g.rotations[1], 0.8f, 1e-5f);
    EXPECT_NEAR(g.rotations[2], 0.0f, 1e-5f);
    EXPECT_NEAR(g.rotations[3], 0.0f, 1e-5f);
}

TEST_F(ActivateTest, SHCoeffsCopiedIdentity) {
    raw.activate(g);
    for (int i = 0; i < N * max_coeffs * 3; i++) {
        EXPECT_FLOAT_EQ(g.sh_coeffs[i], raw.raw_sh_coeffs[i]);
    }
}

TEST_F(ActivateTest, OpacitiesAreSigmoid) {
    raw.activate(g);
    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(g.opacities[i], 0.5f, 1e-5f);  // sigmoid(0) = 0.5
    }
}

TEST_F(ActivateTest, SigmoidLargePositive) {
    raw.raw_opacities[0] = 10.0f;
    raw.activate(g);
    EXPECT_NEAR(g.opacities[0], 1.0f / (1.0f + std::exp(-10.0f)), 1e-5f);
}

// --- RasterGradOutput ---
TEST(RasterGradOutput, AllocateAndZero) {
    FrameAllocator alloc(1 << 20);
    RasterGradOutput rgo{};
    rgo.allocate_and_zero(alloc, 10);
    EXPECT_NE(rgo.d_means2D, nullptr);
    EXPECT_NE(rgo.d_conics, nullptr);
    EXPECT_NE(rgo.d_rgb, nullptr);
    EXPECT_NE(rgo.d_opacities_2d, nullptr);

    // Verify zeroed
    for (int i = 0; i < 10 * 2; i++) EXPECT_EQ(rgo.d_means2D[i], 0.0f);
    for (int i = 0; i < 10 * 3; i++) EXPECT_EQ(rgo.d_conics[i], 0.0f);
    for (int i = 0; i < 10 * 3; i++) EXPECT_EQ(rgo.d_rgb[i], 0.0f);
    for (int i = 0; i < 10; i++) EXPECT_EQ(rgo.d_opacities_2d[i], 0.0f);
}

// --- GradientOutput ---
TEST(GradientOutput, AllocateAndZero) {
    FrameAllocator alloc(1 << 20);
    GradientOutput go{};
    go.allocate_and_zero(alloc, 8, 4);
    EXPECT_NE(go.d_raw_positions, nullptr);
    EXPECT_NE(go.d_raw_scales, nullptr);
    EXPECT_NE(go.d_raw_rotations, nullptr);
    EXPECT_NE(go.d_raw_sh_coeffs, nullptr);
    EXPECT_NE(go.d_raw_opacities, nullptr);

    for (int i = 0; i < 8 * 3; i++) EXPECT_EQ(go.d_raw_positions[i], 0.0f);
    for (int i = 0; i < 8 * 4; i++) EXPECT_EQ(go.d_raw_rotations[i], 0.0f);
    for (int i = 0; i < 8 * 4 * 3; i++) EXPECT_EQ(go.d_raw_sh_coeffs[i], 0.0f);
    for (int i = 0; i < 8; i++) EXPECT_EQ(go.d_raw_opacities[i], 0.0f);
}

// --- TrainConfig defaults ---
TEST(TrainConfig, DefaultValues) {
    TrainConfig cfg{};
    EXPECT_FLOAT_EQ(cfg.lr_position_init,  0.00016f);
    EXPECT_FLOAT_EQ(cfg.lr_position_final, 0.0000016f);
    EXPECT_FLOAT_EQ(cfg.lr_feature,        0.0025f);
    EXPECT_FLOAT_EQ(cfg.lr_opacity,        0.025f);
    EXPECT_FLOAT_EQ(cfg.lr_scaling,        0.005f);
    EXPECT_FLOAT_EQ(cfg.lr_rotation,       0.001f);
    EXPECT_EQ(cfg.max_steps, 30000);
}

// --- lr_schedule ---
TEST(LrSchedule, StartsAtInit) {
    float lr = lr_schedule(0.01f, 0.0001f, 0, 1000);
    EXPECT_NEAR(lr, 0.01f, 1e-6f);
}

TEST(LrSchedule, EndsAtFinal) {
    float lr = lr_schedule(0.01f, 0.0001f, 1000, 1000);
    EXPECT_NEAR(lr, 0.0001f, 1e-6f);
}

TEST(LrSchedule, MidpointIsGeometricMean) {
    float init = 0.01f, fin = 0.0001f;
    float lr = lr_schedule(init, fin, 500, 1000);
    float expected = std::sqrt(init * fin); // geometric mean at t=0.5
    EXPECT_NEAR(lr, expected, 1e-6f);
}

TEST(LrSchedule, ClampsNegativeStep) {
    float lr = lr_schedule(0.01f, 0.0001f, -5, 1000);
    EXPECT_NEAR(lr, 0.01f, 1e-6f);
}

TEST(LrSchedule, ClampsOvershootStep) {
    float lr = lr_schedule(0.01f, 0.0001f, 2000, 1000);
    EXPECT_NEAR(lr, 0.0001f, 1e-6f);
}

// --- SGDOptimizerState ---
TEST(SGDOptimizerState, DefaultStepZero) {
    SGDOptimizerState state{};
    EXPECT_EQ(state.step_count, 0);
}
