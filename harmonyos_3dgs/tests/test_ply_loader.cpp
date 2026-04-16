#include <gtest/gtest.h>
#include "ply_loader.h"
#include <cmath>

TEST(PLYLoader, LoadTiny) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    EXPECT_EQ(model.data.count, 3);
    EXPECT_EQ(model.data.sh_degree, 3);
    EXPECT_EQ(model.data.max_coeffs, 16);
    EXPECT_NE(model.data.positions, nullptr);
    EXPECT_NE(model.data.sh_coeffs, nullptr);
    EXPECT_NE(model.data.scales, nullptr);
    EXPECT_NE(model.data.rotations, nullptr);
    EXPECT_NE(model.data.opacities, nullptr);
    model.free();
}

TEST(PLYLoader, Positions) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    EXPECT_NEAR(model.data.positions[0], 0.0f, 1e-5f); // G0.x
    EXPECT_NEAR(model.data.positions[1], 0.0f, 1e-5f); // G0.y
    EXPECT_NEAR(model.data.positions[2], 5.0f, 1e-5f); // G0.z
    EXPECT_NEAR(model.data.positions[3], 1.0f, 1e-5f); // G1.x
    model.free();
}

TEST(PLYLoader, ActivationOpacity) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    // sigmoid(0) = 0.5
    EXPECT_NEAR(model.data.opacities[0], 0.5f, 1e-5f);
    // sigmoid(2) ~ 0.8808
    EXPECT_NEAR(model.data.opacities[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-4f);
    // sigmoid(-2) ~ 0.1192
    EXPECT_NEAR(model.data.opacities[2], 1.0f / (1.0f + std::exp(2.0f)), 1e-4f);
    model.free();
}

TEST(PLYLoader, ActivationScale) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    // exp(0) = 1
    EXPECT_NEAR(model.data.scales[0], 1.0f, 1e-5f);
    EXPECT_NEAR(model.data.scales[1], 1.0f, 1e-5f);
    EXPECT_NEAR(model.data.scales[2], 1.0f, 1e-5f);
    // exp(1) ~ 2.718
    EXPECT_NEAR(model.data.scales[3], std::exp(1.0f), 1e-4f);
    model.free();
}

TEST(PLYLoader, ActivationRotation) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    // rot=(1,0,0,0) normalized -> (1,0,0,0)
    EXPECT_NEAR(model.data.rotations[0], 1.0f, 1e-5f);
    EXPECT_NEAR(model.data.rotations[1], 0.0f, 1e-5f);
    EXPECT_NEAR(model.data.rotations[2], 0.0f, 1e-5f);
    EXPECT_NEAR(model.data.rotations[3], 0.0f, 1e-5f);
    // rot=(1,1,0,0) normalized -> (1/sqrt(2), 1/sqrt(2), 0, 0)
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(model.data.rotations[8], inv_sqrt2, 1e-4f);
    EXPECT_NEAR(model.data.rotations[9], inv_sqrt2, 1e-4f);
    EXPECT_NEAR(model.data.rotations[10], 0.0f, 1e-5f);
    EXPECT_NEAR(model.data.rotations[11], 0.0f, 1e-5f);
    model.free();
}

TEST(PLYLoader, SH_DC_Coefficients) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    // DC coeffs for Gaussian 0 should be the f_dc values from PLY
    // sh_coeffs[0] = f_dc_0 (R), sh_coeffs[1] = f_dc_1 (G), sh_coeffs[2] = f_dc_2 (B)
    // (verify they are the values we set in generate_test_ply.py)
    EXPECT_NE(model.data.sh_coeffs[0], 0.0f); // Should have some value
    model.free();
}
