#include <gtest/gtest.h>
#include "sh_eval.h"
#include <cmath>

TEST(SHEval, Constants_MatchCUDA) {
    EXPECT_NEAR(SH_C0, 0.28209479177387814f, 1e-10f);
    EXPECT_NEAR(SH_C1, 0.4886025119029199f, 1e-10f);
    EXPECT_NEAR(SH_C2[0], 1.0925484305920792f, 1e-10f);
    EXPECT_NEAR(SH_C2[1], -1.0925484305920792f, 1e-10f);
    EXPECT_NEAR(SH_C2[2], 0.31539156525252005f, 1e-10f);
    EXPECT_NEAR(SH_C3[0], -0.5900435899266435f, 1e-10f);
    EXPECT_NEAR(SH_C3[6], -0.5900435899266435f, 1e-10f);
}

TEST(SHEval, Degree0_DC_Only) {
    float sh[3] = {0.5f, 0.3f, 0.1f};
    float pos[3] = {1, 0, 0};
    float cam[3] = {0, 0, 0};
    float rgb[3];
    computeColorFromSH(0, 1, sh, pos, cam, rgb);
    EXPECT_NEAR(rgb[0], SH_C0 * 0.5f + 0.5f, 1e-5f);
    EXPECT_NEAR(rgb[1], SH_C0 * 0.3f + 0.5f, 1e-5f);
    EXPECT_NEAR(rgb[2], SH_C0 * 0.1f + 0.5f, 1e-5f);
}

TEST(SHEval, ClampNegative) {
    float sh[3] = {-100.0f, -100.0f, -100.0f};
    float pos[3] = {1, 0, 0};
    float cam[3] = {0, 0, 0};
    float rgb[3];
    computeColorFromSH(0, 1, sh, pos, cam, rgb);
    EXPECT_GE(rgb[0], 0.0f);
    EXPECT_GE(rgb[1], 0.0f);
    EXPECT_GE(rgb[2], 0.0f);
}

TEST(SHEval, Degree1_ViewDependent) {
    float sh[12] = {
        0.0f, 0.0f, 0.0f,  // sh[0] DC
        1.0f, 0.0f, 0.0f,  // sh[1]
        0.0f, 1.0f, 0.0f,  // sh[2]
        0.0f, 0.0f, 1.0f,  // sh[3]
    };
    float pos[3] = {1, 0, 0};
    float cam[3] = {0, 0, 0};
    float rgb[3];
    // dir = (1,0,0), x=1, y=0, z=0
    // degree 1 contribution:
    // R: -SH_C1*0*1 + SH_C1*0*0 - SH_C1*1*0 = 0
    // G: 0
    // B: -SH_C1*0*0 + SH_C1*0*0 - SH_C1*1*1 = -SH_C1
    computeColorFromSH(1, 4, sh, pos, cam, rgb);
    EXPECT_NEAR(rgb[0], 0.5f, 1e-5f);  // 0 + 0.5
    EXPECT_NEAR(rgb[1], 0.5f, 1e-5f);  // 0 + 0.5
    float expected_b = -SH_C1 * 1.0f + 0.5f;
    EXPECT_NEAR(rgb[2], std::max(0.0f, expected_b), 1e-5f);
}

TEST(SHEval, TrainingMode_NoUpperClamp) {
    // SH_C0 * 3.0 + 0.5 = 0.2821 * 3.0 + 0.5 = 1.346 > 1.0
    float sh[3] = {3.0f, 3.0f, 3.0f};
    float pos[3] = {1, 0, 0};
    float cam[3] = {0, 0, 0};
    float rgb_inf[3], rgb_train[3];

    // Inference mode (default): should clamp to 1.0
    computeColorFromSH(0, 1, sh, pos, cam, rgb_inf);
    for (int c = 0; c < 3; c++)
        EXPECT_FLOAT_EQ(rgb_inf[c], 1.0f);

    // Training mode: should NOT clamp upper bound
    computeColorFromSH(0, 1, sh, pos, cam, rgb_train, /*training=*/true);
    float expected = SH_C0 * 3.0f + 0.5f;
    for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(rgb_train[c], expected, 1e-5f);
        EXPECT_GT(rgb_train[c], 1.0f);
    }
}

TEST(SHEval, TrainingMode_LowerClampStillWorks) {
    // Very negative coefficient: should clamp to 0 in both modes
    float sh[3] = {-100.0f, -100.0f, -100.0f};
    float pos[3] = {1, 0, 0};
    float cam[3] = {0, 0, 0};
    float rgb[3];

    computeColorFromSH(0, 1, sh, pos, cam, rgb, /*training=*/true);
    for (int c = 0; c < 3; c++)
        EXPECT_FLOAT_EQ(rgb[c], 0.0f);
}

TEST(SHEval, Degree0_AllDirections_Same) {
    // DC-only coefficient should produce same color regardless of view direction
    float sh[3] = {1.0f, 0.5f, 0.25f};
    float pos1[3] = {1, 0, 0};
    float pos2[3] = {0, 1, 0};
    float pos3[3] = {0, 0, 1};
    float cam[3] = {0, 0, 0};
    float rgb1[3], rgb2[3], rgb3[3];
    computeColorFromSH(0, 1, sh, pos1, cam, rgb1);
    computeColorFromSH(0, 1, sh, pos2, cam, rgb2);
    computeColorFromSH(0, 1, sh, pos3, cam, rgb3);
    for (int c = 0; c < 3; c++) {
        EXPECT_FLOAT_EQ(rgb1[c], rgb2[c]);
        EXPECT_FLOAT_EQ(rgb2[c], rgb3[c]);
    }
}
