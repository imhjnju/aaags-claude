#include <gtest/gtest.h>
#include "math_utils.h"
#include <cmath>
#include <cstring>

TEST(MathUtils, TransformPoint4x3_Identity) {
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float p[3] = {1.0f, 2.0f, 3.0f};
    float out[3];
    transformPoint4x3(p, identity, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST(MathUtils, TransformPoint4x3_Translation) {
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 5,6,7,1};
    float p[3] = {1.0f, 2.0f, 3.0f};
    float out[3];
    transformPoint4x3(p, m, out);
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
}

TEST(MathUtils, TransformPoint4x4_HomogeneousDivide) {
    float m[16] = {2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,1};
    float p[3] = {1.0f, 2.0f, 3.0f};
    float out[4];
    transformPoint4x4(p, m, out);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 6.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST(MathUtils, TransformPoint4x3_ColumnMajorRotation) {
    // 90-degree rotation around Z: col0=(0,1,0,0), col1=(-1,0,0,0), col2=(0,0,1,0), col3=(0,0,0,1)
    float m[16] = {0,1,0,0, -1,0,0,0, 0,0,1,0, 0,0,0,1};
    float p[3] = {1.0f, 0.0f, 0.0f};
    float out[3];
    transformPoint4x3(p, m, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
}

TEST(MathUtils, Ndc2Pix) {
    EXPECT_FLOAT_EQ(ndc2Pix(0.0f, 1920), 959.5f);
    EXPECT_FLOAT_EQ(ndc2Pix(-1.0f, 1920), -0.5f);
    EXPECT_FLOAT_EQ(ndc2Pix(1.0f, 1920), 1919.5f);
}

TEST(MathUtils, GetRect_SingleTile) {
    float p[2] = {8.0f, 8.0f};
    int rect_min[2], rect_max[2];
    getRect(p, 5, 120, 68, 16, 16, rect_min, rect_max);
    EXPECT_EQ(rect_min[0], 0);
    EXPECT_EQ(rect_min[1], 0);
    EXPECT_EQ(rect_max[0], 1);
    EXPECT_EQ(rect_max[1], 1);
}

TEST(MathUtils, GetRect_Clamped) {
    float p[2] = {0.0f, 0.0f};
    int rect_min[2], rect_max[2];
    getRect(p, 100, 120, 68, 16, 16, rect_min, rect_max);
    EXPECT_EQ(rect_min[0], 0);
    EXPECT_EQ(rect_min[1], 0);
    EXPECT_LE(rect_max[0], 120);
    EXPECT_LE(rect_max[1], 68);
}

TEST(MathUtils, InFrustum_BehindCamera) {
    float vm[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float vpm[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float point[3] = {0, 0, -1.0f};
    float p_view[3];
    EXPECT_FALSE(inFrustum(point, vm, vpm, p_view));
}

TEST(MathUtils, InFrustum_InFront) {
    float vm[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float vpm[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float point[3] = {0, 0, 5.0f};
    float p_view[3];
    EXPECT_TRUE(inFrustum(point, vm, vpm, p_view));
    EXPECT_FLOAT_EQ(p_view[2], 5.0f);
}

TEST(MathUtils, ComputeCov3D_IdentityRotation) {
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float cov3d[6];
    computeCov3D(scale, 1.0f, rot, cov3d);
    EXPECT_NEAR(cov3d[0], 1.0f, 1e-5f);
    EXPECT_NEAR(cov3d[1], 0.0f, 1e-5f);
    EXPECT_NEAR(cov3d[2], 0.0f, 1e-5f);
    EXPECT_NEAR(cov3d[3], 1.0f, 1e-5f);
    EXPECT_NEAR(cov3d[4], 0.0f, 1e-5f);
    EXPECT_NEAR(cov3d[5], 1.0f, 1e-5f);
}

TEST(MathUtils, ComputeCov3D_ScaleModifier) {
    float scale[3] = {2.0f, 3.0f, 4.0f};
    float rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float cov3d[6];
    computeCov3D(scale, 0.5f, rot, cov3d);
    EXPECT_NEAR(cov3d[0], 1.0f, 1e-5f);    // (0.5*2)^2 = 1
    EXPECT_NEAR(cov3d[3], 2.25f, 1e-5f);   // (0.5*3)^2 = 2.25
    EXPECT_NEAR(cov3d[5], 4.0f, 1e-5f);    // (0.5*4)^2 = 4
}

TEST(MathUtils, ComputeCov2D_IdentityViewSimple) {
    float mean[3] = {0, 0, 5.0f};
    float cov3d[6] = {1, 0, 0, 1, 0, 1}; // Identity 3x3
    float vm[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // Identity view
    float focal_x = 500.0f, focal_y = 500.0f;
    float tan_fovx = 0.5f, tan_fovy = 0.5f;
    float cov2d[3];
    computeCov2D(mean, cov3d, vm, focal_x, focal_y, tan_fovx, tan_fovy, cov2d);
    // With identity view, point at (0,0,5), the result should be a well-defined 2x2 matrix
    EXPECT_GT(cov2d[0], 0.0f); // a > 0 (variance in x)
    EXPECT_GT(cov2d[2], 0.0f); // c > 0 (variance in y)
    // For identity cov3d and centered point, should be symmetric: a == c
    EXPECT_NEAR(cov2d[0], cov2d[2], 1e-4f);
    EXPECT_NEAR(cov2d[1], 0.0f, 1e-4f); // no off-diagonal for centered point
}
