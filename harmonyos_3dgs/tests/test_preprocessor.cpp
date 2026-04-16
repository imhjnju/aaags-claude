#include <gtest/gtest.h>
#include "types.h"
#include "cpu/preprocessor_cpu.h"
#include <cmath>
#include <cstring>

// Helper: create a test camera looking down +Z
Camera makeTestCamera(int w, int h, float fov_deg = 60.0f) {
    Camera cam{};
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    memcpy(cam.view_matrix, identity, sizeof(identity));
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)h / (float)w;
    cam.width = w; cam.height = h;
    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0;
    // Simple perspective viewproj (column-major)
    float n = 0.01f, f = 100.0f;
    float r = cam.tan_fovx * n, t = cam.tan_fovy * n;
    memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0]  = n / r;
    cam.viewproj_matrix[5]  = n / t;
    cam.viewproj_matrix[10] = f / (f - n);
    cam.viewproj_matrix[11] = 1.0f;
    cam.viewproj_matrix[14] = -(f*n)/(f-n);
    return cam;
}

TEST(PreprocessorCPU, SingleGaussian_InView) {
    FrameAllocator alloc(1024 * 1024);
    GaussianData g{};
    g.count = 1; g.sh_degree = 0; g.max_coeffs = 1;
    float pos[3] = {0, 0, 5};
    float sh[3] = {0.5f, 0.5f, 0.5f};
    float scale[3] = {1, 1, 1};
    float rot[4] = {1, 0, 0, 0};
    float opa[1] = {0.9f};
    g.positions = pos; g.sh_coeffs = sh; g.scales = scale;
    g.rotations = rot; g.opacities = opa;

    Camera cam = makeTestCamera(320, 240);
    RenderConfig config{}; config.sh_degree = 0;

    PreprocessorCPU pp;
    auto out = pp.process(g, cam, config, alloc);
    EXPECT_GT(out.radii[0], 0);
    EXPECT_NEAR(out.depths[0], 5.0f, 1e-5f);
    EXPECT_GT(out.tiles_touched[0], 0);
    EXPECT_GT(out.opacities_2d[0], 0.0f);
    // RGB should be non-zero (DC = 0.5, result = SH_C0*0.5 + 0.5)
    EXPECT_GT(out.rgb[0], 0.0f);
}

TEST(PreprocessorCPU, BehindCamera_Culled) {
    FrameAllocator alloc(1024 * 1024);
    GaussianData g{};
    g.count = 1; g.sh_degree = 0; g.max_coeffs = 1;
    float pos[3] = {0, 0, -5};
    float sh[3] = {0.5f, 0.5f, 0.5f};
    float scale[3] = {1, 1, 1};
    float rot[4] = {1, 0, 0, 0};
    float opa[1] = {0.9f};
    g.positions = pos; g.sh_coeffs = sh; g.scales = scale;
    g.rotations = rot; g.opacities = opa;

    Camera cam = makeTestCamera(320, 240);
    RenderConfig config{}; config.sh_degree = 0;

    PreprocessorCPU pp;
    auto out = pp.process(g, cam, config, alloc);
    EXPECT_EQ(out.radii[0], 0);
    EXPECT_EQ(out.tiles_touched[0], 0);
}

TEST(PreprocessorCPU, MultipleGaussians) {
    FrameAllocator alloc(4 * 1024 * 1024);
    GaussianData g{};
    g.count = 3; g.sh_degree = 0; g.max_coeffs = 1;
    float pos[9] = {0,0,5, 1,0,5, 0,0,-5}; // G0 visible, G1 visible, G2 behind
    float sh[9] = {0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f};
    float scale[9] = {1,1,1, 1,1,1, 1,1,1};
    float rot[12] = {1,0,0,0, 1,0,0,0, 1,0,0,0};
    float opa[3] = {0.9f, 0.8f, 0.7f};
    g.positions = pos; g.sh_coeffs = sh; g.scales = scale;
    g.rotations = rot; g.opacities = opa;

    Camera cam = makeTestCamera(320, 240);
    RenderConfig config{}; config.sh_degree = 0;

    PreprocessorCPU pp;
    auto out = pp.process(g, cam, config, alloc);
    EXPECT_GT(out.radii[0], 0);  // G0 visible
    EXPECT_GT(out.radii[1], 0);  // G1 visible
    EXPECT_EQ(out.radii[2], 0);  // G2 behind camera
}
