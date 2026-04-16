#include <gtest/gtest.h>
#include "renderer.h"
#include "ply_loader.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "image_io.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>

static Camera makeE2ECamera(int w, int h, float fov_deg = 60.0f) {
    Camera cam{};
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    memcpy(cam.view_matrix, identity, sizeof(identity));
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)h / (float)w;
    cam.width = w; cam.height = h;
    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0;
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

TEST(RendererE2E, SmallScene_Renders) {
    auto model = loadPly(TEST_DATA_DIR "/tiny_3gaussians.ply");
    Camera cam = makeE2ECamera(64, 48);
    RenderConfig config{};
    config.sh_degree = model.data.sh_degree;

    Renderer renderer(
        std::make_unique<PreprocessorCPU>(),
        std::make_unique<TileBinnerCPU>(),
        std::make_unique<SorterCPU>(),
        std::make_unique<RasterizerCPU>(),
        64 * 1024 * 1024  // 64MB enough for tiny scene
    );

    std::vector<float> image(64 * 48 * 3, 0.0f);
    renderer.render(model.data, cam, config, image.data());

    // Should produce non-trivial output (not all zeros)
    float sum = 0;
    for (float v : image) sum += v;
    EXPECT_GT(sum, 0.0f);

    model.free();
}

TEST(RendererE2E, BackgroundOnly_NoGaussians) {
    GaussianData g{};
    g.count = 0; g.sh_degree = 0; g.max_coeffs = 1;
    Camera cam = makeE2ECamera(32, 32);
    RenderConfig config{};
    config.bg_color[0] = 0.5f; config.bg_color[1] = 0.5f; config.bg_color[2] = 0.5f;

    Renderer renderer(
        std::make_unique<PreprocessorCPU>(),
        std::make_unique<TileBinnerCPU>(),
        std::make_unique<SorterCPU>(),
        std::make_unique<RasterizerCPU>(),
        4 * 1024 * 1024
    );

    std::vector<float> image(32 * 32 * 3, 0.0f);
    renderer.render(g, cam, config, image.data());

    // All pixels should be background color
    for (int i = 0; i < 32 * 32; i++) {
        EXPECT_FLOAT_EQ(image[i * 3 + 0], 0.5f);
        EXPECT_FLOAT_EQ(image[i * 3 + 1], 0.5f);
        EXPECT_FLOAT_EQ(image[i * 3 + 2], 0.5f);
    }
}

TEST(ImageIO, WritePPM_Basic) {
    const int w = 2, h = 2;
    float image[w * h * 3] = {
        1.0f, 0.0f, 0.0f,  // red
        0.0f, 1.0f, 0.0f,  // green
        0.0f, 0.0f, 1.0f,  // blue
        1.0f, 1.0f, 1.0f   // white
    };
    const char* path = "/tmp/test_output.ppm";
    EXPECT_TRUE(writePPM(path, image, w, h));

    // Verify file exists and has correct size
    FILE* f = fopen(path, "rb");
    ASSERT_NE(f, nullptr);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    // Header "P6\n2 2\n255\n" = 13 bytes, data = 2*2*3 = 12 bytes
    EXPECT_GT(size, 12); // at least pixel data
    std::remove(path);
}
