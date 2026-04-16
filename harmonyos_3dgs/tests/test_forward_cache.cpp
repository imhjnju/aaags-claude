#include <gtest/gtest.h>
#include "types.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>

// Helper: create a camera looking down +Z with identity view matrix
static Camera makeCacheTestCamera(int w, int h, float fov_deg = 60.0f) {
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

// Helper: create a 1-Gaussian scene at (0,0,z)
struct SimpleScene {
    std::vector<float> positions;
    std::vector<float> sh_coeffs;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> opacities;
    GaussianData data;

    SimpleScene(float z = 5.0f) {
        positions = {0.0f, 0.0f, z};
        // SH degree 0: 1 coeff * 3 channels
        sh_coeffs = {0.5f, 0.5f, 0.5f};
        scales = {0.5f, 0.5f, 0.5f};
        rotations = {1.0f, 0.0f, 0.0f, 0.0f}; // identity quaternion
        opacities = {0.9f};

        data.count = 1;
        data.sh_degree = 0;
        data.max_coeffs = 1;
        data.positions = positions.data();
        data.sh_coeffs = sh_coeffs.data();
        data.scales = scales.data();
        data.rotations = rotations.data();
        data.opacities = opacities.data();
        data.filter_3D = nullptr;
    }
};

TEST(ForwardCache, PreprocessorPopulatesCache) {
    SimpleScene scene(5.0f);
    Camera cam = makeCacheTestCamera(32, 32);
    RenderConfig cfg{};
    cfg.sh_degree = 0;
    cfg.eval_3D = false;

    FrameAllocator alloc(16 * 1024 * 1024); // 16 MB
    ForwardCache cache{};

    PreprocessorCPU pp;
    auto pre = pp.process(scene.data, cam, cfg, alloc, &cache);

    // Cache arrays should be non-null
    ASSERT_NE(cache.cov3D, nullptr);
    ASSERT_NE(cache.p_view, nullptr);
    ASSERT_NE(cache.p_hom_w, nullptr);
    ASSERT_NE(cache.cov2D, nullptr);
    ASSERT_NE(cache.cov2D_det, nullptr);

    // The single Gaussian should have been processed (radius > 0)
    ASSERT_GT(pre.radii[0], 0) << "Gaussian was culled, test setup issue";

    // p_view: z should be ~5.0 (identity view matrix, position at z=5)
    EXPECT_NEAR(cache.p_view[2], 5.0f, 0.01f);

    // p_hom_w: should be positive (the w from clip-space projection)
    EXPECT_GT(cache.p_hom_w[0], 0.0f);

    // cov3D: should have non-zero values (scale=0.5, identity rotation)
    float cov3d_sum = 0;
    for (int j = 0; j < 6; j++) cov3d_sum += std::fabs(cache.cov3D[j]);
    EXPECT_GT(cov3d_sum, 0.0f);

    // cov2D: 2D covariance after +0.3 filter should have non-zero diagonal
    EXPECT_GT(cache.cov2D[0], 0.3f);  // diagonal must be > 0.3 (original + 0.3)
    EXPECT_GT(cache.cov2D[2], 0.3f);

    // cov2D_det: determinant of filtered cov2D should be positive
    EXPECT_GT(cache.cov2D_det[0], 0.0f);
}

TEST(ForwardCache, PreprocessorNullCacheBackwardCompat) {
    // Verify that passing nullptr for cache still works (no crash)
    SimpleScene scene(5.0f);
    Camera cam = makeCacheTestCamera(32, 32);
    RenderConfig cfg{};
    cfg.sh_degree = 0;

    FrameAllocator alloc(16 * 1024 * 1024);
    PreprocessorCPU pp;
    auto pre = pp.process(scene.data, cam, cfg, alloc, nullptr);

    ASSERT_GT(pre.radii[0], 0);
}

TEST(ForwardCache, FullPipelinePopulatesRasterizerCache) {
    SimpleScene scene(5.0f);
    int W = 32, H = 32;
    Camera cam = makeCacheTestCamera(W, H);
    RenderConfig cfg{};
    cfg.sh_degree = 0;
    cfg.eval_3D = false;

    FrameAllocator alloc(64 * 1024 * 1024); // 64 MB
    ForwardCache cache{};

    // Preprocess
    PreprocessorCPU pp;
    auto pre = pp.process(scene.data, cam, cfg, alloc, &cache);
    ASSERT_GT(pre.radii[0], 0) << "Gaussian was culled";

    // Bin
    TileBinnerCPU binner;
    auto bin = binner.bin(pre, scene.data.count, cam, cfg, alloc);

    // Sort
    SorterCPU sorter;
    if (bin.total_pairs > 0)
        sorter.sort(bin, alloc);

    // Rasterize with cache
    std::vector<float> image(W * H * 3, 0.0f);
    RasterizerCPU rast;
    rast.rasterize(pre, bin, cam, cfg, image.data(), nullptr, &cache, &alloc);

    // Cache per-pixel arrays should be populated
    ASSERT_NE(cache.T_final, nullptr);
    ASSERT_NE(cache.n_contrib, nullptr);

    // Center pixel should have been affected by the Gaussian
    int cx = W / 2, cy = H / 2;
    int center_pix = cy * W + cx;

    // T_final should be < 1.0 at center (some opacity was absorbed)
    EXPECT_LT(cache.T_final[center_pix], 1.0f)
        << "Center pixel T_final should be < 1.0 (Gaussian contributed)";
    EXPECT_GT(cache.T_final[center_pix], 0.0f)
        << "Center pixel T_final should be > 0.0 (not fully opaque)";

    // n_contrib should be > 0 at center
    EXPECT_GT(cache.n_contrib[center_pix], 0)
        << "Center pixel should have at least 1 contributing Gaussian";
}

TEST(ForwardCache, RasterizerNullCacheBackwardCompat) {
    // Verify rasterizer works with nullptr cache (no crash)
    SimpleScene scene(5.0f);
    int W = 32, H = 32;
    Camera cam = makeCacheTestCamera(W, H);
    RenderConfig cfg{};
    cfg.sh_degree = 0;

    FrameAllocator alloc(64 * 1024 * 1024);

    PreprocessorCPU pp;
    auto pre = pp.process(scene.data, cam, cfg, alloc);

    TileBinnerCPU binner;
    auto bin = binner.bin(pre, scene.data.count, cam, cfg, alloc);

    SorterCPU sorter;
    if (bin.total_pairs > 0)
        sorter.sort(bin, alloc);

    std::vector<float> image(W * H * 3, 0.0f);
    RasterizerCPU rast;
    // Call without cache args (default nullptr)
    rast.rasterize(pre, bin, cam, cfg, image.data());

    // Should produce non-zero output at center
    int cx = W / 2, cy = H / 2;
    int center_pix = cy * W + cx;
    float sum = image[center_pix*3] + image[center_pix*3+1] + image[center_pix*3+2];
    EXPECT_GT(sum, 0.0f);
}
