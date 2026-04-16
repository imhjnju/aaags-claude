#include <gtest/gtest.h>
#include "trainer.h"
#include "image_io.h"
#include "renderer.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <random>

// Helper: create a camera looking down +Z
static Camera makeTestCamera(int w, int h, float fov_deg = 60.0f) {
    Camera cam{};
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    memcpy(cam.view_matrix, identity, sizeof(identity));
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)h / (float)w;
    cam.width = w; cam.height = h;
    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0;
    float n = 0.01f, f = 100.0f;
    memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0]  = 1.0f / cam.tan_fovx;
    cam.viewproj_matrix[5]  = 1.0f / cam.tan_fovy;
    cam.viewproj_matrix[10] = f / (f - n);
    cam.viewproj_matrix[11] = 1.0f;
    cam.viewproj_matrix[14] = -(f * n) / (f - n);
    return cam;
}

// Helper: create RawGaussianParams for a single Gaussian
struct SingleGaussianSetup {
    float pos[3], sc[3], rot[4], sh[3], op[1];
    RawGaussianParams raw;

    SingleGaussianSetup(float px, float py, float pz,
                        float sx, float sy, float sz,
                        float sh_val, float opacity_logit) {
        pos[0] = px; pos[1] = py; pos[2] = pz;
        sc[0] = logf(sx); sc[1] = logf(sy); sc[2] = logf(sz);
        rot[0] = 1; rot[1] = 0; rot[2] = 0; rot[3] = 0;
        sh[0] = sh_val; sh[1] = sh_val; sh[2] = sh_val;
        op[0] = opacity_logit;
        raw.count = 1; raw.sh_degree = 0; raw.max_coeffs = 1;
        raw.raw_positions = pos;
        raw.raw_scales = sc;
        raw.raw_rotations = rot;
        raw.raw_sh_coeffs = sh;
        raw.raw_opacities = op;
    }
};

// Render a GaussianData to an image using the full pipeline
static std::vector<float> renderGaussians(const GaussianData& g,
                                           const Camera& cam,
                                           const RenderConfig& cfg) {
    auto renderer = std::make_unique<Renderer>(
        std::make_unique<PreprocessorCPU>(),
        std::make_unique<TileBinnerCPU>(),
        std::make_unique<SorterCPU>(),
        std::make_unique<RasterizerCPU>(),
        128ULL * 1024 * 1024);
    int npix = cam.width * cam.height;
    std::vector<float> img(npix * 3, 0.0f);
    renderer->render(g, cam, cfg, img.data());
    return img;
}

// ---------------------------------------------------------------------------
// Test: synthetic data training loop
// Creates GT from known Gaussians, trains from a different initialization,
// and verifies loss decreases and trained model produces similar images.
// ---------------------------------------------------------------------------
TEST(TrainLoop, SyntheticSingleView) {
    const int W = 16, H = 16;
    Camera cam = makeTestCamera(W, H);

    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = 0;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // Create ground truth: a bright Gaussian at (0, 0, 5)
    GaussianData gt_g;
    gt_g.count = 1; gt_g.sh_degree = 0; gt_g.max_coeffs = 1;
    float gt_pos[3] = {0, 0, 5};
    float gt_sc[3] = {0.3f, 0.3f, 0.3f};
    float gt_rot[4] = {1, 0, 0, 0};
    float gt_sh[3] = {2.0f, 2.0f, 2.0f};  // bright
    float gt_op[1] = {0.9f};
    gt_g.positions = gt_pos; gt_g.scales = gt_sc; gt_g.rotations = gt_rot;
    gt_g.sh_coeffs = gt_sh; gt_g.opacities = gt_op; gt_g.filter_3D = nullptr;

    auto gt_image = renderGaussians(gt_g, cam, cfg);

    float gt_sum = 0;
    for (auto v : gt_image) gt_sum += v;
    ASSERT_GT(gt_sum, 0.0f) << "GT image is all black";
    printf("  GT image sum = %.4f\n", gt_sum);

    // Train from wrong SH starting point
    SingleGaussianSetup setup(0.0f, 0.0f, 5.0f, 0.3f, 0.3f, 0.3f,
                              0.5f,  // wrong SH (target is 2.0)
                              2.197f); // inverse_sigmoid(0.9)

    TrainConfig tcfg;
    tcfg.lr_position_init  = 1e-6f;
    tcfg.lr_position_final = 1e-6f;
    tcfg.lr_feature        = 10.0f;  // learn SH
    tcfg.lr_opacity        = 1e-4f;
    tcfg.lr_scaling        = 1e-4f;
    tcfg.lr_rotation       = 1e-4f;
    tcfg.max_steps         = 500;

    Trainer trainer(256 * 1024 * 1024);

    float first_loss = -1, last_loss = -1;
    for (int iter = 0; iter < 500; iter++) {
        auto result = trainer.step(setup.raw, cam, gt_image.data(), cfg, tcfg, iter);
        if (iter == 0) first_loss = result.loss;
        last_loss = result.loss;
        if (iter < 5 || iter % 100 == 0 || iter == 499) {
            printf("  iter %3d: loss=%.6f  sh=[%.4f,%.4f,%.4f]\n",
                   iter, result.loss, setup.sh[0], setup.sh[1], setup.sh[2]);
        }
    }

    printf("  first_loss=%.6f  last_loss=%.6f  ratio=%.4f\n",
           first_loss, last_loss, last_loss / first_loss);

    // Loss should decrease substantially
    EXPECT_LT(last_loss, first_loss * 0.1f)
        << "Loss should decrease by at least 90% after 500 iterations";

    // SH should converge toward 2.0
    EXPECT_NEAR(setup.sh[0], 2.0f, 0.3f)
        << "SH coefficient should converge to target value";

    // Render trained model and compare with GT
    GaussianData trained_g;
    trained_g.count = 1; trained_g.sh_degree = 0; trained_g.max_coeffs = 1;
    float t_pos[3], t_sc[3], t_rot[4], t_sh[3], t_op[1];
    trained_g.positions = t_pos; trained_g.scales = t_sc; trained_g.rotations = t_rot;
    trained_g.sh_coeffs = t_sh; trained_g.opacities = t_op; trained_g.filter_3D = nullptr;
    setup.raw.activate(trained_g);

    auto trained_image = renderGaussians(trained_g, cam, cfg);

    // Compute L1 difference between trained and GT renders
    float l1_diff = 0;
    for (int i = 0; i < W * H * 3; i++)
        l1_diff += std::abs(trained_image[i] - gt_image[i]);
    l1_diff /= (W * H * 3);
    printf("  Trained vs GT L1 diff = %.6f\n", l1_diff);
    EXPECT_LT(l1_diff, 0.05f) << "Trained render should be close to GT";
}

// ---------------------------------------------------------------------------
// Test: multi-view training (2 views)
// ---------------------------------------------------------------------------
TEST(TrainLoop, SyntheticMultiView) {
    const int W = 16, H = 16;

    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = 0;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // Two cameras: one looking from front, one slightly offset
    Camera cam1 = makeTestCamera(W, H);
    Camera cam2 = makeTestCamera(W, H);
    // cam2: offset slightly in X
    cam2.cam_pos[0] = 0.5f;
    cam2.view_matrix[12] = -0.5f; // translate view

    // GT Gaussian
    GaussianData gt_g;
    gt_g.count = 1; gt_g.sh_degree = 0; gt_g.max_coeffs = 1;
    float gt_pos[3] = {0, 0, 5};
    float gt_sc[3] = {0.3f, 0.3f, 0.3f};
    float gt_rot[4] = {1, 0, 0, 0};
    float gt_sh[3] = {2.0f, 2.0f, 2.0f};
    float gt_op[1] = {0.9f};
    gt_g.positions = gt_pos; gt_g.scales = gt_sc; gt_g.rotations = gt_rot;
    gt_g.sh_coeffs = gt_sh; gt_g.opacities = gt_op; gt_g.filter_3D = nullptr;

    auto gt1 = renderGaussians(gt_g, cam1, cfg);
    auto gt2 = renderGaussians(gt_g, cam2, cfg);

    // Train from wrong SH
    SingleGaussianSetup setup(0.0f, 0.0f, 5.0f, 0.3f, 0.3f, 0.3f,
                              0.5f, 2.197f);

    TrainConfig tcfg;
    tcfg.lr_position_init  = 1e-6f;
    tcfg.lr_position_final = 1e-6f;
    tcfg.lr_feature        = 10.0f;
    tcfg.lr_opacity        = 1e-4f;
    tcfg.lr_scaling        = 1e-4f;
    tcfg.lr_rotation       = 1e-4f;
    tcfg.max_steps         = 500;

    Trainer trainer(256 * 1024 * 1024);
    std::mt19937 rng(42);

    float first_loss = -1, last_loss = -1;
    for (int iter = 0; iter < 500; iter++) {
        // Alternate views randomly (matching 3DGS strategy)
        int view_idx = rng() % 2;
        const Camera& cam = (view_idx == 0) ? cam1 : cam2;
        const float* gt = (view_idx == 0) ? gt1.data() : gt2.data();

        auto result = trainer.step(setup.raw, cam, gt, cfg, tcfg, iter);
        if (iter == 0) first_loss = result.loss;
        last_loss = result.loss;
    }

    printf("  Multi-view: first_loss=%.6f  last_loss=%.6f  ratio=%.4f\n",
           first_loss, last_loss, last_loss / first_loss);

    EXPECT_LT(last_loss, first_loss * 0.5f)
        << "Multi-view loss should decrease by at least 50%";
}

// ---------------------------------------------------------------------------
// Test: PPM round-trip (write then read)
// ---------------------------------------------------------------------------
TEST(TrainLoop, PPMRoundTrip) {
    const int W = 4, H = 3;
    std::vector<float> image(W * H * 3);
    for (int i = 0; i < W * H * 3; i++)
        image[i] = (float)i / (float)(W * H * 3);

    const char* path = "/tmp/gs3d_test_roundtrip.ppm";
    ASSERT_TRUE(writePPM(path, image.data(), W, H));

    int rw, rh;
    auto loaded = readPPM(path, rw, rh);
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(rw, W);
    EXPECT_EQ(rh, H);
    EXPECT_EQ((int)loaded.size(), W * H * 3);

    // Check values (8-bit quantization means ~0.004 max error)
    for (int i = 0; i < W * H * 3; i++) {
        EXPECT_NEAR(loaded[i], image[i], 1.0f / 255.0f + 0.001f)
            << "PPM round-trip mismatch at pixel " << i;
    }
}

// ============================================================
// Adam Training Tests
// ============================================================

// Helper: create a synthetic training scene
static void make_synthetic_scene(
        OwnedRawParams& owned, Camera& cam, RenderConfig& cfg,
        std::vector<float>& gt_image,
        int N, int W, int H, int sh_degree) {
    int max_coeffs = (sh_degree + 1) * (sh_degree + 1);
    owned.sh_degree = sh_degree;
    owned.max_coeffs = max_coeffs;
    owned.resize(N);

    // Gaussians spread in front of camera
    for (int i = 0; i < N; i++) {
        owned.positions[i*3+0] = (float)(i % 5 - 2) * 0.5f;
        owned.positions[i*3+1] = (float)(i / 5 - 1) * 0.5f;
        owned.positions[i*3+2] = 5.0f + (float)i * 0.1f;
        for (int j = 0; j < 3; j++) owned.scales[i*3+j] = std::log(0.3f);
        owned.rotations[i*4+0] = 1.0f;
        for (int j = 1; j < 4; j++) owned.rotations[i*4+j] = 0.0f;
        owned.opacities[i] = 2.0f;  // sigmoid(2) ≈ 0.88
        for (int j = 0; j < max_coeffs * 3; j++)
            owned.sh_coeffs[i * max_coeffs * 3 + j] = (j < 3) ? 1.0f : 0.0f;
    }

    // Camera
    float fov = 60.0f;
    float tan_fov = std::tan(fov * 0.5f * 3.14159265f / 180.0f);
    std::memset(&cam, 0, sizeof(cam));
    cam.view_matrix[0] = cam.view_matrix[5] = cam.view_matrix[10] = cam.view_matrix[15] = 1.0f;
    float near_v = 0.01f, far_v = 100.0f;
    cam.viewproj_matrix[0] = near_v / (tan_fov * near_v);
    cam.viewproj_matrix[5] = near_v / (tan_fov * near_v);
    cam.viewproj_matrix[10] = far_v / (far_v - near_v);
    cam.viewproj_matrix[11] = 1.0f;
    cam.viewproj_matrix[14] = -(far_v * near_v) / (far_v - near_v);
    cam.tan_fovx = cam.tan_fovy = tan_fov;
    cam.width = W; cam.height = H;

    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f; cfg.antialiasing = false; cfg.eval_3D = false;
    cfg.training = true; cfg.sh_degree = sh_degree;
    cfg.tile_w = 16; cfg.tile_h = 16;

    // Render GT from shifted SH
    OwnedRawParams gt_owned = owned;
    for (int i = 0; i < N; i++)
        for (int ch = 0; ch < 3; ch++)
            gt_owned.sh_coeffs[i * max_coeffs * 3 + ch] += 0.5f;

    gt_image.resize(W * H * 3);
    {
        RawGaussianParams gt_raw = gt_owned.as_raw();
        GaussianData g;
        g.count = N; g.sh_degree = sh_degree; g.max_coeffs = max_coeffs;
        std::vector<float> a_p(N*3), a_s(N*3), a_r(N*4), a_sh(N*max_coeffs*3), a_o(N);
        g.positions = a_p.data(); g.scales = a_s.data(); g.rotations = a_r.data();
        g.sh_coeffs = a_sh.data(); g.opacities = a_o.data(); g.filter_3D = nullptr;
        gt_raw.activate(g);

        FrameAllocator alloc(64 * 1024 * 1024);
        PreprocessorCPU pp; TileBinnerCPU bn; SorterCPU sr; RasterizerCPU rs;
        ForwardCache cache{};
        auto pre = pp.process(g, cam, cfg, alloc, &cache);
        auto bin = bn.bin(pre, N, cam, cfg, alloc);
        if (bin.total_pairs > 0) sr.sort(bin, alloc);
        cache.pre = &pre; cache.bin = &bin;
        float* rendered = alloc.allocate_array<float>(W * H * 3);
        rs.rasterize(pre, bin, cam, cfg, rendered, nullptr, &cache, &alloc);
        std::memcpy(gt_image.data(), rendered, W * H * 3 * sizeof(float));
    }
}

TEST(TrainAdam, SH0_ConvergesWithAdam) {
    OwnedRawParams owned; Camera cam; RenderConfig cfg;
    std::vector<float> gt_image;
    make_synthetic_scene(owned, cam, cfg, gt_image, 3, 16, 16, 0);

    TrainConfig tcfg;
    tcfg.use_adam = true;
    tcfg.max_steps = 500;

    Trainer trainer(64 * 1024 * 1024);
    RawGaussianParams params = owned.as_raw();
    float first_loss = 0, last_loss = 0;
    for (int i = 0; i < 500; i++) {
        auto r = trainer.step(params, cam, gt_image.data(), cfg, tcfg, i);
        if (i == 0) first_loss = r.loss;
        last_loss = r.loss;
    }
    EXPECT_GT(first_loss, 0.001f) << "Initial loss should be non-trivial";
    EXPECT_LT(last_loss / first_loss, 0.05f) << "Adam should reduce loss by >95%";
}

TEST(TrainAdam, SH1_ConvergesWithAdam) {
    OwnedRawParams owned; Camera cam; RenderConfig cfg;
    std::vector<float> gt_image;
    make_synthetic_scene(owned, cam, cfg, gt_image, 3, 16, 16, 1);

    TrainConfig tcfg;
    tcfg.use_adam = true;
    tcfg.max_steps = 1000;

    Trainer trainer(64 * 1024 * 1024);
    RawGaussianParams params = owned.as_raw();
    float first_loss = 0, last_loss = 0;
    for (int i = 0; i < 1000; i++) {
        auto r = trainer.step(params, cam, gt_image.data(), cfg, tcfg, i);
        if (i == 0) first_loss = r.loss;
        last_loss = r.loss;
    }
    EXPECT_GT(first_loss, 0.001f);
    EXPECT_LT(last_loss / first_loss, 0.10f) << "Adam+SH1 should reduce loss by >90%";
}

TEST(TrainAdam, SH3_ConvergesWithAdam) {
    OwnedRawParams owned; Camera cam; RenderConfig cfg;
    std::vector<float> gt_image;
    make_synthetic_scene(owned, cam, cfg, gt_image, 3, 16, 16, 3);

    TrainConfig tcfg;
    tcfg.use_adam = true;
    tcfg.max_steps = 1000;

    Trainer trainer(64 * 1024 * 1024);
    RawGaussianParams params = owned.as_raw();
    float first_loss = 0, last_loss = 0;
    for (int i = 0; i < 1000; i++) {
        auto r = trainer.step(params, cam, gt_image.data(), cfg, tcfg, i);
        if (i == 0) first_loss = r.loss;
        last_loss = r.loss;
    }
    EXPECT_GT(first_loss, 0.001f);
    EXPECT_LT(last_loss / first_loss, 0.10f) << "Adam+SH3 should reduce loss by >90%";
}

TEST(TrainAdam, MultiGaussian10_Converges) {
    OwnedRawParams owned; Camera cam; RenderConfig cfg;
    std::vector<float> gt_image;
    make_synthetic_scene(owned, cam, cfg, gt_image, 10, 16, 16, 0);

    TrainConfig tcfg;
    tcfg.use_adam = true;
    tcfg.max_steps = 1000;

    Trainer trainer(64 * 1024 * 1024);
    RawGaussianParams params = owned.as_raw();
    float first_loss = 0, last_loss = 0;
    for (int i = 0; i < 1000; i++) {
        auto r = trainer.step(params, cam, gt_image.data(), cfg, tcfg, i);
        if (i == 0) first_loss = r.loss;
        last_loss = r.loss;
    }
    EXPECT_LT(last_loss / first_loss, 0.05f) << "Adam+10G should reduce loss by >95%";
}
