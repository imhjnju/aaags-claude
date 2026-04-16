#include <gtest/gtest.h>
#include "trainer.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>

// Helper: build camera looking down +Z (same pattern as test_trainer_e2e.cpp)
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

// Helper: inverse sigmoid
static float inv_sigmoid(float y) {
    y = std::max(1e-6f, std::min(1.0f - 1e-6f, y));
    return std::log(y / (1.0f - y));
}

// ---------------------------------------------------------------------------
// Integration test: render GT from known Gaussians, perturb, train, verify
// loss decreases on a multi-Gaussian scene.
// ---------------------------------------------------------------------------
TEST(TrainPipeline, SyntheticScene_LossDecreases) {
    const int W = 32, H = 32;
    Camera cam = makeTestCamera(W, H);

    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = 0;
    cfg.tile_w = 16; cfg.tile_h = 16;

    // --- Create a known scene with 3 Gaussians at different positions ---
    const int N = 3;
    float positions[N * 3] = {
        -0.5f, 0.0f, 5.0f,   // left
         0.5f, 0.0f, 5.0f,   // right
         0.0f, 0.5f, 5.0f    // top
    };
    float scales[N * 3] = {
        0.2f, 0.2f, 0.2f,
        0.3f, 0.3f, 0.3f,
        0.25f, 0.25f, 0.25f
    };
    float rotations[N * 4] = {
        1,0,0,0,
        1,0,0,0,
        1,0,0,0
    };
    float sh_coeffs[N * 3] = {
        2.0f, 0.5f, 0.5f,   // reddish
        0.5f, 2.0f, 0.5f,   // greenish
        0.5f, 0.5f, 2.0f    // bluish
    };
    float opacities[N] = {0.9f, 0.85f, 0.8f};

    GaussianData g;
    g.count = N; g.sh_degree = 0; g.max_coeffs = 1;
    g.positions = positions; g.scales = scales; g.rotations = rotations;
    g.sh_coeffs = sh_coeffs; g.opacities = opacities; g.filter_3D = nullptr;

    // Render ground truth
    std::vector<float> gt(W * H * 3, 0.0f);
    {
        FrameAllocator tmp_alloc(128 * 1024 * 1024);
        PreprocessorCPU pp; TileBinnerCPU bn; SorterCPU sr; RasterizerCPU rs;
        auto pre = pp.process(g, cam, cfg, tmp_alloc);
        auto bin = bn.bin(pre, N, cam, cfg, tmp_alloc);
        if (bin.total_pairs > 0) sr.sort(bin, tmp_alloc);

        // Verify at least some Gaussians are visible
        int visible = 0;
        for (int i = 0; i < N; i++) if (pre.radii[i] > 0) visible++;
        ASSERT_GT(visible, 0) << "No Gaussians visible in GT render";

        rs.rasterize(pre, bin, cam, cfg, gt.data());

        float gt_sum = 0;
        for (auto v : gt) gt_sum += v;
        ASSERT_GT(gt_sum, 0.0f) << "GT image is all black";
        printf("  GT image: %d visible Gaussians, pixel sum=%.2f\n", visible, gt_sum);
    }

    // --- Initialize training params: same geometry, perturbed SH ---
    RawGaussianParams raw;
    raw.count = N; raw.sh_degree = 0; raw.max_coeffs = 1;

    // Positions: same as GT
    float raw_pos[N * 3];
    memcpy(raw_pos, positions, sizeof(raw_pos));
    raw.raw_positions = raw_pos;

    // Scales: log(activated)
    float raw_sc[N * 3];
    for (int i = 0; i < N * 3; i++) raw_sc[i] = std::log(scales[i]);
    raw.raw_scales = raw_sc;

    // Rotations: same
    float raw_rot[N * 4];
    memcpy(raw_rot, rotations, sizeof(raw_rot));
    raw.raw_rotations = raw_rot;

    // SH: perturbed (start at uniform gray instead of colored)
    float raw_sh[N * 3];
    for (int i = 0; i < N * 3; i++) raw_sh[i] = 1.0f;
    raw.raw_sh_coeffs = raw_sh;

    // Opacities: inverse sigmoid
    float raw_op[N];
    for (int i = 0; i < N; i++) raw_op[i] = inv_sigmoid(opacities[i]);
    raw.raw_opacities = raw_op;

    // --- Train ---
    TrainConfig tcfg;
    tcfg.lr_position_init  = 1e-6f;   // freeze geometry
    tcfg.lr_position_final = 1e-6f;
    tcfg.lr_feature        = 5.0f;    // learn color
    tcfg.lr_opacity        = 1e-4f;   // near-frozen
    tcfg.lr_scaling        = 1e-4f;
    tcfg.lr_rotation       = 1e-4f;
    tcfg.max_steps         = 300;

    Trainer trainer(256 * 1024 * 1024);

    float first_loss = -1, last_loss = -1;
    const int ITERS = 300;
    for (int iter = 0; iter < ITERS; iter++) {
        auto result = trainer.step(raw, cam, gt.data(), cfg, tcfg, iter);
        if (iter == 0) first_loss = result.loss;
        last_loss = result.loss;
        if (iter < 5 || iter % 50 == 0 || iter == ITERS - 1) {
            printf("  iter %3d: loss=%.6f\n", iter, result.loss);
        }
    }

    printf("  first_loss=%.6f  last_loss=%.6f  ratio=%.4f\n",
           first_loss, last_loss, last_loss / first_loss);

    EXPECT_LT(last_loss, first_loss * 0.5f)
        << "Loss should decrease by at least 50% after " << ITERS << " iterations";
    EXPECT_LT(last_loss, first_loss * 0.1f)
        << "Loss should decrease by at least 90% after " << ITERS << " iterations";
}

// ---------------------------------------------------------------------------
// Test: initRawFromGaussianData roundtrip (activate -> invert -> activate)
// ---------------------------------------------------------------------------
TEST(TrainPipeline, RawParamsRoundtrip) {
    // Create activated data
    const int N = 2;
    float pos[N * 3] = {1, 2, 3, 4, 5, 6};
    float sc[N * 3]  = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float rot[N * 4] = {1, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f};
    float sh[N * 3]  = {1.5f, 2.0f, 0.5f, -0.3f, 0.8f, 1.2f};
    float op[N]      = {0.9f, 0.3f};

    GaussianData g;
    g.count = N; g.sh_degree = 0; g.max_coeffs = 1;
    g.positions = pos; g.scales = sc; g.rotations = rot;
    g.sh_coeffs = sh; g.opacities = op; g.filter_3D = nullptr;

    // Build raw params from activated data (mimics initRawFromGaussianData)
    RawGaussianParams raw;
    raw.count = N; raw.sh_degree = 0; raw.max_coeffs = 1;
    raw.raw_positions  = new float[N * 3];
    raw.raw_scales     = new float[N * 3];
    raw.raw_rotations  = new float[N * 4];
    raw.raw_sh_coeffs  = new float[N * 3];
    raw.raw_opacities  = new float[N];

    memcpy(raw.raw_positions, pos, sizeof(pos));
    for (int i = 0; i < N * 3; i++) raw.raw_scales[i] = std::log(std::max(1e-12f, sc[i]));
    memcpy(raw.raw_rotations, rot, sizeof(rot));
    memcpy(raw.raw_sh_coeffs, sh, sizeof(sh));
    for (int i = 0; i < N; i++) raw.raw_opacities[i] = inv_sigmoid(op[i]);

    // Activate back
    float out_pos[N * 3], out_sc[N * 3], out_rot[N * 4], out_sh[N * 3], out_op[N];
    GaussianData out;
    out.count = N; out.sh_degree = 0; out.max_coeffs = 1;
    out.positions = out_pos; out.scales = out_sc; out.rotations = out_rot;
    out.sh_coeffs = out_sh; out.opacities = out_op; out.filter_3D = nullptr;
    raw.activate(out);

    // Check roundtrip
    for (int i = 0; i < N * 3; i++)
        EXPECT_NEAR(out.positions[i], pos[i], 1e-5f) << "position[" << i << "]";
    for (int i = 0; i < N * 3; i++)
        EXPECT_NEAR(out.scales[i], sc[i], 1e-4f) << "scale[" << i << "]";
    for (int i = 0; i < N * 3; i++)
        EXPECT_NEAR(out.sh_coeffs[i], sh[i], 1e-5f) << "sh[" << i << "]";
    for (int i = 0; i < N; i++)
        EXPECT_NEAR(out.opacities[i], op[i], 1e-4f) << "opacity[" << i << "]";

    // Rotations should be normalized versions of input
    for (int i = 0; i < N; i++) {
        float len = std::sqrt(rot[i*4]*rot[i*4] + rot[i*4+1]*rot[i*4+1] +
                              rot[i*4+2]*rot[i*4+2] + rot[i*4+3]*rot[i*4+3]);
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(out.rotations[i*4+j], rot[i*4+j]/len, 1e-5f);
    }

    delete[] raw.raw_positions;
    delete[] raw.raw_scales;
    delete[] raw.raw_rotations;
    delete[] raw.raw_sh_coeffs;
    delete[] raw.raw_opacities;
}
