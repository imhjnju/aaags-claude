// train_demo.cpp — Synthetic training demo with loss curve output
// Usage: ./gs3d_train_demo [iterations] [num_gaussians]
// Outputs loss curve to stdout (CSV format for plotting)

#include "trainer.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "image_io.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>

static Camera makeCamera(int w, int h, float fov_deg = 60.0f) {
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

// Render a ground truth image from known Gaussians
static std::vector<float> renderGroundTruth(
    int N, const float* positions, const float* scales, const float* rotations,
    const float* sh_coeffs, const float* opacities, int sh_degree,
    const Camera& cam, const RenderConfig& cfg)
{
    GaussianData g;
    g.count = N;
    g.sh_degree = sh_degree;
    g.max_coeffs = (sh_degree + 1) * (sh_degree + 1);
    g.positions = const_cast<float*>(positions);
    g.scales = const_cast<float*>(scales);
    g.rotations = const_cast<float*>(rotations);
    g.sh_coeffs = const_cast<float*>(sh_coeffs);
    g.opacities = const_cast<float*>(opacities);
    g.filter_3D = nullptr;

    FrameAllocator alloc(128 * 1024 * 1024);
    PreprocessorCPU pp;
    TileBinnerCPU bn;
    SorterCPU sr;
    RasterizerCPU rs;

    auto pre = pp.process(g, cam, cfg, alloc);
    auto bin = bn.bin(pre, N, cam, cfg, alloc);
    if (bin.total_pairs > 0) sr.sort(bin, alloc);

    std::vector<float> image(cam.width * cam.height * 3, 0.0f);
    rs.rasterize(pre, bin, cam, cfg, image.data());
    return image;
}

int main(int argc, char** argv) {
    int iterations = 1000;
    int num_gaussians = 5;
    int W = 64, H = 64;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) num_gaussians = atoi(argv[2]);
    if (argc > 3) W = H = atoi(argv[3]);

    fprintf(stderr, "=== 3DGS Training Demo ===\n");
    fprintf(stderr, "Image: %dx%d, Gaussians: %d, Iterations: %d\n", W, H, num_gaussians, iterations);

    Camera cam = makeCamera(W, H, 60.0f);
    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = 0;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // === Create ground truth scene ===
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> z_dist(4.0f, 8.0f);
    std::uniform_real_distribution<float> scale_dist(0.1f, 0.5f);
    std::uniform_real_distribution<float> sh_dist(0.5f, 3.0f);

    int mc = 1; // max_coeffs for degree 0
    std::vector<float> gt_pos(num_gaussians * 3);
    std::vector<float> gt_scales(num_gaussians * 3);
    std::vector<float> gt_rot(num_gaussians * 4);
    std::vector<float> gt_sh(num_gaussians * mc * 3);
    std::vector<float> gt_op(num_gaussians);

    for (int i = 0; i < num_gaussians; i++) {
        gt_pos[i*3]   = pos_dist(rng);
        gt_pos[i*3+1] = pos_dist(rng);
        gt_pos[i*3+2] = z_dist(rng);
        float s = scale_dist(rng);
        gt_scales[i*3] = gt_scales[i*3+1] = gt_scales[i*3+2] = s;
        gt_rot[i*4] = 1; gt_rot[i*4+1] = 0; gt_rot[i*4+2] = 0; gt_rot[i*4+3] = 0;
        gt_sh[i*3]   = sh_dist(rng);
        gt_sh[i*3+1] = sh_dist(rng);
        gt_sh[i*3+2] = sh_dist(rng);
        gt_op[i] = 0.85f + 0.1f * (float)(rng() % 100) / 100.0f;
    }

    auto gt_image = renderGroundTruth(
        num_gaussians, gt_pos.data(), gt_scales.data(), gt_rot.data(),
        gt_sh.data(), gt_op.data(), 0, cam, cfg);

    float gt_sum = 0;
    for (auto v : gt_image) gt_sum += v;
    fprintf(stderr, "Ground truth image sum: %.2f (non-zero pixels contributing)\n", gt_sum);

    // Save ground truth image
    writePPM("gt_image.ppm", gt_image.data(), W, H);
    fprintf(stderr, "Saved ground truth to gt_image.ppm\n");

    // === Initialize training params with PERTURBED values ===
    RawGaussianParams raw;
    raw.count = num_gaussians;
    raw.sh_degree = 0;
    raw.max_coeffs = mc;

    std::vector<float> train_pos(num_gaussians * 3);
    std::vector<float> train_scales(num_gaussians * 3);
    std::vector<float> train_rot(num_gaussians * 4);
    std::vector<float> train_sh(num_gaussians * mc * 3);
    std::vector<float> train_op(num_gaussians);

    // Start from GT positions/scales/rotations/opacities but WRONG colors
    std::normal_distribution<float> noise(0.0f, 0.3f);
    for (int i = 0; i < num_gaussians * 3; i++)
        train_pos[i] = gt_pos[i]; // same position
    for (int i = 0; i < num_gaussians * 3; i++)
        train_scales[i] = logf(gt_scales[i]); // log-space
    for (int i = 0; i < num_gaussians * 4; i++)
        train_rot[i] = gt_rot[i];
    for (int i = 0; i < num_gaussians * mc * 3; i++)
        train_sh[i] = gt_sh[i] + noise(rng) * 2.0f; // perturbed SH
    for (int i = 0; i < num_gaussians; i++) {
        float s = gt_op[i];
        train_op[i] = logf(s / (1.0f - s)); // inverse sigmoid
    }

    raw.raw_positions = train_pos.data();
    raw.raw_scales = train_scales.data();
    raw.raw_rotations = train_rot.data();
    raw.raw_sh_coeffs = train_sh.data();
    raw.raw_opacities = train_op.data();

    // === Training config ===
    TrainConfig tcfg;
    tcfg.lr_position_init  = 1e-4f;
    tcfg.lr_position_final = 1e-5f;
    tcfg.lr_feature        = 5.0f;     // SH learning rate (main thing to learn)
    tcfg.lr_opacity        = 0.01f;
    tcfg.lr_scaling        = 0.001f;
    tcfg.lr_rotation       = 0.001f;
    tcfg.max_steps         = iterations;

    // === Train ===
    Trainer trainer(512 * 1024 * 1024); // 512MB arena

    // CSV header
    printf("iteration,loss\n");

    fprintf(stderr, "\n--- Training ---\n");
    float first_loss = 0, last_loss = 0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int iter = 0; iter < iterations; iter++) {
        auto result = trainer.step(raw, cam, gt_image.data(), cfg, tcfg, iter);

        if (iter == 0) first_loss = result.loss;
        last_loss = result.loss;

        // CSV output every iteration
        printf("%d,%.8f\n", iter, result.loss);

        // Status to stderr
        if (iter < 10 || iter % (iterations / 20 + 1) == 0 || iter == iterations - 1) {
            fprintf(stderr, "  iter %5d/%d: loss = %.8f\n", iter, iterations, result.loss);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    fprintf(stderr, "\n--- Results ---\n");
    fprintf(stderr, "First loss:  %.8f\n", first_loss);
    fprintf(stderr, "Final loss:  %.8f\n", last_loss);
    fprintf(stderr, "Reduction:   %.2f%%\n", (1.0f - last_loss / first_loss) * 100.0f);
    fprintf(stderr, "Time:        %.2f seconds (%.1f it/s)\n", elapsed, iterations / elapsed);

    // Render final image
    GaussianData g_final;
    g_final.count = num_gaussians;
    g_final.sh_degree = 0;
    g_final.max_coeffs = mc;
    std::vector<float> fpos(num_gaussians*3), fsc(num_gaussians*3), frot(num_gaussians*4);
    std::vector<float> fsh(num_gaussians*mc*3), fop(num_gaussians);
    g_final.positions = fpos.data();
    g_final.scales = fsc.data();
    g_final.rotations = frot.data();
    g_final.sh_coeffs = fsh.data();
    g_final.opacities = fop.data();
    g_final.filter_3D = nullptr;
    raw.activate(g_final);

    auto final_image = renderGroundTruth(
        num_gaussians, fpos.data(), fsc.data(), frot.data(),
        fsh.data(), fop.data(), 0, cam, cfg);

    writePPM("trained_image.ppm", final_image.data(), W, H);
    fprintf(stderr, "Saved trained result to trained_image.ppm\n");
    fprintf(stderr, "Loss curve saved to stdout (CSV format)\n");

    return 0;
}
