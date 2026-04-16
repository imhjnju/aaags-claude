// train_compare_cpp.cpp — Configurable training loop for convergence verification.
// Supports variable N, SH degree, iteration count, and random seed.
// Outputs CSV: iteration,loss
//
// Usage:
//   ./gs3d_train_compare [--iters 500] [--N 10] [--sh 1] [--seed 42] [--res 16]

#include "trainer.h"
#include "train_types.h"
#include "loss.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// Simple PRNG (same as numpy RandomState with seed, approximately)
struct SimpleRNG {
    uint64_t state;
    SimpleRNG(uint64_t seed) : state(seed * 6364136223846793005ULL + 1) {}
    uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(state >> 33);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() & 0xFFFFFF) / (float)0xFFFFFF);
    }
    float randn() {
        // Box-Muller
        float u1 = uniform(1e-7f, 1.0f);
        float u2 = uniform(0.0f, 1.0f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }
};

static int parse_int_arg(int argc, char** argv, const char* name, int default_val) {
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == name) return std::atoi(argv[i+1]);
    return default_val;
}

static bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == name) return true;
    return false;
}

int main(int argc, char** argv) {
    int N = parse_int_arg(argc, argv, "--N", 3);
    int sh_degree = parse_int_arg(argc, argv, "--sh", 0);
    int num_iters = parse_int_arg(argc, argv, "--iters", 100);
    int seed = parse_int_arg(argc, argv, "--seed", 42);
    int W = parse_int_arg(argc, argv, "--res", 16);
    int H = W;
    bool use_densify = has_flag(argc, argv, "--densify");
    int max_coeffs = (sh_degree + 1) * (sh_degree + 1);

    fprintf(stderr, "Config: N=%d, SH=%d, iters=%d, seed=%d, %dx%d\n",
            N, sh_degree, num_iters, seed, W, H);

    // Generate scene deterministically
    SimpleRNG rng(seed);

    std::vector<float> raw_pos(N * 3);
    std::vector<float> raw_sc(N * 3);
    std::vector<float> raw_rot(N * 4);
    std::vector<float> raw_sh(N * max_coeffs * 3);
    std::vector<float> raw_op(N);

    for (int i = 0; i < N; i++) {
        raw_pos[i*3+0] = rng.uniform(-1.5f, 1.5f);
        raw_pos[i*3+1] = rng.uniform(-1.5f, 1.5f);
        raw_pos[i*3+2] = rng.uniform(3.0f, 8.0f);
        raw_sc[i*3+0] = std::log(rng.uniform(0.1f, 0.6f));
        raw_sc[i*3+1] = std::log(rng.uniform(0.1f, 0.6f));
        raw_sc[i*3+2] = std::log(rng.uniform(0.1f, 0.6f));
        raw_rot[i*4+0] = rng.randn() + 1.0f;  // bias toward identity
        raw_rot[i*4+1] = rng.randn();
        raw_rot[i*4+2] = rng.randn();
        raw_rot[i*4+3] = rng.randn();
        for (int j = 0; j < max_coeffs * 3; j++)
            raw_sh[i * max_coeffs * 3 + j] = rng.uniform(-1.0f, 2.0f);
        raw_op[i] = rng.uniform(-1.0f, 3.0f);
    }

    // Camera: identity view, 60 deg FOV
    float fov = 60.0f;
    float tan_fov = std::tan(fov * 0.5f * 3.14159265358979323846f / 180.0f);

    Camera cam{};
    std::memset(cam.view_matrix, 0, sizeof(cam.view_matrix));
    cam.view_matrix[0] = 1.0f; cam.view_matrix[5] = 1.0f;
    cam.view_matrix[10] = 1.0f; cam.view_matrix[15] = 1.0f;

    float near_val = 0.01f, far_val = 100.0f;
    float r = tan_fov * near_val;
    std::memset(cam.viewproj_matrix, 0, sizeof(cam.viewproj_matrix));
    cam.viewproj_matrix[0] = near_val / r;
    cam.viewproj_matrix[5] = near_val / r;
    cam.viewproj_matrix[10] = far_val / (far_val - near_val);
    cam.viewproj_matrix[11] = 1.0f;
    cam.viewproj_matrix[14] = -(far_val * near_val) / (far_val - near_val);

    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0.0f;
    cam.tan_fovx = tan_fov;
    cam.tan_fovy = tan_fov;
    cam.width = W; cam.height = H;

    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = sh_degree;
    cfg.tile_w = 16; cfg.tile_h = 16;

    // GT: same scene but SH DC coeffs += 0.5
    std::vector<float> gt_sh = raw_sh;
    for (int i = 0; i < N; i++)
        for (int ch = 0; ch < 3; ch++)
            gt_sh[i * max_coeffs * 3 + ch] += 0.5f;

    // Render GT
    std::vector<float> gt_image(W * H * 3);
    {
        RawGaussianParams gt_raw;
        gt_raw.count = N; gt_raw.sh_degree = sh_degree; gt_raw.max_coeffs = max_coeffs;
        gt_raw.raw_positions = raw_pos.data();
        gt_raw.raw_scales = raw_sc.data();
        gt_raw.raw_rotations = raw_rot.data();
        gt_raw.raw_sh_coeffs = gt_sh.data();
        gt_raw.raw_opacities = raw_op.data();

        std::vector<float> a_p(N*3), a_s(N*3), a_r(N*4), a_sh(N*max_coeffs*3), a_o(N);
        GaussianData g;
        g.count = N; g.sh_degree = sh_degree; g.max_coeffs = max_coeffs;
        g.positions = a_p.data(); g.scales = a_s.data();
        g.rotations = a_r.data(); g.sh_coeffs = a_sh.data();
        g.opacities = a_o.data(); g.filter_3D = nullptr;
        gt_raw.activate(g);

        FrameAllocator alloc(128 * 1024 * 1024);
        PreprocessorCPU pp; TileBinnerCPU bn; SorterCPU sr; RasterizerCPU rs;
        ForwardCache cache{};
        auto pre = pp.process(g, cam, cfg, alloc, &cache);
        auto bin = bn.bin(pre, N, cam, cfg, alloc);
        if (bin.total_pairs > 0) sr.sort(bin, alloc);
        cache.pre = &pre; cache.bin = &bin;
        float* gt_rendered = alloc.allocate_array<float>(W * H * 3);
        rs.rasterize(pre, bin, cam, cfg, gt_rendered, nullptr, &cache, &alloc);
        std::memcpy(gt_image.data(), gt_rendered, W * H * 3 * sizeof(float));
    }

    // Training config
    TrainConfig train_cfg;
    train_cfg.lr_position_init = 0.001f;
    train_cfg.lr_position_final = 0.001f;
    train_cfg.lr_feature = 0.01f;
    train_cfg.lr_opacity = 0.01f;
    train_cfg.lr_scaling = 0.005f;
    train_cfg.lr_rotation = 0.001f;
    train_cfg.max_steps = num_iters;
    train_cfg.use_adam = true;

    // Training params (owned for densification)
    OwnedRawParams owned;
    owned.sh_degree = sh_degree;
    owned.max_coeffs = max_coeffs;
    owned.positions.assign(raw_pos.begin(), raw_pos.end());
    owned.scales.assign(raw_sc.begin(), raw_sc.end());
    owned.rotations.assign(raw_rot.begin(), raw_rot.end());
    owned.sh_coeffs.assign(raw_sh.begin(), raw_sh.end());
    owned.opacities.assign(raw_op.begin(), raw_op.end());

    // Non-owning view (for non-densify path)
    RawGaussianParams params = owned.as_raw();

    // Densification config
    DensifyConfig dcfg;
    dcfg.enabled = use_densify;
    dcfg.densify_from_iter = std::min(100, num_iters / 5);
    dcfg.densify_until_iter = num_iters * 3 / 4;
    dcfg.densify_interval = 50;
    dcfg.grad_threshold = 0.0002f;
    dcfg.min_opacity = 0.005f;
    dcfg.opacity_reset_interval = 500;

    // Compute scene extent from Gaussian positions
    float scene_extent = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            scene_extent = std::max(scene_extent, std::abs(raw_pos[i*3+j]));
        }
    }
    scene_extent = std::max(scene_extent * 2.0f, 1.0f);

    Trainer trainer(128 * 1024 * 1024);
    printf("iteration,loss,n_gaussians\n");
    float first_loss = 0, last_loss = 0;
    int last_N = N;
    for (int i = 0; i < num_iters; i++) {
        Trainer::StepResult result;
        if (use_densify) {
            result = trainer.step_with_densify(owned, cam, gt_image.data(), cfg,
                                                train_cfg, dcfg, scene_extent, i);
            params = owned.as_raw();
            last_N = owned.count();
        } else {
            result = trainer.step(params, cam, gt_image.data(), cfg, train_cfg, i);
            last_N = N;
        }
        printf("%d,%.8f,%d\n", i, result.loss, last_N);
        if (i == 0) first_loss = result.loss;
        last_loss = result.loss;
    }

    fprintf(stderr, "Loss: %.8f -> %.8f (ratio=%.6f)\n",
            first_loss, last_loss, last_loss / first_loss);

    return 0;
}
