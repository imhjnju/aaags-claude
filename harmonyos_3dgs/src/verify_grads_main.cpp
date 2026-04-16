// verify_grads_main.cpp — Read scene data, run forward+backward, dump gradients
// Used by tools/verify_gradients.py to compare C++ vs Python gradients.
// Usage: ./gs3d_verify_grads <data_dir>

#include "trainer.h"
#include "train_types.h"
#include "loss.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#include "cpu/rasterizer_backward_cpu.h"
#include "cpu/preprocessor_backward_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// Simple JSON value reader
static double readJsonDouble(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos) + 1;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;
    return std::stod(json.substr(pos));
}

static std::vector<float> readBinFloats(const std::string& path, int count) {
    std::vector<float> data(count);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    fread(data.data(), sizeof(float), count, f);
    fclose(f);
    return data;
}

static void writeBinFloats(const std::string& path, const float* data, int count) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];

    // Read metadata
    std::ifstream mf(dir + "/meta.json");
    std::string meta_str((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    mf.close();

    int N = (int)readJsonDouble(meta_str, "N");
    int W = (int)readJsonDouble(meta_str, "W");
    int H = (int)readJsonDouble(meta_str, "H");
    int sh_degree = (int)readJsonDouble(meta_str, "sh_degree");
    int max_coeffs = (int)readJsonDouble(meta_str, "max_coeffs");
    float tan_fovx = (float)readJsonDouble(meta_str, "tan_fovx");
    float tan_fovy = (float)readJsonDouble(meta_str, "tan_fovy");

    printf("Scene: N=%d, %dx%d, SH=%d\n", N, W, H, sh_degree);

    // Read scene data
    auto raw_pos = readBinFloats(dir + "/raw_positions.bin", N * 3);
    auto raw_sc  = readBinFloats(dir + "/raw_scales.bin", N * 3);
    auto raw_rot = readBinFloats(dir + "/raw_rotations.bin", N * 4);
    auto raw_sh  = readBinFloats(dir + "/raw_sh_coeffs.bin", N * max_coeffs * 3);
    auto raw_op  = readBinFloats(dir + "/raw_opacities.bin", N);
    auto gt_img  = readBinFloats(dir + "/gt_image.bin", H * W * 3);
    auto view_m  = readBinFloats(dir + "/view_matrix.bin", 16);
    auto vp_m    = readBinFloats(dir + "/viewproj_matrix.bin", 16);

    // Setup RawGaussianParams
    RawGaussianParams raw;
    raw.count = N;
    raw.sh_degree = sh_degree;
    raw.max_coeffs = max_coeffs;
    raw.raw_positions = raw_pos.data();
    raw.raw_scales = raw_sc.data();
    raw.raw_rotations = raw_rot.data();
    raw.raw_sh_coeffs = raw_sh.data();
    raw.raw_opacities = raw_op.data();

    // Debug: print first few SH values to verify file read
    printf("DEBUG raw_sh[0..5]: ");
    for (int i = 0; i < std::min(6, (int)raw_sh.size()); i++)
        printf("%.6f ", raw_sh[i]);
    printf("\n");

    // Activate
    std::vector<float> act_pos(N*3), act_sc(N*3), act_rot(N*4), act_sh(N*max_coeffs*3), act_op(N);
    GaussianData g;
    g.count = N; g.sh_degree = sh_degree; g.max_coeffs = max_coeffs;
    g.positions = act_pos.data(); g.scales = act_sc.data();
    g.rotations = act_rot.data(); g.sh_coeffs = act_sh.data();
    g.opacities = act_op.data(); g.filter_3D = nullptr;
    raw.activate(g);

    // Camera
    Camera cam{};
    memcpy(cam.view_matrix, view_m.data(), 16 * sizeof(float));
    memcpy(cam.viewproj_matrix, vp_m.data(), 16 * sizeof(float));
    cam.cam_pos[0] = cam.cam_pos[1] = cam.cam_pos[2] = 0;
    cam.tan_fovx = tan_fovx;
    cam.tan_fovy = tan_fovy;
    cam.width = W;
    cam.height = H;

    // Render config
    RenderConfig cfg;
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.antialiasing = false;
    cfg.eval_3D = false;
    cfg.training = true;
    cfg.sh_degree = sh_degree;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    // Forward pass
    FrameAllocator alloc(128 * 1024 * 1024);
    PreprocessorCPU pp;
    TileBinnerCPU bn;
    SorterCPU sr;
    RasterizerCPU rs;

    ForwardCache cache;
    auto pre = pp.process(g, cam, cfg, alloc, &cache);
    auto bin = bn.bin(pre, N, cam, cfg, alloc);
    if (bin.total_pairs > 0) sr.sort(bin, alloc);
    cache.pre = &pre;
    cache.bin = &bin;

    int num_pixels = W * H;
    float* rendered = alloc.allocate_array<float>(num_pixels * 3);
    rs.rasterize(pre, bin, cam, cfg, rendered, nullptr, &cache, &alloc);

    // Loss
    float* d_image = alloc.allocate_array<float>(num_pixels * 3);
    // Use double-precision loss for FD gradient verification accuracy.
    // The float32 d_image gradient is still correct (same sign logic).
    double loss_d = l1_loss_double(rendered, gt_img.data(), H, W, d_image);
    float loss = (float)loss_d;
    printf("C++ loss: %.15e\n", loss_d);

    // Write loss as double for FD precision
    {
        FILE* f = fopen((dir + "/cpp_loss.bin").c_str(), "wb");
        if (f) { fwrite(&loss_d, sizeof(double), 1, f); fclose(f); }
    }

    // Dump rendered image for forward render verification
    writeBinFloats(dir + "/cpp_rendered.bin", rendered, num_pixels * 3);

    // Backward
    RasterGradOutput rgrad;
    rgrad.allocate_and_zero(alloc, N);
    RasterizerBackwardCPU rast_bw;
    rast_bw.backward(pre, bin, cam, cfg, cache, d_image, rgrad);

    GradientOutput grads;
    grads.allocate_and_zero(alloc, N, max_coeffs);
    PreprocessorBackwardCPU pp_bw;
    pp_bw.backward(g, cam, cfg, cache, rgrad, raw, grads);

    // Dump gradients
    writeBinFloats(dir + "/cpp_grad_raw_sh_coeffs.bin", grads.d_raw_sh_coeffs, N * max_coeffs * 3);
    writeBinFloats(dir + "/cpp_grad_raw_opacities.bin", grads.d_raw_opacities, N);
    writeBinFloats(dir + "/cpp_grad_raw_scales.bin", grads.d_raw_scales, N * 3);
    writeBinFloats(dir + "/cpp_grad_raw_positions.bin", grads.d_raw_positions, N * 3);
    writeBinFloats(dir + "/cpp_grad_raw_rotations.bin", grads.d_raw_rotations, N * 4);

    printf("Gradients written to %s/cpp_grad_*.bin\n", dir.c_str());

    // Print some gradient values for quick check
    printf("\nd_raw_sh_coeffs: ");
    for (int i = 0; i < std::min(6, N * max_coeffs * 3); i++)
        printf("%.6f ", grads.d_raw_sh_coeffs[i]);
    printf("\nd_raw_opacities: ");
    for (int i = 0; i < N; i++)
        printf("%.6f ", grads.d_raw_opacities[i]);
    printf("\nd_raw_scales: ");
    for (int i = 0; i < std::min(6, N * 3); i++)
        printf("%.6f ", grads.d_raw_scales[i]);
    printf("\nd_raw_rotations: ");
    for (int i = 0; i < std::min(8, N * 4); i++)
        printf("%.6f ", grads.d_raw_rotations[i]);
    printf("\nd_raw_positions: ");
    for (int i = 0; i < std::min(6, N * 3); i++)
        printf("%.6f ", grads.d_raw_positions[i]);
    printf("\n");

    return 0;
}
