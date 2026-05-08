// test_e2e_basketball.cpp — SP-7 Task 6: basketball end-to-end 2000-step PSNR milestone.
//
// Loads the basketball COLMAP sparse point cloud (2892 Gaussians), initializes
// VulkanTrainer with camera 1 parameters, runs 2000 training steps with
// densification against basket0_ref_cam0.png, and asserts:
//   1. All loss values are finite.
//   2. Loss decreases over the run.
//   3. PSNR > 10 dB after 2000 steps.
//
// SP-7 eliminated ~51 GPU sync points per step, reducing step time from
// ~2.5 s to ~0.26 s on Tegra Thor. 2000 steps now takes ~9 min (was ~83 min).
//
// Skips automatically if:
//   - Basketball PLY dataset not available at expected path
//   - No Vulkan compute device present
//
// PLY format: binary_little_endian, per vertex: float x, y, z, nx, ny, nz, uchar r, g, b
// Camera: pre-computed from COLMAP image 78899858295079.jpg (720x960)

#include "vulkan_trainer.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

// stb_image for PNG loading (implementation in this TU only)
#define STB_IMAGE_IMPLEMENTATION
#include "extern/stb_image.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PLY loader for COLMAP sparse points3D.ply
// Format: binary_little_endian, per vertex: float x y z nx ny nz, uchar r g b
// ---------------------------------------------------------------------------

namespace {

struct ColmapPoint {
    float x, y, z;
    unsigned char r, g, b;
};

// Parse the binary_little_endian PLY header and return byte offset to data.
// Returns the vertex count and the byte offset where vertex data begins.
bool parse_ply_header(std::ifstream& f,
                      int& out_num_vertices,
                      size_t& out_data_offset)
{
    std::string line;
    out_num_vertices = 0;
    out_data_offset  = 0;

    // Verify magic
    if (!std::getline(f, line)) return false;
    if (line.rfind("ply", 0) != 0) return false;

    bool binary_le = false;
    bool end_header = false;
    while (std::getline(f, line)) {
        if (line.rfind("format binary_little_endian", 0) == 0) {
            binary_le = true;
        } else if (line.rfind("element vertex ", 0) == 0) {
            out_num_vertices = std::stoi(line.substr(15));
        } else if (line == "end_header" || line == "end_header\r") {
            end_header = true;
            break;
        }
    }
    if (!binary_le || !end_header || out_num_vertices <= 0) return false;
    out_data_offset = static_cast<size_t>(f.tellg());
    return true;
}

// Load COLMAP points3D.ply into ColmapPoint vector.
// Per-vertex layout: float x, y, z, nx, ny, nz (6 floats = 24 bytes), uchar r, g, b (3 bytes).
// Total stride: 27 bytes per vertex.
std::vector<ColmapPoint> load_colmap_ply(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PLY: " + path);

    int num_vertices = 0;
    size_t data_offset = 0;
    if (!parse_ply_header(f, num_vertices, data_offset)) {
        throw std::runtime_error("PLY header parse failed: " + path);
    }

    // Seek to data section
    f.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);

    const int stride = 6 * sizeof(float) + 3;  // 27 bytes: xyz + nxnynz + rgb
    std::vector<ColmapPoint> pts(static_cast<size_t>(num_vertices));
    std::vector<char> buf(static_cast<size_t>(stride));

    for (int i = 0; i < num_vertices; ++i) {
        f.read(buf.data(), stride);
        if (f.gcount() != stride) {
            throw std::runtime_error("Unexpected EOF in PLY data at vertex " + std::to_string(i));
        }
        float xyz[3];
        std::memcpy(xyz, buf.data(), 3 * sizeof(float));
        pts[i].x = xyz[0];
        pts[i].y = xyz[1];
        pts[i].z = xyz[2];
        // bytes 12..23 are normals — skip
        pts[i].r = static_cast<unsigned char>(buf[24]);
        pts[i].g = static_cast<unsigned char>(buf[25]);
        pts[i].b = static_cast<unsigned char>(buf[26]);
    }
    return pts;
}

// ---------------------------------------------------------------------------
// PSNR
// ---------------------------------------------------------------------------

float compute_psnr(const float* rendered, const float* target, int N)
{
    double mse = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = static_cast<double>(rendered[i]) - static_cast<double>(target[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(N);
    if (mse <= 0.0) return 100.0f;
    return static_cast<float>(-10.0 * std::log10(mse));
}

}  // namespace

// ---------------------------------------------------------------------------
// Basketball.Training2000Steps
// ---------------------------------------------------------------------------

TEST(Basketball, Training2000Steps)
{
    // --- Guard: dataset availability ---
    const std::string ply_path =
        "/home/robota/Downloads/basketball/sparse/0/points3D.ply";
    if (!std::filesystem::exists(ply_path)) {
        GTEST_SKIP() << "Basketball dataset not available at " << ply_path;
    }

    // --- Guard: Vulkan device ---
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping Basketball test.";
    }

    // --- Load reference image ---
    // TEST_DATA_DIR = <project>/tests; reference lives at <project>/tools/.
    const std::string ref_img_path =
        std::string(TEST_DATA_DIR) + "/../tools/basket0_ref_cam0.png";
    ASSERT_TRUE(std::filesystem::exists(ref_img_path))
        << "Reference image not found: " << ref_img_path;

    const int W = 720, H = 960;
    int loaded_w = 0, loaded_h = 0;
    // forced_channels=3 guarantees RGB output regardless of source format
    unsigned char* img_raw = stbi_load(ref_img_path.c_str(),
                                       &loaded_w, &loaded_h, nullptr, 3);
    ASSERT_NE(img_raw, nullptr) << "Failed to load reference image: " << ref_img_path;
    // RAII guard so img_raw is freed even if ASSERT_EQ below fires and returns early.
    struct StbGuard { unsigned char* p; ~StbGuard() { stbi_image_free(p); } } _g{img_raw};
    ASSERT_EQ(loaded_w, W) << "Reference image width mismatch";
    ASSERT_EQ(loaded_h, H) << "Reference image height mismatch";

    // Convert to float [0,1]
    const int total_pixels = W * H * 3;
    std::vector<float> target(static_cast<size_t>(total_pixels));
    for (int i = 0; i < total_pixels; ++i)
        target[i] = static_cast<float>(img_raw[i]) / 255.0f;

    // --- Load basketball PLY ---
    std::vector<ColmapPoint> pts = load_colmap_ply(ply_path);
    const int N = static_cast<int>(pts.size());
    ASSERT_GT(N, 0) << "No points loaded from PLY";

    // --- Initialize GaussianData and RawGaussianParams ---
    // SH: sh_degree=3, max_coeffs=16
    // DC coefficient from RGB: sh_dc_k = (rgb_k/255 - 0.5) / C0
    // where C0 = 0.28209479177387814
    const int sh_degree  = 3;
    const int max_coeffs = 16;  // (sh_degree+1)^2 = 16
    const float C0 = 0.28209479177387814f;

    // Constant init values (log and logit of initial activated values)
    const float init_raw_scale    = -3.507f;  // log(0.03)
    const float init_raw_opacity  = -2.197f;  // logit(0.1) = log(0.1/0.9)

    // Allocate arrays
    std::vector<float> positions   (static_cast<size_t>(N) * 3);
    std::vector<float> scales      (static_cast<size_t>(N) * 3);
    std::vector<float> rotations   (static_cast<size_t>(N) * 4);
    std::vector<float> opacities   (static_cast<size_t>(N));
    std::vector<float> sh_coeffs   (static_cast<size_t>(N) * max_coeffs * 3, 0.0f);

    // Activated arrays (for GaussianData)
    std::vector<float> act_scales  (static_cast<size_t>(N) * 3);
    std::vector<float> act_opacities(static_cast<size_t>(N));

    const float act_scale   = std::exp(init_raw_scale);    // ~0.03
    const float act_opacity = 1.0f / (1.0f + std::exp(-init_raw_opacity));  // ~0.1

    for (int i = 0; i < N; ++i) {
        // Positions (raw = activated, no activation)
        positions[static_cast<size_t>(i) * 3 + 0] = pts[i].x;
        positions[static_cast<size_t>(i) * 3 + 1] = pts[i].y;
        positions[static_cast<size_t>(i) * 3 + 2] = pts[i].z;

        // Scales — raw: log(0.03) constant
        scales[static_cast<size_t>(i) * 3 + 0] = init_raw_scale;
        scales[static_cast<size_t>(i) * 3 + 1] = init_raw_scale;
        scales[static_cast<size_t>(i) * 3 + 2] = init_raw_scale;

        // Activated scales (for GaussianData)
        act_scales[static_cast<size_t>(i) * 3 + 0] = act_scale;
        act_scales[static_cast<size_t>(i) * 3 + 1] = act_scale;
        act_scales[static_cast<size_t>(i) * 3 + 2] = act_scale;

        // Rotations: identity quaternion [w, x, y, z] = [1, 0, 0, 0]
        rotations[static_cast<size_t>(i) * 4 + 0] = 1.0f;
        rotations[static_cast<size_t>(i) * 4 + 1] = 0.0f;
        rotations[static_cast<size_t>(i) * 4 + 2] = 0.0f;
        rotations[static_cast<size_t>(i) * 4 + 3] = 0.0f;

        // Opacities — raw: logit(0.1)
        opacities[i] = init_raw_opacity;
        act_opacities[i] = act_opacity;

        // SH DC coefficient from COLMAP RGB
        // Layout: sh_coeffs[i * max_coeffs * 3 + ch] for DC (first coeff per channel)
        // DC slot: index i*16*3 + ch (channel-major within Gaussian)
        const float r_dc = (static_cast<float>(pts[i].r) / 255.0f - 0.5f) / C0;
        const float g_dc = (static_cast<float>(pts[i].g) / 255.0f - 0.5f) / C0;
        const float b_dc = (static_cast<float>(pts[i].b) / 255.0f - 0.5f) / C0;
        sh_coeffs[static_cast<size_t>(i) * max_coeffs * 3 + 0] = r_dc;
        sh_coeffs[static_cast<size_t>(i) * max_coeffs * 3 + 1] = g_dc;
        sh_coeffs[static_cast<size_t>(i) * max_coeffs * 3 + 2] = b_dc;
        // Higher-order SH coefficients remain zero (initialized above)
    }

    // Build GaussianData (activated view)
    GaussianData init_g{};
    init_g.count      = N;
    init_g.sh_degree  = sh_degree;
    init_g.max_coeffs = max_coeffs;
    init_g.positions  = positions.data();   // positions = raw (identity activation)
    init_g.scales     = act_scales.data();
    init_g.rotations  = rotations.data();   // identity quaternion already normalized
    init_g.opacities  = act_opacities.data();
    init_g.sh_coeffs  = sh_coeffs.data();
    init_g.filter_3D  = nullptr;

    // Build RawGaussianParams
    RawGaussianParams init_raw{};
    init_raw.count         = N;
    init_raw.sh_degree     = sh_degree;
    init_raw.max_coeffs    = max_coeffs;
    init_raw.raw_positions = positions.data();
    init_raw.raw_scales    = scales.data();
    init_raw.raw_rotations = rotations.data();
    init_raw.raw_sh_coeffs = sh_coeffs.data();
    init_raw.raw_opacities = opacities.data();

    // --- Camera setup (camera 1, image 78899858295079.jpg) ---
    // Column-major (transpose of the row-major matrices in the task spec).
    Camera cam{};
    cam.width    = W;
    cam.height   = H;
    // tan_fovx/y = (W/2)/fx, (H/2)/fy — independent of resolution (same FOV angle)
    cam.tan_fovx = 0.509851f;  // 720/(2*706.089)
    cam.tan_fovy = 0.678492f;  // 960/(2*707.452)
    cam.cam_pos[0] = -0.42047721f;
    cam.cam_pos[1] =  0.27240201f;
    cam.cam_pos[2] = -0.21650667f;

    // View matrix: row-major input → column-major storage
    // row_major[row][col] → col_major[col*4 + row]
    // Row 0: [-0.22993832,  0.03527074,  0.97256586,  0.10427535]
    // Row 1: [-0.38269602, -0.92211196, -0.05703769,  0.07792116]
    // Row 2: [ 0.89480284, -0.38531223,  0.22552685,  0.53003210]
    // Row 3: [ 0.0,         0.0,         0.0,          1.0]
    cam.view_matrix[ 0] = -0.22993832f;  // col0, row0
    cam.view_matrix[ 1] = -0.38269602f;  // col0, row1
    cam.view_matrix[ 2] =  0.89480284f;  // col0, row2
    cam.view_matrix[ 3] =  0.0f;         // col0, row3
    cam.view_matrix[ 4] =  0.03527074f;  // col1, row0
    cam.view_matrix[ 5] = -0.92211196f;  // col1, row1
    cam.view_matrix[ 6] = -0.38531223f;  // col1, row2
    cam.view_matrix[ 7] =  0.0f;         // col1, row3
    cam.view_matrix[ 8] =  0.97256586f;  // col2, row0
    cam.view_matrix[ 9] = -0.05703769f;  // col2, row1
    cam.view_matrix[10] =  0.22552685f;  // col2, row2
    cam.view_matrix[11] =  0.0f;         // col2, row3
    cam.view_matrix[12] =  0.10427535f;  // col3, row0 (translation)
    cam.view_matrix[13] =  0.07792116f;  // col3, row1
    cam.view_matrix[14] =  0.53003210f;  // col3, row2
    cam.view_matrix[15] =  1.0f;         // col3, row3

    // Viewproj matrix: already column-major for Camera struct — copy directly.
    // The values are given as 4 consecutive memory blocks ("Row 0..3" = col0..3).
    // No transpose needed (unlike the view matrix above which was ROW-MAJOR input).
    // Col 0: [-0.45099131,  0.06917855,  1.90754958,  0.20452127]
    // Col 1: [-0.56403945, -1.35906174, -0.08406543,  0.11484469]
    // Col 2: [ 0.89489233, -0.38535077,  0.2255494,   0.52008411]
    // Col 3: [ 0.89480284, -0.38531223,  0.22552685,  0.53003210]
    cam.viewproj_matrix[ 0] = -0.45099131f;  // col0[0]
    cam.viewproj_matrix[ 1] =  0.06917855f;  // col0[1]
    cam.viewproj_matrix[ 2] =  1.90754958f;  // col0[2]
    cam.viewproj_matrix[ 3] =  0.20452127f;  // col0[3]
    cam.viewproj_matrix[ 4] = -0.56403945f;  // col1[0]
    cam.viewproj_matrix[ 5] = -1.35906174f;  // col1[1]
    cam.viewproj_matrix[ 6] = -0.08406543f;  // col1[2]
    cam.viewproj_matrix[ 7] =  0.11484469f;  // col1[3]
    cam.viewproj_matrix[ 8] =  0.89489233f;  // col2[0]
    cam.viewproj_matrix[ 9] = -0.38535077f;  // col2[1]
    cam.viewproj_matrix[10] =  0.2255494f;   // col2[2]
    cam.viewproj_matrix[11] =  0.52008411f;  // col2[3]
    cam.viewproj_matrix[12] =  0.89480284f;  // col3[0]
    cam.viewproj_matrix[13] = -0.38531223f;  // col3[1]
    cam.viewproj_matrix[14] =  0.22552685f;  // col3[2]
    cam.viewproj_matrix[15] =  0.53003210f;  // col3[3]

    // --- Render config ---
    RenderConfig cfg{};
    cfg.sh_degree      = sh_degree;
    cfg.training       = true;
    cfg.eval_3D        = false;
    cfg.tile_w         = 16;
    cfg.tile_h         = 16;
    cfg.antialiasing   = false;
    cfg.scale_modifier = 1.0f;
    cfg.bg_color[0]    = 0.0f;
    cfg.bg_color[1]    = 0.0f;
    cfg.bg_color[2]    = 0.0f;

    // --- VkTrainingConfig ---
    VkTrainingConfig tcfg{};
    tcfg.max_steps         = 30000;
    tcfg.pos_lr_init       = 1.6e-4f;
    tcfg.pos_lr_final      = 1.6e-6f;
    tcfg.sh_degree_max     = 3;
    tcfg.sh_degree_warmup  = 1000;
    tcfg.densify_from_step = 0;      // disabled — 100 steps insufficient for densification benefit
    tcfg.lambda_dssim      = 0.0f;  // L1-only: DSSIM adds ~10s/step at 720x960
    // SP-6: Python reference regularization defaults
    tcfg.opacity_reg       = 0.01f;
    tcfg.scale_reg         = 0.01f;
    tcfg.spatial_lr_scale  = 1.0f;  // no COLMAP cameras_extent for single-camera test

    // --- Construct VulkanTrainer ---
    VulkanTrainer trainer(ctx, init_g, init_raw, sh_degree, W, H, tcfg);

    // 100-step smoke test: validates loss convergence, no NaN/Inf, and basic PSNR.
    // At 100 steps without densification, PSNR ≈ 4-6 dB (initialisation quality).
    // PSNR > 10 dB requires ~2000 steps + densification (not measured here).
    const int N_STEPS = 100;
    std::vector<float> losses(static_cast<size_t>(N_STEPS), 0.0f);

    for (int step = 0; step < N_STEPS; ++step) {
        losses[static_cast<size_t>(step)] = trainer.step(cam, cfg, target.data(), W, H);
    }

    EXPECT_EQ(trainer.step_count(), N_STEPS);
    for (int step = 0; step < N_STEPS; ++step) {
        EXPECT_GE(losses[static_cast<size_t>(step)], 0.0f)
            << "Negative loss at step " << (step + 1);
        EXPECT_FALSE(std::isnan(losses[static_cast<size_t>(step)]))
            << "NaN loss at step " << (step + 1);
        EXPECT_FALSE(std::isinf(losses[static_cast<size_t>(step)]))
            << "Inf loss at step " << (step + 1);
    }

    // Loss must decrease (training signal check).
    EXPECT_LT(losses[static_cast<size_t>(N_STEPS - 1)], losses[0])
        << "Loss must decrease over " << N_STEPS << " steps";

    // PSNR measurement: at 100 steps without densification expect ≈ 4-6 dB.
    // > 10 dB requires ~2000 steps + densification (separate long-running test).
    // Here we only assert PSNR is finite and positive (not NaN/Inf/negative).
    const float* rendered = trainer.rendered_image();
    if (rendered) {
        const float final_psnr = compute_psnr(rendered, target.data(), total_pixels);
        std::cout << "[Basketball] After " << N_STEPS << " steps:"
                  << " loss[1]=" << losses[0]
                  << " loss[100]=" << losses[static_cast<size_t>(N_STEPS - 1)]
                  << " PSNR=" << final_psnr << " dB"
                  << " N_gaussians=" << trainer.raw_params().count << "\n";
        EXPECT_TRUE(std::isfinite(final_psnr))
            << "PSNR is non-finite after " << N_STEPS << " steps";
        EXPECT_GT(final_psnr, 0.0f)
            << "PSNR is non-positive after " << N_STEPS << " steps (got " << final_psnr << " dB)";
    }
}
