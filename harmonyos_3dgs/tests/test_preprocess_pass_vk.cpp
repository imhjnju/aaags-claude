// test_preprocess_pass_vk.cpp -- T7 golden validation for PreprocessorVulkan.
//
// Reconstructs the tiny fixture scene from input_*.npy, runs
// PreprocessorVulkan::process(), and compares every per-Gaussian output
// against the CUDA-produced preprocess_*.npy golden bytes.
//
// Thresholds (dual abs/rel) mirror the CUDA-vs-CPU tolerances established
// by the CPU preprocessor tests: abs<1e-5 OR rel<1e-4 is "matches". The
// radii and tiles_touched arrays are compared as exact int32 (same bit
// pattern — preprocess.comp writes `int` which is u32-compatible for
// non-negative tile counts).
//
// The two negative tests exercise the hard-error contract from spec §4.4:
// SP-2 rejects eval_3D=true and any tile size other than 16x16. These are
// runtime checks in PreprocessorVulkan::process().

#include "types.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/vk_context.h"

#include "golden/compare.h"
#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// TEST_DATA_DIR points at harmonyos_3dgs/tests/test_data; tiny fixture lives
// at harmonyos_3dgs/tests/golden/tiny/step000001/cam0000 (sibling directory).
std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}

// Convenience wrappers that copy an NpyArray's payload into a std::vector
// of the correct type. The arena-backed GaussianData pointers in process()
// must outlive the dispatch, so we hold the raw bytes in std::vector for
// the duration of the test (std::vector data is heap-allocated and stable).
std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(),
                a.raw.data(),
                a.numel() * sizeof(float));
    return v;
}

}  // namespace

// -----------------------------------------------------------------------------
// Main test: byte-compatible with CUDA golden on the tiny fixture.
// -----------------------------------------------------------------------------
TEST(PreprocessPass, MatchesCUDAGolden_Tiny) {
    const std::string root = tiny_cam0_dir();

    // --- Load inputs --------------------------------------------------------
    auto pos_npy  = load_npy(root + "/input_positions.npy");           // [N, 3]
    auto scl_npy  = load_npy(root + "/input_scales.npy");              // [N, 3]
    auto rot_npy  = load_npy(root + "/input_rotations.npy");           // [N, 4]
    auto opa_npy  = load_npy(root + "/input_opacities.npy");           // [N, 1]
    auto sh_npy   = load_npy(root + "/input_sh.npy");                  // [N, M, 3]
    auto f3d_npy  = load_npy(root + "/input_filter_3D.npy");           // [N]
    auto vm_npy   = load_npy(root + "/input_viewmatrix.npy");          // [4, 4]
    auto pm_npy   = load_npy(root + "/input_projmatrix.npy");          // [4, 4]
    auto fov_npy  = load_npy(root + "/input_fov_size.npy");            // [4]
    auto cp_npy   = load_npy(root + "/input_campos.npy");              // [3]
    auto meta_npy = load_npy(root + "/input_meta.npy");                // [4]

    const int N = static_cast<int>(pos_npy.shape[0]);
    // meta = [sh_degree, sh_coeffs_per_g, H, W]
    ASSERT_EQ(meta_npy.shape.size(), 1u);
    ASSERT_EQ(meta_npy.shape[0], 4u);
    const int sh_degree       = static_cast<int>(meta_npy.f32()[0]);
    const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);
    const int H               = static_cast<int>(meta_npy.f32()[2]);
    const int W               = static_cast<int>(meta_npy.f32()[3]);

    // Sanity checks on shapes (degenerate scene → meaningless test).
    ASSERT_EQ(pos_npy.shape[1], 3u);
    ASSERT_EQ(scl_npy.shape[0], static_cast<size_t>(N));
    ASSERT_EQ(rot_npy.shape[1], 4u);
    ASSERT_EQ(sh_npy.numel(),
              static_cast<size_t>(N) * sh_coeffs_per_g * 3);
    ASSERT_EQ(f3d_npy.numel(), static_cast<size_t>(N));
    ASSERT_EQ(vm_npy.numel(), 16u);
    ASSERT_EQ(pm_npy.numel(), 16u);

    // Hold-onto vectors: the GaussianData struct uses raw pointers; the
    // backing storage must outlive the process() call below.
    std::vector<float> positions  = npy_to_f32_vec(pos_npy);
    std::vector<float> scales     = npy_to_f32_vec(scl_npy);
    std::vector<float> rotations  = npy_to_f32_vec(rot_npy);
    std::vector<float> opacities  = npy_to_f32_vec(opa_npy);     // [N, 1] → N floats
    std::vector<float> sh_coeffs  = npy_to_f32_vec(sh_npy);      // [N, M, 3] → N*M*3 floats
    std::vector<float> filter_3d  = npy_to_f32_vec(f3d_npy);

    // --- Build Camera -------------------------------------------------------
    // The AAA-Gaussians numpy dump stores the transposed W2V in row-major.
    // transformPoint4x3 in math_utils.cpp reads m[12..14] as the translation,
    // which is the 4th row (indices 12..15) of the same row-major buffer —
    // i.e. the exact memcpy matches the C++ "column-major" index math. No
    // manual transpose needed. (dev_notes/master_plan tiny-fixture convention.)
    Camera cam{};
    std::memcpy(cam.view_matrix,     vm_npy.f32(), 16 * sizeof(float));
    std::memcpy(cam.viewproj_matrix, pm_npy.f32(), 16 * sizeof(float));
    cam.cam_pos[0] = cp_npy.f32()[0];
    cam.cam_pos[1] = cp_npy.f32()[1];
    cam.cam_pos[2] = cp_npy.f32()[2];
    cam.tan_fovx   = fov_npy.f32()[0];
    cam.tan_fovy   = fov_npy.f32()[1];
    cam.width      = static_cast<int>(fov_npy.f32()[2]);
    cam.height     = static_cast<int>(fov_npy.f32()[3]);
    ASSERT_EQ(cam.width,  W);
    ASSERT_EQ(cam.height, H);

    // --- Build GaussianData -------------------------------------------------
    GaussianData g{};
    g.count      = N;
    g.sh_degree  = sh_degree;
    g.max_coeffs = sh_coeffs_per_g;
    g.positions  = positions.data();
    g.scales     = scales.data();
    g.rotations  = rotations.data();
    g.opacities  = opacities.data();
    g.sh_coeffs  = sh_coeffs.data();
    g.filter_3D  = filter_3d.data();

    // --- Build RenderConfig -------------------------------------------------
    RenderConfig cfg{};
    cfg.sh_degree     = sh_degree;
    cfg.training      = true;            // spec_training=1 baked into pipeline
    cfg.eval_3D       = false;           // spec_eval_3D=0
    cfg.tile_w        = 16;
    cfg.tile_h        = 16;
    cfg.antialiasing  = false;
    cfg.scale_modifier = 1.0f;

    // --- Run Vulkan preprocess ---------------------------------------------
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init())
        << "Vulkan init failed — no compute-capable device?";

    FrameAllocator alloc(32u * 1024u * 1024u);   // 32 MB arena
    PreprocessorVulkan pp(ctx);
    PreprocessOutput out = pp.process(g, cam, cfg, alloc);

    // --- Load golden outputs -----------------------------------------------
    auto gold_means2D    = load_npy(root + "/preprocess_means2D.npy");       // [N, 2] f32
    auto gold_depths     = load_npy(root + "/preprocess_depths.npy");        // [N]    f32
    auto gold_conic_op   = load_npy(root + "/preprocess_conic_opacity.npy"); // [N, 4] f32
    auto gold_rgb        = load_npy(root + "/preprocess_rgb.npy");           // [N, 3] f32
    auto gold_radii      = load_npy(root + "/preprocess_radii.npy");         // [N] i32
    auto gold_tiles_t    = load_npy(root + "/preprocess_tiles_touched.npy"); // [N] u32

    ASSERT_EQ(gold_means2D.numel(),  static_cast<size_t>(N) * 2);
    ASSERT_EQ(gold_depths.numel(),   static_cast<size_t>(N));
    ASSERT_EQ(gold_conic_op.numel(), static_cast<size_t>(N) * 4);
    ASSERT_EQ(gold_rgb.numel(),      static_cast<size_t>(N) * 3);
    ASSERT_EQ(gold_radii.numel(),    static_cast<size_t>(N));
    ASSERT_EQ(gold_tiles_t.numel(),  static_cast<size_t>(N));

    // --- Compare floats with dual threshold --------------------------------
    constexpr float kAbsTol = 1e-5f;
    constexpr float kRelTol = 1e-4f;

    std::vector<float> got_m2d(out.means2D, out.means2D + N * 2);
    std::vector<float> exp_m2d(gold_means2D.f32(), gold_means2D.f32() + N * 2);
    auto r_m2d = compare_f32(got_m2d, exp_m2d, kAbsTol, kRelTol);
    EXPECT_TRUE(r_m2d.passed)
        << "means2D: num_bad=" << r_m2d.num_bad
        << " first_bad=" << r_m2d.first_bad_index
        << " max_abs=" << r_m2d.max_abs_err
        << " max_rel=" << r_m2d.max_rel_err;

    std::vector<float> got_dep(out.depths, out.depths + N);
    std::vector<float> exp_dep(gold_depths.f32(), gold_depths.f32() + N);
    auto r_dep = compare_f32(got_dep, exp_dep, kAbsTol, kRelTol);
    EXPECT_TRUE(r_dep.passed)
        << "depths: num_bad=" << r_dep.num_bad
        << " first_bad=" << r_dep.first_bad_index
        << " max_abs=" << r_dep.max_abs_err
        << " max_rel=" << r_dep.max_rel_err;

    std::vector<float> got_rgb(out.rgb, out.rgb + N * 3);
    std::vector<float> exp_rgb(gold_rgb.f32(), gold_rgb.f32() + N * 3);
    auto r_rgb = compare_f32(got_rgb, exp_rgb, kAbsTol, kRelTol);
    EXPECT_TRUE(r_rgb.passed)
        << "rgb: num_bad=" << r_rgb.num_bad
        << " first_bad=" << r_rgb.first_bad_index
        << " max_abs=" << r_rgb.max_abs_err
        << " max_rel=" << r_rgb.max_rel_err;

    // Re-pack PreprocessorVulkan's separate conics[N*3] and opacities_2d[N]
    // back into the {conic.a, conic.b, conic.c, opacity} stride-4 layout to
    // match the CUDA golden's packed `preprocess_conic_opacity.npy`.
    std::vector<float> got_cop(static_cast<size_t>(N) * 4);
    for (int i = 0; i < N; ++i) {
        got_cop[i * 4 + 0] = out.conics[i * 3 + 0];
        got_cop[i * 4 + 1] = out.conics[i * 3 + 1];
        got_cop[i * 4 + 2] = out.conics[i * 3 + 2];
        got_cop[i * 4 + 3] = out.opacities_2d[i];
    }
    std::vector<float> exp_cop(gold_conic_op.f32(), gold_conic_op.f32() + N * 4);
    auto r_cop = compare_f32(got_cop, exp_cop, kAbsTol, kRelTol);
    EXPECT_TRUE(r_cop.passed)
        << "conic_opacity: num_bad=" << r_cop.num_bad
        << " first_bad=" << r_cop.first_bad_index
        << " max_abs=" << r_cop.max_abs_err
        << " max_rel=" << r_cop.max_rel_err;

    // --- Exact-match integer outputs ---------------------------------------
    // radii: PreprocessorVulkan writes int32; golden is int32.
    // tiles_touched: Vulkan shader writes int; golden dtype is uint32, but
    // both are non-negative tile counts so the bit patterns must be equal.
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(out.radii[i], gold_radii.i32()[i]) << "radii[" << i << "]";
    }
    for (int i = 0; i < N; ++i) {
        const uint32_t got = static_cast<uint32_t>(out.tiles_touched[i]);
        EXPECT_EQ(got, gold_tiles_t.u32()[i]) << "tiles_touched[" << i << "]";
    }
}

// -----------------------------------------------------------------------------
// Hard-error contracts (SP-2 spec §4.4).
// -----------------------------------------------------------------------------

// Small helper to build a minimal valid GaussianData+Camera for negative
// tests: the actual values don't matter because process() must throw before
// touching them. One Gaussian with identity rotation is enough.
namespace {
struct MinimalScene {
    float pos[3]  = {0, 0, 5};
    float scl[3]  = {1, 1, 1};
    float rot[4]  = {1, 0, 0, 0};
    float opa[1]  = {0.5f};
    float sh[3]   = {0, 0, 0};          // sh_degree=0 → 1 coeff * 3 = 3 floats
    float f3d[1]  = {0.0f};
};
}  // namespace

TEST(PreprocessPass, RejectsEval3D) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    MinimalScene s;
    GaussianData g{};
    g.count = 1; g.sh_degree = 0; g.max_coeffs = 1;
    g.positions = s.pos; g.scales = s.scl; g.rotations = s.rot;
    g.opacities = s.opa; g.sh_coeffs = s.sh; g.filter_3D = s.f3d;

    Camera cam{};
    const float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::memcpy(cam.view_matrix,     id, sizeof(id));
    std::memcpy(cam.viewproj_matrix, id, sizeof(id));
    cam.tan_fovx = 1.0f; cam.tan_fovy = 1.0f;
    cam.width = 64; cam.height = 64;

    RenderConfig cfg{};
    cfg.sh_degree = 0;
    cfg.eval_3D   = true;    // must be rejected
    cfg.tile_w    = 16;
    cfg.tile_h    = 16;

    FrameAllocator alloc(1u * 1024u * 1024u);
    PreprocessorVulkan pp(ctx);
    EXPECT_THROW(pp.process(g, cam, cfg, alloc), std::runtime_error);
}

TEST(PreprocessPass, RejectsNon16Tile) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    MinimalScene s;
    GaussianData g{};
    g.count = 1; g.sh_degree = 0; g.max_coeffs = 1;
    g.positions = s.pos; g.scales = s.scl; g.rotations = s.rot;
    g.opacities = s.opa; g.sh_coeffs = s.sh; g.filter_3D = s.f3d;

    Camera cam{};
    const float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::memcpy(cam.view_matrix,     id, sizeof(id));
    std::memcpy(cam.viewproj_matrix, id, sizeof(id));
    cam.tan_fovx = 1.0f; cam.tan_fovy = 1.0f;
    cam.width = 64; cam.height = 64;

    RenderConfig cfg{};
    cfg.sh_degree = 0;
    cfg.eval_3D   = false;
    cfg.tile_w    = 32;      // must be rejected
    cfg.tile_h    = 16;

    FrameAllocator alloc(1u * 1024u * 1024u);
    PreprocessorVulkan pp(ctx);
    EXPECT_THROW(pp.process(g, cam, cfg, alloc), std::runtime_error);
}
