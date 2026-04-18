// test_forward_cache_vk.cpp -- Validates ForwardCache population by PreprocessorVulkan.
//
// PreprocessorVulkan::process() now accepts a ForwardCache* and populates
// cov3D, p_view, and p_hom_w from the GPU shader.  This test verifies that
// these fields match the CPU-reference values (PreprocessorCPU) to 1e-5
// absolute tolerance.
//
// Fixture: tiny golden (N=103, cam0000) — same as test_preprocess_pass_vk.cpp.
//
// Note: ForwardCache.cov2D and cov2D_det are NOT populated by the Vulkan path
// (the Vulkan backward shader recomputes from conics).  They are left null and
// not compared here.

#include "types.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/vk_context.h"
#include "cpu/preprocessor_cpu.h"

#include "golden/compare.h"
#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

std::string tiny_cam0_dir() {
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}

std::vector<float> npy_to_f32_vec(const NpyArray& a) {
    std::vector<float> v(a.numel());
    std::memcpy(v.data(), a.raw.data(), a.numel() * sizeof(float));
    return v;
}

}  // namespace

// -----------------------------------------------------------------------------
// Main test: Vulkan ForwardCache fields match CPU reference on tiny fixture.
// -----------------------------------------------------------------------------
TEST(ForwardCacheVK, MatchesCPUReference_Tiny) {
    const std::string root = tiny_cam0_dir();

    // --- Load inputs --------------------------------------------------------
    auto pos_npy  = load_npy(root + "/input_positions.npy");
    auto scl_npy  = load_npy(root + "/input_scales.npy");
    auto rot_npy  = load_npy(root + "/input_rotations.npy");
    auto opa_npy  = load_npy(root + "/input_opacities.npy");
    auto sh_npy   = load_npy(root + "/input_sh.npy");
    auto f3d_npy  = load_npy(root + "/input_filter_3D.npy");
    auto vm_npy   = load_npy(root + "/input_viewmatrix.npy");
    auto pm_npy   = load_npy(root + "/input_projmatrix.npy");
    auto fov_npy  = load_npy(root + "/input_fov_size.npy");
    auto cp_npy   = load_npy(root + "/input_campos.npy");
    auto meta_npy = load_npy(root + "/input_meta.npy");

    const int N               = static_cast<int>(pos_npy.shape[0]);
    const int sh_degree       = static_cast<int>(meta_npy.f32()[0]);
    const int sh_coeffs_per_g = static_cast<int>(meta_npy.f32()[1]);

    // --- Build backing vectors (must outlive process() calls) ---------------
    std::vector<float> positions  = npy_to_f32_vec(pos_npy);
    std::vector<float> scales     = npy_to_f32_vec(scl_npy);
    std::vector<float> rotations  = npy_to_f32_vec(rot_npy);
    std::vector<float> opacities  = npy_to_f32_vec(opa_npy);
    std::vector<float> sh_coeffs  = npy_to_f32_vec(sh_npy);
    std::vector<float> filter_3d  = npy_to_f32_vec(f3d_npy);

    // --- Build Camera -------------------------------------------------------
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
    cfg.training      = true;
    cfg.eval_3D       = false;
    cfg.tile_w        = 16;
    cfg.tile_h        = 16;
    cfg.antialiasing  = false;
    cfg.scale_modifier = 1.0f;

    // =========================================================================
    // 1. CPU reference — run PreprocessorCPU with cache to get ground truth.
    // =========================================================================
    FrameAllocator cpu_alloc(64u * 1024u * 1024u);
    ForwardCache cpu_cache{};
    PreprocessorCPU cpu_pp;
    cpu_pp.process(g, cam, cfg, cpu_alloc, &cpu_cache);

    ASSERT_NE(cpu_cache.cov3D,   nullptr) << "CPU cache.cov3D not populated";
    ASSERT_NE(cpu_cache.p_view,  nullptr) << "CPU cache.p_view not populated";
    ASSERT_NE(cpu_cache.p_hom_w, nullptr) << "CPU cache.p_hom_w not populated";

    // =========================================================================
    // 2. Vulkan — run PreprocessorVulkan::process() with cache.
    // =========================================================================
    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device available";

    FrameAllocator vk_alloc(64u * 1024u * 1024u);
    ForwardCache vk_cache{};
    PreprocessorVulkan vk_pp(ctx);
    vk_pp.process(g, cam, cfg, vk_alloc, &vk_cache);

    ASSERT_NE(vk_cache.cov3D,   nullptr) << "Vulkan cache.cov3D not populated";
    ASSERT_NE(vk_cache.p_view,  nullptr) << "Vulkan cache.p_view not populated";
    ASSERT_NE(vk_cache.p_hom_w, nullptr) << "Vulkan cache.p_hom_w not populated";

    // =========================================================================
    // 3. Compare.
    // =========================================================================
    constexpr float kAbsTol = 1e-5f;
    constexpr float kRelTol = 1e-4f;

    // cov3D: [N*6]
    {
        std::vector<float> got(vk_cache.cov3D,   vk_cache.cov3D   + N * 6);
        std::vector<float> exp(cpu_cache.cov3D,  cpu_cache.cov3D  + N * 6);
        auto r = compare_f32(got, exp, kAbsTol, kRelTol);
        EXPECT_TRUE(r.passed)
            << "cov3D: num_bad=" << r.num_bad
            << " first_bad=" << r.first_bad_index
            << " max_abs=" << r.max_abs_err
            << " max_rel=" << r.max_rel_err;
    }

    // p_view: [N*3]
    {
        std::vector<float> got(vk_cache.p_view,  vk_cache.p_view  + N * 3);
        std::vector<float> exp(cpu_cache.p_view, cpu_cache.p_view + N * 3);
        auto r = compare_f32(got, exp, kAbsTol, kRelTol);
        EXPECT_TRUE(r.passed)
            << "p_view: num_bad=" << r.num_bad
            << " first_bad=" << r.first_bad_index
            << " max_abs=" << r.max_abs_err
            << " max_rel=" << r.max_rel_err;
    }

    // p_hom_w: [N]
    {
        std::vector<float> got(vk_cache.p_hom_w,  vk_cache.p_hom_w  + N);
        std::vector<float> exp(cpu_cache.p_hom_w, cpu_cache.p_hom_w + N);
        auto r = compare_f32(got, exp, kAbsTol, kRelTol);
        EXPECT_TRUE(r.passed)
            << "p_hom_w: num_bad=" << r.num_bad
            << " first_bad=" << r.first_bad_index
            << " max_abs=" << r.max_abs_err
            << " max_rel=" << r.max_rel_err;
    }
}

// -----------------------------------------------------------------------------
// Negative test: download_cache() throws if process() has not been called.
// -----------------------------------------------------------------------------
TEST(ForwardCacheVK, DownloadCacheThrowsBeforeProcess) {
    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device available";

    FrameAllocator alloc(1u * 1024u * 1024u);
    ForwardCache cache{};
    PreprocessorVulkan vk_pp(ctx);
    EXPECT_THROW(vk_pp.download_cache(1, cache, alloc), std::runtime_error);
}
