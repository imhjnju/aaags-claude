// test_preprocess_backward_pass_vk.cpp — SP-3 T23: PreprocessBackwardPass smoke test.
//
// N=4, sh_degree=0 with a larger SH buffer to cover inactive coefficient overwrite.
// Constructs plausible input: positions, radii=[1,1,0,1], cov3D (diagonal),
// d_conics, d_rgb, d_means2D (nonzero), simple view matrix.
// Runs pass, downloads d_sh, d_scales, d_rotations, d_means3D.
// Checks: outputs are nonzero/finite for active Gaussians, zero for culled (radii=0).
//
// GTEST_SKIP if no Vulkan device.

#include "vulkan/vk_context.h"
#include "vulkan/preprocess_backward_pass.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/backward_bindings.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

TEST(PreprocessBackwardPass, TinyFixture) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const uint32_t N       = 4;
    const uint32_t K       = 16u;  // allocated coeffs can exceed active sh_degree during warmup
    const uint32_t sh_deg  = 0u;

    // ---- positions (N*3) ---------------------------------------------------
    std::vector<float> positions = {
        0.0f, 0.0f, 5.0f,   // Gaussian 0 (active)
        1.0f, 0.5f, 4.0f,   // Gaussian 1 (active)
        -1.0f, 0.0f, 3.0f,  // Gaussian 2 (culled: radii=0)
        0.5f, -0.5f, 6.0f,  // Gaussian 3 (active)
    };

    // ---- radii (N): 0=culled -----------------------------------------------
    std::vector<int32_t> radii = {1, 1, 0, 1};

    // ---- cov3D (N*6) upper triangle: [xx,xy,xz,yy,yz,zz] -----------------
    // Diagonal covariance for simplicity
    std::vector<float> cov3D_data(N * 6, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        cov3D_data[i*6 + 0] = 1.0f;  // xx
        cov3D_data[i*6 + 3] = 1.0f;  // yy
        cov3D_data[i*6 + 5] = 1.0f;  // zz
    }

    // ---- d_conics (N*3): upstream gradient (a,b,c) -------------------------
    std::vector<float> d_conics(N * 3, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        d_conics[i*3 + 0] = 0.1f;
        d_conics[i*3 + 1] = 0.05f;
        d_conics[i*3 + 2] = 0.1f;
    }

    // ---- d_opacity (N): upstream opacity gradient --------------------------
    std::vector<float> d_opacity(N, 0.1f);

    // ---- sh_coeffs (N * K * 3): degree-0 SH coefficients ------------------
    std::vector<float> sh_coeffs_data(N * K * 3, 0.3f);
    // Vary per Gaussian for interest
    for (uint32_t i = 0; i < N; i++) {
        sh_coeffs_data[i * K * 3 + 0] = 0.2f + 0.1f * i;
        sh_coeffs_data[i * K * 3 + 1] = 0.3f + 0.1f * i;
        sh_coeffs_data[i * K * 3 + 2] = 0.1f + 0.1f * i;
    }

    // ---- scales (N*3): exp-activated ---------------------------------------
    std::vector<float> scales_data(N * 3);
    for (uint32_t i = 0; i < N; i++) {
        scales_data[i*3 + 0] = 0.5f;
        scales_data[i*3 + 1] = 0.5f;
        scales_data[i*3 + 2] = 0.5f;
    }

    // ---- rotations (N*4): identity quaternion (r,x,y,z) = (1,0,0,0) ------
    std::vector<float> rotations_data(N * 4, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        rotations_data[i*4 + 0] = 1.0f;  // r
    }

    // ---- d_rgb (N*3): upstream color gradients -----------------------------
    std::vector<float> d_rgb_data(N * 3, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        d_rgb_data[i*3 + 0] = 0.5f;
        d_rgb_data[i*3 + 1] = 0.3f;
        d_rgb_data[i*3 + 2] = 0.2f;
    }

    // ---- d_means2D (N*2): upstream screen-position gradient ----------------
    std::vector<float> d_means2D_data(N * 2, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        d_means2D_data[i*2 + 0] = 0.01f;
        d_means2D_data[i*2 + 1] = 0.01f;
    }

    // ---- Output zero buffers -----------------------------------------------
    std::vector<float> zeros_m3d(N * 3, 0.0f);
    std::vector<float> poison_sh(N * K * 3, 123.0f);
    std::vector<float> zeros_sc(N * 3, 0.0f);
    std::vector<float> zeros_rot(N * 4, 0.0f);

    // ---- Opacities and raw_rotations data ---------------------------------
    // opacities: sigmoid-activated values (N floats)
    std::vector<float> opacities_data(N, 0.7f);

    // d_raw_opacities output: zero-filled
    std::vector<float> zeros_d_raw_opa(N, 0.0f);

    // raw_rotations: unnormalized quaternions (identity here, so |q|=1)
    std::vector<float> raw_rotations_data(N * 4, 0.0f);
    for (uint32_t i = 0; i < N; i++) {
        raw_rotations_data[i*4 + 0] = 1.0f;  // r (identity, |q|=1)
    }

    // means2D_cache: pixel-space means2D from forward pass.
    // For a Gaussian at z=5 with identity view/proj and W=H=64:
    //   p_hom = proj*[x,y,z,1] => for identity at [0,0,5]: p_hom=[0,0,5,5]
    //   ndc = [0,0], pixel = ((0+1)*64-1)*0.5 = 31.5
    // These values approximate what the forward pass would produce.
    std::vector<float> means2D_cache_data(N * 2, 31.5f);  // center of 64x64 image

    // ---- Build UBO --------------------------------------------------------
    // Simple camera: identity view matrix, looking down +Z, W=H=64
    PreprocessBackwardUBO ubo{};

    // Identity view matrix (column-major)
    ubo.view_matrix[0]  = 1.f; ubo.view_matrix[5]  = 1.f;
    ubo.view_matrix[10] = 1.f; ubo.view_matrix[15] = 1.f;

    // Simple projection matrix: proj[0]*x + proj[4]*y = ndc*w, etc.
    // For testing: simple perspective with fov=1 (tan_fov=1), W=H=64
    float focal = 32.0f; // W/(2*tan_fov) = 64/2 = 32
    // Column-major 4x4 viewproj:
    ubo.proj_matrix[0]  = focal;  // col0[0]
    ubo.proj_matrix[5]  = focal;  // col1[1]
    ubo.proj_matrix[10] = 1.f;    // col2[2]
    ubo.proj_matrix[11] = 1.f;    // col2[3] (w = z)
    // (rest zero)

    ubo.num_gaussians   = N;
    ubo.sh_degree       = sh_deg;
    ubo.sh_coeffs_per_g = K;
    ubo.scale_modifier  = 1.0f;
    ubo.h_x             = focal;  // W/(2*tan_fovx)
    ubo.h_y             = focal;  // H/(2*tan_fovy)
    ubo.tan_fovx        = 1.0f;
    ubo.tan_fovy        = 1.0f;
    ubo.cam_pos[0]      = 0.0f;
    ubo.cam_pos[1]      = 0.0f;
    ubo.cam_pos[2]      = 0.0f;
    ubo.training        = 1u;
    ubo.cam_width       = 64u;   // required by ndc recovery in means2D_cache path
    ubo.cam_height      = 64u;

    // ---- Allocate GPU buffers ---------------------------------------------
    auto pos_buf      = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rad_buf      = std::make_unique<VulkanBuffer>(ctx, N*sizeof(int32_t),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cv3_buf      = std::make_unique<VulkanBuffer>(ctx, N*6*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dcon_buf     = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dopa_buf     = std::make_unique<VulkanBuffer>(ctx, N*sizeof(float),     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sh_buf       = std::make_unique<VulkanBuffer>(ctx, N*K*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto sc_buf       = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rot_buf      = std::make_unique<VulkanBuffer>(ctx, N*4*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto drgb_buf     = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dm2d_buf     = std::make_unique<VulkanBuffer>(ctx, N*2*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dm3d_buf     = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dsh_buf      = std::make_unique<VulkanBuffer>(ctx, N*K*3*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto dsc_buf      = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto drot_buf     = std::make_unique<VulkanBuffer>(ctx, N*4*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Bindings 14, 15 (opacities, d_raw_opacities), 17 (raw_rotations), 18 (means2D_cache)
    auto opa_buf      = std::make_unique<VulkanBuffer>(ctx, N*sizeof(float),     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto d_raw_opa_buf= std::make_unique<VulkanBuffer>(ctx, N*sizeof(float),     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto rawrot_buf   = std::make_unique<VulkanBuffer>(ctx, N*4*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto m2d_cache_buf= std::make_unique<VulkanBuffer>(ctx, N*2*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto ubo_buf      = std::make_unique<VulkanBuffer>(ctx, sizeof(PreprocessBackwardUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    // Bindings 19..22: p_view_cache_in, cov2D_cache_in, cov2D_det_cache_in, p_hom_w_cache_in
    // p_view_cache_in: with identity view matrix, p_view = world position
    auto pview_in_buf = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto cov2d_in_buf = std::make_unique<VulkanBuffer>(ctx, N*3*sizeof(float),   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto c2ddet_in_buf= std::make_unique<VulkanBuffer>(ctx, N*sizeof(float),     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // p_hom_w_cache_in: for identity proj with proj[11]=1, p_hom.w = z
    auto phomw_in_buf = std::make_unique<VulkanBuffer>(ctx, N*sizeof(float),     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // ---- Upload inputs ----------------------------------------------------
    pos_buf     ->upload(positions.data(),          N*3*sizeof(float));
    rad_buf     ->upload(radii.data(),              N*sizeof(int32_t));
    cv3_buf     ->upload(cov3D_data.data(),         N*6*sizeof(float));
    dcon_buf    ->upload(d_conics.data(),           N*3*sizeof(float));
    dopa_buf    ->upload(d_opacity.data(),          N*sizeof(float));
    sh_buf      ->upload(sh_coeffs_data.data(),     N*K*3*sizeof(float));
    sc_buf      ->upload(scales_data.data(),        N*3*sizeof(float));
    rot_buf     ->upload(rotations_data.data(),     N*4*sizeof(float));
    drgb_buf    ->upload(d_rgb_data.data(),         N*3*sizeof(float));
    dm2d_buf    ->upload(d_means2D_data.data(),     N*2*sizeof(float));
    dm3d_buf    ->upload(zeros_m3d.data(),          N*3*sizeof(float));
    dsh_buf     ->upload(poison_sh.data(),           N*K*3*sizeof(float));
    dsc_buf     ->upload(zeros_sc.data(),           N*3*sizeof(float));
    drot_buf    ->upload(zeros_rot.data(),          N*4*sizeof(float));
    opa_buf     ->upload(opacities_data.data(),     N*sizeof(float));
    d_raw_opa_buf->upload(zeros_d_raw_opa.data(),  N*sizeof(float));
    rawrot_buf  ->upload(raw_rotations_data.data(), N*4*sizeof(float));
    m2d_cache_buf->upload(means2D_cache_data.data(),N*2*sizeof(float));
    ubo_buf     ->upload(&ubo,                      sizeof(ubo));

    // p_view_cache_in: with identity view matrix, p_view = world position (x,y,z)
    // Positions: [0,0,5], [1,0.5,4], [-1,0,3], [0.5,-0.5,6]
    pview_in_buf->upload(positions.data(), N*3*sizeof(float));

    // cov2D_cache_in: dilated cov2D for each Gaussian.
    // Camera: identity view, W=H=64, tan_fov=1.0, focal=32.0
    // With identity view and diagonal cov3D: result[0][0] = (focal/t.z)^2 * sigma_x^2
    // For diagonal cov3D = diag(1,1,1) and identity view: fa = fc = (focal/t.z)^2 + 0.3, fb = 0
    // Gaussian 0: z=5 -> fa=fc=(32/5)^2+0.3=41.26, det=41.26^2
    // Gaussian 1: z=4 -> fa=fc=(32/4)^2+0.3=64.30, det=64.30^2
    // Gaussian 2: z=3 (culled, radii=0) -> fa=fc=0 (doesn't matter, shader skips)
    // Gaussian 3: z=6 -> fa=fc=(32/6)^2+0.3=28.58, det=28.58^2
    std::vector<float> cov2d_data(N * 3, 0.0f);
    std::vector<float> c2ddet_data(N, 0.0f);
    // active Gaussians: 0, 1, 3
    float fa0 = (32.0f/5.0f)*(32.0f/5.0f) + 0.3f;  // 41.26
    float fa1 = (32.0f/4.0f)*(32.0f/4.0f) + 0.3f;  // 64.30
    float fa3 = (32.0f/6.0f)*(32.0f/6.0f) + 0.3f;  // 28.58
    cov2d_data[0*3+0] = fa0; cov2d_data[0*3+1] = 0.0f; cov2d_data[0*3+2] = fa0;
    cov2d_data[1*3+0] = fa1; cov2d_data[1*3+1] = 0.0f; cov2d_data[1*3+2] = fa1;
    // Gaussian 2 culled: cov2d = 0 (already set)
    cov2d_data[3*3+0] = fa3; cov2d_data[3*3+1] = 0.0f; cov2d_data[3*3+2] = fa3;
    c2ddet_data[0] = fa0 * fa0;
    c2ddet_data[1] = fa1 * fa1;
    c2ddet_data[3] = fa3 * fa3;
    cov2d_in_buf ->upload(cov2d_data.data(),   N*3*sizeof(float));
    c2ddet_in_buf->upload(c2ddet_data.data(),  N*sizeof(float));

    // p_hom_w_cache_in: for identity proj with proj[11]=1, p_hom.w = z component
    // Positions[i].z: 5, 4, 3, 6 for Gaussians 0..3
    std::vector<float> phomw_data = {5.0f, 4.0f, 3.0f, 6.0f};
    phomw_in_buf ->upload(phomw_data.data(),   N*sizeof(float));

    // ---- Create pass and dispatch -----------------------------------------
    PreprocessBackwardPass pass(ctx);

    PreprocessBackwardPass::Buffers pb{};
    pb.positions       = pos_buf      ->handle();
    pb.radii           = rad_buf      ->handle();
    pb.cov3D           = cv3_buf      ->handle();
    pb.d_conics        = dcon_buf     ->handle();
    pb.d_opacity       = dopa_buf     ->handle();
    pb.sh_coeffs       = sh_buf       ->handle();
    pb.scales          = sc_buf       ->handle();
    pb.rotations       = rot_buf      ->handle();
    pb.d_rgb           = drgb_buf     ->handle();
    pb.d_means2D       = dm2d_buf     ->handle();
    pb.d_means3D       = dm3d_buf     ->handle();
    pb.d_sh            = dsh_buf      ->handle();
    pb.d_scales        = dsc_buf      ->handle();
    pb.d_rotations     = drot_buf     ->handle();
    pb.opacities       = opa_buf      ->handle();
    pb.d_raw_opacities = d_raw_opa_buf->handle();
    pb.raw_rotations      = rawrot_buf   ->handle();
    pb.means2D_cache      = m2d_cache_buf->handle();
    pb.p_view_cache_in    = pview_in_buf ->handle();
    pb.cov2D_cache_in     = cov2d_in_buf ->handle();
    pb.cov2D_det_cache_in = c2ddet_in_buf->handle();
    pb.p_hom_w_cache_in   = phomw_in_buf ->handle();

    pass.bind_buffers(pb, ubo_buf->handle());
    pass.dispatch_sync(N);

    // ---- Download and validate --------------------------------------------
    std::vector<float> out_d_sh(N * K * 3, 0.f);
    std::vector<float> out_d_scales(N * 3, 0.f);
    std::vector<float> out_d_rotations(N * 4, 0.f);
    std::vector<float> out_d_means3D(N * 3, 0.f);

    dsh_buf ->download(out_d_sh.data(),        N*K*3*sizeof(float));
    dsc_buf ->download(out_d_scales.data(),    N*3*sizeof(float));
    drot_buf->download(out_d_rotations.data(), N*4*sizeof(float));
    dm3d_buf->download(out_d_means3D.data(),   N*3*sizeof(float));

    // Gaussian 2 (culled, radii=0): all outputs must be zero
    for (uint32_t ch = 0u; ch < K * 3u; ch++) {
        EXPECT_EQ(out_d_sh[2u * K * 3u + ch], 0.0f)
            << "Culled Gaussian 2 d_sh[" << ch << "] should be zero";
    }
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(out_d_scales[2*3 + c], 0.0f)
            << "Culled Gaussian 2 d_scales[" << c << "] should be zero";
        EXPECT_EQ(out_d_means3D[2*3 + c], 0.0f)
            << "Culled Gaussian 2 d_means3D[" << c << "] should be zero";
    }
    for (int c = 0; c < 4; c++) {
        EXPECT_EQ(out_d_rotations[2*4 + c], 0.0f)
            << "Culled Gaussian 2 d_rotations[" << c << "] should be zero";
    }

    // Active Gaussians (0, 1, 3): d_sh should be nonzero and finite
    for (uint32_t idx : {0u, 1u, 3u}) {
        bool sh_nonzero = false;
        for (uint32_t ch = 0u; ch < K * 3u; ch++) {
            float v = out_d_sh[idx * K * 3u + ch];
            EXPECT_TRUE(std::isfinite(v))
                << "Active Gaussian " << idx << " d_sh[" << ch << "] is not finite";
            if (v != 0.0f) sh_nonzero = true;
        }
        EXPECT_TRUE(sh_nonzero)
            << "Active Gaussian " << idx << " d_sh is all-zero";
        for (uint32_t coeff = 1u; coeff < K; ++coeff) {
            for (uint32_t ch = 0u; ch < 3u; ++ch) {
                const uint32_t off = idx * K * 3u + coeff * 3u + ch;
                EXPECT_EQ(out_d_sh[off], 0.0f)
                    << "Inactive SH coeff retained stale data for Gaussian " << idx
                    << " coeff " << coeff << " channel " << ch;
            }
        }

        // d_means3D should be finite
        for (int c = 0; c < 3; c++) {
            EXPECT_TRUE(std::isfinite(out_d_means3D[idx * 3 + c]))
                << "Active Gaussian " << idx << " d_means3D[" << c << "] is not finite";
        }
    }
}
