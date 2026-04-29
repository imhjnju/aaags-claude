// test_mcmc_densification.cpp — Phase 2 of MCMC densification port.
//
// Validates mcmc::relocate_gs / add_new_gs / densify against Python reference
// behavior in scene/gaussian_model.py. CPU-only (no Vulkan dependency).

#include "mcmc_densification.h"
#include "train_types.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float inverse_sigmoid(float p) { return std::log(p / (1.0f - p)); }

// Build N Gaussians where slot i has:
//   raw_opacity[i] = (i < num_dead) ? raw_opacity_dead : raw_opacity_alive
//   raw_scales = log(0.1)  (exp -> 0.1 isotropic)
//   positions = (i, 0, 0)  (unique per slot)
//   rotations = identity quaternion (1, 0, 0, 0)
//   sh_coeffs = zeros
OwnedRawParams make_params_split(int N, int num_dead,
                                 float raw_opacity_dead,
                                 float raw_opacity_alive) {
    OwnedRawParams p;
    p.sh_degree  = 0;
    p.max_coeffs = 1;
    p.resize(N);
    const float raw_sc = std::log(0.1f);
    for (int i = 0; i < N; ++i) {
        p.positions[i * 3 + 0] = static_cast<float>(i);
        p.positions[i * 3 + 1] = 0.0f;
        p.positions[i * 3 + 2] = 0.0f;
        for (int k = 0; k < 3; ++k) p.scales[i * 3 + k] = raw_sc;
        p.rotations[i * 4 + 0] = 1.0f;
        p.rotations[i * 4 + 1] = 0.0f;
        p.rotations[i * 4 + 2] = 0.0f;
        p.rotations[i * 4 + 3] = 0.0f;
        p.opacities[i] = (i < num_dead) ? raw_opacity_dead : raw_opacity_alive;
    }
    return p;
}

// Convenience: all slots alive at the same opacity.
OwnedRawParams make_params_all_alive(int N, float raw_opacity_alive) {
    return make_params_split(N, 0, raw_opacity_alive, raw_opacity_alive);
}

}  // namespace

// ---------------------------------------------------------------------------
// relocate_gs tests
// ---------------------------------------------------------------------------

TEST(McmcRelocate, NoDeadIsNoOp) {
    // sigmoid(2) ≈ 0.881 > 0.005 → all alive.
    OwnedRawParams p  = make_params_all_alive(20, 2.0f);
    OwnedRawParams p0 = p;  // copy

    int n = mcmc::relocate_gs(p, 0.005f, 42);

    EXPECT_EQ(n, 0);
    EXPECT_EQ(p.count(), 20);
    EXPECT_EQ(p.opacities, p0.opacities);
    EXPECT_EQ(p.positions, p0.positions);
    EXPECT_EQ(p.scales,    p0.scales);
    EXPECT_EQ(p.rotations, p0.rotations);
    EXPECT_EQ(p.sh_coeffs, p0.sh_coeffs);
}

TEST(McmcRelocate, AllDeadIsNoOp) {
    // sigmoid(-10) ≈ 4.5e-5 < 0.005 → all dead, no alive pool.
    OwnedRawParams p  = make_params_all_alive(10, -10.0f);
    OwnedRawParams p0 = p;

    int n = mcmc::relocate_gs(p, 0.005f, 42);

    EXPECT_EQ(n, 0);
    EXPECT_EQ(p.count(), 10);
    EXPECT_EQ(p.opacities, p0.opacities);
}

TEST(McmcRelocate, DeadGetReplacedFromAlive) {
    // 5 dead at front (raw_opacity = -10), 15 alive at back (raw_opacity = 2).
    OwnedRawParams p = make_params_split(20, /*num_dead=*/5, -10.0f, 2.0f);

    int n = mcmc::relocate_gs(p, 0.005f, 42);

    EXPECT_EQ(n, 5);
    EXPECT_EQ(p.count(), 20);

    // Each former-dead slot should now have positive activation-space opacity
    // and a position copied from one of the alive sources (positions 5..19).
    for (int i = 0; i < 5; ++i) {
        const float op_act = sigmoid(p.opacities[i]);
        EXPECT_GT(op_act, 0.0f) << "slot " << i;

        const float x = p.positions[i * 3 + 0];
        bool matched = false;
        for (int j = 5; j < 20; ++j) {
            if (std::fabs(x - static_cast<float>(j)) < 1e-6f) {
                matched = true;
                break;
            }
        }
        EXPECT_TRUE(matched) << "former-dead slot " << i
                             << " has unexpected x=" << x;
    }
}

TEST(McmcRelocate, CopiesAllShCoefficientsForMaxCoeffGreaterThanOne) {
    OwnedRawParams p;
    p.sh_degree = 1;
    p.max_coeffs = 4;
    p.resize(3);

    const float raw_sc = std::log(0.1f);
    for (int i = 0; i < 3; ++i) {
        p.positions[i * 3 + 0] = static_cast<float>(i);
        p.positions[i * 3 + 1] = 0.0f;
        p.positions[i * 3 + 2] = 0.0f;
        for (int k = 0; k < 3; ++k) p.scales[i * 3 + k] = raw_sc;
        p.rotations[i * 4 + 0] = 1.0f;
        p.opacities[i] = (i < 2) ? -10.0f : 2.0f;
    }

    const int mc3 = p.max_coeffs * 3;
    for (int k = 0; k < mc3; ++k) {
        p.sh_coeffs[k] = -1.0f;
        p.sh_coeffs[mc3 + k] = -2.0f;
        p.sh_coeffs[2 * mc3 + k] = 200.0f + static_cast<float>(k);
    }

    int n = mcmc::relocate_gs(p, 0.005f, 42);

    EXPECT_EQ(n, 2);
    for (int dst = 0; dst < 2; ++dst) {
        for (int k = 0; k < mc3; ++k) {
            EXPECT_FLOAT_EQ(p.sh_coeffs[dst * mc3 + k], 200.0f + static_cast<float>(k));
        }
    }
}

TEST(McmcRelocate, Determinism) {
    OwnedRawParams p1 = make_params_split(20, 5, -10.0f, 2.0f);
    OwnedRawParams p2 = p1;  // copy

    mcmc::relocate_gs(p1, 0.005f, 42);
    mcmc::relocate_gs(p2, 0.005f, 42);

    EXPECT_EQ(p1.opacities, p2.opacities);
    EXPECT_EQ(p1.positions, p2.positions);
    EXPECT_EQ(p1.scales,    p2.scales);
    EXPECT_EQ(p1.rotations, p2.rotations);
    EXPECT_EQ(p1.sh_coeffs, p2.sh_coeffs);
}

TEST(McmcRelocate, DifferentSeedsCanDiffer) {
    // With 5 dead and 15 alive (highly skewed sampling), seeds 1 and 999999
    // should produce distinguishable position vectors. (Probabilistic-but-
    // overwhelming: P[identical] < 1e-3.)
    OwnedRawParams p1 = make_params_split(20, 5, -10.0f, 2.0f);
    OwnedRawParams p2 = p1;

    mcmc::relocate_gs(p1, 0.005f, 1u);
    mcmc::relocate_gs(p2, 0.005f, 999999u);

    EXPECT_NE(p1.positions, p2.positions);
}

// ---------------------------------------------------------------------------
// add_new_gs tests
// ---------------------------------------------------------------------------

TEST(McmcAddNewGs, GrowsByFivePercent) {
    OwnedRawParams p = make_params_all_alive(100, 2.0f);
    int added = mcmc::add_new_gs(p, /*cap_max=*/200, 42);
    // floor(1.05 * 100) = 105; 105 - 100 = 5.
    EXPECT_EQ(added, 5);
    EXPECT_EQ(p.count(), 105);
}

TEST(McmcAddNewGs, RespectsCap) {
    OwnedRawParams p = make_params_all_alive(100, 2.0f);
    int added = mcmc::add_new_gs(p, /*cap_max=*/103, 42);
    EXPECT_EQ(added, 3);
    EXPECT_EQ(p.count(), 103);
}

TEST(McmcAddNewGs, NoGrowthAtCap) {
    OwnedRawParams p = make_params_all_alive(100, 2.0f);
    int added = mcmc::add_new_gs(p, /*cap_max=*/100, 42);
    EXPECT_EQ(added, 0);
    EXPECT_EQ(p.count(), 100);
}

TEST(McmcAddNewGs, NoGrowthAboveCap) {
    OwnedRawParams p = make_params_all_alive(100, 2.0f);
    int added = mcmc::add_new_gs(p, /*cap_max=*/50, 42);
    EXPECT_EQ(added, 0);
    EXPECT_EQ(p.count(), 100);
}

TEST(McmcAddNewGs, SmallNTruncatesToZero) {
    OwnedRawParams p = make_params_all_alive(10, 2.0f);
    int added = mcmc::add_new_gs(p, /*cap_max=*/1000, 42);
    // int(1.05 * 10) = int(10.5) = 10; 10 - 10 = 0.
    EXPECT_EQ(added, 0);
    EXPECT_EQ(p.count(), 10);
}

TEST(McmcAddNewGs, AppendedSlotsCopyXyzFromSomeSource) {
    OwnedRawParams p = make_params_all_alive(100, 2.0f);
    int added = mcmc::add_new_gs(p, 200, 42);
    ASSERT_EQ(added, 5);

    // Each appended Gaussian's x must equal one of the original integer x's
    // in [0, 100) (sources are alive sources, all in [0, 99]).
    for (int k = 100; k < 105; ++k) {
        const float x = p.positions[k * 3 + 0];
        const int xi  = static_cast<int>(std::round(x));
        EXPECT_GE(xi, 0);
        EXPECT_LE(xi, 99);
        EXPECT_NEAR(x, static_cast<float>(xi), 1e-4f);
    }
}

TEST(McmcAddNewGs, Determinism) {
    OwnedRawParams p1 = make_params_all_alive(100, 2.0f);
    OwnedRawParams p2 = p1;

    mcmc::add_new_gs(p1, 200, 42);
    mcmc::add_new_gs(p2, 200, 42);

    EXPECT_EQ(p1.count(), p2.count());
    EXPECT_EQ(p1.positions, p2.positions);
    EXPECT_EQ(p1.opacities, p2.opacities);
    EXPECT_EQ(p1.scales,    p2.scales);
}

// ---------------------------------------------------------------------------
// densify (combined) test
// ---------------------------------------------------------------------------

TEST(McmcDensify, DensifyExReportsSourcesAndRelocatedDestinations) {
    OwnedRawParams p = make_params_split(4, /*num_dead=*/3, -10.0f, 2.0f);

    mcmc::DensifyResult r = mcmc::densify_ex(p, 0.005f, /*cap_max=*/4, 42);

    ASSERT_EQ(r.final_count, 4);
    ASSERT_EQ(r.modified_source_indices.size(), 1u);
    EXPECT_EQ(r.modified_source_indices[0], 3);
    ASSERT_EQ(r.replaced_destination_indices.size(), 3u);
    EXPECT_EQ(r.replaced_destination_indices[0], 0);
    EXPECT_EQ(r.replaced_destination_indices[1], 1);
    EXPECT_EQ(r.replaced_destination_indices[2], 2);
}

TEST(McmcDensify, FullSequence) {
    // 100 total: 10 dead at front, 90 alive at back.
    OwnedRawParams p = make_params_split(100, /*num_dead=*/10, -10.0f, 2.0f);

    int final_n = mcmc::densify(p, /*opacity_thresh=*/0.005f,
                                /*cap_max=*/200, /*rng_seed=*/42);

    // relocate keeps count constant (100), then add grows to floor(1.05 * 100) = 105.
    EXPECT_EQ(final_n, 105);
    EXPECT_EQ(p.count(), 105);

    // No remaining dead slots after relocate (they were repopulated).
    int dead = 0;
    for (int i = 0; i < final_n; ++i) {
        if (sigmoid(p.opacities[i]) <= 0.005f) ++dead;
    }
    EXPECT_EQ(dead, 0) << "expected no dead Gaussians after densify";
}

// ---------------------------------------------------------------------------
// Phase 4: integration — VulkanTrainer with MCMC enabled (cap_max > 0).
// Defined in test_mcmc_trainer_integration.cpp (separate file because it
// requires the Vulkan target).
// ---------------------------------------------------------------------------
