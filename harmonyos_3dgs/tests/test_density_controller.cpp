#include <gtest/gtest.h>
#include "density_controller.h"
#include <cmath>
#include <numeric>

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float inv_sigmoid(float x) { return std::log(x / (1.0f - x)); }

// Helper: create OwnedRawParams with N Gaussians at known positions
static OwnedRawParams make_test_params(int N, int sh_degree = 0) {
    OwnedRawParams p;
    p.sh_degree = sh_degree;
    p.max_coeffs = (sh_degree + 1) * (sh_degree + 1);
    p.resize(N);
    for (int i = 0; i < N; i++) {
        p.positions[i*3+0] = (float)i * 0.5f;
        p.positions[i*3+1] = 0.0f;
        p.positions[i*3+2] = 5.0f;
        // Default scale: small (log(0.1) = -2.3)
        for (int j = 0; j < 3; j++) p.scales[i*3+j] = std::log(0.1f);
        p.rotations[i*4+0] = 1.0f;  // identity quaternion
        for (int j = 1; j < 4; j++) p.rotations[i*4+j] = 0.0f;
        p.opacities[i] = inv_sigmoid(0.5f);  // moderate opacity
        for (int j = 0; j < p.max_coeffs * 3; j++)
            p.sh_coeffs[i * p.max_coeffs * 3 + j] = 1.0f;
    }
    return p;
}

// ============================================================
// OwnedRawParams Tests
// ============================================================

TEST(OwnedRawParams, AppendAndCompact) {
    auto p = make_test_params(3);
    EXPECT_EQ(p.count(), 3);

    // Append one
    float pos[3] = {10, 20, 30};
    float sc[3] = {-1, -1, -1};
    float rot[4] = {1, 0, 0, 0};
    float sh[3] = {2, 2, 2};
    p.append(pos, sc, rot, sh, 0.5f);
    EXPECT_EQ(p.count(), 4);
    EXPECT_FLOAT_EQ(p.positions[9], 10.0f);

    // Compact: remove index 1
    std::vector<bool> mask = {false, true, false, false};
    p.compact(mask);
    EXPECT_EQ(p.count(), 3);
    // Index 0 stays, index 2→1, index 3→2
    EXPECT_FLOAT_EQ(p.positions[0], 0.0f);      // original G0
    EXPECT_FLOAT_EQ(p.positions[3], 1.0f);       // original G2 (was at pos 1.0)
    EXPECT_FLOAT_EQ(p.positions[6], 10.0f);      // appended G3
}

TEST(OwnedRawParams, AsRawRoundtrip) {
    auto p = make_test_params(5);
    RawGaussianParams raw = p.as_raw();
    EXPECT_EQ(raw.count, 5);
    EXPECT_EQ(raw.raw_positions, p.positions.data());

    OwnedRawParams p2;
    p2.from_raw(raw);
    EXPECT_EQ(p2.count(), 5);
    EXPECT_FLOAT_EQ(p2.positions[0], p.positions[0]);
}

// ============================================================
// DensityController Tests
// ============================================================

TEST(DensityController, AccumulateStats) {
    DensityController ctrl;
    int N = 5;
    ctrl.reset(N);

    // Simulate d_means2D gradients
    std::vector<float> d_means2D(N * 2);
    for (int i = 0; i < N; i++) {
        d_means2D[i*2] = (float)(i + 1) * 0.01f;   // dx
        d_means2D[i*2+1] = (float)(i + 1) * 0.02f;  // dy
    }

    ctrl.accumulate_stats(d_means2D.data(), nullptr, N);
    ctrl.accumulate_stats(d_means2D.data(), nullptr, N);

    // After 2 accumulations, denom should be 2
    EXPECT_EQ(ctrl.current_N(), N);
}

TEST(DensityController, CloneSmallGaussians) {
    auto p = make_test_params(5);
    // All have scale=0.1 (small), so clone should work if gradient is high enough

    DensityController ctrl;
    ctrl.reset(5);

    // Give high gradient to G0 and G2
    std::vector<float> d_means2D(10, 0.0f);
    d_means2D[0] = 0.1f; d_means2D[1] = 0.1f;  // G0: norm ≈ 0.14
    d_means2D[4] = 0.1f; d_means2D[5] = 0.1f;  // G2: norm ≈ 0.14
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 5);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.001f;  // don't prune anything

    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);

    // G0 and G2 should be cloned (grad > 0.0002, scale < 0.01*100=1.0)
    // Result: 5 original + 2 cloned = 7
    EXPECT_EQ(new_N, 7);
    EXPECT_EQ(p.count(), 7);

    // Cloned Gaussians should have same position as originals
    EXPECT_FLOAT_EQ(p.positions[5*3+0], p.positions[0*3+0]);  // clone of G0
    EXPECT_FLOAT_EQ(p.positions[6*3+0], p.positions[2*3+0]);  // clone of G2
}

TEST(DensityController, SplitLargeGaussians) {
    auto p = make_test_params(3);
    // Make G1 large (scale=2.0 in log-space → exp(2.0)=7.4 > 0.01*100=1.0)
    for (int j = 0; j < 3; j++) p.scales[1*3+j] = std::log(5.0f);

    DensityController ctrl;
    ctrl.reset(3);

    std::vector<float> d_means2D(6, 0.0f);
    d_means2D[2] = 0.5f; d_means2D[3] = 0.5f;  // G1: high gradient
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 3);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.001f;

    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);

    // G1 should be split: remove original, add 2 new = 3-1+2 = 4
    EXPECT_EQ(new_N, 4);

    // New Gaussians should have smaller scale than original
    // Original log(5.0)=1.609, new should be log(5.0) - log(1.6) ≈ 1.609 - 0.47 ≈ 1.14
    // → exp(1.14) ≈ 3.1, which is < 5.0
    for (int i = 0; i < p.count(); i++) {
        float max_s = 0;
        for (int j = 0; j < 3; j++)
            max_s = std::max(max_s, std::exp(p.scales[i*3+j]));
        // No Gaussian should have the original large scale (5.0)
        // (it was split into smaller ones)
    }
}

TEST(DensityController, PruneLowOpacity) {
    auto p = make_test_params(5);
    // Set G1 and G3 to very low opacity
    p.opacities[1] = inv_sigmoid(0.001f);  // below default min_opacity=0.005
    p.opacities[3] = inv_sigmoid(0.002f);  // below min_opacity

    DensityController ctrl;
    ctrl.reset(5);

    // No gradients → no clone/split, but pruning should still happen
    std::vector<float> d_means2D(10, 0.0f);
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 5);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.005f;

    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);

    // G1 and G3 pruned: 5 - 2 = 3
    EXPECT_EQ(new_N, 3);
    // Remaining should all have opacity > min_opacity
    for (int i = 0; i < p.count(); i++) {
        EXPECT_GT(sigmoid(p.opacities[i]), 0.005f);
    }
}

TEST(DensityController, SingleGaussianClone) {
    // Edge case: N=1, clone should produce N=2
    auto p = make_test_params(1);
    DensityController ctrl;
    ctrl.reset(1);

    std::vector<float> d_means2D = {0.5f, 0.5f};
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 1);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.001f;

    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);
    EXPECT_EQ(new_N, 2);
    EXPECT_FLOAT_EQ(p.positions[0], p.positions[3]);  // clone has same position
}

TEST(DensityController, CloneAndSplitSameStep) {
    // G0: small scale + high grad → clone
    // G1: large scale + high grad → split
    // G2: low grad → no action
    auto p = make_test_params(3);
    // G0 small scale (default log(0.1))
    // G1 large scale
    for (int j = 0; j < 3; j++) p.scales[1*3+j] = std::log(5.0f);
    // G2 stays small

    DensityController ctrl;
    ctrl.reset(3);

    std::vector<float> d_means2D = {0.5f, 0.5f,  // G0: high grad
                                     0.5f, 0.5f,  // G1: high grad
                                     0.0f, 0.0f}; // G2: zero grad
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 3);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.001f;

    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);
    // G0 cloned (+1), G1 split (remove original, +2), G2 untouched
    // Before split: 3 + 1 (clone) = 4
    // Split removes G1 from these 4, adds 2: 4 - 1 + 2 = 5
    EXPECT_EQ(new_N, 5);
}

TEST(DensityController, MultipleDensifyCycles) {
    // Run densify → accumulate → densify again
    auto p = make_test_params(2);
    DensityController ctrl;

    // Cycle 1: both get high gradient → both clone
    ctrl.reset(2);
    std::vector<float> d_means2D = {0.5f, 0.5f, 0.5f, 0.5f};
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 2);

    DensifyConfig cfg;
    cfg.grad_threshold = 0.0002f;
    cfg.min_opacity = 0.001f;

    int n1 = ctrl.densify_and_prune(p, cfg, 100.0f);
    EXPECT_EQ(n1, 4);  // 2 + 2 clones

    // Cycle 2: all 4 get gradient → 4 more clones
    d_means2D.assign(n1 * 2, 0.5f);
    ctrl.accumulate_stats(d_means2D.data(), nullptr, n1);

    int n2 = ctrl.densify_and_prune(p, cfg, 100.0f);
    EXPECT_EQ(n2, 8);  // 4 + 4 clones

    // All positions and opacities should be valid
    for (int i = 0; i < n2; i++) {
        EXPECT_FALSE(std::isnan(p.positions[i*3]));
        EXPECT_FALSE(std::isnan(p.opacities[i]));
    }
}

TEST(DensityController, EmptyAfterPrune) {
    // All Gaussians have very low opacity → all pruned
    auto p = make_test_params(3);
    for (int i = 0; i < 3; i++) p.opacities[i] = inv_sigmoid(0.001f);

    DensityController ctrl;
    ctrl.reset(3);
    std::vector<float> d_means2D(6, 0.0f);
    ctrl.accumulate_stats(d_means2D.data(), nullptr, 3);

    DensifyConfig cfg;
    cfg.min_opacity = 0.005f;
    int new_N = ctrl.densify_and_prune(p, cfg, 100.0f);
    EXPECT_EQ(new_N, 0);
}

TEST(DensityController, ResetOpacity) {
    auto p = make_test_params(3);
    p.opacities[0] = inv_sigmoid(0.9f);
    p.opacities[1] = inv_sigmoid(0.5f);
    p.opacities[2] = inv_sigmoid(0.1f);

    DensityController::reset_opacity(p);

    for (int i = 0; i < 3; i++) {
        float op = sigmoid(p.opacities[i]);
        EXPECT_NEAR(op, 0.01f, 1e-4f);
    }
}
