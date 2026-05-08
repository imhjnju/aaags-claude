// test_vulkan_adam.cpp -- GPU Adam unit tests (SP-5 Task 1).
//
// Three tests, all skipped if no Vulkan compute device is present:
//
//   T-adam1  : single group, 1 step — GPU matches CPU to 1e-5.
//   T-adam2  : single group, 10 steps — GPU matches CPU to 1e-4.
//   T-adam3  : multiple groups, 1 step each — GPU matches CPU to 1e-5.
//
// CPU reference: CpuAdam (include/cpu_adam.h, src/cpu_adam.cpp).
// GPU implementation: VulkanAdam (include/vulkan/vulkan_adam.h).

#include "vulkan/vulkan_adam.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "cpu_adam.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Fill a std::vector<float> with uniform random values in [lo, hi].
std::vector<float> rand_vec(int n, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& x : v) x = dist(rng);
    return v;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// T-adam1: single group, 1 step.
// N=1000, beta1=0.9, beta2=0.999, eps=1e-15, lr=0.001.
// ---------------------------------------------------------------------------
TEST(VulkanAdam, SingleGroup_OneStep) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr int     N     = 1000;
    constexpr float   beta1 = 0.9f;
    constexpr float   beta2 = 0.999f;
    constexpr float   eps   = 1e-15f;
    constexpr float   lr    = 0.001f;

    auto params_init = rand_vec(N, -1.0f,  1.0f, 1);
    auto grads       = rand_vec(N, -0.1f,  0.1f, 2);

    // --- GPU path ---
    VulkanAdam gpu_adam(ctx, beta1, beta2, eps);
    gpu_adam.add_group(static_cast<uint32_t>(N), lr);

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);
    VulkanBuffer params_buf(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer grad_buf  (ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    params_buf.upload(params_init.data(), bytes);
    grad_buf.upload(grads.data(), bytes);

    // step = 1 (first step, 1-indexed)
    gpu_adam.step_group(0, params_buf.handle(), grad_buf.handle(),
                        lr, /*step=*/1);

    std::vector<float> gpu_params(static_cast<std::size_t>(N));
    params_buf.download(gpu_params.data(), bytes);

    // --- CPU path ---
    std::vector<float> cpu_params = params_init;
    std::vector<float> cpu_m(static_cast<std::size_t>(N), 0.0f);
    std::vector<float> cpu_v(static_cast<std::size_t>(N), 0.0f);

    CpuAdam cpu_adam(beta1, beta2, eps);
    AdamGroup grp;
    grp.params = cpu_params.data();
    grp.grad   = grads.data();
    grp.m      = cpu_m.data();
    grp.v      = cpu_v.data();
    grp.n      = N;
    grp.lr     = lr;
    cpu_adam.add_group(grp);
    cpu_adam.step();   // step_count becomes 1

    // --- Compare ---
    float diff = max_abs_diff(gpu_params, cpu_params);
    EXPECT_LT(diff, 1e-5f)
        << "Max |GPU - CPU| = " << diff << " exceeds 1e-5 tolerance";
}

// ---------------------------------------------------------------------------
// T-adam2: single group, 10 steps.
// N=100, random. Each step uploads a new (but constant across this test)
// gradient. Assert max |GPU - CPU| < 1e-4 after all 10 steps.
// ---------------------------------------------------------------------------
TEST(VulkanAdam, SingleGroup_TenSteps) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr int     N     = 100;
    constexpr float   beta1 = 0.9f;
    constexpr float   beta2 = 0.999f;
    constexpr float   eps   = 1e-15f;
    constexpr float   lr    = 0.001f;
    constexpr int     STEPS = 10;

    auto params_init = rand_vec(N, -1.0f,  1.0f, 10);
    auto grads       = rand_vec(N, -0.1f,  0.1f, 20);

    // --- GPU path ---
    VulkanAdam gpu_adam(ctx, beta1, beta2, eps);
    gpu_adam.add_group(static_cast<uint32_t>(N), lr);

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);
    VulkanBuffer params_buf(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer grad_buf  (ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    params_buf.upload(params_init.data(), bytes);
    grad_buf.upload(grads.data(), bytes);

    for (int s = 1; s <= STEPS; ++s) {
        gpu_adam.step_group(0, params_buf.handle(), grad_buf.handle(),
                            lr, static_cast<uint32_t>(s));
    }

    std::vector<float> gpu_params(static_cast<std::size_t>(N));
    params_buf.download(gpu_params.data(), bytes);

    // --- CPU path ---
    std::vector<float> cpu_params = params_init;
    std::vector<float> cpu_m(static_cast<std::size_t>(N), 0.0f);
    std::vector<float> cpu_v(static_cast<std::size_t>(N), 0.0f);

    CpuAdam cpu_adam(beta1, beta2, eps);
    AdamGroup grp;
    grp.params = cpu_params.data();
    grp.grad   = grads.data();
    grp.m      = cpu_m.data();
    grp.v      = cpu_v.data();
    grp.n      = N;
    grp.lr     = lr;
    cpu_adam.add_group(grp);
    for (int s = 0; s < STEPS; ++s) {
        cpu_adam.step();
    }

    // --- Compare ---
    float diff = max_abs_diff(gpu_params, cpu_params);
    EXPECT_LT(diff, 1e-4f)
        << "Max |GPU - CPU| = " << diff << " exceeds 1e-4 tolerance after "
        << STEPS << " steps";
}

// ---------------------------------------------------------------------------
// T-adam3: multiple groups, 1 step each.
// Groups: n=100 lr=0.001, n=200 lr=0.01, n=50 lr=1e-4.
// Verify each group independently matches CPU Adam with its correct lr.
// ---------------------------------------------------------------------------
TEST(VulkanAdam, MultipleGroups_OneStep) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps   = 1e-15f;

    struct GroupSpec { int n; float lr; };
    const GroupSpec specs[] = {{100, 0.001f}, {200, 0.01f}, {50, 1e-4f}};
    constexpr int NUM_GROUPS = 3;

    VulkanAdam gpu_adam(ctx, beta1, beta2, eps);

    // Pre-generate data for each group.
    std::vector<std::vector<float>> all_params_init(NUM_GROUPS);
    std::vector<std::vector<float>> all_grads(NUM_GROUPS);

    // We need stable pointers, so allocate on the heap.
    std::vector<std::unique_ptr<VulkanBuffer>> owned_params;
    std::vector<std::unique_ptr<VulkanBuffer>> owned_grads;

    for (int g = 0; g < NUM_GROUPS; ++g) {
        int n = specs[g].n;
        gpu_adam.add_group(static_cast<uint32_t>(n), specs[g].lr);

        all_params_init[g] = rand_vec(n, -1.0f, 1.0f,
                                      static_cast<uint32_t>(100 + g));
        all_grads[g]       = rand_vec(n, -0.1f, 0.1f,
                                      static_cast<uint32_t>(200 + g));

        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        owned_params.push_back(std::make_unique<VulkanBuffer>(
            ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        owned_grads.push_back(std::make_unique<VulkanBuffer>(
            ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

        owned_params.back()->upload(all_params_init[g].data(), bytes);
        owned_grads.back()->upload(all_grads[g].data(), bytes);
    }

    // --- GPU: dispatch step 1 for each group ---
    for (int g = 0; g < NUM_GROUPS; ++g) {
        gpu_adam.step_group(g,
                            owned_params[g]->handle(),
                            owned_grads[g]->handle(),
                            specs[g].lr,
                            /*step=*/1);
    }

    // --- CPU: one step per group with separate CpuAdam instances ---
    std::vector<std::vector<float>> cpu_params(NUM_GROUPS);
    std::vector<std::vector<float>> cpu_m(NUM_GROUPS);
    std::vector<std::vector<float>> cpu_v(NUM_GROUPS);

    for (int g = 0; g < NUM_GROUPS; ++g) {
        int n = specs[g].n;
        cpu_params[g] = all_params_init[g];
        cpu_m[g].assign(static_cast<std::size_t>(n), 0.0f);
        cpu_v[g].assign(static_cast<std::size_t>(n), 0.0f);

        CpuAdam cpu_adam(beta1, beta2, eps);
        AdamGroup grp;
        grp.params = cpu_params[g].data();
        grp.grad   = all_grads[g].data();
        grp.m      = cpu_m[g].data();
        grp.v      = cpu_v[g].data();
        grp.n      = n;
        grp.lr     = specs[g].lr;
        cpu_adam.add_group(grp);
        cpu_adam.step();
    }

    // --- Compare each group ---
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int n = specs[g].n;
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        std::vector<float> gpu_params(static_cast<std::size_t>(n));
        owned_params[g]->download(gpu_params.data(), bytes);

        float diff = max_abs_diff(gpu_params, cpu_params[g]);
        EXPECT_LT(diff, 1e-5f)
            << "Group " << g << " (n=" << n << ", lr=" << specs[g].lr
            << "): Max |GPU - CPU| = " << diff << " exceeds 1e-5 tolerance";
    }
}

TEST(VulkanAdam, ExtendGroupPreservesExistingState) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr int N_old = 4;
    constexpr int N_add = 2;
    constexpr float lr = 0.001f;

    auto params_init = rand_vec(N_old, -1.0f, 1.0f, 99);
    auto grads = rand_vec(N_old, -0.1f, 0.1f, 100);

    VulkanAdam adam(ctx, 0.9f, 0.999f, 1e-15f);
    adam.add_group(static_cast<uint32_t>(N_old), lr);

    const std::size_t bytes_old = static_cast<std::size_t>(N_old) * sizeof(float);
    VulkanBuffer params_buf(ctx, bytes_old, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer grad_buf(ctx, bytes_old, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    params_buf.upload(params_init.data(), bytes_old);
    grad_buf.upload(grads.data(), bytes_old);

    adam.step_group(0, params_buf.handle(), grad_buf.handle(), lr, 1);

    std::vector<float> m_before, v_before;
    adam.download_group_moments(0, m_before, v_before);
    ASSERT_EQ(static_cast<int>(m_before.size()), N_old);
    for (int i = 0; i < N_old; ++i) {
        EXPECT_NE(m_before[i], 0.0f);
        EXPECT_NE(v_before[i], 0.0f);
    }

    adam.extend_group(0, static_cast<uint32_t>(N_add));

    std::vector<float> m_after, v_after;
    adam.download_group_moments(0, m_after, v_after);
    ASSERT_EQ(static_cast<int>(m_after.size()), N_old + N_add);
    for (int i = 0; i < N_old; ++i) {
        EXPECT_FLOAT_EQ(m_after[i], m_before[i]);
        EXPECT_FLOAT_EQ(v_after[i], v_before[i]);
    }
    for (int i = N_old; i < N_old + N_add; ++i) {
        EXPECT_EQ(m_after[i], 0.0f);
        EXPECT_EQ(v_after[i], 0.0f);
    }
}

TEST(VulkanAdam, ShrinkGroupPreservesExistingState) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr int N_full = 6;
    constexpr int N_shrink = 3;
    constexpr float lr = 0.001f;

    auto params_init = rand_vec(N_full, -1.0f, 1.0f, 77);
    auto grads = rand_vec(N_full, -0.1f, 0.1f, 78);

    VulkanAdam adam(ctx, 0.9f, 0.999f, 1e-15f);
    adam.add_group(static_cast<uint32_t>(N_full), lr);

    const std::size_t bytes_full = static_cast<std::size_t>(N_full) * sizeof(float);
    VulkanBuffer params_buf(ctx, bytes_full, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer grad_buf(ctx, bytes_full, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    params_buf.upload(params_init.data(), bytes_full);
    grad_buf.upload(grads.data(), bytes_full);

    adam.step_group(0, params_buf.handle(), grad_buf.handle(), lr, 1);

    std::vector<float> m_before, v_before;
    adam.download_group_moments(0, m_before, v_before);
    ASSERT_EQ(static_cast<int>(m_before.size()), N_full);

    adam.shrink_group(0, static_cast<uint32_t>(N_shrink));

    std::vector<float> m_after, v_after;
    adam.download_group_moments(0, m_after, v_after);
    ASSERT_EQ(static_cast<int>(m_after.size()), N_shrink);
    for (int i = 0; i < N_shrink; ++i) {
        EXPECT_FLOAT_EQ(m_after[i], m_before[i]);
        EXPECT_FLOAT_EQ(v_after[i], v_before[i]);
    }
}

TEST(VulkanAdam, ZeroMomentFloatsSelectivelyZeros) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    constexpr int N = 6;
    constexpr float lr = 0.001f;

    auto params_init = rand_vec(N, -1.0f, 1.0f, 55);
    auto grads = rand_vec(N, -0.1f, 0.1f, 56);

    VulkanAdam adam(ctx, 0.9f, 0.999f, 1e-15f);
    adam.add_group(static_cast<uint32_t>(N), lr);

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);
    VulkanBuffer params_buf(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer grad_buf(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    params_buf.upload(params_init.data(), bytes);
    grad_buf.upload(grads.data(), bytes);

    adam.step_group(0, params_buf.handle(), grad_buf.handle(), lr, 1);

    std::vector<float> m_step, v_step;
    adam.download_group_moments(0, m_step, v_step);
    for (int i = 0; i < N; ++i) {
        EXPECT_NE(m_step[i], 0.0f);
    }

    adam.zero_moment_floats(0, {1u, 3u});

    std::vector<float> m_zero, v_zero;
    adam.download_group_moments(0, m_zero, v_zero);

    EXPECT_EQ(m_zero[1], 0.0f);
    EXPECT_EQ(m_zero[3], 0.0f);
    EXPECT_EQ(v_zero[1], 0.0f);
    EXPECT_EQ(v_zero[3], 0.0f);
    EXPECT_NE(m_zero[0], 0.0f);
    EXPECT_NE(m_zero[2], 0.0f);
    EXPECT_NE(m_zero[4], 0.0f);
    EXPECT_NE(m_zero[5], 0.0f);
}
