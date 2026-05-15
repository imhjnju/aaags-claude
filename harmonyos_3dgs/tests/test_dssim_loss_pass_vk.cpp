#include "dssim.h"
#include "vulkan/dssim_loss_pass.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void run_dssim_case(int W, int H, float lambda_dssim) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const uint32_t N = static_cast<uint32_t>(W * H * 3);
    std::vector<float> rendered(N);
    std::vector<float> target(N);
    unsigned seed = 911u;
    auto lcg = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return 0.15f + 0.7f * (static_cast<float>(seed >> 16) / 65535.0f);
    };
    for (uint32_t i = 0; i < N; ++i) {
        rendered[i] = lcg();
        target[i] = lcg();
    }

    std::vector<float> cpu_grad(N, 0.0f);
    const float cpu_loss = compute_combined_loss_gradient(
        rendered.data(), target.data(), cpu_grad.data(), W, H, lambda_dssim);

    VulkanBuffer rendered_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer target_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer grad_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer alpha_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer beta_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer gamma_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    const uint32_t partial_count = (N + 255u) / 256u;
    VulkanBuffer partials_buf(ctx, static_cast<VkDeviceSize>(partial_count) * sizeof(float));
    rendered_buf.upload(rendered.data(), rendered.size() * sizeof(float));
    target_buf.upload(target.data(), target.size() * sizeof(float));

    DssimLossPass pass(ctx);
    pass.bind_buffers(rendered_buf.handle(), target_buf.handle(), grad_buf.handle(),
                      partials_buf.handle(), alpha_buf.handle(), beta_buf.handle(), gamma_buf.handle());
    pass.dispatch_sync(static_cast<uint32_t>(W), static_cast<uint32_t>(H), lambda_dssim);

    std::vector<float> gpu_grad(N, 0.0f);
    std::vector<float> partials(partial_count, 0.0f);
    grad_buf.download(gpu_grad.data(), gpu_grad.size() * sizeof(float));
    partials_buf.download(partials.data(), partials.size() * sizeof(float));

    float gpu_loss = lambda_dssim;
    for (float v : partials) gpu_loss += v;
    EXPECT_NEAR(gpu_loss, cpu_loss, 5e-4f);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_NEAR(gpu_grad[i], cpu_grad[i], 4e-3f) << "i=" << i;
    }
}

}  // namespace

TEST(DssimLossPassVulkan, ConstantImageHasZeroLossAndGradient) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const int W = 8;
    const int H = 8;
    const float lambda_dssim = 0.2f;
    const uint32_t N = static_cast<uint32_t>(W * H * 3);
    std::vector<float> rendered(N, 0.5f);
    std::vector<float> target(N, 0.5f);

    VulkanBuffer rendered_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer target_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer grad_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer alpha_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer beta_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer gamma_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    const uint32_t partial_count = (N + 255u) / 256u;
    VulkanBuffer partials_buf(ctx, static_cast<VkDeviceSize>(partial_count) * sizeof(float));
    rendered_buf.upload(rendered.data(), rendered.size() * sizeof(float));
    target_buf.upload(target.data(), target.size() * sizeof(float));

    DssimLossPass pass(ctx);
    pass.bind_buffers(rendered_buf.handle(), target_buf.handle(), grad_buf.handle(),
                      partials_buf.handle(), alpha_buf.handle(), beta_buf.handle(), gamma_buf.handle());
    pass.dispatch_sync(static_cast<uint32_t>(W), static_cast<uint32_t>(H), lambda_dssim);

    std::vector<float> gpu_grad(N, 1.0f);
    std::vector<float> partials(partial_count, 0.0f);
    grad_buf.download(gpu_grad.data(), gpu_grad.size() * sizeof(float));
    partials_buf.download(partials.data(), partials.size() * sizeof(float));

    float gpu_loss = lambda_dssim;
    for (float v : partials) gpu_loss += v;
    EXPECT_NEAR(gpu_loss, 0.0f, 1e-5f);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_NEAR(gpu_grad[i], 0.0f, 1e-5f) << "i=" << i;
    }
}

TEST(DssimLossPassVulkan, MatchesCpuDssimNonMultipleSize) {
    run_dssim_case(17, 11, 0.2f);
}

TEST(DssimLossPassVulkan, MatchesCpuDssimBoundaryHeavySmallImage) {
    run_dssim_case(9, 8, 0.2f);
}

TEST(DssimLossPassVulkan, MatchesCpuDssimHighLambda) {
    run_dssim_case(13, 7, 0.9f);
}
