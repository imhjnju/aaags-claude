#include "dssim.h"
#include "vulkan/l1_loss_pass.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

TEST(L1LossPassVulkan, MatchesCpuLambdaZeroNonMultipleSize) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const int W = 17;
    const int H = 11;
    const uint32_t N = static_cast<uint32_t>(W * H * 3);
    std::vector<float> rendered(N);
    std::vector<float> target(N);
    for (uint32_t i = 0; i < N; ++i) {
        rendered[i] = (static_cast<int>(i % 23u) - 11) * 0.03125f;
        target[i] = (static_cast<int>((i * 7u) % 19u) - 9) * 0.025f;
    }
    rendered[5] = target[5];
    rendered[257] = target[257];

    std::vector<float> cpu_grad(N, 0.0f);
    const float cpu_loss = compute_combined_loss_gradient(
        rendered.data(), target.data(), cpu_grad.data(), W, H, 0.0f);

    VulkanBuffer rendered_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer target_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    VulkanBuffer grad_buf(ctx, static_cast<VkDeviceSize>(N) * sizeof(float));
    const uint32_t partial_count = (N + 255u) / 256u;
    VulkanBuffer partials_buf(ctx, static_cast<VkDeviceSize>(partial_count) * sizeof(float));
    rendered_buf.upload(rendered.data(), rendered.size() * sizeof(float));
    target_buf.upload(target.data(), target.size() * sizeof(float));

    L1LossPass pass(ctx);
    pass.bind_buffers(rendered_buf.handle(), target_buf.handle(), grad_buf.handle(), partials_buf.handle());
    pass.dispatch_sync(N);

    std::vector<float> gpu_grad(N, 0.0f);
    std::vector<float> partials(partial_count, 0.0f);
    grad_buf.download(gpu_grad.data(), gpu_grad.size() * sizeof(float));
    partials_buf.download(partials.data(), partials.size() * sizeof(float));

    float gpu_loss = 0.0f;
    for (float v : partials) gpu_loss += v;
    EXPECT_NEAR(gpu_loss, cpu_loss, 1e-6f);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(gpu_grad[i], cpu_grad[i]) << "i=" << i;
    }
}
