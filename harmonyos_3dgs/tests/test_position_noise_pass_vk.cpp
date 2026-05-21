#include "vulkan/position_noise_pass.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

uint32_t hash_u32_host(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float uniform01_host(uint32_t seed) {
    return (static_cast<float>(hash_u32_host(seed) & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

float normal_sample_host(uint32_t gid, uint32_t step, uint32_t component) {
    constexpr float kPi = 3.14159265358979323846f;
    const uint32_t seed0 = gid * 0x9e3779b9u ^ step * 0x85ebca6bu ^ component * 0xc2b2ae35u;
    const uint32_t seed1 = seed0 ^ 0x27d4eb2fu;
    const float u1 = std::max(uniform01_host(seed0), 1e-7f);
    const float u2 = uniform01_host(seed1);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * kPi * u2);
}

std::vector<float> position_noise_expected(std::vector<float> positions,
                                            const std::vector<float>& scales,
                                            const std::vector<float>& rotations,
                                            const std::vector<float>& opacities,
                                            uint32_t step,
                                            float noise_lr,
                                            float pos_lr) {
    const uint32_t N = static_cast<uint32_t>(opacities.size());
    for (uint32_t gid = 0; gid < N; ++gid) {
        const float opacity = opacities[gid];
        const float opacity_factor = 1.0f / (1.0f + std::exp(-100.0f * ((1.0f - opacity) - 0.995f)));
        if (opacity_factor < 1e-6f) continue;

        const uint32_t scale_base = gid * 3u;
        const float sx = scales[scale_base + 0u];
        const float sy = scales[scale_base + 1u];
        const float sz = scales[scale_base + 2u];

        const uint32_t rot_base = gid * 4u;
        const float w = rotations[rot_base + 0u];
        const float x = rotations[rot_base + 1u];
        const float y = rotations[rot_base + 2u];
        const float z = rotations[rot_base + 3u];

        const float r00 = 1.0f - 2.0f * (y * y + z * z);
        const float r01 = 2.0f * (x * y + w * z);
        const float r02 = 2.0f * (x * z - w * y);
        const float r10 = 2.0f * (x * y - w * z);
        const float r11 = 1.0f - 2.0f * (x * x + z * z);
        const float r12 = 2.0f * (y * z + w * x);
        const float r20 = 2.0f * (x * z + w * y);
        const float r21 = 2.0f * (y * z - w * x);
        const float r22 = 1.0f - 2.0f * (x * x + y * y);

        const float l00 = r00 * sx;
        const float l01 = r01 * sy;
        const float l02 = r02 * sz;
        const float l10 = r10 * sx;
        const float l11 = r11 * sy;
        const float l12 = r12 * sz;
        const float l20 = r20 * sx;
        const float l21 = r21 * sy;
        const float l22 = r22 * sz;

        const float scalar = opacity_factor * noise_lr * pos_lr;
        const float e0 = scalar * normal_sample_host(gid, step, 0u);
        const float e1 = scalar * normal_sample_host(gid, step, 1u);
        const float e2 = scalar * normal_sample_host(gid, step, 2u);

        const float s00 = l00 * l00 + l01 * l01 + l02 * l02;
        const float s01 = l00 * l10 + l01 * l11 + l02 * l12;
        const float s02 = l00 * l20 + l01 * l21 + l02 * l22;
        const float s11 = l10 * l10 + l11 * l11 + l12 * l12;
        const float s12 = l10 * l20 + l11 * l21 + l12 * l22;
        const float s22 = l20 * l20 + l21 * l21 + l22 * l22;

        const uint32_t pos_base = gid * 3u;
        positions[pos_base + 0u] += s00 * e0 + s01 * e1 + s02 * e2;
        positions[pos_base + 1u] += s01 * e0 + s11 * e1 + s12 * e2;
        positions[pos_base + 2u] += s02 * e0 + s12 * e1 + s22 * e2;
    }
    return positions;
}

std::vector<float> run_position_noise(VulkanContext& ctx,
                                      const std::vector<float>& positions,
                                      const std::vector<float>& scales,
                                      const std::vector<float>& rotations,
                                      const std::vector<float>& opacities,
                                      uint32_t step,
                                      float noise_lr,
                                      float pos_lr) {
    VulkanBuffer pos_buf(ctx, static_cast<VkDeviceSize>(positions.size() * sizeof(float)));
    VulkanBuffer scales_buf(ctx, static_cast<VkDeviceSize>(scales.size() * sizeof(float)));
    VulkanBuffer rotations_buf(ctx, static_cast<VkDeviceSize>(rotations.size() * sizeof(float)));
    VulkanBuffer opacities_buf(ctx, static_cast<VkDeviceSize>(opacities.size() * sizeof(float)));
    pos_buf.upload(positions.data(), positions.size() * sizeof(float));
    scales_buf.upload(scales.data(), scales.size() * sizeof(float));
    rotations_buf.upload(rotations.data(), rotations.size() * sizeof(float));
    opacities_buf.upload(opacities.data(), opacities.size() * sizeof(float));

    PositionNoisePass pass(ctx);
    pass.bind_buffers(pos_buf.handle(), scales_buf.handle(), rotations_buf.handle(), opacities_buf.handle());
    pass.dispatch_sync(static_cast<uint32_t>(opacities.size()), step, noise_lr, pos_lr);

    std::vector<float> out(positions.size());
    pos_buf.download(out.data(), out.size() * sizeof(float));
    return out;
}

}  // namespace

TEST(PositionNoisePassVulkan, RecordBeforeBindBuffersThrows) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    PositionNoisePass pass(ctx);
    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    EXPECT_THROW(pass.record(cmd, 0u, 0u, 1.0f, 1.0f), std::runtime_error);
    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx.freePrimary(cmd);

    EXPECT_THROW(pass.dispatch_sync(0u, 0u, 1.0f, 1.0f), std::runtime_error);
}

TEST(PositionNoisePassVulkan, MatchesShaderMathAndOpacityGate) {
    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device — skipping.";
    }

    const uint32_t step = 17;
    const float noise_lr = 2.0f;
    const float pos_lr = 0.125f;
    const std::vector<float> positions = {
        0.0f, 0.1f, 0.2f,
        1.0f, 1.1f, 1.2f,
        -0.3f, 0.4f, -0.5f,
        2.0f, -2.0f, 0.25f,
    };
    const std::vector<float> scales = {
        0.7f, 1.3f, 2.1f,
        1.0f, 0.5f, 0.25f,
        1.5f, 0.8f, 0.35f,
        0.9f, 1.7f, 0.6f,
    };
    const std::vector<float> rotations = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.9238795f, 0.3826834f, 0.0f, 0.0f,
        0.8660254f, 0.0f, 0.5f, 0.0f,
        0.7071068f, 0.0f, 0.0f, 0.7071068f,
    };
    const std::vector<float> opacities = {
        0.001f,
        0.999f,
        0.002f,
        0.004f,
    };

    const std::vector<float> expected = position_noise_expected(positions, scales, rotations, opacities,
                                                                step, noise_lr, pos_lr);
    const std::vector<float> actual = run_position_noise(ctx, positions, scales, rotations, opacities,
                                                         step, noise_lr, pos_lr);
    const std::vector<float> repeat = run_position_noise(ctx, positions, scales, rotations, opacities,
                                                         step, noise_lr, pos_lr);

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], repeat[i]) << "determinism[" << i << "]";
        EXPECT_NEAR(actual[i], expected[i], 5e-5f) << "position[" << i << "]";
    }

    for (size_t k = 0; k < 3; ++k) {
        EXPECT_FLOAT_EQ(actual[3u + k], positions[3u + k]) << "high-opacity gate component " << k;
    }

    double moved = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        moved += std::abs(static_cast<double>(actual[i]) - static_cast<double>(positions[i]));
    }
    EXPECT_GT(moved, 1e-4);
}
