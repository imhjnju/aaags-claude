// test_vk_compute.cpp -- TDD gate for Vulkan Phase 1 abstractions.
// Dispatches add_one.comp via VulkanContext / VulkanBuffer / VulkanShader /
// VulkanComputePipeline, verifies each element += 1. If this test passes we
// know the abstraction layer can drive the GPU end-to-end.

#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <gtest/gtest.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace {

void record_and_run_add_one(VulkanContext& ctx,
                            VulkanComputePipeline& pipeline,
                            VkDescriptorSet dset,
                            uint32_t count) {
    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.layout(), 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(count), &count);
    vkCmdDispatch(cmd, (count + 63) / 64, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);
}

}  // namespace

TEST(VkCompute, AddOne_64Elements) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init())
        << "Vulkan context init failed — no compute-capable device?";

    constexpr uint32_t N = 64;
    VulkanBuffer buf(ctx, N * sizeof(float));
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; ++i) input[i] = static_cast<float>(i);
    buf.upload(input.data(), input.size() * sizeof(float));

    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipeline(ctx, shader,
                                   /*num_ssbo_bindings=*/1,
                                   /*push_constant_bytes=*/sizeof(uint32_t));
    VkDescriptorSet dset = pipeline.allocateDescriptorSet({buf.handle()});

    record_and_run_add_one(ctx, pipeline, dset, N);

    std::vector<float> output(N);
    buf.download(output.data(), output.size() * sizeof(float));
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) + 1.0f) << "i=" << i;
    }
}

TEST(VkCompute, AddOne_1024Elements) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    constexpr uint32_t N = 1024;
    VulkanBuffer buf(ctx, N * sizeof(float));
    std::vector<float> input(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) input[i] = static_cast<float>(i) * 0.5f;
    buf.upload(input.data(), input.size() * sizeof(float));

    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipeline(ctx, shader, 1, sizeof(uint32_t));
    VkDescriptorSet dset = pipeline.allocateDescriptorSet({buf.handle()});

    record_and_run_add_one(ctx, pipeline, dset, N);

    std::vector<float> output(N);
    buf.download(output.data(), output.size() * sizeof(float));
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 0.5f + 1.0f)
            << "i=" << i;
    }
}

TEST(VkContext, DeviceQueryableAfterInit) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    EXPECT_NE(ctx.instance(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.physicalDevice(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.device(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.commandPool(), VK_NULL_HANDLE);
    EXPECT_FALSE(ctx.deviceName().empty());
    EXPECT_GE(ctx.apiVersion(), VK_API_VERSION_1_1);
}
