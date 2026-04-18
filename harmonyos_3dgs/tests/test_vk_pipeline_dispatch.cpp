#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include <vector>

#ifndef ADD_ONE_SPV_PATH
#define ADD_ONE_SPV_PATH "shaders/add_one.spv"
#endif

namespace { struct PushConst { uint32_t count; }; }

TEST(VkPipelineDispatch, DispatchSync_AddOne) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipe(ctx, shader, /*num_ssbos=*/1,
                                /*push_bytes=*/sizeof(PushConst));

    const uint32_t N = 128;
    VulkanBuffer data(ctx, N * sizeof(float));
    std::vector<float> host(N);
    for (uint32_t i = 0; i < N; ++i) host[i] = float(i);
    data.upload(host.data(), N * sizeof(float));

    auto ds = pipe.allocateDescriptorSet({data.handle()});
    PushConst pc{N};
    uint32_t gx = (N + 63) / 64;  // add_one uses local_size_x=64
    pipe.dispatch_sync(ds, gx, 1, 1, &pc, sizeof(pc));

    data.download(host.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(host[i], float(i) + 1.0f);
    // Context / buffer / pipeline / shader destructed in reverse declaration
    // order by RAII -- do NOT call ctx.release() here: it would destroy the
    // device before the dependent objects' destructors run.
}

TEST(VkPipelineDispatch, Record_ExternalCommandBuffer) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipe(ctx, shader, 1, sizeof(PushConst));

    const uint32_t N = 64;
    VulkanBuffer data(ctx, N * sizeof(float));
    std::vector<float> host(N, 2.5f);
    data.upload(host.data(), N * sizeof(float));
    auto ds = pipe.allocateDescriptorSet({data.handle()});

    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    PushConst pc{N};
    pipe.record(cmd, ds, (N + 63) / 64, 1, 1, &pc, sizeof(pc));
    insert_compute_barrier(cmd);
    pipe.record(cmd, ds, (N + 63) / 64, 1, 1, &pc, sizeof(pc));

    vkEndCommandBuffer(cmd);
    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);

    data.download(host.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(host[i], 4.5f);  // 2.5 + 1.0 + 1.0
}

TEST(VkPipelineDispatch, PushConstantOverflowThrows) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    EXPECT_THROW(
        { VulkanComputePipeline big(ctx, shader, 1, /*push_bytes=*/1048576); },
        std::runtime_error);
}
