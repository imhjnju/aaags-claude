// Minimal Vulkan compute smoke test, rebuilt on top of gs3d_vk_core RAII
// wrappers. Same semantics as the raw-API version: dispatches add_one.comp
// over N=64 floats and verifies data[i] == i + 1. Exit 0 = pass.

#include "vulkan/vk_buffer.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_pipeline.h"
#include "vulkan/vk_shader.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    VulkanContext ctx;
    if (!ctx.init()) {
        std::fprintf(stderr, "[FAIL] no Vulkan device with compute queue\n");
        return 1;
    }
    std::printf("[info] device: %s (apiVersion %u.%u.%u), compute qf=%u\n",
                ctx.deviceName().c_str(),
                VK_VERSION_MAJOR(ctx.apiVersion()),
                VK_VERSION_MINOR(ctx.apiVersion()),
                VK_VERSION_PATCH(ctx.apiVersion()),
                ctx.computeQueueFamily());

    constexpr uint32_t N = 64;
    VulkanBuffer buf(ctx, N * sizeof(float));
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; ++i) input[i] = static_cast<float>(i);
    buf.upload(input.data(), input.size() * sizeof(float));

    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipeline(ctx, shader, /*num_ssbo_bindings=*/1,
                                   /*push_constant_bytes=*/sizeof(uint32_t));
    VkDescriptorSet dset = pipeline.allocateDescriptorSet({buf.handle()});

    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline.layout(), 0, 1, &dset, 0, nullptr);
    uint32_t count = N;
    vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(count), &count);
    vkCmdDispatch(cmd, (N + 63) / 64, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);

    std::vector<float> output(N);
    buf.download(output.data(), output.size() * sizeof(float));

    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) + 1.0f;
        if (output[i] != expected) {
            if (bad < 5) {
                std::fprintf(stderr, "[FAIL] data[%u] = %f, expected %f\n", i,
                             output[i], expected);
            }
            ++bad;
        }
    }

    if (bad == 0) {
        std::printf("[OK] add_one dispatched on GPU, %u elements verified\n", N);
        return 0;
    }
    std::fprintf(stderr, "[FAIL] %d/%u elements wrong\n", bad, N);
    return 1;
}
