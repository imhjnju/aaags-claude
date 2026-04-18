#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include <vector>

// Embedded by CMake via xxd -i; see SP-1 Tasks 10 & 11.
#include "hello_spv.h"

namespace { struct PushConst { uint32_t n; }; }

TEST(VkHello, ArrayAdd_EmbeddedSPIRV) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    // Byte-stream VulkanShader overload (T9) + xxd-embedded SPIR-V (T10+T11).
    VulkanShader shader(ctx, static_cast<const uint8_t*>(hello_spv), hello_spv_len);
    VulkanComputePipeline pipe(ctx, shader,
                                /*num_ssbos=*/3,
                                /*push_bytes=*/sizeof(PushConst));

    const uint32_t N = 512;
    VulkanBuffer A(ctx, N * sizeof(float));
    VulkanBuffer B(ctx, N * sizeof(float));
    VulkanBuffer C(ctx, N * sizeof(float));

    std::vector<float> ha(N), hb(N), hc(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) { ha[i] = float(i); hb[i] = 0.5f * float(i); }
    A.upload(ha.data(), N * sizeof(float));
    B.upload(hb.data(), N * sizeof(float));
    C.upload(hc.data(), N * sizeof(float));

    auto ds = pipe.allocateDescriptorSet({A.handle(), B.handle(), C.handle()});
    PushConst pc{N};
    // hello.comp local_size_x=256
    pipe.dispatch_sync(ds, (N + 255) / 256, 1, 1, &pc, sizeof(pc));

    C.download(hc.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(hc[i], ha[i] + hb[i])
            << "mismatch at i=" << i;
}

TEST(VkHello, BoundsCheck_PreventsOverrun) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, static_cast<const uint8_t*>(hello_spv), hello_spv_len);
    VulkanComputePipeline pipe(ctx, shader, 3, sizeof(PushConst));

    const uint32_t N = 256;           // buffer has 256 slots
    const uint32_t effective_n = 10;  // only write first 10
    VulkanBuffer A(ctx, N * sizeof(float));
    VulkanBuffer B(ctx, N * sizeof(float));
    VulkanBuffer C(ctx, N * sizeof(float));
    std::vector<float> ha(N, 1.0f), hb(N, 2.0f), hc(N, -999.0f);
    A.upload(ha.data(), N*4); B.upload(hb.data(), N*4); C.upload(hc.data(), N*4);

    auto ds = pipe.allocateDescriptorSet({A.handle(), B.handle(), C.handle()});
    PushConst pc{effective_n};
    pipe.dispatch_sync(ds, 1, 1, 1, &pc, sizeof(pc));

    C.download(hc.data(), N * sizeof(float));
    for (uint32_t i = 0; i < effective_n; ++i) EXPECT_FLOAT_EQ(hc[i], 3.0f);
    for (uint32_t i = effective_n; i < N; ++i) EXPECT_FLOAT_EQ(hc[i], -999.0f)
        << "shader wrote past effective_n at i=" << i;
}
