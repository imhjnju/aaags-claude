#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include <iostream>

TEST(VkCapabilities, MeetsRequiredMinimums) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed; probe requires a device meeting spec §3.1 minimums";
    const auto& c = ctx.capabilities();

    EXPECT_GE(c.api_version, VK_API_VERSION_1_1);
    EXPECT_GE(c.max_push_constants_size, 128u);
    EXPECT_GE(c.max_compute_workgroup_invocations, 256u);
    EXPECT_GE(c.max_compute_shared_memory_size, 16384u);
    EXPECT_GE(c.max_compute_workgroup_size[0], 256u);
    EXPECT_GE(c.max_compute_workgroup_size[1], 256u);
    EXPECT_GE(c.max_compute_workgroup_size[2], 64u);
    EXPECT_TRUE(c.subgroup_supported_stages & VK_SHADER_STAGE_COMPUTE_BIT);
    EXPECT_GT(c.subgroup_size, 0u);

    // atomic_float informational
    std::cout << "has_shader_atomic_float: "
              << (c.has_shader_atomic_float ? "true" : "false") << "\n";

    ctx.release();
}
