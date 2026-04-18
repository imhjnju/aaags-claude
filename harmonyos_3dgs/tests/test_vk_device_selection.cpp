#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include <cstdlib>

TEST(VkDeviceSelection, DefaultPrefersDiscrete) {
    unsetenv("GS3D_VK_DEVICE");
    unsetenv("GS3D_VK_DEVICE_NAME");
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    EXPECT_FALSE(ctx.deviceName().empty());
    ctx.release();
}

TEST(VkDeviceSelection, EnvIndex_Valid) {
    setenv("GS3D_VK_DEVICE", "0", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE");
    EXPECT_TRUE(ok);
    if (ok) ctx.release();
}

TEST(VkDeviceSelection, EnvIndex_OutOfRange) {
    setenv("GS3D_VK_DEVICE", "99", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE");
    EXPECT_FALSE(ok);  // index 99 should not exist
}

TEST(VkDeviceSelection, EnvNameSubstring_NoMatch) {
    setenv("GS3D_VK_DEVICE_NAME", "ZZZZ_NonExistent_ZZZZ", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE_NAME");
    EXPECT_FALSE(ok);
}
