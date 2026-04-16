#include <gtest/gtest.h>
#include "types.h"

TEST(FrameAllocator, AllocateAndReset) {
    FrameAllocator alloc(1024);
    float* a = alloc.allocate_array<float>(10);
    ASSERT_NE(a, nullptr);
    EXPECT_GE(alloc.used(), 40u);

    int* b = alloc.allocate_array<int>(5);
    ASSERT_NE(b, nullptr);

    alloc.reset();
    EXPECT_EQ(alloc.used(), 0u);
}

TEST(FrameAllocator, Alignment) {
    FrameAllocator alloc(4096);
    void* a = alloc.allocate(1, 1);
    float* b = alloc.allocate_array<float>(1);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % 64, 0u);
}

TEST(FrameAllocator, OutOfMemory) {
    FrameAllocator alloc(64);
    EXPECT_THROW(alloc.allocate(128), std::runtime_error);
}

TEST(FrameAllocator, ReuseAfterReset) {
    FrameAllocator alloc(256);
    void* first = alloc.allocate(100);
    alloc.reset();
    void* second = alloc.allocate(100);
    EXPECT_EQ(first, second);
}

TEST(FrameAllocator, CapacityRoundedUp) {
    FrameAllocator alloc(1000);
    EXPECT_EQ(alloc.capacity() % 64, 0u);
    EXPECT_GE(alloc.capacity(), 1000u);
}
