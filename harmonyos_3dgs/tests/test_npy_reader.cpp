#include <gtest/gtest.h>
#include "golden/npy_reader.h"

TEST(NpyReader, LoadFloat32_2D) {
    // Pre-generated fixture: float32 array [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    // Saved via numpy.save("tests/golden/fixtures/f32_2x3.npy", arr)
    auto arr = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    EXPECT_EQ(arr.dtype, NpyDtype::float32);
    EXPECT_EQ(arr.shape.size(), 2u);
    EXPECT_EQ(arr.shape[0], 2u);
    EXPECT_EQ(arr.shape[1], 3u);
    EXPECT_EQ(arr.numel(), 6u);
    float* data = arr.f32();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[5], 6.0f);
}

TEST(NpyReader, LoadInt32_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/i32_5.npy");
    EXPECT_EQ(a.dtype, NpyDtype::int32);
    EXPECT_EQ(a.shape, std::vector<size_t>{5});
    EXPECT_EQ(a.i32()[0], -2);
    EXPECT_EQ(a.i32()[4], 2);
}

TEST(NpyReader, LoadUint32_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/u32_4.npy");
    EXPECT_EQ(a.dtype, NpyDtype::uint32);
    EXPECT_EQ(a.u32()[0], 0u);
    EXPECT_EQ(a.u32()[3], 4294967295u);  // UINT32_MAX
}

TEST(NpyReader, LoadUint64_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/u64_3.npy");
    EXPECT_EQ(a.dtype, NpyDtype::uint64);
    EXPECT_EQ(a.u64()[0], 0ULL);
    EXPECT_EQ(a.u64()[2], 18446744073709551615ULL);  // UINT64_MAX
}

TEST(NpyReader, LoadInt64_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/i64_2.npy");
    EXPECT_EQ(a.dtype, NpyDtype::int64);
    EXPECT_EQ(a.i64()[0], -9223372036854775807LL);
    EXPECT_EQ(a.i64()[1],  9223372036854775807LL);
}

TEST(NpyReader, AssertShape_Pass) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    assert_shape(a, {2u, 3u});  // should not throw
}

TEST(NpyReader, AssertShape_FailThrows) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    EXPECT_THROW(assert_shape(a, {3u, 2u}), std::runtime_error);
}

TEST(NpyReader, MissingFileThrows) {
    EXPECT_THROW(load_npy("/nonexistent/path.npy"), std::runtime_error);
}
