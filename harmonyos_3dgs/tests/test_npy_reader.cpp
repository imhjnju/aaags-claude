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
