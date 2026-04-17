#include <gtest/gtest.h>
#include <vector>
#include "golden/compare.h"

TEST(Compare, F32_ExactMatch_Passes) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};
    auto r = compare_f32(a, b, 1e-6f, 1e-4f);
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.max_abs_err, 0.0f);
}

TEST(Compare, F32_SmallDiff_WithinAbsTolerance_Passes) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.000001f, 2.0f, 3.0f};
    auto r = compare_f32(a, b, 1e-5f, 1e-4f);
    EXPECT_TRUE(r.passed);
}

TEST(Compare, F32_LargeDiff_Fails) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.1f, 2.0f, 3.0f};
    auto r = compare_f32(a, b, 1e-6f, 1e-4f);
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.first_bad_index, 0u);
}

TEST(Compare, F32_NearZero_UsesAbsolute) {
    std::vector<float> a{1e-10f, 1e-10f};
    std::vector<float> b{2e-10f, 1e-10f};
    auto r = compare_f32(a, b, 1e-5f, 1e-4f);
    EXPECT_TRUE(r.passed);
}

TEST(Compare, U32_ExactOnly) {
    std::vector<uint32_t> a{0, 1, 2, 3};
    std::vector<uint32_t> b{0, 1, 2, 3};
    EXPECT_TRUE(compare_u32(a, b));
    std::vector<uint32_t> c{0, 1, 99, 3};
    EXPECT_FALSE(compare_u32(a, c));
}

TEST(Compare, U64_ExactOnly) {
    std::vector<uint64_t> a{0ULL, 1ULL, (1ULL << 40)};
    std::vector<uint64_t> b{0ULL, 1ULL, (1ULL << 40)};
    EXPECT_TRUE(compare_u64(a, b));
}

TEST(Compare, MismatchSize_Fails) {
    std::vector<float> a{1.0f, 2.0f};
    std::vector<float> b{1.0f};
    auto r = compare_f32(a, b, 1e-6f, 1e-4f);
    EXPECT_FALSE(r.passed);
}
