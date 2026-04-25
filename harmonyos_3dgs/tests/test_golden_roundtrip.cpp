#include <gtest/gtest.h>
#include <string>
#include "golden/npy_reader.h"
#include "golden/manifest.h"

namespace {
std::string tiny_root() {
    // TEST_DATA_DIR points at harmonyos_3dgs/tests/test_data; the tiny fixture
    // actually lives under harmonyos_3dgs/tests/golden/tiny/step000001/cam0000
    // (outside the test_data subtree). Step up one level.
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}
}

TEST(GoldenRoundtrip, ManifestLoads) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    EXPECT_EQ(m.step, 1);
    EXPECT_EQ(m.camera_idx, 0);
    EXPECT_GE(m.artifacts.size(), 15u);
}

TEST(GoldenRoundtrip, PreprocessMeans2D) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "preprocess", "means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->dtype, "float32");
    EXPECT_EQ(a->shape.size(), 2u);
    EXPECT_EQ(a->shape[1], 2u);   // [N, 2]

    auto arr = load_npy(tiny_root() + "/" + a->filename);
    EXPECT_EQ(arr.dtype, NpyDtype::float32);
    EXPECT_EQ(arr.shape[0], a->shape[0]);
    EXPECT_EQ(arr.shape[1], 2u);
}

TEST(GoldenRoundtrip, SortKeysSorted_U64) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "sort", "keys_sorted");
    if (a == nullptr) {
        GTEST_SKIP() << "no rendered pairs in tiny fixture (R==0)";
    }
    auto arr = load_npy(tiny_root() + "/" + a->filename);
    EXPECT_EQ(arr.dtype, NpyDtype::uint64);
    // keys must be non-decreasing (sorted) — sanity check
    uint64_t* k = arr.u64();
    for (size_t i = 1; i < arr.numel(); ++i) {
        EXPECT_LE(k[i-1], k[i]) << "unsorted at i=" << i;
    }
}

TEST(GoldenRoundtrip, BackwardDMeans2D_Shape3) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "backward", "d_means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->shape.size(), 2u);
    EXPECT_EQ(a->shape[1], 3u) << "d_means2D must be [P, 3] (CUDA-side allocation)";
}
