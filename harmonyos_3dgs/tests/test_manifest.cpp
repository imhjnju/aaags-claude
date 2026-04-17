#include <gtest/gtest.h>
#include "golden/manifest.h"

TEST(Manifest, Parse) {
    auto m = load_manifest(std::string(TEST_DATA_DIR) + "/golden/fixtures/manifest_example.json");
    EXPECT_EQ(m.step, 1);
    EXPECT_EQ(m.camera_idx, 0);
    EXPECT_EQ(m.seed, 42);
    EXPECT_EQ(m.artifacts.size(), 2u);
    EXPECT_EQ(m.artifacts[0].filename, "preprocess_means2D.npy");
    EXPECT_EQ(m.artifacts[0].op, "preprocess");
    EXPECT_EQ(m.artifacts[0].tensor, "means2D");
    EXPECT_EQ(m.artifacts[0].shape.size(), 2u);
    EXPECT_EQ(m.artifacts[0].shape[0], 100u);
    EXPECT_EQ(m.artifacts[0].dtype, "float32");
}

TEST(Manifest, FindArtifact) {
    auto m = load_manifest(std::string(TEST_DATA_DIR) + "/golden/fixtures/manifest_example.json");
    auto a = find_artifact(m, "preprocess", "means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->filename, "preprocess_means2D.npy");
    auto b = find_artifact(m, "nope", "nope");
    EXPECT_EQ(b, nullptr);
}
