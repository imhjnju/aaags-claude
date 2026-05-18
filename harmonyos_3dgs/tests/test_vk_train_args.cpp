#include <gtest/gtest.h>

#include "vulkan/vk_train_args.h"

#include <vector>

static TrainArgs parseArgs(std::initializer_list<const char*> values) {
    std::vector<char*> argv;
    argv.reserve(values.size());
    for (const char* value : values)
        argv.push_back(const_cast<char*>(value));
    return parseVkTrainArgs(static_cast<int>(argv.size()), argv.data());
}

TEST(VkTrainArgs, DefaultsUseAaaTrainingPreset) {
    TrainArgs args = parseArgs({"gs3d_vk_train"});

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_EQ(args.training_preset, TrainingPreset::AAA);
    EXPECT_FLOAT_EQ(args.lambda_dssim, 0.2f);
    EXPECT_FLOAT_EQ(args.opacity_reg, 0.01f);
    EXPECT_FLOAT_EQ(args.scale_reg, 0.01f);
    EXPECT_FLOAT_EQ(args.noise_lr, 5e5f);
    EXPECT_FLOAT_EQ(args.pos_lr_init, 1.6e-4f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, 1.6e-6f);
    EXPECT_FLOAT_EQ(args.spatial_lr_scale, 1.0f);
    EXPECT_EQ(args.sh_degree_warmup, 1000);
}

TEST(VkTrainArgs, ExplicitAaaPresetUsesReferenceDefaults) {
    TrainArgs args = parseArgs({"gs3d_vk_train", "--training_preset", "aaa"});

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_EQ(args.training_preset, TrainingPreset::AAA);
    EXPECT_STREQ(trainingPresetName(args.training_preset), "aaa");
    EXPECT_FLOAT_EQ(args.lambda_dssim, 0.2f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, 1.6e-6f);
    EXPECT_EQ(args.sh_degree_warmup, 1000);
}

TEST(VkTrainArgs, FastPresetUsesOldBaselineDefaults) {
    TrainArgs args = parseArgs({"gs3d_vk_train", "--training_preset", "fast"});

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_EQ(args.training_preset, TrainingPreset::Fast);
    EXPECT_STREQ(trainingPresetName(args.training_preset), "fast");
    EXPECT_FLOAT_EQ(args.lambda_dssim, 0.0f);
    EXPECT_FLOAT_EQ(args.opacity_reg, 0.0f);
    EXPECT_FLOAT_EQ(args.scale_reg, 0.0f);
    EXPECT_FLOAT_EQ(args.noise_lr, 0.0f);
    EXPECT_FLOAT_EQ(args.pos_lr_init, 1.6e-4f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, args.pos_lr_init);
    EXPECT_EQ(args.sh_degree_warmup, 0);
}

TEST(VkTrainArgs, FastPresetUsesParsedPositionInitWhenFinalNotProvided) {
    TrainArgs args = parseArgs({"gs3d_vk_train", "--training_preset", "fast", "--pos_lr_init", "0.01"});

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_FLOAT_EQ(args.pos_lr_init, 0.01f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, 0.01f);
}

TEST(VkTrainArgs, ExplicitOverridesWinOverPreset) {
    TrainArgs args = parseArgs({
        "gs3d_vk_train",
        "--training_preset", "fast",
        "--lambda_dssim", "0.2",
        "--opacity_reg", "0.03",
        "--scale_reg", "0.04",
        "--noise_lr", "7",
        "--pos_lr_final", "0.000001",
        "--sh_degree_warmup", "250",
    });

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_FLOAT_EQ(args.lambda_dssim, 0.2f);
    EXPECT_FLOAT_EQ(args.opacity_reg, 0.03f);
    EXPECT_FLOAT_EQ(args.scale_reg, 0.04f);
    EXPECT_FLOAT_EQ(args.noise_lr, 7.0f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, 0.000001f);
    EXPECT_EQ(args.sh_degree_warmup, 250);
}

TEST(VkTrainArgs, ExplicitOverridesWinBeforePresetToo) {
    TrainArgs args = parseArgs({
        "gs3d_vk_train",
        "--lambda_dssim", "0.2",
        "--pos_lr_final", "0.000001",
        "--sh_degree_warmup", "250",
        "--training_preset", "fast",
    });

    ASSERT_TRUE(args.parse_error.empty()) << args.parse_error;
    EXPECT_FLOAT_EQ(args.lambda_dssim, 0.2f);
    EXPECT_FLOAT_EQ(args.pos_lr_final, 0.000001f);
    EXPECT_EQ(args.sh_degree_warmup, 250);
}

TEST(VkTrainArgs, InvalidPresetFails) {
    TrainArgs args = parseArgs({"gs3d_vk_train", "--training_preset", "default"});

    EXPECT_FALSE(args.parse_error.empty());
}
