#pragma once

#include "train_types.h"

#include <string>

enum class TrainingPreset {
    AAA,
    Fast,
};

struct TrainArgs {
    const char* ply_path = nullptr;
    const char* gt_path = nullptr;
    const char* cameras_json = nullptr;
    const char* gt_dir = nullptr;
    const char* view_schedule_path = nullptr;
    const char* output_path = "trained.ply";
    int iterations = 30000;
    int width = 512;
    int height = 512;
    float cam_x = 0, cam_y = 0, cam_z = 0;
    float look_x = 0, look_y = 0, look_z = 5;
    float fov = 50.0f;
    int sh_degree = -1;
    int save_every = 0;
    int log_every = 100;
    bool eval_3D = false;
    bool parity_mode = false;
    bool parity_mode_provided = false;
    bool proper_ewa = false;
    bool require_all_gt = false;
    TrainingPreset training_preset = TrainingPreset::AAA;
    float lambda_dssim = VkTrainingConfig{}.lambda_dssim;
    float opacity_reg = VkTrainingConfig{}.opacity_reg;
    float scale_reg = VkTrainingConfig{}.scale_reg;
    float noise_lr = VkTrainingConfig{}.noise_lr;
    float pos_lr_init = VkTrainingConfig{}.pos_lr_init;
    float pos_lr_final = VkTrainingConfig{}.pos_lr_final;
    float spatial_lr_scale = VkTrainingConfig{}.spatial_lr_scale;
    int sh_degree_warmup = VkTrainingConfig{}.sh_degree_warmup;
    bool lambda_dssim_provided = false;
    bool opacity_reg_provided = false;
    bool scale_reg_provided = false;
    bool noise_lr_provided = false;
    bool pos_lr_final_provided = false;
    bool sh_degree_warmup_provided = false;
    bool densify = false;
    int densify_from_step = 500;
    int densify_until_step = 15000;
    int densify_interval = 100;
    int cap_max = 0;
    bool cap_max_provided = false;
    int opacity_reset_interval = 3000;
    std::string parse_error;
};

const char* trainingPresetName(TrainingPreset preset);
TrainArgs parseVkTrainArgs(int argc, char** argv);
