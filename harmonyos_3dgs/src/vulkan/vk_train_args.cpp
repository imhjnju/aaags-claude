#include "vulkan/vk_train_args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

const char* trainingPresetName(TrainingPreset preset) {
    switch (preset) {
    case TrainingPreset::AAA: return "aaa";
    case TrainingPreset::Fast: return "fast";
    }
    return "aaa";
}

static void applyTrainingPreset(TrainArgs& args) {
    if (args.training_preset == TrainingPreset::Fast) {
        if (!args.lambda_dssim_provided) args.lambda_dssim = 0.0f;
        if (!args.opacity_reg_provided) args.opacity_reg = 0.0f;
        if (!args.scale_reg_provided) args.scale_reg = 0.0f;
        if (!args.noise_lr_provided) args.noise_lr = 0.0f;
        if (!args.pos_lr_final_provided) args.pos_lr_final = args.pos_lr_init;
        if (!args.sh_degree_warmup_provided) args.sh_degree_warmup = 0;
    }
}

TrainArgs parseVkTrainArgs(int argc, char** argv) {
    TrainArgs args;
    auto require_value = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc || std::strncmp(argv[i + 1], "--", 2) == 0) {
            args.parse_error = std::string("missing value for ") + flag;
            return nullptr;
        }
        return argv[++i];
    };
    auto parse_int = [&](int& i, const char* flag, int& out) -> bool {
        const char* value = require_value(i, flag);
        if (!value) return false;
        errno = 0;
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
            args.parse_error = std::string("invalid integer for ") + flag + ": " + value;
            return false;
        }
        out = static_cast<int>(parsed);
        return true;
    };
    auto parse_float = [&](int& i, const char* flag, float& out) -> bool {
        const char* value = require_value(i, flag);
        if (!value) return false;
        errno = 0;
        char* end = nullptr;
        float parsed = std::strtof(value, &end);
        if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
            args.parse_error = std::string("invalid finite float for ") + flag + ": " + value;
            return false;
        }
        out = parsed;
        return true;
    };
    auto parse_bool = [&](int& i, const char* flag, bool& out) -> bool {
        int v = 0;
        if (!parse_int(i, flag, v)) return false;
        if (v != 0 && v != 1) {
            args.parse_error = std::string("invalid boolean for ") + flag + ": expected 0 or 1";
            return false;
        }
        out = (v != 0);
        return true;
    };
    auto parse_preset = [&](int& i) -> bool {
        const char* value = require_value(i, "--training_preset");
        if (!value) return false;
        if (std::strcmp(value, "aaa") == 0) {
            args.training_preset = TrainingPreset::AAA;
            return true;
        }
        if (std::strcmp(value, "fast") == 0) {
            args.training_preset = TrainingPreset::Fast;
            return true;
        }
        args.parse_error = std::string("invalid value for --training_preset: expected aaa or fast, got ") + value;
        return false;
    };

    for (int i = 1; i < argc && args.parse_error.empty(); i++) {
        if (std::strcmp(argv[i], "--ply") == 0)            args.ply_path = require_value(i, "--ply");
        else if (std::strcmp(argv[i], "--gt") == 0)        args.gt_path = require_value(i, "--gt");
        else if (std::strcmp(argv[i], "--cameras") == 0)   args.cameras_json = require_value(i, "--cameras");
        else if (std::strcmp(argv[i], "--gt_dir") == 0)    args.gt_dir = require_value(i, "--gt_dir");
        else if (std::strcmp(argv[i], "--view_schedule") == 0) args.view_schedule_path = require_value(i, "--view_schedule");
        else if (std::strcmp(argv[i], "--output") == 0)    args.output_path = require_value(i, "--output");
        else if (std::strcmp(argv[i], "--iterations") == 0) { if (!parse_int(i, "--iterations", args.iterations)) break; }
        else if (std::strcmp(argv[i], "--width") == 0) { if (!parse_int(i, "--width", args.width)) break; }
        else if (std::strcmp(argv[i], "--height") == 0) { if (!parse_int(i, "--height", args.height)) break; }
        else if (std::strcmp(argv[i], "--cam_x") == 0) { if (!parse_float(i, "--cam_x", args.cam_x)) break; }
        else if (std::strcmp(argv[i], "--cam_y") == 0) { if (!parse_float(i, "--cam_y", args.cam_y)) break; }
        else if (std::strcmp(argv[i], "--cam_z") == 0) { if (!parse_float(i, "--cam_z", args.cam_z)) break; }
        else if (std::strcmp(argv[i], "--look_x") == 0) { if (!parse_float(i, "--look_x", args.look_x)) break; }
        else if (std::strcmp(argv[i], "--look_y") == 0) { if (!parse_float(i, "--look_y", args.look_y)) break; }
        else if (std::strcmp(argv[i], "--look_z") == 0) { if (!parse_float(i, "--look_z", args.look_z)) break; }
        else if (std::strcmp(argv[i], "--fov") == 0) { if (!parse_float(i, "--fov", args.fov)) break; }
        else if (std::strcmp(argv[i], "--sh_degree") == 0) { if (!parse_int(i, "--sh_degree", args.sh_degree)) break; }
        else if (std::strcmp(argv[i], "--save_every") == 0) { if (!parse_int(i, "--save_every", args.save_every)) break; }
        else if (std::strcmp(argv[i], "--log_every") == 0) { if (!parse_int(i, "--log_every", args.log_every)) break; }
        else if (std::strcmp(argv[i], "--densify") == 0) { if (!parse_bool(i, "--densify", args.densify)) break; }
        else if (std::strcmp(argv[i], "--densify_from_step") == 0) { if (!parse_int(i, "--densify_from_step", args.densify_from_step)) break; }
        else if (std::strcmp(argv[i], "--densify_until_step") == 0) { if (!parse_int(i, "--densify_until_step", args.densify_until_step)) break; }
        else if (std::strcmp(argv[i], "--densify_interval") == 0) { if (!parse_int(i, "--densify_interval", args.densify_interval)) break; }
        else if (std::strcmp(argv[i], "--cap_max") == 0) {
            if (!parse_int(i, "--cap_max", args.cap_max)) break;
            args.cap_max_provided = true;
        }
        else if (std::strcmp(argv[i], "--opacity_reset_interval") == 0) { if (!parse_int(i, "--opacity_reset_interval", args.opacity_reset_interval)) break; }
        else if (std::strcmp(argv[i], "--eval_3d") == 0) { if (!parse_bool(i, "--eval_3d", args.eval_3D)) break; }
        else if (std::strcmp(argv[i], "--parity_mode") == 0) {
            if (!parse_bool(i, "--parity_mode", args.parity_mode)) break;
            args.parity_mode_provided = true;
        }
        else if (std::strcmp(argv[i], "--proper_ewa") == 0) { if (!parse_bool(i, "--proper_ewa", args.proper_ewa)) break; }
        else if (std::strcmp(argv[i], "--require_all_gt") == 0) { if (!parse_bool(i, "--require_all_gt", args.require_all_gt)) break; }
        else if (std::strcmp(argv[i], "--training_preset") == 0) { if (!parse_preset(i)) break; }
        else if (std::strcmp(argv[i], "--lambda_dssim") == 0) {
            if (!parse_float(i, "--lambda_dssim", args.lambda_dssim)) break;
            args.lambda_dssim_provided = true;
        }
        else if (std::strcmp(argv[i], "--opacity_reg") == 0) {
            if (!parse_float(i, "--opacity_reg", args.opacity_reg)) break;
            args.opacity_reg_provided = true;
        }
        else if (std::strcmp(argv[i], "--scale_reg") == 0) {
            if (!parse_float(i, "--scale_reg", args.scale_reg)) break;
            args.scale_reg_provided = true;
        }
        else if (std::strcmp(argv[i], "--noise_lr") == 0) {
            if (!parse_float(i, "--noise_lr", args.noise_lr)) break;
            args.noise_lr_provided = true;
        }
        else if (std::strcmp(argv[i], "--pos_lr_init") == 0) { if (!parse_float(i, "--pos_lr_init", args.pos_lr_init)) break; }
        else if (std::strcmp(argv[i], "--pos_lr_final") == 0) {
            if (!parse_float(i, "--pos_lr_final", args.pos_lr_final)) break;
            args.pos_lr_final_provided = true;
        }
        else if (std::strcmp(argv[i], "--spatial_lr_scale") == 0) { if (!parse_float(i, "--spatial_lr_scale", args.spatial_lr_scale)) break; }
        else if (std::strcmp(argv[i], "--sh_degree_warmup") == 0) {
            if (!parse_int(i, "--sh_degree_warmup", args.sh_degree_warmup)) break;
            args.sh_degree_warmup_provided = true;
        }
        else args.parse_error = std::string("unknown option: ") + argv[i];
    }
    if (args.parse_error.empty())
        applyTrainingPreset(args);
    return args;
}
