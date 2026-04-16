#include "ply_loader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

void LoadedModel::free() {
    delete[] data.positions;
    delete[] data.sh_coeffs;
    delete[] data.scales;
    delete[] data.rotations;
    delete[] data.opacities;
    delete[] data.filter_3D;
    data.positions = nullptr;
    data.sh_coeffs = nullptr;
    data.scales = nullptr;
    data.rotations = nullptr;
    data.opacities = nullptr;
    data.filter_3D = nullptr;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

LoadedModel loadPly(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open PLY file: ") + path);
    }

    // --- Parse header: collect property names and their indices ---
    int vertex_count = 0;
    bool found_vertex = false;
    std::vector<std::string> prop_names;
    std::string line;

    std::getline(file, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") {
        throw std::runtime_error("Not a PLY file");
    }

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;

        if (line.substr(0, 15) == "element vertex ") {
            vertex_count = std::stoi(line.substr(15));
            found_vertex = true;
        }
        // Collect property names (only "property float <name>")
        if (found_vertex && line.substr(0, 15) == "property float ") {
            prop_names.push_back(line.substr(15));
        }
    }

    if (!found_vertex || vertex_count <= 0) {
        throw std::runtime_error("PLY header: no valid vertex count found");
    }

    int props_per_vertex = (int)prop_names.size();

    // Build name → index map
    std::unordered_map<std::string, int> prop_idx;
    for (int i = 0; i < props_per_vertex; i++)
        prop_idx[prop_names[i]] = i;

    // Verify required properties exist
    auto require = [&](const std::string& name) -> int {
        auto it = prop_idx.find(name);
        if (it == prop_idx.end())
            throw std::runtime_error("PLY missing required property: " + name);
        return it->second;
    };

    int idx_x = require("x"), idx_y = require("y"), idx_z = require("z");
    int idx_dc0 = require("f_dc_0"), idx_dc1 = require("f_dc_1"), idx_dc2 = require("f_dc_2");
    int idx_opacity = require("opacity");
    int idx_s0 = require("scale_0"), idx_s1 = require("scale_1"), idx_s2 = require("scale_2");
    int idx_r0 = require("rot_0"), idx_r1 = require("rot_1"), idx_r2 = require("rot_2"), idx_r3 = require("rot_3");

    // Count f_rest properties to determine SH degree
    int sh_rest_count = 0;
    while (prop_idx.count("f_rest_" + std::to_string(sh_rest_count)))
        sh_rest_count++;

    // Determine SH degree: rest_count = (deg+1)^2 - 1) * 3
    // degree 3: 45, degree 2: 24, degree 1: 9, degree 0: 0
    int sh_degree = 0;
    int sh_rest_per_channel = 0;
    if (sh_rest_count >= 45) { sh_degree = 3; sh_rest_per_channel = 15; }
    else if (sh_rest_count >= 24) { sh_degree = 2; sh_rest_per_channel = 8; }
    else if (sh_rest_count >= 9) { sh_degree = 1; sh_rest_per_channel = 3; }

    int max_coeffs = (sh_degree + 1) * (sh_degree + 1);

    // Collect f_rest indices
    std::vector<int> idx_rest(sh_rest_count);
    for (int i = 0; i < sh_rest_count; i++)
        idx_rest[i] = prop_idx["f_rest_" + std::to_string(i)];

    // Check for AAA-Gaussians filter_3D property
    int idx_filter_3D = -1;
    if (prop_idx.count("filter_3D"))
        idx_filter_3D = prop_idx["filter_3D"];

    // --- Read binary vertex data ---
    std::vector<float> raw(static_cast<size_t>(vertex_count) * props_per_vertex);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("PLY: failed to read all vertex data");
    }

    // --- Allocate output arrays ---
    LoadedModel model;
    model.data.count = vertex_count;
    model.data.sh_degree = sh_degree;
    model.data.max_coeffs = max_coeffs;

    const int N = vertex_count;
    model.data.positions = new float[N * 3];
    model.data.sh_coeffs = new float[N * max_coeffs * 3];
    model.data.scales = new float[N * 3];
    model.data.rotations = new float[N * 4];
    model.data.opacities = new float[N];
    model.data.filter_3D = (idx_filter_3D >= 0) ? new float[N] : nullptr;

    // Initialize SH coeffs to zero (in case of zero-padding)
    std::memset(model.data.sh_coeffs, 0, N * max_coeffs * 3 * sizeof(float));

    // --- Scatter into SoA arrays ---
    for (int i = 0; i < N; ++i) {
        const float* v = raw.data() + (size_t)i * props_per_vertex;

        // Positions
        model.data.positions[i * 3 + 0] = v[idx_x];
        model.data.positions[i * 3 + 1] = v[idx_y];
        model.data.positions[i * 3 + 2] = v[idx_z];

        // SH coefficients: DC + rest with reordering
        float* sh = model.data.sh_coeffs + (size_t)i * max_coeffs * 3;
        sh[0] = v[idx_dc0];
        sh[1] = v[idx_dc1];
        sh[2] = v[idx_dc2];

        // Higher-order SH: reorder from channel-first to basis-interleaved
        // PLY: f_rest_0..N-1 where first sh_rest_per_channel are R, next are G, then B
        // Target: sh[3 + k*3 + c] for basis k, channel c
        for (int k = 0; k < sh_rest_per_channel; ++k) {
            sh[3 + k * 3 + 0] = v[idx_rest[k]];                          // R
            sh[3 + k * 3 + 1] = v[idx_rest[sh_rest_per_channel + k]];    // G
            sh[3 + k * 3 + 2] = v[idx_rest[2 * sh_rest_per_channel + k]]; // B
        }

        // Opacity: sigmoid activation
        model.data.opacities[i] = sigmoid(v[idx_opacity]);

        // Scale: exp activation
        model.data.scales[i * 3 + 0] = std::exp(v[idx_s0]);
        model.data.scales[i * 3 + 1] = std::exp(v[idx_s1]);
        model.data.scales[i * 3 + 2] = std::exp(v[idx_s2]);

        // Rotation: normalize quaternion
        float r0 = v[idx_r0], r1 = v[idx_r1], r2 = v[idx_r2], r3 = v[idx_r3];
        float len = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
        if (len > 0.0f) {
            float inv_len = 1.0f / len;
            r0 *= inv_len; r1 *= inv_len; r2 *= inv_len; r3 *= inv_len;
        }
        model.data.rotations[i * 4 + 0] = r0;
        model.data.rotations[i * 4 + 1] = r1;
        model.data.rotations[i * 4 + 2] = r2;
        model.data.rotations[i * 4 + 3] = r3;

        // AAA-Gaussians filter_3D (already activated, store as-is)
        if (idx_filter_3D >= 0)
            model.data.filter_3D[i] = v[idx_filter_3D];
    }

    return model;
}
