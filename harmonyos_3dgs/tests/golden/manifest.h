#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

struct Artifact {
    std::string filename;
    std::string op;
    std::string tensor;
    std::vector<size_t> shape;
    std::string dtype;
    std::string layout;
};

struct Manifest {
    int step = 0;
    int camera_idx = 0;
    int seed = 0;
    std::string reference_commit;
    std::string config_hash;
    std::vector<Artifact> artifacts;
};

inline Manifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("manifest: cannot open " + path);
    nlohmann::json j; f >> j;
    Manifest m;
    m.step       = j.value("step", 0);
    m.camera_idx = j.value("camera_idx", 0);
    m.seed       = j.value("seed", 0);
    m.reference_commit = j.value("reference_commit", "");
    m.config_hash      = j.value("config_hash", "");
    for (auto& a : j["artifacts"]) {
        Artifact art;
        art.filename = a["filename"];
        art.op       = a["operator"];
        art.tensor   = a["tensor"];
        for (auto& s : a["shape"]) art.shape.push_back(s.get<size_t>());
        art.dtype    = a["dtype"];
        art.layout   = a.value("layout", "row_major");
        m.artifacts.push_back(art);
    }
    return m;
}

inline const Artifact* find_artifact(const Manifest& m, const std::string& op, const std::string& tensor) {
    for (const auto& a : m.artifacts)
        if (a.op == op && a.tensor == tensor) return &a;
    return nullptr;
}
