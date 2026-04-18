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

/// Load a manifest.json file.
///
/// Throws std::runtime_error on:
///   - file open failure
///   - malformed JSON
///   - missing required top-level 'artifacts' array
///   - any artifact missing required fields {filename, operator, tensor, shape, dtype}
///     (error message includes file path and artifact index)
///
/// Optional top-level fields (step, camera_idx, seed, reference_commit, config_hash)
/// default to 0/empty if absent. Per-artifact 'layout' defaults to "row_major".
inline Manifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("manifest: cannot open " + path);
    nlohmann::json j;
    try { f >> j; }
    catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("manifest: failed to parse JSON at " + path + ": " + e.what());
    }

    Manifest m;
    m.step       = j.value("step", 0);
    m.camera_idx = j.value("camera_idx", 0);
    m.seed       = j.value("seed", 0);
    m.reference_commit = j.value("reference_commit", "");
    m.config_hash      = j.value("config_hash", "");

    if (!j.contains("artifacts") || !j["artifacts"].is_array())
        throw std::runtime_error("manifest: '" + path + "' missing or non-array 'artifacts'");

    size_t idx = 0;
    for (auto& a : j["artifacts"]) {
        try {
            Artifact art;
            art.filename = a.at("filename").get<std::string>();
            art.op       = a.at("operator").get<std::string>();
            art.tensor   = a.at("tensor").get<std::string>();
            for (auto& s : a.at("shape")) art.shape.push_back(s.get<size_t>());
            art.dtype    = a.at("dtype").get<std::string>();
            art.layout   = a.value("layout", "row_major");
            m.artifacts.push_back(std::move(art));
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("manifest: '" + path + "' artifact[" +
                                     std::to_string(idx) + "] invalid: " + e.what());
        }
        ++idx;
    }
    return m;
}

/// Find an artifact by (operator, tensor) in a Manifest.
///
/// LIFETIME: returned pointer aliases into `m.artifacts` (a std::vector).
/// It is valid only while `m` is alive AND `m.artifacts` is not mutated.
/// Do NOT call on a temporary:
///     auto* bad = find_artifact(load_manifest(...), "...", "...");  // DANGLING
/// Instead:
///     auto m = load_manifest(...);
///     auto* ok = find_artifact(m, "...", "...");
///
/// Returns nullptr if (op, tensor) not found.
inline const Artifact* find_artifact(const Manifest& m, const std::string& op, const std::string& tensor) {
    for (const auto& a : m.artifacts)
        if (a.op == op && a.tensor == tensor) return &a;
    return nullptr;
}
