// tests/test_data_paths.h — robust resolution of large test data files that
// cannot be git-tracked (basket-aaa.ply ~100 MB, etc.).
//
// Tests should call find_basket_aaa_ply() and GTEST_SKIP if it returns empty,
// instead of hard-coding a single absolute path. The helper searches a list
// of known locations in order and returns the first that exists. Override
// via the BASKET_AAA_PLY env var if you keep the file elsewhere.
//
// Why: developers swap between master and several .claude/worktrees/* copies
// of the repo, and basket-aaa.ply might exist in only some of them. Pinning
// a single hard-coded path (e.g. .../worktrees/vulkan_3d/basket-aaa.ply)
// breaks tests for everyone working in a different worktree.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifndef REPO_ROOT_DIR
#error "REPO_ROOT_DIR must be defined (CMake target_compile_definitions)"
#endif

namespace test_data {

// Search candidates for basket-aaa.ply in priority order:
//   1. $BASKET_AAA_PLY env var (explicit override)
//   2. <current repo root>/basket-aaa.ply (the worktree the test runs from)
//   3. <master repo root>/basket-aaa.ply (canonical location)
//   4. <sibling AAA-Gaussians>/basket-aaa.ply (the dataset's home in the
//      original AAA-Gaussians submodule training tree)
//   5. Each known sibling worktree under .claude/worktrees/ — covers the case
//      where the file lives in only one worktree's tree.
//
// Returns "" if none of the candidates exist.
inline std::string find_basket_aaa_ply() {
    if (const char* env = std::getenv("BASKET_AAA_PLY"); env && *env) {
        return env;
    }
    const std::vector<std::string> candidates = {
        std::string(REPO_ROOT_DIR) + "/basket-aaa.ply",
        "/home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply",
        "/home/robota/h00813233/Graph/AAA-Gaussians/basket-aaa.ply",
        "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply",
        "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/training/basket-aaa.ply",
    };
    for (const auto& p : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) return p;
    }
    return "";
}

// Skip-friendly variant: returns true and writes path to `out` if found,
// false otherwise. Caller can GTEST_SKIP with a useful message.
inline bool resolve_basket_aaa_ply(std::string& out) {
    out = find_basket_aaa_ply();
    return !out.empty();
}

}  // namespace test_data
