# Master Plan

## End Goal

Port AAA-Gaussians (Anti-Aliased 3D Gaussian Splatting) from Python+CUDA to C++17 + **Vulkan Compute** for HarmonyOS NEXT.
Train 2000 steps on basketball dataset, pixel-level consistent with Python reference.
Target: same visual quality (PSNR/SSIM match) and high FPS on HarmonyOS device (Maleoon 920).

## Project Planning Structure

```
Master Plan (this file — strategic direction, milestone table)
  +-- Milestone (dev_notes/master_plan/M{N}_*.md — scoped objective, phases, exit criteria)
        +-- Phase (within milestone — dependency-ordered, ~1-3 sessions each)
```

Detailed Vulkan migration plan: [vulkan_migration_plan.md](vulkan_migration_plan.md)

Key principles:
- Measure first, design second, implement last
- TDD: every compute shader tested against CPU reference before integration
- Final phase is always verification (tests + docs + review + commit)
- Exit criteria must be testable, not vague
- Replan explicitly — update the plan document, don't drift

## Milestones

| # | Name | Status | Sessions | Description |
|---|------|--------|----------|-------------|
| M0 | Foundation | DONE | S1-S2 | Build system, test harness, spec skeleton, verify existing tests pass |
| M1 | Vulkan Infrastructure | IN PROGRESS | S2- | vk_context, buffer, pipeline, shader loading; hello-world compute test |
| M2 | Forward Pipeline | PLANNED | — | Preprocessing, sorting, rasterization compute shaders; match CPU reference |
| M3 | Backward Pipeline | PLANNED | — | Rasterizer + preprocessor backward; gradient verification |
| M4 | Training Pipeline | PLANNED | — | Adam, L1+DSSIM loss, MCMC densification, noise injection; match Python |
| M5 | End-to-End Validation | PLANNED | — | 2000-step basketball training; PSNR/SSIM vs Python reference |
| M6 | HarmonyOS Deployment | PLANNED | — | Cross-compile, Maleoon 920 optimization, on-device validation |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | Adopted dev harness | Proven workflow for agent-driven development |
| 2026-04-16 | Full cycle workflow | Spec-driven approach ensures correctness |
| 2026-04-16 | **OpenCL → Vulkan pivot** | Vulkan 1.3 mandates subgroup ops; better driver guarantees; future on HarmonyOS |
| 2026-04-16 | MCMC densification | Python AAA-Gaussians uses relocate+grow, not clone/split. Must match for consistency. |
| 2026-04-16 | Offline SPIR-V compilation | glslc at build time; no runtime shader compilation overhead |
| 2026-04-16 | Keep CPU as TDD oracle | Vulkan shaders tested against proven CPU implementations |
| 2026-04-16 | Target Vulkan 1.1 SPIR-V | Maximum cross-device compatibility |
| 2026-04-16 | Push constants ≤ 128B | Maleoon 920 minimum guarantee |
