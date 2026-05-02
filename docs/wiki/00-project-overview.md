# Project Overview

## What This Project Is

`harmonyos_3dgs` is a C++17/CMake port of AAA-Gaussians / Anti-Aliased 3D Gaussian Splatting for HarmonyOS, with Vulkan compute as the active GPU backend. The Python/CUDA implementation in `AAA-Gaussians/` is the reference for behavior and visual quality.

The project goal is to match reference visual quality and training behavior first, then optimize for the target platform.

## Success Criteria

Primary success criteria:

- Vulkan/C++ behavior matches the Python/CUDA reference within accepted numerical tolerances.
- Training, rendering, densification, loss, and optimizer behavior are validated by tests and parity harnesses.
- Performance work preserves output equivalence before claiming speedup.
- Operational knowledge is captured in versioned docs so future humans and agents can continue without rediscovering context.

## Current Direction

For current phase, test counts, latest verified baseline, and active parity status, read:

- `dev_notes/session_state.md`
- `memory/MEMORY.md`
- `dev_notes/captains_log.md`

Do not treat this page as the current-status source. It should remain stable and link to the living status files.

## Project Principles

The current operating principles are:

1. Match reference exactly before optimizing.
2. Tests are the executable specification.
3. Spec-first changes: read behavior before editing code.
4. One variable per experiment during parity/debugging work.
5. No silent deferrals: observed issues must be fixed or made visible.
6. Knowledge must be saved in the right layer, not only in chat.

## Major Knowledge Areas

| Area | Entry Point |
|------|-------------|
| Behavioral contracts | `spec/README.md` |
| Vulkan/CUDA parity work | [Vulkan/CUDA Parity](05-vulkan-cuda-parity.md) |
| Training pipeline | [Training Pipeline](06-training-pipeline.md) |
| Testing and validation | [Testing and Validation](04-testing-and-validation.md) |
| Debug methodology | [Debugging Playbook](07-debugging-playbook.md) |
| Team operations | [Operational Rhythm](08-operational-rhythm.md) |

## Non-Goals for This Wiki

This wiki should not become:

- A second copy of every spec.
- A replacement for test code.
- A chronological dumping ground.
- A place to store temporary run outputs.

Use it to route readers to the correct source, explain how the sources fit together, and document durable operating practices.
