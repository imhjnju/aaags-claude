# Testing and Validation

## Authority

Use `harmonyos_3dgs/tests/TEST_PLAN.md` as the test inventory and acceptance source. Use `dev_notes/session_state.md` for latest verified counts and known skips.

## Test Layers

| Layer | Purpose | Typical Signal |
|-------|---------|----------------|
| Unit tests | Local behavior of math, loaders, helpers, controllers. | Fast correctness. |
| Vulkan-focused tests | GPU pipeline parity and trainer smoke tests. | Backend correctness. |
| Reference parity tests | Compare C++/Vulkan with Python/CUDA reference. | Behavioral equivalence. |
| Harness runs | End-to-end training/rendering comparisons. | Real scenario confidence. |
| Performance checks | Validate speedup/regression after hot-path changes. | Runtime and equivalence. |

## Standard Commands

Build:

```bash
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build
```

Full CTest:

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure
```

Targeted CTest examples:

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure -R "DSSIM|TrainingStepVk|VulkanTrainer|VkVsCuda|Mcmc|Densification"
```

For DSSIM/loss changes, confirm the current test inventory includes the slow-reference equivalence coverage, then run at least:

```bash
cd harmonyos_3dgs && ctest --test-dir build --output-on-failure -R "DSSIM"
```

For hot-path training changes, pair correctness tests with a before/after timing command or harness run. The exact current comparison harness and flags should be taken from `dev_notes/session_state.md`, `dev_notes/captains_log.md`, or the relevant `harmonyos_3dgs/tools/compare_*.py` script.

Always compare the observed test count with `TEST_PLAN.md` and `session_state.md`.

## When to Run What

| Change Type | Minimum Validation |
|-------------|--------------------|
| Documentation only | Link/readability check; run review for deliverable or milestone-level docs before commit. No build unless docs include commands or state. |
| Tests changed | Rebuild, targeted tests, update test count, full CTest when feasible. |
| Core math/loss/backward changed | Targeted tests, parity tests, full CTest, performance/equivalence if hot path. |
| Vulkan shader changed | Targeted Vulkan parity tests, full CTest, relevant render/train comparison. |
| CLI/harness changed | `py_compile` for Python, smoke run, targeted/full tests as relevant. |
| Performance optimization | Pre/post timing plus output equivalence. |

## Recording Results

Record durable validation in:

- `dev_notes/session_state.md` for current baseline and test counts.
- `dev_notes/captains_log.md` for narrative and key metrics.
- Commit message for concise rationale.
- Wiki only when the validation method itself becomes reusable team knowledge.

## Expected Skips

Some CTest entries may be skipped by design. Do not treat skip count changes as harmless. If skip status changes, record why.

## Performance Validation

For performance-sensitive work:

1. Identify baseline command and metric.
2. Confirm output equivalence before interpreting speed.
3. Separate algorithm runtime from IO, build, data loading, and instrumentation.
4. Record exact command, dataset, steps, flags, and hardware-relevant context.
5. Save durable results in dev notes.

## Common Failure Handling

- If baseline is red before your change, stop and identify whether it is pre-existing.
- If a test fails after your change, fix the cause rather than weakening the test.
- If reference appears wrong, exhaust local bugs first.
- If multiple things changed, reduce to one variable per experiment.

## Test Inventory Maintenance

When CTest count changes:

- Update `harmonyos_3dgs/tests/TEST_PLAN.md`.
- Update `dev_notes/session_state.md`.
- Mention the added/removed test category in `captains_log.md` if milestone-relevant.
