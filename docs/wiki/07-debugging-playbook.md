# Debugging Playbook

## Debugging Principles

- Observe before theorizing.
- Change one variable per experiment.
- Prefer intermediate-state comparisons over final-output guessing.
- Treat reference mismatch claims as serious; exhaust local bugs first.
- Save evidence in files, not only terminal output.

## First Response to a Regression

1. Confirm the baseline: did it fail before your change?
2. Identify the smallest failing command/test.
3. Record exact command, branch, commit, flags, and artifacts.
4. Compare against spec/reference/tests.
5. Localize first divergence.
6. Fix root cause or leave a visible TODO/task with evidence.

## Parity Debugging Ladder

When Vulkan and CUDA diverge, compare in this order where applicable:

1. Inputs: model, camera, image, config, seed.
2. Forward intermediates: projection, conic, opacity, color, bins, sorted IDs.
3. Render outputs and loss.
4. Backward intermediates.
5. Raw parameter gradients.
6. Adam moments and post-update parameters.
7. Densification/opacity reset effects.
8. Saved PLY and standalone render config.

## Performance Debugging Ladder

When a path is unexpectedly slow:

1. Separate build/setup/IO from runtime.
2. Identify CPU vs GPU time.
3. Compare feature-off vs feature-on runs.
4. Profile algorithmic complexity before micro-optimizing.
5. Preserve output equivalence during optimization.
6. Record before/after timings and command lines.

## Artifact Hygiene

- Keep generated artifacts under workspace/build output locations when possible.
- Do not commit large raw dumps or temporary binary outputs.
- Delete generated files only when confident they are not user-owned.
- Name experiment directories with scenario, date, and key flags.
- Record artifact paths in experiment reports.

## Diagnostic Patch Rules

Diagnostic code is useful but dangerous. Follow `dev_notes/diagnostic_patches/README.md` when present.

At minimum:

- Keep diagnostics narrow and reversible.
- Do not mix diagnostic-only changes with production fixes in one commit unless explicitly intentional.
- Remove or clearly gate diagnostics before finalizing.

## When to Stop and Escalate

Stop and get fresh eyes when:

- Three hypotheses fail.
- A fix does not move the target metric.
- The failing subsystem is unfamiliar this session.
- The next step would be destructive or broad.
- You are tempted to weaken a test.

## Debug Report Template

Use [Investigation Template](templates/investigation.md) for durable debugging records.
