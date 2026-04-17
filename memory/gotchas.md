# Gotchas — Bug Prevention Patterns

Patterns discovered during development. Check after implementing changes.
Canonical source for "don't do X because Y happened."

Cross-references to deeper docs use "See also:" lines.

(None yet — will be populated as bugs are found and root-caused.)

## Known Domain Traps (pre-populated from domain knowledge)

### Floating-Point Determinism
- Build flag `-ffp-contract=off` is MANDATORY on all targets. Without it, FMA instructions produce different results on ARM vs x86, breaking cross-platform numerical matching.
- On cross-compile (HarmonyOS/OHOS), the CMakeLists.txt already forces this flag. Don't remove it.
- See also: `spec/README.md` §Floating-Point Invariants

### Spherical Harmonics Degree
- SH evaluation is degree-dependent. The number of SH coefficients per Gaussian is `(degree+1)^2 * 3`. Mismatched degree between preprocessing and rasterization causes silent output corruption (wrong colors, no crash).

### Tile Binner / Rasterizer Key Ordering
- Tile-based rasterization requires Gaussians to be sorted by (tile_id, depth). Wrong sort order = wrong alpha compositing = incorrect rendering. The sorter must run BEFORE the rasterizer.
