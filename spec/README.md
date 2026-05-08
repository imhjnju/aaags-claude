# harmonyos_3dgs — Behavioral Spec

Behavioral specification: what each subsystem does and why, independent of implementation.
This is the source of truth for behavioral correctness.
The Python reference (`AAA-Gaussians/`) is the ground truth for WHAT/WHY; this spec documents it for C++ implementation.

## How to Use This Document

- **Before coding**: Read the relevant section. Understand the contract before implementing.
- **During review**: Check code against these specifications. Mismatches = bugs.
- **When updating**: A spec change requires evidence (test result, reference output, observed behavior).
- **Coverage ratings**: A (verified by test), B (documented from reference), C (inferred/unknown).

---

## 1. Floating-Point Invariants

### Overview
All floating-point operations must use consistent precision across platforms. The C++ port must produce numerically equivalent results to the Python reference when given identical inputs.

### Contracts
- All builds MUST use `-ffp-contract=off`. This prevents FMA contraction on ARM, ensuring `a*b+c` rounds the same way as on x86.
- On cross-compile (HarmonyOS), `-ffp-contract=off` is set in CMakeLists.txt and MUST NOT be removed.
- Float32 precision throughout (no accidental float64 promotion).

### Invariants
- `gcc -ffp-contract=off` result == `clang -ffp-contract=off` result for same input data
- Output matches Python reference within floating-point epsilon (≤ 1e-5 per-pixel RMSE on reference scenes)

---

## 2. Preprocessing (3D → 2D Projection)

### Overview
Projects 3D Gaussians from world space to screen space. Computes 2D covariance matrices, screen-space positions, and per-tile coverage.

### Key Concepts
| Term | Definition |
|------|-----------|
| Gaussian | A 3D ellipsoid defined by position (xyz), rotation (quaternion), scale (xyz), opacity, and SH coefficients |
| 2D projection | The screen-space ellipse obtained by projecting the 3D Gaussian through the view/projection matrix |
| 3D smoothing filter | AAA-Gaussians' key innovation: applies a 3D filter before projection to eliminate aliasing |
| Bounding method | View-space bounding that prevents popping artifacts when Gaussians extend past the frustum |

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| `preprocess()` | B | Projects all Gaussians; fills per-Gaussian 2D cov, screen pos, tile range, alpha |
| 3D smoothing filter | B | Reduces Gaussian size based on view distance; prevents aliasing at low resolution |
| Frustum culling | B | Gaussians fully outside frustum are marked inactive (alpha = 0) |

### Invariants
- A Gaussian extending beyond the view frustum must NOT pop in/out abruptly (bounding method)
- Screen-space coverage is always a positive definite 2D covariance

---

## 3. Spherical Harmonics Evaluation

### Overview
Converts per-Gaussian SH coefficients to RGB color for a given view direction.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| `eval_sh()` | B | Evaluates SH of given degree (0..3) for a view direction; returns RGB |
| Degree 0 | A | Single coefficient per channel; view-independent color |
| Degree 3 | B | 16 coefficients per channel; view-dependent color |

### Invariants
- Output is clamped to [0, 1] after SH evaluation
- Coefficient count per Gaussian = `(degree+1)^2 * 3`

---

## 4. Tile Binner

### Overview
Assigns each active Gaussian to all screen tiles it covers. Produces a flat list of (tile_id, depth, gaussian_id) triples for sorting.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| `bin_tiles()` | B | Enumerates all (tile, gaussian) pairs within tile coverage bounds |

### Invariants
- Every Gaussian appears in every tile it overlaps (no missed tiles = no missing splats)
- Tile coverage is computed from the 2D bounding box of the screen-space ellipse

---

## 5. Sorter

### Overview
Sorts Gaussians within each tile by depth (front-to-back) for correct alpha compositing.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| Radix sort on (tile_id << 32 | depth_key) | B | Produces globally sorted array; tile_id is primary key, depth is secondary |

### Invariants
- Within a tile, Gaussians are ordered front-to-back (smallest depth first)
- Sort must be STABLE within a tile if depths are equal (consistent tie-breaking)

---

## 6. Rasterizer

### Overview
Composites sorted Gaussians tile by tile using front-to-back alpha blending.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| `rasterize()` | B | Processes each tile; composites Gaussians using premultiplied alpha |
| Alpha blending | B | `C_out = C_in + alpha * color * transmittance; T_out = T_in * (1 - alpha)` |
| Early termination | B | Stop compositing when transmittance < 1/255 (fully opaque) |

### Invariants
- Output pixel color is independent of Gaussian ordering WITHIN the same alpha value (deterministic)
- Each tile processed independently (no cross-tile state)
- Training backward replay must use the exact forward compositing order. In eval_3D non-parity mode this includes per-subtile depth permutation and per-pixel HEAD flush order, not just raw tile-sorted IDs.

---

## 7. Loss Functions

### Overview
Compute training loss to drive Gaussian optimization.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| L1 loss | B | Mean absolute error between rendered and ground-truth image |
| SSIM loss | B | Structural similarity metric (windowed) |
| Combined loss | B | `0.8 * L1 + 0.2 * (1 - SSIM)` (default weights from Python reference) |

---

## 8. Optimizer

### Overview
Updates Gaussian parameters using Adam optimizer.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| Adam step | B | `lr`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8` (default) |
| Gradient clipping | C | TBD — check Python reference |

---

## 9. Density Controller

### Overview
Adds/removes Gaussians during training to improve coverage.

### Contracts
| API | Rating | Behavior |
|-----|--------|----------|
| Clone (large gradient, small Gaussian) | B | Duplicate Gaussian at same position |
| Split (large gradient, large Gaussian) | B | Replace with 2 smaller Gaussians sampled from 3D distribution |
| Prune (low opacity or out-of-bounds) | B | Remove Gaussian |
| Densify interval | B | Every 100 iterations (default from Python reference) |

---

## Coverage Status

| Subsystem | Spec Coverage | Notes |
|-----------|--------------|-------|
| Floating-Point Invariants | A | Build flag enforced; verified in tests |
| Preprocessing | B | From Python reference; pending test verification |
| SH Evaluation | B/A | Degree 0 verified; higher degrees pending |
| Tile Binner | B | Pending test verification |
| Sorter | B | Pending test verification |
| Rasterizer | B | Pending test verification |
| Loss Functions | B | Weights from Python reference |
| Optimizer | B | Parameters from Python reference |
| Density Controller | B | Logic from Python reference |
