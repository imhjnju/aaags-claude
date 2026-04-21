# VK ↔ CUDA eval_3D Parity (PSNR ≥60 dB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the VK-vs-CUDA full-scene `eval_3D` PSNR gap from ~32 dB to ≥60 dB on frozen basketball input, with an automated measurement harness that gates every subsequent change.

**Architecture:** Three phases. Phase 0 builds the harness — extend the existing `tools/render_single.py` (which uses AAA-Gaussians' canonical `MiniCam` + `ExtendedSettings.from_json("aaa.json")` helpers) to dump a raw float32 HWC image + CHW npy + input hash, then a C++ gtest reuses the existing VK `Renderer` path and compares against that golden. Phase 1 dispatches five parallel audit subagents (one per shader concern) to produce a ranked VK↔CUDA divergence list — **including missing VK feature coverage vs `aaa.json` flags** (`proper_ewa_scaling`, `rect_bounding`, `new_aabb`, `hierarchical_4x4_culling`, `tight_opacity_bounding`, `tile_based_culling`, `load_balancing`, sort settings) — then Task 8 writes per-divergence fix tasks *into this plan file*. Phase 2 is a bisect fallback added only if Phase 1 stalls.

**Plan Revision (2026-04-21):** Original Task 1 wrote a new `render_cuda_basketball.py`; that file's `load_camera` had a silent C2W-vs-W2C bug the implementer had to patch mid-execution. Pivoted to extending `tools/render_single.py`, which is already correct (uses `getWorld2View2`, `MiniCam`) and loads the canonical `aaa.json` settings. Consequence: the golden now uses the full AAA feature set (`proper_ewa_scaling=true`, all cullings, `new_aabb`, etc.) rather than being stripped down to VK's current feature set — so the real gap is **feature coverage + numerical divergence**, not just numerical. See the revised Task 1 and the expanded Task 7 audit scope.

**Tech Stack:** C++17, CMake, Vulkan 1.4 + GLSL compute shaders, gtest, stb_image (for diff heatmap PNG), Python 3.10 (`/home/robota/miniconda3/envs/aaa-gs/bin/python3.10`), numpy, torch, AAA-Gaussians `diff_gaussian_rasterization`. GPU: NVIDIA Tegra Thor.

**Reference spec:** `docs/superpowers/specs/2026-04-21-vk-cuda-parity-60db-design.md`

---

## File Structure

### Phase 0 (Harness) — fully concrete tasks below

| Action | Path | Purpose |
|---|---|---|
| Modify | `tools/render_single.py` | Add `--raw-out`, `--npy-out`, `--hash-out` flags; emit HWC float32 raw + CHW npy + SHA256 hash when provided. Default (PNG-only) behavior preserved. |
| Delete | `harmonyos_3dgs/tools/render_cuda_basketball.py` | Superseded by `render_single.py`. |
| Create | `tests/test_vk_vs_cuda_basketball.cpp` | gtest: load PLY + cam 0, run VK `Renderer::render`, load golden, compute PSNR + per-channel + max/mean/p99 abs + diff heatmap PNG. Assert ≥ `kBaselinePSNR` constant. |
| Create/overwrite | `tests/golden/basketball/cam0/cuda_image.raw` | HWC float32 little-endian, shape (960, 720, 3). Generated via `render_single.py --raw-out ...`, checked in (~8 MB). |
| Create/overwrite | `tests/golden/basketball/cam0/cuda_image.npy` | CHW float32. |
| Create/overwrite | `tests/golden/basketball/cam0/golden_hash.txt` | SHA256(ply + cameras.json + aaa.json + render_single.py) — golden-staleness detector. |
| Modify | `harmonyos_3dgs/CMakeLists.txt` | Add `test_vk_vs_cuda_basketball.cpp` to `gs3d_vk_tests`. |

### Phase 1 (Audit + Fixes) — Tasks 7–8 concrete; Tasks 9+ defined by Task 8 from audit output

### Phase 2 (Bisect Fallback) — defined post-Phase-1 if needed

---

## Task 1 (revised): Extend `tools/render_single.py` with raw/npy/hash output

**Why revised:** The original Task 1 wrote a new `render_cuda_basketball.py`. The existing `tools/render_single.py` already loads `basket-aaa.ply` + `cameras.json` + `aaa.json` correctly via AAA-Gaussians' canonical helpers (`MiniCam`, `getWorld2View2`, `ExtendedSettings.from_json`) — bypassing the matrix-convention bug class and matching the authoritative `aaa.json` feature set. Extending the existing tool is the DRY fix.

**Files:**
- Delete: `harmonyos_3dgs/tools/render_cuda_basketball.py`
- Modify: `tools/render_single.py` — add three optional flags (`--raw-out PATH`, `--npy-out PATH`, `--hash-out PATH`); when provided, emit HWC float32 little-endian raw, CHW float32 npy, SHA256 input hash. Default (torchvision PNG save) behavior preserved when flags absent.

- [ ] **Step 1: Delete the superseded script**

```bash
git rm harmonyos_3dgs/tools/render_cuda_basketball.py
```

- [ ] **Step 2: Read the current `tools/render_single.py`**

```bash
cat tools/render_single.py | head -200
```

Identify: (a) where `img = result["render"]` is assigned (this is the CHW float32 tensor we want to dump), (b) where `torchvision.utils.save_image(img, out)` is called (the default save path).

- [ ] **Step 3: Add three CLI flags + dump logic**

Add to the `argparse` block:

```python
parser.add_argument("--raw-out", type=str, default=None,
                    help="If set, also dump HWC float32 little-endian raw bytes to this path.")
parser.add_argument("--npy-out", type=str, default=None,
                    help="If set, also dump CHW float32 .npy to this path.")
parser.add_argument("--hash-out", type=str, default=None,
                    help="If set, also dump SHA256(ply + cameras.json + aaa.json + render_single.py) to this path.")
```

After `img = result["render"]` (the CHW float32 torch tensor on CUDA) and before `torchvision.utils.save_image(...)`, insert:

```python
# Optional raw/npy/hash dumps for VK-vs-CUDA harness (Plan Task 1 revised)
if args.raw_out or args.npy_out or args.hash_out:
    import hashlib
    import numpy as _np
    img_cpu = img.detach().cpu().numpy().astype(_np.float32)  # [3, H, W]
    if args.npy_out:
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.npy_out)), exist_ok=True)
        _np.save(args.npy_out, img_cpu)
        print(f"Saved CHW npy -> {args.npy_out}")
    if args.raw_out:
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.raw_out)), exist_ok=True)
        img_hwc = _np.transpose(img_cpu, (1, 2, 0)).copy()
        img_hwc.astype("<f4").tofile(args.raw_out)
        print(f"Saved HWC raw -> {args.raw_out} ({_os.path.getsize(args.raw_out)} bytes)")
    if args.hash_out:
        def _sha256_file(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.hash_out)), exist_ok=True)
        components = [
            _sha256_file(args.ply),
            _sha256_file(args.cameras),
            _sha256_file(args.config),
            _sha256_file(_os.path.abspath(__file__)),
            str(args.cam_id),
        ]
        final = hashlib.sha256("".join(components).encode()).hexdigest()
        with open(args.hash_out, "w") as fh:
            fh.write(final + "\n")
        print(f"Saved hash -> {args.hash_out}")
```

Also alias `os` at the top of the file if not already done — the existing script already `import os` so just rename references to `_os` OR add `_os = os` right after the existing import. Simpler: use `os` directly (existing import is fine); the `_os` aliasing above is defensive only.

Simplify: if the existing script already imports `os`, use `os.path.dirname(...)`, `os.makedirs(...)`, `os.path.getsize(...)`, `os.path.abspath(...)` directly without the `_os` prefix. Same for `np` if already imported. Drop the `_np = np` aliasing — use `np` directly.

- [ ] **Step 4: Smoke-test — `--help` prints the new flags**

```bash
/home/robota/miniconda3/envs/aaa-gs/bin/python3.10 tools/render_single.py --help
```

Expected: `--raw-out`, `--npy-out`, `--hash-out` appear in the usage. Exit 0.

- [ ] **Step 5: Smoke-test — a render that would save to nowhere (dry-ish)**

Actually skip this; Task 2 will do the real render.

- [ ] **Step 6: Commit**

```bash
git add tools/render_single.py harmonyos_3dgs/tools/render_cuda_basketball.py
git commit -m "refactor(harness): replace render_cuda_basketball.py; extend render_single.py with raw/npy/hash dump flags"
```

---

## Task 2 (revised): Regenerate the CUDA golden via `render_single.py`

**Files:**
- Create/overwrite: `harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.raw`
- Create/overwrite: `harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.npy`
- Create/overwrite: `harmonyos_3dgs/tests/golden/basketball/cam0/golden_hash.txt`

**Important:** The current golden files on disk were generated by the old `render_cuda_basketball.py` (which used `proper_ewa_scaling=false` and no other AAA features). This task overwrites them using `render_single.py` which uses the full `aaa.json` feature set. The file bytes will be different.

- [ ] **Step 1: Note the cameras.json path expected by render_single.py**

`render_single.py` defaults to `cameras.json` at `<worktree>/harmonyos_3dgs/cameras.json`. Verify:

```bash
ls -la harmonyos_3dgs/cameras.json 2>/dev/null || echo "MISSING"
ls -la /home/robota/Downloads/basketball/_sp0_dump_output/cameras.json
```

If `harmonyos_3dgs/cameras.json` is missing, pass `--cameras /home/robota/Downloads/basketball/_sp0_dump_output/cameras.json` in Step 2.

- [ ] **Step 2: Run the extended render_single.py**

```bash
/home/robota/miniconda3/envs/aaa-gs/bin/python3.10 tools/render_single.py \
  --cam_id 0 \
  --output /tmp/cuda_golden_cam0.png \
  --raw-out harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.raw \
  --npy-out harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.npy \
  --hash-out harmonyos_3dgs/tests/golden/basketball/cam0/golden_hash.txt
```

(Add `--cameras <path>` if Step 1 showed `harmonyos_3dgs/cameras.json` is missing.)

Expected:
- Console prints visible-Gaussian count, pixel range, "Saved -> /tmp/cuda_golden_cam0.png"
- Three new lines: "Saved CHW npy", "Saved HWC raw (8294400 bytes)", "Saved hash"
- `cuda_image.raw` is exactly 8,294,400 bytes

- [ ] **Step 3: Spot-check**

```bash
/home/robota/miniconda3/envs/aaa-gs/bin/python3.10 -c "
import numpy as np
img = np.load('harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.npy')
print('shape=', img.shape, 'dtype=', img.dtype,
      'min=', img.min(), 'max=', img.max(), 'mean=', img.mean())
assert img.shape == (3, 960, 720) and img.dtype == np.float32
assert img.max() > 0.1, 'image is all black/near-zero'
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add harmonyos_3dgs/tests/golden/basketball/cam0/
git commit -m "data(golden): regenerate CUDA eval_3D golden via render_single.py + aaa.json"
```

---

## Task 3: Write the failing gtest — `test_vk_vs_cuda_basketball.cpp`

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp`

- [ ] **Step 1: Write the test**

```cpp
// test_vk_vs_cuda_basketball.cpp — Full-scene VK eval_3D vs CUDA golden.
//
// Loads basket-aaa.ply (400000 Gaussians, SH degree 3) @ cam 0 (720x960),
// renders via the Vulkan Renderer path (same as vk_render_main), loads the
// CUDA golden at tests/golden/basketball/cam0/cuda_image.raw, computes
// PSNR + per-channel PSNR + max/mean/p99 abs error, writes a spatial diff
// heatmap PNG to CMAKE_BINARY_DIR, and asserts PSNR >= kBaselinePSNR.
//
// Baseline starts at 5.0 dB (sentinel) and is overwritten in Task 5 after
// measurement. The canonical golden (aaa.json features enabled) differs from
// the prior VK-aligned reference by ~25 dB, so the initial VK-vs-canonical PSNR
// is expected to be substantially lower than the historical 32 dB against the
// old VK-aligned reference. Target: >=60 dB.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "camera_utils.h"
#include "image_io.h"
#include "ply_loader.h"
#include "renderer.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

namespace {

// Initial sentinel — Task 5 overwrites this with floor(measured_PSNR*10)/10 - 0.5
// DO NOT lower this once Task 5 has set it without documenting why.
constexpr float kBaselinePSNR = 5.0f;

constexpr int kW = 720;
constexpr int kH = 960;

const std::string kPlyPath =
    "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply";
const std::string kCamPath =
    "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json";
const std::string kGoldenPath =
    std::string(TEST_DATA_DIR) + "/golden/basketball/cam0/cuda_image.raw";

struct Metrics {
    float psnr;
    float psnr_r, psnr_g, psnr_b;
    float max_abs, mean_abs, p99_abs;
    int num_bad;  // |diff| > 1e-3
};

float psnr_from_mse(float mse) {
    if (mse <= 0.0f) return 100.0f;  // clamp for safety
    return -10.0f * std::log10(mse);
}

Metrics compute_metrics_hwc(const std::vector<float>& vk,
                            const std::vector<float>& cuda) {
    EXPECT_EQ(vk.size(), cuda.size());
    const size_t N = vk.size();
    double sse = 0.0, sse_r = 0.0, sse_g = 0.0, sse_b = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    int num_bad = 0;
    std::vector<float> abs_errs;
    abs_errs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        float d = vk[i] - cuda[i];
        float ad = std::fabs(d);
        float d2 = d * d;
        sse += d2;
        sum_abs += ad;
        if (ad > max_abs) max_abs = ad;
        if (ad > 1e-3f) ++num_bad;
        abs_errs.push_back(ad);
        int ch = static_cast<int>(i % 3);
        if (ch == 0) sse_r += d2;
        else if (ch == 1) sse_g += d2;
        else sse_b += d2;
    }
    std::sort(abs_errs.begin(), abs_errs.end());
    float p99 = abs_errs[static_cast<size_t>(abs_errs.size() * 0.99)];
    const size_t per_ch = N / 3;
    Metrics m{};
    m.psnr     = psnr_from_mse(static_cast<float>(sse / N));
    m.psnr_r   = psnr_from_mse(static_cast<float>(sse_r / per_ch));
    m.psnr_g   = psnr_from_mse(static_cast<float>(sse_g / per_ch));
    m.psnr_b   = psnr_from_mse(static_cast<float>(sse_b / per_ch));
    m.max_abs  = max_abs;
    m.mean_abs = static_cast<float>(sum_abs / N);
    m.p99_abs  = p99;
    m.num_bad  = num_bad;
    return m;
}

void write_diff_heatmap(const std::vector<float>& vk,
                        const std::vector<float>& cuda,
                        const std::string& out_ppm) {
    // HWC float in [0,1] — writePPM will clamp + quantize to 8-bit.
    // Max per pixel across channels; scale so abs_err of 0.05 → full white.
    const float scale = 1.0f / 0.05f;
    std::vector<float> heat(static_cast<size_t>(kW) * kH * 3);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            size_t base = (static_cast<size_t>(y) * kW + x) * 3;
            float dmax = 0.0f;
            for (int c = 0; c < 3; ++c) {
                dmax = std::max(dmax, std::fabs(vk[base + c] - cuda[base + c]));
            }
            float v = std::clamp(dmax * scale, 0.0f, 1.0f);
            heat[base + 0] = v;
            heat[base + 1] = v;
            heat[base + 2] = v;
        }
    }
    writePPM(out_ppm.c_str(), heat.data(), kW, kH);
}

}  // namespace

TEST(VkVsCudaBasketball, Cam0_PsnrAtLeastBaseline) {
    if (!std::filesystem::exists(kPlyPath)) {
        GTEST_SKIP() << "basket-aaa.ply not available at " << kPlyPath;
    }
    if (!std::filesystem::exists(kCamPath)) {
        GTEST_SKIP() << "cameras.json not available at " << kCamPath;
    }
    if (!std::filesystem::exists(kGoldenPath)) {
        GTEST_SKIP() << "CUDA golden not generated — run "
                        "tools/render_single.py --raw-out ... --npy-out ... "
                        "--hash-out ... Missing: "
                     << kGoldenPath;
    }

    VulkanContext ctx;
    if (!ctx.init()) {
        GTEST_SKIP() << "No Vulkan compute device.";
    }

    auto model = loadPly(kPlyPath.c_str());
    Camera cam = loadCameraJson(kCamPath.c_str(), /*cam_id=*/0);
    ASSERT_EQ(cam.width, kW);
    ASSERT_EQ(cam.height, kH);

    RenderConfig cfg{};
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0.0f;
    cfg.sh_degree = model.data.sh_degree;
    cfg.eval_3D = true;
    cfg.antialiasing = false;
    cfg.training = true;  // matches vk_render_main (spec_training=1 in shader)

    const size_t alloc = 4ULL * 1024 * 1024 * 1024;  // 4 GiB for 400k Gaussians
    Renderer renderer(
        std::make_unique<PreprocessorVulkan>(ctx, /*eval_3D=*/true),
        std::make_unique<TileBinnerVulkan>(ctx),
        std::make_unique<SorterVulkan>(ctx),
        std::make_unique<RasterizerVulkan>(ctx, /*eval_3D=*/true),
        alloc);

    std::vector<float> vk_hwc(static_cast<size_t>(kW) * kH * 3, 0.0f);
    renderer.render(model.data, cam, cfg, vk_hwc.data());
    model.free();

    // Load golden
    std::vector<float> golden_hwc(vk_hwc.size());
    std::ifstream in(kGoldenPath, std::ios::binary);
    ASSERT_TRUE(in.good()) << "Cannot open golden: " << kGoldenPath;
    in.read(reinterpret_cast<char*>(golden_hwc.data()),
            static_cast<std::streamsize>(golden_hwc.size() * sizeof(float)));
    ASSERT_EQ(in.gcount(),
              static_cast<std::streamsize>(golden_hwc.size() * sizeof(float)))
        << "Golden file size mismatch";

    Metrics m = compute_metrics_hwc(vk_hwc, golden_hwc);

    std::string diff_ppm =
        std::string(CMAKE_BINARY_DIR) + "/vk_cuda_diff_cam0.ppm";
    write_diff_heatmap(vk_hwc, golden_hwc, diff_ppm);

    std::printf("[VkVsCudaBasketball] PSNR=%.3f dB (R=%.2f G=%.2f B=%.2f)\n",
                m.psnr, m.psnr_r, m.psnr_g, m.psnr_b);
    std::printf("[VkVsCudaBasketball] max_abs=%.5f mean_abs=%.5f "
                "p99_abs=%.5f bad_pixels(>1e-3)=%d\n",
                m.max_abs, m.mean_abs, m.p99_abs, m.num_bad);
    std::printf("[VkVsCudaBasketball] diff heatmap -> %s\n", diff_ppm.c_str());

    EXPECT_GE(m.psnr, kBaselinePSNR)
        << "VK vs CUDA PSNR regressed below baseline " << kBaselinePSNR
        << " dB. Inspect " << diff_ppm;
}
```

- [ ] **Step 2: No extra dependencies — `writePPM` is already declared in `image_io.h`**

The diff heatmap is written as a grayscale PPM via the existing `writePPM(path, float*, W, H)` helper. No new library required.

---

## Task 4: Wire test into CMake

**Files:**
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Locate the existing `gs3d_vk_tests` sources list**

```bash
grep -n "test_forward_pipeline_vk\|gs3d_vk_tests" harmonyos_3dgs/CMakeLists.txt | head -20
```

- [ ] **Step 2: Add the new test to the sources list**

Add the line `tests/test_vk_vs_cuda_basketball.cpp` in the same list that contains `tests/test_forward_pipeline_vk.cpp` (the VK-tests executable), immediately after `test_forward_pipeline_vk.cpp`. Preserve existing indentation and trailing commas/newlines exactly.

- [ ] **Step 3: Verify the test has access to `TEST_DATA_DIR` and `CMAKE_BINARY_DIR` defines**

```bash
grep -n "TEST_DATA_DIR\|CMAKE_BINARY_DIR" harmonyos_3dgs/CMakeLists.txt | head
```

If `gs3d_vk_tests` doesn't yet get these defines, copy the `target_compile_definitions(... TEST_DATA_DIR=... CMAKE_BINARY_DIR=...)` block from the `gs3d_tests` target to `gs3d_vk_tests`.

- [ ] **Step 4: Configure + build**

```bash
cmake --build harmonyos_3dgs/build -j$(nproc) --target gs3d_vk_tests
```

Expected: builds clean. If link error on `stbi_write_png`, confirm STB_IMAGE_WRITE_IMPLEMENTATION is defined exactly once across the test binary (grep for it; if already defined elsewhere in the same target, remove it from the new test and just `#include "stb_image_write.h"`).

- [ ] **Step 5: Run the test to confirm it executes (not asserting PSNR yet — baseline may be off)**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure
```

Expected: test runs, prints PSNR + per-channel + metrics. May fail or pass — next task locks the baseline.

---

## Task 5: Lock baseline PSNR, confirm reproducibility

**Files:**
- Modify: `harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp` (adjust `kBaselinePSNR` only)

- [ ] **Step 1: Run the test three times, record PSNR**

```bash
for i in 1 2 3; do
  ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure \
    2>&1 | grep "PSNR=" | head -1
done
```

Expected: three near-identical PSNR values (ΔPSNR < 0.1 dB across runs; VK is deterministic given identical inputs). If ΔPSNR > 0.1 dB, investigate non-determinism before proceeding — stop the plan and note findings in `investigations/vk_nondeterminism_<date>.md`.

- [ ] **Step 2: Record the observed PSNR**

Copy the first-run PSNR value from Step 1 — call it P0. Expected: P0 ≈ 32.0 dB per user's reported starting point. If significantly higher or lower, that's informative data — note it in the commit message.

- [ ] **Step 3: Set `kBaselinePSNR = floor(P0 * 10) / 10 - 0.5`**

This gives a 0.5 dB regression cushion. Example: P0=32.1 dB → `kBaselinePSNR = 31.5f`. This is the G0 gate. Edit the constant in the test file:

```cpp
constexpr float kBaselinePSNR = <computed value>f;
```

- [ ] **Step 4: Re-run to confirm passing**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Confirm full test suite still green**

```bash
ctest --test-dir harmonyos_3dgs/build --output-on-failure
```

Expected: all prior tests still pass (no regression from the new test + new CMake target additions).

- [ ] **Step 6: Commit**

```bash
git add harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp harmonyos_3dgs/CMakeLists.txt
git commit -m "test(harness): VK vs CUDA eval_3D basketball PSNR regression gate (G0 baseline)"
```

---

## Task 6: Commit the design spec

**Files:**
- The design spec was staged but not yet committed in the brainstorming session.

- [ ] **Step 1: Commit the staged spec**

```bash
git diff --cached --stat docs/superpowers/specs/2026-04-21-vk-cuda-parity-60db-design.md
git commit -m "docs(plan): VK-CUDA eval_3D parity (>=60 dB) design spec"
```

Expected: one file added, commit succeeds.

---

## Task 7: Structural audit — five parallel subagents

**Files:**
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/preprocess.md`
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/rasterize.md`
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/tile_binning.md`
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/sort.md`
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/sh_eval.md`

- [ ] **Step 1: Read the research gate**

```bash
cat .claude/gates/research.md
```

This encodes the project's "understand before coding" discipline. Apply to every fix task the audit spawns.

- [ ] **Step 2: Create the investigations directory**

```bash
mkdir -p investigations/2026-04-21-vk-cuda-divergence-audit
```

- [ ] **Step 3: Dispatch 5 Explore subagents in PARALLEL (single assistant message, 5 tool calls)**

Each subagent is briefed with exact CUDA and VK anchors and produces one report file.

**Subagent 1 — Preprocess:**

```
Scope: preprocess stage numerical divergences VK vs CUDA eval_3D.

Read side-by-side:
- VK: harmonyos_3dgs/src/vulkan/shaders/preprocess.comp (the eval_3D path,
  especially the 3D→2D covariance, dilation_factor, opacity scaling,
  cutoff=min(11.11, 2*log(opacity/ALPHA_THRESHOLD)), and the gauss2screen
  4x4 matrix construction).
- CUDA: AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/
  forward.cu::preprocessCUDA (EVAL_3D=true template branch) +
  consistent_common.cuh::store_gauss2screen + compute_aabb_*.

Produce: investigations/2026-04-21-vk-cuda-divergence-audit/preprocess.md
with one section per divergence candidate. For each: (a) VK line, (b) CUDA
line, (c) exact numerical difference in plain math, (d) a rank:
- HIGH: plausible source of several dB of the 28 dB residual
- MEDIUM: likely sub-dB contribution but still a real divergence
- LOW: cosmetic / unlikely to move PSNR

End with a "Top 3 ranked fixes" summary.
Under 600 words of prose; tables OK.
```

**Subagent 2 — Rasterize / alpha blend:**

```
Scope: rasterize k-buffer + alpha blending divergences.

Read:
- VK: harmonyos_3dgs/src/vulkan/shaders/rasterize.comp (eval_3D branch,
  plane_x/plane_y math, ray-Gaussian intersection, alpha early-exit,
  front-to-back T accumulation, shared-memory batch loop).
- CUDA: forward.cu::renderCUDA (EVAL_3D=true branch) +
  consistent_common.cuh::max_contrib_ray.

Produce investigations/2026-04-21-vk-cuda-divergence-audit/rasterize.md
with the same structure as Subagent 1. Pay particular attention to:
- Early-exit threshold (VK T<? vs CUDA T<0.0001)
- Accumulation order of RGB channels and transmittance
- Use of __frcp_rn / rcp / 1.0/x reciprocals
- Handling of alpha=min(0.99, opacity*exp(-0.5*contrib))
- How opacity is read (VK: conic_opacity[3]; CUDA: conic_opacity.w)
- Matrix transpose / row-major vs column-major on gauss2screen load
- THREADS_PER_GAUSSIAN=4 (CUDA) vs VK's shared-memory layout for g2s

Rank HIGH/MEDIUM/LOW and end with Top 3.
Under 600 words.
```

**Subagent 3 — Tile binning / AABB:**

```
Scope: tile-binning divergences, AABB computation, tiles_touched, scatter
rect math.

Read:
- VK: preprocess.comp (AABB path — both 2D and eval_3D variants, the
  radius_f[N*2] export, cutoff thresholds),
  harmonyos_3dgs/src/vulkan/shaders/scatter.comp (rectangular tile rect
  math, per-tile duplication).
- CUDA: forward.cu::duplicateWithKeys_extended + consistent_common.cuh::
  compute_aabb_view + compute_aabb_screen.

Produce investigations/2026-04-21-vk-cuda-divergence-audit/tile_binning.md.
Key question: does VK use compute_aabb_view or compute_aabb_screen? Does
CUDA default differ? Are tile extents (extent_x, extent_y) computed with
the same cutoff factor and rounding policy? Does VK's tile rect include
the same border-fringe Gaussians as CUDA for boundary-straddling cases?

Rank HIGH/MEDIUM/LOW and end with Top 3.
Under 500 words.
```

**Subagent 4 — Sort semantics:**

```
Scope: 64-bit key sort divergences VK vs CUDA (CUB DeviceRadixSort).

Read:
- VK: harmonyos_3dgs/src/vulkan/shaders/radix_sort_count.comp +
  radix_sort_scatter.comp + prefix_sum.comp. The key format is
  (tile_id:32 | floatBitsToUint(depth):32).
- CUDA: rasterizer_impl.cu lines 213-216 (CUB sort invocation) + CUB
  source if needed for tie-breaking semantics.

Produce investigations/2026-04-21-vk-cuda-divergence-audit/sort.md.
Key questions:
(a) Is tile_id endianness / byte order identical?
(b) Is floatBitsToUint(depth) equivalent to CUB's float-as-uint key? How
    do NEGATIVE depths sort in each? (Gaussians in eval_3D behind the
    camera are culled upstream but verify.)
(c) Are ties (identical tile+depth pairs) broken identically? CUB is
    stable; is VK's radix sort stable?
(d) Bit-radix count: CUDA uses 64-bit via CUB; VK uses how many bits/
    pass and how many passes? Does the total match 64?

Rank HIGH/MEDIUM/LOW and end with Top 3.
Under 500 words.
```

**Subagent 5 — SH evaluation:**

```
Scope: SH (spherical harmonics) evaluation, RGB output clamp, training flag.

Read:
- VK: preprocess.comp (SH constants SH_C0..SH_C3_6, the SH eval block,
  the RGB clamp at end, the spec_training=1 flag, the SH coefficient
  layout [N, max_coeffs, 3]).
- CUDA: forward.cu computeColorFromSH function + its RGB output write.

Produce investigations/2026-04-21-vk-cuda-divergence-audit/sh_eval.md.
Key questions:
(a) Are SH basis constants identical to all 32 bits?
(b) Are SH coefficients indexed in the same order (channel-major vs
    coeff-major)?
(c) Accumulation order of SH terms (parallel summation vs sequential)?
(d) RGB clamp: CUDA clamps RGB >= 0 at the end; VK comment says
    [-1, 11] no upper clamp. That's a divergence if CUDA eval_3D uses
    the same clamp. Is the clamp the same?
(e) Does SH degree 3 evaluate identically (no reordering of the 15 rest
    coefficients)?

Rank HIGH/MEDIUM/LOW and end with Top 3.
Under 500 words.
```

- [ ] **Step 4: Wait for all 5 reports. Do not proceed until all files exist.**

```bash
ls -la investigations/2026-04-21-vk-cuda-divergence-audit/
```

Expected: five .md files, each non-empty.

---

## Task 8: Synthesize audit → extend this plan with fix tasks

**Files:**
- Create: `investigations/2026-04-21-vk-cuda-divergence-audit/synthesis.md`
- Modify: `docs/superpowers/plans/2026-04-21-vk-cuda-parity-60db.md` (this file — append Tasks 9, 10, 11, … below Task 8)

- [ ] **Step 1: Merge the five reports' "Top 3" sections into one combined ranking**

Read all five reports' final summaries. Produce a combined table:

| Rank | Source report | Divergence | Expected PSNR impact |
|---|---|---|---|

HIGH items first (within HIGH, your best judgment of ordering). MEDIUM next. LOW deferred or dropped.

Save to `investigations/2026-04-21-vk-cuda-divergence-audit/synthesis.md`.

- [ ] **Step 2: For each HIGH item, define a fix task in this plan file**

Append to the end of this plan (after this Task 8), using the Task N template below. Each fix task MUST include:

- Exact file + line anchors for the VK change
- Exact file + line anchors for the CUDA reference
- The gate chain: read `.claude/gates/research.md` → spec-read the subsystem
- A TDD cycle: write a buffer-diff micro-test (RED) → fix → buffer-diff passes (GREEN) → harness PSNR delta recorded (informational, not assertion)
- Commit step: single-fix commit, message format `fix(vk): <short> (<audit-rank>, PSNR <before>→<after> dB)`

Template:

```markdown
## Task N: Fix — [<divergence short name>]

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/shaders/<shader>.comp:<line range>`
- Test: `harmonyos_3dgs/tests/test_<subsystem>_vk.cpp` (extend with a new
  `TEST(...)` that diffs the specific buffer produced by this kernel
  against CUDA's equivalent for <N> random inputs, tolerance <epsilon>)

**Divergence evidence:** [quote or cite lines from investigations/.../<report>.md]

**Gate:**
- [ ] Read `.claude/gates/research.md`
- [ ] Read the subsystem spec section at `spec/<subsystem>.md`

**TDD:**
- [ ] Step 1: Write failing buffer-diff test (show the exact test code)
- [ ] Step 2: Run, confirm RED (show exact expected fail message)
- [ ] Step 3: Implement fix (show exact shader diff)
- [ ] Step 4: Run, confirm GREEN
- [ ] Step 5: Run VK-vs-CUDA harness, record PSNR delta
- [ ] Step 6: Confirm no regression on full test suite: `ctest --test-dir harmonyos_3dgs/build`
- [ ] Step 7: Commit
```

- [ ] **Step 3: Update milestones section of this plan**

Add at the top of the appended fix tasks, a "Target milestones" block:

```markdown
### Phase 1 milestones
- After fix N₁: expected PSNR ≥ 40 dB (G1)
- After fix N₂: expected PSNR ≥ 50 dB (G2)
- After fix Nₖ: PSNR ≥ 60 dB (G3, acceptance)
```

- [ ] **Step 4: Commit the plan update + synthesis**

```bash
git add investigations/2026-04-21-vk-cuda-divergence-audit/synthesis.md \
        docs/superpowers/plans/2026-04-21-vk-cuda-parity-60db.md
git commit -m "plan: synthesize VK-CUDA audit → append fix tasks"
```

- [ ] **Step 5: Hand off to the next subagent to execute Task 9 (first fix)**

At this point the plan has gained its concrete per-divergence fix tasks. Execution continues task-by-task using the subagent-driven-development pattern per this plan's header.

---

## Task 9+ (defined by Task 8)

These tasks are produced by Task 8 from the audit synthesis. Do not pre-fabricate them here.

---

## Phase 2 (contingent)

If after executing every Phase 1 fix task the harness still reports PSNR < 60 dB:

- [ ] **Step 1: User approval gate**

STOP. Surface to user: "Phase 1 complete, PSNR = X dB, still < 60 dB. Proceed to Phase 2 bisect (build per-stage buffer-dump instrumentation on both CUDA and VK)? Y/n"

- [ ] **Step 2: If approved, append Phase 2 tasks to this plan**

Phase 2 instruments both pipelines to dump post-preprocess / post-sort / post-tile-range buffers, diffs them, and localizes the remaining residual. Exact task list is defined at that point from the current PSNR signature.

- [ ] **Step 3: If not approved, engage the emergency exit per spec §7**

Document root cause in `investigations/vk_cuda_residual_<date>.md` and finalize with the best-achieved PSNR.

---

## Task 9: Fix `proper_ewa_scaling` in `preprocess.comp`

**Why:** aaa.json sets `proper_ewa_scaling=true` but preprocess.comp line 959 explicitly disables it with an incorrect comment claiming the CUDA golden uses `false`. The CUDA golden uses `aaa.json` which has `"proper_ewa_scaling": true`. This causes every Gaussian's opacity to be wrong by the dilation factor. PSNR impact: HIGH.

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp`

- [ ] **Step 1: Read the research gate**

```bash
cat .claude/gates/research.md
```

- [ ] **Step 2: Read preprocess.comp lines 940–970 (eval_3D opacity section)**

```bash
sed -n '940,970p' harmonyos_3dgs/src/vulkan/shaders/preprocess.comp
```

- [ ] **Step 3: Re-enable opacity dilation in eval_3D path**

Find the disabled line:
```glsl
// opacity_3d *= dilation_factor;  // DISABLED: proper_ewa_scaling=false
```
Replace with:
```glsl
opacity_3d *= dilation_factor;  // proper_ewa_scaling=true (matches aaa.json)
```

- [ ] **Step 4: Fix 2D path — apply convolution_scaling_factor to opacity**

Find the 2D path opacity store (search for `conic_opacity_packed` or `opacity_activated`).

In CUDA (forward_common.h lines 118–142), the 2D path computes:
```cpp
float det_dilated = (cov[0][0]+0.3f)*(cov[1][1]+0.3f) - cov[0][1]*cov[0][1];
float det_orig = cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1];
float convolution_scaling_factor = sqrt(max(0.000025f, det_orig / det_dilated));
conic_opacity.w = opacity * convolution_scaling_factor;
```

In VK's 2D path, locate where `cov2D.x += 0.3; cov2D.z += 0.3;` (the dilation), then find where `opacity_activated` is stored to `conic_opacity_packed`. Add the scaling:
```glsl
// proper_ewa_scaling: scale opacity by sqrt(det_orig / det_dilated)
float det_orig = cov2D_pre_dilate.x * cov2D_pre_dilate.z - cov2D_pre_dilate.y * cov2D_pre_dilate.y;
float det_dilated = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
float convolution_scaling_factor = sqrt(max(0.000025, det_orig / det_dilated));
opacity_activated *= convolution_scaling_factor;
```
You will need to capture `cov2D` before the dilation (e.g., `vec3 cov2D_pre_dilate = cov2D;`) immediately before the `+= 0.3` lines.

- [ ] **Step 5: Build**

```bash
cmake --build harmonyos_3dgs/build -j$(nproc) --target gs3d_vk_tests
```

- [ ] **Step 6: Run PSNR harness — record new PSNR**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure 2>&1 | grep "PSNR="
```

Expected: PSNR increases (was 25.308 dB). If PSNR drops, revert and investigate.

- [ ] **Step 7: Run full suite**

```bash
ctest --test-dir harmonyos_3dgs/build --output-on-failure 2>&1 | tail -10
```

- [ ] **Step 8: Update kBaselinePSNR if PSNR improved**

If new PSNR P1 > P0 (25.308), set `kBaselinePSNR = floor(P1*10)/10 - 0.5` in the test file.

- [ ] **Step 9: Commit**

```bash
git add harmonyos_3dgs/src/vulkan/shaders/preprocess.comp harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp
git commit -m "fix(preprocess): enable proper_ewa_scaling — opacity*dilation_factor in eval_3D and 2D paths (PSNR delta: +X.X dB)"
```
Fill in the measured PSNR delta.

---

## Task 10: Fix `rect_bounding` + `tight_opacity_bounding` in `preprocess.comp` 2D path

**Why:** aaa.json sets `rect_bounding=true` and `tight_opacity_bounding=true`. VK uses an isotropic radius for tile AABB in the 2D path and hardcodes extent=3.33. This causes VK to over-cover tiles for elongated Gaussians, producing different tile-pair sets than CUDA. PSNR impact: MEDIUM (compound tile coverage error).

Note: the eval_3D path already stores asymmetric extents correctly (lines 1074–1075).

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp`

- [ ] **Step 1: Read the research gate**

```bash
cat .claude/gates/research.md
```

- [ ] **Step 2: Read preprocess.comp lines 1140–1215 (2D path AABB/radius section)**

```bash
sed -n '1140,1215p' harmonyos_3dgs/src/vulkan/shaders/preprocess.comp
```

- [ ] **Step 3: Implement tight_opacity_bounding extent**

Find:
```glsl
float radius_f = 3.33 * sqrt(lambda);
```
Replace with (compute adaptive extent based on opacity):
```glsl
// tight_opacity_bounding=true: extent = min(3.33, sqrt(2 * log(opacity / ALPHA_THRESHOLD)))
// ALPHA_THRESHOLD = 1.0/255.0
const float ALPHA_THRESHOLD = 1.0 / 255.0;
float opacity_power_threshold = -log(ALPHA_THRESHOLD / opacity_activated);
float extent_tight = min(3.33, sqrt(max(0.0, 2.0 * opacity_power_threshold)));
float radius_f = extent_tight * sqrt(lambda);
```

- [ ] **Step 4: Implement rect_bounding asymmetric extents**

Find the isotropic rectMin/rectMax calls:
```glsl
ivec2 rmn = rectMin(pix, radius_f, radius_f, grid_x, grid_y, TILE_W, TILE_H);
ivec2 rmx = rectMax(pix, radius_f, radius_f, grid_x, grid_y, TILE_W, TILE_H);
```

Replace with:
```glsl
// rect_bounding=true: use per-axis extents fitted to covariance ellipse
float extent_x = min(extent_tight * sqrt(max(0.0, cov2D.x)), radius_f);
float extent_y = min(extent_tight * sqrt(max(0.0, cov2D.z)), radius_f);
ivec2 rmn = rectMin(pix, extent_x, extent_y, grid_x, grid_y, TILE_W, TILE_H);
ivec2 rmx = rectMax(pix, extent_x, extent_y, grid_x, grid_y, TILE_W, TILE_H);
```
Also update the radius_f_arr storage:
```glsl
radius_f_arr[i*2u + 0u] = extent_x;  // was radius_f
radius_f_arr[i*2u + 1u] = extent_y;  // was radius_f
```

Verify that `rectMin/rectMax` accept separate x/y extents. If they only accept one radius, add a variant or inline the computation:
```glsl
ivec2 rmn = ivec2(
    min(grid_x, max(0, int(floor((pix.x - extent_x) / float(TILE_W))))),
    min(grid_y, max(0, int(floor((pix.y - extent_y) / float(TILE_H)))))
);
ivec2 rmx = ivec2(
    min(grid_x, max(0, int(ceil((pix.x + extent_x) / float(TILE_W))))),
    min(grid_y, max(0, int(ceil((pix.y + extent_y) / float(TILE_H)))))
);
```

- [ ] **Step 5: Build**

```bash
cmake --build harmonyos_3dgs/build -j$(nproc) --target gs3d_vk_tests
```

- [ ] **Step 6: Run PSNR harness — record new PSNR**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure 2>&1 | grep "PSNR="
```

Expected: PSNR increases from Task 9 baseline.

- [ ] **Step 7: Run full suite, confirm no regressions**

```bash
ctest --test-dir harmonyos_3dgs/build --output-on-failure 2>&1 | tail -10
```

- [ ] **Step 8: Update kBaselinePSNR if PSNR improved**

- [ ] **Step 9: Commit**

```bash
git add harmonyos_3dgs/src/vulkan/shaders/preprocess.comp harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp
git commit -m "fix(preprocess): rect_bounding + tight_opacity_bounding — asymmetric AABB, adaptive extent (PSNR delta: +X.X dB)"
```

---

## Task 11: Implement `tile_based_culling` in scatter + preprocess

**Why:** aaa.json sets `tile_based_culling=true` and `load_balancing=true`. CUDA's `duplicateWithKeys_extended` runs a per-tile opacity test and only writes tiles where the Gaussian contributes meaningfully. VK writes all rect tiles unconditionally. This inflates the sort buffer and feeds the rasterizer with extra tiles that CUDA would skip. PSNR impact: HIGH (different tile-pair counts change blend ordering for every tile boundary region).

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/shaders/preprocess.comp`
- Modify: `harmonyos_3dgs/src/vulkan/shaders/scatter.comp`

**CUDA reference:** `forward.cu::duplicateWithKeys_extended` (stopthepop_common.cuh, search for `write_tile`), `auxiliary.h::max_contrib_power_rect_gaussian_float`.

- [ ] **Step 1: Read the research gate**

```bash
cat .claude/gates/research.md
```

- [ ] **Step 2: Read CUDA reference — per-tile opacity test**

```bash
grep -n "write_tile\|max_contrib_power\|opacity_power_threshold\|TILE_BASED_CULLING\|computeTilebasedCulling" \
  /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/stopthepop_common.cuh | head -40
```

```bash
grep -n "write_tile\|max_contrib_power\|opacity_power_threshold\|TILE_BASED_CULLING" \
  /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | head -30
```

- [ ] **Step 3: Read scatter.comp current tile-write loop**

```bash
sed -n '270,330p' harmonyos_3dgs/src/vulkan/shaders/scatter.comp
```

- [ ] **Step 4: Implement per-tile opacity predicate in scatter.comp**

For each tile in the rect bounding box, compute `max_contrib_power_rect_gaussian` — the maximum 2D Gaussian contribution at any point within the tile's pixel bounding box:

For the 2D path: The maximum contribution within a tile rect [tile_min, tile_max] for a 2D Gaussian with conic (a, b, c) and mean (mx, my) is found by clamping the mean to the tile rect and evaluating. The CUDA function `max_contrib_power_rect_gaussian_float` in stopthepop_common.cuh computes this.

Port the CUDA per-tile test to GLSL in scatter.comp:
```glsl
// tile_based_culling: skip tile if max Gaussian contribution is below threshold
float tile_min_x = float(tx * TILE_W);
float tile_max_x = float((tx + 1) * TILE_W - 1);
float tile_min_y = float(ty * TILE_H);
float tile_max_y = float((ty + 1) * TILE_H - 1);
float clamped_x = clamp(mean2D.x, tile_min_x, tile_max_x);
float clamped_y = clamp(mean2D.y, tile_min_y, tile_max_y);
float dx = clamped_x - mean2D.x;
float dy = clamped_y - mean2D.y;
float power = -0.5 * (conic.x * dx*dx + conic.z * dy*dy) - conic.y * dx*dy;
float max_opac_factor = exp(power);
const float ALPHA_THRESHOLD = 1.0 / 255.0;
bool write_tile = (opacity * max_opac_factor >= ALPHA_THRESHOLD);
if (!write_tile) {
    // Write INVALID sentinel key so radix sort pushes this to the end
    keys_out[out_slot] = (uint64_t(0xFFFFFFFFu) << 32) | uint64_t(0xFFFFFFFFu);
    vals_out[out_slot] = 0xFFFFFFFFu;
} else {
    keys_out[out_slot] = (uint64_t(tile_id) << 32) | uint64_t(depth_bits);
    vals_out[out_slot] = i;
}
```

- [ ] **Step 5: Update preprocess.comp tiles_touched count**

The `tiles_touched[i]` count must match the number of tiles that will actually be written (for correct prefix sum). Add the same per-tile test in preprocess.comp's 2D path tile-counting loop.

- [ ] **Step 6: Build**

```bash
cmake --build harmonyos_3dgs/build -j$(nproc) --target gs3d_vk_tests
```

- [ ] **Step 7: Run PSNR harness — record new PSNR**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure 2>&1 | grep "PSNR="
```

- [ ] **Step 8: Update kBaselinePSNR + commit**

```bash
git add harmonyos_3dgs/src/vulkan/shaders/preprocess.comp harmonyos_3dgs/src/vulkan/shaders/scatter.comp harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp
git commit -m "fix(scatter): tile_based_culling — per-tile opacity predicate, INVALID sentinel for culled tiles (PSNR delta: +X.X dB)"
```

---

## Task 12: Implement hierarchical eval_3D rasterizer

**Why:** aaa.json sets `sort_mode=3` (HIERARCHICAL). CUDA uses `sortGaussiansRayHierarchicalCUDA_forward` (hierarchical_render.cuh, 1378 lines) with per-pixel HEAD_WINDOW=4 sorted k-buffer, 2×2 MID_WINDOW=8 intermediate queue, and 4×4 TAIL buffer (64 entries). VK uses a simple global front-to-back blend (sort_mode=0 equivalent). This is the primary driver of the PSNR gap. PSNR impact: CATASTROPHIC (expected +20–30 dB).

**Key parameters from aaa.json:**
- `per_pixel = 4` (HEAD_WINDOW)
- `tile_2x2 = 8` (MID_WINDOW)
- `tile_4x4 = 64` (TAIL buffer size)
- `hierarchical_4x4_culling = true` (CULL_ALPHA)

**CUDA reference:** `hierarchical_render.cuh` (1378 lines, particularly lines 207–1059 for the kernel implementation)

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp`
- Possibly create: new sub-shaders or GLSL include if needed

- [ ] **Step 1: Read the research gate**

```bash
cat .claude/gates/research.md
```

- [ ] **Step 2: Read CUDA hierarchical render kernel**

```bash
sed -n '1,100p' /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh
sed -n '200,500p' /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh
sed -n '500,900p' /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh
sed -n '900,1200p' /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh
sed -n '1200,1378p' /home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh
```

- [ ] **Step 3: Read current rasterize.comp eval_3D path**

```bash
sed -n '180,260p' harmonyos_3dgs/src/vulkan/shaders/rasterize.comp
```

- [ ] **Step 4: Understand CUDA thread layout**

CUDA hierarchical kernel uses `{16, 4, 4}` thread blocks — 16 pixels × 4×4 sub-tile hierarchy.
- `dim3 block(BLOCK_X, 4, 4)` = `{16, 4, 4}` = 256 threads per block
- `threadIdx.x` = pixel within the 16-pixel row; `threadIdx.y` = 2×2 sub-tile x; `threadIdx.z` = 2×2 sub-tile y
- Each 16×16 tile contains 4×4 sub-tiles (each 4×4 pixels), each holding a MID queue

VK: Currently `local_size_x=TILE_W*TILE_H` (256 threads for 16×16 tile). The thread ID maps to `(thread_id % TILE_W, thread_id / TILE_W)`.

The CUDA hierarchy: each 4×4 tile block shares a TAIL queue (64 entries, sorted by tile-center depth). Within each 4×4, the 2×2 groups share MID queues. Within each 2×2, individual pixels have HEAD queues (4 entries, sorted by per-pixel depth).

- [ ] **Step 5: Design VK shared memory layout**

For a 16×16 tile (256 threads), the hierarchical layout requires:
- 1 TAIL buffer: 64 entries per 4×4 tile (1 per 16×16 tile since TAIL covers the whole tile)
- 16 MID buffers: 8 entries each, one per 2×2 sub-tile (16 sub-tiles per 16×16 tile)  
- 256 HEAD buffers: 4 entries each, one per pixel

Shared memory:
```glsl
// 256 pixels × 4-entry head = 1024 depth/contrib entries
shared float head_depth[256][4];
shared uint head_gauss_idx[256][4];
shared int head_size[256];
// 16 2x2-subtiles × 8-entry mid
shared float mid_depth[16][8];
shared uint mid_gauss_idx[16][8];
shared int mid_size[16];
// 1 tail per 16×16 tile × 64 entries
shared float tail_depth[64];
shared uint tail_gauss_idx[64];
shared int tail_size_s;
```

- [ ] **Step 6: Port the hierarchical insertion logic**

The CUDA kernel (hierarchical_render.cuh lines 400–900) processes Gaussians in batches from the sorted list:
1. Load batch into shared memory
2. For each Gaussian in batch, each 4×4 group evaluates `hierarchical_4x4_culling`:
   - Compute alpha at the nearest point of the 4×4 tile rect
   - If alpha < ALPHA_THRESHOLD for whole 4×4 tile: skip this Gaussian for this tile
3. Surviving Gaussians are inserted into TAIL (sorted by 4×4-center depth)
4. When TAIL is full, flush: each 2×2 sub-tile pulls its subset from TAIL into MID (sorted by 2×2-center depth)
5. When MID is full, flush: each pixel pulls from MID into HEAD (sorted by per-pixel depth via max_contrib_ray)
6. When HEAD is full, blend the shallowest entry and pop it

**Implementation note:** This is approximately 400–600 lines of GLSL. Port the CUDA logic faithfully, adapting:
- CUDA `cooperative_groups::tiled_partition<N>` → GLSL `subgroupBallot` / barrier-based voting
- CUDA `__ballot_sync` → `subgroupBallot`
- CUDA `__syncthreads()` → `barrier()` + `memoryBarrierShared()`
- CUDA `atomicOr` / `atomicAdd` on shared → GLSL `atomicAdd` on shared uint arrays

- [ ] **Step 7: Handle flush and final blend**

After all Gaussians processed: flush remaining TAIL→MID→HEAD entries for each pixel, then blend the HEAD entries in depth order into the final color accumulator.

This matches CUDA hierarchical_render.cuh lines 800–1000 (the final-flush phase).

- [ ] **Step 8: Build**

```bash
cmake --build harmonyos_3dgs/build -j$(nproc) --target gs3d_vk_tests
```

Fix any compilation errors before proceeding.

- [ ] **Step 9: Run PSNR harness — record new PSNR**

```bash
ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure 2>&1 | grep "PSNR="
```

Expected: substantial PSNR gain (+20–30 dB). If gain is < 5 dB, stop and file an investigation note before proceeding.

- [ ] **Step 10: Run FullChain_TinyFixture_Eval3D — no regression**

```bash
ctest --test-dir harmonyos_3dgs/build -R FullChain_TinyFixture_Eval3D --output-on-failure
```

Must still pass (≥83 dB on tiny fixture).

- [ ] **Step 11: Update kBaselinePSNR + commit**

```bash
git add harmonyos_3dgs/src/vulkan/shaders/rasterize.comp harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp
git commit -m "feat(rasterize): hierarchical eval_3D sort_mode=3 — HEAD_WINDOW=4 per-pixel k-buffer, 4x4 culling (PSNR delta: +X.X dB)"
```

---

## Task 13: Verify ≥60 dB and lock final baseline

- [ ] **Step 1: Run harness 3× to confirm PSNR ≥ 60 dB**

```bash
for i in 1 2 3; do
  ctest --test-dir harmonyos_3dgs/build -R VkVsCudaBasketball --output-on-failure 2>&1 | grep "PSNR="
done
```

If PSNR < 60 dB, consult the emergency exit clause in the design spec and investigate residual divergence.

- [ ] **Step 2: Run full test suite**

```bash
ctest --test-dir harmonyos_3dgs/build --output-on-failure 2>&1 | tail -20
```

- [ ] **Step 3: Set kBaselinePSNR = floor(PSNR*10)/10 - 0.5**

- [ ] **Step 4: Commit**

```bash
git add harmonyos_3dgs/tests/test_vk_vs_cuda_basketball.cpp
git commit -m "test(harness): lock G3 baseline PSNR (>=60 dB target achieved)"
```
