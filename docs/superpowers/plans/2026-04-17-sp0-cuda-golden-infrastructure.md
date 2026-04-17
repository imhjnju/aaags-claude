# SP-0 CUDA Golden Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the CUDA golden-artifact dumping and C++ consumption infrastructure needed by all downstream Vulkan sub-projects (SP-1 through SP-6).

**Architecture:** Two-layer design. Python layer (`tools/dump_tool.py`) orchestrates step/camera/seed and drives the AAA-Gaussians CUDA extension. CUDA extension layer gains a new `materialize_dump()` function that parses the already-returned `geomBuffer/binningBuffer/imgBuffer` into named tensors (zero-invasive to kernel main chain), plus a `dump_mode` flag on backward to additionally expose `dL_dconic`. C++ test side gets a header-only `npy_reader.h` + comparison + manifest utilities.

**Tech Stack:** Python 3 (aaa-gs conda env), PyTorch, CUDA 11+, PyBind11, C++17, CMake 3.28+, GoogleTest.

---

## Interface Inventory

**Lock this inventory in `dev_notes/interfaces_sp0.md` during Task 1 to prevent drift.** Any implementation that calls a "draft" interface before it is created MUST fail first.

### Existing interfaces (real, already in the codebase / dependencies)

| Interface | Location | Usage |
|-----------|----------|-------|
| `AAA-Gaussians/train.py -m / --model_path` | `AAA-Gaussians/arguments/__init__.py:52-59` | Output directory for baseline runs |
| `AAA-Gaussians/train.py -s` | same | Source dataset path |
| `_C.rasterize_gaussians(...)` → `(int, color, radii, geomBuffer, binningBuffer, imgBuffer)` | `rasterize_points.cu:44-142` | Forward pass; buffers returned as raw bytes |
| `_C.rasterize_gaussians_backward(...)` → 8-tuple | `rasterize_points.cu:144-236` | Backward; d_means2D/colors/opacity/means3D/cov3D/sh/scales/rotations |
| `GaussianRasterizationSettings.debug: bool` | `diff_gaussian_rasterization/__init__.py:254-269` | Snapshot-on-exception flag (DO NOT reuse) |
| `GaussianModel.capture()` / `.restore()` | `scene/gaussian_model.py` | Full model state serialization |
| `torch.Tensor.retain_grad()` | PyTorch built-in | Required to read gradient of non-leaf tensors |
| `CudaRasterizer::GeometryState::fromChunk(chunk, P, requires_cov3D_inv, requires_gauss2screen)` | `rasterizer_impl.cu:175` | Parse geomBuffer byte layout |
| `CudaRasterizer::BinningState::fromChunk(chunk, P)` | `rasterizer_impl.cu:206` | Parse binningBuffer |
| `CudaRasterizer::ImageState::fromChunk(chunk, N)` | `rasterizer_impl.cu:197` | Parse imgBuffer |
| `cub::DeviceRadixSort::SortPairs` | CUDA toolkit | Used for (key, value) sort inside rasterizer |
| conda env `aaa-gs` | `/home/robota/miniconda3/envs/aaa-gs` | Runtime for Python dumper and baseline |
| GoogleTest fetched v1.14.0 | `harmonyos_3dgs/CMakeLists.txt` | C++ unit test framework |

### Draft interfaces (to be created in this plan)

| Interface | Planned location | Created in Task |
|-----------|------------------|-----------------|
| `materialize_dump(geomBuffer, binningBuffer, imgBuffer, P, R, num_tiles, H, W, requires_cov3D_inv, requires_gauss2screen)` → `dict[str, torch.Tensor]` | `rasterize_points.{h,cu}` | Task 10, 11 |
| `_C.materialize_dump(...)` Python binding | `ext.cpp` | Task 12 |
| New optional param `dump_mode: bool` on `RasterizeGaussiansBackwardCUDA` | `rasterize_points.cu:170` | Task 13 |
| `tools/dump_tool.py` CLI (`--fixture`, `--output`, `--ladder`, `--dump_steps`) | `tools/dump_tool.py` | Task 17-22 |
| `tests/golden/npy_reader.h` (header-only, 5 dtypes) | `harmonyos_3dgs/tests/golden/` | Task 4-6 |
| `tests/golden/compare.h` (dual-threshold compare) | same | Task 7 |
| `tests/golden/manifest.h` (manifest.json reader) | same | Task 8 |
| `manifest.json` schema (step/cam/artifacts) | described in Task 16 | Task 16 |
| `dev_notes/dependencies.md` | `dev_notes/dependencies.md` | Task 26 |
| `dev_notes/interfaces_sp0.md` | `dev_notes/interfaces_sp0.md` | Task 1 |

**Enforcement rule:** If any task references an interface not in either list above, plan is broken — fix the plan, do not invent interfaces on the fly.

---

## File Structure

### New files
```
dev_notes/
├── dependencies.md                  # fused_ssim version, CUB determinism verdict
└── interfaces_sp0.md                # inventory above, living doc

tools/
├── dump_tool.py                     # main CLI
├── dump_npy.py                      # NpyDumper (atomic writes + manifest)
└── dump_fixtures.py                 # tiny & basketball fixture loaders

harmonyos_3dgs/tests/golden/
├── npy_reader.h                     # header-only npy v1.0 reader
├── compare.h                        # compare_f32/u32/u64 with dual tolerance
└── manifest.h                       # manifest.json reader (nlohmann::json)

harmonyos_3dgs/tests/
├── test_npy_reader.cpp              # unit tests for npy_reader
├── test_compare.cpp                 # unit tests for compare
├── test_manifest.cpp                # unit tests for manifest reader
└── test_golden_roundtrip.cpp        # end-to-end: python dump → C++ read
```

### Modified files
```
AAA-Gaussians/submodules/diff-gaussian-rasterization/
├── rasterize_points.h               # declare materialize_dump, add dump_mode
├── rasterize_points.cu              # implement materialize_dump, backward dump_mode
├── ext.cpp                          # bind materialize_dump
└── diff_gaussian_rasterization/__init__.py  # (NOT modified — dumper calls _C directly)

harmonyos_3dgs/CMakeLists.txt        # add tests/golden/ include path, link new tests
```

### Golden artifact output (gitignored for basketball; committed for tiny)
```
harmonyos_3dgs/tests/golden/tiny/    # small, committed
  step000001/cam0000/*.npy + manifest.json

dev_notes/ground_truth/              # large, gitignored
  sp5/step{000001..002000}/cam*/*.npy
  sp6/step010000/cam*/*.npy
```

---

## Task 1: Lock interface inventory + scaffolding

**Files:**
- Create: `dev_notes/interfaces_sp0.md`
- Create: `harmonyos_3dgs/tests/golden/.gitkeep`
- Create: `tools/.gitkeep`
- Create: `dev_notes/ground_truth/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory skeletons + Python package init**

```bash
mkdir -p /home/robota/h00813233/Graph/aaags-claude/tools
mkdir -p /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs/tests/golden
mkdir -p /home/robota/h00813233/Graph/aaags-claude/dev_notes/ground_truth
touch /home/robota/h00813233/Graph/aaags-claude/tools/__init__.py
touch /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs/tests/golden/.gitkeep
touch /home/robota/h00813233/Graph/aaags-claude/dev_notes/ground_truth/.gitkeep
```

`tools/__init__.py` makes `tools/` a Python package so `from tools.dump_tiny import run_tiny` works.
All `dump_tool.py` invocations use `python -m tools.dump_tool ...` (run from repo root), **not** `python tools/dump_tool.py`, to keep the project root in `sys.path` so `from tools.dump_tiny import run_tiny` resolves.

- [ ] **Step 2: Write interface inventory**

Copy the "Interface Inventory" section of this plan into `dev_notes/interfaces_sp0.md`. Add header:
```markdown
# SP-0 Interface Inventory

> Living document. Update when interfaces are added or their signature changes.
> Source of truth for what exists vs what is draft.
```

- [ ] **Step 3: Update .gitignore**

Append to `.gitignore`:
```
# SP-0 golden artifacts — large, not committed
dev_notes/ground_truth/**/*.npy
dev_notes/ground_truth/**/*.pth
dev_notes/ground_truth/**/*.ply
# keep .gitkeep
!dev_notes/ground_truth/.gitkeep
!dev_notes/ground_truth/*/.gitkeep
# tiny fixtures ARE committed (see tests/golden/tiny/)
```

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add dev_notes/interfaces_sp0.md dev_notes/ground_truth/.gitkeep \
        harmonyos_3dgs/tests/golden/.gitkeep tools/__init__.py .gitignore
git commit -m "sp0: scaffold directories and lock interface inventory"
```

---

## Task 2: Verify CUB DeviceRadixSort determinism

**Files:**
- Create: `tools/verify_cub_determinism.py`

- [ ] **Step 1: Write determinism probe**

```python
# tools/verify_cub_determinism.py
"""Verify CUB DeviceRadixSort gives bit-identical output across runs.

Runs rasterize_gaussians on a fixed-seed tiny scene 10 times and compares
sort_keys_sorted / sort_values_sorted byte-for-byte. If all 10 match,
CUB is deterministic on this hardware; if not, sort golden verification
must downgrade from exact to 'tile-grouping consistent' (see spec §1.2).
"""
import sys, os, hashlib
import numpy as np
import torch

# This probe uses a synthetic tiny scene to avoid dataset dependency.
# It does NOT require materialize_dump — it manually parses geomBuffer
# bytes using the known GeometryState layout.
# TODO: update after Task 11 ships to use materialize_dump instead.

def main():
    torch.manual_seed(42)
    # Synthetic 100-gaussian scene, random positions + random view
    # This is intentionally before materialize_dump — we use checksums
    # of the raw binningBuffer bytes (which contain point_list + keys) as
    # the determinism signal.
    N = 100
    from diff_gaussian_rasterization import GaussianRasterizationSettings, \
        GaussianRasterizer, ExtendedSettings, SortSettings, SortMode, \
        GlobalSortOrder, CullingSettings

    device = torch.device("cuda")
    means3D = torch.randn(N, 3, device=device) * 2
    scales  = torch.randn(N, 3, device=device) * 0.1
    rots    = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=1)
    opac    = torch.rand(N, 1, device=device)
    sh      = torch.randn(N, 16, 3, device=device) * 0.1
    filt    = torch.ones(N, device=device) * 0.01

    view = torch.eye(4, device=device)
    proj = torch.eye(4, device=device)
    invvp = torch.eye(4, device=device)

    settings = ExtendedSettings(
        sort_settings=SortSettings(sort_mode=SortMode.GLOBAL,
                                    sort_order=GlobalSortOrder.Z_DEPTH),
        culling_settings=CullingSettings(),
        load_balancing=False, proper_ewa_scaling=False, eval_3D=False)

    rs = GaussianRasterizationSettings(
        image_height=256, image_width=256,
        tanfovx=1.0, tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=proj, inv_viewprojmatrix=invvp,
        sh_degree=3,
        campos=torch.zeros(3, device=device),
        prefiltered=False, settings=settings, render_depth=False, debug=False)

    ra = GaussianRasterizer(raster_settings=rs)

    hashes = []
    for run in range(10):
        color, radii = ra(means3D=means3D, means2D=torch.zeros_like(means3D),
                          opacities=opac, filter3D=filt, shs=sh,
                          scales=scales, rotations=rots)
        torch.cuda.synchronize()
        h = hashlib.sha256(color.detach().cpu().numpy().tobytes()).hexdigest()
        hashes.append(h)
        print(f"run {run}: image sha256 = {h}")

    if len(set(hashes)) == 1:
        print("DETERMINISTIC: all 10 runs produced identical rendered image.")
        sys.exit(0)
    else:
        print(f"NON-DETERMINISTIC: {len(set(hashes))} distinct hashes in 10 runs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python tools/verify_cub_determinism.py
```

Expected: `DETERMINISTIC: all 10 runs produced identical rendered image.` (exit 0)
If non-deterministic: document the finding in Task 26.

- [ ] **Step 3: Commit probe + result**

```bash
git add tools/verify_cub_determinism.py
git commit -m "sp0: add CUB radix sort determinism probe"
```

---

## Task 3: Start `tests/golden/npy_reader.h` — stub + failing test

**Files:**
- Create: `harmonyos_3dgs/tests/golden/npy_reader.h`
- Create: `harmonyos_3dgs/tests/test_npy_reader.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Write the failing test first**

Create `harmonyos_3dgs/tests/test_npy_reader.cpp`:
```cpp
#include <gtest/gtest.h>
#include "golden/npy_reader.h"

TEST(NpyReader, LoadFloat32_2D) {
    // Pre-generated fixture: float32 array [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    // Saved via numpy.save("tests/golden/fixtures/f32_2x3.npy", arr)
    auto arr = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    EXPECT_EQ(arr.dtype, NpyDtype::float32);
    EXPECT_EQ(arr.shape.size(), 2u);
    EXPECT_EQ(arr.shape[0], 2u);
    EXPECT_EQ(arr.shape[1], 3u);
    EXPECT_EQ(arr.numel(), 6u);
    float* data = arr.f32();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[5], 6.0f);
}
```

- [ ] **Step 2: Create minimal stub header**

Create `harmonyos_3dgs/tests/golden/npy_reader.h`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class NpyDtype { float32, int32, int64, uint32, uint64 };

struct NpyArray {
    std::vector<uint8_t> raw;
    std::vector<size_t>  shape;
    NpyDtype             dtype = NpyDtype::float32;
    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    float*    f32() { return reinterpret_cast<float*>(raw.data()); }
    int32_t*  i32() { return reinterpret_cast<int32_t*>(raw.data()); }
    uint32_t* u32() { return reinterpret_cast<uint32_t*>(raw.data()); }
    int64_t*  i64() { return reinterpret_cast<int64_t*>(raw.data()); }
    uint64_t* u64() { return reinterpret_cast<uint64_t*>(raw.data()); }
};

// NOT YET IMPLEMENTED — test should fail at link time
NpyArray load_npy(const std::string& path);
```

- [ ] **Step 3: Register test in CMakeLists**

Modify `harmonyos_3dgs/CMakeLists.txt` — find the `gs3d_tests` section (around line 130-170) and add:
```cmake
target_sources(gs3d_tests PRIVATE tests/test_npy_reader.cpp)
target_include_directories(gs3d_tests PRIVATE tests)
```

- [ ] **Step 4: Create fixture file**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
mkdir -p harmonyos_3dgs/tests/test_data/golden/fixtures
conda run -n aaa-gs python -c "import numpy as np; \
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32); \
np.save('harmonyos_3dgs/tests/test_data/golden/fixtures/f32_2x3.npy', arr)"
ls harmonyos_3dgs/tests/test_data/golden/fixtures/f32_2x3.npy
```

Expected: file exists, size ≈ 152 bytes (128B header + 24B data).

- [ ] **Step 5: Build and run test — expect link failure**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build 2>&1 | tail -20
```

Expected: link error on `load_npy` (undefined reference). This confirms the test is wired up and the stub correctly has no implementation.

- [ ] **Step 6: Commit the failing state**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/golden/npy_reader.h \
        harmonyos_3dgs/tests/test_npy_reader.cpp \
        harmonyos_3dgs/tests/test_data/golden/fixtures/f32_2x3.npy \
        harmonyos_3dgs/CMakeLists.txt
git commit -m "sp0: add npy_reader header stub + failing test"
```

---

## Task 4: Implement npy_reader.h — parse NPY v1.0 header

**Files:**
- Modify: `harmonyos_3dgs/tests/golden/npy_reader.h`

NPY v1.0 spec: `\x93NUMPY<major><minor><HEADER_LEN:u16 LE><HEADER_TEXT>\n<raw data>`
Header text is a Python dict literal: `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`

- [ ] **Step 1: Add implementation into the same header (inline)**

Append to `harmonyos_3dgs/tests/golden/npy_reader.h`:
```cpp
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstring>

inline NpyDtype parse_descr(const std::string& descr) {
    // descr format: "<f4", "<i4", "<i8", "<u4", "<u8" (little-endian assumed)
    if (descr == "<f4") return NpyDtype::float32;
    if (descr == "<i4") return NpyDtype::int32;
    if (descr == "<i8") return NpyDtype::int64;
    if (descr == "<u4") return NpyDtype::uint32;
    if (descr == "<u8") return NpyDtype::uint64;
    throw std::runtime_error("npy_reader: unsupported dtype: " + descr);
}

inline size_t dtype_size(NpyDtype d) {
    switch (d) {
        case NpyDtype::float32: case NpyDtype::int32: case NpyDtype::uint32: return 4;
        case NpyDtype::int64:   case NpyDtype::uint64:                       return 8;
    }
    throw std::runtime_error("npy_reader: unknown dtype size");
}

inline std::string extract_dict_value(const std::string& header, const std::string& key) {
    // Extract value for "'key': VALUE," from Python dict literal header.
    auto pos = header.find("'" + key + "':");
    if (pos == std::string::npos) throw std::runtime_error("npy_reader: key not found: " + key);
    pos += key.size() + 3;  // past "'key':"
    while (pos < header.size() && header[pos] == ' ') ++pos;
    // Collect until next ',' at top level (shape uses parens, descr/fortran_order do not)
    int paren = 0;
    std::string val;
    while (pos < header.size()) {
        char c = header[pos];
        if (c == '(') ++paren;
        else if (c == ')') --paren;
        else if (c == ',' && paren == 0) break;
        val += c; ++pos;
    }
    return val;
}

inline std::vector<size_t> parse_shape(const std::string& shape_str) {
    // shape_str like "(2, 3)" or "(5,)"
    std::vector<size_t> shape;
    size_t i = shape_str.find('(') + 1;
    std::string num;
    while (i < shape_str.size() && shape_str[i] != ')') {
        char c = shape_str[i];
        if (c >= '0' && c <= '9') num += c;
        else if (c == ',' || c == ' ') {
            if (!num.empty()) { shape.push_back(std::stoul(num)); num.clear(); }
        }
        ++i;
    }
    if (!num.empty()) shape.push_back(std::stoul(num));
    return shape;
}

inline NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_reader: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_reader: bad magic at " + path);
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (major != 1) throw std::runtime_error("npy_reader: only v1.0 supported, got v"
                                             + std::to_string(major));
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    NpyArray arr;
    std::string descr = extract_dict_value(header, "descr");
    // strip surrounding quotes
    descr.erase(std::remove(descr.begin(), descr.end(), '\''), descr.end());
    // trim
    while (!descr.empty() && descr.front() == ' ') descr.erase(0, 1);
    while (!descr.empty() && descr.back() == ' ')  descr.pop_back();
    arr.dtype = parse_descr(descr);

    std::string shape_val = extract_dict_value(header, "shape");
    arr.shape = parse_shape(shape_val);

    std::string fo = extract_dict_value(header, "fortran_order");
    if (fo.find("True") != std::string::npos)
        throw std::runtime_error("npy_reader: fortran_order=True not supported");

    size_t total_bytes = arr.numel() * dtype_size(arr.dtype);
    arr.raw.resize(total_bytes);
    f.read(reinterpret_cast<char*>(arr.raw.data()), total_bytes);
    if (f.gcount() != static_cast<std::streamsize>(total_bytes))
        throw std::runtime_error("npy_reader: short read at " + path);
    return arr;
}

inline void assert_shape(const NpyArray& a, std::vector<size_t> expected) {
    if (a.shape != expected) {
        std::ostringstream oss;
        oss << "npy_reader: shape mismatch. got [";
        for (auto s : a.shape) oss << s << ",";
        oss << "], expected [";
        for (auto s : expected) oss << s << ",";
        oss << "]";
        throw std::runtime_error(oss.str());
    }
}

inline void assert_dtype(const NpyArray& a, NpyDtype expected) {
    if (a.dtype != expected)
        throw std::runtime_error("npy_reader: dtype mismatch");
}
```

Also add `#include <algorithm>` at the top (for `std::remove`).

- [ ] **Step 2: Run test — should now pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build && ctest --test-dir build -R NpyReader --output-on-failure
```

Expected: `NpyReader.LoadFloat32_2D` passes.

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/golden/npy_reader.h
git commit -m "sp0: implement npy_reader v1.0 parser (float32)"
```

---

## Task 5: Extend npy_reader tests to all 5 dtypes + edge cases

**Files:**
- Modify: `harmonyos_3dgs/tests/test_npy_reader.cpp`
- Create: fixtures for int32/int64/uint32/uint64

- [ ] **Step 1: Add failing tests for all dtypes**

Append to `tests/test_npy_reader.cpp`:
```cpp
TEST(NpyReader, LoadInt32_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/i32_5.npy");
    EXPECT_EQ(a.dtype, NpyDtype::int32);
    EXPECT_EQ(a.shape, std::vector<size_t>{5});
    EXPECT_EQ(a.i32()[0], -2);
    EXPECT_EQ(a.i32()[4], 2);
}

TEST(NpyReader, LoadUint32_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/u32_4.npy");
    EXPECT_EQ(a.dtype, NpyDtype::uint32);
    EXPECT_EQ(a.u32()[0], 0u);
    EXPECT_EQ(a.u32()[3], 4294967295u);  // UINT32_MAX
}

TEST(NpyReader, LoadUint64_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/u64_3.npy");
    EXPECT_EQ(a.dtype, NpyDtype::uint64);
    EXPECT_EQ(a.u64()[0], 0ULL);
    EXPECT_EQ(a.u64()[2], 18446744073709551615ULL);  // UINT64_MAX
}

TEST(NpyReader, LoadInt64_1D) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/i64_2.npy");
    EXPECT_EQ(a.dtype, NpyDtype::int64);
    EXPECT_EQ(a.i64()[0], -9223372036854775807LL);
    EXPECT_EQ(a.i64()[1],  9223372036854775807LL);
}

TEST(NpyReader, AssertShape_Pass) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    assert_shape(a, {2u, 3u});  // should not throw
}

TEST(NpyReader, AssertShape_FailThrows) {
    auto a = load_npy(std::string(TEST_DATA_DIR) + "/golden/fixtures/f32_2x3.npy");
    EXPECT_THROW(assert_shape(a, {3u, 2u}), std::runtime_error);
}

TEST(NpyReader, MissingFileThrows) {
    EXPECT_THROW(load_npy("/nonexistent/path.npy"), std::runtime_error);
}
```

- [ ] **Step 2: Generate fixtures**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -c "
import numpy as np
base = 'harmonyos_3dgs/tests/test_data/golden/fixtures'
np.save(f'{base}/i32_5.npy', np.array([-2, -1, 0, 1, 2], dtype=np.int32))
np.save(f'{base}/u32_4.npy', np.array([0, 1, 2, 2**32-1], dtype=np.uint32))
np.save(f'{base}/u64_3.npy', np.array([0, 1, 2**64-1], dtype=np.uint64))
np.save(f'{base}/i64_2.npy', np.array([-(2**63-1), 2**63-1], dtype=np.int64))
print('fixtures ok')
"
ls harmonyos_3dgs/tests/test_data/golden/fixtures/
```

Expected: 5 .npy files total.

- [ ] **Step 3: Build and run — expect all pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build && ctest --test-dir build -R NpyReader --output-on-failure
```

Expected: 7 NpyReader tests all pass.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/test_npy_reader.cpp \
        harmonyos_3dgs/tests/test_data/golden/fixtures/
git commit -m "sp0: cover all 5 dtypes + edge cases in npy_reader tests"
```

---

## Task 6: `compare.h` — dual-threshold float comparison

**Files:**
- Create: `harmonyos_3dgs/tests/golden/compare.h`
- Create: `harmonyos_3dgs/tests/test_compare.cpp`

- [ ] **Step 1: Write the failing test**

Create `harmonyos_3dgs/tests/test_compare.cpp`:
```cpp
#include <gtest/gtest.h>
#include <vector>
#include "golden/compare.h"

TEST(Compare, F32_ExactMatch_Passes) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};
    auto r = compare_f32(a, b, /*abs=*/1e-6f, /*rel=*/1e-4f);
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.max_abs_err, 0.0f);
}

TEST(Compare, F32_SmallDiff_WithinAbsTolerance_Passes) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.000001f, 2.0f, 3.0f};
    auto r = compare_f32(a, b, /*abs=*/1e-5f, /*rel=*/1e-4f);
    EXPECT_TRUE(r.passed);
}

TEST(Compare, F32_LargeDiff_Fails) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.1f, 2.0f, 3.0f};  // 10% off
    auto r = compare_f32(a, b, /*abs=*/1e-6f, /*rel=*/1e-4f);
    EXPECT_FALSE(r.passed);
    EXPECT_EQ(r.first_bad_index, 0u);
}

TEST(Compare, F32_NearZero_UsesAbsolute) {
    // near-zero values: relative check unstable, absolute check used
    std::vector<float> a{1e-10f, 1e-10f};
    std::vector<float> b{2e-10f, 1e-10f};  // rel err 100% but abs err 1e-10
    auto r = compare_f32(a, b, /*abs=*/1e-5f, /*rel=*/1e-4f);
    EXPECT_TRUE(r.passed);  // passes because abs err < abs_tol
}

TEST(Compare, U32_ExactOnly) {
    std::vector<uint32_t> a{0, 1, 2, 3};
    std::vector<uint32_t> b{0, 1, 2, 3};
    EXPECT_TRUE(compare_u32(a, b));
    std::vector<uint32_t> c{0, 1, 99, 3};
    EXPECT_FALSE(compare_u32(a, c));
}

TEST(Compare, U64_ExactOnly) {
    std::vector<uint64_t> a{0ULL, 1ULL, (1ULL << 40)};
    std::vector<uint64_t> b{0ULL, 1ULL, (1ULL << 40)};
    EXPECT_TRUE(compare_u64(a, b));
}

TEST(Compare, MismatchSize_Fails) {
    std::vector<float> a{1.0f, 2.0f};
    std::vector<float> b{1.0f};
    auto r = compare_f32(a, b, 1e-6f, 1e-4f);
    EXPECT_FALSE(r.passed);
}
```

- [ ] **Step 2: Register test**

Modify `harmonyos_3dgs/CMakeLists.txt`, add `tests/test_compare.cpp` to `target_sources(gs3d_tests PRIVATE ...)`.

- [ ] **Step 3: Build — expect link failure**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build 2>&1 | tail -10
```

Expected: link error on `compare_f32` / `compare_u32` / `compare_u64`.

- [ ] **Step 4: Implement compare.h**

Create `harmonyos_3dgs/tests/golden/compare.h`:
```cpp
#pragma once
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <vector>
#include <limits>

struct CompareResult {
    bool    passed        = false;
    float   max_abs_err   = 0.0f;
    float   max_rel_err   = 0.0f;
    size_t  first_bad_index = std::numeric_limits<size_t>::max();
    size_t  num_bad       = 0;
};

// Float32 comparison with dual threshold:
//   pass iff (abs_err <= abs_tol) OR (rel_err <= rel_tol)
// This prevents near-zero values (where rel_err is unstable) from failing
// on noise while still catching true drift.
inline CompareResult compare_f32(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 float abs_tol, float rel_tol) {
    CompareResult r;
    if (a.size() != b.size()) return r;
    for (size_t i = 0; i < a.size(); ++i) {
        float abs_err = std::fabs(a[i] - b[i]);
        float rel_err = abs_err / (std::fabs(b[i]) + 1e-30f);
        bool bad = (abs_err > abs_tol) && (rel_err > rel_tol);
        if (bad) {
            if (r.num_bad == 0) r.first_bad_index = i;
            ++r.num_bad;
        }
        if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;
        if (rel_err > r.max_rel_err) r.max_rel_err = rel_err;
    }
    r.passed = (r.num_bad == 0);
    return r;
}

inline bool compare_u32(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

inline bool compare_u64(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}
```

- [ ] **Step 5: Build and run — expect all pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build && ctest --test-dir build -R Compare --output-on-failure
```

Expected: 7 Compare tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/golden/compare.h \
        harmonyos_3dgs/tests/test_compare.cpp \
        harmonyos_3dgs/CMakeLists.txt
git commit -m "sp0: add compare.h dual-threshold float compare"
```

---

## Task 7: `manifest.h` — read manifest.json

**Files:**
- Create: `harmonyos_3dgs/tests/golden/manifest.h`
- Create: `harmonyos_3dgs/tests/test_manifest.cpp`
- Create: `harmonyos_3dgs/tests/test_data/golden/fixtures/manifest_example.json`

nlohmann::json is already vendored at `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/json/json.hpp`. Check if it's available in harmonyos_3dgs; if not, fetch via CMake.

- [ ] **Step 1: Verify/add nlohmann::json dependency**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
grep -n "nlohmann" CMakeLists.txt || echo "not yet added"
```

If not present, append to `CMakeLists.txt` (near the FetchContent block):
```cmake
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(gs3d_tests PRIVATE nlohmann_json::nlohmann_json)
```

- [ ] **Step 2: Write fixture manifest**

Create `harmonyos_3dgs/tests/test_data/golden/fixtures/manifest_example.json`:
```json
{
  "step": 1,
  "camera_idx": 0,
  "seed": 42,
  "reference_commit": "abc1234",
  "config_hash": "sha256:deadbeef",
  "artifacts": [
    {
      "filename": "preprocess_means2D.npy",
      "operator": "preprocess",
      "tensor": "means2D",
      "shape": [100, 2],
      "dtype": "float32",
      "layout": "row_major"
    },
    {
      "filename": "sort_keys_sorted.npy",
      "operator": "sort",
      "tensor": "keys_sorted",
      "shape": [500],
      "dtype": "uint64",
      "layout": "flat"
    }
  ]
}
```

- [ ] **Step 3: Write failing test**

Create `harmonyos_3dgs/tests/test_manifest.cpp`:
```cpp
#include <gtest/gtest.h>
#include "golden/manifest.h"

TEST(Manifest, Parse) {
    auto m = load_manifest(std::string(TEST_DATA_DIR) + "/golden/fixtures/manifest_example.json");
    EXPECT_EQ(m.step, 1);
    EXPECT_EQ(m.camera_idx, 0);
    EXPECT_EQ(m.seed, 42);
    EXPECT_EQ(m.artifacts.size(), 2u);
    EXPECT_EQ(m.artifacts[0].filename, "preprocess_means2D.npy");
    EXPECT_EQ(m.artifacts[0].op, "preprocess");
    EXPECT_EQ(m.artifacts[0].tensor, "means2D");
    EXPECT_EQ(m.artifacts[0].shape.size(), 2u);
    EXPECT_EQ(m.artifacts[0].shape[0], 100u);
    EXPECT_EQ(m.artifacts[0].dtype, "float32");
}

TEST(Manifest, FindArtifact) {
    auto m = load_manifest(std::string(TEST_DATA_DIR) + "/golden/fixtures/manifest_example.json");
    auto a = find_artifact(m, "preprocess", "means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->filename, "preprocess_means2D.npy");
    auto b = find_artifact(m, "nope", "nope");
    EXPECT_EQ(b, nullptr);
}
```

- [ ] **Step 4: Create manifest.h stub**

Create `harmonyos_3dgs/tests/golden/manifest.h`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

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

Manifest load_manifest(const std::string& path);
const Artifact* find_artifact(const Manifest& m, const std::string& op, const std::string& tensor);
```

- [ ] **Step 5: Register test + build — expect link fail**

Modify CMakeLists.txt: add `tests/test_manifest.cpp` to `gs3d_tests`. Build:
```bash
cmake --build build 2>&1 | tail -10
```
Expected: link error on `load_manifest` and `find_artifact`.

- [ ] **Step 6: Implement manifest.h**

Append to `manifest.h`:
```cpp
#include <nlohmann/json.hpp>
#include <fstream>

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
```

- [ ] **Step 7: Build and run — expect all pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build && ctest --test-dir build -R Manifest --output-on-failure
```

Expected: 2 Manifest tests pass.

- [ ] **Step 8: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/golden/manifest.h \
        harmonyos_3dgs/tests/test_manifest.cpp \
        harmonyos_3dgs/tests/test_data/golden/fixtures/manifest_example.json \
        harmonyos_3dgs/CMakeLists.txt
git commit -m "sp0: add manifest.h reader + tests"
```

---

## Task 8: Declare `materialize_dump` in `rasterize_points.h`

**Files:**
- Modify: `AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.h`

- [ ] **Step 1: Read current header**

```bash
cat /home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.h
```

Note the existing function signatures.

- [ ] **Step 2: Add materialize_dump declaration**

Append to `rasterize_points.h` before the final `#endif` (or at end of top-level):
```cpp
// SP-0: Parse raw geomBuffer/binningBuffer/imgBuffer into named tensors.
// Non-invasive: does NOT modify forward/backward main chain. Called AFTER
// RasterizeGaussiansCUDA returns. Returns a dict keyed by tensor name.
//
// Args mirror the rasterizer internal state layout (see rasterizer_impl.h):
//   geom_buffer, binning_buffer, img_buffer: raw byte tensors from forward
//   P: num_gaussians
//   R: num_rendered (sum of tiles_touched), from forward return int
//   num_tiles: (tile_grid_x * tile_grid_y)
//   H, W: image size
//   requires_cov3D_inv: matches the forward-time flag (StopThePop kbuffer mode)
//   requires_gauss2screen: matches forward-time eval_3D flag
std::map<std::string, torch::Tensor>
MaterializeDumpCUDA(
    const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& img_buffer,
    int P, int R, int num_tiles, int H, int W,
    bool requires_cov3D_inv,
    bool requires_gauss2screen);
```

Also add `#include <map>` and `#include <string>` at the top.

- [ ] **Step 3: Build — expect linker failure on new symbol when binding is added**

The declaration alone doesn't break build yet (no bindings reference it). Skip build, proceed to implementation.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.h
git commit -m "sp0: declare MaterializeDumpCUDA (no impl yet)"
```

---

## Task 9: Implement `MaterializeDumpCUDA` in `rasterize_points.cu`

**Files:**
- Modify: `AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.cu`

The function parses raw buffers using `GeometryState::fromChunk`, `BinningState::fromChunk`, `ImageState::fromChunk`, then `cudaMemcpy` each internal array into a freshly-allocated `torch::Tensor`.

- [ ] **Step 1: Add implementation at end of `rasterize_points.cu`**

```cpp
// SP-0: materialize dump — parse raw buffers into named tensors
#include "cuda_rasterizer/rasterizer_impl.h"

std::map<std::string, torch::Tensor>
MaterializeDumpCUDA(
    const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& img_buffer,
    int P, int R, int num_tiles, int H, int W,
    bool requires_cov3D_inv,
    bool requires_gauss2screen)
{
    torch::Device device(torch::kCUDA);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto u32 = torch::TensorOptions().dtype(torch::kUInt32).device(device);
    auto u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);

    std::map<std::string, torch::Tensor> out;
    if (P == 0) return out;

    // --- GeometryState ---
    char* geom_ptr = reinterpret_cast<char*>(
        const_cast<void*>(geom_buffer.data_ptr()));
    auto geom = CudaRasterizer::GeometryState::fromChunk(
        geom_ptr, P, requires_cov3D_inv, requires_gauss2screen);

    // means2D: float2* → [P, 2] float32
    out["preprocess_means2D"] = torch::empty({P, 2}, f32);
    cudaMemcpy(out["preprocess_means2D"].data_ptr(),
               geom.means2D, P * 2 * sizeof(float), cudaMemcpyDeviceToDevice);

    // depths: float* → [P]
    out["preprocess_depths"] = torch::empty({P}, f32);
    cudaMemcpy(out["preprocess_depths"].data_ptr(),
               geom.depths, P * sizeof(float), cudaMemcpyDeviceToDevice);

    // conic_opacity: float4* → [P, 4] float32, packed {a, b, c, opacity}
    out["preprocess_conic_opacity"] = torch::empty({P, 4}, f32);
    cudaMemcpy(out["preprocess_conic_opacity"].data_ptr(),
               geom.conic_opacity, P * 4 * sizeof(float), cudaMemcpyDeviceToDevice);

    // rgb: float* → [P, 3]
    out["preprocess_rgb"] = torch::empty({P, 3}, f32);
    cudaMemcpy(out["preprocess_rgb"].data_ptr(),
               geom.rgb, P * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    // tiles_touched: uint32_t* → [P]
    out["preprocess_tiles_touched"] = torch::empty({P}, u32);
    cudaMemcpy(out["preprocess_tiles_touched"].data_ptr(),
               geom.tiles_touched, P * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // point_offsets: uint32_t* → [P]
    out["preprocess_point_offsets"] = torch::empty({P}, u32);
    cudaMemcpy(out["preprocess_point_offsets"].data_ptr(),
               geom.point_offsets, P * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    if (requires_gauss2screen && geom.gauss2screen != nullptr) {
        out["preprocess_gauss2screen"] = torch::empty({P, 16}, f32);
        cudaMemcpy(out["preprocess_gauss2screen"].data_ptr(),
                   geom.gauss2screen, P * 16 * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // --- BinningState (only if R > 0) ---
    if (R > 0) {
        char* bin_ptr = reinterpret_cast<char*>(
            const_cast<void*>(binning_buffer.data_ptr()));
        auto bin = CudaRasterizer::BinningState::fromChunk(bin_ptr, R);

        out["sort_keys_unsorted"] = torch::empty({R}, u64);
        cudaMemcpy(out["sort_keys_unsorted"].data_ptr(),
                   bin.point_list_keys_unsorted, R * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        out["sort_keys_sorted"] = torch::empty({R}, u64);
        cudaMemcpy(out["sort_keys_sorted"].data_ptr(),
                   bin.point_list_keys, R * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        out["sort_values_unsorted"] = torch::empty({R}, u32);
        cudaMemcpy(out["sort_values_unsorted"].data_ptr(),
                   bin.point_list_unsorted, R * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        out["sort_values_sorted"] = torch::empty({R}, u32);
        cudaMemcpy(out["sort_values_sorted"].data_ptr(),
                   bin.point_list, R * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    }

    // --- ImageState ---
    char* img_ptr = reinterpret_cast<char*>(
        const_cast<void*>(img_buffer.data_ptr()));
    auto img = CudaRasterizer::ImageState::fromChunk(img_ptr, H * W);

    out["sort_tile_ranges"] = torch::empty({num_tiles, 2}, u32);
    cudaMemcpy(out["sort_tile_ranges"].data_ptr(),
               img.ranges, num_tiles * 2 * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    out["rasterize_n_contrib"] = torch::empty({H * W}, u32);
    cudaMemcpy(out["rasterize_n_contrib"].data_ptr(),
               img.n_contrib, H * W * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    out["rasterize_transmittance"] = torch::empty({H * W}, f32);
    cudaMemcpy(out["rasterize_transmittance"].data_ptr(),
               img.accum_alpha, H * W * sizeof(float), cudaMemcpyDeviceToDevice);

    return out;
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.cu
git commit -m "sp0: implement MaterializeDumpCUDA parsing geom/binning/img buffers"
```

---

## Task 10: Add `dump_mode` to backward, return `dL_dconic`

**Files:**
- Modify: `AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.h`
- Modify: `AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.cu`

- [ ] **Step 1: Add dump_mode parameter to backward declaration**

In `rasterize_points.h`, find the declaration of `RasterizeGaussiansBackwardCUDA` and add `bool dump_mode = false` as the last parameter. Change the return type: when `dump_mode=true`, the tuple gets one extra tensor (`dL_dconic [P, 2, 2]`). Because C++ std::tuple return types are static, define a **new** function rather than changing the existing one:

```cpp
// Extended backward with optional dL_dconic return.
// Returns 9-tuple (vs 8-tuple from RasterizeGaussiansBackwardCUDA) when
// dump_mode=true. The first 8 elements match the original backward return
// order exactly; dL_dconic is appended as element 8.
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
           torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
RasterizeGaussiansBackwardDumpCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &opacities,
    const torch::Tensor &colors,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &inv_viewprojmatrix,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &pixel_colors,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const nlohmann::json& settings_dict,
    const bool debug);
```

- [ ] **Step 2: Implement the new backward in rasterize_points.cu**

Add at end of `rasterize_points.cu` (copy the existing `RasterizeGaussiansBackwardCUDA` body and modify the return):

```cpp
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
           torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
RasterizeGaussiansBackwardDumpCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &opacities,
    const torch::Tensor &colors,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &inv_viewprojmatrix,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &pixel_colors,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const nlohmann::json& settings_dict,
    const bool debug)
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);
    int M = 0; if (sh.size(0) != 0) M = sh.size(1);

    torch::Tensor dL_dmeans3D      = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D      = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors       = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic        = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity      = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D        = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dgauss2screen = torch::zeros({P, 4, 4}, means3D.options());
    torch::Tensor dL_dsh           = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales       = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations    = torch::zeros({P, 4}, means3D.options());

    CudaRasterizer::SplattingSettings settings =
        settings_dict.get<CudaRasterizer::SplattingSettings>();

    if (P != 0) {
        CudaRasterizer::Rasterizer::backward(
            P, degree, M, R,
            background.contiguous().data<float>(), W, H, settings,
            means3D.contiguous().data<float>(), sh.contiguous().data<float>(),
            opacities.contiguous().data<float>(), colors.contiguous().data<float>(),
            scales.data_ptr<float>(), scale_modifier,
            rotations.data_ptr<float>(), cov3D_precomp.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            inv_viewprojmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(), tan_fovx, tan_fovy,
            pixel_colors.contiguous().data<float>(),
            radii.contiguous().data<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data<float>(),
            dL_dmeans2D.contiguous().data<float>(),
            dL_dconic.contiguous().data<float>(),
            dL_dopacity.contiguous().data<float>(),
            dL_dcolors.contiguous().data<float>(),
            dL_dmeans3D.contiguous().data<float>(),
            dL_dcov3D.contiguous().data<float>(),
            dL_dgauss2screen.contiguous().data<float>(),
            dL_dsh.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_drotations.contiguous().data<float>(),
            debug);
    }

    // Return order: first 8 match RasterizeGaussiansBackwardCUDA exactly,
    // 9th (dL_dconic) is the dump_mode extra.
    return std::make_tuple(
        dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D,
        dL_dsh, dL_dscales, dL_drotations,
        dL_dconic);
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.{h,cu}
git commit -m "sp0: add RasterizeGaussiansBackwardDumpCUDA returning dL_dconic"
```

---

## Task 11: Bind new functions in `ext.cpp`

**Files:**
- Modify: `AAA-Gaussians/submodules/diff-gaussian-rasterization/ext.cpp`

- [ ] **Step 1: Add bindings**

Replace `ext.cpp` with:
```cpp
#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_gaussians_backward_dump", &RasterizeGaussiansBackwardDumpCUDA,
        "SP-0: backward returning extra dL_dconic as 9th tuple element");
  m.def("materialize_dump", &MaterializeDumpCUDA,
        "SP-0: parse raw geom/binning/img buffers into named tensors");
  m.def("mark_visible", &markVisible);
  m.def("compute_relocation", &ComputeRelocationCUDA);
}
```

- [ ] **Step 2: Rebuild extension**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians/submodules/diff-gaussian-rasterization
conda run -n aaa-gs pip install -e . 2>&1 | tail -20
```

Expected: `Successfully installed diff-gaussian-rasterization-...` (or similar).

- [ ] **Step 3: Verify bindings exist**

```bash
conda run -n aaa-gs python -c "
from diff_gaussian_rasterization import _C
print('materialize_dump:', hasattr(_C, 'materialize_dump'))
print('backward_dump:', hasattr(_C, 'rasterize_gaussians_backward_dump'))
"
```

Expected:
```
materialize_dump: True
backward_dump: True
```

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add AAA-Gaussians/submodules/diff-gaussian-rasterization/ext.cpp
git commit -m "sp0: bind materialize_dump and backward_dump to Python"
```

---

## Task 12: Smoke test — call `materialize_dump` from Python

**Files:**
- Create: `tools/smoke_materialize_dump.py`

- [ ] **Step 1: Write smoke test**

```python
# tools/smoke_materialize_dump.py
"""Verify materialize_dump returns all expected tensors with correct shapes."""
import torch
from diff_gaussian_rasterization import _C, GaussianRasterizer, \
    GaussianRasterizationSettings, ExtendedSettings, SortSettings, SortMode, \
    GlobalSortOrder, CullingSettings

torch.manual_seed(0)
device = torch.device("cuda")
N = 50

means3D = torch.randn(N, 3, device=device) * 2
scales  = torch.rand(N, 3, device=device) * 0.2
rots    = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=1)
opac    = torch.rand(N, 1, device=device) * 0.5 + 0.3
sh      = torch.randn(N, 16, 3, device=device) * 0.1
filt    = torch.ones(N, device=device) * 0.01

view = torch.eye(4, device=device); view[2, 3] = -5.0  # back up camera
proj = torch.eye(4, device=device); proj[3, 3] = 0; proj[3, 2] = 1; proj[2, 2] = 1
invvp = torch.eye(4, device=device)

settings = ExtendedSettings(
    sort_settings=SortSettings(sort_mode=SortMode.GLOBAL,
                                sort_order=GlobalSortOrder.Z_DEPTH),
    culling_settings=CullingSettings(),
    load_balancing=False, proper_ewa_scaling=False, eval_3D=False)

rs = GaussianRasterizationSettings(
    image_height=64, image_width=64,
    tanfovx=1.0, tanfovy=1.0,
    bg=torch.zeros(3, device=device),
    scale_modifier=1.0,
    viewmatrix=view, projmatrix=proj, inv_viewprojmatrix=invvp,
    sh_degree=3, campos=torch.zeros(3, device=device),
    prefiltered=False, settings=settings, render_depth=False, debug=False)

# Call _C directly to get the raw buffers
args = (rs.bg, means3D, torch.Tensor([]), opac, scales, rots, filt,
        rs.scale_modifier, torch.Tensor([]),
        rs.viewmatrix, rs.projmatrix, rs.inv_viewprojmatrix,
        rs.tanfovx, rs.tanfovy, rs.image_height, rs.image_width,
        sh, rs.sh_degree, rs.campos, rs.prefiltered,
        rs.settings.to_dict(), rs.render_depth, rs.debug)

num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
    _C.rasterize_gaussians(*args)
print(f"num_rendered R = {num_rendered}")

# Compute num_tiles
import math
TILE = 16
num_tiles_x = (rs.image_width + TILE - 1) // TILE
num_tiles_y = (rs.image_height + TILE - 1) // TILE
num_tiles = num_tiles_x * num_tiles_y
print(f"num_tiles = {num_tiles}")

out = _C.materialize_dump(geomBuffer, binningBuffer, imgBuffer,
    N, num_rendered, num_tiles, rs.image_height, rs.image_width,
    False, False)  # requires_cov3D_inv=False, requires_gauss2screen=False

for k, v in sorted(out.items()):
    print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}")

# Assert expected keys are present
required = ["preprocess_means2D", "preprocess_depths", "preprocess_conic_opacity",
            "preprocess_rgb", "preprocess_tiles_touched", "preprocess_point_offsets",
            "sort_tile_ranges", "rasterize_n_contrib", "rasterize_transmittance"]
if num_rendered > 0:
    required += ["sort_keys_unsorted", "sort_keys_sorted",
                 "sort_values_unsorted", "sort_values_sorted"]
missing = [k for k in required if k not in out]
assert not missing, f"missing tensors: {missing}"
print("OK: all required tensors present")
```

- [ ] **Step 2: Run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python tools/smoke_materialize_dump.py
```

Expected output: `OK: all required tensors present` + printed shapes matching spec §2.3.

- [ ] **Step 3: Commit**

```bash
git add tools/smoke_materialize_dump.py
git commit -m "sp0: smoke test materialize_dump from Python"
```

---

## Task 13: `dump_npy.py` — atomic writes + manifest accumulator

**Files:**
- Create: `tools/dump_npy.py`

- [ ] **Step 1: Write NpyDumper**

```python
# tools/dump_npy.py
"""Atomic NPY writer + manifest accumulator.

NpyDumper writes each tensor to {output_dir}/step{step:06d}/cam{cam:04d}/{op}_{tensor}.npy
and accumulates an entry in manifest.json. Call .finalize() at end to flush manifest.

Atomic = write to tmp file, rename. Avoids partial reads if crashed mid-dump.
"""
import json
import os
import numpy as np
import torch
from pathlib import Path


class NpyDumper:
    def __init__(self, output_dir: str, step: int, cam_idx: int,
                 seed: int, reference_commit: str = "", config_hash: str = ""):
        self.dir = Path(output_dir) / f"step{step:06d}" / f"cam{cam_idx:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "step": int(step),
            "camera_idx": int(cam_idx),
            "seed": int(seed),
            "reference_commit": reference_commit,
            "config_hash": config_hash,
            "artifacts": [],
        }

    def _torch_to_numpy(self, t: torch.Tensor):
        return t.detach().cpu().numpy()

    def _dtype_name(self, a: np.ndarray) -> str:
        m = {np.dtype("float32"): "float32", np.dtype("int32"): "int32",
             np.dtype("int64"): "int64", np.dtype("uint32"): "uint32",
             np.dtype("uint64"): "uint64"}
        if a.dtype not in m:
            raise TypeError(f"NpyDumper: unsupported dtype {a.dtype}; cast upstream.")
        return m[a.dtype]

    def dump(self, op: str, tensor: str, arr, layout: str = "row_major"):
        if isinstance(arr, torch.Tensor):
            arr = self._torch_to_numpy(arr)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"NpyDumper: need ndarray or Tensor, got {type(arr)}")
        # Enforce contiguous row-major
        arr = np.ascontiguousarray(arr)
        fname = f"{op}_{tensor}.npy"
        tmp = self.dir / (fname + ".tmp")
        final = self.dir / fname
        np.save(tmp, arr)
        os.replace(tmp, final)
        self.manifest["artifacts"].append({
            "filename": fname,
            "operator": op,
            "tensor": tensor,
            "shape": list(arr.shape),
            "dtype": self._dtype_name(arr),
            "layout": layout,
        })

    def finalize(self):
        tmp = self.dir / "manifest.json.tmp"
        final = self.dir / "manifest.json"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, final)
```

- [ ] **Step 2: Quick CLI smoke**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -c "
from tools.dump_npy import NpyDumper
import numpy as np, tempfile, os, json
with tempfile.TemporaryDirectory() as d:
    dp = NpyDumper(d, step=1, cam_idx=0, seed=42)
    dp.dump('preprocess', 'means2D', np.zeros((10, 2), dtype=np.float32))
    dp.dump('sort', 'keys_sorted', np.zeros(20, dtype=np.uint64))
    dp.finalize()
    mfp = os.path.join(d, 'step000001', 'cam0000', 'manifest.json')
    m = json.load(open(mfp))
    assert m['step'] == 1
    assert len(m['artifacts']) == 2
    assert m['artifacts'][0]['dtype'] == 'float32'
    assert m['artifacts'][1]['dtype'] == 'uint64'
    print('NpyDumper OK')
"
```

Expected: `NpyDumper OK`.

- [ ] **Step 3: Commit**

```bash
git add tools/dump_npy.py
git commit -m "sp0: add NpyDumper atomic writer with manifest accumulator"
```

---

## Task 14: `dump_fixtures.py` — tiny fixture generator

**Files:**
- Create: `tools/dump_fixtures.py`

- [ ] **Step 1: Write tiny fixture generator**

```python
# tools/dump_fixtures.py
"""Tiny synthetic fixtures for algorithmic unit tests.

A tiny fixture is a deterministic small scene (5-50 Gaussians, synthetic camera)
that runs quickly and can be committed to git. Produced artifacts are used by
C++ unit tests that do NOT depend on basketball dataset.
"""
import torch
import numpy as np


def build_tiny_scene(N: int = 20, seed: int = 42, H: int = 64, W: int = 64):
    """Build a minimal deterministic scene.

    Returns dict with keys: means3D, scales, rotations, opacities, sh, filter_3D,
    viewmatrix, projmatrix, inv_viewprojmatrix, campos, tan_fovx, tan_fovy,
    H, W, sh_degree, sh_coeffs_per_g.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = torch.device("cuda")

    means3D = (torch.randn((N, 3), generator=g, device=device) * 1.5)
    scales  = (torch.rand ((N, 3), generator=g, device=device) * 0.2 + 0.05)
    rots    = torch.nn.functional.normalize(
                torch.randn((N, 4), generator=g, device=device), dim=1)
    opac    = torch.rand((N, 1), generator=g, device=device) * 0.5 + 0.3
    sh_deg  = 3
    M       = (sh_deg + 1) ** 2
    sh      = torch.randn((N, M, 3), generator=g, device=device) * 0.1
    filt    = torch.ones(N, device=device) * 0.01

    # Simple pinhole camera: look-at origin from (0, 0, -5), fov ~90deg
    view = torch.eye(4, device=device)
    view[2, 3] = -5.0
    tan_fov = 1.0
    proj = torch.eye(4, device=device)
    # simple perspective: zeroed w,z already; we approximate
    proj[0, 0] = 1.0 / tan_fov
    proj[1, 1] = 1.0 / tan_fov
    proj[2, 2] = 1.0
    proj[3, 2] = 1.0
    proj[3, 3] = 0.0
    # view-proj combined
    viewproj = proj @ view
    inv_viewproj = torch.linalg.inv(viewproj)
    campos = torch.tensor([0.0, 0.0, 5.0], device=device)

    return dict(
        means3D=means3D, scales=scales, rotations=rots, opacities=opac,
        sh=sh, filter_3D=filt,
        viewmatrix=view, projmatrix=viewproj,
        inv_viewprojmatrix=inv_viewproj,
        campos=campos, tan_fovx=tan_fov, tan_fovy=tan_fov,
        H=H, W=W, sh_degree=sh_deg, sh_coeffs_per_g=M,
    )
```

- [ ] **Step 2: Smoke**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -c "
from tools.dump_fixtures import build_tiny_scene
s = build_tiny_scene(N=20, seed=42)
assert s['means3D'].shape == (20, 3)
assert s['sh'].shape == (20, 16, 3)
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add tools/dump_fixtures.py
git commit -m "sp0: add tiny fixture generator"
```

---

## Task 15: `dump_tool.py` skeleton — CLI + fixture dispatch

**Files:**
- Create: `tools/dump_tool.py`

- [ ] **Step 1: Write CLI skeleton**

```python
#!/usr/bin/env python3
# tools/dump_tool.py
"""SP-0 CUDA golden dumper.

Orchestrates step/camera/seed and calls AAA-Gaussians CUDA extension to
produce .npy golden artifacts at specified ladder steps.

Usage:
  dump_tool.py --fixture=tiny --output=harmonyos_3dgs/tests/golden/tiny
  dump_tool.py --fixture=basketball --output=dev_notes/ground_truth \\
               --ladder=1,10,100,500,1000,2000
  dump_tool.py --fixture=basketball --output=dev_notes/ground_truth \\
               --dump_steps=1234,1235  # on-demand for bisection
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", choices=["tiny", "basketball"], required=True)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--ladder", default="",
                   help="Comma-separated step numbers to dump (e.g. 1,10,100)")
    p.add_argument("--dump_steps", default="",
                   help="Additional on-demand step numbers (e.g. 1234,1235)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source_path", default="/home/robota/Downloads/basketball")
    p.add_argument("--ply_path",
                   default="/home/robota/Downloads/basketball/sparse/0/points3D.ply")
    p.add_argument("--iterations", type=int, default=2000,
                   help="Total training steps when fixture=basketball")
    p.add_argument("--densify_from_iter", type=int, default=5000)
    p.add_argument("--densify_until_iter", type=int, default=10000)
    p.add_argument("--densification_interval", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    steps = set()
    if args.ladder:
        steps.update(int(s) for s in args.ladder.split(",") if s)
    if args.dump_steps:
        steps.update(int(s) for s in args.dump_steps.split(",") if s)

    if args.fixture == "tiny":
        from tools.dump_tiny import run_tiny
        run_tiny(output_dir=str(out), seed=args.seed)
    elif args.fixture == "basketball":
        if not steps:
            print("--ladder or --dump_steps required for basketball", file=sys.stderr)
            sys.exit(2)
        from tools.dump_basketball import run_basketball
        run_basketball(output_dir=str(out), steps=sorted(steps),
                       source_path=args.source_path, ply_path=args.ply_path,
                       iterations=args.iterations, seed=args.seed,
                       densify_from_iter=args.densify_from_iter,
                       densify_until_iter=args.densify_until_iter,
                       densification_interval=args.densification_interval)
    else:
        raise ValueError(args.fixture)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI smoke — help**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -m tools.dump_tool --help
```

Expected: help text listing all flags.

- [ ] **Step 3: Commit**

```bash
git add tools/dump_tool.py
git commit -m "sp0: dump_tool.py CLI skeleton"
```

---

## Task 16: `dump_tiny.py` — tiny fixture dump runner

**Files:**
- Create: `tools/dump_tiny.py`

- [ ] **Step 1: Write tiny runner**

```python
# tools/dump_tiny.py
"""Run a tiny synthetic fixture and dump one forward+backward checkpoint.

Step=1, cam=0. Produces all preprocess/sort/rasterize/backward artifacts
in the required schema, committed alongside C++ tests.
"""
import torch
import math
from tools.dump_npy import NpyDumper
from tools.dump_fixtures import build_tiny_scene


def run_tiny(output_dir: str, seed: int = 42):
    from diff_gaussian_rasterization import _C, \
        GaussianRasterizationSettings, GaussianRasterizer, \
        ExtendedSettings, SortSettings, SortMode, GlobalSortOrder, CullingSettings

    torch.manual_seed(seed)
    s = build_tiny_scene(seed=seed)

    settings = ExtendedSettings(
        sort_settings=SortSettings(sort_mode=SortMode.GLOBAL,
                                    sort_order=GlobalSortOrder.Z_DEPTH),
        culling_settings=CullingSettings(),
        load_balancing=False, proper_ewa_scaling=False, eval_3D=False)

    rs = GaussianRasterizationSettings(
        image_height=s["H"], image_width=s["W"],
        tanfovx=s["tan_fovx"], tanfovy=s["tan_fovy"],
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=s["viewmatrix"],
        projmatrix=s["projmatrix"],
        inv_viewprojmatrix=s["inv_viewprojmatrix"],
        sh_degree=s["sh_degree"],
        campos=s["campos"],
        prefiltered=False, settings=settings, render_depth=False, debug=False)

    # Forward: use _C directly to capture buffers
    args = (rs.bg, s["means3D"], torch.Tensor([]).cuda(), s["opacities"],
            s["scales"], s["rotations"], s["filter_3D"],
            rs.scale_modifier, torch.Tensor([]).cuda(),
            rs.viewmatrix, rs.projmatrix, rs.inv_viewprojmatrix,
            rs.tanfovx, rs.tanfovy, rs.image_height, rs.image_width,
            s["sh"], rs.sh_degree, rs.campos,
            rs.prefiltered, rs.settings.to_dict(), rs.render_depth, rs.debug)

    R, color, radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(*args)

    TILE = 16
    nt_x = (s["W"] + TILE - 1) // TILE
    nt_y = (s["H"] + TILE - 1) // TILE
    num_tiles = nt_x * nt_y

    out = _C.materialize_dump(geomBuf, binBuf, imgBuf,
        s["means3D"].shape[0], R, num_tiles, s["H"], s["W"],
        False, False)

    # Keep color and radii for rasterize output
    out["rasterize_image"] = color       # [3, H, W]
    out["preprocess_radii"] = radii       # [P] int32

    # --- Backward ---
    # We need an upstream gradient. Use fixed random mask for determinism.
    gmask = torch.randn_like(color)

    # retain_grad + backward through Python autograd wrapper:
    color_ag = color.detach().clone().requires_grad_(True)
    # Cannot rewire backprop through detached tensor directly — we use _C.backward_dump
    # which accepts dL_dout_color as explicit input.
    dL_dout_color = gmask.contiguous()

    bw_args = (rs.bg, s["means3D"], radii, s["opacities"],
               torch.Tensor([]).cuda(), s["scales"], s["rotations"],
               rs.scale_modifier, torch.Tensor([]).cuda(),
               rs.viewmatrix, rs.projmatrix, rs.inv_viewprojmatrix,
               rs.tanfovx, rs.tanfovy, color, dL_dout_color,
               s["sh"], rs.sh_degree, rs.campos,
               geomBuf, R, binBuf, imgBuf,
               rs.settings.to_dict(), rs.debug)

    (d_means2D, d_colors, d_opacity, d_means3D, d_cov3D,
     d_sh, d_scales, d_rotations, d_conic) = \
        _C.rasterize_gaussians_backward_dump(*bw_args)

    dumper = NpyDumper(output_dir, step=1, cam_idx=0, seed=seed)
    # Forward
    for key, ten in out.items():
        dumper.dump(*_split_op_tensor(key), ten)
    # dL_dout_color input
    dumper.dump("backward", "dL_dout_color", dL_dout_color)
    # Backward outputs
    dumper.dump("backward", "d_means2D",   d_means2D)
    dumper.dump("backward", "d_colors",    d_colors)
    dumper.dump("backward", "d_opacity",   d_opacity)
    dumper.dump("backward", "d_means3D",   d_means3D)
    dumper.dump("backward", "d_cov3D",     d_cov3D)
    dumper.dump("backward", "d_sh",        d_sh)
    dumper.dump("backward", "d_scales",    d_scales)
    dumper.dump("backward", "d_rotations", d_rotations)
    dumper.dump("backward", "d_conic",     d_conic)
    dumper.finalize()
    print(f"tiny fixture dumped to {output_dir}/step000001/cam0000/")


def _split_op_tensor(key: str):
    # "preprocess_means2D" → ("preprocess", "means2D")
    idx = key.index("_")
    return key[:idx], key[idx+1:]
```

- [ ] **Step 2: Run tiny fixture dump**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -m tools.dump_tool --fixture=tiny \
    --output=harmonyos_3dgs/tests/golden/tiny
```

Expected: `tiny fixture dumped to .../step000001/cam0000/`.

- [ ] **Step 3: Verify produced files**

```bash
ls harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/
cat harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/manifest.json | head -20
```

Expected: 15+ .npy files + manifest.json.

- [ ] **Step 4: Commit tiny fixture**

```bash
git add tools/dump_tiny.py harmonyos_3dgs/tests/golden/tiny/
git commit -m "sp0: generate and commit tiny fixture golden artifacts"
```

---

## Task 17: C++ round-trip test — read the tiny fixture we just dumped

**Files:**
- Create: `harmonyos_3dgs/tests/test_golden_roundtrip.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Write test**

```cpp
// tests/test_golden_roundtrip.cpp
#include <gtest/gtest.h>
#include <string>
#include "golden/npy_reader.h"
#include "golden/manifest.h"

namespace {
std::string tiny_root() {
    return std::string(TEST_DATA_DIR) + "/../golden/tiny/step000001/cam0000";
}
}

TEST(GoldenRoundtrip, ManifestLoads) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    EXPECT_EQ(m.step, 1);
    EXPECT_EQ(m.camera_idx, 0);
    EXPECT_GE(m.artifacts.size(), 15u);
}

TEST(GoldenRoundtrip, PreprocessMeans2D) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "preprocess", "means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->dtype, "float32");
    EXPECT_EQ(a->shape.size(), 2u);
    EXPECT_EQ(a->shape[1], 2u);   // [N, 2]

    auto arr = load_npy(tiny_root() + "/" + a->filename);
    EXPECT_EQ(arr.dtype, NpyDtype::float32);
    EXPECT_EQ(arr.shape[0], a->shape[0]);
    EXPECT_EQ(arr.shape[1], 2u);
}

TEST(GoldenRoundtrip, SortKeysSorted_U64) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "sort", "keys_sorted");
    if (a == nullptr) {
        GTEST_SKIP() << "no rendered pairs in tiny fixture (R==0)";
    }
    auto arr = load_npy(tiny_root() + "/" + a->filename);
    EXPECT_EQ(arr.dtype, NpyDtype::uint64);
    // keys must be non-decreasing (sorted) — sanity check
    uint64_t* k = arr.u64();
    for (size_t i = 1; i < arr.numel(); ++i)
        EXPECT_LE(k[i-1], k[i]) << "unsorted at i=" << i;
}

TEST(GoldenRoundtrip, BackwardDMeans2D_Shape3) {
    auto m = load_manifest(tiny_root() + "/manifest.json");
    auto a = find_artifact(m, "backward", "d_means2D");
    ASSERT_TRUE(a != nullptr);
    EXPECT_EQ(a->shape.size(), 2u);
    EXPECT_EQ(a->shape[1], 3u) << "d_means2D must be [P, 3] (CUDA-side allocation)";
}
```

- [ ] **Step 2: Register test**

Modify `harmonyos_3dgs/CMakeLists.txt`: add `tests/test_golden_roundtrip.cpp` to `gs3d_tests`.

- [ ] **Step 3: Build and run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
cmake --build build && ctest --test-dir build -R GoldenRoundtrip --output-on-failure
```

Expected: 4 tests, all pass.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add harmonyos_3dgs/tests/test_golden_roundtrip.cpp harmonyos_3dgs/CMakeLists.txt
git commit -m "sp0: C++ round-trip test against tiny fixture golden"
```

---

## Task 18: `dump_basketball.py` — basketball ladder dump

**Files:**
- Create: `tools/dump_basketball.py`

This runs the AAA-Gaussians `train.py` loop up to each ladder step and dumps artifacts. We shell out to `train.py` once but inject a patch that stops and dumps at each ladder step. The cleanest approach: import `train.py`'s machinery as a module (it is not module-safe), OR: re-implement the required bits of the train loop in `dump_basketball.py`.

For SP-0, the simpler approach: subclass/wrap the scene loader + renderer from `AAA-Gaussians.scene` and `gaussian_renderer`, running the training loop ourselves. This keeps the dump tool independent of `train.py`'s CLI.

- [ ] **Step 1: Write basketball runner (draft — focus on ladder semantics)**

```python
# tools/dump_basketball.py
"""Basketball ladder dump: run training for N steps, dump at ladder steps.

Re-implements enough of AAA-Gaussians/train.py to drive the pipeline,
captures the stochastic artifacts (camera index, noise randn, multinomial
samples) AS USED by the reference implementation — these become the
PythonReplay inputs for SP-4/SP-5 validation.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make AAA-Gaussians importable
AAA_ROOT = Path(__file__).resolve().parents[1] / "AAA-Gaussians"
sys.path.insert(0, str(AAA_ROOT))

from tools.dump_npy import NpyDumper


def run_basketball(output_dir: str, steps, source_path: str, ply_path: str,
                   iterations: int, seed: int,
                   densify_from_iter: int, densify_until_iter: int,
                   densification_interval: int):
    """Run basketball training and dump at specified ladder steps.

    Args:
        steps: sorted list of step numbers to dump at
        iterations: total training steps
    """
    from argparse import Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams, OptimizationParams, \
        get_combined_args
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss
    from utils.general_utils import safe_state, build_scaling_rotation
    from diff_gaussian_rasterization import _C
    try:
        from fused_ssim import fused_ssim
    except ImportError:
        print("WARNING: fused_ssim not installed; SSIM artifacts will be absent",
              file=sys.stderr)
        fused_ssim = None

    # Seed everything
    safe_state(silent=False)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Minimal args construction — mirror train.py's parse
    # ... [implementation: populate ModelParams, OptimizationParams, PipelineParams
    #      matching basketball defaults and SP-4 overrides]

    # Training loop — same ordering as train.py:80-148
    # At each step in `steps`, call dump_step() which writes all artifacts.

    # This function body is intentionally deferred — it depends on the
    # exact AAA-Gaussians API surface. For SP-0 task 18, scope is limited
    # to getting step=1 dumped correctly. Subsequent steps (10, 100, ...)
    # follow the same pattern.

    raise NotImplementedError(
        "dump_basketball.py: wire up full training loop in follow-up sub-task. "
        "This is intentional: requires AAA-Gaussians internal API traversal "
        "best done with execution feedback loop. See plan Task 19.")


def dump_step_artifacts(dumper: NpyDumper, gaussians, scene, cam_idx: int,
                        step: int, rng_state: dict):
    """Dump all forward + backward + RNG artifacts for one step.

    rng_state must contain: cam_index (int), noise_randn (tensor),
        densify_reinit_idx (tensor or None), densify_add_idx (tensor or None).
    """
    # Forward via _C.rasterize_gaussians + _C.materialize_dump (Task 16 pattern)
    # Backward via _C.rasterize_gaussians_backward_dump
    # Adam state dump via optimizer.state_dict()
    # Noise dump via retain_grad pattern
    # ...
    pass
```

- [ ] **Step 2: Recognize this task is incomplete — add placeholder commit**

This task is a structural placeholder. Full implementation requires interactive debugging with the AAA-Gaussians API. Task 19 will do it.

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git add tools/dump_basketball.py
git commit -m "sp0: dump_basketball.py skeleton — deferred implementation (Task 19)"
```

---

## Task 19: Complete `dump_basketball.py` — step=1 end-to-end

**Files:**
- Modify: `tools/dump_basketball.py`

Implement full step=1 dump, verify, then generalize to ladder.

- [ ] **Step 1: Write minimal training loop**

Populate `run_basketball` by closely mirroring `AAA-Gaussians/train.py:32-148`. Keep only what is required to reach step=1 and dump. Key imports and arg construction must match reference exactly (model params, optimizer params).

The implementation MUST:
1. Load scene from `source_path` and init `GaussianModel` from `ply_path`
2. Build optimizer with 6 param groups (xyz, f_dc, f_rest, opacity, scaling, rotation) and eps=1e-15
3. Per step: compute LR, sample camera (save cam_idx), render, compute loss, retain_grad on image, backward, dump artifacts at ladder steps, densify, adam step, zero_grad, compute noise (save randn), apply noise

At each ladder step, after loss.backward() but before densify/adam:
- Call `_C.materialize_dump` on cached buffers
- Call `_C.rasterize_gaussians_backward_dump` (re-run is inefficient; better: use the grads accumulated by autograd, plus call backward_dump just for d_conic)
- Record `rendered_image.grad` as `backward_dL_dout_color`

After adam step but before noise:
- Dump `xyz_after_adam`
- Dump all 6 param groups' `m`, `v`, `param_after` from `optimizer.state_dict()`

After noise injection:
- Dump `xyz_after_noise`

At densify steps, additionally dump `reinit_idx`, `add_idx`, `new_opacity`, `new_scaling`.

- [ ] **Step 2: Run for step=1 only**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -m tools.dump_tool --fixture=basketball \
    --output=dev_notes/ground_truth/sp5 --ladder=1 --iterations=1
```

Expected: `dev_notes/ground_truth/sp5/step000001/cam####/manifest.json` exists with full artifact set.

- [ ] **Step 3: Validate artifact shapes**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -c "
import json, numpy as np, os, glob
root = glob.glob('dev_notes/ground_truth/sp5/step000001/cam*/manifest.json')[0]
m = json.load(open(root))
cam_dir = os.path.dirname(root)
print(f'{len(m[\"artifacts\"])} artifacts at step 1')
# Verify critical shape invariants
for a in m['artifacts']:
    arr = np.load(os.path.join(cam_dir, a['filename']))
    assert list(arr.shape) == a['shape'], (a['filename'], arr.shape, a['shape'])
    assert str(arr.dtype) == a['dtype'], (a['filename'], arr.dtype, a['dtype'])
print('all shape/dtype manifest entries match disk')
# Spot check: d_means2D must be [P, 3]
dm2d = np.load(os.path.join(cam_dir, 'backward_d_means2D.npy'))
assert dm2d.shape[1] == 3, dm2d.shape
print('d_means2D confirmed [P, 3]')
"
```

Expected: `all shape/dtype manifest entries match disk` + `d_means2D confirmed [P, 3]`.

- [ ] **Step 4: Commit**

```bash
git add tools/dump_basketball.py
git commit -m "sp0: implement basketball step=1 dump with full artifact set"
```

---

## Task 20: Extend to full ladder — 1, 10, 100, 500, 1000, 2000

**Files:**
- Modify: `tools/dump_basketball.py`

- [ ] **Step 1: Ladder support**

In `run_basketball`, loop steps 1..iterations. At each step in the `steps` set, call `dump_step_artifacts`. Ensure the training loop continues normally after dumping — **no side effect on model state from dumping**.

- [ ] **Step 2: Run full ladder (may take ~30 minutes on GPU)**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -m tools.dump_tool --fixture=basketball \
    --output=dev_notes/ground_truth/sp5 \
    --ladder=1,10,100,500,1000,2000 --iterations=2000
```

Expected: 6 step directories created, each with full manifest.

- [ ] **Step 3: Verify**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
ls dev_notes/ground_truth/sp5/
# Expect: step000001 step000010 step000100 step000500 step001000 step002000

# Spot check count consistency
conda run -n aaa-gs python -c "
import json, glob
for d in sorted(glob.glob('dev_notes/ground_truth/sp5/step*/cam0000/manifest.json')):
    m = json.load(open(d))
    print(f'step {m[\"step\"]}: {len(m[\"artifacts\"])} artifacts')
"
```

Expected: all 6 steps report similar artifact count (with extra relocation artifacts only on densify steps — but none are in this range since densify_from=5000).

- [ ] **Step 4: Commit (metadata only — artifacts gitignored)**

```bash
git add tools/dump_basketball.py
git commit -m "sp0: support full basketball ladder 1,10,100,500,1000,2000"
```

Artifacts themselves are not committed (gitignored per Task 1). They are regenerated locally by running `dump_tool.py`.

---

## Task 21: `--dump_steps` on-demand mode for bisection

**Files:**
- Modify: `tools/dump_tool.py`
- Modify: `tools/dump_basketball.py`

- [ ] **Step 1: Already wired**

`dump_tool.py` already accepts `--dump_steps`. In `dump_basketball.py`, the `steps` set is already unioned. Verify with:

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -m tools.dump_tool --fixture=basketball \
    --output=dev_notes/ground_truth/sp5 \
    --dump_steps=1234 --iterations=2000
```

Expected: `step001234/` directory appears.

- [ ] **Step 2: Commit (if any tweaks)**

Only commit if changes were needed. Otherwise skip.

---

## Task 22: `dev_notes/dependencies.md` — fused_ssim + CUB verdict

**Files:**
- Create: `dev_notes/dependencies.md`

- [ ] **Step 1: Locate fused_ssim source**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
conda run -n aaa-gs python -c "
import fused_ssim, inspect, os
print('fused_ssim:', fused_ssim.__file__)
print('dir:', os.path.dirname(fused_ssim.__file__))
"
ls $(conda run -n aaa-gs python -c "import fused_ssim, os; print(os.path.dirname(fused_ssim.__file__))")
```

- [ ] **Step 2: Record version, install path, commit or release info**

Create `dev_notes/dependencies.md`:
```markdown
# SP-0 External Dependency Inventory

## fused_ssim
- Package: `fused_ssim`
- Install location (aaa-gs env): `<paste from step 1>`
- Source availability: <if source in site-packages, note path; if only .so, note "binary wheel — fetch source from git">
- Git source: <URL if known; otherwise "TBD: locate and pin">
- Pinned version/commit: <paste from `pip show fused_ssim` → Version>
- Downstream usage: `AAA-Gaussians/train.py:103` (`fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))`)
- SP-4 impact: Vulkan SSIM shader MUST match this implementation's output.
  If source not available: SP-4 SSIM blocked until source is located OR
  numerical probing fits an equivalent pure-PyTorch SSIM.

## CUB DeviceRadixSort determinism
- Probe: `tools/verify_cub_determinism.py`
- Result (paste from Task 2 run): <DETERMINISTIC | NON-DETERMINISTIC>
- Hardware tested: <GPU name from `nvidia-smi`>
- Implication:
  - If DETERMINISTIC: sort golden verification uses exact match.
  - If NON-DETERMINISTIC: downgrade to tile-grouping consistent (see spec §1.2).

## PyTorch / CUDA toolchain
- PyTorch version: <paste from `pip show torch` in aaa-gs>
- CUDA version: <paste from `nvcc --version`>
- Extension build flags: see `AAA-Gaussians/submodules/diff-gaussian-rasterization/setup.py`
```

Paste real values collected in step 1 and from the Task 2 run.

- [ ] **Step 3: Commit**

```bash
git add dev_notes/dependencies.md
git commit -m "sp0: document fused_ssim version and CUB determinism verdict"
```

---

## Task 23: Final validation — all 137 + new tests green

**Files:** none (run validation only)

- [ ] **Step 1: Full build clean**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_OPENCL=OFF
cmake --build build 2>&1 | tail -10
```

Expected: build completes, no errors.

- [ ] **Step 2: Run full test suite**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -20
```

Expected: `100% tests passed, 0 tests failed out of <137 + new>`.
The new tests added by this plan:
- NpyReader (7 tests)
- Compare (7 tests)
- Manifest (2 tests)
- GoldenRoundtrip (4 tests)
Total expected: 137 + 20 = 157 (allow ±2 based on GPU test skips).

- [ ] **Step 3: Tag SP-0 complete**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git tag -a sp0-infrastructure-ready -m "SP-0 infrastructure complete: dump_tool + npy_reader + compare + manifest, tiny fixture + basketball ladder reproducible, dependencies documented"
```

- [ ] **Step 4: Commit (no files, just tag)**

Tag is local until pushed. SP-0 exit criteria met.

---

## SP-0 Exit Criteria (verification before moving to SP-1)

- [ ] All 23 tasks completed
- [ ] `ctest` shows ≥ 157 passing tests (137 baseline + 20 new)
- [ ] `harmonyos_3dgs/tests/golden/tiny/step000001/cam0000/manifest.json` exists and is committed
- [ ] `dev_notes/ground_truth/sp5/step00{0001,0010,0100,0500,1000,2000}/` all present locally (gitignored, regeneratable)
- [ ] `dev_notes/dependencies.md` has concrete fused_ssim version and CUB verdict
- [ ] `tools/dump_tool.py --help` works in aaa-gs env
- [ ] `_C.materialize_dump` and `_C.rasterize_gaussians_backward_dump` callable from Python
- [ ] Git tag `sp0-infrastructure-ready` exists
