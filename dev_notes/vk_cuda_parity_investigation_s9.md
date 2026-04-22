# VK-CUDA Parity 调查报告 (Session 9, 2026-04-22)

## 目标
VK eval_3D 光栅化器 vs CUDA golden PSNR ≥60 dB。当前 **42.789 dB**，差距 **17.2 dB**。

---

## 已排除的原因

| 尝试 | PSNR | 结论 |
|------|------|------|
| z/w 深度键修正（HEAD 用 z_ndc/w_ndc） | 42.784 dB | 深度键公式不是问题 |
| HEAD_W=16 | 42.832 dB | HEAD 窗口大小不是瓶颈 |
| HEAD_W=64 | 42.798 dB | 同上，系统天花板 |
| 持久化 per-sub-tile TAIL（无裁剪） | 18.212 dB | 子 tile 收集溢出（MAX_TAIL=256，4 批×256=1024） |
| 持久化 per-sub-tile TAIL + z 范围裁剪 | 18.254 dB | 裁剪不足以防止溢出 |
| Flat collect + HEAD_W=8 | 41.846 dB | 无 batch 排序时 HEAD 提供 +4.9 dB |
| Flat collect + HEAD_W=64 | 42.798 dB | ≈ 当前基线，证明 HEAD 大小不是天花板 |
| Flat loop（无 batch，无 HEAD，直接混合） | 36.857 dB | 全局内存读取本身工作正常 |

**核心结论**：42.8 dB 是一个系统性上限，与 HEAD 溢出和跨批次排序无关。HEAD_W>8 不再带来显著收益。

---

## 像素级诊断结果

### 整体指标
| 指标 | 值 |
|------|-----|
| PSNR | 42.789 dB |
| max_abs | 0.24252 |
| mean_abs | 0.00327 |
| p99_abs | 0.02785 |
| bad pixels (>1e-3) | 708,015 / 2,073,600 (34.1%) |

### 前景 vs 背景
| 区域 | 像素数 | PSNR | mean_abs |
|------|--------|------|----------|
| 前景 (CUDA lum > 0.01) | 678,493 | 42.71 dB | — |
| 背景 (CUDA lum ≤ 0.01) | 12,707 | 78.69 dB | — |

**背景几乎完美，所有误差在前景。**

### 亮度偏差分析
| 方向 | 像素数 | 百分比 |
|------|--------|--------|
| VK 更亮 | 219,003 | 31.7% |
| CUDA 更亮 | 17,738 | 2.6% |
| 近似相等 | 454,459 | 65.7% |

**VK 系统性地偏亮**，所有通道正偏差：R=+0.0024, G=+0.0033, B=+0.0034。

### 误差 vs 亮度（CUDA luminance 分桶）
| 亮度范围 | 像素数 | mean_rmse | PSNR |
|----------|--------|-----------|------|
| [0.00, 0.05) | 46,131 | 0.00007 | 67.9 dB |
| [0.05, 0.20) | 82,370 | 0.00023 | 59.6 dB |
| [0.20, 0.40) | 136,377 | 0.00041 | 53.2 dB |
| [0.40, 0.60) | 81,685 | 0.00060 | 50.6 dB |
| [0.60, 0.80) | 82,710 | 0.00110 | 48.6 dB |
| **[0.80, 1.00)** | **261,784** | **0.00792** | **38.9 dB** |
| [1.00, 2.00) | 143 | 0.00075 | 51.7 dB |

**高亮度区域（0.8-1.0）是误差主力**，占 26 万像素，PSNR 仅 38.9 dB。

### 空间 PSNR 分布
| 区域 | PSNR | mean_abs |
|------|------|----------|
| 左上 | 57.54 dB | 0.00019 |
| 右上 | 50.94 dB | 0.00044 |
| 左下 | 37.45 dB | 0.01030 |
| 右下 | 46.81 dB | 0.00214 |
| 中心 | 50.19 dB | 0.00108 |

**左下象限最差**（37.45 dB），对应深度复杂的前景区域。

### 最差 16×16 tiles
Top tiles 都集中在 y=448, 576, 928-944 附近——几何复杂区域，PSNR 低至 30.2 dB。

### 误差直方图
| 区间 | 像素数 |
|------|--------|
| [0.00, 0.05) | 691,069 |
| [0.05, 0.10) | 100 |
| [0.10, 0.15) | 20 |
| [0.15, 0.20) | 11 |
| ≥ 0.20 | 0 |

99.98% 的像素误差 <0.05，但微小系统性偏差累积成 42.8 dB 的天花板。

---

## CUDA vs VK 架构差异

### CUDA 层次渲染器（sort_mode=3）
```
全局排序（per-tile PER_TILE_DEPTH_MAXPOS + radix sort）
    ↓
TAIL: 64 个槽位/4×4 sub-tile, bitonic sort (per-pixel depth)
    ↓ 推送最浅的到 MID
MID: 8-12 个槽位/4×4 sub-tile, 归并排序
    ↓ 推送最浅的到 HEAD
HEAD: 4 个槽位/像素, 插入排序
    ↓ 满时弹出最浅的执行 alpha blend
```

### VK 当前实现
```
全局排序（per-tile PER_TILE_DEPTH_MAXPOS + radix sort） — 匹配 CUDA ✓
    ↓
每 256-Gaussian 批次内: sub-tile center sort (16 sub-tiles × 深度排序)
    ↓ 按排序顺序送入 HEAD
HEAD: 8 个槽位/像素, 插入排序
    ↓ 满时弹出最浅的执行 alpha blend
```

### 关键差异
| 组件 | CUDA | VK | 影响 |
|------|------|-----|------|
| 跨批次持久排序 | TAIL 跨批次累积 | 每批次独立排序 | 高 |
| 逐像素深度重排 | TAIL 64-slot bitonic sort | 无 | 高 |
| 4×4 MID 层 | 8-slot refine | 无 | 中 |
| HEAD 大小 | 4 | 8 | VK 更大（已验证不关键） |

---

## 正在调查的新方向

### 1. 投影矩阵构建差异（优先级最高）

**VK** (`camera_utils.cpp`):
```cpp
proj[0] = 1.0f / tan_fovx;
proj[5] = 1.0f / tan_fovy;
proj[10] = zfar / (zfar - znear);
proj[11] = 1.0f;
proj[14] = -(zfar * znear) / (zfar - znear);
```

**Python/CUDA** (`graphics_utils.py`):
```python
P[0,0] = 2.0 * znear / (right - left)    # = 1/tan(fov/2) 当 left=-right
P[1,1] = 2.0 * znear / (top - bottom)     # = 1/tan(fov/2)
P[2,2] = zfar / (zfar - znear)
P[2,3] = -(zfar * znear) / (zfar - znear)
P[3,2] = 1.0  # z_sign = 1.0
```

数学上应等价（`1/tan(fov/2) = 2*znear/(2*tan(fov/2)*znear)`），但需逐元素验证。

**View-Projection 乘法顺序**：
- VK: `viewproj = proj × view`
- Python: `full_proj = world_view × projection`

在列主序矩阵中，`proj × view` 和 `world_view^T × proj^T` 可能不同，取决于矩阵存储约定。需确认两个 ViewProj 矩阵元素完全一致。

### 2. 预处理阶段中间结果对比
如果投影矩阵一致，需对比：
- means2D（屏幕坐标）
- gauss2screen 矩阵
- 激活后的 opacity 值
- SH 计算的 RGB 值

定位哪个阶段开始产生差异。

### 3. 其他未验证原因
- SH 系数常数是否完全匹配 CUDA
- filter_3D 参数是否正确传递
- PLY 加载中 SH 系数重排是否正确

---

## 下一步行动

1. **逐元素对比 ViewProj 矩阵** — 在 VK 测试中打印完整 4×4 矩阵，与 Python 计算的矩阵对比
2. **如果矩阵一致** — dump 预处理中间结果（means2D, opacity, rgb），定位差异起点
3. **如果矩阵不一致** — 修复矩阵构建，可能直接突破 42.8 dB
4. **考虑 CUDA 3 级层次排序的完整移植** — 这是架构性差异，即使预处理完全一致，排序精度差异也会限制 PSNR 上限
