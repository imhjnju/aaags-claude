# 默认配置对比：原始 Python 3DGS vs HarmonyOS C++ 移植

## 对比结果

| 配置项 | 原始 Python 3DGS 默认值 | C++ 移植默认值 | 一致 | 来源 |
|--------|------------------------|---------------|------|------|
| antialiasing | `False` | `false` | ✅ | `arguments/__init__.py` PipelineParams |
| white_background | `False` (黑色背景) | `0.0` (黑色背景) | ✅ | `arguments/__init__.py` ModelParams |
| sh_degree | `3` | `3` (从 PLY 模型读取) | ✅ | `arguments/__init__.py` ModelParams |
| scale_modifier | `1.0` | `1.0f` | ✅ | `gaussian_renderer/__init__.py` render() |
| tile_size | `16×16` (BLOCK_X/BLOCK_Y) | `16×16` (tile_w/tile_h) | ✅ | CUDA `config.h` |
| convert_SHs_python | `False` (CUDA kernel 内计算) | N/A (kernel 内计算) | ✅ | PipelineParams |
| compute_cov3D_python | `False` (CUDA kernel 内计算) | N/A (kernel 内计算) | ✅ | PipelineParams |
| prefiltered | `False` | 未实现 (等效 `False`) | ✅ | `gaussian_renderer/__init__.py` |
| debug | `False` | 无 debug 模式 (等效 `False`) | ✅ | PipelineParams |

## 环境变量控制

C++ 移植版通过环境变量覆盖默认配置，与原版命令行参数对应：

| 环境变量 | 默认 | 说明 | 对应原版参数 |
|----------|------|------|-------------|
| `AA=1` | OFF | 开启 Mip-Splatting 抗锯齿 | `--antialiasing` |
| `BG_WHITE=1` | 黑色 | 使用白色背景 | `--white_background` |
| `SH_DEGREE=N` | 从模型读取 | 覆盖 SH 阶数 (0-3) | `--sh_degree N` |
| `DUMP_DIAG=1` | OFF | 输出诊断信息 | 无对应 |
| `--gpu` | CPU | 使用 OpenCL GPU 后端 | 无对应 (原版仅 CUDA) |

## 确认方法

### 原始 Python 代码位置

```
arguments/__init__.py       → PipelineParams.antialiasing = False
                            → ModelParams._white_background = False
                            → ModelParams.sh_degree = 3

gaussian_renderer/__init__.py → render(scaling_modifier=1.0)
                              → prefiltered=False
                              → antialiasing=pipe.antialiasing

render.py                   → bg_color = [1,1,1] if white_background else [0,0,0]
```

### C++ 移植代码位置

```
include/types.h             → RenderConfig 默认值
src/main.cpp                → 环境变量读取和覆盖逻辑
```

## 结论

所有默认配置与原始 Python 3DGS **完全一致**。移植未改变任何渲染行为的默认值。
