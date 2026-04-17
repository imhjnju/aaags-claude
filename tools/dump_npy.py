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
        # np.save auto-appends ".npy" if path doesn't end with it; pass a file
        # handle to avoid that so we can atomic-rename the exact tmp path.
        with open(tmp, "wb") as f:
            np.save(f, arr)
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
