"""Tiny synthetic fixtures for algorithmic unit tests.

A tiny fixture is a deterministic small scene (5-50 Gaussians, synthetic camera)
that runs quickly and can be committed to git. Produced artifacts are used by
C++ unit tests that do NOT depend on the basketball dataset.

Matrix convention
-----------------
The AAA-Gaussians CUDA rasterizer consumes ``viewmatrix`` and ``projmatrix`` in
*transposed* form (same convention as ``AAA-Gaussians/scene/cameras.py``'s
``world_view_transform``), so translation lives at ``view[3, 0:3]`` rather than
``view[0:3, 3]``. A naive math-form ``view[2, 3] = -5.0`` places the world
origin outside the frustum and ``num_rendered`` collapses to 0 — the tests
become vacuous.

Scene setup here mirrors ``tools/smoke_materialize_dump.py`` (the T11 end-to-end
smoke) and ``tools/verify_cub_determinism.py`` (T2 CUB probe): set
``view[3, 2] = 5.0`` and cluster Gaussians near world z=-3 so camera-space
z ≈ +2 (> the 0.2 near-plane guard in ``in_frustum``).
"""
import torch
import numpy as np  # noqa: F401 — re-exported for downstream fixture dumpers


def build_tiny_scene(N: int = 20, seed: int = 42, H: int = 64, W: int = 64):
    """Build a minimal deterministic scene.

    Uses the same matrix conventions as ``tools/smoke_materialize_dump.py``
    (AAA-Gaussians transposed W2V + perspective projection transposed from
    math form). This convention is verified to produce non-zero
    ``num_rendered`` on the smoke path.

    Parameters
    ----------
    N : int
        Number of Gaussians. Default 20 (small enough to commit outputs).
    seed : int
        Seed for the CUDA RNG used to generate tensors. Deterministic.
    H, W : int
        Image height / width (pixels). Default 64x64.

    Returns
    -------
    dict with keys:
        means3D        : (N, 3) float32, clustered near world z=-3
        scales         : (N, 3) float32, in [0.05, 0.25]
        rotations      : (N, 4) float32, unit quaternions
        opacities      : (N, 1) float32, in [0.3, 0.8]
        sh             : (N, 16, 3) float32, small (* 0.1) so alpha < 1
        filter_3D      : (N,) float32, constant 0.01
        viewmatrix     : (4, 4) float32, transposed W2V (AAA convention)
        projmatrix     : (4, 4) float32, full view-projection (transposed)
        inv_viewprojmatrix : (4, 4) float32, inverse of projmatrix
        campos         : (3,) float32, world-space camera origin
        tan_fovx       : float, tangent of half-HFOV (1.0 → 90 deg HFOV)
        tan_fovy       : float, tangent of half-VFOV
        H, W           : int, image dimensions
        sh_degree      : int, 3
        sh_coeffs_per_g: int, 16 = (sh_degree + 1) ** 2
    """
    device = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)

    # --- Gaussians clustered near world z=-3 (see module docstring) ---------
    means3D = torch.randn((N, 3), generator=g, device=device) * 0.5
    means3D[:, 2] -= 3.0
    scales = torch.rand((N, 3), generator=g, device=device) * 0.2 + 0.05
    rotations = torch.nn.functional.normalize(
        torch.randn((N, 4), generator=g, device=device), dim=1
    )
    opacities = torch.rand((N, 1), generator=g, device=device) * 0.5 + 0.3

    sh_degree = 3
    M = (sh_degree + 1) ** 2  # 16
    sh = torch.randn((N, M, 3), generator=g, device=device) * 0.1
    filter_3D = torch.ones(N, device=device) * 0.01

    # --- Camera (transposed W2V per AAA-Gaussians/scene/cameras.py) ---------
    # view[3, 2] = 5.0 puts world origin at camera-space z=+5; Gaussians at
    # world z=-3 land at cam z=+2 (in frustum, past the 0.2 near-plane guard).
    view = torch.eye(4, device=device)
    view[3, 2] = 5.0

    # Perspective projection built in math form, then transposed to match the
    # rasterizer's row-major-as-transposed storage convention.
    tan_fov = 1.0
    znear, zfar = 0.1, 100.0
    proj_math = torch.zeros(4, 4, device=device)
    proj_math[0, 0] = 1.0 / tan_fov
    proj_math[1, 1] = 1.0 / tan_fov
    proj_math[2, 2] = zfar / (zfar - znear)
    proj_math[2, 3] = -znear * zfar / (zfar - znear)
    proj_math[3, 2] = 1.0
    proj = proj_math.T.contiguous()

    # Combined view-projection (both already transposed) and its inverse.
    viewproj = (view @ proj).contiguous()
    inv_viewproj = torch.linalg.inv(viewproj).contiguous()

    # campos matches tools/smoke_materialize_dump.py's working convention.
    # With the transposed W2V above the camera is treated as being at the
    # world origin relative to the clustered Gaussians — zeroing campos has
    # been verified (T11 smoke) to produce non-zero num_rendered.
    campos = torch.zeros(3, device=device)

    return dict(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh=sh,
        filter_3D=filter_3D,
        viewmatrix=view,
        projmatrix=viewproj,
        inv_viewprojmatrix=inv_viewproj,
        campos=campos,
        tan_fovx=float(tan_fov),
        tan_fovy=float(tan_fov),
        H=int(H),
        W=int(W),
        sh_degree=int(sh_degree),
        sh_coeffs_per_g=int(M),
    )
