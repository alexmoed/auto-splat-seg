#!/usr/bin/env python3
"""Render a gaussian splat PLY from a camera. One primitive, one output.

Does ONE thing: load a gsplat PLY, place a camera, rasterize the scene to a
PNG. No cropping, no cleanup, no segmentation, no overlays.

Usage:
    python view.py scene.ply out.png --yaw 30 --pitch 15 --fov 70

Requires: gsplat (with CUDA), torch, plyfile, numpy, Pillow.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. See references/gaussian-splat-viewer-install.md")
    sys.exit(1)

try:
    from gsplat import rasterization
except ImportError:
    print("ERROR: gsplat not installed. See references/gaussian-splat-viewer-install.md")
    sys.exit(1)


SH_C0 = 0.28209479177387814


# ---------- PLY loader ----------

def load_gsplat_ply(path):
    """Parse a gaussian splat PLY. Returns dict with torch tensors on CUDA for:
    means (N,3), quats (N,4 wxyz), scales (N,3), opacities (N,), colors (N,3) from f_dc.
    SH higher degrees are ignored here; colors are DC only, which is what we need
    for a simple overview render.
    """
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = v.data.dtype.names

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    colors = f_dc * SH_C0 + 0.5

    if all(n in names for n in ("rot_0", "rot_1", "rot_2", "rot_3")):
        quats = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1
        ).astype(np.float32)
    else:
        quats = np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (len(means), 1))

    if all(n in names for n in ("scale_0", "scale_1", "scale_2")):
        log_scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=1
        ).astype(np.float32)
        scales = np.exp(log_scales)
    else:
        scales = np.full((len(means), 3), 0.01, dtype=np.float32)

    if "opacity" in names:
        logit = v["opacity"].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-logit))
    else:
        opacities = np.ones(len(means), dtype=np.float32)

    t = lambda a: torch.from_numpy(a).contiguous().cuda()
    return {
        "means": t(means),
        "quats": t(quats),
        "scales": t(scales),
        "opacities": t(opacities),
        "colors": t(colors),
    }


# ---------- Camera math ----------

def rotation_matrix_from_yaw_pitch(yaw_deg, pitch_deg):
    """World-to-camera rotation given yaw around Y and pitch around X (after yaw).
    The camera looks along -Z (OpenCV/typical gsplat convention).
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    Ry = np.array([
        [np.cos(y),  0, np.sin(y)],
        [0,          1, 0],
        [-np.sin(y), 0, np.cos(y)],
    ], dtype=np.float32)
    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(p),    -np.sin(p)],
        [0, np.sin(p),     np.cos(p)],
    ], dtype=np.float32)
    return Rx @ Ry


def viewmat_look_at(eye, target, up=(0, 1, 0)):
    """OpenCV-style world-to-camera matrix. Camera looks at target; -Z is forward."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.stack([right, -true_up, -forward], axis=0)
    t = -R @ eye
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def build_K(fov_deg, width, height):
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([
        [f,  0, width * 0.5],
        [0,  f, height * 0.5],
        [0,  0, 1],
    ], dtype=np.float32)
    return K


# ---------- Rendering ----------

def render_splat(scene, V_wc, K, width, height, bg=(1, 1, 1)):
    flip_z = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    V_gs = flip_z @ V_wc
    V = torch.from_numpy(V_gs).float().cuda().unsqueeze(0)
    Kt = torch.from_numpy(K).float().cuda().unsqueeze(0)
    bg_t = torch.tensor(bg, dtype=torch.float32, device="cuda").reshape(3)
    img, _alpha, _meta = rasterization(
        means=scene["means"],
        quats=scene["quats"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        colors=scene["colors"],
        viewmats=V,
        Ks=Kt,
        width=width,
        height=height,
        backgrounds=bg_t,
    )
    rgb = img[0].clamp(0, 1).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("out_png")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fov", type=float, default=70)
    p.add_argument("--yaw", type=float, default=0, help="camera yaw around world Y, degrees")
    p.add_argument("--pitch", type=float, default=0, help="camera pitch (down positive), degrees")
    p.add_argument("--distance", type=float, default=None,
                   help="eye distance from scene center; default = scene radius * 1.8")
    p.add_argument("--bg", default="1,1,1", help="background rgb, floats 0-1")
    p.add_argument("--y-down", action="store_true",
                   help="scene uses -Y up (COLMAP-style); flip the up vector")
    p.add_argument("--eye", default=None,
                   help="explicit camera eye position 'x,y,z' in world coords; "
                        "overrides yaw/pitch/distance auto-placement")
    p.add_argument("--target", default=None,
                   help="explicit look-at target 'x,y,z'; required when --eye "
                        "is set; defaults to scene center otherwise")
    p.add_argument("--up", default=None,
                   help="explicit world-up vector 'x,y,z' (e.g. '0,0,1' for "
                        "axis-aligned topdowns where forward is parallel to "
                        "world y and the default y-only up gimbal-locks). "
                        "Default: (0,1,0) y-up, or (0,-1,0) with --y-down.")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA not available; gsplat needs a GPU"

    scene = load_gsplat_ply(args.ply)
    means_np = scene["means"].detach().cpu().numpy()
    print(f"[view] loaded {len(means_np):,} splats")

    bb_min = means_np.min(axis=0)
    bb_max = means_np.max(axis=0)
    center = (bb_min + bb_max) * 0.5
    radius = float(np.linalg.norm(bb_max - bb_min) * 0.5)
    distance = args.distance if args.distance is not None else radius * 1.8
    if args.up is not None:
        up = np.array([float(v) for v in args.up.split(",")], dtype=np.float32)
        up = up / max(np.linalg.norm(up), 1e-9)
    else:
        up = np.array([0, -1 if args.y_down else 1, 0], dtype=np.float32)
    if args.eye is not None:
        eye = np.array([float(v) for v in args.eye.split(",")],
                       dtype=np.float32)
        target = np.array(
            [float(v) for v in args.target.split(",")], dtype=np.float32
        ) if args.target else center
        V = viewmat_look_at(eye, target, up)
    else:
        base_eye = center + np.array([0, 0, distance], dtype=np.float32)
        R = rotation_matrix_from_yaw_pitch(args.yaw, args.pitch)
        eye = center + R.T @ (base_eye - center)
        V = viewmat_look_at(eye, center, up)
    K = build_K(args.fov, args.width, args.height)

    bg_rgb = tuple(float(x) for x in args.bg.split(","))
    img = render_splat(scene, V, K, args.width, args.height, bg=bg_rgb)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    Image.fromarray(img).save(args.out_png)
    print(f"[view] wrote {args.out_png}")


if __name__ == "__main__":
    main()
