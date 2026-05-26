#!/usr/bin/env python3
"""Render before/after wipe pairs.

For each named object: render the object PLY alone AND scene_background.ply
with the SAME camera, so a website can wipe-cross-dissolve between them.

Camera convention per object:
  - position = on the room-center side of the object, pushed back enough
    to fit the object at FOV 35° with ~70% framing
  - target   = object centroid
  - up       = world up (y-down convention)
  - pitch    = slightly elevated above object (looking down ~12°)

Output: 1920×1080 PNGs in <scene>/docs/wipe/.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
PYTHON = "/home/ubuntu/anaconda3/envs/claude_seg/bin/python"

W, H = 1920, 1080
FOV = 35.0  # tighter than canonical 60 → cleaner front-on framing
ELEVATION_PITCH_DEG = 12.0  # slight look-down
FRAMING_MARGIN = 2.6  # object fills ~38% of frame — lots of breathing room


def load_xyz(ply: Path) -> np.ndarray:
    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def compute_camera(obj_xyz: np.ndarray, scene_xyz: np.ndarray,
                   yaw_offset_deg: float = 0.0,
                   orbit_right_m: float = 0.0,
                   pan_chair_right_m: float = 0.0,
                   lift_m: float = 0.0,
                   margin: float = FRAMING_MARGIN):
    """Return (eye, target, distance) for a front-on slightly-elevated cam.

    "Front-on" = direction from scene_center → object_centroid, projected
    onto the xz-plane (we lift the camera vertically separately). Pitch
    elevation is applied by raising the eye in -y (y-down convention).
    `yaw_offset_deg` rotates the camera around the object's vertical axis.
    """
    # Object bbox extents — use the GEOMETRIC bbox center, not mean.
    # The mean gets dragged by outlier halos / floor residue; bbox center
    # keeps the object visually centered in the frame.
    # Use 5/95 percentiles to ignore extreme halo outliers.
    lo = np.percentile(obj_xyz, 2, axis=0)
    hi = np.percentile(obj_xyz, 98, axis=0)
    obj_c = (lo + hi) / 2
    ext = hi - lo
    # Diagonal in xz (front-projection size)
    diag_xz = math.hypot(ext[0], ext[2])
    height_y = abs(ext[1])
    # Camera distance — frame the LARGER of the xz-diag and the y-height
    aspect = W / H
    tan_h = math.tan(math.radians(FOV / 2))
    fit = max(diag_xz / aspect, height_y) * margin
    dist = fit / (2 * tan_h)

    # Room-side direction (in xz)
    scene_c_xz = np.array([scene_xyz[:, 0].mean(), scene_xyz[:, 2].mean()])
    obj_c_xz = np.array([obj_c[0], obj_c[2]])
    away = obj_c_xz - scene_c_xz  # from scene center → object
    n = np.linalg.norm(away)
    if n < 1e-6:
        # object IS at scene center — fall back to looking along -z
        dir_xz = np.array([0.0, -1.0])
    else:
        # Camera should be on the scene-center side, so direction FROM camera
        # TO object = (object - scene_center) normalized → camera at obj - dir*dist
        dir_xz = away / n

    # Apply optional yaw rotation around the object's vertical axis
    if abs(yaw_offset_deg) > 1e-6:
        a = math.radians(yaw_offset_deg)
        c, s = math.cos(a), math.sin(a)
        dx, dz = dir_xz
        dir_xz = np.array([c * dx - s * dz, s * dx + c * dz])

    # Eye position in xz: object - dir * dist (camera is between scene center
    # and the object, far enough back to frame it)
    eye_x = obj_c[0] - dir_xz[0] * dist
    eye_z = obj_c[2] - dir_xz[1] * dist

    # Vertical elevation in y-down: -tan(pitch) * dist makes camera ABOVE
    # the target (which in y-down means smaller y value)
    rise = -math.tan(math.radians(ELEVATION_PITCH_DEG)) * dist
    eye_y = obj_c[1] + rise  # rise is negative → eye_y < target y

    cam_right = np.array([dir_xz[1], -dir_xz[0]])

    # Orbit: shift ONLY the eye sideways. Chair stays centered, view angle
    # changes.
    if abs(orbit_right_m) > 1e-6:
        eye_x += orbit_right_m * cam_right[0]
        eye_z += orbit_right_m * cam_right[1]

    # Pan: shift BOTH eye and target sideways. The chair moves in frame.
    # Positive pan_chair_right_m → chair shifts screen-right (camera pans
    # left, so the object stays put while the framing shifts left).
    tx, tz = obj_c[0], obj_c[2]
    if abs(pan_chair_right_m) > 1e-6:
        shift = -pan_chair_right_m * cam_right
        eye_x += shift[0]
        eye_z += shift[1]
        tx += shift[0]
        tz += shift[1]

    # Lift: raise camera (in y-down, positive lift = smaller eye_y).
    eye_y -= lift_m

    eye = (eye_x, eye_y, eye_z)
    target = (tx, obj_c[1], tz)
    return eye, target, dist


def render(ply: Path, out: Path, eye, target):
    cmd = [PYTHON, VIEW_PY, str(ply), str(out),
           f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
           f"--target={target[0]:.4f},{target[1]:.4f},{target[2]:.4f}",
           "--y-down", "--fov", str(FOV),
           "--width", str(W), "--height", str(H)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--objects", nargs="+", required=True,
                    help="object slugs (e.g. 02_grey_armchair); append "
                         "':<yaw_deg>' to rotate the camera around the "
                         "object's vertical axis (e.g. 02_grey_armchair:90)")
    ap.add_argument("--stage", default="5_sweep_fallback",
                    help="which stage PLY to use (basename w/o .ply)")
    ap.add_argument("--background-ply", default="scene_reassembled.ply",
                    help="which PLY to render as the 'before' background "
                         "(default scene_reassembled.ply — full room)")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_ply = scene / args.background_ply
    if not bg_ply.exists():
        sys.exit(f"missing {bg_ply}")
    print(f"[load] background: {bg_ply.name}")
    bg_xyz = load_xyz(bg_ply)
    print(f"  splats: {len(bg_xyz):,}")

    cameras = {}
    for spec in args.objects:
        orbit_right = pan_chair = lift = 0.0
        margin = FRAMING_MARGIN
        if ":" in spec:
            slug, params = spec.split(":", 1)
            parts = params.split(",")
            yaw_offset = float(parts[0]) if parts[0] else 0.0
            if len(parts) > 1 and parts[1]:
                orbit_right = float(parts[1])
            if len(parts) > 2 and parts[2]:
                pan_chair = float(parts[2])
            if len(parts) > 3 and parts[3]:
                lift = float(parts[3])
            if len(parts) > 4 and parts[4]:
                margin = float(parts[4])
        else:
            slug, yaw_offset = spec, 0.0
        obj_dir = scene / slug
        # Pick the highest-numbered stage that exists
        candidates = [
            f"{args.stage}.ply",
            "5_subtracted.ply", "5_sweep_fallback.ply",
            "4_sam_tight.ply", "3_floor_drop.ply",
            "2_sam_wide.ply", "1_visual_hull.ply",
        ]
        obj_ply = None
        for c in candidates:
            p = obj_dir / c
            if p.exists():
                obj_ply = p
                break
        if obj_ply is None:
            print(f"[warn] no PLY in {obj_dir}, skipping")
            continue

        short = slug.replace("02_", "")
        print(f"\n[{short}] using {obj_ply.name}")
        obj_xyz = load_xyz(obj_ply)
        print(f"  splats: {len(obj_xyz):,}")

        eye, target, dist = compute_camera(obj_xyz, bg_xyz, yaw_offset,
                                           orbit_right, pan_chair, lift,
                                           margin)
        cameras[short] = {"eye": list(eye), "target": list(target),
                          "fov": FOV, "width": W, "height": H,
                          "source_ply": str(obj_ply)}
        print(f"  eye    = ({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})")
        print(f"  target = ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        print(f"  dist   = {dist:.3f}")

        obj_png = out_dir / f"{short}_object.png"
        bg_png = out_dir / f"{short}_background.png"
        render(obj_ply, obj_png, eye, target)
        render(bg_ply, bg_png, eye, target)
        print(f"  → {obj_png}")
        print(f"  → {bg_png}")

    cam_path = out_dir / "cameras.json"
    cam_path.write_text(json.dumps(cameras, indent=2))
    print(f"\n[done] cameras → {cam_path}")


if __name__ == "__main__":
    main()
