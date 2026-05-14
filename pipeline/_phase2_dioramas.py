#!/usr/bin/env python3
"""Phase 2 diorama renders — 4 quadrants of the cleaned scene.

Per quadrant: keep ONLY that quadrant's splats (drop other 3 quadrants),
render axis-aligned ACROSS the inner cut wall.
- NE / NW: camera south of cz, looking +z
- SE / SW: camera north of cz, looking -z

Reads:  <scene>/_phase1_temp/scene_minus_phase1.ply
Writes: <scene>/_phase2_temp/diorama_<NE|NW|SE|SW>.ply
        <scene>/_phase2_temp/quad_<NE|NW|SE|SW>.png   (4K, FOV 70)
        <scene>/_phase2_temp/cameras.json             (single source of truth
                                                       for the 4 diorama cams +
                                                       room bounds — phase 3
                                                       reads this; never
                                                       re-derive elsewhere)

Locked params:
  BACK = 2.0m              (camera distance from cut face)
  eye_y = centroid_y - 1m  (camera height — above centroid in y-down)
  Render: 3840×2160, FOV 70

Usage:
    python _phase2_dioramas.py <scene_dir>
"""
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
W, H = 3840, 2160
FOV = 70.0
BACK = 2.0


def main():
    scene = Path(sys.argv[1]).resolve()
    ply = scene / "_phase1_temp" / "scene_minus_phase1.ply"
    out = scene / "_phase2_temp"
    out.mkdir(parents=True, exist_ok=True)

    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    xs = np.array(v["x"]).astype(np.float64)
    ys = np.array(v["y"]).astype(np.float64)
    zs = np.array(v["z"]).astype(np.float64)
    xb_min, xb_max = float(np.percentile(xs, 15)), float(np.percentile(xs, 85))
    zb_min, zb_max = float(np.percentile(zs, 15)), float(np.percentile(zs, 85))
    cx, cz = (xb_min + xb_max) / 2, (zb_min + zb_max) / 2
    cy = float(ys.mean())
    eye_y = cy - 1.0

    print(f"x [{xb_min:.2f},{xb_max:.2f}]  z [{zb_min:.2f},{zb_max:.2f}]"
          f"  eye_y={eye_y:.2f}  back={BACK}m")

    quads = [
        ("NE", (xs >= cx) & (zs >= cz), (cx + xb_max) / 2, cz - BACK,
                                         (cx + xb_max) / 2, zb_max),
        ("NW", (xs <= cx) & (zs >= cz), (xb_min + cx) / 2, cz - BACK,
                                         (xb_min + cx) / 2, zb_max),
        ("SE", (xs >= cx) & (zs <= cz), (cx + xb_max) / 2, cz + BACK,
                                         (cx + xb_max) / 2, zb_min),
        ("SW", (xs <= cx) & (zs <= cz), (xb_min + cx) / 2, cz + BACK,
                                         (xb_min + cx) / 2, zb_min),
    ]
    cameras_out = {}
    for tag, keep, ex, ez, tx, tz in quads:
        sub_data = v.data[keep]
        sub_ply = out / f"diorama_{tag}.ply"
        PlyData([PlyElement.describe(sub_data, "vertex")],
                text=False).write(str(sub_ply))
        eye = (ex, eye_y, ez)
        tgt = (tx, eye_y, tz)
        png = out / f"quad_{tag}.png"
        subprocess.run([
            sys.executable, VIEW_PY, str(sub_ply), str(png),
            f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
            f"--target={tgt[0]:.4f},{tgt[1]:.4f},{tgt[2]:.4f}",
            "--up=0,-1,0", "--y-down",
            "--fov", str(FOV), "--width", str(W), "--height", str(H),
        ], check=True, capture_output=True)
        target_dist = float(np.linalg.norm(np.array(tgt) - np.array(eye)))
        cameras_out[tag] = {
            "eye":         [float(c) for c in eye],
            "target":      [float(c) for c in tgt],
            "up":          [0.0, -1.0, 0.0],
            "fov":         FOV,
            "width":       W,
            "height":      H,
            "y_down":      True,
            "target_dist": target_dist,
        }
        print(f"{tag}: keep={int(keep.sum()):,}/{len(xs):,}  "
              f"eye={eye} → tgt={tgt}  dist={target_dist:.2f}m")

    cameras_out["bounds"] = {
        "cx": cx, "cz": cz,
        "xb_min": xb_min, "xb_max": xb_max,
        "zb_min": zb_min, "zb_max": zb_max,
        "eye_y": eye_y, "cy": cy,
        "back":  BACK,
    }
    cam_json = out / "cameras.json"
    cam_json.write_text(json.dumps(cameras_out, indent=2))
    print(f"\n[cameras] {cam_json}")


if __name__ == "__main__":
    main()
