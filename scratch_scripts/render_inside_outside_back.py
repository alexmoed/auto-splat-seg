#!/usr/bin/env python3
"""render_inside_outside_back.py — re-render 6_inside_outside canonical 5
with the cameras pulled back. Overrides the default 1.55× margin.

Usage:
    python render_inside_outside_back.py <obj_dir> [--margin 2.5]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/pipeline")
from sam_carve import (  # noqa: E402
    CANONICAL_YAWS, CANONICAL_PITCH, CANONICAL_W, CANONICAL_H,
    CANONICAL_TOPDOWN_PITCH, FOV, Y_DOWN, build_camera,
)
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--margin", type=float, default=2.5)
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    ply = obj / "6_inside_outside.ply"
    out = obj / "renders" / "6_inside_outside"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    print(f"[load] {ply}")
    scene = load_gsplat_ply(str(ply))
    means = scene["means"].detach().cpu().numpy()
    lo = np.percentile(means, 2, axis=0)
    hi = np.percentile(means, 98, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = max(float((hi - lo).max()), 0.15)
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * args.margin) / (2 * tan_half) + 0.5
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m margin={args.margin}")
    for yaw_deg in CANONICAL_YAWS:
        V, K, _ = build_camera(center, yaw_deg, CANONICAL_PITCH, distance,
                                FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
        img = render_splat(scene, V, K, CANONICAL_W, CANONICAL_H,
                            bg=(1.0, 1.0, 1.0))
        op = out / f"y{int(yaw_deg)}.png"
        Image.fromarray(img).save(op)
        print(f"[save] {op}")
    V, K, _ = build_camera(center, 0, CANONICAL_TOPDOWN_PITCH, distance,
                            FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
    img = render_splat(scene, V, K, CANONICAL_W, CANONICAL_H,
                        bg=(1.0, 1.0, 1.0))
    op = out / "topdown.png"
    Image.fromarray(img).save(op)
    print(f"[save] {op}")


if __name__ == "__main__":
    main()
