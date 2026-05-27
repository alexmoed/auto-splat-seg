#!/usr/bin/env python3
"""low_ring_preview.py — render a handful of LOW ring camera views from
the current sam_tight result so we can eyeball whether SAM/Qwen would
recognize the chair from these angles. No SAM/vote, no PLY write.

Usage:
    python low_ring_preview.py <obj_dir>
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/pipeline")
from sam_tight import build_camera, FOV, W, H, Y_DOWN  # noqa: E402
from extract_one import RENDER_MARGIN  # noqa: E402
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--in-ply", type=str, default="4_sam_tight.ply")
    ap.add_argument("--pitches", type=str, default="-5,5")
    ap.add_argument("--yaws", type=str, default="0,90,180,270")
    ap.add_argument("--margin-mult", type=float, default=1.5)
    args = ap.parse_args()

    obj = args.obj_dir.resolve()
    in_ply = obj / args.in_ply
    out_dir = obj / "diagnostics" / "low_ring_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    pitches = [float(p) for p in args.pitches.split(",")]
    yaws = [float(y) for y in args.yaws.split(",")]

    print(f"[load] {in_ply}")
    scene = load_gsplat_ply(str(in_ply))
    means = scene["means"].detach().cpu().numpy()
    print(f"[load] {len(means):,} splats")

    center = np.median(means, axis=0).astype(np.float32)
    raw_lo = means.min(axis=0)
    raw_hi = means.max(axis=0)
    extent = float((raw_hi - raw_lo).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN * args.margin_mult) / (2 * tan_half)
    print(f"[frame] center={center.tolist()}  extent={extent:.2f}m  "
          f"dist={distance:.2f}m  margin={RENDER_MARGIN}×{args.margin_mult}")

    for pitch in pitches:
        for yaw in yaws:
            V, K, eye = build_camera(center, yaw, pitch, distance,
                                      FOV, W, H, y_down=Y_DOWN)
            img = render_splat(scene, V, K, W, H)
            tag = f"y{int(round(yaw))}_p{int(round(pitch))}"
            out = out_dir / f"{tag}.png"
            Image.fromarray(img).save(out)
            print(f"[save] {out}")


if __name__ == "__main__":
    main()
