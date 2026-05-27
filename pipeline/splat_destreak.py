#!/usr/bin/env python3
"""splat_destreak.py — drop big+dark Gaussian splat streaks.

Runs after stage_pick produces 7_final.ply. Scans per-splat for the
joint failure mode: max scale axis > 0.10m AND mean SH-deg-0 brightness
< 0.25. Almost always = ellipsoid streaks from a bad optimizer local
minimum (one axis blew up, color collapsed to ~black). Drops them in
place; legitimate dark splats stay small, legitimate big splats stay
bright, so neither trips the AND.

Validated 2026-05-27 on light_wood_bookshelf: 181 splats dropped, the
foreground vertical streaks vanished, no legitimate material lost.

Usage:
    python splat_destreak.py <obj_dir>
        [--in-ply 7_final.ply] [--out-ply 7_final.ply]
        [--min-scale 0.10] [--max-brightness 0.25]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from sam_carve import render_canonical_5  # noqa: E402

DEFAULT_MIN_SCALE_M = 0.10
DEFAULT_MAX_BRIGHTNESS = 0.25


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--in-ply", default="7_final.ply",
                    help="source PLY inside obj_dir (default 7_final.ply)")
    ap.add_argument("--out-ply", default="7_final.ply",
                    help="output PLY inside obj_dir (default overwrites in-ply)")
    ap.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE_M,
                    help=f"drop splats whose largest scale axis exceeds this "
                         f"AND are dark. meters. default {DEFAULT_MIN_SCALE_M}")
    ap.add_argument("--max-brightness", type=float,
                    default=DEFAULT_MAX_BRIGHTNESS,
                    help=f"drop splats below this mean rough-RGB brightness "
                         f"AND big. default {DEFAULT_MAX_BRIGHTNESS}")
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    src = obj / args.in_ply
    dst = obj / args.out_ply

    if not src.exists():
        print(f"[destreak] SKIPPED — no {src.name} in {obj}")
        return

    p = PlyData.read(str(src))
    v = p["vertex"]
    n_in = len(v)

    # Scales stored in log space — exp to get meters.
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                      axis=1).astype(np.float64)
    max_scale = np.exp(scales).max(axis=1)

    # SH degree-0 → rough RGB ≈ 0.5 + sh * 0.28
    rgb_sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
                      axis=1).astype(np.float64)
    brightness = (rgb_sh * 0.28 + 0.5).mean(axis=1)

    drop = (max_scale > args.min_scale) & (brightness < args.max_brightness)
    n_drop = int(drop.sum())
    n_kept = n_in - n_drop
    print(f"[destreak] {n_in:,} → {n_kept:,} "
          f"(dropped {n_drop} big+dark splats, "
          f"thresholds scale>{args.min_scale}m bright<{args.max_brightness})")

    if n_drop == 0:
        print(f"[destreak] no streaks — {src.name} unchanged")
        return

    PlyData([PlyElement.describe(v.data[~drop], "vertex")],
            text=False).write(str(dst))
    print(f"[destreak] wrote {dst.name}")

    # Re-render canonical 5 of the cleaned PLY
    out_renders = obj / "renders" / dst.stem
    render_canonical_5(dst, out_renders)
    print(f"[render] canonical 5 → {out_renders}")

    diag = obj / "diagnostics" / "8_destreak"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "report.json").write_text(json.dumps({
        "stage": "splat_destreak",
        "in_ply": str(src), "out_ply": str(dst),
        "n_in": n_in, "n_kept": n_kept, "n_dropped": n_drop,
        "min_scale_m": args.min_scale,
        "max_brightness": args.max_brightness,
    }, indent=2))


if __name__ == "__main__":
    main()
