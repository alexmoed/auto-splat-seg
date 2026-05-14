#!/usr/bin/env python3
"""mask_region_from_background.py — surgically remove a 3D xz-bbox + y-range
region from scene_background.ply. Useful for cleaning up failed-extract
artifacts like the unmoved black refrigerator splats.

Usage:
    python mask_region_from_background.py <scene_dir> \\
        --xmin 1.0 --xmax 2.5 --zmin 2.0 --zmax 3.5 \\
        --ymin -1.0 --ymax 1.9 \\
        --label "black_refrigerator"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--xmin", type=float, required=True)
    ap.add_argument("--xmax", type=float, required=True)
    ap.add_argument("--zmin", type=float, required=True)
    ap.add_argument("--zmax", type=float, required=True)
    ap.add_argument("--ymin", type=float, default=-3.0,
                    help="ymin (y-down: smaller=higher up). Default -3 = no upper bound")
    ap.add_argument("--ymax", type=float, default=3.0,
                    help="ymax (y-down: larger=lower down). Default 3 = no lower bound")
    ap.add_argument("--label", default="manual",
                    help="annotation for the diagnostics report")
    ap.add_argument("--in-ply", default="scene_background.ply",
                    help="input PLY filename (relative to scene_dir)")
    ap.add_argument("--out-ply", default=None,
                    help="output PLY filename. Default: overwrites the input.")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    in_p = scene / args.in_ply
    out_p = scene / (args.out_ply or args.in_ply)

    print(f"[load] {in_p}")
    pl = PlyData.read(str(in_p))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    n = len(xyz)
    print(f"  {n:,} splats")

    in_box = ((xyz[:, 0] >= args.xmin) & (xyz[:, 0] <= args.xmax) &
               (xyz[:, 2] >= args.zmin) & (xyz[:, 2] <= args.zmax) &
               (xyz[:, 1] >= args.ymin) & (xyz[:, 1] <= args.ymax))
    n_dropped = int(in_box.sum())
    keep = ~in_box
    n_kept = int(keep.sum())

    print(f"\n[mask] '{args.label}'")
    print(f"  x∈[{args.xmin}, {args.xmax}]  z∈[{args.zmin}, {args.zmax}]  "
          f"y∈[{args.ymin}, {args.ymax}]")
    print(f"  drop: {n_dropped:,}  keep: {n_kept:,} ({100*n_kept/n:.1f}%)")

    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_p))
    print(f"[save] {out_p}")

    # Append to mask history
    history_path = scene / "diagnostics" / "background_masks.json"
    history_path.parent.mkdir(exist_ok=True)
    history = []
    if history_path.exists():
        try:
            history = json.load(open(history_path))
        except Exception:
            history = []
    history.append({
        "label": args.label,
        "xmin": args.xmin, "xmax": args.xmax,
        "ymin": args.ymin, "ymax": args.ymax,
        "zmin": args.zmin, "zmax": args.zmax,
        "n_dropped": n_dropped,
        "n_kept_after": n_kept,
    })
    history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
