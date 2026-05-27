#!/usr/bin/env python3
"""aniso_filter.py — drop highly-anisotropic gaussian splats from a PLY.

Outlier splats with one scale axis ~0 render as needle/smear artefacts
that survive multi-view voting (because they're 3D-positioned at the
object's surface). This filter drops any splat whose
max_scale / min_scale ratio exceeds `--thresh`.

Usage:
  ./aniso_filter.py <in.ply> <out.ply> --thresh 100
"""
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_ply", type=Path)
    ap.add_argument("out_ply", type=Path)
    ap.add_argument("--thresh", type=float, default=100.0,
                    help="drop splats with max_scale/min_scale > thresh "
                         "(default 100)")
    args = ap.parse_args()

    pl = PlyData.read(str(args.in_ply))
    v = pl["vertex"]
    n = len(v.data)
    s0 = np.exp(np.asarray(v["scale_0"]))
    s1 = np.exp(np.asarray(v["scale_1"]))
    s2 = np.exp(np.asarray(v["scale_2"]))
    scales = np.stack([s0, s1, s2], axis=1)
    sorted_s = np.sort(scales, axis=1)
    maxs = sorted_s[:, 2]
    mins = sorted_s[:, 0]
    aniso = maxs / np.maximum(mins, 1e-9)

    keep = aniso <= args.thresh
    n_kept = int(keep.sum())
    n_dropped = n - n_kept
    print(f"[in]      {n:,} splats")
    print(f"[thresh]  aniso <= {args.thresh}")
    print(f"[dropped] {n_dropped:,} ({100*n_dropped/n:.2f}%)")
    print(f"[kept]    {n_kept:,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(
        str(args.out_ply))
    print(f"[save]    {args.out_ply}")


if __name__ == "__main__":
    main()
