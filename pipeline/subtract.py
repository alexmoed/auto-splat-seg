#!/usr/bin/env python3
"""subtract.py — remove child splats from parent PLYs.

Reads <scene>/scene_hierarchy.json (written by group.py). For each parent
with children, drops splats from the parent's 4_sam_tight.ply that fall
inside any child's tight 3D AABB (with a small safety pad). Saves the
result as 5_subtracted.ply + canonical renders.

Objects with no children get NO 5_subtracted.ply — their 4_sam_tight.ply
remains the final output. The manifest tells you which is the canonical
final per object.

Usage:
    python subtract.py <scene_dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
# Use the SAME canonical render the rest of the pipeline uses — never
# invent new orbit / camera math.
from sam_carve import render_canonical_5  # noqa: E402

# Tighter AABB for child than parent grouping rule used (we want to ONLY
# remove splats that are unambiguously the child, not borderline ones).
CHILD_AABB_LO = 2     # p2
CHILD_AABB_HI = 98    # p98
CHILD_AABB_PAD = 0.02 # 2cm pad — safety margin so we don't cut into parent


def child_tight_aabb(child_ply: Path):
    pl = PlyData.read(str(child_ply))
    v = pl["vertex"]
    if len(v.data) == 0:
        return None
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    lo = np.percentile(xyz, CHILD_AABB_LO,  axis=0) - CHILD_AABB_PAD
    hi = np.percentile(xyz, CHILD_AABB_HI, axis=0) + CHILD_AABB_PAD
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    hier_path = scene / "scene_hierarchy.json"
    if not hier_path.exists():
        sys.exit(f"[fatal] missing {hier_path} — run group.py first")
    hier = json.load(open(hier_path))
    groups = hier.get("groups", [])
    if not groups:
        print("[subtract] no parent-child groups — nothing to do")
        return

    print(f"[subtract] processing {len(groups)} parents")
    for g in groups:
        parent_slug = g["parent"]
        children    = g.get("children", [])
        if not children:
            continue

        parent_dir = scene / parent_slug
        parent_ply = parent_dir / "4_sam_tight.ply"
        if not parent_ply.exists():
            print(f"  [{parent_slug}] SKIP — no 4_sam_tight.ply")
            continue

        # Collect each child's AABB
        child_aabbs = []
        for c_slug in children:
            c_ply = scene / c_slug / "4_sam_tight.ply"
            if not c_ply.exists():
                print(f"  [{parent_slug}] WARN — child {c_slug} has no 4_sam_tight.ply, skipping")
                continue
            aabb = child_tight_aabb(c_ply)
            if aabb is not None:
                child_aabbs.append((c_slug, aabb))
        if not child_aabbs:
            print(f"  [{parent_slug}] no usable child AABBs, skipping")
            continue

        # Mask parent splats: drop those inside ANY child AABB
        pl = PlyData.read(str(parent_ply))
        v  = pl["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        n0 = len(xyz)

        drop = np.zeros(n0, dtype=bool)
        per_child_drops = {}
        for c_slug, (lo, hi) in child_aabbs:
            inside = ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
                      (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
                      (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2]))
            n_hit = int(inside.sum())
            per_child_drops[c_slug] = n_hit
            drop |= inside

        keep = ~drop
        n_kept = int(keep.sum())
        n_dropped = int(drop.sum())

        out_ply = parent_dir / "5_subtracted.ply"
        PlyData([PlyElement.describe(v.data[keep], "vertex")],
                text=False).write(str(out_ply))

        print(f"  [{parent_slug}] {n0:,} → {n_kept:,}  "
              f"(dropped {n_dropped:,})  per-child:{per_child_drops}")

        # Canonical renders via the shared locked function (same as
        # sam_tight, floor_drop, sam_carve all use).
        renders_dir = parent_dir / "renders" / "5_subtracted"
        render_canonical_5(out_ply, renders_dir)
        print(f"    renders: {renders_dir}")

    print(f"\n[subtract] done")


if __name__ == "__main__":
    main()
