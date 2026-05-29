#!/usr/bin/env python3
"""group.py — detect logical parent-child relationships among extracted objects.

Reads each <scene>/02_<slug>/{4_sam_tight.ply or 1_visual_hull.ply},
computes a 3D AABB (p5/p95 percentile, robust to outliers), and assigns:

  A is parent of B  iff
    1. B's xz footprint is inside A's xz footprint (with tolerance), AND
    2. B's y_min is at or just below A's y_max (B sits on A's top, ±10cm).

  NOTE: a third intended rule — "the floor / a rug isn't a parent" (reject a
  candidate parent whose y_max is near the room floor) — is NOT currently
  enforced. FLOOR_TOL is recorded in the output params but no comparison uses
  it; find_room_floor_y is unused. Enabling it is a tracked follow-up.

For each B, of all qualifying A, pick the SMALLEST (closest match — speaker
sits on TV stand, not on the room as a whole).

Outputs:
  Updates each child's info.json with "parent": "02_<parent_slug>".
  Writes <scene>/scene_hierarchy.json with groups[] + orphans[].

Does NOT touch any .ply files. Splats stay in their original folders;
hierarchy is metadata only. Spatial display works as-is because all PLYs
share the world coordinate frame.

Usage:
    python group.py <scene_dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

# Tolerances
XZ_PAD       = 0.15   # 15 cm — child's xz can overhang parent's xz this much.
                      # Bumped from 5cm 2026-05-06 because TVs/speakers commonly
                      # extend past the back of their stand by 10-15cm and the
                      # 5cm tolerance was missing real parent-child relationships.
Y_STACK_TOL  = 0.10   # 10 cm — child y_min within this distance of parent y_max
FLOOR_TOL    = 0.20   # 20 cm — A's y_max within this of room floor → not a parent


def aabb_robust(ply_path: Path):
    """Return percentile-based 3D AABB: dict with x_min/x_max/y_min/y_max/
    z_min/z_max + volume."""
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"]
    if len(v.data) == 0:
        return None
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    p05 = np.percentile(xyz, 5,  axis=0)
    p95 = np.percentile(xyz, 95, axis=0)
    return {
        "x_min": float(p05[0]), "x_max": float(p95[0]),
        "y_min": float(p05[1]), "y_max": float(p95[1]),
        "z_min": float(p05[2]), "z_max": float(p95[2]),
        "volume": float(np.prod(p95 - p05)),
    }


def xz_contains(parent, child, pad: float = XZ_PAD) -> bool:
    return ((parent["x_min"] - pad) <= child["x_min"] and
            (parent["x_max"] + pad) >= child["x_max"] and
            (parent["z_min"] - pad) <= child["z_min"] and
            (parent["z_max"] + pad) >= child["z_max"])


def stacks_on_top(parent, child, tol: float = Y_STACK_TOL) -> bool:
    """child's bottom is within tol of parent's top (y-down → 'top' = y_min,
    'bottom' = y_max).
    BUT this scene uses y-down convention where smaller y = up. Need to figure
    out which way y points.
    Actually let's stay agnostic: for stacking, we want child to be ABOVE
    parent. In y-down, ABOVE = smaller y. In y-up, ABOVE = larger y.
    Caller passes scene's `y_axis_up` flag — but simpler: the side facing
    upward is whichever side of the parent's y range is closer to the
    canonical 'sky'. Without that info, treat both directions and accept
    either one."""
    # y-up: child y_min ≈ parent y_max  (child sits on top)
    if abs(child["y_min"] - parent["y_max"]) <= tol:
        return True
    # y-down: child y_max ≈ parent y_min (child sits on top in y-down)
    if abs(child["y_max"] - parent["y_min"]) <= tol:
        return True
    return False


def detect_parent(child, candidates):
    """candidates = [(slug, aabb), ...] sorted by ascending volume.
    Return the smallest qualifying parent, or None."""
    best = None
    for slug, aabb in candidates:
        if aabb["volume"] <= child["volume"]:
            continue   # parent must be larger
        if not xz_contains(aabb, child):
            continue
        if not stacks_on_top(aabb, child):
            continue
        # First (smallest) match wins because candidates are sorted ascending.
        best = slug
        break
    return best


def find_room_floor_y(objects: dict):
    """Estimate room floor y by looking at the LARGEST object's y range."""
    if not objects:
        return None
    biggest_slug = max(objects, key=lambda s: objects[s]["volume"])
    a = objects[biggest_slug]
    return a["y_max"], a["y_min"]   # both extremes — caller decides


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    obj_dirs = sorted([d for d in scene.iterdir()
                       if d.is_dir() and d.name.startswith("02_")])
    if not obj_dirs:
        sys.exit(f"[fatal] no 02_*/ folders in {scene}")

    # Load AABB per object — prefer 4_sam_tight.ply if present
    objects = {}
    for od in obj_dirs:
        for fname in ("4_sam_tight.ply", "1_visual_hull.ply"):
            p = od / fname
            if p.exists():
                aabb = aabb_robust(p)
                if aabb:
                    objects[od.name] = aabb
                    objects[od.name]["src"] = fname
                break
    if not objects:
        sys.exit("[fatal] no PLYs found in any 02_*/")

    print(f"[load] {len(objects)} objects with AABBs")

    # Sort by volume ascending — small objects considered as candidate
    # children first; when finding parents we iterate the same sorted list
    # so smallest-qualifying-parent is picked.
    sorted_slugs = sorted(objects.keys(), key=lambda s: objects[s]["volume"])
    candidates_for_parent = [(s, objects[s]) for s in sorted_slugs]

    # Detect parents
    children_of = {}     # parent_slug → [child_slug, ...]
    parent_of   = {}     # child_slug  → parent_slug
    orphans     = []
    for child_slug in sorted_slugs:
        child = objects[child_slug]
        # Filter candidates to those LARGER than this child
        bigger = [(s, a) for s, a in candidates_for_parent
                  if a["volume"] > child["volume"] and s != child_slug]
        parent = detect_parent(child, bigger)
        if parent:
            parent_of[child_slug] = parent
            children_of.setdefault(parent, []).append(child_slug)
        else:
            orphans.append(child_slug)

    print(f"[group] {len(parent_of)} parent-child links, {len(orphans)} orphans")
    for parent, kids in children_of.items():
        print(f"  {parent}  ←  {kids}")

    # Update child info.json with parent
    for child_slug, parent_slug in parent_of.items():
        info_path = scene / child_slug / "info.json"
        if info_path.exists():
            try:
                info = json.load(open(info_path))
            except Exception:
                info = {}
            info["parent"] = parent_slug
            info_path.write_text(json.dumps(info, indent=2))

    # Write scene-level hierarchy
    hierarchy = {
        "scene_dir": str(scene),
        "groups": [
            {"parent": p, "children": sorted(kids)}
            for p, kids in sorted(children_of.items())
        ],
        "orphans": sorted(orphans),
        "params":  {"xz_pad": XZ_PAD, "y_stack_tol": Y_STACK_TOL,
                    "floor_tol": FLOOR_TOL},
    }
    out = scene / "scene_hierarchy.json"
    out.write_text(json.dumps(hierarchy, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
