#!/usr/bin/env python3
"""Rebuild scene_background.ply using Open3D proximity subtraction.

For each BG splat, computes nearest-neighbor distance to the union of
all object splats. Drops BG splats within --radius meters of any object
splat. Then applies outlier cleanup (drops splats far from centroid or
with extreme scales).

Compared to extract_background.py (exact-xyz cKDTree match at radius
1e-5m), this leaves cleaner background — no halo of object surface
splats — at the cost of slightly biting into walls/floor where objects
touched them.

Usage:
    python rebuild_background_open3d.py <scene_dir> \
        [--source-ply step7_cardinal_aligned.ply] \
        [--radius 0.03] \
        [--outlier-pos-max 15.0] \
        [--outlier-scale-max 0.5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


# Match the stage-preference chain extract_background.py uses
OBJECT_STAGE_CANDIDATES = [
    "8_final.ply",   # v32 final stage (stage_pick → destreak → 8_final)
    "7_final.ply",
    "6_inside_outside.ply",
    "5_subtracted.ply",
    "5_bookshelf_sweep.ply",
    "4_rug.ply",
    "5_sweep_fallback.ply",
    "4b_sam_tight_low.ply",
    "4_sam_tight.ply",
    "3_floor_drop.ply",
    "2_sam_wide.ply",
    "1_visual_hull.ply",
]


# Per-object stage overrides for objects whose viewer .splat does NOT use
# 7_final.ply. Keep in sync with splat-demo/viewer.html object swaps.
OBJECT_PLY_OVERRIDES = {
    "02_wooden_shelving_unit": "5b_bookshelf_sweep_low.ply",
    "02_light_wood_sideboard": "6_inside_outside.ply",
}


def find_object_plys(obj_dir: Path):
    """Return ONLY the final-stage PLY for this object (single PLY,
    in a list for API compat). Unioning coarse hulls (1_visual_hull,
    2_sam_wide) makes subtraction carve floor under + wall behind
    objects, leaving "bomb crater" holes in the BG. Just use 7_final.
    Fall back to next-best stage if 7_final missing.

    Per-object override: OBJECT_PLY_OVERRIDES wins when present."""
    if obj_dir.name in OBJECT_PLY_OVERRIDES:
        p = obj_dir / OBJECT_PLY_OVERRIDES[obj_dir.name]
        if p.exists():
            return [p]
    for n in OBJECT_STAGE_CANDIDATES:  # 7_final first, then 6_inside_outside, etc.
        p = obj_dir / n
        if p.exists():
            return [p]
    return []


def load_xyz(ply_path: Path) -> np.ndarray:
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--source-ply", type=Path, default=None,
                    help="background source PLY (default: step7_cardinal_aligned.ply)")
    ap.add_argument("--radius", type=float, default=0.03,
                    help="proximity threshold in meters (default 0.03 = 3 cm). "
                         "BG splats within this distance of any object splat are dropped.")
    ap.add_argument("--outlier-pos-max", type=float, default=15.0,
                    help="drop splats whose pos is farther than N meters from centroid "
                         "(default 15.0; set 0 to skip)")
    ap.add_argument("--outlier-scale-max", type=float, default=0.5,
                    help="drop splats whose any-axis scale exceeds N meters "
                         "(default 0.5; set 0 to skip). NOTE: requires reading the "
                         "scale fields from the PLY.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PLY (default <scene>/scene_background_o3d.ply)")
    ap.add_argument("--exclude", type=str, default="",
                    help="comma-separated 02_* folder names to NOT subtract "
                         "(they stay baked into the background shell = no holes "
                         "for non-keeper objects)")
    args = ap.parse_args()
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    scene = args.scene_dir.resolve()
    src = args.source_ply or (scene / "step7_cardinal_aligned.ply")
    if not src.exists():
        sys.exit(f"[fatal] source PLY missing: {src}")

    out_ply = args.out or (scene / "scene_background_o3d.ply")

    # ──────────────────────────────────────────────────────────────
    # 1) Load background
    # ──────────────────────────────────────────────────────────────
    print(f"[load] background source: {src}")
    t0 = time.time()
    bg_pl = PlyData.read(str(src))
    bg_v = bg_pl["vertex"]
    bg_xyz = np.stack([bg_v["x"], bg_v["y"], bg_v["z"]], axis=1).astype(np.float64)
    n_bg = len(bg_xyz)
    print(f"  {n_bg:,} splats  ({time.time()-t0:.1f}s)")

    # ──────────────────────────────────────────────────────────────
    # 2) Build the union object cloud from every 02_* stage PLY
    # ──────────────────────────────────────────────────────────────
    obj_dirs = sorted([d for d in scene.iterdir()
                        if d.is_dir() and d.name.startswith("02_")
                        and d.name not in exclude])
    if exclude:
        print(f"\n[exclude] NOT subtracting (left in shell): {sorted(exclude)}")
    print(f"\n[gather] {len(obj_dirs)} object folders")
    obj_clouds = []
    obj_meta = []
    for od in obj_dirs:
        plys = find_object_plys(od)
        if not plys:
            print(f"  [{od.name}] SKIP — no candidate PLY")
            continue
        n_obj = 0
        for ply in plys:
            xyz = load_xyz(ply)
            obj_clouds.append(xyz)
            n_obj += len(xyz)
        obj_meta.append({"obj": od.name, "n_splats_unioned": n_obj,
                         "n_stage_plys": len(plys)})
        print(f"  [{od.name:40s}] {n_obj:>9,} splats from {len(plys)} stages")

    if not obj_clouds:
        sys.exit("[fatal] no object PLYs found")

    obj_xyz = np.concatenate(obj_clouds, axis=0)
    print(f"\n[union] total object splats: {len(obj_xyz):,}")

    # ──────────────────────────────────────────────────────────────
    # 3) Open3D compute_point_cloud_distance: bg → nearest object
    # ──────────────────────────────────────────────────────────────
    print(f"\n[open3d] computing distance bg → nearest object splat ...")
    t0 = time.time()
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(bg_xyz)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_xyz)
    dists = np.asarray(bg_pcd.compute_point_cloud_distance(obj_pcd))
    print(f"  done ({time.time()-t0:.1f}s)")
    print(f"  distance stats: min={dists.min():.4f}  median={np.median(dists):.4f}  "
          f"p90={np.percentile(dists,90):.4f}  max={dists.max():.4f}")

    # Drop BG splats within radius of any object
    near_object = dists < args.radius
    n_near = int(near_object.sum())
    print(f"\n[subtract] BG splats within {args.radius}m of object: "
          f"{n_near:,}/{n_bg:,} ({100*n_near/n_bg:.1f}%) — DROPPED")

    keep = ~near_object

    # ──────────────────────────────────────────────────────────────
    # 4) Outlier cleanup (optional)
    # ──────────────────────────────────────────────────────────────
    if args.outlier_pos_max > 0:
        cx, cy, cz = np.median(bg_xyz, axis=0)
        d = np.linalg.norm(bg_xyz - np.array([cx, cy, cz]), axis=1)
        far = d > args.outlier_pos_max
        n_far = int(far.sum())
        print(f"\n[outlier] |pos - centroid| > {args.outlier_pos_max}m: "
              f"{n_far:,} ({100*n_far/n_bg:.2f}%) — DROPPED")
        keep &= ~far

    if args.outlier_scale_max > 0:
        # Pull scale fields; they're log-scale in gsplat PLYs. The actual
        # linear scale is exp(scale_i). We drop where any axis's linear
        # scale exceeds outlier_scale_max.
        try:
            s = np.exp(np.stack([bg_v["scale_0"], bg_v["scale_1"], bg_v["scale_2"]],
                                  axis=1).astype(np.float32))
            big = (s > args.outlier_scale_max).any(axis=1)
            n_big = int(big.sum())
            print(f"[outlier] max scale > {args.outlier_scale_max}m: "
                  f"{n_big:,} ({100*n_big/n_bg:.2f}%) — DROPPED")
            keep &= ~big
        except (ValueError, KeyError) as e:
            print(f"[outlier] scale fields missing in PLY ({e}) — skipping scale filter")

    # ──────────────────────────────────────────────────────────────
    # 5) Save
    # ──────────────────────────────────────────────────────────────
    n_kept = int(keep.sum())
    n_dropped = n_bg - n_kept
    print(f"\n[save] kept {n_kept:,}/{n_bg:,} ({100*n_kept/n_bg:.1f}%) — "
          f"dropped {n_dropped:,}")

    PlyData([PlyElement.describe(bg_v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] → {out_ply}")

    # Report
    diag = scene / "diagnostics"
    diag.mkdir(exist_ok=True)
    (diag / "scene_background_o3d_report.json").write_text(json.dumps({
        "source_ply": str(src),
        "n_source": n_bg,
        "n_object_splats_union": len(obj_xyz),
        "subtract_radius_m": args.radius,
        "n_dropped_near_object": n_near,
        "outlier_pos_max_m": args.outlier_pos_max,
        "outlier_scale_max_m": args.outlier_scale_max,
        "n_kept": n_kept,
        "n_dropped_total": n_dropped,
        "kept_pct": round(100 * n_kept / n_bg, 2),
        "n_objects_processed": len(obj_meta),
        "per_object": obj_meta,
        "out_ply": str(out_ply),
    }, indent=2))
    print(f"[report] {diag / 'scene_background_o3d_report.json'}")


if __name__ == "__main__":
    main()
