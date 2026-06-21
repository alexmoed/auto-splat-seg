#!/usr/bin/env python3
"""extract_background.py — produce the room background PLY.

Loads the FULL rotated room PLY (step7_cardinal_aligned.ply by default
— cardinal-aligned, unsliced, ceiling and walls intact) and removes
every splat that was extracted as part of an object. The result is
the architectural shell of the room: walls, floor, ceiling, plus any
leftover splats that weren't extracted as objects.

Why the unsliced source: step7_sliced.ply has the top 8% of y dropped
(ceiling region) for downstream object extraction. For the background
we want the full envelope back. cardinal_aligned shares xyz with sliced
(rotation chain doesn't change splat positions), so object PLYs derived
from sliced still match cardinal_aligned by exact-coordinate KDTree
lookup.

Method:
  1. Load source PLY.
  2. Build a KDTree of source xyz.
  3. For each <scene>/02_<obj>/<final_stage>.ply, query the tree at
     each object splat with a tiny radius (default 1e-5 m) — object
     splats are descendants of the source, so they share exact xyz
     positions, and we mark each match for removal.
  4. Write <scene>/scene_background.ply with the surviving splats.
  5. Render canonical 5 views.

Usage:
    python extract_background.py <scene_dir>
    python extract_background.py <scene_dir> --source-ply <path>
    python extract_background.py <scene_dir> --radius 0.01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from sam_carve import render_canonical_5  # noqa: E402

# (Retired 2026-05-29: dead local OBJECT_STAGE_PREFERENCE — the background
# subtract uses the UNION of SUBTRACT_STAGES via find_object_plys, not a
# single most-final pick. NOTE: SUBTRACT_STAGES still omits 8_final/7_final/
# 6_inside_outside — that is a SEPARATE open audit finding, not fixed here.)
SUBTRACT_STAGES = [
    "1_visual_hull", "2_sam_wide", "3_floor_drop",
    "4_sam_tight", "4_rug",
    "5_sweep_fallback", "5_bookshelf_sweep", "5_subtracted",
    "2_pitch_sweep_refined",
]


def find_object_plys(obj_dir: Path) -> list[Path]:
    """Return EVERY stage PLY that exists for this object. The background
    subtract uses the UNION so any splat any stage thought belonged to the
    object is removed from the background — fixes the wall-art leftover
    problem (refined was tiny, wider hull splats stayed in background).
    The puzzle piece in final_outputs/ still uses the most-refined stage."""
    return [obj_dir / f"{s}.ply" for s in SUBTRACT_STAGES
             if (obj_dir / f"{s}.ply").exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--source-ply", type=Path, default=None,
                    help="rotated full-room source PLY. Default: looks "
                         "for step7_cardinal_aligned.ply (unsliced, "
                         "ceiling intact) in scene_dir, then in the "
                         "sibling Kitchen_living_dining/ scene")
    ap.add_argument("--radius", type=float, default=1e-5,
                    help="KDTree match radius in meters (default 1e-5 — "
                         "exact xyz match for splats from the same rotation)")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()

    # 1) Locate source PLY
    src = args.source_ply
    if src is None:
        # Prefer step8_density_filtered.ply (outliers removed, ceiling
        # intact). step7_cardinal_aligned has the same object splats but
        # also far-field outliers (x/z to ±8000m on Kitchen_living_dining)
        # which balloon the canonical_5 render extent so the room appears
        # as a tiny dot. step8 is the cleanest "rotated full scan" source.
        candidates = [
            scene / "step8_density_filtered.ply",
            scene / "step7_cardinal_aligned.ply",
            scene.parent / "Kitchen_living_dining" / "step8_density_filtered.ply",
            scene.parent / "Kitchen_living_dining" / "step7_cardinal_aligned.ply",
        ]
        for c in candidates:
            if c.exists():
                src = c
                break
        if src is None:
            sys.exit(f"[fatal] couldn't find step8_density_filtered.ply or "
                     f"step7_cardinal_aligned.ply in {scene}; pass "
                     f"--source-ply explicitly")

    if not src.exists():
        sys.exit(f"[fatal] source PLY {src} not found")

    print(f"[load] source: {src}")
    src_pl = PlyData.read(str(src))
    src_v = src_pl["vertex"]
    src_xyz = np.stack([src_v["x"], src_v["y"], src_v["z"]],
                        axis=1).astype(np.float64)
    n_total = len(src_xyz)
    print(f"  {n_total:,} splats")
    print(f"  x: [{src_xyz[:,0].min():.2f}, {src_xyz[:,0].max():.2f}]")
    print(f"  y: [{src_xyz[:,1].min():.2f}, {src_xyz[:,1].max():.2f}]")
    print(f"  z: [{src_xyz[:,2].min():.2f}, {src_xyz[:,2].max():.2f}]")

    # 2) Build KDTree of source
    print(f"\n[kdtree] building tree of {n_total:,} source splats...")
    tree = cKDTree(src_xyz)

    # 3) For each object, mark matches
    obj_dirs = sorted([d for d in scene.iterdir()
                        if d.is_dir() and d.name.startswith("02_")])
    print(f"\n[match] {len(obj_dirs)} objects, radius={args.radius}m")

    drop = np.zeros(n_total, dtype=bool)
    per_obj = []
    for od in obj_dirs:
        plys = find_object_plys(od)
        if not plys:
            print(f"  [{od.name}] SKIP — no candidate PLY found")
            continue
        # Union all stage splats for the subtract mask. Each stage's
        # splat positions are descendants of the source via the rotation
        # chain — exact xyz match with the tree.
        stage_hits = 0
        n_obj_total = 0
        stages_used = []
        for ply in plys:
            op = PlyData.read(str(ply))
            oxyz = np.stack([op["vertex"]["x"], op["vertex"]["y"],
                              op["vertex"]["z"]], axis=1).astype(np.float64)
            d, i = tree.query(oxyz, k=1, distance_upper_bound=args.radius)
            hits = d < args.radius
            n_obj_total += len(oxyz)
            stage_hits += int(hits.sum())
            drop[i[hits]] = True
            stages_used.append(ply.name)
        ply = plys[-1]   # report against the most-refined stage
        n_obj = n_obj_total
        n_hit = stage_hits

        per_obj.append({
            "obj": od.name,
            "src_ply": ply.name,
            "stages_unioned": stages_used,
            "n_obj_splats": n_obj,
            "n_matched": n_hit,
            "match_pct": round(100.0 * n_hit / max(n_obj, 1), 1),
        })
        print(f"  [{od.name:35s}] {ply.name:25s} "
              f"{n_obj:>8,} obj → {n_hit:>8,} matched "
              f"({100.0*n_hit/max(n_obj,1):.1f}%)")

    # 4) Save background
    keep = ~drop
    n_kept = int(keep.sum())
    n_dropped = int(drop.sum())
    print(f"\n[save] kept {n_kept:,}/{n_total:,} ({100.0*n_kept/n_total:.1f}%)  "
          f"dropped {n_dropped:,}")

    out_ply = scene / "scene_background.ply"
    PlyData([PlyElement.describe(src_v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # 5) Render canonical views
    renders_dir = scene / "renders" / "scene_background"
    print(f"[render] canonical 5 → {renders_dir}")
    render_canonical_5(out_ply, renders_dir)

    # Report
    diag = scene / "diagnostics"
    diag.mkdir(exist_ok=True)
    (diag / "scene_background_report.json").write_text(json.dumps({
        "source_ply": str(src),
        "n_source": n_total,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "kept_pct": round(100.0 * n_kept / n_total, 2),
        "match_radius_m": args.radius,
        "n_objects_processed": len(per_obj),
        "per_object": per_obj,
    }, indent=2))
    print(f"[report] {diag / 'scene_background_report.json'}")


if __name__ == "__main__":
    main()
