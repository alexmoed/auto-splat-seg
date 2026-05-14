#!/usr/bin/env python3
"""merge_scene.py — concatenate scene_background.ply + every per-object
final PLY into a single reassembled-room PLY and render inside views.

Validates that background + objects reconstruct the original room
visually. If anything is missing in the reassembly, the extraction
or background subtraction had a bug.

Inputs:
  <scene>/scene_background.ply
  <scene>/02_<obj>/<latest_stage>.ply  (one per object)

Outputs:
  <scene>/scene_reassembled.ply
  <scene>/renders/reassembled/{y0,y90,y180,y270,topdown}.png

Usage:
    python merge_scene.py <scene_dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))

OBJECT_STAGE_PREFERENCE = [
    "5_subtracted",
    "5_bookshelf_sweep",
    "4_rug",
    "5_sweep_fallback",
    "4_sam_tight",
    "3_floor_drop",
    "2_pitch_sweep_refined",   # phase 4 wall art
    "1_visual_hull",           # companions
]


def find_object_ply(obj_dir: Path) -> tuple[Path | None, str | None]:
    for stage in OBJECT_STAGE_PREFERENCE:
        p = obj_dir / f"{stage}.ply"
        if p.exists():
            return p, stage
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()

    scene = args.scene_dir.resolve()

    # 1) Load background
    bg_path = scene / "scene_background.ply"
    if not bg_path.exists():
        sys.exit(f"[fatal] {bg_path} not found — run extract_background.py first")

    print(f"[load] background: {bg_path}")
    bg_pl = PlyData.read(str(bg_path))
    bg_v = bg_pl["vertex"]
    n_bg = len(bg_v.data)
    print(f"  {n_bg:,} splats")

    # Reference dtype — all PLYs in the chain SHOULD share this since
    # they all descend from step7_cardinal_aligned.ply.
    ref_dtype = bg_v.data.dtype

    # 2) Collect object PLYs
    obj_dirs = sorted([d for d in scene.iterdir()
                        if d.is_dir() and d.name.startswith("02_")])
    print(f"\n[collect] {len(obj_dirs)} object folders")

    arrays = [bg_v.data]   # start with background
    per_obj = []
    for od in obj_dirs:
        ply, stage = find_object_ply(od)
        if ply is None:
            print(f"  [{od.name}] SKIP — no PLY")
            continue
        op = PlyData.read(str(ply))
        ov = op["vertex"]
        if ov.data.dtype != ref_dtype:
            print(f"  [{od.name}] WARN — dtype mismatch, casting")
            # Pad missing fields with zeros + cast
            cast = np.zeros(len(ov.data), dtype=ref_dtype)
            for name in ref_dtype.names:
                if name in ov.data.dtype.names:
                    cast[name] = ov.data[name]
            arrays.append(cast)
        else:
            arrays.append(ov.data)
        per_obj.append({"obj": od.name, "stage": stage,
                         "n_splats": len(ov.data)})
        print(f"  [{od.name:35s}] {stage:20s} {len(ov.data):>8,} splats")

    # 3) Concatenate
    print(f"\n[merge] concatenating {len(arrays)} arrays...")
    merged = np.concatenate(arrays)
    n_merged = len(merged)
    print(f"  {n_merged:,} total splats")

    # 4) Save
    out_ply = scene / "scene_reassembled.ply"
    PlyData([PlyElement.describe(merged, "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # 5) Render inside views via render_inside_views.py
    print(f"\n[render] inside views of reassembled scene")
    import subprocess
    rc = subprocess.run([sys.executable,
                          str(ITERATION_DIR / "render_inside_views.py"),
                          str(scene),
                          "--target-ply", str(out_ply),
                          "--out-subdir", "renders/reassembled"]).returncode
    if rc != 0:
        print(f"[warn] render_inside_views.py exited {rc}")

    # Report
    sum_obj = sum(o["n_splats"] for o in per_obj)
    diag = scene / "diagnostics"
    diag.mkdir(exist_ok=True)
    (diag / "scene_reassembled_report.json").write_text(json.dumps({
        "background_splats": n_bg,
        "n_objects": len(per_obj),
        "object_total_splats": sum_obj,
        "merged_total": n_merged,
        "expected_if_no_overlap": n_bg + sum_obj,
        "delta": n_merged - (n_bg + sum_obj),
        "per_object": per_obj,
    }, indent=2))
    print(f"\n[report] {diag / 'scene_reassembled_report.json'}")
    print(f"[summary] background {n_bg:,} + objects {sum_obj:,} = {n_merged:,}")


if __name__ == "__main__":
    main()
