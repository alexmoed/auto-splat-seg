#!/usr/bin/env python3
"""extract_final_outputs.py — package per-object final outputs +
scene background into a single deliverable folder of .splat files
(32-byte-per-splat binary format used by web viewers).

For each <scene>/02_<obj>/, picks the most-final PLY available
(5_subtracted > 5_bookshelf_sweep > 4_rug > 5_sweep_fallback >
4_sam_tight), converts it to .splat, and writes
<scene>/final_outputs/<obj>.splat. Also converts scene_background.ply
if present.

Pass --keep-ply to also include the source PLY alongside each .splat.

Usage:
    python extract_final_outputs.py <scene_dir>
    python extract_final_outputs.py <scene_dir> --keep-ply
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ITERATION_DIR = Path(__file__).resolve().parent

OBJECT_STAGE_PREFERENCE = [
    "5_subtracted",
    "5_bookshelf_sweep",
    "4_rug",
    "5_sweep_fallback",
    "4_sam_tight",
    "3_floor_drop",
    "2_pitch_sweep_refined",   # phase 4 wall art
    "1_visual_hull",           # companions (TV speaker/remote, shelf items)
]


def find_final_ply(obj_dir: Path) -> tuple[Path | None, str | None]:
    for stage in OBJECT_STAGE_PREFERENCE:
        p = obj_dir / f"{stage}.ply"
        if p.exists():
            return p, stage
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--out-name", default="final_outputs",
                    help="output folder name under scene_dir (default: final_outputs)")
    ap.add_argument("--keep-ply", action="store_true",
                    help="also copy the source PLY alongside the .splat")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    out_dir = scene / args.out_name
    if out_dir.exists():
        print(f"[clean] removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Collect objects
    obj_dirs = sorted([d for d in scene.iterdir()
                        if d.is_dir() and d.name.startswith("02_")])
    print(f"\n[collect] {len(obj_dirs)} object folders → {out_dir}")

    manifest = {"scene_dir": str(scene), "objects": []}
    n_linked = 0
    n_skipped = 0
    for od in obj_dirs:
        ply, stage = find_final_ply(od)
        slug = od.name.removeprefix("02_")
        if ply is None:
            print(f"  [{od.name:35s}] SKIP — no final PLY")
            n_skipped += 1
            manifest["objects"].append({
                "obj": od.name, "stage": None, "ply": None, "skipped": True,
            })
            continue
        # Convert PLY → .splat
        splat_dst = out_dir / f"{slug}.splat"
        rc = subprocess.run([sys.executable,
                              str(ITERATION_DIR / "ply_to_splat.py"),
                              str(ply), str(splat_dst)],
                             capture_output=True).returncode
        if rc != 0:
            print(f"  [{od.name:35s}] FAIL converting {ply.name} → .splat (rc={rc})")
            n_skipped += 1
            manifest["objects"].append({
                "obj": od.name, "stage": stage, "splat": None, "skipped": True,
            })
            continue
        # Optionally also drop the source PLY alongside.
        ply_dst = None
        if args.keep_ply:
            ply_dst = out_dir / f"{slug}.ply"
            try:
                ply_dst.hardlink_to(ply)
            except (PermissionError, OSError):
                shutil.copy2(ply, ply_dst)
        # Pick label from info.json or 1_visual_hull_meta.json
        label = None
        for meta_name in ("info.json", "1_visual_hull_meta.json"):
            mp = od / meta_name
            if mp.exists():
                try:
                    m = json.load(open(mp))
                    label = m.get("object_type") or m.get("label")
                    if label:
                        break
                except Exception:
                    pass
        size_mb = round(splat_dst.stat().st_size / 1e6, 2)
        print(f"  [{od.name:35s}] {stage:20s} → {slug}.splat ({size_mb} MB)")
        manifest["objects"].append({
            "obj": od.name,
            "label": label,
            "stage": stage,
            "splat": f"{slug}.splat",
            "ply": f"{slug}.ply" if ply_dst else None,
            "size_mb": size_mb,
        })
        n_linked += 1

    # Background — convert PLY to .splat
    bg_src = scene / "scene_background.ply"
    if bg_src.exists():
        bg_splat = out_dir / "_background.splat"
        rc = subprocess.run([sys.executable,
                              str(ITERATION_DIR / "ply_to_splat.py"),
                              str(bg_src), str(bg_splat)],
                             capture_output=True).returncode
        if rc == 0:
            bg_size_mb = round(bg_splat.stat().st_size / 1e6, 2)
            print(f"\n[background] scene_background.ply → _background.splat "
                  f"({bg_size_mb} MB)")
            manifest["background"] = {
                "splat": "_background.splat",
                "size_mb": bg_size_mb,
            }
            if args.keep_ply:
                bg_ply = out_dir / "_background.ply"
                try:
                    bg_ply.hardlink_to(bg_src)
                except (PermissionError, OSError):
                    shutil.copy2(bg_src, bg_ply)
                manifest["background"]["ply"] = "_background.ply"
        else:
            print(f"\n[background] FAIL converting scene_background.ply (rc={rc})")
            manifest["background"] = None
    else:
        print(f"\n[background] scene_background.ply NOT FOUND — "
              f"run extract_background.py first")
        manifest["background"] = None

    # Scene hierarchy (parent-child grouping)
    hier = scene / "scene_hierarchy.json"
    if hier.exists():
        dst = out_dir / "_hierarchy.json"
        shutil.copy2(hier, dst)
        manifest["hierarchy"] = "_hierarchy.json"

    # Write manifest
    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest] {manifest_path}")
    print(f"[done] {n_linked} object PLYs linked, {n_skipped} skipped")


if __name__ == "__main__":
    main()
