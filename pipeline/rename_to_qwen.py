#!/usr/bin/env python3
"""rename_to_qwen.py — rename every <scene>/02_<slug>/ directory to use
the Qwen-refined parent label from sam_carve step 2 (the first pipe-union
term in diagnostics/2_sam_wide/sam_prompt.txt, stripped of {soft}/{hard}
tags and slugified).

Why: inventory.py picks a coarse label ("dark cabinet"), but Qwen often
refines it during sam_carve step 2 ("wooden media console"). The folder
should reflect the refined name so the final .splat outputs (named from
the folder slug) are accurate.

Run AT THE END of the pipeline, after step 2 dispatch is complete and
before extract_final_outputs / merge_scene. Re-run those two after this
script so the final outputs pick up the new names.

Usage:
    python rename_to_qwen.py <scene_dir>

Collision handling: if the target slug already exists, append _2, _3...
"""
import argparse
import json
import re
import sys
from pathlib import Path

ITER = Path(__file__).resolve().parent
sys.path.insert(0, str(ITER))
from sam_carve import parse_tagged_prompts  # noqa: E402


def slugify(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_")
    return s or "object"


def extract_refined_label(obj_dir: Path) -> str | None:
    """Read the first piped term in sam_prompt.txt, strip the tag."""
    p = obj_dir / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not p.exists():
        return None
    pipe = p.read_text().strip()
    tagged = parse_tagged_prompts(pipe)
    if not tagged:
        return None
    return tagged[0][0]  # first prompt's text (already untagged)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print intended renames without doing them.")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    obj_dirs = sorted([d for d in scene.iterdir()
                        if d.is_dir() and d.name.startswith("02_")])
    if not obj_dirs:
        print(f"[fatal] no 02_*/ dirs under {scene}")
        return 1
    print(f"[scan] {len(obj_dirs)} object dirs")

    # Track every slug that will exist after renames so concurrent
    # collisions (e.g. 3 framed prints all wanting the same base slug)
    # each get a unique suffix.
    claimed = set()
    keepers = set()
    for od in obj_dirs:
        # First pass: figure out which dirs are keepers (no rename)
        current_slug = od.name.removeprefix("02_")
        refined_label = extract_refined_label(od)
        if not refined_label:
            keepers.add(current_slug)
            continue
        new_slug = slugify(refined_label)
        if new_slug == current_slug:
            keepers.add(current_slug)
    claimed |= keepers

    renames = []
    for od in obj_dirs:
        current_slug = od.name.removeprefix("02_")
        refined_label = extract_refined_label(od)
        if not refined_label:
            print(f"  [{od.name}] no sam_prompt.txt — keep")
            continue
        base_slug = slugify(refined_label)
        if base_slug == current_slug:
            print(f"  [{od.name}] already matches '{refined_label}' — keep")
            continue
        # Pick a unique slug not already claimed by another dir.
        new_slug = base_slug
        i = 2
        while new_slug in claimed:
            new_slug = f"{base_slug}_{i}"
            i += 1
        claimed.add(new_slug)
        if new_slug == current_slug:
            # Collision suffix bounced us back to our current name.
            print(f"  [{od.name}] (no change after collision) — keep")
            continue
        target = scene / f"02_{new_slug}"
        renames.append((od, target, refined_label))
        print(f"  [{od.name}] → 02_{new_slug}   (label='{refined_label}')")

    if not renames:
        print("[done] no renames needed")
        return 0

    if args.dry_run:
        print(f"\n[dry-run] {len(renames)} renames would happen")
        return 0

    # Two-phase rename via temp names — avoids mid-rename collisions
    # when a chain like potted_plant_4 → potted_plant_3 fires.
    import uuid
    temps = []  # (src, temp_path, final_dst, refined_label)
    for src, dst, label in renames:
        tmp = scene / f"_renaming_{uuid.uuid4().hex[:8]}_{src.name}"
        src.rename(tmp)
        temps.append((src.name, tmp, dst, label))
    for old_name, tmp, dst, label in temps:
        tmp.rename(dst)
        # Update info.json's object_ply path if present
        info_p = dst / "info.json"
        if info_p.exists():
            try:
                info = json.load(open(info_p))
                if "object_ply" in info:
                    info["object_ply"] = info["object_ply"].replace(
                        f"{old_name}/", f"{dst.name}/")
                    info_p.write_text(json.dumps(info, indent=2))
            except Exception as e:
                print(f"  [{dst.name}] info.json patch failed: {e}")

    # Drop any cached hierarchy / manifest so they regenerate
    for stale in ("scene_hierarchy.json", "scene_manifest.json"):
        p = scene / stale
        if p.exists():
            p.unlink()
            print(f"[clean] removed stale {stale}")

    print(f"\n[done] renamed {len(renames)} dirs")
    print("       re-run group.py / extract_final_outputs.py / merge_scene.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
