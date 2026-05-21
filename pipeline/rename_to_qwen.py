#!/usr/bin/env python3
"""rename_to_qwen.py — rename every <scene>/02_<slug>/ directory using
the rich descriptive name from info.py's info.json (preferred), with
fallback to sam_carve step 2's coarse refined label when info.json is
absent.

Label source priority:
  1. <obj>/info.json["object_type"]    — rich description from Qwen
     looking at the FINAL extracted images (4_sam_tight / 7_final
     renders). Examples: "wooden media cabinet with flat-screen tv and
     speakers", "beige sectional sofa with green and pink pillows".
  2. <obj>/diagnostics/2_sam_wide/sam_prompt.txt first pipe-union term
     — coarse refined label from sam_carve step 2 looking at the hull.
     Examples: "wooden cabinet", "beige sectional sofa".

Why prefer info.json: it's derived from the FINAL extracted shape, so
it captures what actually ended up in the splat (e.g. the cabinet
ended up with TV + speakers on it, so the rich name reflects that).
The sam_carve step 2 label is derived from the hull before any carve.

Run AT THE END of the pipeline, after info.py has run on every object,
and before extract_final_outputs / merge_scene. Re-run those two after
this script so the final outputs pick up the new names.

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

# Cap slug length so rich descriptions don't blow up filesystem limits.
# 60 chars covers things like "wooden_media_cabinet_with_flat_screen_tv_and_speakers".
MAX_SLUG_LEN = 60


def slugify(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_")
    if len(s) > MAX_SLUG_LEN:
        s = s[:MAX_SLUG_LEN].rstrip("_")
    return s or "object"


def extract_refined_label(obj_dir: Path) -> str | None:
    """Prefer Qwen's free-form name from info.json (independent review of
    the FINAL renders). Fall back to legacy object_type, then to the
    sam_carve step-2 coarse label as a last resort."""
    # Source 1: info.json's `name` (Qwen names the object freely
    # looking at 13 rendered angles of the picked PLY)
    info_p = obj_dir / "info.json"
    if info_p.exists():
        try:
            info = json.load(open(info_p))
            name = (info.get("name") or info.get("object_type") or "").strip()
            if name:
                return name
        except Exception:
            pass
    # Source 2: sam_carve step-2 first pipe-union term (coarse, from hull)
    p = obj_dir / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not p.exists():
        return None
    pipe = p.read_text().strip()
    tagged = parse_tagged_prompts(pipe)
    if not tagged:
        return None
    return tagged[0][0]


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
