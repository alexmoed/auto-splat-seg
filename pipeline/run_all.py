#!/usr/bin/env python3
"""run_all.py — Per-scene orchestrator (under construction).

--step 1: rough visual-hull extraction per item, both topdown + sides.
  a. inventory.py → topdown 4K + qwen_items.json (4-pass disjoint)
  b. extract_one.py per item → 02_<slug>/1_visual_hull.ply (topdown bbox)
  c. subtract phase-1 cones from full scan → _phase1_temp/scene_minus_phase1.ply
  d. _phase2_dioramas.py → 4 axis-aligned across-cut quadrant renders
  e. _phase2_detect.py → Qwen per quadrant (excluded items + min-area + label-blacklist)
  f. _phase3_extract_one.py per phase-2 item → 02_<slug>/1_visual_hull.ply
     (auto-renders through the diorama camera that produced the bbox)

No SAM, no floor drop, no info.json yet — that's later steps.

Usage:
    python run_all.py <scene_dir> --step 1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402


def run_detect_room_type(scene_dir: Path, room_type_override: str | None):
    """Step 1.0 — detect room type if not overridden. Writes
    <scene>/_inventory_temp/room_type.json which inventory.py and
    _phase2_detect.py read."""
    out = scene_dir / "_inventory_temp" / "room_type.json"
    if room_type_override:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "room_type": room_type_override,
            "source": "user_override",
        }, indent=2))
        print(f"\n[step 1.0] room_type override → {room_type_override}")
        return
    if out.exists():
        try:
            existing = json.loads(out.read_text()).get("room_type")
            print(f"\n[step 1.0] room_type already detected: {existing}")
            return
        except Exception:
            pass
    # Need a topdown to detect from. inventory.py renders one if missing,
    # so we may need to render it first if it doesn't exist yet.
    topdown = scene_dir / "_inventory_temp" / "topdown_for_qwen.png"
    if not topdown.exists():
        print(f"\n[step 1.0] no topdown yet; running inventory render-only "
              f"first then detecting")
        # Skip detection; inventory will run with default 'mixed' and
        # then we'd want to re-detect. Cleaner: ask inventory to render
        # the topdown, then detect, then re-run with room-specific
        # categories. For simplicity now, just default to mixed if no
        # topdown — user can pass --room-type to be explicit.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "room_type": "mixed",
            "source": "default_no_topdown",
        }, indent=2))
        print(f"  → defaulting to 'mixed' (pass --room-type to override)")
        return
    print(f"\n[step 1.0] detecting room type from topdown")
    cmd = [sys.executable, str(ITERATION_DIR / "detect_room_type.py"),
            str(scene_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [warn] detect_room_type.py exited {r.returncode}: {r.stderr}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "room_type": "mixed",
            "source": "fallback_after_failure",
            "error": r.stderr,
        }, indent=2))
        return
    rt = r.stdout.strip()
    print(f"  detected: {rt}")


def run_inventory(scene_dir: Path):
    out = scene_dir / "_inventory_temp" / "qwen_items.json"
    if out.exists():
        print(f"\n[step 1.a] inventory.py — SKIP (exists: {out.name})")
        return
    print(f"\n[step 1.a] inventory.py on {scene_dir}")
    cmd = [sys.executable, str(ITERATION_DIR / "inventory.py"), str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fatal] inventory.py exited {r.returncode}")
    if not out.exists():
        sys.exit(f"[fatal] inventory.py finished but did not write {out}")


def _slug_for_item(label: str, idx: int, scene: Path):
    """Replicate extract_one.py's slug-with-suffix logic for skip-check."""
    import re
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (label or "").strip().lower()).strip("_") or "item"
    base = s
    n = 2
    while (scene / f"02_{s}").exists() and (scene / f"02_{s}" / "1_visual_hull_meta.json").exists():
        try:
            m = json.load(open(scene / f"02_{s}" / "1_visual_hull_meta.json"))
            if m.get("source_index_in_inventory") == idx:
                return s
        except Exception:
            pass
        s = f"{base}_{n}"
        n += 1
    return s


def run_extract_one_per_item(scene_dir: Path):
    items_path = scene_dir / "_inventory_temp" / "qwen_items.json"
    if not items_path.exists():
        sys.exit(f"[fatal] missing {items_path}")
    inv = json.load(open(items_path))
    items = inv.get("items", [])
    if not items:
        sys.exit("[fatal] no items in qwen_items.json")

    print(f"\n[step 1.b] extract_one.py per item ({len(items)} items)")
    failures = []
    skipped = 0
    for i, it in enumerate(items):
        label = it.get("label", "?")
        slug = _slug_for_item(label, i, scene_dir)
        out_ply = scene_dir / f"02_{slug}" / "1_visual_hull.ply"
        if out_ply.exists():
            try:
                m = json.load(open(scene_dir / f"02_{slug}" / "1_visual_hull_meta.json"))
                if m.get("source_index_in_inventory") == i:
                    skipped += 1
                    continue
            except Exception:
                pass
        print(f"\n--- item {i}: '{label}' ---")
        cmd = [sys.executable, str(ITERATION_DIR / "extract_one.py"),
               str(scene_dir), "--index", str(i)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[warn] extract_one.py exited {r.returncode} on item {i} ('{label}')")
            failures.append((i, label, r.returncode))

    ran = len(items) - len(failures) - skipped
    print(f"\n[step 1.b] DONE — ran {ran}, skipped {skipped}, failed {len(failures)} / {len(items)}")
    if failures:
        for i, lbl, rc in failures:
            print(f"  - item {i} ('{lbl}'): exit {rc}")


def subtract_phase1_from_scene(scene_dir: Path):
    """For each inventory item, project the TIGHT (unpadded) bbox cone
    through the saved topdown camera into step7_cardinal_aligned.ply and
    remove those splats. Output a single PLY = scene minus all phase-1
    items, with ceiling intact. Dioramas read this file, so we use the
    UNSLICED source so the side-view dioramas show the full vertical
    silhouette of any tall objects (bookshelves etc.) — slicing dropped
    the top 8% which can clip those.
    """
    out_ply = scene_dir / "_phase1_temp" / "scene_minus_phase1.ply"
    if out_ply.exists():
        print(f"\n[step 1.c] subtract phase-1 — SKIP (exists: {out_ply.name})")
        return
    items_path = scene_dir / "_inventory_temp" / "qwen_items.json"
    inv = json.load(open(items_path))
    items = inv["items"]
    cam = inv["camera"]
    img_w, img_h = inv["image_size"]

    src_ply = scene_dir / "step7_cardinal_aligned.ply"
    print(f"\n[step 1.c] subtract phase-1 items (tight bbox, no padding)")
    print(f"  source: {src_ply}  (ceiling intact)")

    pl = PlyData.read(str(src_ply))
    v = pl["vertex"]
    n_total = len(v.data)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])
    u, vp, in_front = project_to_pixels(xyz, V, K)

    subtract_mask = np.zeros(n_total, dtype=bool)
    for i, it in enumerate(items):
        bbox = it.get("bbox_pixels")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        in_box = (u >= x0) & (u <= x1) & (vp >= y0) & (vp <= y1)
        hit = in_front & in_box
        n_hit = int(hit.sum())
        subtract_mask |= hit
        print(f"  item {i:2d} '{it.get('label')}'  bbox={bbox}  splats={n_hit:,}")


    keep = ~subtract_mask
    n_kept = int(keep.sum())
    n_removed = int(subtract_mask.sum())
    print(f"\n  removed: {n_removed:,} / {n_total:,}  kept: {n_kept:,}")

    out_dir = scene_dir / "_phase1_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "scene_minus_phase1.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"  saved: {out_ply}")

    # Topdown render at 1920x1080 using inventory's camera (if available)
    inv_path = scene_dir / "_inventory_temp" / "qwen_items.json"
    if inv_path.exists():
        out_png = out_dir / "scene_minus_phase1_topdown.png"
        try:
            inv = json.load(open(inv_path))
            cam = inv["camera"]
            eye = cam["eye"]; target = cam["target"]; up = cam["up"]
            cmd = [sys.executable,
                   "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py",
                   str(out_ply), str(out_png),
                   f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
                   f"--target={target[0]:.4f},{target[1]:.4f},{target[2]:.4f}",
                   f"--up={up[0]},{up[1]},{up[2]}", "--y-down",
                   "--fov", str(cam["fov"]), "--width", "1920", "--height", "1080"]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  topdown: {out_png}")
        except Exception as e:
            print(f"  [warn] topdown render skipped: {e}")


def render_phase2_dioramas(scene_dir: Path):
    out = scene_dir / "_phase2_temp" / "cameras.json"
    if out.exists():
        print(f"\n[step 1.d] phase 2 dioramas — SKIP (exists: {out.name})")
        return
    print(f"\n[step 1.d] phase 2 diorama renders (4 quadrants)")
    cmd = [sys.executable, str(ITERATION_DIR / "_phase2_dioramas.py"),
           str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fatal] _phase2_dioramas.py exited {r.returncode}")
    if not out.exists():
        sys.exit(f"[fatal] _phase2_dioramas.py finished but did not write {out}")


def run_phase2_detect(scene_dir: Path):
    out = scene_dir / "_phase2_temp" / "qwen_phase2_items.json"
    if out.exists():
        print(f"\n[step 1.e] phase 2 detect — SKIP (exists: {out.name})")
        return
    print(f"\n[step 1.e] phase 2 Qwen detect on quadrants")
    cmd = [sys.executable, str(ITERATION_DIR / "_phase2_detect.py"),
           str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fatal] _phase2_detect.py exited {r.returncode}")
    if not out.exists():
        sys.exit(f"[fatal] _phase2_detect.py finished but did not write {out}")


def run_phase4_art_detect(scene_dir: Path):
    """Phase 4a: Qwen scans the 4 quadrant dioramas for wall art
    (paintings, framed prints, posters). Writes
    _phase4_temp/qwen_art_items.json."""
    out = scene_dir / "_phase4_temp" / "qwen_art_items.json"
    if out.exists():
        print(f"\n[step 1.e2] phase 4 art detect — SKIP (exists: {out.name})")
        return
    print(f"\n[step 1.e2] phase 4 Qwen art detect on quadrants")
    cmd = [sys.executable, str(ITERATION_DIR / "_phase4_art_detect.py"),
            str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[warn] _phase4_art_detect.py exited {r.returncode}")
        return
    if not out.exists():
        print(f"[warn] _phase4_art_detect.py finished but did not write {out}")


def run_phase4_art_extract_per_item(scene_dir: Path):
    """Phase 4b: for each art piece, face-on render perpendicular to its
    wall + Qwen bbox + visual hull from raw rotated PLY."""
    items_path = scene_dir / "_phase4_temp" / "qwen_art_items.json"
    if not items_path.exists():
        print(f"\n[step 1.f2] phase 4 art extract — SKIP (no items detected)")
        return
    pdata = json.load(open(items_path))
    by_q = pdata.get("by_quadrant", {})
    total = sum(len(v) for v in by_q.values())
    if total == 0:
        print(f"\n[step 1.f2] phase 4 art extract — SKIP (0 items)")
        return
    # Index existing phase-4 hulls
    existing_phase4 = {}
    for od in scene_dir.iterdir():
        if not (od.is_dir() and od.name.startswith("02_")):
            continue
        mp = od / "1_visual_hull_meta.json"
        if not mp.exists():
            continue
        try:
            m = json.load(open(mp))
            if m.get("phase") == 4:
                existing_phase4[(m.get("quadrant"), m.get("source_index_in_quadrant"))] = od
        except Exception:
            pass

    print(f"\n[step 1.f2] phase 4 art extract ({total} pieces, "
          f"{len(existing_phase4)} already extracted)")
    failures = []
    skipped = 0
    for q in ["NE", "NW", "SE", "SW"]:
        q_items = by_q.get(q, [])
        for i, it in enumerate(q_items):
            label = it.get("label", "?")
            if (q, i) in existing_phase4:
                skipped += 1
                continue
            print(f"\n--- art {q}[{i}]: '{label}' ---")
            cmd = [sys.executable, str(ITERATION_DIR / "_phase4_art_extract.py"),
                    str(scene_dir), "--quadrant", q, "--index", str(i)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[warn] _phase4_art_extract.py exited {r.returncode} on art {q}[{i}]")
                failures.append((q, i, label, r.returncode))
    ran = total - len(failures) - skipped
    print(f"\n[step 1.f2] DONE — ran {ran}, skipped {skipped}, failed {len(failures)} / {total}")


def run_phase3_extract_per_item(scene_dir: Path):
    """Per-item visual-hull extract from full scan via diorama camera+bbox.
    Each call auto-renders QA through the SAME diorama camera (LOCKED in
    _phase3_extract_one.py). Skips if 02_<slug>/1_visual_hull.ply exists
    AND its meta matches the requested quadrant + index."""
    import re
    items_path = scene_dir / "_phase2_temp" / "qwen_phase2_items.json"
    if not items_path.exists():
        sys.exit(f"[fatal] missing {items_path}")
    pdata = json.load(open(items_path))
    by_q = pdata.get("by_quadrant", {})
    total = sum(len(v) for v in by_q.values())
    # Build index of existing phase-3 extractions: (quadrant, idx) → folder
    existing_phase3 = {}
    for od in scene_dir.iterdir():
        if not (od.is_dir() and od.name.startswith("02_")):
            continue
        mp = od / "1_visual_hull_meta.json"
        if not mp.exists():
            continue
        try:
            m = json.load(open(mp))
            if m.get("phase") == 3:
                existing_phase3[(m.get("quadrant"), m.get("source_index_in_quadrant"))] = od
        except Exception:
            pass

    print(f"\n[step 1.f] phase 3 extract per item ({total} items, {len(existing_phase3)} already extracted)")
    failures = []
    skipped = 0
    for q in ["NE", "NW", "SE", "SW"]:
        q_items = by_q.get(q, [])
        for i, it in enumerate(q_items):
            label = it.get("label", "?")
            if (q, i) in existing_phase3:
                skipped += 1
                continue
            print(f"\n--- {q}[{i}]: '{label}' ---")
            cmd = [sys.executable, str(ITERATION_DIR / "_phase3_extract_one.py"),
                   str(scene_dir), "--quadrant", q, "--index", str(i)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[warn] _phase3_extract_one.py exited {r.returncode} on {q}[{i}] ('{label}')")
                failures.append((q, i, label, r.returncode))
    ran = total - len(failures) - skipped
    print(f"\n[step 1.f] DONE — ran {ran}, skipped {skipped}, failed {len(failures)} / {total}")
    if failures:
        for q, i, lbl, rc in failures:
            print(f"  - {q}[{i}] ('{lbl}'): exit {rc}")


def run_dispatch_per_object(scene_dir: Path):
    """For each 02_<slug>/ with 1_visual_hull.ply, hand off to
    procedure_dispatch.py — it picks the right procedure (general / tv /
    bookshelf / skip) based on the object's label and runs the chain."""
    obj_dirs = sorted([d for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("02_")
                       and (d / "1_visual_hull.ply").exists()])
    print(f"\n[step 1.h] procedure_dispatch × {len(obj_dirs)} objects")
    summary = {}
    for od in obj_dirs:
        cmd = [sys.executable, str(ITERATION_DIR / "procedure_dispatch.py"),
               str(scene_dir), str(od)]
        r = subprocess.run(cmd)
        summary[od.name] = "ok" if r.returncode == 0 else f"fail({r.returncode})"
    ok = sum(1 for v in summary.values() if v == "ok")
    print(f"\n[step 1.h] DONE — {ok}/{len(summary)} objects through dispatch")


def write_manifest(scene_dir: Path):
    """Single source of truth for what's in this scene at the end of step 1.
    Always re-runs (it's a summary, cheap to regenerate)."""
    manifest = {"scene_dir": str(scene_dir), "objects": []}
    obj_dirs = sorted([d for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("02_")])
    for od in obj_dirs:
        meta_path = od / "1_visual_hull_meta.json"
        hull_path = od / "1_visual_hull.ply"
        entry = {
            "slug":    od.name,
            "obj_dir": str(od.relative_to(scene_dir)),
            "phase":   None,
            "label":   None,
            "source":  None,
            "splats":  None,
            "qa_render": None,
            "status":  "missing_hull",
        }
        if meta_path.exists():
            try:
                m = json.load(open(meta_path))
                entry["label"] = m.get("label")
                if m.get("phase") == 3:
                    entry["phase"]  = 3
                    entry["source"] = f"phase2_{m.get('quadrant')}"
                    entry["qa_render"] = f"1_visual_hull_{m.get('quadrant')}cam.png"
                else:
                    entry["phase"]  = 1
                    entry["source"] = "phase1_topdown"
                    entry["qa_render"] = "1_visual_hull_topdown.png"
                entry["splats"] = m.get("n_splats_kept")
            except Exception as ex:
                entry["status"] = f"meta_read_error: {ex}"
        if hull_path.exists():
            entry["status"] = "hull_extracted"
        manifest["objects"].append(entry)

    out = scene_dir / "scene_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    n = len(manifest["objects"])
    by_phase = {1: 0, 3: 0, None: 0}
    for o in manifest["objects"]:
        by_phase[o["phase"]] = by_phase.get(o["phase"], 0) + 1
    print(f"\n[step 1.g] manifest written: {out}")
    print(f"  total objects: {n}  phase1={by_phase.get(1,0)}  phase3={by_phase.get(3,0)}")


def step_1(scene_dir: Path, room_type_override: str | None = None):
    """Empty scene → all rough bbox extractions (visual hulls).

    Bookshelves are detected by phase 2 detect on the existing
    quadrant dioramas (which already see across-wall cuts of the
    room). No separate wall-camera pass needed.

    Room-type-aware: detect_room_type runs first (or honor --room-type
    override). Inventory + phase 2 detect read room_type.json and load
    the matching category lists / prompt extras from room_config.py.
    """
    run_detect_room_type(scene_dir, room_type_override)
    run_inventory(scene_dir)
    # If we couldn't detect because the topdown didn't exist yet, try
    # again now that inventory has rendered it. This happens on first run
    # of a brand-new scene.
    if not room_type_override:
        rt_file = scene_dir / "_inventory_temp" / "room_type.json"
        if rt_file.exists():
            try:
                d = json.loads(rt_file.read_text())
                if d.get("source") == "default_no_topdown":
                    run_detect_room_type(scene_dir, None)
                    # Re-run inventory with the now-correct room type
                    items_file = scene_dir / "_inventory_temp" / "qwen_items.json"
                    if items_file.exists():
                        items_file.unlink()
                    run_inventory(scene_dir)
            except Exception:
                pass
    run_extract_one_per_item(scene_dir)
    subtract_phase1_from_scene(scene_dir)
    render_phase2_dioramas(scene_dir)
    run_phase2_detect(scene_dir)
    run_phase3_extract_per_item(scene_dir)
    run_phase4_art_detect(scene_dir)
    run_phase4_art_extract_per_item(scene_dir)
    write_manifest(scene_dir)


def run_group_and_subtract(scene_dir: Path):
    """Detect parent-child relationships among extracted objects (3D
    AABB containment + on-top stacking) and subtract child splats from
    parent PLYs so the TV / speakers / accessories aren't double-counted
    inside the cabinet they sit on. Writes:
      - <scene>/scene_hierarchy.json
      - <scene>/02_<parent>/5_subtracted.ply (only for parents that have
        children — leaf objects keep 4_sam_tight.ply as their final).
    Idempotent: rewrites scene_hierarchy.json each run; only writes
    5_subtracted.ply for parents that have children."""
    print(f"\n[group] group.py")
    cmd = [sys.executable, str(ITERATION_DIR / "group.py"), str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fatal] group.py exited {r.returncode}")
    print(f"\n[subtract] subtract.py")
    cmd = [sys.executable, str(ITERATION_DIR / "subtract.py"), str(scene_dir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[fatal] subtract.py exited {r.returncode}")


def run_inside_outside_per_object(scene_dir: Path):
    """Stage 6 — final inside/outside refinement per object.

    For each 02_<slug>/, inside_outside.py reads the object's final PLY
    and its 4_sam_tight masks, computes per-splat insideness, and drops
    the clearly-outside splats (Qwen-picked threshold via --auto, with
    legs/feet/supports as the absolute priority). Writes
    6_inside_outside.ply, which extract_final_outputs prefers.

    Bookshelves / open shelving are exempt — handled inside
    inside_outside.py (passed through unchanged)."""
    obj_dirs = sorted([d for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("02_")])
    print(f"\n[step 2.c] inside_outside × {len(obj_dirs)} objects")
    for od in obj_dirs:
        cmd = [sys.executable, str(ITERATION_DIR / "inside_outside.py"),
               str(od), "--auto"]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[warn] inside_outside.py exited {r.returncode} on {od.name}")


def step_2(scene_dir: Path):
    """Per-object extraction via procedure dispatch (general / tv / bookshelf
    / rug / skip). Each object gets its 1_visual_hull.ply refined into
    4_sam_tight.ply + info.json — or marked skipped/not_implemented.

    After the per-object loop finishes, automatically detects parent-child
    groups (TV+speakers on a stand → stand is parent) and subtracts child
    splats from parent PLYs (5_subtracted.ply per parent). Finally a
    Stage-6 inside/outside refinement trims clearly-outside splats."""
    run_dispatch_per_object(scene_dir)
    run_group_and_subtract(scene_dir)
    run_inside_outside_per_object(scene_dir)
    write_manifest(scene_dir)


def step_3(scene_dir: Path):
    """Standalone group + subtract + inside/outside — same work as the
    tail of step 2. Kept for re-running after manual fixes to per-object
    PLYs without redoing the full dispatch loop."""
    run_group_and_subtract(scene_dir)
    run_inside_outside_per_object(scene_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--step", choices=["1", "2", "3"], required=True,
                    help="which orchestrator step to run")
    ap.add_argument("--room-type", default=None,
                    help="Override auto-detect: living_room | dining_room | "
                          "kitchen | bedroom | office | bathroom | hallway | "
                          "mixed. If omitted, detect_room_type.py runs on "
                          "the topdown (or 4 dioramas if available) to pick.")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    if args.step == "1":
        step_1(scene, args.room_type)
    elif args.step == "2":
        step_2(scene)
    elif args.step == "3":
        step_3(scene)
    else:
        sys.exit(f"[fatal] step '{args.step}' not implemented yet")


if __name__ == "__main__":
    main()
