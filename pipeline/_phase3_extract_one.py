#!/usr/bin/env python3
"""_phase3_extract_one.py — Visual-hull extract one phase-2 item from the
full original scan, using its diorama camera + bbox.

Re-derives the diorama camera (must match _phase2_dioramas.py exactly):
  - room bounds: 15/85 percentile xs,zs of _phase1_temp/scene_minus_phase1.ply
  - eye_y = mean(ys) - 1.0           (1m above centroid in y-down)
  - BACK  = 2.0m
  - FOV   = 70°
  - W,H   = 3840,2160
  - up    = (0,-1,0)
  - per-quadrant eye/target as in _phase2_dioramas.py

Bbox source: <scene>/_phase2_temp/qwen_phase2_items.json.
Splat source: <scene>/step7_sliced.ply (the FULL ORIGINAL SCAN, sliced).

Cone is naturally bounded by a quadrant filter on splat xz so it doesn't
sweep the opposite half of the room. No raycasting needed — phase 1
already removed major furniture, SAM cleanup runs downstream.

Output:
  <scene>/02_<slug>/1_visual_hull.ply
  <scene>/02_<slug>/1_visual_hull_topdown.png
  <scene>/02_<slug>/1_visual_hull_meta.json

Usage:
    python _phase3_extract_one.py <scene> --quadrant SW --label sideboard
    python _phase3_extract_one.py <scene> --quadrant SW --index 0
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import (  # noqa: E402
    viewmat_look_at, build_K, project_to_pixels, slugify, render_topdown_simple)

# Diorama params — MUST MATCH _phase2_dioramas.py
W, H = 3840, 2160
FOV = 70.0
BACK = 2.0


def load_quadrant_cameras(scene: Path):
    """Read cameras.json written by _phase2_dioramas.py — single source of
    truth for the 4 diorama cameras + room bounds. Phase 3 must NOT
    re-derive cameras with duplicated math; that creates silent drift if
    _phase2_dioramas.py changes its logic."""
    cam_path = scene / "_phase2_temp" / "cameras.json"
    if not cam_path.exists():
        sys.exit(f"[fatal] missing {cam_path} — run _phase2_dioramas.py first")
    return json.load(open(cam_path))


def quadrant_xz_mask(xs, zs, q: str, cx: float, cz: float):
    if q == "NE":
        return (xs >= cx) & (zs >= cz)
    if q == "NW":
        return (xs <= cx) & (zs >= cz)
    if q == "SE":
        return (xs >= cx) & (zs <= cz)
    if q == "SW":
        return (xs <= cx) & (zs <= cz)
    sys.exit(f"[fatal] unknown quadrant {q}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--quadrant", choices=["NE", "NW", "SE", "SW"], required=True)
    ap.add_argument("--label", type=str, default=None,
                    help="match item by label substring within quadrant")
    ap.add_argument("--index", type=int, default=None,
                    help="0-based index into by_quadrant[<q>]")
    ap.add_argument("--pad-pct", type=float, default=0.04)
    ap.add_argument("--source-ply", type=Path, default=None,
                    help="default: <scene>/step7_sliced.ply (FULL scan)")
    ap.add_argument("--no-quadrant-filter", action="store_true",
                    help="skip xz quadrant mask (debug)")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    # Load phase-2 items
    items_path = scene / "_phase2_temp" / "qwen_phase2_items.json"
    if not items_path.exists():
        sys.exit(f"[fatal] missing {items_path} — run _phase2_detect.py first")
    pdata = json.load(open(items_path))
    by_q = pdata.get("by_quadrant", {})
    q_items = by_q.get(args.quadrant, [])
    if not q_items:
        sys.exit(f"[fatal] no items in quadrant {args.quadrant}")

    chosen = None
    chosen_idx = None
    if args.index is not None:
        if not (0 <= args.index < len(q_items)):
            sys.exit(f"[fatal] --index out of range [0,{len(q_items)-1}]")
        chosen = q_items[args.index]
        chosen_idx = args.index
    elif args.label:
        wanted = args.label.strip().lower()
        for i, it in enumerate(q_items):
            if wanted in it.get("label", "").lower():
                chosen = it
                chosen_idx = i
                break
        if chosen is None:
            sys.exit(f"[fatal] no {args.quadrant} item matching '{args.label}'")
    else:
        sys.exit("[fatal] specify --label or --index")

    bbox_px = chosen.get("bbox_pixels")
    if not bbox_px or len(bbox_px) != 4:
        sys.exit(f"[fatal] item has no bbox_pixels: {chosen}")

    print(f"[pick] {args.quadrant}[{chosen_idx}] label='{chosen.get('label')}' "
          f"conf={chosen.get('confidence', '?')}")

    # Read locked cameras from cameras.json
    cams = load_quadrant_cameras(scene)
    cam = cams[args.quadrant]
    bounds = cams["bounds"]
    print(f"[cam ] eye={cam['eye']}  tgt={cam['target']}  fov={cam['fov']}")
    print(f"[bnd ] cx={bounds['cx']:.3f}  cz={bounds['cz']:.3f}  "
          f"x[{bounds['xb_min']:.2f},{bounds['xb_max']:.2f}]  "
          f"z[{bounds['zb_min']:.2f},{bounds['zb_max']:.2f}]")

    # Pad bbox
    img_w, img_h = cam["width"], cam["height"]
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    px = bw * args.pad_pct
    py = bh * args.pad_pct
    padded = [
        max(0, int(x0 - px)),
        max(0, int(y0 - py)),
        min(img_w, int(x1 + px)),
        min(img_h, int(y1 + py)),
    ]
    print(f"[bbox] tight  : {bbox_px}")
    print(f"[bbox] padded : {padded}  (+{args.pad_pct*100:.0f}% per side)")

    # Project
    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])

    source_ply = args.source_ply or (scene / "step7_sliced.ply")
    if not source_ply.exists():
        sys.exit(f"[fatal] source PLY missing: {source_ply}")
    print(f"[src ] {source_ply}")
    pl = PlyData.read(str(source_ply))
    vdata = pl["vertex"]
    xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
    print(f"[src ] {len(xyz):,} splats")

    u, v_img, in_front = project_to_pixels(xyz, V, K)
    inside = ((u >= padded[0]) & (u <= padded[2]) &
              (v_img >= padded[1]) & (v_img <= padded[3]))
    if args.no_quadrant_filter:
        qmask = np.ones(len(xyz), dtype=bool)
    else:
        qmask = quadrant_xz_mask(xyz[:, 0], xyz[:, 2],
                                 args.quadrant, bounds["cx"], bounds["cz"])
    keep = in_front & inside & qmask
    n_kept = int(keep.sum())
    print(f"[hull] in_front={int(in_front.sum()):,}  "
          f"in_bbox={int(inside.sum()):,}  in_quad={int(qmask.sum()):,}  "
          f"kept={n_kept:,}")
    if n_kept == 0:
        sys.exit("[fatal] 0 splats kept — bbox or camera mismatch?")

    # Save with auto-suffix on slug collision (multiple items in the same
    # quadrant can share a label, e.g. NE has 3× "wooden bowl with vegetables")
    base_slug = slugify(chosen.get("label", f"{args.quadrant}_item{chosen_idx}"))
    slug = base_slug
    n = 2
    while (scene / f"02_{slug}").exists() and (
            scene / f"02_{slug}" / "1_visual_hull_meta.json").exists():
        try:
            existing = json.load(open(scene / f"02_{slug}" / "1_visual_hull_meta.json"))
            if (existing.get("quadrant") == args.quadrant and
                    existing.get("source_index_in_quadrant") == chosen_idx):
                break  # this folder belongs to *this* item — overwrite
        except Exception:
            pass
        slug = f"{base_slug}_{n}"
        n += 1
    out_dir = scene / f"02_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "1_visual_hull.ply"
    new_v = vdata.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(out_ply))
    print(f"[save] {out_ply}  ({n_kept:,} splats)")

    # Canonical 5-view set in renders/1_visual_hull/ — same locked
    # render_canonical_5 used by all later stages.
    from sam_carve import render_canonical_5  # noqa: E402
    renders_dir = out_dir / "renders" / "1_visual_hull"
    render_canonical_5(out_ply, renders_dir)
    print(f"[rndr] canonical 5 → {renders_dir}")

    # LOCKED PHASE-3 QA RENDER — through the SAME diorama camera that
    # produced the bbox image. Sits alongside the canonical 5 in the same
    # renders folder, named <QUAD>cam.png to make it obvious which camera.
    out_png = renders_dir / f"{args.quadrant}cam.png"
    subprocess.run([
        sys.executable, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py",
        str(out_ply), str(out_png),
        f"--eye={cam['eye'][0]:.4f},{cam['eye'][1]:.4f},{cam['eye'][2]:.4f}",
        f"--target={cam['target'][0]:.4f},{cam['target'][1]:.4f},{cam['target'][2]:.4f}",
        f"--up={cam['up'][0]},{cam['up'][1]},{cam['up'][2]}", "--y-down",
        "--fov", str(cam["fov"]), "--width", "1920", "--height", "1080",
    ], check=True, capture_output=True)
    print(f"[rndr] {out_png}  (via {args.quadrant} diorama camera)")

    meta = {
        "phase": 3,
        "quadrant": args.quadrant,
        "label": chosen.get("label"),
        "confidence": chosen.get("confidence"),
        "source_index_in_quadrant": chosen_idx,
        "bbox_pixels_tight": bbox_px,
        "bbox_pixels_padded": padded,
        "pad_pct_per_side": args.pad_pct,
        "source_ply": str(source_ply),
        "camera": cam,
        "quadrant_filter": not args.no_quadrant_filter,
        "n_splats_kept": n_kept,
        "n_splats_total": len(xyz),
    }
    (out_dir / "1_visual_hull_meta.json").write_text(json.dumps(meta, indent=2))

    # Qwen wall-adjacency check — same as extract_one.py. Builds a wider
    # hull at +10% per side, renders 4 views, asks Qwen one yes/no.
    # Writes wall_adjacent.json so sam_tight + sweep_fallback can gate
    # wall-skip on it.
    from sam_carve import check_wall_adjacent_via_qwen  # noqa: E402
    check_wall_adjacent_via_qwen(scene, out_dir)

    print(f"\n[done] STOP")
    print(f"  PLY:     {out_ply}")
    print(f"  qa-render: {out_png}")


if __name__ == "__main__":
    main()
