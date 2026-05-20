#!/usr/bin/env python3
"""extract_one.py — Visual-hull extract a single inventory item from the
saved topdown camera + bbox.

Reads:
  <scene>/_inventory_temp/qwen_items.json  (camera + items)
  <scene>/step7_sliced.ply                  (source PLY)

Picks one item by --label (first match) or --index. Pads the item's
bbox 15% per side. Projects every splat through the saved camera and
keeps those whose 2D image projection lands inside the padded bbox AND
in front of the camera.

Camera math matches view.py's viewmat_look_at + build_K (OpenCV-style:
camera looks toward -Z, image y points down).

Output:
  <scene>/02_<slug>/1_visual_hull.ply
  <scene>/02_<slug>/1_visual_hull_topdown.png
  <scene>/02_<slug>/1_visual_hull_meta.json

Usage:
    python extract_one.py <scene_dir> --label armchair
    python extract_one.py <scene_dir> --index 16 --pad-pct 0.15
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

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"

# RENDER MARGIN — single source of truth across iteration_1 scripts.
# 1.6 → 2.0 → 3.5 (2026-05-03). Bumped because 2.0 still cut chair legs
# at frame bottom — combined with the percentile-extent bug
# (p5/p95 of a 1.26m chair gave only 0.73m, so distance was 75% of what
# it should have been). Fix: use FULL extent (min/max) AND margin 3.5.
RENDER_MARGIN = 1.55


def viewmat_look_at(eye, target, up):
    """Mirrors view.py exactly. OpenCV-style world→camera (camera looks
    toward -Z, image y points down)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.stack([right, -true_up, -forward], axis=0)
    t = -R @ eye
    V = np.eye(4, dtype=np.float64)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def build_K(fov_deg, width, height):
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([
        [f, 0, width * 0.5],
        [0, f, height * 0.5],
        [0, 0, 1],
    ], dtype=np.float64)


def project_to_pixels(xyz, V, K):
    """Return (u, v, in_front) in image pixel coords."""
    homog = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    cam = homog @ V.T  # (N, 4)
    z = cam[:, 2]
    in_front = z < 0
    safe_z = np.where(z < 0, -z, 1.0)
    u = K[0, 0] * cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe_z + K[1, 2]
    return u, v, in_front


def slugify(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    return s.strip("_") or "item"


def render_topdown_simple(ply: Path, out: Path, fov, w, h):
    """Render a quick topdown of the extracted PLY for verification."""
    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    if len(v.data) == 0:
        return False
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    # FULL extent (min/max) — not p5/p95. Object outliers must fit in frame.
    x_lo, z_lo = xyz[:, 0].min(), xyz[:, 2].min()
    x_hi, z_hi = xyz[:, 0].max(), xyz[:, 2].max()
    cx, cz = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    xe, ze = float(x_hi - x_lo), float(z_hi - z_lo)
    yf = float(xyz[:, 1].max())
    aspect = w / h
    tan_h = math.tan(math.radians(fov / 2))
    dist = max((xe * RENDER_MARGIN) / (2 * tan_h * aspect),
               (ze * RENDER_MARGIN) / (2 * tan_h),
               1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, VIEW_PY, str(ply), str(out),
           f"--eye={cx:.4f},{yf - dist:.4f},{cz:.4f}",
           f"--target={cx + 0.001:.4f},{yf:.4f},{cz:.4f}",
           "--up=0,0,-1", "--y-down", "--fov", str(fov),
           "--width", str(w), "--height", str(h)]
    subprocess.run(cmd, check=True, capture_output=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--label", type=str, default=None,
                    help="match item by label substring")
    ap.add_argument("--index", type=int, default=None,
                    help="0-based index in qwen_items.json items[]")
    ap.add_argument("--pad-pct", type=float, default=0.04,
                    help="fraction per side (default 0.04 = 4%%)")
    ap.add_argument("--source-ply", type=Path, default=None,
                    help="default: <scene>/step7_sliced.ply "
                         "(rotated, ceiling cut)")
    ap.add_argument("--top-extend-m", type=float, default=2.0,
                    help="after back-projection, extend the kept-splats "
                         "AABB upward by this many meters to capture items "
                         "sitting on top (default 2.0)")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    items_path = scene / "_inventory_temp" / "qwen_items.json"
    if not items_path.exists():
        sys.exit(f"[fatal] missing {items_path}\n  run inventory.py first")
    inv = json.load(open(items_path))
    items = inv["items"]
    cam = inv["camera"]
    img_w, img_h = inv["image_size"]
    print(f"[load] {len(items)} items, image {img_w}×{img_h}")

    # Pick item
    chosen = None
    chosen_idx = None
    if args.index is not None:
        if not (0 <= args.index < len(items)):
            sys.exit(f"[fatal] --index {args.index} out of range [0,{len(items)-1}]")
        chosen = items[args.index]
        chosen_idx = args.index
    elif args.label:
        wanted = args.label.strip().lower()
        for i, it in enumerate(items):
            if wanted in it.get("label", "").lower():
                chosen = it
                chosen_idx = i
                break
        if chosen is None:
            sys.exit(f"[fatal] no item with label matching '{args.label}'")
    else:
        sys.exit("[fatal] specify --label or --index")
    print(f"[pick] [{chosen_idx}] label='{chosen.get('label')}' "
          f"category='{chosen.get('category', '?')}' "
          f"conf={chosen.get('confidence', '?')}")
    bbox_px = chosen.get("bbox_pixels")
    if not bbox_px or len(bbox_px) != 4:
        sys.exit(f"[fatal] item has no bbox_pixels: {chosen}")

    # Pad bbox 15% per side
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

    # Camera
    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])

    # Source PLY: rotated + light (3%) ceiling cut for extraction —
    # preserves tall bookshelf / cabinet tops that the 10% ID-cut eats.
    # Falls back to step7_sliced.ply (the 10% one) when the extract PLY
    # isn't present.
    source_ply = args.source_ply
    if source_ply is None:
        extract_ply = scene / "step7_sliced_extract.ply"
        source_ply = extract_ply if extract_ply.exists() else (scene / "step7_sliced.ply")
    if not source_ply.exists():
        sys.exit(f"[fatal] source PLY missing: {source_ply}")
    print(f"[source] {source_ply}")
    pl = PlyData.read(str(source_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[source] {len(xyz):,} splats")

    # Project
    u, v_img, in_front = project_to_pixels(xyz, V, K)
    inside = ((u >= padded[0]) & (u <= padded[2]) &
              (v_img >= padded[1]) & (v_img <= padded[3]))
    keep = in_front & inside
    n_kept = int(keep.sum())
    print(f"[hull] in_front={int(in_front.sum()):,}  "
          f"inside_padded={int(inside.sum()):,}  kept={n_kept:,}")

    if n_kept == 0:
        sys.exit("[fatal] 0 splats kept — bbox or camera mismatch?")

    # Top extension: extend the kept-splats' y_min upward by
    # top_extend_m to catch items sitting on top of the parent. But
    # clip the x-z range of the extension to the ORIGINAL bbox cone's
    # x-z range — NOT the kept-splats' AABB (which inflates wider than
    # the bbox because the cone widens at the floor, capturing floor/
    # wall splats that aren't part of the object). Y convention:
    # y-down (lower y = higher in the world).
    if args.top_extend_m > 0:
        ky = xyz[keep, 1]
        y_hi = float(ky.max())   # bottom of object (largest y, y-down)
        y_lo = float(ky.min())   # top of object (smallest y, y-down)
        y_top_extended = y_lo - args.top_extend_m
        # Extension is x-z clipped to the original bbox cone — same
        # pixel test as the main hull, but only fires for splats in the
        # y-band (y_top_extended, y_lo). Splats below y_lo are already
        # in `keep`. This keeps the extension's x-z width matched to
        # what the bbox actually drew.
        ay_band = (xyz[:, 1] >= y_top_extended) & (xyz[:, 1] < y_lo)
        extend_mask = in_front & inside & ay_band & ~keep
        n_extra = int(extend_mask.sum())
        keep = keep | extend_mask
        n_kept = int(keep.sum())
        print(f"[top-extend] +{args.top_extend_m:.2f}m above object top, "
              f"clipped to bbox cone → +{n_extra:,} extra splats "
              f"(total {n_kept:,})")

    # Save — auto-suffix _2/_3/... if the same slug already exists
    slug = slugify(chosen.get("label", f"item{chosen_idx}"))
    base_slug = slug
    n = 2
    while (scene / f"02_{slug}").exists() and (
            scene / f"02_{slug}" / "1_visual_hull_meta.json").exists():
        # Existing folder for a different item index → suffix
        try:
            existing_meta = json.load(open(scene / f"02_{slug}" / "1_visual_hull_meta.json"))
            if existing_meta.get("source_index_in_inventory") == chosen_idx:
                break  # same item → reuse / overwrite
        except Exception:
            pass
        slug = f"{base_slug}_{n}"
        n += 1
    out_dir = scene / f"02_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "1_visual_hull.ply"
    new_v = pl["vertex"].data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(out_ply))
    print(f"[save] {out_ply}  ({n_kept:,} splats)")

    # Quick-check single topdown render at root (legacy)
    out_png = out_dir / "1_visual_hull_topdown.png"
    render_topdown_simple(out_ply, out_png, fov=70, w=1920, h=1080)
    print(f"[render] {out_png}")

    # Canonical 5-view set in renders/1_visual_hull/ — uses the SAME locked
    # render_canonical_5 from sam_carve so all stages share camera math.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sam_carve import render_canonical_5  # noqa: E402
    renders_dir = out_dir / "renders" / "1_visual_hull"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    meta = {
        "label": chosen.get("label"),
        "category": chosen.get("category"),
        "confidence": chosen.get("confidence"),
        "source_index_in_inventory": chosen_idx,
        "bbox_pixels_tight": bbox_px,
        "bbox_pixels_padded": padded,
        "pad_pct_per_side": args.pad_pct,
        "source_ply": str(source_ply),
        "camera": cam,
        "n_splats_kept": n_kept,
        "n_splats_total": len(xyz),
    }
    (out_dir / "1_visual_hull_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[done] STOP")
    print(f"  PLY:     {out_ply}")
    print(f"  topdown: {out_png}")
    print(f"  meta:    {out_dir / '1_visual_hull_meta.json'}")


if __name__ == "__main__":
    main()
