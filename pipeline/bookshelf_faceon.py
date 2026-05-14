#!/usr/bin/env python3
"""bookshelf_faceon.py — Stage 5 for bookshelves.

Reads 4_sam_tight.ply (the SAM-cleaned bookshelf) and locks in the
face-on framing. Does NOT re-run SAM. Pure geometric pass that:

  1. Loads 4_sam_tight.ply (cleaned bookshelf).
  2. Computes centroid + extents.
  3. Picks the depth axis = SHORTEST horizontal extent (x or z).
     Bookshelves are wide+tall, narrow front-to-back.
  4. Renders face-on from BOTH directions along the depth axis. Picks
     the one with the largest non-white area (= the front face).
  5. Projects splats into that face-on view → tight 2D bbox.
  6. Crops to bbox + depth cap → 5_faceon_hull.ply.

This stage doesn't recover splats SAM rejected; it just gives a clean
face-on framed result aligned to the bookshelf's principal axis.

Usage:
    python bookshelf_faceon.py <scene_dir> <obj_dir>
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402
from sam_carve import render_canonical_5  # noqa: E402

FOV = 60.0
W, H = 1920, 1080
RENDER_MARGIN = 1.4
BBOX_PAD_PCT = 0.05      # 5% pad on the 2D face-on bbox
DEPTH_PAD_FACTOR = 1.5   # depth cap = (camera-target distance) ± depth_pad
                         # where depth_pad = extent_depth × this factor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir",   type=Path)
    args = ap.parse_args()

    obj = args.obj_dir.resolve()

    # Source: 4_sam_tight.ply — the SAM-cleaned bookshelf. Stage 5 takes
    # this as input, doesn't redo SAM, just locks face-on framing.
    src_ply = obj / "4_sam_tight.ply"
    if not src_ply.exists():
        sys.exit(f"[fatal] no 4_sam_tight.ply in {obj} — run sam_tight first")
    print(f"[bookshelf_faceon] source: {src_ply.name}")

    # Compute centroid + extents from the cleaned bookshelf
    pl_shape = PlyData.read(str(src_ply))
    v_shape = pl_shape["vertex"]
    xyz_shape = np.stack([v_shape["x"], v_shape["y"], v_shape["z"]],
                          axis=1).astype(np.float64)
    if len(xyz_shape) == 0:
        sys.exit("[fatal] source PLY is empty")

    center = np.median(xyz_shape, axis=0)
    p5  = np.percentile(xyz_shape, 5,  axis=0)
    p95 = np.percentile(xyz_shape, 95, axis=0)
    extent_xyz = p95 - p5
    print(f"[bookshelf_faceon] center={center.tolist()}")
    print(f"[bookshelf_faceon] extents (xyz)={extent_xyz.tolist()}")

    # Depth axis = shorter of x or z (vertical y excluded).
    depth_axis = 0 if extent_xyz[0] < extent_xyz[2] else 2
    width_axis = 2 if depth_axis == 0 else 0
    extent_depth  = float(extent_xyz[depth_axis])
    extent_width  = float(extent_xyz[width_axis])
    extent_height = float(extent_xyz[1])
    print(f"[bookshelf_faceon] depth_axis={'xz'[depth_axis//2] if False else ['x','y','z'][depth_axis]}  "
          f"depth={extent_depth:.2f}  width={extent_width:.2f}  height={extent_height:.2f}")

    # Camera distance to fit width AND height
    tan_h = math.tan(math.radians(FOV / 2))
    aspect = W / H
    dist_w = (extent_width  * RENDER_MARGIN) / (2 * tan_h * aspect)
    dist_h = (extent_height * RENDER_MARGIN) / (2 * tan_h)
    distance = max(dist_w, dist_h)

    # Render both directions, pick the one with more non-white pixels.
    diag = obj / "diagnostics" / "5_faceon_hull"
    diag.mkdir(parents=True, exist_ok=True)
    scene = load_gsplat_ply(str(src_ply))

    candidates = []
    for sign, tag in [(+1, "pos"), (-1, "neg")]:
        eye = center.copy()
        eye[depth_axis] += sign * distance
        target = center.copy()
        up = np.array([0.0, -1.0, 0.0])   # y-down image space
        V = viewmat_look_at(eye, target, up)
        K = build_K(FOV, W, H)
        img = render_splat(scene, V.astype(np.float32), K.astype(np.float32),
                           W, H, bg=(1.0, 1.0, 1.0))
        png = diag / f"candidate_{tag}.png"
        Image.fromarray(img).save(png)
        # score = non-white pixel count
        non_white = int(((img < 250).any(axis=-1)).sum())
        candidates.append({"tag": tag, "sign": sign, "score": non_white,
                           "eye": eye.tolist(), "target": target.tolist(),
                           "up": up.tolist(),
                           "V": V, "K": K, "png": str(png)})
        print(f"  candidate {tag}: non-white pixels = {non_white:,}")

    best = max(candidates, key=lambda c: c["score"])
    print(f"[bookshelf_faceon] picked: {best['tag']} (eye={best['eye']})")

    # Project SHAPE splats through best camera → tight 2D bbox of visible body
    V = best["V"]; K = best["K"]
    u_s, v_s, in_front_s = project_to_pixels(xyz_shape, V, K)
    visible_s = (in_front_s & (u_s >= 0) & (u_s < W) &
                 (v_s >= 0) & (v_s < H))
    if int(visible_s.sum()) == 0:
        sys.exit("[fatal] no shape splats projected into face-on view")
    u_v = u_s[visible_s]; v_v = v_s[visible_s]
    x0, y0 = float(u_v.min()), float(v_v.min())
    x1, y1 = float(u_v.max()), float(v_v.max())
    bw, bh = x1 - x0, y1 - y0
    pad = BBOX_PAD_PCT
    bbox = [
        max(0, x0 - bw * pad),
        max(0, y0 - bh * pad),
        min(W, x1 + bw * pad),
        min(H, y1 + bh * pad),
    ]
    print(f"[bookshelf_faceon] face-on bbox (px): {bbox} "
          f"(+{int(pad*100)}% pad)")

    # Crop SOURCE splats by 2D bbox + depth cap. Source already projected
    # above (u_s, v_s, in_front_s) — reuse those.
    in_bbox = ((u_s >= bbox[0]) & (u_s <= bbox[2]) &
               (v_s >= bbox[1]) & (v_s <= bbox[3]))

    homog = np.concatenate([xyz_shape, np.ones((len(xyz_shape), 1))], axis=1)
    cam = homog @ V.T
    cam_z = -cam[:, 2]
    depth_pad = max(extent_depth * DEPTH_PAD_FACTOR, 0.5)
    in_depth = ((cam_z >= distance - depth_pad) &
                (cam_z <= distance + depth_pad))

    keep = in_front_s & in_bbox & in_depth
    n_kept = int(keep.sum())
    n_total = len(xyz_shape)
    print(f"[bookshelf_faceon] source splats {n_total:,}  "
          f"in_front {int(in_front_s.sum()):,}  "
          f"in_bbox {int(in_bbox.sum()):,}  "
          f"in_depth {int(in_depth.sum()):,}  "
          f"kept {n_kept:,} ({100*n_kept/max(1,n_total):.1f}%)")
    if n_kept == 0:
        sys.exit("[fatal] 0 splats survived — bbox/depth/camera mismatch")

    # Save
    out_ply = obj / "5_faceon_hull.ply"
    new_v = v_shape.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # Canonical renders + face-on render alongside the canonical 5
    renders_dir = obj / "renders" / "5_faceon_hull"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")
    # Also keep the face-on view we used (so user can see what's in / out of bbox)
    faceon_png = renders_dir / "faceon.png"
    pl_out = PlyData.read(str(out_ply))
    scene_out = load_gsplat_ply(str(out_ply))
    img = render_splat(scene_out,
                       np.asarray(V, dtype=np.float32),
                       np.asarray(K, dtype=np.float32),
                       W, H, bg=(1.0, 1.0, 1.0))
    Image.fromarray(img).save(faceon_png)
    print(f"[render] face-on → {faceon_png}")

    # Report
    (diag / "report.json").write_text(json.dumps({
        "stage": "5_faceon_hull",
        "src_ply": str(src_ply),
        "depth_axis":   ["x", "y", "z"][depth_axis],
        "extent_xyz":   extent_xyz.tolist(),
        "center":       center.tolist(),
        "distance":     distance,
        "candidates":   [{"tag": c["tag"], "score": c["score"]} for c in candidates],
        "picked":       best["tag"],
        "camera": {"eye": best["eye"], "target": best["target"],
                   "up": best["up"], "fov": FOV, "width": W, "height": H},
        "bbox_px":      bbox,
        "bbox_pad_pct": BBOX_PAD_PCT,
        "depth_pad_m":  float(depth_pad),
        "n_total":      n_total,
        "n_kept":       n_kept,
    }, indent=2))


if __name__ == "__main__":
    main()
