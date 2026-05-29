#!/usr/bin/env python3
"""tv_carve.py — Pitch-sweep SAM carve for thin flat items (TVs / monitors /
pictures sitting on furniture).

Differs from sam_carve.py: instead of orbiting yaws around the object,
we keep the SOURCE diorama camera's yaw fixed (the direction from which
the TV was originally seen) and SWEEP THE PITCH over [0, -15, -30, -45]
in y-down raw. Yaw orbit fails for thin flat items because the object
disappears at side angles — pitch sweep keeps the screen face visible
across all 4 views and uses majority vote at vote-frac 0.5.

Reads:
  <obj>/1_visual_hull.ply
  <obj>/1_visual_hull_meta.json    (uses meta.camera.eye/target for forward yaw)

Writes:
  <obj>/diagnostics/2_sam_wide/input_p<P>.png  (4 pitch views, 1080p)
  <obj>/diagnostics/2_sam_wide/sam_prompt.txt
  <obj>/diagnostics/2_sam_wide/mask_p<P>.png
  <obj>/diagnostics/2_sam_wide/cameras.json
  <obj>/4_sam_tight.ply                         (final, vote ≥50% across pitches)
  <obj>/renders/4_sam_tight/{y0,y90,y180,y270,topdown}.png

No floor_drop (TVs don't sit on the floor). info.py runs after this in
procedure_dispatch.

Usage:
    python tv_carve.py <scene_dir> <obj_dir>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402
import sys as _sys
_sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402
from sam_carve import sam_segment, morph_clean, render_canonical_5  # noqa: E402

PITCHES_DEG = [0, -15, -30, -45]   # straight-on + 3 elevated angles looking down.
                                    # -60 was tried but SAM stops recognizing
                                    # a thin flat-screen TV from that high
                                    # (only the top edge is visible).
W, H        = 1920, 1080
FOV         = 60.0
RENDER_MARGIN = 1.55
SAM_THRESHOLD = 0.4
VOTE_FRAC     = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir",   type=Path)
    args = ap.parse_args()

    obj = args.obj_dir.resolve()
    hull_ply  = obj / "1_visual_hull.ply"
    meta_path = obj / "1_visual_hull_meta.json"
    if not hull_ply.exists() or not meta_path.exists():
        sys.exit(f"[fatal] missing 1_visual_hull.* in {obj}")
    meta  = json.load(open(meta_path))
    label = meta.get("label", "object")
    print(f"[tv_carve] obj={obj.name}  label='{label}'")

    src_cam = meta.get("camera")
    if not src_cam:
        sys.exit("[fatal] meta has no source camera — cannot derive front yaw")
    src_eye = np.array(src_cam["eye"], dtype=np.float64)
    src_tgt = np.array(src_cam["target"], dtype=np.float64)

    # Forward yaw from the source diorama camera (ignore y component — we sweep
    # pitch separately).
    forward_xz = src_tgt - src_eye
    forward_xz[1] = 0.0
    nrm = np.linalg.norm(forward_xz)
    if nrm < 1e-6:
        sys.exit("[fatal] source camera has zero xz forward — bad meta")
    forward_xz /= nrm
    print(f"[tv_carve] forward_xz={forward_xz.tolist()}")

    # Object centroid + extent (median + p5/p95, same convention as sam_carve)
    pl = PlyData.read(str(hull_ply))
    v  = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    center = np.median(xyz, axis=0)
    p5  = np.percentile(xyz, 5,  axis=0)
    p95 = np.percentile(xyz, 95, axis=0)
    extent = float((p95 - p5).max())
    tan_h = math.tan(math.radians(FOV / 2))
    distance = (extent * RENDER_MARGIN) / (2 * tan_h)
    print(f"[tv_carve] center={center.tolist()} extent={extent:.2f} dist={distance:.2f}")

    # Diagnostics dir
    diag = obj / "diagnostics" / "2_sam_wide"
    diag.mkdir(parents=True, exist_ok=True)
    # Clean stale renders
    for f in list(diag.glob("input_*.png")) + list(diag.glob("mask_*.png")):
        f.unlink()

    # Build cameras — same yaw direction (forward_xz), varying pitch.
    # World is y-DOWN for this pipeline (floor_plane.json: small y = top,
    # large y = floor). A negative pitch RAISES eye.y above center.y — and
    # since larger y is toward the floor, that places the camera BELOW the
    # object looking UP at it. (The carve works regardless; this note just
    # keeps the geometry honest for anyone tuning PITCHES_DEG or the sign.)
    # Formula: eye.y = center.y - sin(pitch_rad) * distance
    #   pitch =   0°  → sin=0       → eye.y = center.y                (level)
    #   pitch = -15°  → sin=-0.26   → eye.y = center.y + 0.26*dist    (slightly below, looking up)
    #   pitch = -60°  → sin=-0.87   → eye.y = center.y + 0.87*dist    (well below, looking up)
    scene_g = load_gsplat_ply(str(hull_ply))
    cameras = []
    for pitch_deg in PITCHES_DEG:
        pitch_rad = math.radians(pitch_deg)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        eye_xz_offset = -forward_xz * distance * cos_p
        eye_y_offset  = -sin_p * distance   # negative sign — see comment above
        eye = np.array([
            center[0] + eye_xz_offset[0],
            center[1] + eye_y_offset,
            center[2] + eye_xz_offset[2],
        ], dtype=np.float64)
        target = center.copy()
        up = np.array([0.0, -1.0, 0.0])   # y-down
        V = viewmat_look_at(eye, target, up)
        K = build_K(FOV, W, H)
        # Render
        img = render_splat(scene_g, V.astype(np.float32), K.astype(np.float32),
                           W, H, bg=(1.0, 1.0, 1.0))
        ptag = f"p{int(pitch_deg)}"
        png = diag / f"input_{ptag}.png"
        Image.fromarray(img).save(png)
        cameras.append({"tag": ptag, "pitch_deg": float(pitch_deg),
                        "eye": eye.tolist(), "target": target.tolist(),
                        "up": up.tolist(), "fov": FOV, "width": W, "height": H,
                        "V": V.tolist(), "K": K.tolist(),
                        "png": str(png)})
        print(f"  [{ptag}] eye={eye.tolist()}")

    cam_json = diag / "cameras.json"
    cam_json.write_text(json.dumps(cameras, indent=2))

    # Step 2: SAM prompt = the label (no Qwen sub-item discovery for TV — too
    # narrow a target, no reliable sub-items).
    prompt_path = diag / "sam_prompt.txt"
    prompt_path.write_text(label)
    print(f"[prompt] {label}")

    # Step 3: SAM each view
    masks = {}
    for cam_data in cameras:
        ptag = cam_data["tag"]
        m, scores = sam_segment(cam_data["png"], label, threshold=SAM_THRESHOLD)
        if m is None or m.sum() == 0:
            print(f"  [{ptag}] no mask")
            continue
        m_bool = (m > 0).astype("uint8")
        m_bool = morph_clean(m_bool, r=5)
        if m_bool.sum() < 200:
            print(f"  [{ptag}] mask too small ({int(m_bool.sum())} px)")
            continue
        masks[ptag] = (m_bool, cam_data)
        Image.fromarray((m_bool * 255).astype("uint8")).save(diag / f"mask_{ptag}.png")
        top = max(scores) if scores else 0.0
        print(f"  [{ptag}] mask {int(m_bool.sum())} px (top score={top:.2f})")

    if not masks:
        sys.exit("[fatal] no usable SAM masks across all 4 pitches")

    # Step 4: vote — for each splat, count how many masks include its projection
    n_views = len(masks)
    min_votes = max(1, int(math.ceil(VOTE_FRAC * n_views)))
    print(f"[vote] {n_views} views, need ≥{min_votes} votes ({VOTE_FRAC*100:.0f}%)")

    vote = np.zeros(len(xyz), dtype=np.int32)
    for ptag, (m, cam_data) in masks.items():
        V = np.asarray(cam_data["V"])
        K = np.asarray(cam_data["K"])
        u, v_img, in_front = project_to_pixels(xyz, V, K)
        u_int = np.round(u).astype(np.int64)
        v_int = np.round(v_img).astype(np.int64)
        in_bounds = ((u_int >= 0) & (u_int < W) &
                     (v_int >= 0) & (v_int < H) & in_front)
        idx = np.where(in_bounds)[0]
        # m[v_int[idx], u_int[idx]] is True → this splat in this mask
        hit = m[v_int[idx], u_int[idx]] > 0
        vote[idx[hit]] += 1

    keep = vote >= min_votes
    n_kept = int(keep.sum())
    print(f"[vote] kept {n_kept:,} / {len(xyz):,} ({100*n_kept/max(1,len(xyz)):.1f}%)")
    if n_kept == 0:
        sys.exit("[fatal] vote kept 0 splats")

    # Save 4_sam_tight.ply directly (skip 2_sam_wide / 3_floor_drop for TV)
    out_ply = obj / "4_sam_tight.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # Canonical renders (use shared locked function)
    renders_dir = obj / "renders" / "4_sam_tight"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    # Write a minimal report.json so dispatcher's marker checks work
    report = {"procedure": "tv", "n_pitches": len(PITCHES_DEG),
              "n_masks": n_views, "min_votes": min_votes,
              "n_splats_kept": n_kept, "n_splats_total": len(xyz)}
    (diag / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
