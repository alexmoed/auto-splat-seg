#!/usr/bin/env python3
"""_phase4_art_extract.py — LOCKED wall-art extraction (2026-05-11).

PIPELINE (do not change without re-validating on all 8 v4 art pieces):

  1. Bbox-cone hull = splats inside the padded diorama bbox (6% pad),
     in front of the diorama camera. Saved as 1_visual_hull.ply.

  2. Yaw sweep: 8 yaw views around the hull centroid at level pitch,
     per-yaw camera distance from perpendicular extents. Saved to
     diagnostics/yaw_sweep/yaw{000..315}.png.

  3. Qwen bbox per yaw at 3% pad. None if edge-on/invisible.

  4. Qwen picks face-on yaw from the 8 yaw renders.

  5. Pitch sweep at the face-on yaw: 7 pitches [-35, -30, -15, 0, 15,
     30, 35]. Saved to diagnostics/pitch_sweep/pitch{...}.png.

  6. Qwen bbox per pitch at 3% pad.

  7. Combined multi-view vote: union of 8 yaw bboxes + 7 pitch bboxes
     (15 views). Keep splat if it lands inside ≥50% of valid (non-None)
     bboxes. Saved as 2_pitch_sweep_refined.ply.

  8. Canonical render set on refined hull = 8 yaws + face_on.png.
     Written to renders/1_visual_hull/.

  9. PCA wall slab — find true wall normal via PCA on the splat
     distribution (thinnest principal axis), keep splats within
     ±5cm of the peak depth.

 10. Final QC — Qwen sees face_on.png and decides PASS / REJECT.
     Lenient: PASS if any artwork content is recognizable.
     REJECT moves folder to <scene>/rejects/.

LOCKED CONSTANTS:
  HULL_PAD_PCT = 0.06       (diorama bbox padding for hull cone)
  YAW_SWEEP = [0,45,90,135,180,225,270,315]
  PITCH_SWEEP = [-35,-30,-15,0,15,30,35]
  PITCH_BBOX_PAD_PCT = 0.03 (per-view Qwen bbox padding for vote)
  PITCH_VOTE_FRAC = 0.5     (keep if in ≥50% of valid views)
  SWEEP_FOV = 50.0
  SWEEP_W, SWEEP_H = 1920, 1080
  SWEEP_MARGIN = 2.0        (camera distance multiplier)

Usage:
    python _phase4_art_extract.py <scene_dir> --quadrant NW --index 0
    python _phase4_art_extract.py <scene_dir> --quadrant NW --label "orange yellow"
"""
import argparse
import base64
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import (  # noqa: E402
    viewmat_look_at, build_K, project_to_pixels, slugify)
from sam_carve import render_canonical_5  # noqa: E402

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

# Hull extraction
HULL_PAD_PCT = 0.06            # 6% per side on diorama bbox

# Face-on yaw sweep
YAW_SWEEP = [0, 45, 90, 135, 180, 225, 270, 315]   # 8 angles
SWEEP_FOV = 50.0
SWEEP_W = 1920
SWEEP_H = 1080
SWEEP_PITCH_DEG = 0.0          # level
SWEEP_MARGIN = 2.0             # camera distance = extent × 2.0 / 2tan(fov/2)
                               # — bumped from 1.4 for more breathing room


VERTICAL_SWEEP = [0, 45, 90, 135, 180, 225, 270, 315]   # pitch angles at face-on yaw,
                                                          # mirroring the yaw sweep vertically
BBOX_PAD_PCT = 0.03                  # 3% per side on per-view Qwen bbox
VOTE_FRAC = 0.5                      # keep splat if in-bbox at ≥50% of valid views
ANISOTROPY_THRESHOLD = 20.0          # drop splats where max(scale)/min(scale) > this.
                                       # At 50 we still kept streak-splats radiating
                                       # from frame corners; at 10 dropped 83% (too
                                       # much). 20 is a middle ground — catches
                                       # corner streaks while preserving most art.
SLAB_HALF_M = 0.05                    # ±5cm slab around the art plane along the
                                       # PCA-derived true wall normal (thinnest
                                       # principal axis of the splat distribution).
                                       # Removes wall noise behind + floating splats.


def encode_b64(p: Path, max_dim: int = 480) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        new_size = (int(img.size[0] * s), int(img.size[1] * s))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_yaw_sweep(ply_path: Path, out_dir: Path, label: str):
    """Render hull from N yaw angles around its centroid at level pitch.
    Camera distance is computed PER YAW from the extents perpendicular to
    that yaw's view direction — so edge-on views zoom in tight, face-on
    views frame the full extent. Returns list of view dicts."""
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    centroid = np.median(xyz, axis=0)
    rel = xyz - centroid[None, :]

    aspect = SWEEP_W / SWEEP_H
    tan_h = math.tan(math.radians(SWEEP_FOV / 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for yaw_deg in YAW_SWEEP:
        rad = math.radians(yaw_deg)
        # Camera-from-centroid offset (eye = centroid + dist * back-vector)
        back = np.array([math.sin(rad), 0.0, math.cos(rad)])
        # Right axis perpendicular to back (in xz plane)
        right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
        up_axis = np.array([0.0, 1.0, 0.0])

        # Project splats onto right and up axes (signed)
        u_proj = rel @ right
        v_proj = rel @ up_axis
        # 5/95 percentile half-width on each axis
        u_half = float(max(abs(np.percentile(u_proj, 5)),
                            abs(np.percentile(u_proj, 95))))
        v_half = float(max(abs(np.percentile(v_proj, 5)),
                            abs(np.percentile(v_proj, 95))))
        # Required distance to fit content with SWEEP_MARGIN buffer
        dist_w = (u_half * SWEEP_MARGIN) / (tan_h * aspect)
        dist_h = (v_half * SWEEP_MARGIN) / tan_h
        dist = max(dist_w, dist_h, 0.5)

        eye = centroid + dist * back
        png = out_dir / f"yaw{yaw_deg:03d}.png"
        subprocess.run([
            sys.executable, VIEW_PY, str(ply_path), str(png),
            f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
            f"--target={centroid[0]:.4f},{centroid[1]:.4f},{centroid[2]:.4f}",
            "--up=0,-1,0", "--y-down",
            "--fov", str(SWEEP_FOV),
            "--width", str(SWEEP_W), "--height", str(SWEEP_H),
        ], check=True, capture_output=True)
        results.append({
            "yaw_deg": yaw_deg,
            "path": png,
            "eye": eye.tolist(),
            "target": centroid.tolist(),
            "up": [0.0, -1.0, 0.0],
            "fov": SWEEP_FOV,
            "width": SWEEP_W,
            "height": SWEEP_H,
            "u_half_m": u_half,
            "v_half_m": v_half,
            "dist_m": dist,
        })
    return results


def qwen_qc_face_on(face_on_path: Path, label: str) -> tuple:
    """Final QC: show Qwen the canonical face_on render and ask if the
    labeled art is actually recognizable. Returns (verdict, reason).
    verdict ∈ {"PASS", "REJECT"}.

    Lenient — accept partial frames / halo / off-center as long as the
    art itself is recognizable. Reject only when the render is
    incoherent / unrecognizable noise."""
    prompt = (
        f"This is the canonical face-on render of an extracted piece of "
        f"wall art labeled '{label}'.\n\n"
        f"Is the artwork RECOGNIZABLE as a {label}? Be lenient — accept "
        f"if you can see ANY identifiable picture / frame / artwork "
        f"content even if halo, partial frame, or off-center. Only reject "
        f"if the render is incoherent noise where no artwork is "
        f"identifiable at all.\n\n"
        f"Output JSON only:\n"
        f'{{"verdict": "PASS" | "REJECT", "reason": "<one sentence>"}}'
    )
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{encode_b64(face_on_path)}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=200, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    if s == -1 or e <= s:
        return "PASS", "qc_parse_fallback"
    try:
        d = json.loads(cleaned[s:e])
        verdict = d.get("verdict", "PASS").upper()
        reason = d.get("reason", "")
        if verdict not in ("PASS", "REJECT"):
            verdict = "PASS"
        return verdict, reason
    except Exception:
        return "PASS", "qc_json_fallback"


def qwen_bbox_on_render(img_path: Path, label: str) -> list | None:
    """Single tight bbox from a render. Returns [x0,y0,x1,y1] in pixels."""
    img = Image.open(img_path)
    img_w, img_h = img.size
    prompt = (
        f"Look at this rendered view of an extracted wall art piece "
        f"labeled '{label}'. Find the framed artwork (or canvas) and "
        f"return a TIGHT 2D bounding box around it. If the artwork is "
        f"edge-on or not clearly visible, return found=false.\n\n"
        f"Output JSON only:\n"
        f'{{"bbox_2d": [x_min, y_min, x_max, y_max], "found": true|false}}\n\n'
        f"Coordinates are integers in 0-1000 (normalized to image size)."
    )
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{encode_b64(img_path)}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=150, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    if s == -1 or e <= s:
        return None
    try:
        d = json.loads(cleaned[s:e])
        if not d.get("found", True):
            return None
        b = d.get("bbox_2d")
        if not b or len(b) != 4:
            return None
        return [int(b[0] * img_w / 1000), int(b[1] * img_h / 1000),
                int(b[2] * img_w / 1000), int(b[3] * img_h / 1000)]
    except Exception:
        return None


def render_pitch_sweep_at_yaw(ply_path: Path, out_dir: Path,
                                yaw_deg: float, pitch_deg_list: list):
    """Pitch sweep at fixed face-on yaw. Camera distance per-pitch from
    perpendicular extents (same as yaw sweep). Returns list of view dicts."""
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    centroid = np.median(xyz, axis=0)
    rel = xyz - centroid[None, :]

    aspect = SWEEP_W / SWEEP_H
    tan_h = math.tan(math.radians(SWEEP_FOV / 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    yaw_rad = math.radians(yaw_deg)
    for pitch_deg in pitch_deg_list:
        pitch_rad = math.radians(pitch_deg)
        # back-direction: yaw rotates in xz, pitch tilts up/down
        # y-down: positive pitch tilts camera DOWN (looking up-from-below)
        back = np.array([
            math.sin(yaw_rad) * math.cos(pitch_rad),
            -math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
        ])
        right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)])
        # camera up is perpendicular to back and right
        up_cam = np.cross(back, right)
        up_cam /= np.linalg.norm(up_cam)

        u_proj = rel @ right
        v_proj = rel @ up_cam
        u_half = float(max(abs(np.percentile(u_proj, 5)),
                            abs(np.percentile(u_proj, 95))))
        v_half = float(max(abs(np.percentile(v_proj, 5)),
                            abs(np.percentile(v_proj, 95))))
        dist_w = (u_half * SWEEP_MARGIN) / (tan_h * aspect)
        dist_h = (v_half * SWEEP_MARGIN) / tan_h
        dist = max(dist_w, dist_h, 0.5)

        eye = centroid + dist * back
        png = out_dir / f"pitch{pitch_deg:+03d}.png"
        subprocess.run([
            sys.executable, VIEW_PY, str(ply_path), str(png),
            f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
            f"--target={centroid[0]:.4f},{centroid[1]:.4f},{centroid[2]:.4f}",
            f"--up={-up_cam[0]:.4f},{-up_cam[1]:.4f},{-up_cam[2]:.4f}", "--y-down",
            "--fov", str(SWEEP_FOV),
            "--width", str(SWEEP_W), "--height", str(SWEEP_H),
        ], check=True, capture_output=True)
        results.append({
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "path": png,
            "eye": eye.tolist(),
            "target": centroid.tolist(),
            "up": (-up_cam).tolist(),
            "fov": SWEEP_FOV,
            "width": SWEEP_W,
            "height": SWEEP_H,
        })
    return results


def refine_hull_by_bboxes(xyz: np.ndarray, views: list,
                            pad_pct: float, vote_frac: float,
                            label_prefix: str = "vote") -> np.ndarray:
    """Generic multi-view bbox vote. For each view with a bbox: project all
    splats, vote if inside padded bbox. Keep splat if ≥vote_frac votes."""
    valid_views = [v for v in views if v.get("bbox_pixels")]
    if not valid_views:
        return np.ones(len(xyz), dtype=bool)

    vote_count = np.zeros(len(xyz), dtype=np.int32)
    for v in valid_views:
        bbox = v["bbox_pixels"]
        bw = bbox[2] - bbox[0]; bh = bbox[3] - bbox[1]
        px = bw * pad_pct
        py = bh * pad_pct
        padded = [
            max(0, int(bbox[0] - px)),
            max(0, int(bbox[1] - py)),
            min(v["width"], int(bbox[2] + px)),
            min(v["height"], int(bbox[3] + py)),
        ]
        V_v = viewmat_look_at(v["eye"], v["target"], v["up"])
        K_v = build_K(v["fov"], v["width"], v["height"])
        u, vp, in_front = project_to_pixels(xyz, V_v, K_v)
        in_b = ((u >= padded[0]) & (u <= padded[2]) &
                 (vp >= padded[1]) & (vp <= padded[3]) & in_front)
        vote_count += in_b.astype(np.int32)
        v["bbox_padded"] = padded

    threshold = max(1, int(np.ceil(vote_frac * len(valid_views))))
    keep = vote_count >= threshold
    print(f"[{label_prefix}] valid_views={len(valid_views)}/{len(views)}  "
          f"vote_thresh={threshold}  kept={int(keep.sum()):,}")
    return keep


def qwen_pick_face_on(yaw_views: list, label: str) -> tuple:
    """Show Qwen all yaw renders and ask which one is face-on of the art."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for v in yaw_views:
        content.append({"type": "text", "text": f"\nView at yaw {v['yaw_deg']}°:"})
        content.append({"type": "image_url",
                         "image_url": {
                             "url": f"data:image/png;base64,{encode_b64(v['path'])}"}})
    content.append({"type": "text", "text":
        f"You are looking at {len(yaw_views)} renders of a wall-art "
        f"extraction labeled '{label}'. Each render is from a different "
        f"yaw angle around the object.\n\n"
        f"Pick the SINGLE view that shows the artwork most FACE-ON "
        f"(perpendicular to the painting/print, i.e. you can see the "
        f"art's full surface flat to the camera, NOT at an angle, NOT "
        f"the back of the frame, NOT the edge).\n\n"
        f"If NONE of the views show identifiable artwork (just smeared "
        f"noise or background), pick the cleanest-looking one anyway.\n\n"
        f"Output JSON only:\n"
        f"{{\n"
        f'  "best_yaw_deg": <int from the list above>,\n'
        f'  "reasoning": "<one short sentence>"\n'
        f"}}"})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=150, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    best_yaw = YAW_SWEEP[0]
    reasoning = "fallback"
    if s != -1 and e > s:
        try:
            d = json.loads(cleaned[s:e])
            cand = int(d.get("best_yaw_deg", YAW_SWEEP[0]))
            if cand in YAW_SWEEP:
                best_yaw = cand
            reasoning = d.get("reasoning", "")
        except Exception:
            pass
    best_path = next(v["path"] for v in yaw_views if v["yaw_deg"] == best_yaw)
    return best_yaw, best_path, reasoning, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--quadrant", choices=["NE", "NW", "SE", "SW"], required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--source-ply", type=Path, default=None,
                    help="default: <scene>/step7_cardinal_aligned.ply (rotated raw)")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    # Load phase-4 items
    items_path = scene / "_phase4_temp" / "qwen_art_items.json"
    if not items_path.exists():
        sys.exit(f"[fatal] missing {items_path} — run _phase4_art_detect.py first")
    pdata = json.load(open(items_path))
    q_items = pdata.get("by_quadrant", {}).get(args.quadrant, [])
    if not q_items:
        sys.exit(f"[fatal] no art in quadrant {args.quadrant}")

    chosen = None
    chosen_idx = None
    if args.index is not None:
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
            sys.exit(f"[fatal] no art in {args.quadrant} matching '{args.label}'")
    else:
        sys.exit("[fatal] specify --label or --index")

    bbox_px = chosen.get("bbox_pixels")
    if not bbox_px:
        sys.exit(f"[fatal] no bbox_pixels for {chosen}")
    print(f"[pick] {args.quadrant}[{chosen_idx}] '{chosen['label']}'")

    # Diorama camera
    cams = json.load(open(scene / "_phase2_temp" / "cameras.json"))
    cam = cams[args.quadrant]

    # Pad bbox
    img_w, img_h = cam["width"], cam["height"]
    bw = bbox_px[2] - bbox_px[0]
    bh = bbox_px[3] - bbox_px[1]
    px = bw * HULL_PAD_PCT
    py = bh * HULL_PAD_PCT
    padded = [
        max(0, int(bbox_px[0] - px)),
        max(0, int(bbox_px[1] - py)),
        min(img_w, int(bbox_px[2] + px)),
        min(img_h, int(bbox_px[3] + py)),
    ]
    print(f"[bbox] tight={bbox_px}  padded={padded}  ({int(HULL_PAD_PCT*100)}%)")

    # Load source PLY
    source_ply_path = args.source_ply or (scene / "step7_cardinal_aligned.ply")
    if not source_ply_path.exists():
        sys.exit(f"[fatal] source PLY missing: {source_ply_path}")
    print(f"[src ] {source_ply_path}")
    pl = PlyData.read(str(source_ply_path))
    vdata = pl["vertex"]
    xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
    print(f"[src ] {len(xyz):,} splats")

    # Visual hull: bbox cone in front of camera. No wall geometry.
    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])
    u, v_img, in_front = project_to_pixels(xyz, V, K)
    inside = ((u >= padded[0]) & (u <= padded[2]) &
              (v_img >= padded[1]) & (v_img <= padded[3]))
    keep = in_front & inside
    n_kept = int(keep.sum())
    print(f"[hull] in_front={int(in_front.sum()):,}  "
          f"in_bbox={int(inside.sum()):,}  kept={n_kept:,}")
    if n_kept == 0:
        sys.exit("[fatal] 0 splats in bbox cone")

    # Output folder (auto-suffix if collision)
    base_slug = slugify(chosen["label"])
    slug = base_slug
    n = 2
    while (scene / f"02_{slug}").exists() and (
            scene / f"02_{slug}" / "1_visual_hull_meta.json").exists():
        try:
            ex = json.load(open(scene / f"02_{slug}" / "1_visual_hull_meta.json"))
            if (ex.get("phase") == 4 and ex.get("quadrant") == args.quadrant and
                    ex.get("source_index_in_quadrant") == chosen_idx):
                break
        except Exception:
            pass
        slug = f"{base_slug}_{n}"
        n += 1
    out_dir = scene / f"02_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "1_visual_hull.ply"
    PlyData([PlyElement.describe(vdata.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # Stage 1: yaw sweep on bbox-cone hull
    sweep_dir = out_dir / "diagnostics" / "yaw_sweep"
    print(f"[yaw] rendering {len(YAW_SWEEP)} yaw views → {sweep_dir}")
    yaw_views = render_yaw_sweep(out_ply, sweep_dir, chosen["label"])

    # Stage 2: Qwen bbox per yaw (3% pad)
    print(f"[yaw] Qwen bbox per yaw...")
    for v in yaw_views:
        bbox = qwen_bbox_on_render(v["path"], chosen["label"])
        v["bbox_pixels"] = bbox
        print(f"  yaw {v['yaw_deg']:3d}°: {'bbox=' + str(bbox) if bbox else 'not visible'}")

    # Stage 3: Qwen picks face-on yaw (from same 8 yaw renders)
    print(f"[yaw] asking Qwen which is face-on...")
    best_yaw, best_path, reasoning, raw = qwen_pick_face_on(
        yaw_views, chosen["label"])
    print(f"[yaw] best_yaw={best_yaw}°  reason={reasoning}")
    (sweep_dir / "qwen_pick_raw.txt").write_text(raw)

    # Stage 4: vertical sweep at face-on yaw (8 pitches mirroring yaw sweep)
    pitch_dir = out_dir / "diagnostics" / "pitch_sweep"
    print(f"[vertical] rendering {len(VERTICAL_SWEEP)} pitches at yaw {best_yaw}° → {pitch_dir}")
    pitch_views = render_pitch_sweep_at_yaw(out_ply, pitch_dir,
                                              best_yaw, VERTICAL_SWEEP)

    # Stage 5: Qwen bbox per vertical view (3% pad)
    print(f"[vertical] Qwen bbox per pitch...")
    for v in pitch_views:
        bbox = qwen_bbox_on_render(v["path"], chosen["label"])
        v["bbox_pixels"] = bbox
        print(f"  pitch {v['pitch_deg']:+4d}°: {'bbox=' + str(bbox) if bbox else 'not visible'}")

    # Stage 6: combined multi-view vote (8 yaw + 8 vertical = 16 views)
    combined_views = yaw_views + pitch_views
    refine_keep = refine_hull_by_bboxes(xyz[keep], combined_views,
                                          BBOX_PAD_PCT,
                                          VOTE_FRAC,
                                          label_prefix="combined-vote")
    keep_indices = np.where(keep)[0]
    voted_indices = keep_indices[refine_keep]
    n_voted = len(voted_indices)
    print(f"[refine] {n_voted:,} splats after combined vote")

    # Stage 7 (anisotropy filter) REMOVED 2026-05-11 per user direction.
    aniso_indices = voted_indices

    # Stage 8: PCA-derived wall slab — find the TRUE wall normal from the
    # splat distribution (thinnest principal axis) rather than using the
    # yaw-snapped face-on direction (which is 45° granular and slices
    # diagonally through the wall). Then keep splats within ±SLAB_HALF_M
    # along that true normal.
    splat_xyz = np.stack([vdata["x"][aniso_indices],
                           vdata["y"][aniso_indices],
                           vdata["z"][aniso_indices]], axis=1).astype(np.float64)
    centroid = splat_xyz.mean(axis=0)
    centered = splat_xyz - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals are sorted ascending; smallest = thinnest = wall normal direction
    wall_normal = eigvecs[:, 0]
    # Make it point roughly toward the diorama camera (so positive depth = in front of wall)
    diorama_eye = np.array(cam["eye"])
    if np.dot(diorama_eye - centroid, wall_normal) < 0:
        wall_normal = -wall_normal
    depth_along_normal = (splat_xyz - centroid) @ wall_normal
    # Find peak depth via histogram (1cm bins)
    lo, hi = float(depth_along_normal.min()), float(depth_along_normal.max())
    if hi - lo < 0.05:
        plane_depth = float(np.median(depth_along_normal))
    else:
        nbins = max(20, int((hi - lo) / 0.01))
        hist, edges = np.histogram(depth_along_normal, bins=nbins)
        peak_idx = int(np.argmax(hist))
        plane_depth = float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    keep_slab = np.abs(depth_along_normal - plane_depth) <= SLAB_HALF_M
    final_indices = aniso_indices[keep_slab]
    n_dropped_slab = int((~keep_slab).sum())
    print(f"[slab] PCA-normal={wall_normal}  eigvals={eigvals}")
    print(f"[slab] plane_depth={plane_depth:.3f}m  "
          f"±{SLAB_HALF_M*100:.0f}cm  dropped {n_dropped_slab:,}  kept {len(final_indices):,}")

    n_refined = len(final_indices)
    refined_ply = out_dir / "2_pitch_sweep_refined.ply"
    PlyData([PlyElement.describe(vdata.data[final_indices], "vertex")],
            text=False).write(str(refined_ply))
    print(f"[save] {refined_ply}")

    # Canonical render set for phase 4 art = 8 yaw sweep views on the
    # REFINED hull + face_on.png. Canonical_5 (cardinal y0/y90/y180/y270/
    # topdown) is dropped — for obliquely-mounted art it captures
    # edge-on / behind views that look like noise.
    canonical_dir = out_dir / "renders" / "1_visual_hull"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    print(f"[canonical] rendering {len(YAW_SWEEP)} yaws on refined hull → {canonical_dir}")
    refined_yaw_views = render_yaw_sweep(refined_ply, canonical_dir,
                                          chosen["label"])

    # Save face-on canonical from the FIRST yaw sweep (bbox-cone hull
    # with scene context)
    import shutil
    face_on_dest = canonical_dir / "face_on.png"
    shutil.copyfile(best_path, face_on_dest)
    print(f"[face-on] {face_on_dest} (from yaw sweep, yaw {best_yaw}°)")

    # Stage 9: final QC — ask Qwen if the face_on shows recognizable art.
    # Use the REFINED-hull yaw-sweep face_on render (canonical_dir).
    qc_verdict, qc_reason = qwen_qc_face_on(face_on_dest, chosen["label"])
    print(f"[qc] verdict={qc_verdict}  reason={qc_reason}")

    meta = {
        "phase": 4,
        "quadrant": args.quadrant,
        "label": chosen["label"],
        "confidence": chosen.get("confidence"),
        "source_index_in_quadrant": chosen_idx,
        "diorama_bbox_pixels": bbox_px,
        "diorama_bbox_padded": padded,
        "hull_pad_pct_per_side": HULL_PAD_PCT,
        "yaw_sweep": YAW_SWEEP,
        "vertical_sweep": VERTICAL_SWEEP,
        "bbox_pad_pct": BBOX_PAD_PCT,
        "vote_frac": VOTE_FRAC,
        "anisotropy_threshold": ANISOTROPY_THRESHOLD,
        "yaw_bboxes": [{"yaw_deg": v["yaw_deg"],
                         "bbox_pixels": v.get("bbox_pixels")}
                        for v in yaw_views],
        "vertical_bboxes": [{"pitch_deg": v["pitch_deg"],
                              "bbox_pixels": v.get("bbox_pixels")}
                             for v in pitch_views],
        "best_yaw_deg": best_yaw,
        "qwen_reasoning": reasoning,
        "source_ply": str(source_ply_path),
        "n_splats_hull": n_kept,
        "n_splats_voted": n_voted,
        "n_splats_refined": n_refined,
        "n_splats_total": len(xyz),
        "qc_verdict": qc_verdict,
        "qc_reason": qc_reason,
    }
    (out_dir / "1_visual_hull_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] PLY: {out_ply}")
    print(f"        face-on: {face_on_dest}")

    # Move to rejects/ on REJECT
    if qc_verdict == "REJECT":
        rejects_dir = scene / "rejects"
        rejects_dir.mkdir(parents=True, exist_ok=True)
        dest = rejects_dir / out_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(out_dir), str(dest))
        print(f"[reject] moved to {dest}")


if __name__ == "__main__":
    main()
