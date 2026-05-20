#!/usr/bin/env python3
"""sam_carve.py — Stage 2 multi-view SAM carve on a visual hull.

Ports the locked plugin Stage 2 cameras: 12 yaws × 2 pitches (−15°, −45°)
+ topdown = 25 views at 1080p, all rendered in-process (PLY loaded once,
no reloads).

Per-object output layout (7-stage pipeline; this script handles stage 2):
  02_<slug>/
    visual_hull.ply        (Stage 1 — extract_one.py)
    sam_wide.ply           (Stage 2 — this script's step 4 vote, later)
    floor_drop.ply         (Stage 3 — RANSAC floor)
    sam_tight.ply          (Stage 4 — 2nd SAM tight pad)
    aabb_filter.ply        (Stage 5 — Qwen tight xz AABB)
    floor_band.ply         (Stage 6 — soft only)
    export.ply             (Stage 7 — final shippable)
    renders/<stage>/{y0,y90,y180,y270,topdown}.png     (5 renders per stage)
    diagnostics/<stage>/   (per-stage diagnostics)

Stage 2 (sam_wide) diagnostics this script writes:
  diagnostics/sam_wide/
    input_y{N}_p{P}.png         (step 1 — 24 oblique views, all 1080p)
    input_topdown.png           (step 1)
    cameras.json                (step 1 — V/K per view)
    sam_prompt.txt              (step 2 — Qwen-derived pipe-union)
    sam_prompt_raw.txt          (step 2 — Qwen raw response)
    sam_prompt_history.json     (step 3 — retry attempts log)
    mask_y{N}_p{P}.png          (step 3 — raw post-morph mask)
    mask_padded_y{N}_p{P}.png   (step 3 — dilated by sam_pad)
    mask_topdown.png            (step 3)
    mask_padded_topdown.png     (step 3)
    report.json                 (step 3 — per-view scores + attempts)
Yaw tag format: y0, y90, y180 (no leading zeros). Pitch: p-15, p-45.

Steps (run one at a time):
  --step 1   render all 25 views into 02_<slug>/sam_views/
             save camera (V, K) per view to sam_views_cameras.json
  --step 2   Qwen describes object + sub-items from 4 cardinal views,
             outputs a pipe-union SAM prompt (e.g.
             "armchair|throw blanket|striped pillow"). Saves to
             02_<slug>/sam_prompt.txt
  --step 3   SAM each of 25 views with the pipe-union prompt; per-prompt
             skip-if-empty (<50 px), union via np.maximum, view skip
             if <200 px, morph closing + fill_holes, dilate by
             sam_pad × focal/depth.
             HEALTH CHECK: if main-prompt mask hits <MIN_HIT_VIEWS=3
             views, loop back to Qwen with failure context and get a
             refined prompt; retry SAM (up to MAX_ATTEMPTS=3 total).
             Saves mask_<tag>.png + mask_padded_<tag>.png +
             sam_masks_info.json
  --step 4   project visual_hull.ply splats through every saved camera,
             vote across dilated masks → sam_carved.ply
             [TODO]

Reuses iteration_1/extract_one.py for camera math and slug.
Skill imports (gsplat-viewer) for in-process rendering — plugin scripts
are reference-only, not imported.

Usage:
    python sam_carve.py <scene_dir> 02_<slug>/ --step 1
    python sam_carve.py <scene_dir> 02_<slug>/ --step 2
"""
import argparse
import base64
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from openai import OpenAI
from plyfile import PlyData, PlyElement

# Skill (not plugin) — OK to import, never edit.
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import (  # noqa: E402
    load_gsplat_ply, render_splat, rotation_matrix_from_yaw_pitch,
)

# Iteration_1 sibling — reuse without editing.
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402

# Locked Stage 2 view set (mirrors plugin's vh_topdown.py defaults)
YAWS_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
PITCHES_DEG = [-15.0, -45.0]
TOPDOWN_PITCH = -89.0

FOV = 70.0
W, H = 1920, 1080  # 1080p
Y_DOWN = True

# RENDER MARGIN — single source of truth. Distance multiplier on extent.
# Bumped from 1.6 (SAM views) / 1.4 (canonical) per TODO_TOMORROW.md item 2:
# objects were getting cut at frame edges. 2.0 gives consistent breathing
# room around the object. Used by EVERY render function in this script
# AND must match extract_one.py's render_topdown_simple margin.
RENDER_MARGIN = 2.0

# Step 2: views Qwen sees to derive SAM prompt (canonical tags, no leading zeros)
PROMPT_DERIVE_VIEWS = ["y0_p-15", "y90_p-15", "y180_p-15", "y270_p-15"]
QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

# Step 3: SAM
SAM_THRESHOLD = 0.4              # plugin lock
SAM_PAD_M = 0.15                 # mask dilation in meters. Was 0.5 (way too
                                  # wide, no carving), 0.2 default for a while.
                                  # 2026-05-20 dropped to 0.15 — tighter carve
                                  # when sam_carve runs with the for-Qwen
                                  # clean prompt (5 terms, no neighbors).
                                  # If carve too aggressive: ladder up
                                  # 0.15 → 0.20 → 0.30 → 0.40.
MIN_PROMPT_PX = 50               # skip a prompt if its raw mask < this
MIN_VIEW_PX = 200                # skip a view if union mask < this
MIN_HIT_VIEWS = 3                # health threshold for main prompt
MAX_ATTEMPTS = 3                 # initial + 2 reprompts

# Step 4: vote + render
MIN_VIEWS_FRAC = 0.6             # plugin lock — keep splat if voted in by ≥60%
CANONICAL_YAWS = [0, 90, 180, 270]
CANONICAL_PITCH = -20.0
CANONICAL_TOPDOWN_PITCH = -89.0
CANONICAL_W, CANONICAL_H = 1920, 1080   # 1080p, locked


def build_camera(center, yaw_deg, pitch_deg, distance, fov, w, h, y_down=True):
    """Mirrors plugin's vh_topdown.build_camera (orbit around `center`)."""
    pitch_eff = pitch_deg if y_down else -pitch_deg
    center = np.asarray(center, dtype=np.float32)
    base_eye = center + np.array([0, 0, distance], dtype=np.float32)
    R = rotation_matrix_from_yaw_pitch(yaw_deg, pitch_eff)
    eye = center + R.T @ (base_eye - center)
    up = np.array([0, -1 if y_down else 1, 0], dtype=np.float32)
    V = viewmat_look_at(eye, center, up)
    K = build_K(fov, w, h)
    return V, K, eye


def compute_wall_skip(scene_dir: Path, means):
    """Detect the nearest room wall to the object hull, return
    (wall_axis, wall_back, eye_behind_object_callable).

    `eye_behind_object(eye)` returns True if `eye` sits on the wall-side
    of the object's hull back face (i.e., between the back face and the
    wall, or past the wall). Used to skip orbit cameras that would
    render the object's sparse back through wall splats.

    Only applies when the hull is within 2.5m of a wall; otherwise the
    callable always returns False (free-standing object — keep all
    orbit cameras)."""
    room_src = scene_dir / "step8_density_filtered.ply"
    if not room_src.exists():
        room_src = scene_dir / "step7_cardinal_aligned.ply"
    if not room_src.exists():
        return None, None, (lambda eye: False)

    rpl = PlyData.read(str(room_src))['vertex']
    rxz = np.stack([rpl['x'], rpl['z']], axis=1)
    room_xmin, room_xmax = np.percentile(rxz[:, 0], [1, 99])
    room_zmin, room_zmax = np.percentile(rxz[:, 1], [1, 99])
    hxz = np.stack([means[:, 0], means[:, 2]], axis=1)
    hull_xmin, hull_xmax = np.percentile(hxz[:, 0], [2, 98])
    hull_zmin, hull_zmax = np.percentile(hxz[:, 1], [2, 98])
    candidates = [
        ("x_min", hull_xmin - room_xmin, hull_xmin),
        ("x_max", room_xmax - hull_xmax, hull_xmax),
        ("z_min", hull_zmin - room_zmin, hull_zmin),
        ("z_max", room_zmax - hull_zmax, hull_zmax),
    ]
    candidates.sort(key=lambda c: c[1])
    nearest_axis, nearest_dist, nearest_back = candidates[0]
    print(f"[room-bounds] x=[{room_xmin:.2f},{room_xmax:.2f}] "
          f"z=[{room_zmin:.2f},{room_zmax:.2f}]")
    print(f"[hull-bounds] x=[{hull_xmin:.2f},{hull_xmax:.2f}] "
          f"z=[{hull_zmin:.2f},{hull_zmax:.2f}]")
    print(f"[wall-adj] nearest wall: {nearest_axis} dist={nearest_dist:.2f}m")

    if nearest_dist > 2.5:
        return None, None, (lambda eye: False)

    wall_axis = nearest_axis
    wall_back = nearest_back

    def eye_behind_object(eye):
        if wall_axis == "x_min": return eye[0] < wall_back
        if wall_axis == "x_max": return eye[0] > wall_back
        if wall_axis == "z_min": return eye[2] < wall_back
        if wall_axis == "z_max": return eye[2] > wall_back
        return False

    return wall_axis, wall_back, eye_behind_object


# ─────────────────────────────────────────────────────────────────────
# Topdown-bbox neighbor subtraction for the Qwen identification step.
# Built 2026-05-20. SCOPE: produces 1_visual_hull_for_qwen.ply + a parallel
# input_qwen_*.png render set at step 1. Step 2 prefers these renders so
# Qwen identifies parts of THIS target without seeing already-extracted
# neighbors. Downstream stages (SAM step 3, vote step 4, floor_drop,
# sam_tight, sam_low_refine, inside_outside) all keep using
# 1_visual_hull.ply / input_*.png — the carved hull is a dead-end
# artifact whose only consumer is Qwen at step 2.
QWEN_CARVE_OVERLAP_THRESH = 0.25   # neighbor bbox >25% inside target padded bbox
QWEN_CARVE_NEIGHBOR_PAD = 0.01     # 1% per-side pad on each neighbor bbox


def _bbox_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _overlap_area(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _find_inside_neighbors(scene_dir: Path, target_dir: Path,
                              target_padded_bbox,
                              overlap_thresh: float = QWEN_CARVE_OVERLAP_THRESH):
    """List sibling 02_*/ objects whose tight bbox has > overlap_thresh of
    its OWN area inside the target's padded bbox. Both bboxes are in
    phase-1 inventory topdown image coords (3840×2160)."""
    out = []
    for od in sorted(scene_dir.iterdir()):
        if not (od.is_dir() and od.name.startswith("02_") and
                od.name != target_dir.name):
            continue
        mp = od / "1_visual_hull_meta.json"
        if not mp.exists():
            continue
        om = json.load(open(mp))
        ob = om.get("bbox_pixels_tight")
        if not ob:
            continue
        a = _bbox_area(ob)
        if a == 0:
            continue
        r = _overlap_area(ob, target_padded_bbox) / a
        if r > overlap_thresh:
            out.append({"name": od.name, "label": om.get("label", "?"),
                         "tight_bbox": ob, "inside_ratio": r})
    return out


def _pad_bbox(bbox, pct: float):
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pct, bh * pct
    return [int(x0 - px), int(y0 - py), int(x1 + px), int(y1 + py)]


def _carve_hull_for_qwen(scene_dir: Path, obj_dir: Path, scene: dict,
                            target_means_np):
    """Project target hull splats through the phase-1 topdown camera and
    drop splats whose (u,v) falls inside any padded neighbor bbox.
    Returns (carved_scene_dict, audit_dict) or (None, None) if no
    neighbors meet the threshold (fall through to input_*.png at step 2).

    The carved scene dict is rendered at sam_carve step 1 cameras as
    input_qwen_*.png. NEVER used for SAM segmentation (step 3) or
    vote_carve (step 4). Only consumer is Qwen at step 2."""
    import torch as _t
    meta_path = obj_dir / "1_visual_hull_meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.load(open(meta_path))
    tgt_padded = meta.get("bbox_pixels_padded")
    cam = meta.get("camera")
    if tgt_padded is None or cam is None:
        return None, None
    neighbors = _find_inside_neighbors(scene_dir, obj_dir, tgt_padded)
    if not neighbors:
        return None, None
    print(f"\n[step1] carve-for-qwen — {len(neighbors)} neighbor(s) with "
          f">{int(QWEN_CARVE_OVERLAP_THRESH*100)}% bbox inside target padded bbox")
    for n in neighbors:
        print(f"  {n['name']:50s} '{n['label'][:32]}' "
              f"inside={n['inside_ratio']*100:5.1f}%")

    # Project hull splats through phase-1 topdown camera
    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])
    u, v_img, _ = project_to_pixels(target_means_np, V, K)

    drop = np.zeros(len(target_means_np), dtype=bool)
    for n in neighbors:
        pb = _pad_bbox(n["tight_bbox"], QWEN_CARVE_NEIGHBOR_PAD)
        in_box = ((u >= pb[0]) & (u <= pb[2]) &
                  (v_img >= pb[1]) & (v_img <= pb[3]))
        n_in = int(in_box.sum())
        print(f"  drop {n_in:6d} splats inside {n['name']} padded bbox {pb}")
        n["padded_bbox"] = pb
        n["splats_dropped"] = n_in
        drop |= in_box
    keep_np = ~drop
    keep = _t.from_numpy(keep_np).to(scene["means"].device)
    n_full = len(scene["means"])
    new_scene = {}
    for k, val in scene.items():
        if hasattr(val, "shape") and len(val) == n_full:
            new_scene[k] = val[keep]
        else:
            new_scene[k] = val
    n_kept = int(keep_np.sum())
    print(f"  [carve-for-qwen] kept {n_kept:,}/{n_full:,} "
          f"({100*n_kept/n_full:.1f}%) — {n_full-n_kept:,} dropped")
    audit = {
        "overlap_thresh": QWEN_CARVE_OVERLAP_THRESH,
        "neighbor_pad_pct_per_side": QWEN_CARVE_NEIGHBOR_PAD,
        "target_padded_bbox": tgt_padded,
        "neighbors": neighbors,
        "splats_in": n_full,
        "splats_out": n_kept,
    }
    return new_scene, audit


def step1_render_views(scene_dir: Path, obj_dir: Path):
    hull_ply = obj_dir / "1_visual_hull.ply"
    if not hull_ply.exists():
        sys.exit(f"[fatal] missing {hull_ply}\n  run extract_one.py first")

    diag = obj_dir / "diagnostics" / "2_sam_wide"
    diag.mkdir(parents=True, exist_ok=True)
    # Clear stale per-view inputs from prior runs
    for f in diag.glob("input_*.png"):
        f.unlink()

    print(f"[load] {hull_ply}")
    scene = load_gsplat_ply(str(hull_ply))
    means = scene["means"].detach().cpu().numpy()
    n_splats = len(means)
    print(f"[load] {n_splats:,} splats")

    # Robust center+extent — uses median + p5/p95 instead of min/max midpoint.
    # Phase-3 hulls (and any cone-extracted PLY) carry capture-noise outliers
    # far behind the wall; min/max midpoint would pivot the orbit around
    # those outliers, drifting the actual object off-frame on most yaws.
    # Median is unaffected; p5/p95 still fits the densest part of the hull.
    center = np.median(means, axis=0).astype(np.float32)
    p5  = np.percentile(means, 5,  axis=0)
    p95 = np.percentile(means, 95, axis=0)
    extent = float((p95 - p5).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m margin={RENDER_MARGIN}")

    wall_axis, wall_back, eye_behind_object = compute_wall_skip(
        scene_dir, means)

    cameras = []
    skipped = []

    # 12 yaws × 2 pitches = 24 oblique views.
    # Tag format: y{int}_p{int} matching plugin canonical (e.g. y0_p-15).
    for pitch_deg in PITCHES_DEG:
        ptag = f"p{int(round(pitch_deg))}"
        for yaw_deg in YAWS_DEG:
            ytag = f"y{int(round(yaw_deg))}"
            tag = f"{ytag}_{ptag}"
            V, K, eye = build_camera(center, yaw_deg, pitch_deg, distance,
                                      FOV, W, H, y_down=Y_DOWN)
            if eye_behind_object(eye):
                skipped.append(tag)
                print(f"  [{tag}] SKIP — eye behind object "
                      f"({eye[0]:.2f},{eye[2]:.2f})")
                continue
            img = render_splat(scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
            out_png = diag / f"input_{tag}.png"
            Image.fromarray(img).save(out_png)
            cameras.append({
                "tag": tag, "yaw_deg": float(yaw_deg),
                "pitch_deg": float(pitch_deg), "fov": FOV,
                "width": W, "height": H,
                "V": V.tolist(), "K": K.tolist(),
                "eye": eye.tolist(), "target": center.tolist(),
                "png": str(out_png),
            })
            print(f"  [{tag}] → input_{tag}.png")

    # Topdown
    V, K, eye = build_camera(center, 0.0, TOPDOWN_PITCH, distance,
                              FOV, W, H, y_down=Y_DOWN)
    img = render_splat(scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
    out_png = diag / "input_topdown.png"
    Image.fromarray(img).save(out_png)
    cameras.append({
        "tag": "topdown", "yaw_deg": 0.0, "pitch_deg": TOPDOWN_PITCH,
        "fov": FOV, "width": W, "height": H,
        "V": V.tolist(), "K": K.tolist(),
        "eye": eye.tolist(), "target": center.tolist(),
        "png": str(out_png),
    })
    print(f"  [topdown] → input_topdown.png")

    cam_json = diag / "cameras.json"
    cam_json.write_text(json.dumps({
        "ply_path": str(hull_ply),
        "n_splats": n_splats,
        "fov": FOV, "width": W, "height": H,
        "y_down": Y_DOWN,
        "yaws_deg": YAWS_DEG,
        "pitches_deg": PITCHES_DEG,
        "topdown_pitch_deg": TOPDOWN_PITCH,
        "center": center.tolist(),
        "extent": extent,
        "distance": distance,
        "cameras": cameras,
    }, indent=2))
    # ───── Qwen identification carve (2026-05-20) ─────
    # Produce a parallel render set from a sibling-bbox-subtracted hull,
    # for the Qwen identification step (step 2). All downstream stages
    # keep using input_*.png + 1_visual_hull.ply — this is dead-end.
    # If no neighbors meet the threshold, no extra files are written and
    # step 2 falls back to input_*.png.
    # Clear stale input_qwen_*.png from any prior run.
    for f in diag.glob("input_qwen_*.png"):
        f.unlink()
    (obj_dir / "1_visual_hull_for_qwen.ply").unlink(missing_ok=True)
    (obj_dir / "1_visual_hull_for_qwen_audit.json").unlink(missing_ok=True)

    carved_scene, audit = _carve_hull_for_qwen(scene_dir, obj_dir, scene, means)
    if carved_scene is not None:
        # Render the SAME camera set from the carved hull
        for cam in cameras:
            V = np.array(cam["V"], dtype=np.float32)
            K = np.array(cam["K"], dtype=np.float32)
            img = render_splat(carved_scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
            out_png = diag / f"input_qwen_{cam['tag']}.png"
            Image.fromarray(img).save(out_png)
        print(f"  [carve-for-qwen] wrote {len(cameras)} input_qwen_*.png")
        # Save the carved PLY + audit alongside the original hull (do NOT
        # overwrite 1_visual_hull.ply — that's the source of truth that
        # SAM step 3 + vote step 4 + downstream operate on).
        out_ply = obj_dir / "1_visual_hull_for_qwen.ply"
        cm = carved_scene["means"].detach().cpu().numpy()
        # Carry over scales/quats/opacities/colors via the keep mask
        # already applied in _carve_hull_for_qwen — we need to write a
        # PLY in the same gsplat format. The full PLY can't be reduced
        # here without re-reading the source; just save the means as a
        # diagnostic point cloud (Qwen never sees the PLY directly, only
        # the renders from it).
        from plyfile import PlyElement
        elt = np.empty(len(cm), dtype=[('x','f4'),('y','f4'),('z','f4')])
        elt['x'] = cm[:, 0]; elt['y'] = cm[:, 1]; elt['z'] = cm[:, 2]
        PlyData([PlyElement.describe(elt, 'vertex')],
                text=False).write(str(out_ply))
        audit["out_ply"] = str(out_ply)
        (obj_dir / "1_visual_hull_for_qwen_audit.json").write_text(
            json.dumps(audit, indent=2))
        print(f"  [carve-for-qwen] saved {out_ply.name}")
    else:
        print(f"  [carve-for-qwen] no neighbors — step 2 will use input_*.png")

    print(f"\n[step1] DONE — {len(cameras)} views rendered")
    print(f"  diagnostics: {diag}")
    print(f"  cameras:     {cam_json}")
    print(f"\n  STOP — review, then run --step 2 to derive SAM prompt")


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_TAG_RE = re.compile(r"\s*\{(soft|hard)\}\s*$", re.IGNORECASE)


def parse_tagged_prompts(pipe_str: str) -> list[tuple[str, str]]:
    """Split a pipe-union prompt and extract per-term {soft}|{hard} tags.

    Input:  'beige armchair {soft}|wooden chair legs {hard}'
    Output: [('beige armchair', 'soft'), ('wooden chair legs', 'hard')]

    Untagged terms default to 'soft' (more permissive — easier to over-
    include than to lose material). Strips trailing punctuation/whitespace.
    """
    out = []
    for raw in pipe_str.split("|"):
        term = raw.strip()
        if not term:
            continue
        m = _TAG_RE.search(term)
        if m:
            tag = m.group(1).lower()
            text = _TAG_RE.sub("", term).strip()
        else:
            tag = "soft"
            text = term
        if text:
            out.append((text, tag))
    return out


def step2_derive_sam_prompt(scene_dir: Path, obj_dir: Path):
    """Qwen looks at 4 cardinal views of the visual hull and describes
    the object + any sub-items (pillows, throws, blankets, etc).
    Output is a pipe-union SAM prompt saved to sam_prompt.txt.

    Two routes based on object class (2026-05-20):
      - Cabinets/countertops/sideboards/consoles: RICH prompt with part
        decomposition (lamp -> shade+base, etc.) — captures the many
        distinct items on display surfaces.
      - Everything else: V13-STYLE SIMPLE prompt — just main + items on
        top + legs. The rich prompt over-decomposes for tables/sofas
        and pulls neighboring chairs/stools into the pipe-union, which
        pollutes the SAM vote and kills the legs (v25 dining_table)."""
    meta_path = obj_dir / "1_visual_hull_meta.json"
    if not meta_path.exists():
        sys.exit(f"[fatal] missing {meta_path}\n  run extract_one.py first")
    meta = json.load(open(meta_path))
    label = meta.get("label", "object")

    # Class router: use the rich prompt only for display surfaces that
    # actually need part-decomposition. Tables/sofas/chairs use simple.
    CABINET_TOKENS = ("cabinet", "countertop", "sideboard", "console",
                      "buffet", "hutch", "credenza", "shelving unit",
                      "display shelf")
    _ll = (label or "").lower()
    use_rich = any(tok in _ll for tok in CABINET_TOKENS)
    print(f"[step2] prompt route: {'RICH (cabinet/countertop)' if use_rich else 'SIMPLE (v13-style)'}")

    diag = obj_dir / "diagnostics" / "2_sam_wide"
    # Prefer the for-Qwen carved views if step 1 produced them (sibling
    # 02_*/ objects with topdown bbox overlap >25% got subtracted from
    # the hull's projection — Qwen identifies the target without seeing
    # already-extracted neighbors). Falls back to input_*.png when no
    # neighbors qualified (first object / isolated object / old scenes).
    use_qwen_views = (diag / f"input_qwen_{PROMPT_DERIVE_VIEWS[0]}.png").exists()
    prefix = "input_qwen_" if use_qwen_views else "input_"
    print(f"[step2] view source: {prefix}*.png "
          f"({'sibling-bbox subtracted' if use_qwen_views else 'full hull'})")
    images = []
    for tag in PROMPT_DERIVE_VIEWS:
        p = diag / f"{prefix}{tag}.png"
        if p.exists():
            images.append((tag, p))
    if not images:
        # All p-15 cardinals skipped (wall-eye check). Fall back to any
        # views we have with the chosen prefix. Scan by-startswith so the
        # 'input_*' glob doesn't accidentally match 'input_qwen_*'.
        candidates = sorted([
            p for p in diag.iterdir()
            if p.name.startswith(prefix) and
            (p.name.endswith("_p-15.png") or p.name.endswith("_p-45.png"))
        ])
        for p in candidates:
            tag = p.stem[len(prefix):]
            images.append((tag, p))
            if len(images) >= 4:
                break
    if not images:
        sys.exit(f"[fatal] no {prefix}*.png views in {diag}\n  run --step 1 first")

    print(f"[step2] inventory label: '{label}'")
    print(f"[step2] sending {len(images)} cardinal views to Qwen")

    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag, p in images:
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    if use_rich:
        rich_text = (
        f"You are looking at 4 views of a SINGLE piece of furniture.\n\n"
        f"⚠️ THE TARGET OBJECT IS: '{label}'.\n"
        f"This is the ONLY object you should describe. Other objects may be "
        f"visible in the views (a coffee table next to a sofa, a side table "
        f"next to a chair, a lamp on a nearby surface, etc.) — these are "
        f"NEIGHBORS, not parts of the target. DO NOT include neighbors in "
        f"the pipe-union, even if they are clearly visible.\n\n"
        f"What COUNTS as part of the target:\n"
        f"  - The target object itself (the '{label}')\n"
        f"  - Items SITTING ON it (pillows on a sofa, books on a table, "
        f"speakers on a cabinet)\n"
        f"  - Items DRAPED OVER it (throw blanket on a sofa)\n"
        f"  - Items ATTACHED to it (legs, hardware, frame)\n\n"
        f"What DOES NOT count (treat as neighbor, exclude):\n"
        f"  - Furniture next to it (coffee table next to sofa, chair next to table)\n"
        f"  - Lamps standing on the floor or on a different surface\n"
        f"  - Objects on the floor near it\n"
        f"  - Walls, floor, rug behind/under it\n\n"
        f"Build a pipe-union SAM3 prompt with THREE categories, in order:\n\n"
        f"1. MAIN — the target object's upholstered/solid mass. The starting "
        f"label is '{label}'. You SHOULD refine this name if you can describe "
        f"the target object more accurately (e.g. 'green tufted chesterfield "
        f"sofa' instead of just 'green sofa', 'round walnut coffee table' "
        f"instead of 'wooden coffee table'). The refined name should still "
        f"refer to the SAME physical object — don't substitute a neighbor.\n\n"
        f"2. SUB-ITEMS — items resting ON / draped OVER / attached TO the "
        f"target only. Pillows, throws, blankets, cushions, books, decor "
        f"sitting on the target's top surface. If nothing on top, skip.\n"
        f"   IMPORTANT — break each COMPOUND sub-item into its visible "
        f"PARTS, each as its OWN pipe-union term, so SAM keeps all of it, "
        f"not just one part. Examples of decomposing:\n"
        f"     - a table lamp -> 'table lamp' AND 'lamp shade' AND "
        f"'lamp base' (list all, not just 'lamp')\n"
        f"     - a potted plant -> 'potted plant' AND 'flowers' / "
        f"'dried grass' AND 'flower pot' / 'vase' / 'planter'\n"
        f"     - a framed photo -> 'picture frame' AND 'photograph'\n"
        f"   When unsure how an item decomposes, list the whole item AND "
        f"every part you can name — more terms is safer, SAM picks the "
        f"ones that match. Never collapse a multi-part item to one term.\n\n"
        f"3. SUPPORTS (MANDATORY if visible) — the target's OWN structural "
        f"pieces: its legs, base, frame, pedestal, dowels, spindles, "
        f"casters, plinth. Not legs of a neighboring table. Name them as "
        f"their own pipe-union entry — DO NOT lump them into the main "
        f"object name. SAM3 needs these as separate prompts to keep thin "
        f"supports.\n"
        f"   - If supports are clearly visible: name them.\n"
        f"   - If supports are hidden by a skirt: skip.\n\n"
        f"After EACH term in the pipe-union, append a class tag in curly "
        f"braces: '{{soft}}' or '{{hard}}'.\n"
        f"  - {{soft}} = upholstered/fabric/diffuse-edge material\n"
        f"  - {{hard}} = rigid support / hard surface with crisp edges\n\n"
        f"Output a pipe-union string in this exact format:\n"
        f'<main> {{tag}}|<sub-item 1> {{tag}}|...|<support 1> {{tag}}|...\n\n'
        f"Examples:\n"
        f'- target=beige armchair: beige armchair {{soft}}|striped pillow {{soft}}|brown throw blanket {{soft}}|wooden chair legs {{hard}}\n'
        f'- target=wooden coffee table: wooden coffee table top {{hard}}|thin wooden table legs {{hard}}\n'
        f'- target=green sofa: green sofa {{soft}}|orange pillow {{soft}}|gray throw {{soft}}\n'
        f'- target=wooden sideboard with a lamp + plant on top: wooden sideboard {{hard}}|table lamp {{hard}}|lamp shade {{soft}}|lamp base {{hard}}|potted plant {{hard}}|flower pot {{hard}}|picture frame {{hard}}|wooden sideboard legs {{hard}}\n'
        f'- target=wooden tv stand: wooden tv stand {{hard}}|wooden tv stand legs {{hard}}\n\n'
        f"In the last example, even if a TV and speakers are visible ON "
        f"the cabinet, they would only be included if you judge they are "
        f"physically resting on the target's top surface (which speakers "
        f"and a TV typically are — include them). A coffee table NEXT TO "
        f"the cabinet would NOT be included.\n\n"
        f"Output ONLY the pipe-union string with tags. No commentary, no "
        f"markdown, no quotes, no JSON, no explanation.")
        content.append({"type": "text", "text": rich_text})
    else:
        # SIMPLE v13-style: short prompt, no part-decomposition,
        # no neighbor-collection language. Just main + items on top
        # + legs. Matches what v13's sam_carve produced for tables/
        # sofas/chairs and worked end-to-end.
        simple_text = (
            f"You are looking at 4 views of a SINGLE piece of furniture.\n\n"
            f"⚠️ THE TARGET OBJECT IS: '{label}'.\n"
            f"This is the ONLY object to describe. Other furniture nearby "
            f"(chairs around a table, side table near a sofa, etc.) is NOT "
            f"part of the target — EXCLUDE them entirely from the prompt.\n\n"
            f"Build a short pipe-union SAM3 prompt with up to THREE kinds "
            f"of terms:\n"
            f"  1. The main object (refine the name if you can: 'tufted "
            f"chesterfield sofa' instead of 'sofa').\n"
            f"  2. SMALL ITEMS resting on its top surface (book on table, "
            f"pillow on sofa, cup on table). Name each as a single term; "
            f"DO NOT decompose into parts. If nothing on top, skip.\n"
            f"  3. The target's OWN structural pieces (its legs, base, "
            f"pedestal) — ONLY if clearly visible and not hidden by a "
            f"skirt. NOT legs of a neighboring chair.\n\n"
            f"Strict exclusions: NO chairs around a dining table, NO "
            f"cushions on those chairs, NO floor lamps next to a sofa, "
            f"NO neighboring furniture. If you see a chair around the "
            f"target table, the chair is NOT part of the target.\n\n"
            f"Append a class tag to each term: '{{soft}}' for "
            f"upholstered/fabric, '{{hard}}' for rigid.\n\n"
            f"Output format:\n"
            f"<main> {{tag}}|<item 1> {{tag}}|...|<legs> {{tag}}\n\n"
            f"Examples (v13-style — short, no neighbors, no decomposition):\n"
            f'- target=light wood dining table: light wood dining table {{hard}}|small potted plant {{hard}}|bowl of fruit {{hard}}|wooden mug {{hard}}|wooden table legs {{hard}}\n'
            f'- target=wooden coffee table: wooden coffee table {{hard}}|small decorative items on table {{hard}}|wooden table legs {{hard}}\n'
            f'- target=grey armchair: grey armchair {{soft}}|striped pillow {{soft}}|wooden chair legs {{hard}}\n'
            f'- target=green sofa: green sofa {{soft}}|orange pillow {{soft}}|gray throw {{soft}}\n\n'
            f"Output ONLY the pipe-union string with tags. No commentary, "
            f"no markdown, no quotes, no JSON, no explanation."
        )
        content.append({"type": "text", "text": simple_text})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=200, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    # Clean: strip markdown fences, surrounding quotes
    prompt = raw.strip()
    if prompt.startswith("```"):
        lines = [l for l in prompt.split("\n") if not l.startswith("```")]
        prompt = "\n".join(lines).strip()
    prompt = prompt.strip('"').strip("'").strip("`").strip()
    # If Qwen wrote a multi-line response, take first non-empty line with a |
    for line in prompt.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if "|" in line or line:
            prompt = line
            break

    out_path = diag / "sam_prompt.txt"
    out_path.write_text(prompt + "\n")
    raw_path = diag / "sam_prompt_raw.txt"
    raw_path.write_text(raw)

    print(f"\n[qwen raw] {raw}")
    print(f"\n[step2] DONE")
    print(f"  prompt: {prompt}")
    print(f"  saved:  {out_path}")
    print(f"\n  STOP — review the prompt, then run --step 3 to SAM each view")


# ---------- SAM3 (copied from plugin sam3_segment.py — no plugin import) ----------
import torch  # noqa: E402
torch.backends.cudnn.enabled = False  # vLLM/cuDNN coexistence
from transformers import Sam3Model, Sam3Processor  # noqa: E402

_SAM_PROC = None
_SAM_MODEL = None
_SAM_DEVICE = None


def _sam_load():
    global _SAM_PROC, _SAM_MODEL, _SAM_DEVICE
    if _SAM_MODEL is None:
        _SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[sam] loading facebook/sam3 on {_SAM_DEVICE}...")
        _SAM_PROC = Sam3Processor.from_pretrained("facebook/sam3")
        _SAM_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(_SAM_DEVICE).eval()
    return _SAM_PROC, _SAM_MODEL, _SAM_DEVICE


def sam_segment(image_path, prompt, threshold=SAM_THRESHOLD):
    """OR-combined instance mask + scores for the prompt."""
    proc, model, device = _sam_load()
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    results = proc.post_process_instance_segmentation(
        out, threshold=threshold, target_sizes=[(img.height, img.width)]
    )[0]
    scores = results.get("scores", torch.tensor([]))
    masks = results.get("masks", torch.tensor([]))
    if len(masks) == 0:
        return np.zeros((img.height, img.width), dtype=np.uint8), []
    m = (masks.cpu().numpy() > 0).any(axis=0).astype(np.uint8) * 255
    return m, scores.cpu().tolist()


def dilate_mask(mask, radius_px):
    if radius_px <= 0:
        return mask
    from scipy.ndimage import binary_dilation
    r = int(round(radius_px))
    if r < 1:
        return mask
    if r <= 50:
        ys, xs = np.ogrid[-r:r+1, -r:r+1]
        disk = (xs**2 + ys**2) <= r**2
        return binary_dilation(mask > 0, structure=disk).astype(np.uint8) * 255
    iters = max(1, int(round(r / 1.4)))
    return binary_dilation(mask > 0, iterations=iters).astype(np.uint8) * 255


def morph_clean(mask, r=5):
    from scipy.ndimage import binary_closing, binary_fill_holes
    bin_mask = mask > 0
    ys, xs = np.ogrid[-r:r+1, -r:r+1]
    disk = (xs**2 + ys**2) <= r**2
    return (binary_fill_holes(binary_closing(bin_mask, structure=disk))
            .astype(np.uint8) * 255)


def qwen_reprompt(views, attempts_history, label):
    """Ask Qwen to refine a SAM prompt that didn't latch."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag, p in views:
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    history_lines = []
    for i, attempt in enumerate(attempts_history, 1):
        prompt = attempt["prompt"]
        hits = ", ".join(f"'{k}'={v}" for k, v in attempt["per_prompt_hits"].items())
        history_lines.append(f"  Attempt {i}: \"{prompt}\" → hits: {hits}")
    history = "\n".join(history_lines)
    content.append({"type": "text", "text":
        f"You are looking at 4 views of a single piece of furniture. "
        f"The inventory label is: '{label}'.\n\n"
        f"Previous SAM3 prompts you suggested did not latch reliably "
        f"(out of 25 views, the main prompt should hit at least 3):\n"
        f"{history}\n\n"
        f"Refine the pipe-union prompt. Things to try:\n"
        f"  - Different color descriptors (e.g. 'cream' vs 'beige', 'tan' vs 'brown')\n"
        f"  - More common object names ('lounge chair' vs 'armchair')\n"
        f"  - Drop sub-items if they weren't matching\n"
        f"  - Simpler / more generic main term\n\n"
        f"Append a class tag '{{soft}}' or '{{hard}}' AFTER EACH term:\n"
        f"  - {{soft}} = upholstered/fabric/diffuse-edge material\n"
        f"  - {{hard}} = rigid support / hard surface with crisp edges\n"
        f"Example: 'beige lounge chair {{soft}}|wooden chair legs {{hard}}'\n\n"
        f"Output ONLY the new pipe-union string with tags. No commentary, "
        f"no markdown, no quotes, no JSON, no explanation."})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=200, temperature=0.2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    # Same cleanup as step 2
    prompt = raw.strip()
    if prompt.startswith("```"):
        lines = [l for l in prompt.split("\n") if not l.startswith("```")]
        prompt = "\n".join(lines).strip()
    prompt = prompt.strip('"').strip("'").strip("`").strip()
    for line in prompt.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if "|" in line or line:
            prompt = line
            break
    return prompt, raw


def _sam_pass_on_views(cam_data, prompts, sam_pad):
    """Run SAM on every view with each prompt in pipe-union; per-prompt
    skip-if-empty; union via np.maximum; per-view skip if <MIN_VIEW_PX.
    Returns (masks_info[], per_prompt_hits{}).
    """
    masks_info = []
    per_prompt_hits = {p: 0 for p in prompts}
    for cam in cam_data["cameras"]:
        tag = cam["tag"]
        img_path = Path(cam["png"])
        K = np.array(cam["K"])
        eye = np.array(cam["eye"])
        target = np.array(cam["target"])
        depth = float(np.linalg.norm(eye - target))

        mask = None
        scores_pp = {}
        for pr in prompts:
            m, s = sam_segment(img_path, pr)
            scores_pp[pr] = [round(x, 3) for x in s]
            if not s or m is None or (m > 0).sum() < MIN_PROMPT_PX:
                continue
            per_prompt_hits[pr] += 1
            mask = m if mask is None else np.maximum(mask, m)

        if mask is None or (mask > 0).sum() < MIN_VIEW_PX:
            print(f"  [{tag}] skip (no usable mask) scores={scores_pp}")
            continue
        mask = morph_clean(mask)
        rpx = sam_pad * float(K[0, 0]) / max(depth, 0.1)
        mask_d = dilate_mask(mask, rpx)
        masks_info.append({
            "tag": tag,
            "mask": mask, "mask_d": mask_d,
            "n_pixels": int((mask > 0).sum()),
            "n_pixels_dilated": int((mask_d > 0).sum()),
            "scores_per_prompt": scores_pp,
            "depth": depth, "dilation_px": float(rpx),
            "V": cam["V"], "K": cam["K"],
        })
        print(f"  [{tag}] mask_px={int((mask>0).sum()):,} "
              f"dilated={int((mask_d>0).sum()):,} dilate_r={rpx:.0f}px")
    return masks_info, per_prompt_hits


def step3_sam_each_view(scene_dir, obj_dir):
    diag = obj_dir / "diagnostics" / "2_sam_wide"
    prompt_path = diag / "sam_prompt.txt"
    if not prompt_path.exists():
        sys.exit(f"[fatal] missing {prompt_path}\n  run --step 2 first")
    cam_json_path = diag / "cameras.json"
    if not cam_json_path.exists():
        sys.exit(f"[fatal] missing {cam_json_path}\n  run --step 1 first")
    cam_data = json.load(open(cam_json_path))
    n_views = len(cam_data["cameras"])

    meta = json.load(open(obj_dir / "1_visual_hull_meta.json"))
    label = meta.get("label", "object")

    derive_views = []
    for tag in PROMPT_DERIVE_VIEWS:
        p = diag / f"input_{tag}.png"
        if p.exists():
            derive_views.append((tag, p))

    # Clear stale per-view masks from prior runs
    for f in diag.glob("mask_*.png"):
        f.unlink()

    attempts_history = []
    best_attempt = None
    accepted = False
    current_prompt = prompt_path.read_text().strip()

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[step3] === ATTEMPT {attempt_num}/{MAX_ATTEMPTS} ===")
        print(f"[step3] prompt: {current_prompt}")
        # Strip {soft}/{hard} tags before sending to SAM (tags are for
        # downstream pad selection — sam_carve uses uniform SAM_PAD_M).
        tagged = parse_tagged_prompts(current_prompt)
        prompts = [t for t, _tag in tagged]
        main_prompt = prompts[0]

        # Clear masks from prior attempt within this step run
        for f in diag.glob("mask_*.png"):
            f.unlink()

        masks_info, per_prompt_hits = _sam_pass_on_views(
            cam_data, prompts, SAM_PAD_M)

        main_hits = per_prompt_hits[main_prompt]
        view_hits = len(masks_info)
        print(f"\n[step3] attempt {attempt_num}: views_with_mask={view_hits}/{n_views}, "
              f"main='{main_prompt}' hit {main_hits}")

        attempt_record = {
            "attempt": attempt_num,
            "prompt": current_prompt,
            "prompts": prompts,
            "per_prompt_hits": per_prompt_hits,
            "n_views_with_mask": view_hits,
            "main_prompt": main_prompt,
            "main_hits": main_hits,
        }
        attempts_history.append(attempt_record)

        # Track best by main_hits (tie-break: total view_hits)
        if best_attempt is None or (
            (main_hits, view_hits) >
            (best_attempt["main_hits"], best_attempt["n_views_with_mask"])
        ):
            best_attempt = attempt_record
            best_masks_info = masks_info

        if main_hits >= MIN_HIT_VIEWS:
            print(f"[step3] PASS — main prompt hit ≥{MIN_HIT_VIEWS} views")
            accepted = True
            best_attempt = attempt_record
            best_masks_info = masks_info
            break

        if attempt_num < MAX_ATTEMPTS:
            print(f"[step3] only {main_hits} main hits — asking Qwen to refine...")
            current_prompt, raw = qwen_reprompt(derive_views, attempts_history, label)
            print(f"[qwen retry raw] {raw}")
            print(f"[qwen retry] new prompt: {current_prompt}")
            (diag / f"sam_prompt_attempt{attempt_num+1}.txt").write_text(
                current_prompt + "\n")

    if not accepted:
        print(f"\n[step3] no attempt hit ≥{MIN_HIT_VIEWS} — accepting best "
              f"(attempt {best_attempt['attempt']}, main_hits="
              f"{best_attempt['main_hits']})")
        print(f"[step3] re-running best prompt to regen masks: "
              f"'{best_attempt['prompt']}'")
        for f in diag.glob("mask_*.png"):
            f.unlink()
        prompts = [t for t, _tag in parse_tagged_prompts(best_attempt["prompt"])]
        best_masks_info, _ = _sam_pass_on_views(cam_data, prompts, SAM_PAD_M)

    # Persist masks to diagnostics/
    for mv in best_masks_info:
        Image.fromarray(mv["mask"], mode="L").save(diag / f"mask_{mv['tag']}.png")
        Image.fromarray(mv["mask_d"], mode="L").save(diag / f"mask_padded_{mv['tag']}.png")

    # Promote final accepted prompt to canonical sam_prompt.txt
    (diag / "sam_prompt.txt").write_text(best_attempt["prompt"] + "\n")

    history_json = diag / "sam_prompt_history.json"
    history_json.write_text(json.dumps({
        "final_prompt": best_attempt["prompt"],
        "accepted_at_attempt": best_attempt["attempt"],
        "accepted_via_health_check": accepted,
        "min_hit_views": MIN_HIT_VIEWS,
        "max_attempts": MAX_ATTEMPTS,
        "attempts": attempts_history,
    }, indent=2))

    report_json = diag / "report.json"
    report_json.write_text(json.dumps({
        "stage": "stage2_sam_carve",
        "final_prompt": best_attempt["prompt"],
        "n_views_total": n_views,
        "n_views_with_mask": best_attempt["n_views_with_mask"],
        "per_prompt_hits": best_attempt["per_prompt_hits"],
        "sam_pad_m": SAM_PAD_M,
        "sam_threshold": SAM_THRESHOLD,
        "min_prompt_px": MIN_PROMPT_PX,
        "min_view_px": MIN_VIEW_PX,
        "views": [
            {"tag": mv["tag"], "n_pixels": mv["n_pixels"],
             "n_pixels_dilated": mv["n_pixels_dilated"],
             "scores_per_prompt": mv["scores_per_prompt"],
             "depth": mv["depth"], "dilation_px": mv["dilation_px"]}
            for mv in best_masks_info
        ],
    }, indent=2))

    print(f"\n[step3] DONE")
    print(f"  final prompt: {best_attempt['prompt']}")
    print(f"  attempts:     {len(attempts_history)}")
    print(f"  views w/mask: {best_attempt['n_views_with_mask']}/{n_views}")
    print(f"  diagnostics:  {diag}")
    print(f"  history:      {history_json}")
    print(f"  report:       {report_json}")
    print(f"\n  STOP — review masks, then run --step 4 to vote")


def render_canonical_5(ply_path: Path, out_dir: Path):
    """Render 5 canonical 1080p views: y0/y90/y180/y270 at pitch -20 + topdown.

    LOCKED-CAMERA RULE: if `<obj>/display_cameras.json` exists (written
    by floor_drop), this function uses those EXACT cameras instead of
    recomputing framing from the current PLY. This guarantees later
    stages (sam_tight, bookshelf_sweep, sweep_fallback) produce
    pixel-comparable canonical_5 renders to floor_drop's.

    Otherwise (1_visual_hull and 2_sam_wide renders, which run before
    floor_drop) the framing is computed from p2/p98 extents per PLY.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    scene = load_gsplat_ply(str(ply_path))
    means = scene["means"].detach().cpu().numpy()
    if len(means) == 0:
        print(f"[render] {ply_path} is empty — skipping")
        return

    # Try locked display cameras first. out_dir is <obj>/renders/<stage>.
    obj_dir = out_dir.parent.parent
    locked_path = obj_dir / "display_cameras.json"
    if locked_path.exists():
        locked = json.load(open(locked_path))
        for cam in locked["cameras"]:
            V = np.array(cam["V"], dtype=np.float32)
            K = np.array(cam["K"], dtype=np.float32)
            img = render_splat(scene, V, K, locked["width"], locked["height"],
                                bg=(1.0, 1.0, 1.0))
            Image.fromarray(img).save(out_dir / f"{cam['tag']}.png")
        return

    # No locked cameras yet — compute per-PLY framing.
    lo = np.percentile(means, 2, axis=0)
    hi = np.percentile(means, 98, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = max(float((hi - lo).max()), 0.15)
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * 1.55) / (2 * tan_half) + 0.5  # +0.5m push-back
    for yaw_deg in CANONICAL_YAWS:
        V, K, _ = build_camera(center, yaw_deg, CANONICAL_PITCH, distance,
                                FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
        img = render_splat(scene, V, K, CANONICAL_W, CANONICAL_H,
                            bg=(1.0, 1.0, 1.0))
        Image.fromarray(img).save(out_dir / f"y{int(yaw_deg)}.png")
    V, K, _ = build_camera(center, 0, CANONICAL_TOPDOWN_PITCH, distance,
                            FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
    img = render_splat(scene, V, K, CANONICAL_W, CANONICAL_H,
                        bg=(1.0, 1.0, 1.0))
    Image.fromarray(img).save(out_dir / "topdown.png")


def step4_vote_carve(scene_dir: Path, obj_dir: Path):
    """Project visual_hull splats through every SAM-survived camera; keep
    splats voted in by ≥MIN_VIEWS_FRAC of views (and visible in ≥same).
    Writes sam_wide.ply at root + renders/sam_wide/."""
    diag = obj_dir / "diagnostics" / "2_sam_wide"
    cam_json_path = diag / "cameras.json"
    if not cam_json_path.exists():
        sys.exit(f"[fatal] missing {cam_json_path}\n  run --step 1 first")
    cam_data = json.load(open(cam_json_path))

    hull_ply = obj_dir / "1_visual_hull.ply"
    if not hull_ply.exists():
        sys.exit(f"[fatal] missing {hull_ply}")
    pl = PlyData.read(str(hull_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[step4] visual_hull: {len(xyz):,} splats")

    masks_info = []
    for cam in cam_data["cameras"]:
        tag = cam["tag"]
        mp_path = diag / f"mask_padded_{tag}.png"
        if not mp_path.exists():
            continue  # view didn't survive SAM (empty mask)
        mask_d = np.array(Image.open(mp_path).convert("L"))
        masks_info.append({
            "tag": tag,
            "V": np.array(cam["V"], dtype=np.float64),
            "K": np.array(cam["K"], dtype=np.float64),
            "mask_d": mask_d,
            "W": int(cam["width"]),
            "H": int(cam["height"]),
        })
    n_views = len(masks_info)
    print(f"[step4] {n_views} views with SAM padded masks "
          f"(of {len(cam_data['cameras'])} total)")
    if n_views == 0:
        sys.exit("[fatal] no SAM masks found — run --step 3 first")

    hp = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    votes = np.zeros(len(xyz), dtype=np.int32)
    valid = np.zeros(len(xyz), dtype=np.int32)
    for mv in masks_info:
        V, K, mask_d = mv["V"], mv["K"], mv["mask_d"]
        W, H = mv["W"], mv["H"]
        cam_xyz = (hp @ V.T)[:, :3]
        zc = -cam_xyz[:, 2]
        in_front = zc > 0.01
        xs = K[0, 0] * cam_xyz[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]
        ys = K[1, 1] * cam_xyz[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
        xi = xs.astype(np.int32)
        yi = ys.astype(np.int32)
        in_img = in_front & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        good = np.where(in_img)[0]
        valid[good] += 1
        vals = mask_d[yi[good].clip(0, H - 1), xi[good].clip(0, W - 1)]
        votes[good[vals > 0]] += 1

    required = int(math.ceil(MIN_VIEWS_FRAC * n_views))
    keep = (valid >= required) & (votes >= required)
    n_kept = int(keep.sum())
    print(f"[step4] required ≥{required}/{n_views} votes "
          f"(min_views_frac={MIN_VIEWS_FRAC})")
    print(f"[step4] kept {n_kept:,} / {len(xyz):,}")

    out_ply = obj_dir / "2_sam_wide.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[step4] wrote {out_ply}")

    render_dir = obj_dir / "renders" / "2_sam_wide"
    render_canonical_5(out_ply, render_dir)
    print(f"[step4] rendered 5 canonical views → {render_dir}")

    report_path = diag / "report.json"
    report = json.load(open(report_path)) if report_path.exists() else {}
    report["vote"] = {
        "min_views_frac": MIN_VIEWS_FRAC,
        "n_views": n_views,
        "required": required,
        "n_kept": n_kept,
        "n_total": len(xyz),
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n[step4] DONE — sam_wide stage complete")
    print(f"  PLY:     {out_ply}")
    print(f"  renders: {render_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path,
                    help="path to 02_<slug>/ (must contain visual_hull.ply)")
    ap.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4])
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()
    if args.step == 1:
        step1_render_views(scene, obj)
    elif args.step == 2:
        step2_derive_sam_prompt(scene, obj)
    elif args.step == 3:
        step3_sam_each_view(scene, obj)
    elif args.step == 4:
        step4_vote_carve(scene, obj)


if __name__ == "__main__":
    main()
