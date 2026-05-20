#!/usr/bin/env python3
"""sam_tight.py — Stage 4 of the per-object pipeline.

Second multi-view SAM pass with TIGHT padding on the floor-cleaned PLY.
Where sam_wide was conservative (sam_pad 0.2m, vote 0.6) to confirm
"this is the object", sam_tight is precise (sam_pad 0.05m, vote 0.8) to
trim every halo splat that sam_wide and floor_drop left behind.

Pipeline position (locked):
  visual_hull (Stage 1)
  → sam_wide (Stage 2)
  → floor_drop (Stage 3)
  → sam_tight (Stage 4, this script)
  → aabb_filter (Stage 5)
  → floor_band (Stage 6)
  → export (Stage 7)

Reads:
  <scene>/02_<slug>/floor_drop.ply
  <scene>/02_<slug>/diagnostics/sam_wide/sam_prompt.txt
    (same Qwen-derived pipe-union prompt — same chair, same prompt)

Writes:
  <scene>/02_<slug>/sam_tight.ply
  <scene>/02_<slug>/renders/sam_tight/{y0,y90,y180,y270,topdown}.png
  <scene>/02_<slug>/diagnostics/sam_tight/
    input_<tag>.png × 25       (re-rendered from floor_drop.ply)
    cameras.json
    mask_<tag>.png × N         (raw post-morph SAM masks)
    mask_padded_<tag>.png × N  (dilated by SAM_PAD_M)
    report.json                (per-view scores + vote stats)

Differences vs sam_wide:
  - Input PLY: floor_drop.ply (not visual_hull.ply)
  - SAM_PAD_M: 0.05 (was 0.2) — tight, edge-precision dilation
  - MIN_VIEWS_FRAC: 0.8 (was 0.6) — stricter vote, drops halo splats
  - No Qwen prompt-derivation (reuses sam_wide's prompt)
  - No retry loop (single-shot SAM)

Reuses sam_carve.py for the heavy lifting (camera math, render, SAM,
morph cleanup, dilation, projection). This script is the driver.

Usage:
    python sam_tight.py <scene_dir> 02_<slug>/
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
from openai import OpenAI
from PIL import Image
from plyfile import PlyData, PlyElement

# Skill — OK to import.
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402

# iteration_1 siblings — reuse without editing.
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from extract_one import viewmat_look_at, build_K, RENDER_MARGIN  # noqa: E402
from sam_carve import (  # noqa: E402
    YAWS_DEG, PITCHES_DEG, TOPDOWN_PITCH,
    FOV, W, H, Y_DOWN,
    build_camera, render_canonical_5,
    sam_segment, dilate_mask, morph_clean,
    MIN_PROMPT_PX, MIN_VIEW_PX, SAM_THRESHOLD,
    parse_tagged_prompts, compute_wall_skip, get_wall_skip_callable,
)

# TIGHT-pass parameters (override sam_wide defaults)
SAM_PAD_HARD_M = 0.035      # tight: legs, frames, hard surfaces
SAM_PAD_FABRIC_M = 0.10     # wider: upholstery, pillows, blankets — soft edges spread

# Per-view Qwen-bbox crop. Constrains SAM to ONLY look at pixels inside
# the parent object's bbox (with an asymmetric top pad to keep
# items-on-top inside the crop). Prevents SAM from latching onto
# neighboring furniture / wall / decor outside the bbox.
QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"
CROP_TOP_PAD_PCT = 0.12      # fallback upward pad on y_min (catch plant on top)
CROP_TOP_PAD_M = 2.0         # world-space upward headroom — crop top clamps to
                             # the image edge so tall on-top items (lamp) are
                             # never clipped out of the crop before SAM sees them
CROP_SIDE_PAD_PCT = 0.03     # left / right / bottom pad (small, just slack)


def _qwen_encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def qwen_view_bbox(img_path: Path, label: str):
    """Ask Qwen for a tight pixel bbox of `label` in the given view.
    Returns (x0, y0, x1, y1) in normalized 0-1000 space, or None."""
    prompt = (
        f"This is a single view of a single piece of furniture and its "
        f"surroundings.\n\n"
        f"TASK: return a bounding box that contains THE {label.upper()} "
        f"AND EVERYTHING resting ON TOP OF it or inside it — lamps and "
        f"lamp shades, vases, plants, bowls, books, picture frames: every "
        f"object sitting on it. The box's TOP edge MUST be ABOVE the "
        f"highest point of the tallest item on top — do NOT draw the box "
        f"tight to the furniture, it must enclose the whole stack. "
        f"Exclude ONLY neighboring furniture, walls, paintings on the "
        f"wall, and items on the floor.\n\n"
        f"If the {label} is not visible in this view, return "
        f'{{"found": false}}.\n\n'
        f'Otherwise return: {{"found": true, "bbox_2d": [x0, y0, x1, y1]}}\n'
        f"Coordinates in 0-1000 normalized space, integers, x0<x1, y0<y1.\n"
        f"Output ONLY the JSON object."
    )
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    try:
        r = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_qwen_encode_b64(img_path)}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=120, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [qwen-bbox] exception: {e}")
        return None
    s = raw.find("{")
    e = raw.rfind("}") + 1
    if s == -1 or e <= s:
        return None
    try:
        data = json.loads(raw[s:e])
    except Exception:
        return None
    if not data.get("found"):
        return None
    bbox = data.get("bbox_2d") or data.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return tuple(int(c) for c in bbox)


def crop_for_sam(img_path: Path, bbox_norm, W_img: int, H_img: int,
                  out_path: Path, top_extra_px=None):
    """Crop `img_path` to bbox + asymmetric pads, save to `out_path`.
    Returns (crop_x0, crop_y0, crop_w, crop_h) so masks can be mapped
    back to full image coords.

    top_extra_px: explicit upward headroom in pixels (caller converts
    CROP_TOP_PAD_M world-metres -> pixels). If None, falls back to the
    bbox-relative CROP_TOP_PAD_PCT."""
    x0, y0, x1, y1 = bbox_norm
    x0 = int(x0 * W_img / 1000)
    y0 = int(y0 * H_img / 1000)
    x1 = int(x1 * W_img / 1000)
    y1 = int(y1 * H_img / 1000)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    top_extra = (int(top_extra_px) if top_extra_px is not None
                 else int(bh * CROP_TOP_PAD_PCT))
    side_pad = int(max(bw, bh) * CROP_SIDE_PAD_PCT)
    cx0 = max(0, x0 - side_pad)
    cy0 = max(0, y0 - top_extra - side_pad)
    cx1 = min(W_img, x1 + side_pad)
    cy1 = min(H_img, y1 + side_pad)
    cw, ch = cx1 - cx0, cy1 - cy0
    Image.open(img_path).crop((cx0, cy0, cx1, cy1)).save(out_path)
    return cx0, cy0, cw, ch
MIN_VIEWS_FRAC = 0.7        # was 0.8 — too strict, killed bodies

def render_25_views(in_ply: Path, diag: Path, scene_dir: Path = None,
                    obj_dir: Path = None, pitches: list = None):
    """Render the SAM views from in_ply into diag/input_<tag>.png +
    save cameras.json. Mirrors sam_carve.step1_render_views but takes
    an input PLY argument. If scene_dir is provided, applies the same
    wall-side camera skip as sam_carve step 1 (skip cameras whose eye
    sits on the wall-side of the hull's back face).

    `pitches` selects the pitch ring set:
      - None / PITCHES_DEG  → Pass A high cameras (-15/-45) + topdown.
      - [0.0, 15.0]         → Pass B low cameras (level / looking up).
    The two passes are run separately (sam_tight = A, sam_low_refine = B)
    so the low cameras never enter Pass A's vote-carve."""
    diag.mkdir(parents=True, exist_ok=True)
    for f in diag.glob("input_*.png"):
        f.unlink()

    print(f"[load] {in_ply}")
    scene = load_gsplat_ply(str(in_ply))
    means = scene["means"].detach().cpu().numpy()
    n_splats = len(means)
    print(f"[load] {n_splats:,} splats")

    lo = means.min(axis=0)
    hi = means.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = float((hi - lo).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m margin={RENDER_MARGIN}")

    # Wall-skip gated on Qwen verdict (2026-05-20 v2): the blanket disable
    # from earlier today over-corrected — it fixed wall-adjacent tables
    # (thin legs surviving) but broke wall-adjacent cabinets/sideboards
    # (back face flush against wall, backside views see wall-through-body
    # and vote drops the cabinet body). get_wall_skip_callable reads
    # wall_adjacent.json (written by check_wall_adjacent_via_qwen at the
    # 1_visual_hull stage) — true → real wall-skip, false → no-op.
    _, _, eye_behind_object = get_wall_skip_callable(scene_dir, obj_dir, means)

    # Extend YAWS_DEG with ±5° offsets around y0 (= y355 and y5) so
    # SAM/Qwen also see slightly off-axis front views — helpful when the
    # canonical y0 view catches wall/perpendicular-wall material from a
    # corner-adjacent object.
    sam_tight_yaws = list(YAWS_DEG) + [355.0, 5.0]

    # Pitch set is caller-selected (Pass A high vs Pass B low). Pass A is
    # the -15/-45 above-object rings; Pass B (sam_low_refine.py) supplies
    # [0, 15] low rings separately. The topdown camera is only added for
    # the high pass (Pass A) — a topdown is meaningless for the low pass.
    sam_tight_pitches = list(pitches) if pitches is not None else list(PITCHES_DEG)
    is_low_pass = pitches is not None and all(p >= 0 for p in pitches)

    cameras = []
    for pitch_deg in sam_tight_pitches:
        ptag = f"p{int(round(pitch_deg))}"
        for yaw_deg in sam_tight_yaws:
            ytag = f"y{int(round(yaw_deg))}"
            tag = f"{ytag}_{ptag}"
            V, K, eye = build_camera(center, yaw_deg, pitch_deg, distance,
                                      FOV, W, H, y_down=Y_DOWN)
            if eye_behind_object(eye):
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
    if not is_low_pass:
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
    cam_json = diag / "cameras.json"
    cam_json.write_text(json.dumps({
        "ply_path": str(in_ply),
        "n_splats": n_splats,
        "fov": FOV, "width": W, "height": H,
        "y_down": Y_DOWN,
        "yaws_deg": YAWS_DEG,
        "pitches_deg": sam_tight_pitches,
        "topdown_pitch_deg": TOPDOWN_PITCH,
        "center": center.tolist(),
        "extent": extent,
        "distance": distance,
        "cameras": cameras,
    }, indent=2))
    print(f"[render] {len(cameras)} views → {diag}")


def sam_each_view(diag: Path, prompts: list, prompt_pads: dict,
                   parent_label: str | None = None,
                   qwen_miss: str = "skip"):
    """Run SAM on each input view. PER-PROMPT dilation — fabric prompts
    get SAM_PAD_FABRIC_M, hard prompts get SAM_PAD_HARD_M. Each prompt's
    mask is dilated separately, then unioned via np.maximum.

    prompt_pads: {prompt_str: pad_meters} mapping.

    If parent_label is provided, the input PNG is first cropped by a
    per-view Qwen bbox of the parent (with asymmetric top pad) before
    being fed to SAM. SAM can only segment pixels inside that crop —
    prevents picking up neighboring furniture / wall material.

    qwen_miss controls what happens when qwen_view_bbox returns None:
      - "skip" (default, sam_tight behavior): drop the view entirely.
        Full-frame SAM on a missed view picks the biggest non-parent
        thing in the scene and pollutes the vote → carves the parent.
        v26 chair (37,051 splats) was made with this behavior.
      - "full_frame" (sam_low_refine behavior): run SAM on the
        uncropped image. Useful for low cameras where Qwen often can't
        find the object but SAM can still produce a usable silhouette.
        The masks feed inside_outside multi-pool, not vote_carve.
    """
    cam_data = json.load(open(diag / "cameras.json"))
    for f in diag.glob("mask_*.png"):
        f.unlink()
    for f in diag.glob("crop_*.png"):
        f.unlink()

    masks_info = []
    per_prompt_hits = {p: 0 for p in prompts}
    for cam in cam_data["cameras"]:
        tag = cam["tag"]
        img_path = Path(cam["png"])
        K = np.array(cam["K"])
        eye = np.array(cam["eye"])
        target = np.array(cam["target"])
        depth = float(np.linalg.norm(eye - target))
        f_px = float(K[0, 0])
        W_img = int(cam["width"])
        H_img = int(cam["height"])

        # Crop to Qwen bbox so SAM only sees parent-object region.
        sam_input_path = img_path
        crop_x0 = crop_y0 = 0
        crop_w, crop_h = W_img, H_img
        if parent_label:
            bbox_norm = qwen_view_bbox(img_path, parent_label)
            if bbox_norm is None:
                if qwen_miss == "full_frame":
                    print(f"  [{tag}] Qwen didn't find '{parent_label}' — "
                          f"using full frame (caller policy)")
                    # leave sam_input_path = img_path, no crop
                else:  # "skip" — sam_tight v26-equivalent behavior
                    print(f"  [{tag}] Qwen didn't find '{parent_label}' — "
                          f"skipping view")
                    continue
            else:
                crop_path = diag / f"crop_{tag}.png"
                top_extra_px = CROP_TOP_PAD_M * f_px / max(depth, 0.1)
                crop_x0, crop_y0, crop_w, crop_h = crop_for_sam(
                    img_path, bbox_norm, W_img, H_img, crop_path,
                    top_extra_px=top_extra_px)
                sam_input_path = crop_path

        # raw_union: unioned undilated mask (for diagnostics)
        # padded_union: per-prompt dilated then unioned
        raw_union = None
        padded_union = None
        scores_pp = {}
        per_prompt_dilation_px = {}
        for pr in prompts:
            m_crop, s = sam_segment(sam_input_path, pr)
            scores_pp[pr] = [round(x, 3) for x in s]
            if not s or m_crop is None or (m_crop > 0).sum() < MIN_PROMPT_PX:
                continue
            per_prompt_hits[pr] += 1
            # Map crop-space mask back to full image coords.
            if parent_label:
                m = np.zeros((H_img, W_img), dtype=m_crop.dtype)
                m[crop_y0:crop_y0 + crop_h,
                  crop_x0:crop_x0 + crop_w] = m_crop[:crop_h, :crop_w]
            else:
                m = m_crop
            # Morph the per-prompt mask first
            m_clean = morph_clean(m)
            # Per-prompt dilation
            pad_m = prompt_pads[pr]
            rpx = pad_m * f_px / max(depth, 0.1)
            per_prompt_dilation_px[pr] = float(rpx)
            m_dilated = dilate_mask(m_clean, rpx)
            raw_union = m_clean if raw_union is None else np.maximum(raw_union, m_clean)
            padded_union = (m_dilated if padded_union is None
                             else np.maximum(padded_union, m_dilated))

        if padded_union is None or (padded_union > 0).sum() < MIN_VIEW_PX:
            print(f"  [{tag}] skip — no usable mask")
            continue
        Image.fromarray(raw_union, mode="L").save(diag / f"mask_{tag}.png")
        Image.fromarray(padded_union, mode="L").save(diag / f"mask_padded_{tag}.png")
        masks_info.append({
            "tag": tag,
            "V": np.array(cam["V"], dtype=np.float64),
            "K": np.array(cam["K"], dtype=np.float64),
            "mask_d": padded_union,
            "W": int(cam["width"]),
            "H": int(cam["height"]),
            "n_pixels_raw": int((raw_union > 0).sum()),
            "n_pixels_padded": int((padded_union > 0).sum()),
            "scores_per_prompt": scores_pp,
            "depth": depth,
            "per_prompt_dilation_px": per_prompt_dilation_px,
        })
        print(f"  [{tag}] raw={int((raw_union>0).sum()):,} "
              f"padded={int((padded_union>0).sum()):,}")
    print(f"[sam] {len(masks_info)} views with usable masks "
          f"(of {len(cam_data['cameras'])} total)")
    return masks_info, per_prompt_hits


def vote_carve(in_ply: Path, masks_info: list, min_views_frac: float):
    """Project in_ply splats through every saved camera; keep splats voted
    in by ≥ceil(min_views_frac × n_views) of views. Returns (keep, n_kept,
    n_in, required, n_views) and the new vertex data."""
    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n_in = len(xyz)
    n_views = len(masks_info)
    if n_views == 0:
        return None, 0, n_in, 0, 0, v

    hp = np.concatenate([xyz, np.ones((n_in, 1))], axis=1)
    votes = np.zeros(n_in, dtype=np.int32)
    valid = np.zeros(n_in, dtype=np.int32)
    for mv in masks_info:
        V, K, mask_d = mv["V"], mv["K"], mv["mask_d"]
        Wv, Hv = mv["W"], mv["H"]
        cam_xyz = (hp @ V.T)[:, :3]
        zc = -cam_xyz[:, 2]
        in_front = zc > 0.01
        xs = K[0, 0] * cam_xyz[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]
        ys = K[1, 1] * cam_xyz[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
        xi = xs.astype(np.int32)
        yi = ys.astype(np.int32)
        in_img = in_front & (xi >= 0) & (xi < Wv) & (yi >= 0) & (yi < Hv)
        good = np.where(in_img)[0]
        valid[good] += 1
        vals = mask_d[yi[good].clip(0, Hv - 1), xi[good].clip(0, Wv - 1)]
        votes[good[vals > 0]] += 1

    required = int(math.ceil(min_views_frac * n_views))
    keep = (valid >= required) & (votes >= required)
    return keep, int(keep.sum()), n_in, required, n_views, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path,
                    help="path to 02_<slug>/ (must contain floor_drop.ply)")
    ap.add_argument("--sam-pad-hard-m", type=float, default=SAM_PAD_HARD_M,
                    help=f"tight pad for hard surfaces (default {SAM_PAD_HARD_M})")
    ap.add_argument("--sam-pad-fabric-m", type=float, default=SAM_PAD_FABRIC_M,
                    help=f"wider pad for fabric/upholstery (default {SAM_PAD_FABRIC_M})")
    ap.add_argument("--min-views-frac", type=float, default=MIN_VIEWS_FRAC,
                    help=f"strict vote threshold (default {MIN_VIEWS_FRAC})")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()

    in_ply = obj / "3_floor_drop.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing {in_ply}\n  run floor_drop.py first")
    prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not prompt_path.exists():
        sys.exit(f"[fatal] missing {prompt_path}\n  run sam_carve.py through step 2 first")
    pipe_prompt = prompt_path.read_text().strip()
    # Parse {soft}/{hard} tags inline. SAM gets clean text; tags pick pad.
    tagged = parse_tagged_prompts(pipe_prompt)
    prompts = [t for t, _tag in tagged]
    prompt_classes = {t: tag for t, tag in tagged}
    prompt_pads = {t: (args.sam_pad_fabric_m if tag == "soft"
                       else args.sam_pad_hard_m)
                   for t, tag in tagged}
    print(f"[prompt] {pipe_prompt}")
    print(f"[prompt] {len(prompts)} terms in pipe-union")
    for pr in prompts:
        print(f"[pad] '{pr}' → {prompt_classes[pr]} pad={prompt_pads[pr]}m")

    diag = obj / "diagnostics" / "4_sam_tight"
    diag.mkdir(parents=True, exist_ok=True)

    # Step A: render 25 views from floor_drop.ply
    print(f"\n[A] rendering 25 views from floor_drop.ply...")
    render_25_views(in_ply, diag, scene_dir=scene, obj_dir=obj)

    # Step B: SAM each view (per-prompt pad). Pass the main prompt
    # (first pipe-union term) as parent_label so SAM only sees the
    # parent-object bbox region.
    print(f"\n[B] SAM with per-prompt pads + per-view Qwen-bbox crop...")
    parent_label = prompts[0] if prompts else None
    masks_info, per_prompt_hits = sam_each_view(
        diag, prompts, prompt_pads, parent_label=parent_label)
    print(f"[sam] per-prompt hits: {per_prompt_hits}")

    if not masks_info:
        sys.exit("[fatal] no usable SAM masks — check the prompt or input PLY")

    # Step C: vote and write sam_tight.ply
    print(f"\n[C] voting at min_views_frac={args.min_views_frac}...")
    keep, n_kept, n_in, required, n_views, v = vote_carve(
        in_ply, masks_info, args.min_views_frac)
    print(f"[vote] required ≥{required}/{n_views} votes")
    print(f"[vote] kept {n_kept:,} / {n_in:,} ({100*n_kept/n_in:.1f}%)")

    out_ply = obj / "4_sam_tight.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    render_dir = obj / "renders" / "4_sam_tight"
    render_canonical_5(out_ply, render_dir)
    print(f"[render] 5 canonical views → {render_dir}")

    (diag / "report.json").write_text(json.dumps({
        "stage": "sam_tight",
        "input_ply": str(in_ply),
        "output_ply": str(out_ply),
        "prompt": pipe_prompt,
        "prompts": prompts,
        "prompt_pads_m": prompt_pads,
        "prompt_classes": prompt_classes,
        "sam_pad_hard_m": args.sam_pad_hard_m,
        "sam_pad_fabric_m": args.sam_pad_fabric_m,
        "min_views_frac": args.min_views_frac,
        "n_views_total": n_views,
        "required_votes": required,
        "n_in": n_in,
        "n_kept": n_kept,
        "per_prompt_hits": per_prompt_hits,
        "views": [
            {"tag": mv["tag"], "n_pixels_raw": mv["n_pixels_raw"],
             "n_pixels_padded": mv["n_pixels_padded"],
             "scores_per_prompt": mv["scores_per_prompt"],
             "depth": mv["depth"],
             "per_prompt_dilation_px": mv["per_prompt_dilation_px"]}
            for mv in masks_info
        ],
    }, indent=2))

    print(f"\n[done]")
    print(f"  PLY:     {out_ply}")
    print(f"  renders: {render_dir}")
    print(f"  report:  {diag / 'report.json'}")


if __name__ == "__main__":
    main()
