#!/usr/bin/env python3
"""floor_drop.py — Stage 3 of the per-object pipeline.

Drops splats within FLOOR_THRESH meters of the saved scene floor plane.
Runs AFTER sam_wide. SAM has already conservatively carved obvious
non-object splats; floor_drop now removes the wood-floor halo that
sam_wide retained because of mask dilation tangent to the chair feet.

Pipeline position (locked):
  visual_hull (Stage 1)
  → sam_wide (Stage 2)
  → floor_drop (Stage 3, this script)
  → sam_tight (Stage 4)
  → aabb_filter (Stage 5)
  → floor_band (Stage 6)
  → export (Stage 7)

Reads:
  <scene>/01_shell_removed/floor_plane.json   (RANSAC plane equation)
  <scene>/02_<slug>/sam_wide.ply

Writes:
  <scene>/02_<slug>/floor_drop.ply
  <scene>/02_<slug>/renders/floor_drop/{y0,y90,y180,y270,topdown}.png
  <scene>/02_<slug>/diagnostics/floor_drop/
    iter{0..N}/{y0,y90,y180,y270,topdown}.png   (per-iteration renders)
    iter{0..N}.ply                                (per-iteration PLYs)
    qwen_iter{N}_raw.txt                          (per-iter Qwen response)
    history.json                                  (all attempts + final pick)
    report.json                                   (final selection stats)

Qwen feedback loop (default ON):
  - ALWAYS runs MAX_ITERATIONS (default 5). No early STOP — Qwen tends
    to rubber-stamp the first attempt and miss residual floor.
  - Iter 0: carve at INITIAL_THRESH (default 0.05m), render, ask Qwen
    to suggest a new threshold to try next.
  - Iter N>0: Qwen sees current iter's renders + history of all prior
    attempts and suggests the next threshold to try.
  - After all iterations: Qwen sees all attempts side-by-side and
    picks the BEST iter.
  - Promote winner to canonical floor_drop.ply + renders/floor_drop/.

Use --no-qwen-loop to disable the loop and run a single carve at
--floor-thresh.

Usage:
    python floor_drop.py <scene_dir> 02_<slug>/
    python floor_drop.py <scene_dir> 02_<slug>/ --no-qwen-loop --floor-thresh 0.15
    python floor_drop.py <scene_dir> 02_<slug>/ --max-iterations 3
"""
import os
import argparse
import base64
import io
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from openai import OpenAI
from plyfile import PlyData, PlyElement

# Skill (gsplat-viewer) — OK to import.
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import (  # noqa: E402
    load_gsplat_ply, render_splat, rotation_matrix_from_yaw_pitch,
)

# pipeline sibling — reuse without editing.
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from extract_one import viewmat_look_at, build_K, RENDER_MARGIN  # noqa: E402

INITIAL_THRESH_M = 0.10   # was 0.05 — start higher so 5 iters cover 0.10-0.35m range
THRESH_MIN = 0.01
THRESH_MAX = 0.50
UPPER_THRESH_M = 0.18     # CAP on how far ABOVE floor we drop (protects chair body/legs).
                           # Qwen-tuned thresh applies only BELOW floor.
MAX_ITERATIONS = 5
QWEN_VIEWS = ["topdown", "y180", "y0", "y90"]   # views Qwen sees per iter
QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-awq")

# Normals-guided drop. A splat is only eligible for drop if its own
# normal aligns with the floor plane normal (|n_splat · n_plane| > thresh).
# 0.5 ≈ within 60° of vertical → drops "kinda flat" floor splats whose
# normals are noisy. Matches the plugin's soft-surface lock. Higher =
# stricter (preserves more legs). Lower = more aggressive floor drop.
NORMAL_COS_THRESH = 0.3   # was 0.5 — looser → drops more in-band splats (within ~73° of vertical)

FOV = 70.0
W, H = 1920, 1080
Y_DOWN = True
CANONICAL_YAWS = [0, 90, 180, 270]
CANONICAL_PITCH = -20.0
CANONICAL_TOPDOWN_PITCH = -89.0


def build_camera(center, yaw_deg, pitch_deg, distance, fov, w, h, y_down=True):
    pitch_eff = pitch_deg if y_down else -pitch_deg
    center = np.asarray(center, dtype=np.float32)
    base_eye = center + np.array([0, 0, distance], dtype=np.float32)
    R = rotation_matrix_from_yaw_pitch(yaw_deg, pitch_eff)
    eye = center + R.T @ (base_eye - center)
    up = np.array([0, -1 if y_down else 1, 0], dtype=np.float32)
    V = viewmat_look_at(eye, center, up)
    K = build_K(fov, w, h)
    return V, K, eye


def render_canonical_5(ply_path: Path, out_dir: Path):
    """Render the 5 canonical display views. THIS step is where the
    display-camera framing is computed and locked — once floor_drop runs,
    `<obj>/display_cameras.json` records the exact camera config and
    every later stage (sam_tight, bookshelf_sweep, sweep_fallback) must
    reuse those cameras so the per-stage canonical_5 renders are pixel-
    comparable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    scene = load_gsplat_ply(str(ply_path))
    means = scene["means"].detach().cpu().numpy()
    if len(means) == 0:
        print(f"[render] {ply_path} is empty — skipping")
        return
    # FULL extent (min/max) — not p5/p95. Outliers must fit in frame.
    lo = means.min(axis=0)
    hi = means.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = float((hi - lo).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    cameras_log = []
    for yaw_deg in CANONICAL_YAWS:
        V, K, eye = build_camera(center, yaw_deg, CANONICAL_PITCH, distance,
                                  FOV, W, H, y_down=Y_DOWN)
        img = render_splat(scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
        Image.fromarray(img).save(out_dir / f"y{int(yaw_deg)}.png")
        cameras_log.append({
            "tag": f"y{int(yaw_deg)}",
            "yaw_deg": float(yaw_deg), "pitch_deg": float(CANONICAL_PITCH),
            "V": V.tolist(), "K": K.tolist(),
            "eye": eye.tolist(), "target": center.tolist(),
        })
    V, K, eye = build_camera(center, 0, CANONICAL_TOPDOWN_PITCH, distance,
                              FOV, W, H, y_down=Y_DOWN)
    img = render_splat(scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
    Image.fromarray(img).save(out_dir / "topdown.png")
    cameras_log.append({
        "tag": "topdown",
        "yaw_deg": 0.0, "pitch_deg": float(CANONICAL_TOPDOWN_PITCH),
        "V": V.tolist(), "K": K.tolist(),
        "eye": eye.tolist(), "target": center.tolist(),
    })

    # Lock display cameras at floor_drop. `out_dir` is
    # <obj>/renders/3_floor_drop, so obj_dir = out_dir.parent.parent.
    if out_dir.name == "3_floor_drop":
        obj_dir = out_dir.parent.parent
        (obj_dir / "display_cameras.json").write_text(json.dumps({
            "locked_at": "3_floor_drop",
            "center": center.tolist(),
            "extent": extent,
            "distance": distance,
            "fov": FOV, "width": W, "height": H, "y_down": Y_DOWN,
            "cameras": cameras_log,
        }, indent=2))


def carve_at_thresh(v_data, signed_dist, thresh,
                     normal_align=None, normal_cos_thresh=NORMAL_COS_THRESH,
                     upper_thresh=UPPER_THRESH_M):
    """Asymmetric band: drop splats from -upper_thresh (above floor) down
    to +thresh (below floor). Above-floor cap protects chair body/legs;
    below-floor extent is Qwen-tuned. Splats only drop if also normal-
    aligned with floor plane normal (>cos_thresh).

    Convention reminder (y-down): signed_dist < 0 = above floor (chair),
    signed_dist > 0 = below floor (subfloor noise).
    """
    abs_dist = np.abs(signed_dist)
    in_band = (signed_dist >= -upper_thresh) & (signed_dist <= thresh)
    n_in_band = int(in_band.sum())
    n_above = int((signed_dist < -upper_thresh).sum())
    n_below = int((signed_dist > thresh).sum())

    if normal_align is not None:
        # Only drop splats that are BOTH in the band AND horizontal-aligned.
        floor_aligned = normal_align > normal_cos_thresh
        drop = in_band & floor_aligned
        n_in_band_floor = int(drop.sum())
        n_in_band_kept = n_in_band - n_in_band_floor
    else:
        drop = in_band
        n_in_band_floor = n_in_band
        n_in_band_kept = 0

    keep = ~drop
    n_dropped = int(drop.sum())
    return (keep, n_dropped, n_above, n_in_band, n_below,
            n_in_band_floor, n_in_band_kept)


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def qwen_judge_iter(render_dir: Path, current_thresh: float,
                     n_kept: int, n_in: int, label: str,
                     prior_history: list) -> dict:
    """Show Qwen the current iter's renders + tell it the threshold.
    Qwen returns verdict + new threshold + reasoning."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag in QWEN_VIEWS:
        p = render_dir / f"{tag}.png"
        if not p.exists():
            continue
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})

    history_str = ""
    if prior_history:
        lines = []
        for h in prior_history:
            lines.append(f"  iter {h['iter']}: thresh={h['thresh']:.3f}m → "
                          f"verdict={h['verdict']} ({h['reasoning'][:60]})")
        history_str = "Prior attempts:\n" + "\n".join(lines) + "\n\n"

    content.append({"type": "text", "text":
        f"You are tuning a floor-removal threshold for a {label} extraction. "
        f"Splats within ±THRESHOLD meters of the floor plane get dropped — "
        f"BUT a normal-aware filter PRESERVES vertical surfaces (chair legs), "
        f"so legs SHOULD survive even at high thresholds.\n\n"
        f"CURRENT: threshold = {current_thresh:.3f}m, "
        f"kept {n_kept:,} / {n_in:,} splats.\n\n"
        f"{history_str}"
        f"PRIORITY ORDER (read carefully):\n"
        f"  1. CHAIR LEGS MUST BE PRESERVED. This is the most important "
        f"     check. Look at y180 — count visible legs and trace each one "
        f"     from the seat down to the floor. If ANY leg is missing, "
        f"     truncated, or 'floating' (gap between leg bottom and former "
        f"     floor line), the threshold is TOO HIGH — DECREASE.\n"
        f"  2. Then check floor removal. In the topdown, look AROUND the "
        f"     {label}'s edges for wood plank texture (parallel brown "
        f"     stripes / beige boards). In y180/y0/y90, look BEHIND/BESIDE "
        f"     the {label} for wood-grain bands.\n\n"
        f"Decision rules:\n"
        f"  - Legs cut → DECREASE (always wins over floor cleanup).\n"
        f"  - Legs intact + floor still visible → INCREASE.\n"
        f"  - Legs intact + floor mostly clean → small INCREASE to test "
        f"    if you can clean any halo without losing legs.\n\n"
        f"This is iteration {len(prior_history)} of multiple. ALWAYS suggest "
        f"a new threshold (don't STOP yet) — we'll pick the best at the end.\n\n"
        f"Range: {THRESH_MIN} to {THRESH_MAX} meters. Typical step: "
        f"±0.05 to ±0.10m so 5 iterations cover a wide range. Be CONSERVATIVE "
        f"about legs — preserving legs > removing floor halo.\n\n"
        f"Output ONLY this JSON, no commentary, no markdown:\n"
        f"{{\n"
        f'  "verdict": "INCREASE" | "DECREASE",\n'
        f'  "new_thresh_m": <float, MUST differ from current by ≥0.01>,\n'
        f'  "floor_visible": true | false,\n'
        f'  "legs_preserved": true | false,\n'
        f'  "n_legs_visible": <integer, count of distinct legs you can see>,\n'
        f'  "reasoning": "<one short sentence>"\n'
        f"}}"})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=300, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    return parse_judge_response(raw, current_thresh), raw


def parse_judge_response(raw: str, fallback_thresh: float) -> dict:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    if s != -1 and e > s:
        try:
            data = json.loads(cleaned[s:e])
            verdict = str(data.get("verdict", "STOP")).upper()
            if verdict not in ("STOP", "INCREASE", "DECREASE"):
                verdict = "STOP"
            new_t = float(data.get("new_thresh_m", fallback_thresh))
            new_t = max(THRESH_MIN, min(THRESH_MAX, new_t))
            return {
                "verdict": verdict,
                "new_thresh_m": new_t,
                "floor_visible": bool(data.get("floor_visible", False)),
                "legs_preserved": bool(data.get("legs_preserved", True)),
                "reasoning": str(data.get("reasoning", ""))[:200],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as ex:
            print(f"[parse] JSON error: {ex} — falling back to STOP")
    return {"verdict": "STOP", "new_thresh_m": fallback_thresh,
            "floor_visible": False, "legs_preserved": True,
            "reasoning": "parse_error"}


def qwen_pick_best(diag: Path, attempts: list, label: str) -> int:
    """After max_iterations exhausted without a STOP, send Qwen a
    side-by-side of all attempts (topdown + y180 thumbnails) and ask
    which iter index was BEST."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for a in attempts:
        idx = a["iter"]
        thresh = a["thresh"]
        for view in ("topdown", "y180"):
            p = diag / f"iter{idx}" / f"{view}.png"
            if not p.exists():
                continue
            content.append({"type": "text",
                             "text": f"\nIter {idx} ({view}, thresh={thresh:.3f}m):"})
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    summary = "\n".join(
        f"  iter {a['iter']}: thresh={a['thresh']:.3f}m, "
        f"floor_visible={a['floor_visible']}, legs_preserved={a['legs_preserved']}, "
        f"reasoning={a['reasoning'][:80]}"
        for a in attempts)
    content.append({"type": "text", "text":
        f"You ran {len(attempts)} iterations of floor-removal threshold tuning "
        f"for a {label} extraction.\n\n"
        f"Iterations:\n{summary}\n\n"
        f"Pick the iter index with the BEST balance, prioritized:\n"
        f"  1. ALL LEGS preserved (most important — never pick an iter "
        f"     where legs are truncated or missing).\n"
        f"  2. THEN floor removal — choose the iter that cleans the most "
        f"     wood-floor halo while still keeping all legs.\n\n"
        f"If multiple iters preserve legs equally, prefer the one with "
        f"least floor visible (typically the higher threshold).\n\n"
        f"Output ONLY this JSON:\n"
        f"{{\n"
        f'  "best_iter": <int>,\n'
        f'  "reasoning": "<one short sentence>"\n'
        f"}}"})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=200, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    try:
        data = json.loads(cleaned[s:e])
        idx = int(data.get("best_iter", 0))
        idx = max(0, min(len(attempts) - 1, idx))
        return idx, data.get("reasoning", ""), raw
    except Exception:
        return 0, "fallback to iter 0", raw


def carve_and_save_iter(obj_dir: Path, diag: Path, vx_data, signed_dist,
                          thresh: float, iter_idx: int,
                          normal_align=None,
                          normal_cos_thresh=NORMAL_COS_THRESH) -> dict:
    """Carve at threshold (with optional normal-aware filter), save iter
    PLY + iter renders. Returns stats dict."""
    keep, n_dropped, n_above, n_in_band, n_below, n_band_floor, n_band_kept = \
        carve_at_thresh(vx_data, signed_dist, thresh,
                         normal_align, normal_cos_thresh)
    iter_ply = diag / f"iter{iter_idx}.ply"
    PlyData([PlyElement.describe(vx_data[keep], "vertex")],
            text=False).write(str(iter_ply))
    iter_render_dir = diag / f"iter{iter_idx}"
    render_canonical_5(iter_ply, iter_render_dir)
    return {
        "iter": iter_idx,
        "thresh": thresh,
        "n_kept": int(keep.sum()),
        "n_dropped": n_dropped,
        "n_above": n_above,
        "n_in_band": n_in_band,
        "n_in_band_floor_aligned": n_band_floor,
        "n_in_band_legs_preserved": n_band_kept,
        "n_below": n_below,
        "ply_path": str(iter_ply),
        "render_dir": str(iter_render_dir),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path,
                    help="path to 02_<slug>/ (must contain sam_wide.ply)")
    ap.add_argument("--floor-thresh", type=float, default=INITIAL_THRESH_M,
                    help=f"initial threshold in m (default {INITIAL_THRESH_M})")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                    help=f"max Qwen-driven iterations (default {MAX_ITERATIONS})")
    ap.add_argument("--no-qwen-loop", action="store_true",
                    help="skip Qwen loop; single-shot at --floor-thresh")
    ap.add_argument("--normal-cos-thresh", type=float,
                    default=NORMAL_COS_THRESH,
                    help=f"only drop band splats whose normal aligns with "
                         f"floor plane normal (|n_splat·n_plane| > thresh). "
                         f"Higher = preserves more legs. Default {NORMAL_COS_THRESH}")
    ap.add_argument("--no-normal-filter", action="store_true",
                    help="disable normal-aware filtering — drop everything in band "
                         "(WILL kill chair legs touching floor)")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()

    fp_path = scene / "01_shell_removed" / "floor_plane.json"
    if not fp_path.exists():
        sys.exit(f"[fatal] missing {fp_path}")
    fp = json.load(open(fp_path))
    pa, pb, pc, pd = fp["plane"]
    norm = math.sqrt(pa * pa + pb * pb + pc * pc)
    print(f"[plane] equation: {pa:.6f}*x + {pb:.6f}*y + {pc:.6f}*z + {pd:.6f}")
    print(f"[plane] floor y ≈ {-pd / pb:.3f}")

    in_ply = obj / "2_sam_wide.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing {in_ply}\n  run sam_carve.py through step 4 first")
    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n_in = len(xyz)
    print(f"[load] {in_ply}: {n_in:,} splats")

    signed_dist = (pa * xyz[:, 0] + pb * xyz[:, 1] + pc * xyz[:, 2] + pd) / norm

    # Per-splat normal: derived from quaternion + scales (NOT nx/ny/nz —
    # those are typically zeros in gsplat PLYs). Each splat is an ellipsoid;
    # its surface normal is the axis of SMALLEST scale (most flattened
    # direction) rotated by the splat's quaternion.
    if args.no_normal_filter:
        normal_align = None
        print("[normals] DISABLED — will drop everything in band")
    else:
        rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                         axis=1).astype(np.float64)
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                           axis=1).astype(np.float64)
        # Normalize quaternions [w, x, y, z]
        q_norm = np.linalg.norm(rots, axis=1, keepdims=True)
        q = rots / np.maximum(q_norm, 1e-9)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Rotation matrix columns
        col0 = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], axis=1)
        col1 = np.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], axis=1)
        col2 = np.stack([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)], axis=1)
        all_cols = np.stack([col0, col1, col2], axis=1)  # (N, 3, 3)
        # Smallest-scale axis = surface normal direction in local frame
        min_axis = np.argmin(scales, axis=1)
        normals = all_cols[np.arange(len(min_axis)), min_axis]  # (N, 3)
        # Renormalize to unit (floating point drift)
        n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(n_norm, 1e-9)
        plane_n = np.array([pa, pb, pc]) / norm
        normal_align = np.abs(normals @ plane_n)
        print(f"[normals] derived from quaternion + smallest-scale axis, "
              f"cos_thresh={args.normal_cos_thresh}")
        print(f"[normals] alignment stats: median={float(np.median(normal_align)):.3f} "
              f"p95={float(np.percentile(normal_align, 95)):.3f}")

    # Read label from visual_hull_meta.json (for Qwen prompt context)
    meta_path = obj / "1_visual_hull_meta.json"
    label = "object"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        label = meta.get("label", "object")
    print(f"[label] {label}")

    diag = obj / "diagnostics" / "3_floor_drop"
    diag.mkdir(parents=True, exist_ok=True)
    # Clear stale per-iter files from prior runs
    for f in diag.glob("iter*"):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
    for f in diag.glob("qwen_iter*.txt"):
        f.unlink()

    if args.no_qwen_loop:
        print(f"\n[mode] single-shot, thresh={args.floor_thresh}")
        stats = carve_and_save_iter(obj, diag, v.data, signed_dist,
                                     args.floor_thresh, 0,
                                     normal_align=normal_align,
                                     normal_cos_thresh=args.normal_cos_thresh)
        winner = stats
        history = [{**stats, "verdict": "MANUAL", "floor_visible": None,
                    "legs_preserved": None, "reasoning": "no qwen loop"}]
    else:
        print(f"\n[mode] Qwen feedback loop, ALWAYS {args.max_iterations} iterations")
        history = []
        current_thresh = args.floor_thresh

        for i in range(args.max_iterations):
            # Avoid suggesting an already-tested threshold
            if any(abs(h["thresh"] - current_thresh) < 0.005 for h in history):
                # Nudge so we don't repeat
                current_thresh = min(THRESH_MAX,
                                      current_thresh +
                                      (0.03 if i % 2 == 0 else -0.03))
                current_thresh = max(THRESH_MIN, current_thresh)
                print(f"[iter {i}] nudged duplicate thresh → {current_thresh:.3f}m")

            print(f"\n[iter {i}] thresh={current_thresh:.3f}m")
            stats = carve_and_save_iter(obj, diag, v.data, signed_dist,
                                         current_thresh, i,
                                         normal_align=normal_align,
                                         normal_cos_thresh=args.normal_cos_thresh)
            print(f"[iter {i}] kept {stats['n_kept']:,} / {n_in:,} "
                  f"(dropped {stats['n_dropped']:,} in band)")

            print(f"[iter {i}] asking Qwen...")
            judge, raw = qwen_judge_iter(
                Path(stats["render_dir"]), current_thresh,
                stats["n_kept"], n_in, label,
                history)
            (diag / f"qwen_iter{i}_raw.txt").write_text(raw)
            print(f"[iter {i}] qwen: verdict={judge['verdict']} "
                  f"new_thresh={judge['new_thresh_m']:.3f} "
                  f"floor_vis={judge['floor_visible']} "
                  f"legs={judge['legs_preserved']}")
            print(f"[iter {i}] reason: {judge['reasoning']}")

            history.append({**stats, **judge})
            current_thresh = judge["new_thresh_m"]

        # All iterations done — let Qwen pick best
        print(f"\n[loop] {args.max_iterations} iterations complete")
        print(f"[loop] sending all attempts to Qwen for best pick...")
        best_idx, best_reason, best_raw = qwen_pick_best(diag, history, label)
        (diag / "qwen_pick_best_raw.txt").write_text(best_raw)
        print(f"[loop] Qwen picked iter {best_idx}: {best_reason}")
        winner = next(h for h in history if h["iter"] == best_idx)

    # Promote winner to canonical floor_drop.ply + renders/floor_drop/
    out_ply = obj / "3_floor_drop.ply"
    shutil.copy(winner["ply_path"], out_ply)
    out_render_dir = obj / "renders" / "3_floor_drop"
    if out_render_dir.exists():
        shutil.rmtree(out_render_dir)
    shutil.copytree(winner["render_dir"], out_render_dir)
    print(f"\n[promote] iter {winner['iter']} → floor_drop.ply + renders/floor_drop/")

    (diag / "history.json").write_text(json.dumps({
        "stage": "floor_drop",
        "input_ply": str(in_ply),
        "output_ply": str(out_ply),
        "label": label,
        "winner_iter": winner["iter"],
        "winner_thresh_m": winner["thresh"],
        "n_in": n_in,
        "n_kept": winner["n_kept"],
        "iterations": history,
    }, indent=2))
    (diag / "report.json").write_text(json.dumps({
        "stage": "floor_drop",
        "winner_iter": winner["iter"],
        "winner_thresh_m": winner["thresh"],
        "n_in": n_in,
        "n_kept": winner["n_kept"],
        "n_dropped": winner["n_dropped"],
    }, indent=2))

    print(f"\n[done]")
    print(f"  PLY:     {out_ply}")
    print(f"  renders: {out_render_dir}")
    print(f"  history: {diag / 'history.json'}")


if __name__ == "__main__":
    main()
