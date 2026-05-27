#!/usr/bin/env python3
"""splat_destreak.py — drop big+dark Gaussian splat streaks.

Runs as Stage 7 on the stage_pick output (7_picked.ply). Scans per-splat
for the joint failure mode: max scale axis > THRESH_S AND mean SH-deg-0
brightness < THRESH_B. Almost always = ellipsoid streaks from a bad
optimizer local minimum (one axis blew up, color collapsed to ~black).
Legitimate dark splats stay small, legitimate big splats stay bright,
so neither trips the AND.

--auto mode: sweeps 3 (min_scale, max_brightness) settings, renders each
result, sends ALL three to Qwen via a single multi-image call. Qwen
picks the best (1/2/3) — most streaks removed without eroding legitimate
material. Mirrors inside_outside --auto flow.

Validated 2026-05-27 on light_wood_bookshelf: 181 splats dropped at
default, the foreground vertical streaks vanished, no legitimate
material lost.

Usage:
    # Single threshold (legacy):
    python splat_destreak.py <obj_dir>
        [--in-ply 7_picked.ply] [--out-ply 7_destreak.ply]
        [--min-scale 0.10] [--max-brightness 0.25]

    # Auto sweep + Qwen pick:
    python splat_destreak.py <obj_dir> --auto
        [--in-ply 7_picked.ply] [--out-ply 7_destreak.ply]
"""
import argparse
import base64
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from sam_carve import render_canonical_5  # noqa: E402
from view import load_gsplat_ply, render_splat  # noqa: E402
from sam_carve import (  # noqa: E402
    build_camera, FOV, CANONICAL_PITCH, CANONICAL_W, CANONICAL_H, Y_DOWN,
    QWEN_URL, QWEN_MODEL,
)

DEFAULT_MIN_SCALE_M = 0.10
DEFAULT_MAX_BRIGHTNESS = 0.25

# 3-threshold sweep for --auto. Ordered conservative → aggressive.
# Conservative drops fewer (only the biggest+darkest); aggressive drops
# more (catches dimmer / smaller streaks at the cost of nibbling some
# legit material). Qwen picks visually.
AUTO_THRESHOLDS = [
    {"min_scale": 0.15, "max_brightness": 0.20, "tag": "conservative"},
    {"min_scale": 0.10, "max_brightness": 0.25, "tag": "default"},
    {"min_scale": 0.06, "max_brightness": 0.35, "tag": "aggressive"},
]


def _compute_drop_mask(v, min_scale_m: float, max_brightness: float):
    """Returns (drop_bool_mask, n_drop)."""
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                      axis=1).astype(np.float64)
    max_scale = np.exp(scales).max(axis=1)
    rgb_sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
                      axis=1).astype(np.float64)
    brightness = (rgb_sh * 0.28 + 0.5).mean(axis=1)
    drop = (max_scale > min_scale_m) & (brightness < max_brightness)
    return drop, int(drop.sum())


def _write_filtered_ply(v, drop_mask, out_path: Path):
    PlyData([PlyElement.describe(v.data[~drop_mask], "vertex")],
            text=False).write(str(out_path))


def _render_y0(ply_path: Path, out_png: Path):
    """Render y0 with per-PLY centroid framing for Qwen pick."""
    scene = load_gsplat_ply(str(ply_path))
    means = scene["means"].detach().cpu().numpy()
    lo = np.percentile(means, 2, axis=0)
    hi = np.percentile(means, 98, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = max(float((hi - lo).max()), 0.15)
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * 1.55) / (2 * tan_half) + 0.5
    V, K, _ = build_camera(center, 0, CANONICAL_PITCH, distance,
                            FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
    img = render_splat(scene, V, K, CANONICAL_W, CANONICAL_H,
                       bg=(1.0, 1.0, 1.0))
    Image.fromarray(img).save(out_png)


def _b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                          Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _qwen_pick(candidates):
    """Qwen multi-image pick among destreak candidates.
    Returns (pick_index, raw_reply)."""
    content = []
    for i, c in enumerate(candidates):
        content.append({"type": "text",
                         "text": f"\n--- CANDIDATE {i+1} ({c['tag']}) ---"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{_b64(c['render'])}"}})
    n = len(candidates)
    content.append({"type": "text", "text":
        f"Each candidate shows the same extracted object. A 'destreak' "
        f"cleanup tries to drop large dark Gaussian splat streaks (an "
        f"optimizer failure mode that paints big black smears, vertical "
        f"black bars, fuzzy dark halos, or random black blobs near or "
        f"across the object).\n\n"
        f"Candidate 1 (skip): NO destreaking — source as-is.\n"
        f"Candidate 2 (conservative): drops only the biggest, darkest streaks.\n"
        f"Candidate 3 (default): moderate cleanup.\n"
        f"Candidate 4 (aggressive): drops dimmer/smaller streaks too. May "
        f"start nibbling legitimate dark material like dark wood, dark "
        f"fabric, or shadow detail.\n\n"
        f"DEFAULT BIAS: prefer to remove streaks. Black smears, vertical "
        f"black bars, fuzzy dark blobs floating near the object, or dark "
        f"halos around edges are ALWAYS artifacts and should be removed. "
        f"Pick candidate 2 or 3 in most cases.\n\n"
        f"Pick candidate 1 (skip) ONLY if you are CONFIDENT the source has "
        f"zero visible black smears or halos AND every destreak candidate "
        f"removes legitimate dark material the object actually has (dark "
        f"wood, dark cushion fabric, etc.). When in doubt between skip and "
        f"conservative, pick conservative.\n\n"
        f"Pick candidate 4 (aggressive) only when streaks are SEVERE and "
        f"the object has little legitimate dark material.\n\n"
        f"Reply with ONLY the number (1 to {n}). No other text."})

    payload = json.dumps({
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 50, "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(QWEN_URL + "/chat/completions",
                                  data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    pick = None
    for tok in raw.replace(",", " ").replace(".", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(candidates):
            pick = int(tok) - 1
            break
    if pick is None:
        print(f"[qwen] could not parse pick from '{raw}' — defaulting to 1 (conservative)")
        pick = 0
    return pick, raw


def run_auto_sweep(obj: Path, in_ply_name: str, out_ply_name: str):
    """Sweep 3 thresholds, render each, Qwen picks best."""
    src = obj / in_ply_name
    dst = obj / out_ply_name
    if not src.exists():
        print(f"[destreak] SKIPPED — no {src.name} in {obj}")
        return

    import shutil
    p = PlyData.read(str(src))
    v = p["vertex"]
    n_in = len(v)

    # Quick check — if even the most aggressive setting drops 0, no
    # streaks exist anywhere. Skip qwen entirely, copy input through.
    most_aggressive = AUTO_THRESHOLDS[-1]
    _, n_max = _compute_drop_mask(v, most_aggressive["min_scale"],
                                  most_aggressive["max_brightness"])
    if n_max == 0:
        print(f"[destreak] no streaks at any threshold — copying {src.name} → {dst.name}")
        shutil.copy(src, dst)
        out_renders = obj / "renders" / dst.stem
        render_canonical_5(dst, out_renders)
        diag = obj / "diagnostics" / "7_destreak"
        diag.mkdir(parents=True, exist_ok=True)
        (diag / "report.json").write_text(json.dumps({
            "stage": "splat_destreak_auto",
            "in_ply": str(src), "out_ply": str(dst),
            "n_in": n_in, "n_kept": n_in, "n_dropped": 0,
            "picked_tag": "no_op", "auto_skipped": True,
        }, indent=2))
        return

    # Generate 4 candidates: candidate 1 = SKIP (source as-is), 2-4 = destreak
    diag = obj / "diagnostics" / "7_destreak"
    diag.mkdir(parents=True, exist_ok=True)
    candidates = []

    # Candidate 1 — SKIP (no destreak applied, just source)
    skip_ply = diag / "candidate_skip.ply"
    shutil.copy(src, skip_ply)
    skip_render = diag / "candidate_skip.png"
    _render_y0(skip_ply, skip_render)
    print(f"[destreak] {'skip':13s}: drop    0  (source as-is)")
    candidates.append({"tag": "skip", "cfg": {"min_scale": None, "max_brightness": None},
                       "ply": skip_ply, "render": skip_render, "n_drop": 0})

    # Candidates 2-4 — destreak at 3 thresholds
    for cfg in AUTO_THRESHOLDS:
        drop, n_drop = _compute_drop_mask(v, cfg["min_scale"],
                                          cfg["max_brightness"])
        cand_ply = diag / f"candidate_{cfg['tag']}.ply"
        _write_filtered_ply(v, drop, cand_ply)
        cand_render = diag / f"candidate_{cfg['tag']}.png"
        _render_y0(cand_ply, cand_render)
        print(f"[destreak] {cfg['tag']:13s}: drop {n_drop:>4d}  "
              f"(scale>{cfg['min_scale']}m, bright<{cfg['max_brightness']})")
        candidates.append({"tag": cfg["tag"], "cfg": cfg, "ply": cand_ply,
                           "render": cand_render, "n_drop": n_drop})

    pick_idx, raw = _qwen_pick(candidates)
    picked = candidates[pick_idx]
    print(f"[qwen] picked {pick_idx+1} ({picked['tag']}) — raw='{raw}'")

    # Copy picked candidate to final out_ply
    import shutil
    shutil.copy(picked["ply"], dst)
    out_renders = obj / "renders" / dst.stem
    render_canonical_5(dst, out_renders)
    print(f"[destreak] wrote {dst.name}  ({n_in - picked['n_drop']:,} splats)")
    print(f"[render] canonical 5 → {out_renders}")

    (diag / "report.json").write_text(json.dumps({
        "stage": "splat_destreak_auto",
        "in_ply": str(src), "out_ply": str(dst),
        "n_in": n_in,
        "candidates": [{"tag": c["tag"], "min_scale": c["cfg"]["min_scale"],
                         "max_brightness": c["cfg"]["max_brightness"],
                         "n_drop": c["n_drop"]} for c in candidates],
        "picked_tag": picked["tag"],
        "picked_min_scale": picked["cfg"]["min_scale"],
        "picked_max_brightness": picked["cfg"]["max_brightness"],
        "n_dropped": picked["n_drop"],
        "n_kept": n_in - picked["n_drop"],
        "qwen_raw": raw,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--in-ply", default="7_picked.ply")
    ap.add_argument("--out-ply", default="7_destreak.ply")
    ap.add_argument("--auto", action="store_true",
                    help="sweep 3 thresholds + Qwen picks best (recommended)")
    ap.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE_M)
    ap.add_argument("--max-brightness", type=float,
                    default=DEFAULT_MAX_BRIGHTNESS)
    args = ap.parse_args()
    obj = args.obj_dir.resolve()

    if args.auto:
        run_auto_sweep(obj, args.in_ply, args.out_ply)
        return

    # Single-threshold legacy path
    src = obj / args.in_ply
    dst = obj / args.out_ply
    if not src.exists():
        print(f"[destreak] SKIPPED — no {src.name} in {obj}")
        return

    p = PlyData.read(str(src))
    v = p["vertex"]
    n_in = len(v)
    drop, n_drop = _compute_drop_mask(v, args.min_scale, args.max_brightness)
    n_kept = n_in - n_drop
    print(f"[destreak] {n_in:,} → {n_kept:,} "
          f"(dropped {n_drop} big+dark splats, "
          f"thresholds scale>{args.min_scale}m bright<{args.max_brightness})")

    if n_drop == 0:
        print(f"[destreak] no streaks — {src.name} unchanged")
        import shutil
        if dst != src:
            shutil.copy(src, dst)
        return

    _write_filtered_ply(v, drop, dst)
    print(f"[destreak] wrote {dst.name}")
    out_renders = obj / "renders" / dst.stem
    render_canonical_5(dst, out_renders)
    print(f"[render] canonical 5 → {out_renders}")

    diag = obj / "diagnostics" / "7_destreak"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "report.json").write_text(json.dumps({
        "stage": "splat_destreak",
        "in_ply": str(src), "out_ply": str(dst),
        "n_in": n_in, "n_kept": n_kept, "n_dropped": n_drop,
        "min_scale_m": args.min_scale,
        "max_brightness": args.max_brightness,
    }, indent=2))


if __name__ == "__main__":
    main()
