#!/usr/bin/env python3
"""Stages 7 + 8 — final stage picker + destreak.

Picks one candidate PLY from the pipeline chain (see CANDIDATE_PLYS:
3_floor_drop, 4_sam_tight, 4b_sam_tight_low, 5_bookshelf_sweep,
5b_bookshelf_sweep_low, 5_sweep_fallback, 6_inside_outside), copies it to
7_picked.ply, runs `splat_destreak --auto` to 7_destreak.ply, then
promotes the result to **8_final.ply** (the shipped deliverable), converts
to final.splat, renders canonical 5, and renames the obj_dir to the
refined label slug. NOTE: the current code writes 8_final.ply, NOT
7_final.ply — every finalize consumer resolves the final stage via the
shared stage_preference.py list, which leads with 8_final.

PICK METHOD (2026-05-20): hardcoded preference order
  inside_outside → sweep_fallback → sam_tight_low → sam_tight
(first one that exists wins). Qwen-pick was tried first but Qwen
consistently favored the largest silhouette regardless of halo —
unreliable for visual cleanliness judgment across differently-framed
renders. qwen_pick() function kept in this file for future revisit.

Usage:
    python stage_pick.py <obj_dir>
"""
import argparse
import base64
import io
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from plyfile import PlyData
from PIL import Image

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from view import load_gsplat_ply, render_splat  # noqa
from sam_carve import (  # noqa
    build_camera, parse_tagged_prompts, render_canonical_5,
    FOV, CANONICAL_PITCH, CANONICAL_W, CANONICAL_H, Y_DOWN,
    QWEN_URL, QWEN_MODEL,
)
from extract_one import slugify  # noqa

ITERATION_DIR = Path(__file__).resolve().parent


CANDIDATE_PLYS = [
    ("floor_drop",          "3_floor_drop.ply"),       # table chain final
    ("sam_tight",           "4_sam_tight.ply"),
    ("sam_tight_low",       "4b_sam_tight_low.ply"),
    ("bookshelf_sweep",     "5_bookshelf_sweep.ply"),  # bookshelf chain final (high pitch)
    ("bookshelf_sweep_low", "5b_bookshelf_sweep_low.ply"),  # 2026-05-22 second pass at low pitches
    ("sweep_fallback",      "5_sweep_fallback.ply"),
    ("inside_outside",      "6_inside_outside.ply"),
]
# 2026-05-20: Qwen-pick across stages parked. Visual comparison across
# differently-framed candidates was unreliable (sweep_fallback's wider
# silhouette kept winning despite more halo). Hardcoded preference
# order — pick the most-refined stage that exists. Different chains
# reach different final stages:
#   - general chain → inside_outside (best)
#   - bookshelf chain → 5_bookshelf_sweep (no inside_outside; sweep is final)
#   - table chain → 3_floor_drop (no sam_tight; floor_drop is final)
# Picker code below kept intact for future revisit; just bypassed.
PREFERRED_ORDER = [
    "inside_outside",
    "bookshelf_sweep",
    "sweep_fallback",
    "sam_tight_low",
    "sam_tight",
    "floor_drop",
]


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                          Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_y0(ply_path: Path, out_png: Path, locked_cam=None):
    """Render y0 at locked camera if available, else per-PLY framing."""
    scene = load_gsplat_ply(str(ply_path))
    means = scene["means"].detach().cpu().numpy()
    if locked_cam is not None:
        V = np.array(locked_cam["V"], dtype=np.float32)
        K = np.array(locked_cam["K"], dtype=np.float32)
        W_, H_ = locked_cam["width"], locked_cam["height"]
    else:
        lo = np.percentile(means, 2, axis=0)
        hi = np.percentile(means, 98, axis=0)
        center = ((lo + hi) * 0.5).astype(np.float32)
        extent = max(float((hi - lo).max()), 0.15)
        tan_half = math.tan(math.radians(FOV) / 2)
        distance = (extent * 1.55) / (2 * tan_half) + 0.5
        V, K, _ = build_camera(center, 0, CANONICAL_PITCH, distance,
                                FOV, CANONICAL_W, CANONICAL_H, y_down=Y_DOWN)
        W_, H_ = CANONICAL_W, CANONICAL_H
    img = render_splat(scene, V, K, W_, H_, bg=(1.0, 1.0, 1.0))
    Image.fromarray(img).save(out_png)


def qwen_pick(candidates, label, pipe_union):
    """Single multi-image Qwen call. Returns (pick_index, raw_reply).

    Decision is VISUAL ONLY — splat counts / stage names are NOT shown
    to Qwen to avoid biasing toward biggest=best. Qwen sees only the
    rendered images and the candidate number."""
    content = []
    for i, c in enumerate(candidates):
        content.append({"type": "text",
                         "text": f"\n--- CANDIDATE {i+1} ---"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encode_b64(c['render'])}"}})
    content.append({"type": "text", "text":
        f"Each numbered candidate shows the SAME extracted '{label}'. "
        f"The pipe-union the upstream SAM step used was:\n\n"
        f"  {pipe_union}\n\n"
        f"Pick the candidate that looks best overall, balancing two things:\n\n"
        f"- CLEAN: less floor halo, less wisps, less smear around the object, "
        f"tighter silhouette against the white background.\n"
        f"- INTACT: the '{label}' looks whole — no visible holes or "
        f"carved-out chunks in the main body, no missing legs, no missing "
        f"sub-items from the pipe-union.\n\n"
        f"Weigh both. A candidate with some mild halo but a fully intact "
        f"object is usually better than a candidate that looks clean but "
        f"has a chunk missing from the body or a leg gone. On the other "
        f"hand, a candidate that's slightly less halo-y is preferable when "
        f"both are intact. Use your judgment.\n\n"
        f"Slightly thinner supports, slight edge wisps, slightly fewer "
        f"detail splats — those are minor. Visible holes in the solid body "
        f"or completely-missing parts are the things to weight against.\n\n"
        f"Reply with ONLY the candidate number (1 to {len(candidates)}). "
        f"No other text."})

    payload = json.dumps({
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100, "temperature": 0.1,
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
        # Default to last (most-processed) on parse failure
        print(f"[qwen] could not parse pick from '{raw}' — defaulting "
              f"to last candidate")
        pick = len(candidates) - 1
    return pick, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj_dir", type=Path)
    args = ap.parse_args()
    obj = args.obj_dir.resolve()

    # Find available candidates
    cands = []
    for tag, fname in CANDIDATE_PLYS:
        p = obj / fname
        if not p.exists():
            print(f"[skip] {tag} — {fname} not present")
            continue
        cands.append({"tag": tag, "ply": p, "fname": fname})
    if not cands:
        sys.exit(f"[fatal] no candidate PLYs found in {obj}")

    # Count splats per candidate (for diagnostics only — NOT used to pick)
    for c in cands:
        pl = PlyData.read(str(c["ply"]))
        c["n_splats"] = len(pl["vertex"].data)
        print(f"  {c['tag']}: {c['n_splats']:,} splats  ({c['fname']})")

    # Label + pipe-union (same priority chain as sweep_fallback)
    label = "object"
    pipe_union = ""
    prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if prompt_path.exists():
        pipe_union = prompt_path.read_text().strip()
        try:
            tagged = parse_tagged_prompts(pipe_union)
            if tagged:
                label = tagged[0][0]
        except Exception:
            pass
    if label == "object":
        meta_path = obj / "1_visual_hull_meta.json"
        if meta_path.exists():
            try:
                label = json.load(open(meta_path)).get("label", "object")
            except Exception:
                pass

    # Pick: when both sam_tight and inside_outside exist, ask Qwen to
    # compare their y0 renders and choose the cleanest. inside_outside is
    # the "most processed" stage but for compound objects (cabinet + items
    # on top) it can slice the body in half because the body's insideness
    # scores sit right on the chosen threshold. sam_tight often preserves
    # the whole assembly with mild floor halo. Let Qwen judge visually.
    # Falls back to hardcoded preference order when only one is present.
    by_tag = {c["tag"]: c for c in cands}
    chosen = None
    if "sam_tight" in by_tag and "inside_outside" in by_tag:
        pick_cands = []
        for tag in ("sam_tight", "inside_outside"):
            c = by_tag[tag]
            render = obj / "renders" / tag / "y0.png"
            if not render.exists():
                # Render falls back if a stage skipped renders
                tmp = obj / "diagnostics" / f"stage_pick_y0_{tag}.png"
                render_y0(c["ply"], tmp)
                render = tmp
            pick_cands.append({"tag": tag, "render": render, "ply": c["ply"],
                               "n_splats": c["n_splats"]})
        try:
            pick_idx, raw = qwen_pick(pick_cands, label, pipe_union)
            chosen = by_tag[pick_cands[pick_idx]["tag"]]
            print(f"[pick] qwen compared sam_tight vs inside_outside → "
                  f"{chosen['tag']} ({chosen['n_splats']:,} splats)  "
                  f"raw='{raw}'")
        except Exception as e:
            print(f"[pick] qwen comparison failed ({e}) — falling back "
                  f"to hardcoded order")
            chosen = None
    if chosen is None:
        for tag in PREFERRED_ORDER:
            if tag in by_tag:
                chosen = by_tag[tag]
                break
        if chosen is None:
            chosen = cands[0]
        print(f"[pick] hardcoded order picked: {chosen['tag']} "
              f"({chosen['n_splats']:,} splats)")

    # Copy chosen stage → 7_picked.ply (intermediate before destreak)
    picked_path = obj / "7_picked.ply"
    shutil.copy(chosen["ply"], picked_path)
    print(f"[save] {picked_path}")

    (obj / "final_pick.json").write_text(json.dumps({
        "picked_tag": chosen["tag"],
        "picked_ply": chosen["fname"],
        "picked_n_splats": chosen["n_splats"],
        "candidates": [{"tag": c["tag"], "ply": c["fname"],
                         "n_splats": c["n_splats"]} for c in cands],
        "pick_method": "qwen_or_hardcoded_preference",
        "preferred_order": PREFERRED_ORDER,
        "label": label,
    }, indent=2))

    # Render 7_picked canonical 5 (diagnostics for the pre-destreak stage)
    picked_renders = obj / "renders" / "7_picked"
    render_canonical_5(picked_path, picked_renders)
    print(f"[render] canonical 5 → {picked_renders}")

    # Stage 7 — destreak --auto: 4-way Qwen pick (skip + 3 thresholds) → 7_destreak.ply
    print(f"\n[stage7] destreak --auto on 7_picked.ply")
    destreak_rc = subprocess.run([
        sys.executable, str(ITERATION_DIR / "splat_destreak.py"),
        str(obj), "--auto",
        "--in-ply", "7_picked.ply",
        "--out-ply", "7_destreak.ply",
    ]).returncode
    if destreak_rc != 0:
        print(f"[stage7] destreak FAILED rc={destreak_rc} — falling back to 7_picked")
        destreak_path = picked_path
    else:
        destreak_path = obj / "7_destreak.ply"

    # Stage 8 — final: copy destreak → 8_final.ply, render, splat
    final_path = obj / "8_final.ply"
    shutil.copy(destreak_path, final_path)
    print(f"[stage8] {final_path}")

    renders_dir = obj / "renders" / "8_final"
    render_canonical_5(final_path, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    # Convert 8_final.ply → final.splat for web-viewer use.
    splat_path = obj / "final.splat"
    rc = subprocess.run([
        sys.executable, str(ITERATION_DIR / "ply_to_splat.py"),
        str(final_path), str(splat_path),
    ]).returncode
    if rc != 0:
        print(f"[splat] FAIL ply_to_splat returned {rc} — final.splat "
              f"not written")
    else:
        size_mb = splat_path.stat().st_size / 1024 / 1024
        print(f"[splat] {splat_path}  ({size_mb:.1f} MB)")

    # Write refined slug to a marker file. procedure_dispatch reads this
    # AFTER qc_gate + info + split_children finish, and does the actual
    # folder rename there. Renaming here used to break those downstream
    # steps because the obj_dir Path they held went stale (2026-05-27 fix).
    new_slug = slugify(label)
    (obj / "stage_pick_refined_slug.json").write_text(json.dumps({
        "label": label,
        "refined_slug": new_slug,
        "current_slug": obj.name[3:] if obj.name.startswith("02_") else obj.name,
        "rename_pending": new_slug not in ("object",) and
                          new_slug != (obj.name[3:] if obj.name.startswith("02_") else obj.name),
    }, indent=2))
    print(f"\n[rename] deferred to procedure_dispatch (refined_slug='{new_slug}')")

    print(f"\n[done] picked: {chosen['tag']} → 7_picked.ply → 7_destreak.ply → 8_final.ply + final.splat")


if __name__ == "__main__":
    main()
