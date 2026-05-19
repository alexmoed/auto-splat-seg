#!/usr/bin/env python3
"""cleanup.py — conservative cleanup on the cardinal-aligned PLY.

Input:  <scene>/step7_cardinal_aligned.ply  (locked, never modified)
Output per step.

  Step 1: Qwen-picked voxel density filter (port of snapshot's
          auto_dome_density.py). Bins splats into 10cm cubes, computes
          per-splat voxel occupancy, sweeps min-density [3, 10, 25, 60, 150],
          renders topdown of each, Qwen picks the threshold where dome is
          gone but room is intact. SH-SAFE — only masks splats, no rotation.
          Output: step8_density_filtered.ply

NO obliteration of walls, ceiling, furniture, or wall items.

Usage:
    python cleanup.py <scene_dir> --step 1
"""
import argparse
import base64
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from openai import OpenAI
from plyfile import PlyData, PlyElement

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

FOV = 70.0
W, H = 1920, 1080
TOPDOWN_MARGIN = 3.0

# Step 1 — density filter
VOXEL_SIZE_M = 0.10
DENSITY_CANDIDATES = [3, 10, 25, 60, 150]
DENSITY_LABELS = list("ABCDE")


def encode_b64(p: Path, max_dim: int = 512) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_topdown(ply: Path, out: Path):
    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    x_lo, z_lo = np.percentile(xyz[:, [0, 2]], 15, axis=0)
    x_hi, z_hi = np.percentile(xyz[:, [0, 2]], 85, axis=0)
    cx, cz = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    xe, ze = float(x_hi - x_lo), float(z_hi - z_lo)
    yf = float(np.percentile(xyz[:, 1], 85))
    aspect = W / H
    tan_h = math.tan(math.radians(FOV / 2))
    dist = max((xe * TOPDOWN_MARGIN) / (2 * tan_h * aspect),
               (ze * TOPDOWN_MARGIN) / (2 * tan_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python", VIEW_PY, str(ply), str(out),
           f"--eye={cx:.4f},{yf - dist:.4f},{cz:.4f}",
           f"--target={cx + 0.001:.4f},{yf:.4f},{cz:.4f}",
           "--up=0,0,-1", "--y-down", "--fov", str(FOV),
           "--width", str(W), "--height", str(H)]
    subprocess.run(cmd, check=True, capture_output=True)


def voxel_density(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Per-splat voxel occupancy count."""
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(vox, axis=0, return_inverse=True,
                                    return_counts=True)
    return counts[inverse]


def qwen_pick_density(images: dict) -> tuple:
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for lbl in images:
        content.append({"type": "text", "text": f"\nCandidate {lbl}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(images[lbl])}"}})
    content.append({"type": "text", "text":
        "Each candidate is a topdown render of the same room with different "
        "amounts of sparse-density scatter (scan dome / capture noise) "
        "filtered out.\n\n"
        "Pick the candidate where:\n"
        "  1. The dark / fluffy / scatter noise around and above the room "
        "     is GONE — only the room body is visible against a clean "
        "     white background.\n"
        "  2. The walls + windows + floor + furniture are FULLY INTACT — "
        "     nothing carved off the room body itself.\n\n"
        "If multiple satisfy both, pick the LEAST aggressive (earliest "
        "letter A < B < C ...).\n\n"
        "Reply on ONE LINE — WINNER FIRST:\n"
        "WINNER=<letter>; REASON=<short sentence>"})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=120, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    out = r.choices[0].message.content.strip()
    print(f"[qwen] {out}")
    for lbl in images:
        if f"WINNER={lbl}" in out.upper():
            return lbl, out
    fallback = list(images.keys())[len(images) // 2]
    print(f"[qwen] couldn't parse — fallback={fallback}")
    return fallback, out


def get_input_ply(scene: Path) -> Path:
    p = scene / "step7_cardinal_aligned.ply"
    if not p.exists():
        sys.exit(f"[fatal] missing input: {p}\n  run cardinal_pick.py through step 3 first")
    return p


def step1_density_filter(scene: Path):
    in_ply = get_input_ply(scene)
    out_dir = scene / "_cleanup_temp"
    cand_dir = out_dir / "density_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for f in cand_dir.glob("*.png"):
        f.unlink()

    print(f"[step1] input: {in_ply}")
    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[step1] {len(xyz):,} splats")

    print(f"[step1] computing voxel density at {VOXEL_SIZE_M}m...")
    density = voxel_density(xyz, VOXEL_SIZE_M)
    print(f"[step1] density: median={int(np.median(density))} max={int(density.max())}")

    images = {}
    label_meta = []
    for lbl, min_d in zip(DENSITY_LABELS, DENSITY_CANDIDATES):
        keep = density >= min_d
        ply = cand_dir / f"{lbl}_dens{min_d}.ply"
        new_v = v.data[keep]
        PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(ply))
        png = cand_dir / f"{lbl}_dens{min_d}.png"
        render_topdown(ply, png)
        images[lbl] = png
        label_meta.append({"label": lbl, "min_density": min_d,
                           "n_kept": int(keep.sum()), "n_total": len(keep),
                           "ply": str(ply), "png": str(png)})
        print(f"  [{lbl}] min_density={min_d}: kept {int(keep.sum()):,}/{len(keep):,}")

    winner, reasoning = qwen_pick_density(images)
    chosen = next(c for c in label_meta if c["label"] == winner)
    print(f"[winner] {winner}: min_density={chosen['min_density']}, kept {chosen['n_kept']:,}")

    out_ply = scene / "step8_density_filtered.ply"
    shutil.copy(chosen["ply"], out_ply)

    log = {"winner": winner, "reasoning": reasoning,
           "voxel_size_m": VOXEL_SIZE_M,
           "chosen_min_density": chosen["min_density"],
           "input_ply": str(in_ply),
           "output_ply": str(out_ply),
           "candidates": label_meta}
    (out_dir / "density_choice.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step1] DONE")
    print(f"  output PLY:  {out_ply}")
    print(f"  winner png:  {chosen['png']}")
    print(f"  choice log:  {out_dir / 'density_choice.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--step", type=int, required=True, choices=[1])
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    if args.step == 1:
        step1_density_filter(scene)


if __name__ == "__main__":
    main()
