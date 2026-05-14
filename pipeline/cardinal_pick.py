#!/usr/bin/env python3
"""cardinal_pick.py — pick the correct cardinal rotation via Qwen on a
temporary aggressive strip of the oriented PLY.

Runs in stages so the user can confirm each:
  Step 1: y-axis strip (ceiling p40 + floor cap + 7ft up cap) + density
          filter (5cm voxel, min 15). NO world-xz wall crop — that cuts
          diagonally when room still rotated.
          → render temp topdown → STOP for approval

  Step 2: render N camera-yaw candidates of the temp (PLY unchanged),
          send to Qwen with floor+furniture biased prompt, pick winner,
          apply chosen yaw to the FULL aligned PLY via plugin's
          y_axis_align.py (SH rotation via Wigner-D).
          → step7_cardinal_aligned.ply + final topdown render

Usage:
    python cardinal_pick.py <scene_dir> --step 1
    python cardinal_pick.py <scene_dir> --step 2
"""
import os
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

PLUGIN = Path("/home/ubuntu/.claude/local-plugins-marketplace/plugins/pointcloud-segmentation/scripts")
Y_AXIS_ALIGN = PLUGIN / "y_axis_align.py"
VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-awq")

FOV = 70.0
W, H = 1920, 1080
TOPDOWN_MARGIN = 3.0
SEVEN_FEET_M = 2.13
CEILING_PCTL = 40.0
DENSITY_VOXEL_M = 0.05
DENSITY_MIN_COUNT = 15

# Step 2 sweep: ±30° in 5° steps (13 candidates).
ANGLE_CANDIDATES = [float(a) for a in range(-30, 31, 5)]
ANGLE_LABELS = list("ABCDEFGHIJKLM")  # 13 letters for 13 candidates


def render_topdown_rotated(ply: Path, out: Path, yaw_deg: float = 0.0):
    """Render topdown with camera up vector rotated by yaw_deg around +y.
    PLY is NOT modified — image rotates within frame."""
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
    rad = math.radians(yaw_deg)
    up_x = -math.sin(rad)
    up_z = -math.cos(rad)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python", VIEW_PY, str(ply), str(out),
           f"--eye={cx:.4f},{yf - dist:.4f},{cz:.4f}",
           f"--target={cx + 0.001:.4f},{yf:.4f},{cz:.4f}",
           f"--up={up_x:.6f},0,{up_z:.6f}", "--y-down",
           "--fov", str(FOV), "--width", str(W), "--height", str(H)]
    subprocess.run(cmd, check=True, capture_output=True)


def encode_b64(p: Path, max_dim: int = 512) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def qwen_pick_rotation(images: dict) -> tuple:
    """Send candidates to Qwen. Prompt biased toward floor + furniture,
    ignore abstract shapes / dome scatter."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for lbl in images:
        content.append({"type": "text", "text": f"\nCandidate {lbl}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(images[lbl])}"}})
    content.append({"type": "text", "text":
        "Each candidate is a topdown render of the same room rotated by a "
        "different angle.\n\n"
        "Pick the candidate where the room is BEST aligned with the image axes.\n\n"
        "Base your decision ONLY on:\n"
        "  - Wood floor planks: should run HORIZONTALLY or VERTICALLY (parallel "
        "    to image edges), NOT diagonally.\n"
        "  - Rectangular furniture (sofas, tables, beds, cabinets, rugs, "
        "    countertops): edges should be parallel to image edges.\n\n"
        "IGNORE:\n"
        "  - Dark dome scatter / capture noise around the room edges\n"
        "  - Abstract / curved / decorative shapes\n"
        "  - Pillows, plants, round objects\n\n"
        "Reply on ONE LINE — the WINNER letter MUST come first:\n"
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


def step1_strip_and_render(scene: Path):
    """Aggressive strip on a temp copy of orient.py's aligned PLY.
    - Reads scene/orient_status.json to find the final aligned PLY
    - Drops top CEILING_PCTL% of y (ceiling region in y-down)
    - Drops outer WALL_PCTL% of x and z
    - Caps to 7ft above floor
    - Renders temp topdown
    - Returns temp PLY + render paths
    """
    status_path = scene / "orient_status.json"
    if not status_path.exists():
        sys.exit(f"[fatal] no orient_status.json — run orient.py first")
    status = json.load(open(status_path))
    aligned_ply = Path(status["final_aligned_ply"])
    floor_plane_path = Path(status["floor_plane_json"])
    if not aligned_ply.exists():
        sys.exit(f"[fatal] aligned PLY missing: {aligned_ply}")
    print(f"[step1] aligned PLY: {aligned_ply}")

    # Get floor y from floor_plane.json
    fp = json.load(open(floor_plane_path))
    pa, pb, pc, pd = fp["plane"]
    floor_y = -pd / pb
    print(f"[step1] floor y = {floor_y:.3f}")

    # Load + strip
    pl = PlyData.read(str(aligned_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[step1] input splats: {len(xyz):,}")

    # Ceiling cut: drop top CEILING_PCTL% of y (in y-down small y = top)
    y_ceil_cut = float(np.percentile(xyz[:, 1], CEILING_PCTL))
    # 7ft cap: in y-down, "above floor" = smaller y. Keep y >= floor_y - 7ft
    y_seven_ft = floor_y - SEVEN_FEET_M
    # Floor cap: in y-down, "below floor" = larger y. Drop everything below floor.
    keep_y = ((xyz[:, 1] > y_ceil_cut) & (xyz[:, 1] >= y_seven_ft) &
              (xyz[:, 1] <= floor_y))
    print(f"[step1] after ceiling+7ft+floor_cap: {int(keep_y.sum()):,} / {len(xyz):,}")

    # Density filter: voxelize at DENSITY_VOXEL_M, drop splats whose voxel
    # has < DENSITY_MIN_COUNT splats. Rotation-invariant. Erodes walls
    # (temp only) but cleanly drops dome scatter.
    xyz_keep = xyz[keep_y]
    voxel_idx = np.floor(xyz_keep / DENSITY_VOXEL_M).astype(np.int64)
    _, inverse, counts = np.unique(voxel_idx, axis=0,
                                    return_inverse=True, return_counts=True)
    voxel_count = counts[inverse]
    keep_density_local = voxel_count >= DENSITY_MIN_COUNT
    print(f"[step1] after density (vox={DENSITY_VOXEL_M}m, min={DENSITY_MIN_COUNT}): "
          f"{int(keep_density_local.sum()):,} / {len(xyz_keep):,}")

    # Combine into global mask
    keep = np.zeros(len(xyz), dtype=bool)
    keep_y_idx = np.where(keep_y)[0]
    keep[keep_y_idx[keep_density_local]] = True
    n_kept = int(keep.sum())
    print(f"[step1] FINAL kept: {n_kept:,} / {len(xyz):,}")

    out_dir = scene / "_cardinal_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_ply = out_dir / "temp_strip.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(temp_ply))
    temp_png = out_dir / "temp_strip_topdown.png"
    render_topdown(temp_ply, temp_png)
    print(f"\n[step1] DONE")
    print(f"  temp PLY:  {temp_ply}")
    print(f"  temp PNG:  {temp_png}")
    print(f"\n  STOP — review the topdown, then run with --step 2 to proceed")


def _sweep_and_pick(temp_ply: Path, cand_dir: Path, angles: list,
                    labels: list, pass_name: str) -> tuple:
    """Render N yaw candidates, send to Qwen, return (chosen_label,
    chosen_yaw, label_meta, reasoning)."""
    images = {}
    label_meta = []
    for lbl, ang in zip(labels, angles):
        png = cand_dir / f"{pass_name}_{lbl}_yaw{ang:+05.1f}.png"
        render_topdown_rotated(temp_ply, png, ang)
        images[lbl] = png
        label_meta.append({"label": lbl, "yaw_deg": ang, "png": str(png)})
        print(f"  [{pass_name} {lbl}] yaw={ang:+.1f}° → {png.name}")
    winner, reasoning = qwen_pick_rotation(images)
    chosen = next(c for c in label_meta if c["label"] == winner)
    return winner, chosen["yaw_deg"], label_meta, reasoning


def step2_qwen_sweep(scene: Path):
    """Two-pass sweep on temp PLY:
      Pass 1 (coarse): ±30° by 5° → Qwen picks winner
      Pass 2 (fine):   winner ± 4° by 1° → Qwen picks final
    Stops after fine pick. Does NOT modify the full aligned PLY.
    """
    temp_ply = scene / "_cardinal_temp" / "temp_strip.ply"
    if not temp_ply.exists():
        sys.exit(f"[fatal] temp PLY missing — run step 1 first: {temp_ply}")

    out_dir = scene / "_cardinal_temp"
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for f in cand_dir.glob("*.png"):
        f.unlink()

    # PASS 1: coarse ±30 by 5
    print(f"\n[step2] PASS 1 — coarse sweep ±30° by 5°")
    coarse_winner, coarse_yaw, coarse_meta, coarse_reason = _sweep_and_pick(
        temp_ply, cand_dir, ANGLE_CANDIDATES, ANGLE_LABELS, "coarse")
    print(f"[coarse winner] {coarse_winner}: yaw={coarse_yaw:+.1f}°")
    print(f"[coarse reason] {coarse_reason}")

    # PASS 2: fine ± 4 by 1 around coarse winner
    print(f"\n[step2] PASS 2 — fine sweep around {coarse_yaw:+.1f}° ± 4° by 1°")
    fine_angles = [coarse_yaw + d for d in range(-4, 5)]  # 9 candidates
    fine_labels = list("ABCDEFGHI")  # 9 letters
    fine_winner, fine_yaw, fine_meta, fine_reason = _sweep_and_pick(
        temp_ply, cand_dir, fine_angles, fine_labels, "fine")
    print(f"[fine winner] {fine_winner}: yaw={fine_yaw:+.1f}°")
    print(f"[fine reason] {fine_reason}")

    chosen_png = next(c["png"] for c in fine_meta if c["label"] == fine_winner)

    log = {"final_chosen_yaw_deg": fine_yaw,
           "coarse_pass": {"winner": coarse_winner, "yaw_deg": coarse_yaw,
                            "reasoning": coarse_reason, "candidates": coarse_meta},
           "fine_pass": {"winner": fine_winner, "yaw_deg": fine_yaw,
                          "reasoning": fine_reason, "candidates": fine_meta},
           "applied_to_full_ply": False}
    (out_dir / "cardinal_choice.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step2] DONE — STOP, no PLY modified")
    print(f"  final yaw:      {fine_yaw:+.1f}°")
    print(f"  winner image:   {chosen_png}")
    print(f"  choice log:     {out_dir / 'cardinal_choice.json'}")
    print(f"\n  Review the winner. If approved, step 3 will apply the angle to the full PLY.")


def step3_apply_to_full_ply(scene: Path):
    """Read chosen yaw from step 2's cardinal_choice.json. Apply to the
    FULL aligned PLY (orient.py output) via plugin's y_axis_align.py
    (SH rotation via Wigner-D Monte Carlo). Render confirmation topdown.
    """
    choice_path = scene / "_cardinal_temp" / "cardinal_choice.json"
    if not choice_path.exists():
        sys.exit(f"[fatal] no cardinal_choice.json — run step 2 first")
    choice = json.load(open(choice_path))
    chosen_yaw = float(choice["final_chosen_yaw_deg"])

    status_path = scene / "orient_status.json"
    if not status_path.exists():
        sys.exit(f"[fatal] no orient_status.json — run orient.py first")
    status = json.load(open(status_path))
    aligned_ply = Path(status["final_aligned_ply"])
    if not aligned_ply.exists():
        sys.exit(f"[fatal] aligned PLY missing: {aligned_ply}")

    # Camera-yaw rotation in render_topdown_rotated and PLY rotation in
    # y_axis_align use opposite sign conventions. Negate to match.
    apply_angle = -chosen_yaw

    out_ply = scene / "step7_cardinal_aligned.ply"
    print(f"[step3] input:  {aligned_ply}")
    print(f"[step3] output: {out_ply}")
    print(f"[step3] camera_yaw_picked: {chosen_yaw:+.4f}°  →  ply_rotation: {apply_angle:+.4f}°")

    if abs(apply_angle) < 0.01:
        print(f"[step3] angle≈0 — copying input as-is")
        shutil.copy(aligned_ply, out_ply)
    else:
        print(f"[step3] y_axis_align --angle-deg={apply_angle:.4f} (SH rotation)")
        cmd = ["python", str(Y_AXIS_ALIGN), str(aligned_ply), str(out_ply),
               "--angle-deg", f"{apply_angle:.4f}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[fatal] y_axis_align failed rc={r.returncode}")
            print(r.stderr[-1500:])
            sys.exit(r.returncode)
        print(r.stdout[-500:] if r.stdout else "")

    final_png = scene / "_cardinal_temp" / "step3_full_ply_topdown.png"
    render_topdown_rotated(out_ply, final_png, 0.0)

    choice["applied_to_full_ply"] = True
    choice["output_cardinal_ply"] = str(out_ply)
    choice["step3_topdown"] = str(final_png)
    choice_path.write_text(json.dumps(choice, indent=2))

    print(f"\n[step3] DONE")
    print(f"  cardinal-aligned PLY: {out_ply}")
    print(f"  confirmation render:  {final_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--step", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    if args.step == 1:
        step1_strip_and_render(scene)
    elif args.step == 2:
        step2_qwen_sweep(scene)
    else:
        step3_apply_to_full_ply(scene)


if __name__ == "__main__":
    main()
