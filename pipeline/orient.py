#!/usr/bin/env python3
"""orient.py — bring a raw gsplat PLY into axis-aligned y-down state.

No audits, no destructive percentile picks. Just orientation.

Pipeline:
  1. Vertical axis detection (smallest extent = vertical) → rotate if z/x-up
  2. remove_shell (defaults: ceiling p10, walls preserved) → scene_after.ply
  3. If orient_ok=false → tilt_correct → re-shell
  4. y_axis_align (PCA on floor splats) → re-shell
  5. Render topdown
  6. Hough residual angle on topdown → apply corrective rotation, re-shell
  7. Re-render topdown
  8. Qwen yaw sweep (5 candidates at 1920×1080) — loop until Qwen picks center
  9. Preference rotation: 90° if z-extent > x-extent (longer-dim-horizontal)
  10. Y-vertical assertion + auto-correct (if X or Z became vertical, rotate back)
  11. Render canonical topdown

Output: <scene>/01_shell_removed/scene_after.ply + floor_plane.json
        <scene>/01_shell_removed/views/topdown.png
        <scene>/orient_status.json (chain log)

Usage:
    python orient.py <scene_dir> [--raw-ply <path>]
"""
import argparse
import base64
import io
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image
from plyfile import PlyData

PLUGIN = Path("/home/ubuntu/.claude/local-plugins-marketplace/plugins/pointcloud-segmentation/scripts")
ROTATE_ZUP = PLUGIN / "rotate_zup_to_ydown.py"
REMOVE_SHELL = PLUGIN / "remove_shell.py"
TILT_CORRECT = PLUGIN / "tilt_correct.py"
Y_AXIS_ALIGN = PLUGIN / "y_axis_align.py"
VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"

import os as _os
QWEN_URL = _os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = _os.environ.get("QWEN_MODEL", "qwen36-awq")

FOV = 70.0
TOPDOWN_W, TOPDOWN_H = 1920, 1080
TOPDOWN_MARGIN = 3.0
QWEN_SWEEP_RANGES = [20.0, 8.0, 3.0, 1.0]  # narrowing each round
MAX_QWEN_ATTEMPTS = 5
HOUGH_THRESH_DEG = 0.5  # below this Hough doesn't apply correction


# ────────────────────────── helpers ──────────────────────────


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"[fatal] {' '.join(str(c) for c in cmd[:4])}... rc={r.returncode}")
        if r.stderr:
            print(r.stderr[-1500:])
        sys.exit(r.returncode)
    return r


def detect_vertical_axis(ply_path: Path) -> str:
    """Return 'x' / 'y' / 'z' — axis with smallest p15-85 extent (= vertical).

    Uses percentile range, not full min-max, so dome/scatter outliers
    (which can reach 10+ km on raw scans) don't dominate the comparison.
    """
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"].data
    exts = {a: float(np.percentile(v[a], 85) - np.percentile(v[a], 15))
            for a in ("x", "y", "z")}
    print(f"[detect] p15-85 extents x={exts['x']:.2f} y={exts['y']:.2f} z={exts['z']:.2f}")
    return min(exts, key=exts.get)


def shell_remove(in_ply: Path, scene_dir: Path) -> tuple:
    """Run plugin's remove_shell.py with defaults. Returns (scene_after, floor_plane)."""
    out_dir = scene_dir / "01_shell_removed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "scene_after.ply"
    floor_json = out_dir / "floor_plane.json"
    print(f"[shell] {in_ply.name} → scene_after.ply")
    run([sys.executable, str(REMOVE_SHELL), str(in_ply), str(out_ply)])
    return out_ply, floor_json


def render_topdown(ply: Path, out: Path, yaw_deg: float = 0.0):
    """Render axis-aligned topdown. yaw_deg rotates the camera up vector
    (used for Qwen yaw sweep — visualizes a rotation without rewriting the PLY)."""
    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    x_lo, z_lo = np.percentile(xyz[:, [0, 2]], 15, axis=0)
    x_hi, z_hi = np.percentile(xyz[:, [0, 2]], 85, axis=0)
    cx, cz = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    xe, ze = float(x_hi - x_lo), float(z_hi - z_lo)
    yf = float(np.percentile(xyz[:, 1], 85))
    aspect = TOPDOWN_W / TOPDOWN_H
    tan_h = math.tan(math.radians(FOV / 2))
    dist = max((xe * TOPDOWN_MARGIN) / (2 * tan_h * aspect),
               (ze * TOPDOWN_MARGIN) / (2 * tan_h))
    rad = math.radians(yaw_deg)
    up_x = -math.sin(rad)
    up_z = -math.cos(rad)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, VIEW_PY, str(ply), str(out),
           f"--eye={cx:.4f},{yf - dist:.4f},{cz:.4f}",
           f"--target={cx + 0.001:.4f},{yf:.4f},{cz:.4f}",
           f"--up={up_x:.6f},0,{up_z:.6f}", "--y-down",
           "--fov", str(FOV), "--width", str(TOPDOWN_W), "--height", str(TOPDOWN_H)]
    subprocess.run(cmd, check=True, capture_output=True)


def hough_residual_deg(topdown_path: Path) -> float:
    """Find dominant line angle on topdown via Hough. Returns residual mod 90 in [-45, 45]."""
    img = cv2.imread(str(topdown_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0.0
    angles = []
    for r in lines[:200]:
        _, theta = r[0]
        deg = math.degrees(theta)
        angles.append(((deg + 45) % 90) - 45)
    return float(np.median(angles)) if angles else 0.0


def encode_b64(p: Path) -> str:
    """Encode image as base64 — full resolution (no downscale)."""
    img = Image.open(p).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def qwen_yaw_pick(images: dict) -> tuple:
    """Show 5 yaw candidates to Qwen at 1920×1080. Returns (winner_label, raw_response)."""
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for lbl, p in images.items():
        content.append({"type": "text", "text": f"\nCandidate {lbl}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    content.append({"type": "text", "text":
        "Each candidate is the same room rotated by a different small angle. "
        "Pick the candidate where the room walls and wood-floor planks run "
        "MOST PARALLEL to the image edges — perfectly horizontal AND vertical "
        "lines, NO diagonal tilt. Reply on ONE LINE — WINNER FIRST:\n"
        "WINNER=<letter>; REASON=<short>"})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=120, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    out = r.choices[0].message.content.strip()
    print(f"[qwen-yaw] {out}")
    for lbl in images:
        if f"WINNER={lbl}" in out.upper():
            return lbl, out
    return list(images.keys())[len(images) // 2], out


# ────────────────────────── main ──────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--raw-ply", type=Path, default=None,
                    help="default: <scene>/raw_ydown.ply or <scene>/raw.ply")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    scene.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chain = []

    # Locate raw PLY
    raw = args.raw_ply
    if raw is None:
        for n in ("raw_ydown.ply", "raw.ply"):
            cand = scene / n
            if cand.exists():
                raw = cand
                break
    if raw is None or not raw.exists():
        sys.exit(f"[fatal] no raw PLY at {scene}/raw_ydown.ply or raw.ply")
    print(f"[init] scene={scene}  raw={raw}")
    chain.append({"step": "init", "raw_ply": str(raw)})

    # 1 — vertical axis
    vertical = detect_vertical_axis(raw)
    current_ply = raw
    if vertical in ("x", "z"):
        rotated = scene / "step1_ydown.ply"
        print(f"[1] {vertical}-up detected → rotating to y-down")
        run([sys.executable, str(ROTATE_ZUP), str(raw), str(rotated),
             "--from-axis", vertical])
        current_ply = rotated
    chain.append({"step": "vertical_check", "vertical_axis": vertical,
                  "rotated": vertical != "y", "current_ply": str(current_ply)})

    # 2 — preliminary shell removal (ceiling p10 only — walls preserved)
    print(f"[2] preliminary remove_shell")
    scene_after, floor_json = shell_remove(current_ply, scene)
    fp = json.load(open(floor_json))
    chain.append({"step": "preshell", "orient_ok": fp.get("orient_ok"),
                  "normal_unit": fp.get("normal_unit")})

    # 3 — tilt-correct if needed
    if not fp.get("orient_ok", False):
        print(f"[3] orient_ok=false → tilt_correct")
        tilted = scene / "step3_tilt_corrected.ply"
        run([sys.executable, str(TILT_CORRECT), str(current_ply), str(tilted),
             "--plane", str(floor_json)])
        current_ply = tilted
        scene_after, floor_json = shell_remove(current_ply, scene)
        chain.append({"step": "tilt_correct", "current_ply": str(current_ply)})
    else:
        print(f"[3] tilt OK — skipping")
        chain.append({"step": "tilt_correct", "skipped": True})

    # 4 — y-axis align via PCA
    print(f"[4] y_axis_align (PCA auto)")
    yaligned = scene / "step4_yaligned.ply"
    run([sys.executable, str(Y_AXIS_ALIGN), str(current_ply), str(yaligned)])
    current_ply = yaligned
    scene_after, floor_json = shell_remove(current_ply, scene)
    chain.append({"step": "y_align_pca", "current_ply": str(current_ply)})

    # 5 — render topdown for Hough
    topdown = scene / "01_shell_removed" / "views" / "topdown.png"
    render_topdown(scene_after, topdown)

    # 6 — Hough bulk correction
    residual = hough_residual_deg(topdown)
    print(f"[6] Hough residual = {residual:+.3f}°")
    chain.append({"step": "hough", "residual_deg": residual})
    if abs(residual) > HOUGH_THRESH_DEG:
        print(f"[6] applying corrective rotation -{residual:.3f}°")
        rotated = scene / "step6_hough_corrected.ply"
        run([sys.executable, str(Y_AXIS_ALIGN), str(current_ply), str(rotated),
             "--angle-deg", f"{-residual:.4f}"])
        current_ply = rotated
        scene_after, floor_json = shell_remove(current_ply, scene)
        render_topdown(scene_after, topdown)

    # 7-8 — Qwen yaw sweep convergence
    cumulative_yaw = 0.0
    for attempt in range(1, MAX_QWEN_ATTEMPTS + 1):
        sweep = QWEN_SWEEP_RANGES[min(attempt - 1, len(QWEN_SWEEP_RANGES) - 1)]
        deltas = [-sweep, -sweep / 2, 0.0, sweep / 2, sweep]
        labels = ["A", "B", "C", "D", "E"]
        cand_dir = scene / "01_shell_removed" / "views" / f"_yaw_round{attempt}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        print(f"[7] yaw sweep round {attempt}, ±{sweep:.2f}°")
        candidates = {}
        for lbl, d in zip(labels, deltas):
            png = cand_dir / f"{lbl}_d{d:+.3f}.png"
            render_topdown(scene_after, png, yaw_deg=d)
            candidates[lbl] = png
        winner, reasoning = qwen_yaw_pick(candidates)
        winner_delta = deltas[labels.index(winner)]
        chain.append({"step": "qwen_yaw_sweep", "round": attempt,
                      "sweep_range": sweep, "winner_label": winner,
                      "winner_delta": winner_delta, "reasoning": reasoning})
        if abs(winner_delta) < 1e-6:
            print(f"[7] converged at round {attempt}")
            break
        cumulative_yaw += winner_delta
        rotated = scene / f"step7_yaw_round{attempt}.ply"
        run([sys.executable, str(Y_AXIS_ALIGN), str(current_ply), str(rotated),
             "--angle-deg", f"{winner_delta:.4f}"])
        current_ply = rotated
        scene_after, floor_json = shell_remove(current_ply, scene)
        render_topdown(scene_after, topdown)

    # 9 — preference rotation (longer-dim-horizontal)
    pl = PlyData.read(str(current_ply))
    vx = np.asarray(pl["vertex"]["x"])
    vz = np.asarray(pl["vertex"]["z"])
    xe = float(np.percentile(vx, 85) - np.percentile(vx, 15))
    ze = float(np.percentile(vz, 85) - np.percentile(vz, 15))
    print(f"[9] room extents x={xe:.2f}m z={ze:.2f}m")
    if ze > xe:
        print(f"[9] z>x → applying 90° preference rotation")
        rotated = scene / "step9_pref_rotated.ply"
        run([sys.executable, str(Y_AXIS_ALIGN), str(current_ply), str(rotated),
             "--angle-deg", "90.0"])
        current_ply = rotated
        scene_after, floor_json = shell_remove(current_ply, scene)
        chain.append({"step": "preference_rotation", "applied_deg": 90,
                      "x_ext": xe, "z_ext": ze})
    else:
        chain.append({"step": "preference_rotation", "skipped": True,
                      "x_ext": xe, "z_ext": ze})

    # 10 — Y-vertical assertion + auto-correct (locked 2026-05-27).
    # After every rotation in the chain, Y MUST still be the smallest
    # extent (= the world vertical axis). The Qwen yaw-sweep (step 8) and
    # preference rotation (step 9) are non-deterministic and can leave the
    # scene with X or Z as vertical. That breaks every downstream script
    # (sam_carve, dioramas, floor_drop, inside_outside all hard-assume
    # y-down) and packages raw bbox cones as "final" outputs.
    # This step catches that and auto-rotates back to y-down.
    def _ext(arr):
        return float(np.percentile(arr, 85) - np.percentile(arr, 15))

    pl_chk = PlyData.read(str(current_ply))
    xe_f = _ext(np.asarray(pl_chk["vertex"]["x"]))
    ye_f = _ext(np.asarray(pl_chk["vertex"]["y"]))
    ze_f = _ext(np.asarray(pl_chk["vertex"]["z"]))
    print(f"[10] final-axis check: p15-85 x={xe_f:.2f} y={ye_f:.2f} z={ze_f:.2f}")
    extents = {"x": xe_f, "y": ye_f, "z": ze_f}
    vertical_now = min(extents, key=extents.get)
    if vertical_now != "y":
        print(f"[10] WRONG — {vertical_now} is smallest (vertical), should be y. auto-correcting.")
        corrected = scene / "step10_axis_corrected.ply"
        run([sys.executable, str(ROTATE_ZUP), str(current_ply), str(corrected),
             "--from-axis", vertical_now])
        current_ply = corrected
        scene_after, floor_json = shell_remove(current_ply, scene)
        pl_chk = PlyData.read(str(current_ply))
        xe_p = _ext(np.asarray(pl_chk["vertex"]["x"]))
        ye_p = _ext(np.asarray(pl_chk["vertex"]["y"]))
        ze_p = _ext(np.asarray(pl_chk["vertex"]["z"]))
        print(f"[10] post-correct: x={xe_p:.2f} y={ye_p:.2f} z={ze_p:.2f}")
        if not (ye_p < xe_p and ye_p < ze_p):
            chain.append({"step": "y_vertical_assert", "passed": False,
                          "before": extents,
                          "after_correct": {"x": xe_p, "y": ye_p, "z": ze_p},
                          "tried_rotate_from": vertical_now})
            (scene / "orient_status.json").write_text(json.dumps(
                {"chain": chain, "fatal": "axis-correction failed"}, indent=2))
            sys.exit(f"[FATAL] orient axis-correction failed. After rotating "
                     f"from {vertical_now}-up to y-down, y={ye_p:.2f}m is still "
                     f"not the smallest extent (x={xe_p:.2f}, z={ze_p:.2f}). "
                     f"Manual intervention required.")
        chain.append({"step": "y_vertical_assert", "passed": True,
                      "auto_corrected_from": vertical_now,
                      "before": extents,
                      "after_correct": {"x": xe_p, "y": ye_p, "z": ze_p}})
    else:
        chain.append({"step": "y_vertical_assert", "passed": True,
                      "auto_corrected_from": None, "extents": extents})

    # 11 — final canonical topdown render
    render_topdown(scene_after, topdown)

    # Status
    elapsed = time.time() - t0
    status = {
        "scene": str(scene),
        "raw_ply": str(raw),
        "final_aligned_ply": str(current_ply),
        "scene_after_ply": str(scene_after),
        "floor_plane_json": str(floor_json),
        "topdown_png": str(topdown),
        "elapsed_s": round(elapsed, 1),
        "chain": chain,
    }
    status_path = scene / "orient_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(f"\n=== orient done ({elapsed/60:.1f} min) ===")
    print(f"  aligned PLY:  {current_ply}")
    print(f"  scene_after:  {scene_after}")
    print(f"  topdown:      {topdown}")
    print(f"  status:       {status_path}")


if __name__ == "__main__":
    main()
