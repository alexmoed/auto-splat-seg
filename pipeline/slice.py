#!/usr/bin/env python3
"""slice.py — geometric pre-cleanup slicing on the cardinal-aligned PLY.

Runs BEFORE cleanup.py. Conservative, deterministic, no Qwen.

Input:  <scene>/step7_cardinal_aligned.ply  (locked, never modified)
Output: <scene>/step7_sliced.ply             (passed to cleanup.py later)

Steps:
  Step 1: drop top 8% of y (ceiling region) AND drop everything below
          floor_y + 0.35m. Floor plane read from orient_status.json →
          floor_plane.json.

  Step 2: max-scale filter on step7_sliced.ply. Drops splats whose
          largest axis scale exceeds MAX_SCALE_M (default 0.5m). Targets
          oversized noise blobs (dome scatter, capture artifacts).

  Step 3: anisotropy filter on step7_sliced.ply. Drops splats whose
          max/min scale ratio exceeds ANISO_RATIO (default 20). Restricted
          to top ANISO_TOP_PCTL% of y (default 10%). Targets stretched
          dome/halo splats above the room body.

  Step 4: aggressive max-scale filter in top band only. Drops splats
          whose max axis scale exceeds TOP_MAX_SCALE_M (default 0.10m),
          restricted to top TOP_SCALE_PCTL% of y (default 10%).

  Step 5: density filter in top band only. Voxelizes at DENS_VOXEL_M
          (default 10cm), drops top-band splats whose voxel has <
          DENS_MIN (default 25) splats.

  Step 6: local SH (DC color) gradient outliers in TOP BAND ONLY.
          Drops splats above OUTLIER_PCTL distance (default p95) within
          top SH_TOP_PCTL% of y (default 10%).

  Step 7: scene-wide SH outlier (NO band restriction). Conservative:
          drops only above SH_ALL_PCTL distance (default p99 = top 1%).

All steps write back to step7_sliced.ply.

NO obliteration of walls, ceiling-mounted items, or wall art.

Usage:
    python slice.py <scene_dir> --step {1..6}
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"

FOV = 70.0
W, H = 1920, 1080
TOPDOWN_MARGIN = 3.0

# Step 1
TOP_PCTL = 10.0          # heavy 10% ceiling cut → step7_sliced.ply (used by
                          # inventory.py for Qwen identification — cleaner view
                          # without ceiling clutter).
TOP_PCTL_EXTRACT = 3.0   # light 3% ceiling cut → step7_sliced_extract.ply
                          # (used by extract_one for back-projection — keeps
                          # tall bookshelf / cabinet tops).
FLOOR_BUFFER_M = 0.35    # keep splats up to floor_y + 35cm (preserve floor thickness)

# Step 2
MAX_SCALE_M = 0.5        # drop splats with max axis scale > 0.5m (oversized noise)

# Step 3
ANISO_RATIO = 20.0       # drop splats with max/min scale ratio > 20 (extreme stretch)
ANISO_TOP_PCTL = 10.0    # only apply aniso filter to top 10% of y (upper region)

# Step 4
TOP_MAX_SCALE_M = 0.10   # aggressive: drop splats with max axis scale > 10cm
TOP_SCALE_PCTL = 10.0    # only apply to top 10% of y

# Step 5
DENS_VOXEL_M = 0.10      # 10cm voxel
DENS_MIN = 25            # drop top-band splats whose voxel has < 25 splats
DENS_TOP_PCTL = 10.0     # only apply to top 10% of y

# Step 6
SH_VOXEL_M = 0.20        # 20cm voxel for color-mean aggregation
OUTLIER_PCTL = 95.0      # drop splats above this distance percentile
SH_TOP_PCTL = 10.0       # only apply to top 10% of y

# Step 7
SH_ALL_PCTL = 99.0       # scene-wide: drop only above p99 (top 1% most extreme)


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


def get_input_ply(scene: Path) -> Path:
    p = scene / "step7_cardinal_aligned.ply"
    if not p.exists():
        sys.exit(f"[fatal] missing input: {p}\n  run cardinal_pick.py through step 3 first")
    return p


def step1_top_and_below_floor(scene: Path):
    in_ply = get_input_ply(scene)

    # Get floor_y from orient_status.json → floor_plane.json
    status_path = scene / "orient_status.json"
    if not status_path.exists():
        sys.exit(f"[fatal] no orient_status.json — run orient.py first")
    status = json.load(open(status_path))
    fp_path = Path(status["floor_plane_json"])
    fp = json.load(open(fp_path))
    pa, pb, pc, pd = fp["plane"]
    floor_y = -pd / pb
    print(f"[step1] floor y = {floor_y:.3f}")

    # Load PLY
    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[step1] {len(xyz):,} splats")

    # Two top cuts: heavy 10% for identification (Qwen topdown), light
    # 3% for back-projection (preserves tall bookshelf / cabinet tops).
    y_top_cut_id = float(np.percentile(xyz[:, 1], TOP_PCTL))
    y_top_cut_extract = float(np.percentile(xyz[:, 1], TOP_PCTL_EXTRACT))
    print(f"[step1] top-{TOP_PCTL}% y cutoff (id):      {y_top_cut_id:.3f}")
    print(f"[step1] top-{TOP_PCTL_EXTRACT}% y cutoff (extract): {y_top_cut_extract:.3f}")

    # Below floor: drop y > floor_y + FLOOR_BUFFER_M
    floor_cut = floor_y + FLOOR_BUFFER_M
    print(f"[step1] floor cutoff (with {FLOOR_BUFFER_M}m buffer): {floor_cut:.3f}")

    keep_id = (xyz[:, 1] >= y_top_cut_id) & (xyz[:, 1] <= floor_cut)
    keep_extract = (xyz[:, 1] >= y_top_cut_extract) & (xyz[:, 1] <= floor_cut)
    n_id = int(keep_id.sum())
    n_extract = int(keep_extract.sum())
    print(f"[step1] id PLY:      kept {n_id:,} / {len(xyz):,}")
    print(f"[step1] extract PLY: kept {n_extract:,} / {len(xyz):,}")

    out_ply_id = scene / "step7_sliced.ply"
    PlyData([PlyElement.describe(v.data[keep_id], "vertex")],
            text=False).write(str(out_ply_id))
    out_ply_extract = scene / "step7_sliced_extract.ply"
    PlyData([PlyElement.describe(v.data[keep_extract], "vertex")],
            text=False).write(str(out_ply_extract))

    out_dir = scene / "_slice_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(out_ply_id, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply_id": str(out_ply_id),
           "output_ply_extract": str(out_ply_extract),
           "floor_y": floor_y,
           "top_pctl_id": TOP_PCTL,
           "top_pctl_extract": TOP_PCTL_EXTRACT,
           "y_top_cut_id": y_top_cut_id,
           "y_top_cut_extract": y_top_cut_extract,
           "floor_buffer_m": FLOOR_BUFFER_M,
           "floor_cut": floor_cut,
           "n_kept_id": n_id,
           "n_kept_extract": n_extract,
           "n_total": len(xyz)}
    (out_dir / "slice_step1.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step1] DONE")
    print(f"  id PLY:      {out_ply_id}")
    print(f"  extract PLY: {out_ply_extract}")
    print(f"  topdown:     {out_png}")


def step2_max_scale_filter(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    if "scale_0" not in v.data.dtype.names:
        sys.exit(f"[fatal] PLY has no scale_0/1/2 fields — not a gsplat PLY")

    scales_log = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                           axis=1).astype(np.float64)
    max_log = np.max(scales_log, axis=1)
    max_scale_m = np.exp(max_log)
    print(f"[step2] max_scale stats: median={float(np.median(max_scale_m)):.3f}m "
          f"p95={float(np.percentile(max_scale_m, 95)):.3f}m "
          f"max={float(max_scale_m.max()):.3f}m")
    print(f"[step2] threshold: max_axis_scale <= {MAX_SCALE_M}m")

    keep = max_scale_m <= MAX_SCALE_M
    n_kept = int(keep.sum())
    print(f"[step2] kept {n_kept:,} / {len(scales_log):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "max_scale_m_threshold": MAX_SCALE_M,
           "n_kept": n_kept,
           "n_total": len(scales_log)}
    (out_dir / "slice_step2.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step2] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def step3_anisotropy_filter(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    if "scale_0" not in v.data.dtype.names:
        sys.exit(f"[fatal] PLY has no scale_0/1/2 fields — not a gsplat PLY")

    scales_log = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                           axis=1).astype(np.float64)
    scales_m = np.exp(scales_log)
    max_s = np.max(scales_m, axis=1)
    min_s = np.min(scales_m, axis=1)
    ratio = max_s / np.maximum(min_s, 1e-9)
    print(f"[step3] aniso ratio stats: median={float(np.median(ratio)):.2f} "
          f"p95={float(np.percentile(ratio, 95)):.2f} "
          f"max={float(ratio.max()):.2f}")

    # Restrict aniso filter to top ANISO_TOP_PCTL% of y (in y-down: smallest y)
    y = np.array(v["y"]).astype(np.float64)
    y_cutoff = float(np.percentile(y, ANISO_TOP_PCTL))
    in_top_band = y < y_cutoff
    print(f"[step3] aniso filter scope: top {ANISO_TOP_PCTL}% of y "
          f"(y < {y_cutoff:.3f}) — {int(in_top_band.sum()):,} splats")
    print(f"[step3] threshold: ratio <= {ANISO_RATIO} (only in top band)")

    drop = in_top_band & (ratio > ANISO_RATIO)
    keep = ~drop
    n_dropped = int(drop.sum())
    n_kept = int(keep.sum())
    print(f"[step3] dropped {n_dropped:,} (top-band aniso outliers)")
    print(f"[step3] kept {n_kept:,} / {len(scales_log):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "aniso_ratio_threshold": ANISO_RATIO,
           "n_kept": n_kept,
           "n_total": len(scales_log)}
    (out_dir / "slice_step3.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step3] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def step4_top_band_max_scale(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    if "scale_0" not in v.data.dtype.names:
        sys.exit(f"[fatal] PLY has no scale_0/1/2 fields — not a gsplat PLY")

    scales_log = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                           axis=1).astype(np.float64)
    max_scale_m = np.exp(np.max(scales_log, axis=1))
    print(f"[step4] max_scale stats: median={float(np.median(max_scale_m)):.3f}m "
          f"p95={float(np.percentile(max_scale_m, 95)):.3f}m "
          f"max={float(max_scale_m.max()):.3f}m")

    y = np.array(v["y"]).astype(np.float64)
    y_cutoff = float(np.percentile(y, TOP_SCALE_PCTL))
    in_top_band = y < y_cutoff
    print(f"[step4] scale filter scope: top {TOP_SCALE_PCTL}% of y "
          f"(y < {y_cutoff:.3f}) — {int(in_top_band.sum()):,} splats")
    print(f"[step4] threshold: max_axis_scale <= {TOP_MAX_SCALE_M}m (only in top band)")

    drop = in_top_band & (max_scale_m > TOP_MAX_SCALE_M)
    keep = ~drop
    n_dropped = int(drop.sum())
    n_kept = int(keep.sum())
    print(f"[step4] dropped {n_dropped:,} (top-band oversized)")
    print(f"[step4] kept {n_kept:,} / {len(scales_log):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "top_max_scale_m": TOP_MAX_SCALE_M,
           "top_scale_pctl": TOP_SCALE_PCTL,
           "n_kept": n_kept,
           "n_total": len(scales_log)}
    (out_dir / "slice_step4.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step4] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def step5_top_band_density(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    print(f"[step5] {len(xyz):,} splats")
    print(f"[step5] computing voxel density at {DENS_VOXEL_M}m...")
    vox = np.floor(xyz / DENS_VOXEL_M).astype(np.int64)
    _, inverse, counts = np.unique(vox, axis=0, return_inverse=True,
                                    return_counts=True)
    density = counts[inverse]
    print(f"[step5] density stats: median={int(np.median(density))} "
          f"max={int(density.max())}")

    y = xyz[:, 1]
    y_cutoff = float(np.percentile(y, DENS_TOP_PCTL))
    in_top_band = y < y_cutoff
    print(f"[step5] density filter scope: top {DENS_TOP_PCTL}% of y "
          f"(y < {y_cutoff:.3f}) — {int(in_top_band.sum()):,} splats")
    print(f"[step5] threshold: density >= {DENS_MIN} (only in top band)")

    drop = in_top_band & (density < DENS_MIN)
    keep = ~drop
    n_dropped = int(drop.sum())
    n_kept = int(keep.sum())
    print(f"[step5] dropped {n_dropped:,} (top-band sparse)")
    print(f"[step5] kept {n_kept:,} / {len(xyz):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "voxel_size_m": DENS_VOXEL_M,
           "min_density": DENS_MIN,
           "top_pctl": DENS_TOP_PCTL,
           "n_kept": n_kept,
           "n_total": len(xyz)}
    (out_dir / "slice_step5.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step5] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def step6_local_sh_outliers(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    if "f_dc_0" not in v.data.dtype.names:
        sys.exit(f"[fatal] PLY has no f_dc_* fields — not a gsplat PLY")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
                    axis=1).astype(np.float64)
    print(f"[step6] {len(xyz):,} splats")

    print(f"[step6] voxelizing at {SH_VOXEL_M}m + computing voxel mean DC color")
    vox = np.floor(xyz / SH_VOXEL_M).astype(np.int64)
    _, inverse = np.unique(vox, axis=0, return_inverse=True)
    n_vox = inverse.max() + 1
    counts = np.bincount(inverse)
    voxel_mean = np.zeros((n_vox, 3))
    for i in range(3):
        voxel_mean[:, i] = np.bincount(inverse, weights=fdc[:, i]) / counts
    splat_voxel_mean = voxel_mean[inverse]
    dist = np.linalg.norm(fdc - splat_voxel_mean, axis=1)
    print(f"[step6] color distance stats: median={float(np.median(dist)):.3f} "
          f"p95={float(np.percentile(dist, 95)):.3f} "
          f"max={float(dist.max()):.3f}")

    y = xyz[:, 1]
    y_cutoff = float(np.percentile(y, SH_TOP_PCTL))
    in_top_band = y < y_cutoff
    band_dist = dist[in_top_band]
    threshold = float(np.percentile(band_dist, OUTLIER_PCTL))
    print(f"[step6] scope: top {SH_TOP_PCTL}% of y "
          f"(y < {y_cutoff:.3f}) — {int(in_top_band.sum()):,} splats")
    print(f"[step6] threshold: distance <= {threshold:.3f} "
          f"(p{OUTLIER_PCTL} within band)")

    drop = in_top_band & (dist > threshold)
    keep = ~drop
    n_dropped = int(drop.sum())
    n_kept = int(keep.sum())
    print(f"[step6] dropped {n_dropped:,} (top-band SH outliers)")
    print(f"[step6] kept {n_kept:,} / {len(xyz):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "voxel_size_m": SH_VOXEL_M,
           "outlier_pctl": OUTLIER_PCTL,
           "top_pctl": SH_TOP_PCTL,
           "threshold_dist": threshold,
           "n_kept": n_kept,
           "n_total": len(xyz)}
    (out_dir / "slice_step6.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step6] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def step7_scenewide_sh_outliers(scene: Path):
    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run --step 1 first")

    pl = PlyData.read(str(in_ply))
    v = pl["vertex"]
    if "f_dc_0" not in v.data.dtype.names:
        sys.exit(f"[fatal] PLY has no f_dc_* fields — not a gsplat PLY")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
                    axis=1).astype(np.float64)
    print(f"[step7] {len(xyz):,} splats")

    print(f"[step7] voxelizing at {SH_VOXEL_M}m + computing voxel mean DC color")
    vox = np.floor(xyz / SH_VOXEL_M).astype(np.int64)
    _, inverse = np.unique(vox, axis=0, return_inverse=True)
    n_vox = inverse.max() + 1
    counts = np.bincount(inverse)
    voxel_mean = np.zeros((n_vox, 3))
    for i in range(3):
        voxel_mean[:, i] = np.bincount(inverse, weights=fdc[:, i]) / counts
    splat_voxel_mean = voxel_mean[inverse]
    dist = np.linalg.norm(fdc - splat_voxel_mean, axis=1)
    print(f"[step7] color distance stats: median={float(np.median(dist)):.3f} "
          f"p95={float(np.percentile(dist, 95)):.3f} "
          f"p99={float(np.percentile(dist, 99)):.3f} "
          f"max={float(dist.max()):.3f}")

    threshold = float(np.percentile(dist, SH_ALL_PCTL))
    print(f"[step7] scope: SCENE-WIDE")
    print(f"[step7] threshold: distance <= {threshold:.3f} (p{SH_ALL_PCTL})")

    drop = dist > threshold
    keep = ~drop
    n_dropped = int(drop.sum())
    n_kept = int(keep.sum())
    print(f"[step7] dropped {n_dropped:,} (scene-wide SH outliers)")
    print(f"[step7] kept {n_kept:,} / {len(xyz):,}")

    new_v = v.data[keep]
    PlyData([PlyElement.describe(new_v, "vertex")], text=False).write(str(in_ply))

    out_dir = scene / "_slice_temp"
    out_png = out_dir / "step7_sliced_topdown.png"
    render_topdown(in_ply, out_png)

    log = {"input_ply": str(in_ply),
           "output_ply": str(in_ply),
           "voxel_size_m": SH_VOXEL_M,
           "outlier_pctl": SH_ALL_PCTL,
           "threshold_dist": threshold,
           "n_kept": n_kept,
           "n_total": len(xyz)}
    (out_dir / "slice_step7.json").write_text(json.dumps(log, indent=2))

    print(f"\n[step7] DONE")
    print(f"  output PLY: {in_ply}")
    print(f"  topdown:    {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    if args.step == 1:
        step1_top_and_below_floor(scene)
    elif args.step == 2:
        step2_max_scale_filter(scene)
    elif args.step == 3:
        step3_anisotropy_filter(scene)
    elif args.step == 4:
        step4_top_band_max_scale(scene)
    elif args.step == 5:
        step5_top_band_density(scene)
    elif args.step == 6:
        step6_local_sh_outliers(scene)
    elif args.step == 7:
        step7_scenewide_sh_outliers(scene)


if __name__ == "__main__":
    main()
