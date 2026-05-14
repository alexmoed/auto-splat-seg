#!/usr/bin/env python3
"""render_inside_views.py — render a scene from the center of the
room looking outward in 4 cardinal directions + topdown.

Picks the room centre as the median xz of an in-room reference PLY
(default: step7_sliced.ply — the bounded slab, NOT the unsliced
source, so wild outliers don't pull the centroid). Then renders the
target scene from that point at eye-height looking ±x and ±z.

Default target: scene_background.ply (the rotated room minus
extracted objects, with ceiling intact). Override with --target-ply
to render the unsliced source or any other PLY in the same coordinate
frame.

Usage:
    python render_inside_views.py <scene_dir>
    python render_inside_views.py <scene_dir> --target-ply step7_cardinal_aligned.ply
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402
from extract_one import viewmat_look_at, build_K  # noqa: E402
from PIL import Image

FOV = 80.0          # wider FOV — more of the room visible from a single eye
W, H = 1920, 1080
EYE_HEIGHT_FROM_FLOOR_M = 1.6   # standing-ish viewpoint


def find_in_room_reference(scene: Path) -> Path:
    """Pick a PLY whose median xz reflects the actual room centre
    (i.e. NOT the unsliced source which has noise outliers)."""
    for name in ("step7_sliced.ply", "scene_background.ply",
                 "step8_density_filtered.ply"):
        p = scene / name
        if p.exists():
            return p
        sib = scene.parent / "Kitchen_living_dining" / name
        if sib.exists():
            return sib
    raise FileNotFoundError(f"no in-room reference PLY found near {scene}")


def find_target_ply(scene: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for name in ("scene_background.ply", "step7_sliced.ply"):
        p = scene / name
        if p.exists():
            return p
        sib = scene.parent / "Kitchen_living_dining" / name
        if sib.exists():
            return sib
    raise FileNotFoundError(f"no target PLY found near {scene}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--target-ply", type=Path, default=None,
                    help="PLY to render (default: scene_background.ply)")
    ap.add_argument("--reference-ply", type=Path, default=None,
                    help="PLY for centroid calc (default: step7_sliced.ply)")
    ap.add_argument("--out-subdir", default="renders/inside_views")
    ap.add_argument("--fov", type=float, default=FOV)
    args = ap.parse_args()

    scene = args.scene_dir.resolve()

    ref_ply = args.reference_ply or find_in_room_reference(scene)
    target_ply = find_target_ply(scene, args.target_ply)

    print(f"[ref]    {ref_ply}  (for centroid)")
    print(f"[target] {target_ply}  (rendered)")

    # 1) Compute centroid + eye position from reference
    ref_v = PlyData.read(str(ref_ply))["vertex"]
    ref_xyz = np.stack([ref_v["x"], ref_v["y"], ref_v["z"]],
                        axis=1).astype(np.float64)
    # Use percentile-based centre to ignore stray outliers even in the
    # reference PLY (step7_sliced has some).
    cx = float(np.median(ref_xyz[:, 0]))
    cz = float(np.median(ref_xyz[:, 2]))
    floor_y = float(np.percentile(ref_xyz[:, 1], 95))   # y-down: floor = high y
    eye_y = floor_y - EYE_HEIGHT_FROM_FLOOR_M

    print(f"\n[centre] xz=({cx:.2f}, {cz:.2f})  floor_y={floor_y:.2f}  eye_y={eye_y:.2f}")

    out_dir = scene / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load target for rendering
    print(f"\n[load] target {target_ply}")
    target = load_gsplat_ply(str(target_ply))

    # 3) Render 4 cardinal yaws (looking outward from center)
    eye = np.array([cx, eye_y, cz], dtype=np.float64)
    K = build_K(args.fov, W, H)

    yaws = [
        ("y0",   ( 0.0, 0.0,  1.0)),   # +z
        ("y90",  (-1.0, 0.0,  0.0)),   # -x  (90° = looking left in XZ)
        ("y180", ( 0.0, 0.0, -1.0)),   # -z
        ("y270", ( 1.0, 0.0,  0.0)),   # +x
    ]
    for tag, fwd in yaws:
        target_pt = eye + np.array(fwd, dtype=np.float64) * 5.0
        # y-down convention: up = (0,-1,0)
        V = viewmat_look_at(eye.tolist(), target_pt.tolist(), [0.0, -1.0, 0.0])
        img = render_splat(target, V.astype(np.float32),
                            K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
        out = out_dir / f"{tag}.png"
        Image.fromarray(img).save(out)
        print(f"  [{tag}] {out}")

    # 4) Topdown for context
    topdown_eye = np.array([cx, eye_y - 4.0, cz])  # 4m above eye
    topdown_target = np.array([cx + 0.001, floor_y, cz])
    V = viewmat_look_at(topdown_eye.tolist(), topdown_target.tolist(),
                         [0.0, 0.0, -1.0])
    img = render_splat(target, V.astype(np.float32),
                        K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
    out = out_dir / "topdown.png"
    Image.fromarray(img).save(out)
    print(f"  [topdown] {out}")

    print(f"\n[done] 5 renders in {out_dir}")


if __name__ == "__main__":
    main()
