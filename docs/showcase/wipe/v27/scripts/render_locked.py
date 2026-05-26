#!/usr/bin/env python3
"""Render the locked wipe pairs at a NEW scene version using the SAME
hand-iterated cameras as the original.

Reads docs/showcase/wipe/cameras.json (the v13/v14/v26 eye/target that
were hand-locked over tasks #475-#520) and re-renders each requested
object + same-camera background using a different scene version.

Use case: scene re-extracted in v27 → swap PLY paths to v27 → render →
get visually identical framing of the new extraction.

Outputs:
  <out-dir>/<slug>_object.png      (object PLY at locked eye/target)
  <out-dir>/<slug>_background.png  (background PLY at locked eye/target)

Then run build_showcase.py on the same <out-dir> to build the pairs.

Usage:
  ./render_locked.py \
      --cameras docs/showcase/wipe/cameras.json \
      --scene-root /home/ubuntu/room_pipeline_v002/pipeline/Kitchen_living_dining_v27 \
      --object-stage 7_final \
      --background-ply scene_background.ply \
      --slugs grey_armchair=02_beige_upholstered_armchair_with_wooden_legs \
              wooden_sideboard=02_light_wood_sideboard \
              wooden_coffee_table=02_light_oak_coffee_table_with_black_metal_frame \
      --out-dir docs/showcase/wipe/v27
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
PYTHON = "/home/ubuntu/anaconda3/envs/claude_seg/bin/python"


def render(ply: Path, out: Path, eye, target, fov, w, h):
    cmd = [PYTHON, VIEW_PY, str(ply), str(out),
           f"--eye={eye[0]:.5f},{eye[1]:.5f},{eye[2]:.5f}",
           f"--target={target[0]:.5f},{target[1]:.5f},{target[2]:.5f}",
           "--y-down", "--fov", str(fov),
           "--width", str(w), "--height", str(h)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True,
                    help="path to existing cameras.json (locked eye/target)")
    ap.add_argument("--scene-root", type=Path, required=True,
                    help="new scene root (e.g. Kitchen_living_dining_v27)")
    ap.add_argument("--object-stage", default="7_final",
                    help="object PLY basename (no .ply)")
    ap.add_argument("--background-ply", default="scene_background.ply",
                    help="background PLY filename inside --scene-root")
    ap.add_argument("--slugs", nargs="+", required=True,
                    help="cam_slug=new_obj_dir mappings. cam_slug is the "
                         "key inside cameras.json['objects']. new_obj_dir "
                         "is the v27 02_<...> folder name.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cams = json.loads(args.cameras.read_text())
    common = cams.get("common", {})
    fov = common.get("fov", 50)
    w = common.get("width", 1920)
    h = common.get("height", 1080)

    bg_ply = args.scene_root / args.background_ply
    if not bg_ply.exists():
        sys.exit(f"missing background PLY: {bg_ply}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the common block + a fresh objects block pointing at v27 PLYs.
    out_cams = {"common": common, "objects": {}}

    for spec in args.slugs:
        if "=" not in spec:
            sys.exit(f"slug spec must be cam_slug=obj_dir, got: {spec}")
        cam_slug, obj_dir = spec.split("=", 1)
        if cam_slug not in cams.get("objects", {}):
            sys.exit(f"cam_slug '{cam_slug}' not in {args.cameras}")
        entry = cams["objects"][cam_slug]
        eye = entry["eye"]
        target = entry["target"]

        obj_ply = args.scene_root / obj_dir / f"{args.object_stage}.ply"
        if not obj_ply.exists():
            sys.exit(f"missing object PLY: {obj_ply}")

        print(f"\n[{cam_slug}]")
        print(f"  obj_ply: {obj_ply}")
        print(f"  bg_ply : {bg_ply}")
        print(f"  eye    = {eye}")
        print(f"  target = {target}")
        print(f"  fov    = {fov}  size = {w}x{h}")

        obj_png = args.out_dir / f"{cam_slug}_object.png"
        bg_png = args.out_dir / f"{cam_slug}_background.png"
        render(obj_ply, obj_png, eye, target, fov, w, h)
        render(bg_ply, bg_png, eye, target, fov, w, h)
        print(f"  → {obj_png}")
        print(f"  → {bg_png}")

        out_cams["objects"][cam_slug] = {
            "scene": args.scene_root.name,
            "object_ply": f"{obj_dir}/{args.object_stage}.ply",
            "background_ply": args.background_ply,
            "eye": eye,
            "target": target,
        }

    cam_out = args.out_dir / "cameras.json"
    cam_out.write_text(json.dumps(out_cams, indent=2))
    print(f"\n[done] {len(out_cams['objects'])} objects rendered")
    print(f"[done] cameras.json → {cam_out}")


if __name__ == "__main__":
    main()
