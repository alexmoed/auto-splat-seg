#!/usr/bin/env python3
"""sam_low_refine.py — Pass B: low-camera SAM refine.

Runs AFTER sam_tight (Pass A). Pass A carves with the high cameras
(-15/-45) only — safe, never sees under a flat top. Pass B re-renders
Pass A's result (4_sam_tight.ply) from the LOW cameras (0 deg level,
+15 deg looking up), runs SAM, and vote-carves again. The low cameras
silhouette under-object floor smear against the background so the vote
trims it — material the high-only pass leaves behind.

Chained, not parallel: Pass B's input IS Pass A's output, so the low
cameras only ever carve an already-tight object.

TABLE EXEMPT: flat-topped objects (table / desk) skip the low carve —
the low cameras look UNDER the tabletop and give SAM a garbage
silhouette. For those, 4b_sam_tight_low.ply is just a copy of
4_sam_tight.ply so downstream always finds the file.

Reads:
  <obj>/4_sam_tight.ply
  <obj>/diagnostics/2_sam_wide/sam_prompt.txt
Writes:
  <obj>/4b_sam_tight_low.ply
  <obj>/diagnostics/4b_sam_tight_low/  (masks + cameras.json)
  <obj>/renders/4b_sam_tight_low/

Usage:
    python sam_low_refine.py <scene_dir> 02_<slug>/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from plyfile import PlyData, PlyElement

sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from sam_tight import (  # noqa: E402
    render_25_views, sam_each_view, vote_carve, render_canonical_5,
    parse_tagged_prompts, SAM_PAD_HARD_M, SAM_PAD_FABRIC_M, MIN_VIEWS_FRAC,
)

LOW_PITCHES = [0.0, 15.0]
TABLE_TOKENS = ("table", "desk")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path, help="path to 02_<slug>/")
    ap.add_argument("--sam-pad-hard-m", type=float, default=SAM_PAD_HARD_M)
    ap.add_argument("--sam-pad-fabric-m", type=float, default=SAM_PAD_FABRIC_M)
    ap.add_argument("--min-views-frac", type=float, default=MIN_VIEWS_FRAC)
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()

    in_ply = obj / "4_sam_tight.ply"
    out_ply = obj / "4b_sam_tight_low.ply"
    if not in_ply.exists():
        print(f"[sam_low_refine] SKIPPED — no {in_ply.name} (Pass A did not "
              f"produce a sam_tight result); nothing to refine.")
        return

    prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not prompt_path.exists():
        sys.exit(f"[fatal] missing {prompt_path}")
    pipe_prompt = prompt_path.read_text().strip()

    # TABLE EXEMPT — low cameras look under the flat top; skip the carve.
    if any(tok in pipe_prompt.lower() for tok in TABLE_TOKENS):
        print(f"[sam_low_refine] table/desk detected — low carve SKIPPED; "
              f"copying 4_sam_tight.ply -> 4b_sam_tight_low.ply")
        shutil.copy(str(in_ply), str(out_ply))
        render_canonical_5(out_ply, obj / "renders" / "4b_sam_tight_low")
        diag = obj / "diagnostics" / "4b_sam_tight_low"
        diag.mkdir(parents=True, exist_ok=True)
        (diag / "report.json").write_text(json.dumps({
            "stage": "sam_low_refine", "skipped": True,
            "reason": "table/desk class — low cameras look under flat top",
            "input_ply": str(in_ply), "output_ply": str(out_ply),
        }, indent=2))
        print(f"[done] {out_ply} (passthrough — table exempt)")
        return

    tagged = parse_tagged_prompts(pipe_prompt)
    prompts = [t for t, _tag in tagged]
    prompt_pads = {t: (args.sam_pad_fabric_m if tag == "soft"
                       else args.sam_pad_hard_m)
                   for t, tag in tagged}

    diag = obj / "diagnostics" / "4b_sam_tight_low"
    diag.mkdir(parents=True, exist_ok=True)

    print(f"\n[B] rendering LOW-camera views ({LOW_PITCHES}) from "
          f"4_sam_tight.ply...")
    render_25_views(in_ply, diag, scene_dir=scene, pitches=LOW_PITCHES)

    print(f"\n[B] SAM each low view...")
    parent_label = prompts[0] if prompts else None
    masks_info, per_prompt_hits = sam_each_view(
        diag, prompts, prompt_pads, parent_label=parent_label)
    print(f"[sam] per-prompt hits: {per_prompt_hits}")

    if not masks_info:
        # No usable low masks — keep Pass A's result unchanged.
        print("[sam_low_refine] no usable low-camera masks — keeping "
              "4_sam_tight.ply unchanged as 4b_sam_tight_low.ply")
        shutil.copy(str(in_ply), str(out_ply))
        render_canonical_5(out_ply, obj / "renders" / "4b_sam_tight_low")
        (diag / "report.json").write_text(json.dumps({
            "stage": "sam_low_refine", "skipped": True,
            "reason": "no usable low-camera SAM masks",
            "input_ply": str(in_ply), "output_ply": str(out_ply),
        }, indent=2))
        return

    print(f"\n[B] voting at min_views_frac={args.min_views_frac}...")
    keep, n_kept, n_in, required, n_views, v = vote_carve(
        in_ply, masks_info, args.min_views_frac)
    print(f"[vote] required >={required}/{n_views}  kept {n_kept:,}/{n_in:,} "
          f"({100*n_kept/n_in:.1f}%)")

    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")
    render_canonical_5(out_ply, obj / "renders" / "4b_sam_tight_low")

    (diag / "report.json").write_text(json.dumps({
        "stage": "sam_low_refine",
        "input_ply": str(in_ply), "output_ply": str(out_ply),
        "prompt": pipe_prompt, "low_pitches": LOW_PITCHES,
        "min_views_frac": args.min_views_frac,
        "n_views": n_views, "required_votes": required,
        "n_in": n_in, "n_kept": n_kept,
        "per_prompt_hits": per_prompt_hits,
    }, indent=2))
    print(f"\n[done] {out_ply}  ({n_kept:,}/{n_in:,} kept)")


if __name__ == "__main__":
    main()
