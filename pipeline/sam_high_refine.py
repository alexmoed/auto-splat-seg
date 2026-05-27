#!/usr/bin/env python3
"""sam_high_refine.py — Pass C: steep high-camera SAM refine.

Runs AFTER sam_low_refine (Pass B). Three rings (center / low / high):
  - Pass A sam_tight       pitches [-15, -30, -45]   margin 1.0×
  - Pass B sam_low_refine  pitches [0, +15]          margin 1.5×
  - Pass C sam_high_refine pitches [-60, -75, -89]   margin 1.25×

The high ring sees the object from way above (near topdown). Floor and
tabletop neighbours look small from this elevation and get reliably
silhouetted against the object body. Cameras stand a bit farther back
(1.25×) so the object stays fully framed at steep angles.

Chained, not parallel: Pass C's input IS Pass B's output, so the steep
cameras only ever carve an already-tight object.

Reads:
  <obj>/4b_sam_tight_low.ply  (or 4_sam_tight.ply as fallback)
  <obj>/diagnostics/2_sam_wide/sam_prompt.txt
Writes:
  <obj>/4c_sam_tight_high.ply
  <obj>/diagnostics/4c_sam_tight_high/  (masks + cameras.json)
  <obj>/renders/4c_sam_tight_high/

Usage:
    python sam_high_refine.py <scene_dir> 02_<slug>/
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

HIGH_PITCHES = [-70.0]   # single high ring — looking down 70°
HIGH_MARGIN_MULT = 1.25


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

    in_ply = obj / "4b_sam_tight_low.ply"
    if not in_ply.exists():
        in_ply = obj / "4_sam_tight.ply"
    out_ply = obj / "4c_sam_tight_high.ply"
    if not in_ply.exists():
        print(f"[sam_high_refine] SKIPPED — no Pass A/B PLY in {obj}; "
              f"nothing to refine.")
        return

    prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not prompt_path.exists():
        sys.exit(f"[fatal] missing {prompt_path}")
    pipe_prompt = prompt_path.read_text().strip()

    tagged = parse_tagged_prompts(pipe_prompt)
    prompts = [t for t, _tag in tagged]
    prompt_pads = {t: (args.sam_pad_fabric_m if tag == "soft"
                       else args.sam_pad_hard_m)
                   for t, tag in tagged}

    diag = obj / "diagnostics" / "4c_sam_tight_high"
    diag.mkdir(parents=True, exist_ok=True)

    print(f"\n[C] rendering HIGH-camera views ({HIGH_PITCHES}) from "
          f"{in_ply.name}, margin ×{HIGH_MARGIN_MULT}...")
    render_25_views(in_ply, diag, scene_dir=scene, obj_dir=obj,
                    pitches=HIGH_PITCHES, margin_mult=HIGH_MARGIN_MULT)

    print(f"\n[C] SAM each high view...")
    parent_label = prompts[0] if prompts else None
    masks_info, per_prompt_hits = sam_each_view(
        diag, prompts, prompt_pads, parent_label=parent_label,
        qwen_miss="full_frame")
    print(f"[sam] per-prompt hits: {per_prompt_hits}")

    if not masks_info:
        print("[sam_high_refine] no usable high-camera masks — keeping "
              f"{in_ply.name} unchanged as 4c_sam_tight_high.ply")
        shutil.copy(str(in_ply), str(out_ply))
        render_canonical_5(out_ply, obj / "renders" / "4c_sam_tight_high")
        (diag / "report.json").write_text(json.dumps({
            "stage": "sam_high_refine", "skipped": True,
            "reason": "no usable high-camera SAM masks",
            "input_ply": str(in_ply), "output_ply": str(out_ply),
        }, indent=2))
        return

    n_prompts = len(prompts)
    if n_prompts >= 3:
        eff_frac = max(0.40, args.min_views_frac - 0.10 * (n_prompts - 2))
        print(f"[C] {n_prompts}-term prompt → scaling min_views_frac "
              f"{args.min_views_frac:.2f} → {eff_frac:.2f}")
    else:
        eff_frac = args.min_views_frac
    print(f"\n[C] voting at min_views_frac={eff_frac:.2f}...")
    keep, n_kept, n_in, required, n_views, v = vote_carve(
        in_ply, masks_info, eff_frac)
    print(f"[vote] required >={required}/{n_views}  kept {n_kept:,}/{n_in:,} "
          f"({100*n_kept/n_in:.1f}%)")

    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")
    render_canonical_5(out_ply, obj / "renders" / "4c_sam_tight_high")

    (diag / "report.json").write_text(json.dumps({
        "stage": "sam_high_refine",
        "input_ply": str(in_ply), "output_ply": str(out_ply),
        "prompt": pipe_prompt, "high_pitches": HIGH_PITCHES,
        "margin_mult": HIGH_MARGIN_MULT,
        "min_views_frac": args.min_views_frac,
        "n_views": n_views, "required_votes": required,
        "n_in": n_in, "n_kept": n_kept,
        "per_prompt_hits": per_prompt_hits,
    }, indent=2))
    print(f"\n[done] {out_ply}  ({n_kept:,}/{n_in:,} kept)")


if __name__ == "__main__":
    main()
