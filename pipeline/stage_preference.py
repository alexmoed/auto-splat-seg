#!/usr/bin/env python3
"""stage_preference.py — single source of truth for "which stage PLY is an
object's final/best output".

Several finalize-time consumers need to pick one PLY from an object's
02_<slug>/ directory: the most-refined stage that actually got produced.
Historically each consumer (info.py, qc_reject.py, merge_scene.py,
extract_final_outputs.py) kept its OWN hardcoded list, and the lists drifted
apart:

  * stage_pick.py was changed (commit 6b298e6) to write the picked + destreaked
    result as **8_final.ply** (via 7_picked -> 7_destreak -> 8_final), but the
    consumer lists still led with **7_final**, which the current pipeline never
    writes. The consumers therefore fell through to a PRE-pick / PRE-destreak
    stage (6_inside_outside / 4_sam_tight), silently discarding stage_pick's
    work for every object.
  * the lists also disagreed with each other on the 4_sam_tight vs
    5_sweep_fallback tiebreak, and qc_reject was missing 6_inside_outside.

This module exists so there is exactly ONE ordered preference. Import it; do
not redefine the order locally.

Order = most-refined / latest stage first. The first entry whose
``<obj>/<name>.ply`` exists is the object's final output.
"""

# Canonical, most-refined-first. The union of every stage name the finalize
# consumers historically referenced, plus 8_final at the top (the fix).
#
# Rationale for the notable orderings:
#   8_final            stage_pick's picked + destreaked result — THE deliverable
#   7_final            legacy stage_pick output name (older code); kept so an
#                      object carrying only a stale 7_final still resolves
#   6_inside_outside   multi-mask insideness carve (last stage before pick)
#   5_subtracted       parent body with child AABBs carved out (group/subtract)
#   5_bookshelf_sweep  bookshelf-route class final
#   4_rug              rug-route class final
#   5_sweep_fallback   Qwen-bbox vote refinement that runs AFTER 4_sam_tight,
#                      so it ranks ABOVE it (chain-correct; matches what
#                      merge_scene/extract_final_outputs already did)
#   4_sam_tight        SAM tight carve (Pass A)
#   3_floor_drop       geometric floor carve (table-route final)
#   2_pitch_sweep_refined  phase-4 wall-art final
#   2_sam_wide         wide SAM carve
#   1_visual_hull      coarse hull (companions / last-ditch fallback)
STAGE_PREFERENCE = [
    "8_final",
    "7_final",
    "6_inside_outside",
    "5_subtracted",
    "5_bookshelf_sweep",
    "4_rug",
    "5_sweep_fallback",
    "4_sam_tight",
    "3_floor_drop",
    "2_pitch_sweep_refined",
    "2_sam_wide",
    "1_visual_hull",
]


def pick_stage(obj_dir, candidates=None):
    """Return (stage_name, ply_path) for the most-refined existing PLY in
    ``obj_dir``, or (None, None) if none of the candidate stages exist.

    ``obj_dir`` may be a pathlib.Path or str. ``candidates`` defaults to the
    canonical STAGE_PREFERENCE; pass a subset only if a consumer genuinely
    needs to restrict the set.
    """
    from pathlib import Path
    obj_dir = Path(obj_dir)
    for stage in (candidates or STAGE_PREFERENCE):
        p = obj_dir / f"{stage}.ply"
        if p.exists():
            return stage, p
    return None, None
