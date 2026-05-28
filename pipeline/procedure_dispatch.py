#!/usr/bin/env python3
"""procedure_dispatch.py — Per-object procedure router.

Picks a procedure based on the object's label (cheap rule-based first,
Qwen tiebreaker reserved for ambiguous cases — TODO) and runs the right
chain of EXISTING scripts. Does NOT copy/fork sam_carve.py etc. — the
real algorithms stay shared; only the orchestration differs per class.

Procedures:
  general   — sam_carve (4 sub-steps) → sam_tight (A center) →
              sam_low_refine (B low) → sam_high_refine (C steep) →
              sweep_fallback → inside_outside → stage_pick → info.
              floor_drop RETIRED 2026-05-27 — sam_tight sources
              2_sam_wide.ply directly. The geometric floor band drop
              was eating soft-furniture skirts; the SAM chain handles
              floor isolation now.
  tv        — pitch-sweep SAM (no floor_drop, TVs aren't on the floor).
              NOT YET IMPLEMENTED — stub.
  bookshelf — face-on sam_carve variant (clutter-aware).
              NOT YET IMPLEMENTED — stub.
  skip      — no-op, the object will fall through into leftover.

Each stage is idempotent — skipped if its output marker exists. Re-running
on a finished object is a no-op.

Usage:
    python procedure_dispatch.py <scene_dir> <obj_dir>
    python procedure_dispatch.py <scene_dir> <obj_dir> --procedure general
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ITERATION_DIR = Path(__file__).resolve().parent


# ───── shared stage definitions (referenced by procedures below) ──────

# (stage_name, cmd_factory(scene, obj) -> argv list, output_marker_relative)
STAGE_SAM_CARVE_S1 = (
    "sam_carve_s1",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_carve.py"),
                  str(s), str(o), "--step", "1"],
    "diagnostics/2_sam_wide/cameras.json",
)
STAGE_SAM_CARVE_S2 = (
    "sam_carve_s2",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_carve.py"),
                  str(s), str(o), "--step", "2"],
    "diagnostics/2_sam_wide/sam_prompt.txt",
)
STAGE_SAM_CARVE_S3 = (
    "sam_carve_s3",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_carve.py"),
                  str(s), str(o), "--step", "3"],
    "diagnostics/2_sam_wide/report.json",
)
STAGE_SAM_CARVE_S4 = (
    "sam_carve_s4",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_carve.py"),
                  str(s), str(o), "--step", "4"],
    "2_sam_wide.ply",
)
STAGE_FLOOR_DROP = (
    "floor_drop",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "floor_drop.py"),
                  str(s), str(o)],
    "3_floor_drop.ply",
)
# 2026-05-27 — floor_drop RE-ADDED to general chain but in NEW POSITION:
# AFTER sam_tight (Pass A center), BEFORE sam_low (Pass B). Sources
# 4_sam_tight.ply, writes 4a_floor_drop.ply. The RANSAC plane fit + band
# carve trims floor halo around the base before the steep-camera SAM
# passes get to see it, so they don't have to vote against it.
STAGE_FLOOR_DROP_AFTER_TIGHT = (
    "floor_drop_after_tight",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "floor_drop.py"),
                  str(s), str(o),
                  "--src-ply", "4_sam_tight.ply",
                  "--out-stage-name", "4a_floor_drop"],
    "4a_floor_drop.ply",
)
STAGE_SAM_TIGHT_FROM_FLOOR = (
    "sam_tight",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_tight.py"),
                  str(s), str(o)],
    "4_sam_tight.ply",
)
# Bookshelf-specific sam_tight: looser vote (0.5 vs 0.7) + bigger pads
# (hard 0.05/fabric 0.15) — defaults nuke the body of a bookshelf because
# only the front-facing yaws have valid SAM masks. Validated 2026-05-05
# on 02_bookshelf.
STAGE_SAM_TIGHT_BOOKSHELF = (
    "sam_tight",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_tight.py"),
                  str(s), str(o),
                  "--min-views-frac", "0.5",
                  "--sam-pad-hard-m", "0.05",
                  "--sam-pad-fabric-m", "0.15"],
    "4_sam_tight.ply",
)
STAGE_BOOKSHELF_SWEEP = (
    "bookshelf_sweep",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "bookshelf_sweep.py"),
                  str(s), str(o)],
    "5_bookshelf_sweep.ply",
)
# Second bookshelf_sweep pass at low pitches (sees up under the shelves
# / catches the bottom plinth and base). Same Qwen-bbox-vote mechanism
# as the main pass — just different camera elevations. Output PLY lives
# alongside 5_bookshelf_sweep.ply so stage_pick can compare both.
STAGE_BOOKSHELF_SWEEP_LOW = (
    "bookshelf_sweep_low",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "bookshelf_sweep.py"),
                  str(s), str(o),
                  "--pitches", "0,15",
                  "--out-name", "5b_bookshelf_sweep_low",
                  "--src-ply", "4_sam_tight.ply"],
    "5b_bookshelf_sweep_low.ply",
)
# sam_low_refine — generates the low-camera SAM masks that inside_outside
# pools with the 4_sam_tight (high) masks for the multi-mask insideness
# carve. NOT a standalone carve — its 4b_sam_tight_low.ply output is
# secondary; what matters is the masks dir. Best-effort: if Qwen can't
# find the object in some low views, the masks that DO get written are
# still useful. Marker is the cameras.json (proves at least the views
# were rendered).
STAGE_SAM_LOW_REFINE = (
    "sam_low_refine",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_low_refine.py"),
                  str(s), str(o)],
    "diagnostics/4b_sam_tight_low/cameras.json",
    True,  # optional — failures don't halt the chain
)
# sam_high_refine — Pass C, steep high cameras (pitches -60/-75/-89).
# Reads 4b_sam_tight_low.ply if present else 4_sam_tight.ply, carves at
# steep elevation where floor / tabletop neighbours silhouette cleanly
# against the object body. Writes 4c_sam_tight_high.ply + masks dir.
# Optional — chains gracefully if input absent.
STAGE_SAM_HIGH_REFINE = (
    "sam_high_refine",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_high_refine.py"),
                  str(s), str(o)],
    "4c_sam_tight_high.ply",
    True,  # optional
)
# sweep_fallback — Qwen-bbox vote refinement on top of 4_sam_tight.
# MUST run BEFORE inside_outside so 5_sweep_fallback.ply exists as the
# input to inside_outside (priority above 4b_sam_tight_low). This is
# the order that produced v26's 36,642-splat chair: sweep_fallback
# refined 4_sam_tight → 5_sweep_fallback (37,043), then inside_outside
# took 5_sweep_fallback as input + multi-mask carve → 36,642. The
# safety-net invocation in _post_extract_qc still exists for the
# 4_sam_tight-missing recovery case (sources from 3_floor_drop); it
# skips when 5_sweep_fallback.ply already exists.
STAGE_SWEEP_FALLBACK = (
    "sweep_fallback",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sweep_fallback.py"),
                  str(s), str(o), "--source-stage", "4_sam_tight"],
    "5_sweep_fallback.ply",
)
STAGE_INSIDE_OUTSIDE = (
    "inside_outside",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "inside_outside.py"),
                  str(o), "--auto"],
    "6_inside_outside.ply",
)
# Final cross-stage pick: Qwen looks at sam_tight + sam_tight_low +
# sweep_fallback + inside_outside renders, picks the cleanest+intact
# one, copies to final.ply. Each stage excels at different objects
# (wall-flush cabinets often peak at sam_tight; soft sofas peak at
# inside_outside; cabriole tables peak at sweep_fallback). One Qwen
# call per object. Writes final_pick.json with choice + reason.
STAGE_FINAL_PICK = (
    "stage_pick",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "stage_pick.py"),
                  str(o)],
    "8_final.ply",
)
# Drop big+dark Gaussian splat streaks (one axis blew up, color collapsed
# ~black). Locked 2026-05-27 after light_wood_bookshelf had visible black
# vertical smears in the front view. Rewrites 7_final.ply in place when
# it finds streaks; no-op otherwise. Optional — no streaks ≠ failure.
STAGE_DESTREAK = (
    "splat_destreak",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "splat_destreak.py"),
                  str(o)],
    "diagnostics/8_destreak/report.json",
    True,  # optional — no streaks means no report, that's fine
)
STAGE_INFO = (
    "info",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "info.py"),
                  str(s), str(o)],
    "info.json",
)
# Final QC: lenient Qwen check on the latest stage's 5 canonical
# renders. PASS = keep. REJECT (heavily damaged / unrecognizable noise)
# = move whole folder to <scene>/rejects/. Marker is qc_reject.json,
# which the script always writes — re-running on a kept object is a
# no-op via the marker check. (Rejected folders move out, so they
# can't be re-checked here.)
STAGE_QC_REJECT = (
    "qc_reject",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "qc_reject.py"),
                  str(s), str(o)],
    "qc_reject.json",
)


def _run_chain(scene: Path, obj_dir: Path, stages: list) -> dict:
    """Run a list of stages in order. Each stage is a tuple of
    (name, cmd_factory, marker) or (name, cmd_factory, marker, optional).
    Skip stages whose marker exists. Required stages halt the chain on
    failure; optional stages log the failure and continue.
    Return dict of stage→status."""
    status = {}
    for stage in stages:
        if len(stage) == 4:
            name, mkcmd, marker, optional = stage
        else:
            name, mkcmd, marker = stage
            optional = False
        mpath = obj_dir / marker
        if mpath.exists():
            status[name] = "skip"
            print(f"  [{name}] skip (marker exists: {marker})")
            continue
        print(f"  [{name}] running...")
        r = subprocess.run(mkcmd(scene, obj_dir))
        if r.returncode != 0:
            status[name] = f"fail({r.returncode})"
            print(f"  [{name}] FAIL exit={r.returncode}"
                  f"{' (optional, continuing)' if optional else ''}")
            if not optional:
                return status   # stop on first hard failure
            continue
        status[name] = "ok"
    return status


# ───── procedures ──────────────────────────────────────────────────────

GENERAL_PRE_QC_STAGES = [
    STAGE_SAM_CARVE_S1, STAGE_SAM_CARVE_S2,
    STAGE_SAM_CARVE_S3, STAGE_SAM_CARVE_S4,
    # floor_drop MOVED 2026-05-27 from the position before sam_tight to
    # the position AFTER sam_tight. sam_tight now sources 2_sam_wide.ply
    # (via fallback), and floor_drop runs on its output to carve floor
    # halo before the steep-camera SAM passes see it.
    STAGE_SAM_TIGHT_FROM_FLOOR,    # Pass A center, sources 2_sam_wide.ply
    STAGE_FLOOR_DROP_AFTER_TIGHT,  # RANSAC on 4_sam_tight → 4a_floor_drop
    STAGE_SAM_LOW_REFINE,          # Pass B low — reads 4a_floor_drop (fallback 4_sam_tight)
    STAGE_SAM_HIGH_REFINE,         # Pass C steep — reads 4b_sam_tight_low (chains via low)
    STAGE_SWEEP_FALLBACK,          # Qwen-bbox vote refinement on top of 4_sam_tight
    STAGE_INSIDE_OUTSIDE,          # multi-mask insideness carve
    STAGE_FINAL_PICK,              # Stage 7+8: Qwen picks stage → 7_picked → destreak --auto → 7_destreak → 8_final.ply
]

# Table chain — same as general but STOPS at 3_floor_drop.
# sam_tight is omitted because its vote_carve eats wall-adjacent table
# legs: compute_wall_skip drops every back-hemisphere camera, so back
# legs never accumulate the ≥0.7·N_views needed to survive the vote
# (v25 dining_table 2026-05-20 — kept 70,097 with legs gone, vs
# floor_drop's 80,480 with legs intact). 3_floor_drop.ply is a pure
# geometric carve (RANSAC floor + normal-aware band drop) and keeps
# the legs cleanly.
TABLE_PRE_QC_STAGES = [
    STAGE_SAM_CARVE_S1, STAGE_SAM_CARVE_S2,
    STAGE_SAM_CARVE_S3, STAGE_SAM_CARVE_S4,
    STAGE_FLOOR_DROP,
    # STAGE_SAM_TIGHT_FROM_FLOOR — SKIPPED for tables (legs)
    STAGE_FINAL_PICK,       # 2026-05-20 — falls through to 3_floor_drop.ply.
]

# Floor / standing lamp chain (validated 2026-05-28 on
# 02_floor_standing_lamp). A floor lamp is a THIN pole + base + shade.
# sam_tight's vote_carve erodes the pole (it never accumulates the
# ≥0.7·N_views needed to survive — a 3cm-wide pole projects to a few px
# and misses most yaw masks), collapsing the lamp 9,042 → 922 splats.
# inside_outside's multi-mask insideness carve has the same effect on the
# stem. So this chain SKIPS both: sam_carve s1-s4 → promote 2_sam_wide →
# sweep_fallback → stage_pick(+destreak --auto). The destreak --auto
# inside stage_pick runs the geom bottom-25% pre-clean (drops the
# big+elongated+isolated floor-shadow streaks at the base) + the color
# streak sweep. Result: 3,868-splat readable lamp with pole + shade + base
# intact and base halo cleaned.
LAMP_PRE_QC_STAGES = [
    STAGE_SAM_CARVE_S1, STAGE_SAM_CARVE_S2,
    STAGE_SAM_CARVE_S3, STAGE_SAM_CARVE_S4,
    # STAGE_SAM_TIGHT_FROM_FLOOR + STAGE_INSIDE_OUTSIDE — SKIPPED (thin pole)
    STAGE_SWEEP_FALLBACK,   # sources 4_sam_tight (= promoted 2_sam_wide)
    STAGE_FINAL_PICK,       # stage_pick → destreak --auto (geom + color) → 8_final
]


def _read_qc_verdict(obj_dir: Path) -> str | None:
    p = obj_dir / "qc_reject.json"
    if not p.exists():
        return None
    try:
        return str(json.load(open(p)).get("verdict", "")).strip().upper()
    except Exception:
        return None


def _move_to_rejects(scene: Path, obj_dir: Path, status: dict) -> None:
    """Move <obj_dir> → <scene>/rejects/<name>/, with collision suffix."""
    rejects_root = scene / "rejects"
    rejects_root.mkdir(exist_ok=True)
    dest = rejects_root / obj_dir.name
    if dest.exists():
        i = 2
        while (rejects_root / f"{obj_dir.name}_{i}").exists():
            i += 1
        dest = rejects_root / f"{obj_dir.name}_{i}"
    print(f"  [reject] moving {obj_dir.name} → rejects/{dest.name}")
    import shutil
    shutil.move(str(obj_dir), str(dest))
    status["rejected_to"] = str(dest)


def _run_qc(scene: Path, obj_dir: Path, no_move: bool) -> tuple[str, int]:
    """Run qc_reject.py and return (verdict, exit_code). Always
    overwrites qc_reject.json (no marker check)."""
    cmd = [sys.executable, str(ITERATION_DIR / "qc_reject.py"),
           str(scene), str(obj_dir)]
    if no_move:
        cmd.append("--no-move")
    r = subprocess.run(cmd)
    return _read_qc_verdict(obj_dir) or "UNKNOWN", r.returncode


def _run_sweep_fallback(scene: Path, obj_dir: Path,
                          source_stage: str = "auto") -> int:
    """Run sweep_fallback.py. source_stage='auto' picks the latest
    stage that exists: 4_sam_tight (preferred — refines the SAM result
    with a Qwen-bbox vote that drops off-axis neighbours SAM kept),
    else 3_floor_drop (recovery — sam_tight failed)."""
    if source_stage == "auto":
        if (obj_dir / "4_sam_tight.ply").exists():
            source_stage = "4_sam_tight"
        else:
            source_stage = "3_floor_drop"
    return subprocess.run([sys.executable, str(ITERATION_DIR / "sweep_fallback.py"),
                           str(scene), str(obj_dir),
                           "--source-stage", source_stage]).returncode


def _apply_deferred_rename(obj_dir: Path) -> Path:
    """Read stage_pick_refined_slug.json (if stage_pick wrote one) and
    rename obj_dir to 02_<refined_slug> with _2/_3 suffix on collision.
    Returns the new path (or obj_dir unchanged if no rename pending).

    Lives here, not in stage_pick.py, so qc_gate + info + split_children
    can run against the original path before the rename takes effect.
    """
    if not obj_dir.exists():
        return obj_dir
    marker = obj_dir / "stage_pick_refined_slug.json"
    if not marker.exists():
        return obj_dir
    try:
        info = json.load(open(marker))
    except Exception:
        return obj_dir
    if not info.get("rename_pending"):
        print(f"[rename] slug unchanged ('{info.get('current_slug', '?')}') — no rename")
        return obj_dir
    new_slug = info.get("refined_slug", "object")
    if new_slug in (None, "", "object"):
        return obj_dir
    base_target = obj_dir.parent / f"02_{new_slug}"
    target = base_target
    n = 2
    while target.exists():
        target = obj_dir.parent / f"02_{new_slug}_{n}"
        n += 1
    print(f"[rename] {obj_dir.name} → {target.name}")
    obj_dir.rename(target)
    print(f"[rename] new path: {target}")
    return target


def _run_info(scene: Path, obj_dir: Path, status: dict, key: str) -> int:
    """Run info.py, deleting any prior info.json so it picks the
    latest stage. Returns the exit code."""
    (obj_dir / "info.json").unlink(missing_ok=True)
    rc = subprocess.run([sys.executable, str(ITERATION_DIR / "info.py"),
                         str(scene), str(obj_dir)]).returncode
    status[key] = "ok" if rc == 0 else f"fail({rc})"
    return rc


MAIN_STAGE_PLYS = ("5_bookshelf_sweep", "4_rug", "4_sam_tight")


def _post_extract_qc(scene: Path, obj_dir: Path, status: dict,
                      allow_sweep_fallback: bool = True) -> dict:
    """Common tail for any procedure:
      1. ALWAYS run sweep_fallback as a safety PLY when 4_sam_tight.ply
         or 3_floor_drop.ply exists. When sam_tight succeeded, the
         sweep sources from 4_sam_tight (acts as a Qwen-bbox-vote
         REFINEMENT on top of SAM, dropping off-axis neighbours SAM
         kept). When sam_tight failed, it sources from 3_floor_drop
         (acts as recovery, rebuilding from the pre-SAM stage).
      2. QC GATE — qc_reject.py lenient PASS/REJECT on the PICKED
         7_final. PASS continues; REJECT (only absolute trash) ->
         <scene>/rejects/, discarded.
      3. info — name + describe (only when the gate PASSed).

    The gate runs ONCE, after stage_pick, on the final picked result —
    never mid-chain, so an upstream bug can't bury a good object before
    it gets its full chain. stage_pick is independent and untouched.

    `allow_sweep_fallback=False` for procedures whose output the
    sweep_fallback can't recover (e.g. rugs — sweep_fallback sources
    from 3_floor_drop, which for rugs still contains on-top furniture,
    and its furniture-shaped Qwen prompt doesn't fit a flat rug)."""
    # 1. Run sweep_fallback as automatic safety net.
    has_main_output = any(
        (obj_dir / f"{p}.ply").exists() for p in MAIN_STAGE_PLYS
    )
    has_any_source = (obj_dir / "4_sam_tight.ply").exists() or \
                       (obj_dir / "3_floor_drop.ply").exists()
    if allow_sweep_fallback and \
            not (obj_dir / "5_sweep_fallback.ply").exists() and \
            has_any_source:
        if has_main_output:
            print("  [sweep_fallback] running as safety net (refines 4_sam_tight)")
        else:
            print("  [sweep_fallback] no main-stage PLY → primary recovery from 3_floor_drop")
        rc = _run_sweep_fallback(scene, obj_dir)
        status["sweep_fallback"] = "ok" if rc == 0 else f"fail({rc})"
        if rc != 0 and not has_main_output:
            return status

    # QC GATE — lenient PASS/REJECT on the PICKED 7_final result.
    # stage_pick already chose the best candidate; this judges whether
    # that best is keepable. PASS -> name + describe. REJECT (only
    # absolute trash — noise / unrecognizable) -> <scene>/rejects/,
    # stop (no naming). Never runs mid-chain.
    verdict, rc = _run_qc(scene, obj_dir, no_move=True)
    status["qc_gate"] = verdict if rc == 0 else f"fail({rc})"
    if verdict == "REJECT":
        _move_to_rejects(scene, obj_dir, status)
        return status

    _run_info(scene, obj_dir, status, "info")
    return status


def run_general(scene: Path, obj_dir: Path) -> dict:
    """sam_carve → floor_drop → sam_tight → [bbox-sweep fallback if
    needed] → info → qc_reject (with fallback retry) → reject if both
    QC passes fail. Then split_children to break items-on-top out of
    the parent (sideboard → lamp, picture frame, vases, etc.)."""
    status = _run_chain(scene, obj_dir, GENERAL_PRE_QC_STAGES)
    status = _post_extract_qc(scene, obj_dir, status)
    # If the object survived QC (still on disk), split items-on-top.
    if obj_dir.exists() and (obj_dir / "diagnostics" / "2_sam_wide" /
                              "sam_prompt.txt").exists():
        if (obj_dir / "children" / "manifest.json").exists():
            status["split_children"] = "skip"
        else:
            rc = subprocess.run([sys.executable,
                                  str(ITERATION_DIR / "split_children.py"),
                                  str(scene), str(obj_dir)]).returncode
            status["split_children"] = "ok" if rc == 0 else f"fail({rc})"
    return status


def run_tv(scene: Path, obj_dir: Path) -> dict:
    """TV / monitor / picture-on-furniture: pitch-sweep SAM at fixed front
    yaw + vote-frac 0.5, no floor_drop. See tv_carve.py.
    After extraction, runs companion_search.py to find supporting items
    (soundbar, remote, set-top box) and extract them as separate child
    objects."""
    status = {}
    # tv_carve produces 4_sam_tight.ply directly (skips 2_sam_wide + 3_floor_drop).
    if (obj_dir / "4_sam_tight.ply").exists():
        status["tv_carve"] = "skip"
    else:
        r = subprocess.run([sys.executable, str(ITERATION_DIR / "tv_carve.py"),
                            str(scene), str(obj_dir)])
        if r.returncode != 0:
            status["tv_carve"] = f"fail({r.returncode})"
            return status
        status["tv_carve"] = "ok"
    # info + qc with bbox-sweep fallback (TVs use 4_sam_tight as their
    # main output — same fallback story as general).
    status = _post_extract_qc(scene, obj_dir, status)
    # Companion search: find supporting items near the TV (soundbar, remote,
    # set-top box, gaming console, etc.) and extract as separate children.
    if (obj_dir / "companions.json").exists():
        status["companion_search"] = "skip"
    else:
        r = subprocess.run([sys.executable,
                             str(ITERATION_DIR / "companion_search.py"),
                             str(scene), str(obj_dir)])
        status["companion_search"] = "ok" if r.returncode == 0 else f"fail({r.returncode})"
    return status


BOOKSHELF_PRE_QC_STAGES = [
    STAGE_SAM_CARVE_S1, STAGE_SAM_CARVE_S2,
    STAGE_SAM_CARVE_S3, STAGE_SAM_CARVE_S4,
    # STAGE_FLOOR_DROP RETIRED 2026-05-27 — same reason as general chain.
    STAGE_SAM_TIGHT_BOOKSHELF,
    STAGE_BOOKSHELF_SWEEP,
    STAGE_BOOKSHELF_SWEEP_LOW,  # 2026-05-22 — second pass at low pitches
                                 # (same Qwen-bbox-vote mechanism as
                                 # bookshelf_sweep, just from below).
    STAGE_FINAL_PICK,       # picks best from available stage outputs.
    STAGE_DESTREAK,         # drop big+dark streak splats from 7_final.ply (2026-05-27)
]


def run_bookshelf(scene: Path, obj_dir: Path) -> dict:
    """Bookshelf chain (validated 2026-05-05 on 02_bookshelf):
    sam_carve → floor_drop → sam_tight (looser: 0.5/0.05/0.15) →
    bookshelf_sweep → info → qc_reject with bbox-sweep fallback retry.
    Then companion_search to extract individual shelf items (books, vases,
    picture frames, baskets, plants, etc.) as separate children."""
    status = _run_chain(scene, obj_dir, BOOKSHELF_PRE_QC_STAGES)
    status = _post_extract_qc(scene, obj_dir, status)
    if (obj_dir / "companions.json").exists():
        status["companion_search"] = "skip"
    else:
        rc = subprocess.run([sys.executable,
                              str(ITERATION_DIR / "companion_search.py"),
                              str(scene), str(obj_dir)]).returncode
        status["companion_search"] = "ok" if rc == 0 else f"fail({rc})"
    return status


def run_rug(scene: Path, obj_dir: Path) -> dict:
    """Rug chain: topdown-only Qwen bbox cone at narrow FOV=25, y-band
    cut to keep only the floor-flush slab, auto-subtract sibling
    on-top objects (coffee table etc.), then info + qc with
    sweep_fallback retry. Validated 2026-05-06 on 02_large_beige_area_rug
    (24,176 splats → clean rug rectangle, coffee table hole carved)."""
    status = {}
    if (obj_dir / "4_rug.ply").exists():
        status["rug_extract"] = "skip"
    else:
        rc = subprocess.run([sys.executable, str(ITERATION_DIR / "rug_extract.py"),
                             str(scene), str(obj_dir)]).returncode
        status["rug_extract"] = "ok" if rc == 0 else f"fail({rc})"
        if rc != 0:
            return status
    # No sweep_fallback retry for rugs — it would source from
    # 3_floor_drop (which still has furniture on top of the rug) and
    # use a furniture-shaped Qwen prompt that doesn't fit a flat rug.
    return _post_extract_qc(scene, obj_dir, status,
                             allow_sweep_fallback=False)


def run_table(scene: Path, obj_dir: Path) -> dict:
    """Table chain: sam_carve → floor_drop → STOP.
    Skips sam_tight (would carve back-legs on wall-adjacent tables —
    see TABLE_PRE_QC_STAGES note). Promotes 3_floor_drop.ply →
    4_sam_tight.ply so downstream (info, qc_reject, split_children)
    finds the expected file name."""
    import shutil as _sh
    status = _run_chain(scene, obj_dir, TABLE_PRE_QC_STAGES)
    floor = obj_dir / "3_floor_drop.ply"
    tight = obj_dir / "4_sam_tight.ply"
    if floor.exists():
        _sh.copy(str(floor), str(tight))
        status["sam_tight"] = "skipped_table_route → 4_sam_tight = 3_floor_drop"
    status = _post_extract_qc(scene, obj_dir, status)
    # Same split_children as run_general (table can have items on top —
    # bowl/plant/pitcher/etc.).
    if obj_dir.exists() and (obj_dir / "diagnostics" / "2_sam_wide" /
                              "sam_prompt.txt").exists():
        if (obj_dir / "children" / "manifest.json").exists():
            status["split_children"] = "skip"
        else:
            rc = subprocess.run([sys.executable,
                                  str(ITERATION_DIR / "split_children.py"),
                                  str(scene), str(obj_dir)]).returncode
            status["split_children"] = "ok" if rc == 0 else f"fail({rc})"
    return status


def run_lamp(scene: Path, obj_dir: Path) -> dict:
    """Floor / standing lamp chain (validated 2026-05-28 on
    02_floor_standing_lamp): sam_carve s1-s4 → promote 2_sam_wide →
    sweep_fallback → stage_pick(+destreak --auto) → final.

    Skips sam_tight (its vote_carve erodes the thin pole — 9,042 → 922)
    AND inside_outside (multi-mask carve also eats the stem). Promotes
    2_sam_wide.ply → 4_sam_tight.ply so sweep_fallback / stage_pick /
    info / split_children all find the expected file name. The geom
    bottom-25% destreak inside stage_pick's `destreak --auto` cleans the
    base floor-shadow streaks; the color sweep drops dark streaks."""
    import shutil as _sh
    # sam_carve s1-s4 → 2_sam_wide.ply
    status = _run_chain(scene, obj_dir, LAMP_PRE_QC_STAGES[:4])
    wide = obj_dir / "2_sam_wide.ply"
    tight = obj_dir / "4_sam_tight.ply"
    if wide.exists() and not tight.exists():
        _sh.copy(str(wide), str(tight))
        status["sam_tight"] = "skipped_lamp_route → 4_sam_tight = 2_sam_wide"
    # sweep_fallback (from 4_sam_tight) → stage_pick (+destreak --auto)
    status.update(_run_chain(scene, obj_dir, LAMP_PRE_QC_STAGES[4:]))
    status = _post_extract_qc(scene, obj_dir, status)
    # Split items-on-top (a floor lamp rarely has any, but keep parity in
    # case Qwen sees a small accent table / books beside it).
    if obj_dir.exists() and (obj_dir / "diagnostics" / "2_sam_wide" /
                              "sam_prompt.txt").exists():
        if (obj_dir / "children" / "manifest.json").exists():
            status["split_children"] = "skip"
        else:
            rc = subprocess.run([sys.executable,
                                  str(ITERATION_DIR / "split_children.py"),
                                  str(scene), str(obj_dir)]).returncode
            status["split_children"] = "ok" if rc == 0 else f"fail({rc})"
    return status


def run_skip(scene: Path, obj_dir: Path) -> dict:
    print("  [skip] no extraction; object will roll into leftover")
    return {"skip": "ok"}


PROCEDURES = {
    "general":   run_general,
    "tv":        run_tv,
    "bookshelf": run_bookshelf,
    "rug":       run_rug,
    "table":     run_table,
    "lamp":      run_lamp,
    "skip":      run_skip,
}


# ───── decision ────────────────────────────────────────────────────────

# Word-boundary regexes — "tv" must be a standalone word, not a substring.
# Otherwise "wooden tv stand" (furniture) would route to the tv procedure.
TV_PATTERN        = re.compile(
    r"\b(tv|television|monitor|screen|flat[- ]?screen)\b", re.IGNORECASE)
TV_EXCLUDE        = re.compile(r"\b(stand|console|cabinet|unit|table)\b",
                               re.IGNORECASE)
BOOKSHELF_PATTERN = re.compile(
    r"\b(bookshelf|book[- ]?shelf|bookcase|shelving(?:\s+unit)?|open\s+shelving)\b",
    re.IGNORECASE)
RUG_PATTERN = re.compile(
    r"\b(rug|carpet|area\s+rug|floor\s+mat|runner)\b", re.IGNORECASE)
# Match standalone "table" or "desk" — covers dining/coffee/side/console
# table + desk. Does NOT match sideboard (which is furniture, has lots
# of items on top, needs sam_tight + multi-mask inside_outside).
TABLE_PATTERN = re.compile(r"\b(table|desk)\b", re.IGNORECASE)
# Don't route sideboards / cabinets to the table chain — they need the
# full sam_tight + on-top-items path.
TABLE_EXCLUDE = re.compile(r"\b(sideboard|cabinet|console|nightstand)\b",
                           re.IGNORECASE)
# Floor / standing lamp — thin pole + base + shade. Routes to the lamp
# chain (skips sam_tight + inside_outside, both of which erode the pole).
# Requires "floor"/"standing"/"torchiere" so it does NOT catch a "table
# lamp" / "desk lamp" (those sit on furniture → general route / child).
LAMP_PATTERN = re.compile(
    r"\b(floor[- ]?lamp|floor[- ]?standing\s+lamp|standing\s+lamp|torch[ie]+re)\b",
    re.IGNORECASE)


def decide_procedure(label: str) -> str:
    """Rule-based first cut. Qwen tiebreaker reserved for ambiguous cases
    (not yet wired — add when the rule misclassifies in practice)."""
    lo = (label or "")
    # TV: must mention tv/screen/monitor as a standalone word AND must NOT be
    # furniture-that-holds-a-TV (stand / console / cabinet / unit / table).
    if TV_PATTERN.search(lo) and not TV_EXCLUDE.search(lo):
        return "tv"
    # Floor / standing lamp — thin pole; route to the lamp chain (skips
    # sam_tight + inside_outside). Checked before table so a "floor lamp"
    # never falls through; "table lamp" / "desk lamp" don't match
    # LAMP_PATTERN (no floor/standing) and stay on their existing route.
    if LAMP_PATTERN.search(lo):
        return "lamp"
    # Bookshelf route RETIRED from auto-routing 2026-05-22 — the general
    # route extracts bookshelves at least as cleanly (validated on the
    # light-wood + tall metal-frame bookshelves) and keeps the pipeline
    # simpler. The bookshelf procedure is still defined and reachable via
    # an explicit `--procedure bookshelf` (kept for website renders).
    # BOOKSHELF_PATTERN is left in place for that manual path.
    if RUG_PATTERN.search(lo):
        return "rug"
    if TABLE_PATTERN.search(lo) and not TABLE_EXCLUDE.search(lo):
        return "table"
    return "general"


# ───── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir",   type=Path,
                    help="02_<slug>/ folder for the object")
    ap.add_argument("--procedure", choices=list(PROCEDURES.keys()),
                    default=None, help="override the decided procedure")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    obj   = args.obj_dir.resolve()
    if not (obj / "1_visual_hull.ply").exists():
        sys.exit(f"[fatal] no 1_visual_hull.ply in {obj}")

    # Read label from meta
    meta_path = obj / "1_visual_hull_meta.json"
    label = "?"
    if meta_path.exists():
        try:
            label = json.load(open(meta_path)).get("label", "?")
        except Exception:
            pass

    procedure = args.procedure or decide_procedure(label)
    print(f"[dispatch] obj={obj.name}  label='{label}'  procedure={procedure}")

    fn = PROCEDURES[procedure]
    status = fn(scene, obj)

    # Apply deferred rename from stage_pick (2026-05-27 fix — was
    # originally inside stage_pick.py but ran before qc_gate + info +
    # split_children, breaking their obj_dir Path). Now runs as the last
    # step of dispatch so all downstream stages see the original path.
    obj_final = _apply_deferred_rename(obj)
    if obj_final != obj:
        status["renamed_to"] = obj_final.name

    print(f"[dispatch] DONE  {obj_final.name}  status={status}")

    # Persist procedure choice into meta so subsequent runs know
    if meta_path.exists():
        try:
            m = json.load(open(meta_path))
            m["procedure"] = procedure
            m["procedure_status"] = status
            meta_path.write_text(json.dumps(m, indent=2))
        except Exception:
            pass

    if any(str(v).startswith("fail") for v in status.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
