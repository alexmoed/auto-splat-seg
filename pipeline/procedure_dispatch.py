#!/usr/bin/env python3
"""procedure_dispatch.py — Per-object procedure router.

Picks a procedure based on the object's label (cheap rule-based first,
Qwen tiebreaker reserved for ambiguous cases — TODO) and runs the right
chain of EXISTING scripts. Does NOT copy/fork sam_carve.py etc. — the
real algorithms stay shared; only the orchestration differs per class.

Procedures:
  general   — sam_carve (4 sub-steps) → floor_drop → sam_tight → info
              The default flow validated on the armchair.
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
STAGE_SAM_TIGHT_FROM_FLOOR = (
    "sam_tight",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_tight.py"),
                  str(s), str(o)],
    "4_sam_tight.ply",
)
# Pass B — low-camera SAM refine. Reruns SAM on 4_sam_tight.ply from the
# 0/+15 low cameras and re-carves. Chained off Pass A so the low cameras
# only ever carve an already-tight object. Tables are exempt inside the
# script (low cameras look under a flat top). Always produces
# 4b_sam_tight_low.ply (a copy of 4_sam_tight.ply when skipped).
STAGE_SAM_LOW_REFINE = (
    "sam_low_refine",
    lambda s, o: [sys.executable, str(ITERATION_DIR / "sam_low_refine.py"),
                  str(s), str(o)],
    "4b_sam_tight_low.ply",
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
    """Run a list of (name, cmd_factory, marker) stages in order. Skip
    stages whose marker exists. Return dict of stage→status."""
    status = {}
    for name, mkcmd, marker in stages:
        mpath = obj_dir / marker
        if mpath.exists():
            status[name] = "skip"
            print(f"  [{name}] skip (marker exists: {marker})")
            continue
        print(f"  [{name}] running...")
        r = subprocess.run(mkcmd(scene, obj_dir))
        if r.returncode != 0:
            status[name] = f"fail({r.returncode})"
            print(f"  [{name}] FAIL exit={r.returncode}")
            return status   # stop on first failure
        status[name] = "ok"
    return status


# ───── procedures ──────────────────────────────────────────────────────

GENERAL_PRE_QC_STAGES = [
    STAGE_SAM_CARVE_S1, STAGE_SAM_CARVE_S2,
    STAGE_SAM_CARVE_S3, STAGE_SAM_CARVE_S4,
    STAGE_FLOOR_DROP,
    STAGE_SAM_TIGHT_FROM_FLOOR,
    STAGE_SAM_LOW_REFINE,
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
    stage that exists: 4b_sam_tight_low (preferred — the low-camera
    refine of the SAM result), else 4_sam_tight (high-camera SAM
    result), else 3_floor_drop (recovery — sam_tight failed)."""
    if source_stage == "auto":
        if (obj_dir / "4b_sam_tight_low.ply").exists():
            source_stage = "4b_sam_tight_low"
        elif (obj_dir / "4_sam_tight.ply").exists():
            source_stage = "4_sam_tight"
        else:
            source_stage = "3_floor_drop"
    return subprocess.run([sys.executable, str(ITERATION_DIR / "sweep_fallback.py"),
                           str(scene), str(obj_dir),
                           "--source-stage", source_stage]).returncode


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
      2. info + qc on the canonical (sam_tight) stage.
      3. If qc REJECTs: rename 4_sam_tight.ply → 4_sam_tight_rejected.ply
         so info/qc fall through to 5_sweep_fallback.ply, then re-run
         info + qc on the safety PLY.
      4. If both REJECT: move folder to <scene>/rejects/.

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

    _run_info(scene, obj_dir, status, "info")

    verdict1, rc1 = _run_qc(scene, obj_dir, no_move=True)
    status["qc_reject"] = "ok" if rc1 == 0 else f"fail({rc1})"
    if rc1 != 0 or verdict1 != "REJECT":
        return status

    # First QC said REJECT — sam_tight produced a bad result. The
    # safety sweep we already ran was sourced from 4_sam_tight (a
    # refinement OF the bad result), so it's also bad. Re-run
    # sweep_fallback sourcing from 3_floor_drop instead — that
    # rebuilds the object from the pre-SAM stage and ignores the
    # bad SAM mask entirely.
    main_ply = obj_dir / "4_sam_tight.ply"
    safety_ply = obj_dir / "5_sweep_fallback.ply"
    floor_ply = obj_dir / "3_floor_drop.ply"
    if main_ply.exists() and floor_ply.exists():
        print("  [swap] qc REJECTED 4_sam_tight → re-running sweep_fallback "
              "from 3_floor_drop, then renaming so info/qc inspect it")
        # Delete the refinement-from-sam_tight sweep; rebuild from floor_drop.
        if safety_ply.exists():
            safety_ply.unlink()
        rc_resweep = _run_sweep_fallback(scene, obj_dir,
                                           source_stage="3_floor_drop")
        status["sweep_fallback_recovery"] = "ok" if rc_resweep == 0 else f"fail({rc_resweep})"
        if rc_resweep != 0 or not safety_ply.exists():
            _move_to_rejects(scene, obj_dir, status)
            return status
        # Rename main so the new safety PLY becomes the canonical stage.
        rejected_ply = obj_dir / "4_sam_tight_rejected.ply"
        if rejected_ply.exists():
            rejected_ply.unlink()
        main_ply.rename(rejected_ply)
        status["sam_tight_rejected_renamed"] = "ok"
        _run_info(scene, obj_dir, status, "info_post_fallback")
        verdict2, rc2 = _run_qc(scene, obj_dir, no_move=True)
        status["qc_reject_post_fallback"] = "ok" if rc2 == 0 else f"fail({rc2})"
        if verdict2 != "REJECT":
            return status

    # Still REJECT (or no safety PLY available). Move out.
    _move_to_rejects(scene, obj_dir, status)
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
    STAGE_FLOOR_DROP,
    STAGE_SAM_TIGHT_BOOKSHELF,
    STAGE_BOOKSHELF_SWEEP,
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


def run_skip(scene: Path, obj_dir: Path) -> dict:
    print("  [skip] no extraction; object will roll into leftover")
    return {"skip": "ok"}


PROCEDURES = {
    "general":   run_general,
    "tv":        run_tv,
    "bookshelf": run_bookshelf,
    "rug":       run_rug,
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


def decide_procedure(label: str) -> str:
    """Rule-based first cut. Qwen tiebreaker reserved for ambiguous cases
    (not yet wired — add when the rule misclassifies in practice)."""
    lo = (label or "")
    # TV: must mention tv/screen/monitor as a standalone word AND must NOT be
    # furniture-that-holds-a-TV (stand / console / cabinet / unit / table).
    if TV_PATTERN.search(lo) and not TV_EXCLUDE.search(lo):
        return "tv"
    if BOOKSHELF_PATTERN.search(lo):
        return "bookshelf"
    if RUG_PATTERN.search(lo):
        return "rug"
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
    print(f"[dispatch] DONE  {obj.name}  status={status}")

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
