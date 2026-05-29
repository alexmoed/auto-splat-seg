# Bug audit — 2026-05-29 codebase review

Source: exhaustive read-only review (1 reachability map → 8 module audits →
53 adversarially-verified claims). Only claims whose verification verdict was
`confirmed` are listed here. 24 other claims were **refuted** by verification
(things that *looked* dead/buggy but are actually live/correct) — see the
"Refuted — do NOT touch" section at the bottom.

Branch: `cleanup/codebase-review` (off clean `main`). Every fix is a separate
commit so it can be reverted individually.

---

## FIXED + committed

### The `7_final` → `8_final` finalize cluster (5 bugs, 1 root cause) — commit `90f5ccf`
`stage_pick.py` writes the picked + destreaked deliverable as **`8_final.ply`**
(via `7_picked → 7_destreak → 8_final`); it never writes `7_final.ply`. But four
finalize consumers ranked the never-written `7_final` first and fell through to a
pre-pick / pre-destreak stage:

| File | Was | Now |
|------|-----|-----|
| `info.py:154` | led with `7_final` → described `6_inside_outside` | shared `STAGE_PREFERENCE` (8_final first) |
| `qc_reject.py:53` | led with `7_final`, **missing** `6_inside_outside` | shared list |
| `merge_scene.py:32` | led with `7_final` | shared list |
| `extract_final_outputs.py:27` | led with `7_final` | shared list |
| `stage_pick.py:6` | docstring said "copies to 7_final.ply" | corrected to 8_final |

Verified live on v32: e.g. `02_wooden_cabinet` shipped `8_final` but `info.json`
described `6_inside_outside` (2,815 vs 6,314 splats). New
`pipeline/stage_preference.py` is the single source of truth (also fixes the
prior inconsistency between the four lists). **No change to extraction — only
which existing PLY is selected as the final.**

### Other fixes — commit `8017231`
- **#14 `extract_final_outputs.py:104`** — manifest label read `object_type or
  label`, dropping the rich Qwen `name`. Now `name or object_type or label`
  (matches `rename_to_qwen.py`). Verified on `02_wooden_cabinet`
  (`name="light wood media console"` was being lost).
- **#9 `render_inside_views.py:47,60`** — removed the hardcoded
  `scene.parent/"Kitchen_living_dining"` sibling fallback; now raises if a scene
  lacks its own reference/target PLY. Prevents silently rendering the **wrong
  room**. (Dormant for scenes that have their own `step7_sliced.ply`; the
  inside_outside step takes the safe path.)
- **#15 `tv_carve.py:111`** — camera-pitch comment claimed "y-UP"; verified
  against `floor_plane.json` (`convention: y-down`, floor at y≈1.55) that it is
  y-down. Comment corrected. **Doc-only — carve math unchanged.**
- **#12a `group.py:10`** — docstring claimed a "floor/rug isn't a parent" rule
  that the code never enforces. Docstring made honest. **Doc-only.**

### Dispatcher — commit `e5ea5f2`
- **bookshelf/shelving routing** — `decide_procedure` had bookshelf auto-routing
  retired, so bookshelf + shelving fell through to `general` (which applies
  `floor_drop`/RANSAC → `4a_floor_drop`). Re-enabled `BOOKSHELF_PATTERN` →
  `bookshelf` procedure (no RANSAC). Now a first-class procedure, not a manual
  `--procedure` override. Other routes verified unchanged.

---

## DEFERRED — confirmed bugs, fixes ready, awaiting go-ahead

These change **scene-level** behavior (foundation / detection / grouping) that
the 3 per-object verification runs (armchair, art, bookshelf) do **not**
exercise, so applying them blind conflicts with "preserve functionality / don't
break things". Each has a verified fix; apply after a scene-level test.

- **#6 `slice.py:196`** — steps 2-7 mutate `step7_sliced.ply` in place with no
  idempotency guard; re-running step 1 discards 2-7's work, and re-running a
  percentile step double-filters. *Fix:* per-step state marker (`slice_state.json`)
  or write each step to a distinct filename. *Verify:* run `slice.py` twice on a
  throwaway scene copy.
- **#7 `_phase2_detect.py:464`** — `cross_quadrant_dedup` writes to `items` but
  phase 3 reads the un-deduped `by_quadrant`, so singular appliances
  (fridge/stove/range hood) extract twice. *Fix:* have phase 3 consume the
  deduped list, or fold the drop back into `by_quadrant`. *Verify:* full-scene
  inventory run.
- **#8 `_phase4_art_extract.py:667` / `run_all.py:327`** — rejected art is
  re-extracted every run (skip-index only scans `02_*`, not `rejects/`). *Fix:*
  also scan `rejects/*/1_visual_hull_meta.json`, or leave a tombstone marker.
- **#11 `procedure_dispatch.py:518` (bookshelf `STAGE_DESTREAK`)** — redundant
  re-destreak after `stage_pick` (which already destreaks → `8_final`); its
  marker `diagnostics/8_destreak/report.json` never matches the script's
  `7_destreak/report.json`, so it re-runs every time and its output is unused.
  *Fix:* drop `STAGE_DESTREAK` from `BOOKSHELF_PRE_QC_STAGES`. **Being observed
  in the 2026-05-29 bookshelf verification run now that the bookshelf procedure
  is the live path.**
- **#13 `subtract.py:80`** — child AABB always read from the child's
  `4_sam_tight.ply` with no fallback; a child whose final is `8_final`/
  `6_inside_outside` but lacks `4_sam_tight` is silently skipped (double-counted
  in parent + child). Deeper issue: the parent ships `8_final` (un-carved) while
  `5_subtracted` ranks below it. *Fix:* resolve child + parent PLY via
  `stage_preference` before computing AABBs. *Verify:* scene with parent-child
  grouping.
- **#12b `group.py`** — actually *enforce* the floor/rug-not-a-parent rule
  (currently `FLOOR_TOL`/`find_room_floor_y` are unused). Behavior change to
  grouping; docstring already made honest.
- **#10 `sam_server.py:40`** — the resident SAM3 server has **zero clients**
  (all segmentation is in-process via `sam_carve.sam_segment`); it double-loads
  SAM3 into GPU for the container lifetime. *Fix (pick one):* route
  `sam_segment` through `SAM_URL`, **or** stop launching it in `entrypoint.sh`
  + delete `sam_server.py` (its `/health` is live — gates startup + `SAM_URL`
  export — so remove launch + probe + export **together**). Needs an
  `entrypoint.sh` change + image rebuild; GPU-optimization, not correctness.

---

## Confirmed-dead (removed) — commit `be03975`
- `pipeline/bookshelf_faceon.py` (superseded by `bookshelf_sweep.py`; output read
  by nothing).
- `pipeline/archive/bookshelf_route/` (byte-identical duplicate of
  `bookshelf_sweep.py` + its README).

### Confirmed-dead, NOT removed (reported only — in live files, low value/risk)
- `sam_server.py::/segment` (tied to #10).
- `group.py::find_room_floor_y` + `FLOOR_TOL` (FLOOR_TOL is still written to
  output metadata, so not a clean strip; tied to #12).
- `extract_background.py::find_object_ply` (singular, unused wrapper — distinct
  from the live `merge_scene.find_object_ply`).

---

## Refuted by verification — do NOT "fix" these
18+ claims were challenged and found to have **no reachable failure path**.
Notable: `docker/view.py` helpers (`rotation_matrix_from_yaw_pitch`,
`viewmat_look_at`, `build_K`) are heavily used via the vendored-subprocess path;
`sam_server.health` gates container startup; `cleanup.py` reading
`step7_cardinal_aligned.ply` (not the sliced PLY) is intentional;
`cardinal_pick.py`'s `-chosen_yaw` negation is mathematically exact;
`floor_drop.py`'s no-early-STOP loop is by design; several "silent-swallow" /
"wrong-index" claims return byte-identical output. Do not delete or "harden"
these — doing so would change correct behavior.
