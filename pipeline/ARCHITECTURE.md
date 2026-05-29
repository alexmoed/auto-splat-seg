# Pipeline architecture — module map + current chain

Authoritative as of 2026-05-29 (regenerated from the code during the codebase
review). If `PIPELINE_ORDER.md` disagrees about the per-object chain, THIS file
+ `procedure_dispatch.py` win — `PIPELINE_ORDER.md`'s stage table predates the
`8_final` rework.

## Why the layout is flat (do NOT move modules into subpackages)

Modules invoke each other in two move-fragile ways:
1. **Subprocess by name** — `procedure_dispatch.py` and `run_all.py` build
   commands as `[sys.executable, str(ITERATION_DIR / 'name.py'), ...]` where
   `ITERATION_DIR = Path(__file__).resolve().parent`. Moving a target into a
   subdir breaks every dispatch literal.
2. **Flat imports** — ~12 modules do `from extract_one import ...`,
   `from sam_carve import ...`, `from view import ...` (the last resolves to the
   Docker-vendored `_vendored_view.py` via a `sys.path.insert`). The Dockerfile
   also `COPY pipeline/ /workspace/pipeline/` and symlinks `docker/view.py` over
   the skill path.

Moving files would require touching every dispatch literal + every import + the
Dockerfile + the `sys.path` inserts — high blast radius. So the layout stays
flat; this doc provides the logical grouping instead.

## Logical groups (all live in `pipeline/`, flat)

| Group | Modules |
|-------|---------|
| **foundation** (per-scene align/clean) | `orient.py`, `cardinal_pick.py`, `slice.py`, `cleanup.py`, `room_config.py`, `detect_room_type.py` |
| **detection** (what's in the scene) | `inventory.py`, `_phase2_dioramas.py`, `_phase2_detect.py`, `_phase4_art_detect.py` |
| **extraction → visual hull** | `extract_one.py`, `_phase3_extract_one.py`, `_phase4_art_extract.py` |
| **per-object refine (SAM chain)** | `sam_carve.py`, `sam_tight.py`, `sam_low_refine.py`, `sam_high_refine.py`, `floor_drop.py`, `sweep_fallback.py`, `inside_outside.py`, `render_inside_views.py`, `splat_destreak.py`, `stage_pick.py` |
| **class routes** | `tv_carve.py`, `rug_extract.py`, `bookshelf_sweep.py`, `companion_search.py` |
| **grouping / children** | `group.py`, `subtract.py`, `split_children.py` |
| **finalize** | `info.py`, `qc_reject.py`, `rename_to_qwen.py`, `merge_scene.py`, `extract_background.py`, `extract_final_outputs.py`, `ply_to_splat.py` |
| **shared / orchestration** | `run_all.py`, `procedure_dispatch.py`, `stage_preference.py`, `sam_server.py` (service), `_vendored_view.py` / `docker/view.py` (render primitive — never move/merge) |

## End-to-end scene flow (`run_full.sh`)

```
orient.py (1-3)            -> step6_hough_corrected.ply       [host, conda]
cardinal_pick.py (1-3)     -> step7_cardinal_aligned.ply
slice.py (1-7)             -> step7_sliced.ply
cleanup.py (1)             -> step8_density_filtered.ply
run_pipeline.sh:                                              [docker]
  run_all.py --step 1      -> detect_room_type, inventory, extract_one xN,
                              _phase2_dioramas, _phase2_detect, _phase4 art,
                              _phase3_extract_one xM, scene_manifest.json
                              => one 02_<slug>/1_visual_hull.ply per object
  run_all.py --step 2      -> procedure_dispatch.py per object (see below)
  run_all.py --step 3      -> group.py + subtract.py (parent/child)
  rename_to_qwen.py        -> folders renamed to Qwen-refined labels
  extract_background.py    -> scene_background.ply
  extract_final_outputs.py -> final_outputs/<obj>.splat (+ manifest)
  merge_scene.py           -> scene_reassembled.ply + verification renders
```

## Per-object chain (`procedure_dispatch.py`, routed by label)

`decide_procedure(label)` → one of: `tv`, `lamp`, `bookshelf`, `rug`, `table`,
`general` (default). All chains end at **`8_final.ply`** (the shipped
deliverable), resolved everywhere via `stage_preference.py`.

- **general** — `sam_carve` s1-4 → `2_sam_wide` → `sam_tight` (Pass A, from
  `2_sam_wide`) → `floor_drop` (RANSAC, → `4a_floor_drop`) → `sam_low_refine`
  (Pass B) → `sam_high_refine` (Pass C) → `sweep_fallback` (`5_sweep_fallback`)
  → `inside_outside` (`6_inside_outside`) → `stage_pick`
  (`7_picked` → `7_destreak` → **`8_final`**) → `_post_extract_qc`
  (`qc_reject` + `info`) → `split_children`.
- **bookshelf** — like general but **SKIPS `floor_drop`/RANSAC** (wrong for tall
  shelving): `sam_carve` → `sam_tight_bookshelf` (looser vote) →
  `bookshelf_sweep` + `bookshelf_sweep_low` → `stage_pick` → (`STAGE_DESTREAK` —
  redundant, see BUGS_FIXED.md #11). Matches bookshelf/bookcase/shelving.
- **lamp** — SKIPS `sam_tight` + `inside_outside` (erode the thin pole): promote
  `2_sam_wide` → `sweep_fallback` → `stage_pick` (+geom destreak).
- **table** — `sam_carve` → `floor_drop` → STOP (skips `sam_tight`, which carves
  wall-adjacent back legs); promotes `3_floor_drop` → `4_sam_tight`.
- **tv** — `tv_carve` (pitch-sweep SAM, no floor_drop) → `companion_search`.
- **rug** — `rug_extract` (narrow-FOV topdown, on-top items subtracted), no
  sweep_fallback.

## Recommended (not yet done) de-duplication

Verification found pervasive copy-paste. Extracting these into flat shared
modules (import-only, never subprocessed — so dispatch paths are unaffected)
would cut the most duplication without touching the topology:
- `stage_preference.py` — **done** (the `8_final` cluster fix).
- `qwen_io.py` — `encode_b64` (~5 copies) + JSON-fence parsing + a `qwen_pick`
  picker (3 variants: inside_outside / splat_destreak / stage_pick).
- `render_util.py` — `render_canonical_5` (2 copies: sam_carve vs floor_drop) +
  the topdown camera-math block duplicated across foundation modules.
- `qwen_bbox.py` — bbox-sweep helpers duplicated across rug_extract /
  bookshelf_sweep / sweep_fallback.
- camera math (`viewmat_look_at`/`build_K`/`project_to_pixels`/`slugify`) lives
  in `extract_one.py` and is imported by ~12 modules; could move to
  `camera_math.py` with `extract_one` kept as a re-export shim (it is also
  subprocessed, so the shim preserves the dispatch literal).
