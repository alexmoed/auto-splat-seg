# Pipeline order — authoritative source of truth

This file is THE order. If anything (script docs, my memory, our chat) disagrees with this file, this file wins. Updated as we build.

Last updated: 2026-05-05

---

## Orchestrator: `run_all.py`

The full chain is driven by `run_all.py` with three idempotent steps. Re-running `run_all.py --step N` skips any stage whose output marker exists.

| step | What happens | Outputs |
|---|---|---|
| `--step 1` | Empty scene → all rough bbox extracts (visual hulls, no SAM yet) | `_inventory_temp/`, `02_<slug>/1_visual_hull.ply`, `_phase1_temp/scene_minus_phase1.ply`, `_phase2_temp/{cameras.json, qwen_phase2_items.json, quad_*.png}`, more `02_<slug>/1_visual_hull.ply` from phase-3, `scene_manifest.json` |
| `--step 2` | Per-object dispatch — each `02_<slug>/` routed to a procedure (general / tv / bookshelf / skip) | `02_<slug>/4_sam_tight.ply` + `info.json` per object that has a working procedure |
| `--step 3` | Logical grouping + splat subtract: detect parent-child via 3D AABB, carve child splats out of parent | `scene_hierarchy.json`, `02_<parent>/5_subtracted.ply` |

### Step 1 sub-steps (managed inside `run_all.py`)

1. `inventory.py` — 4K topdown + Qwen 4-pass disjoint
2. `extract_one.py` × N — phase 1 visual hull per topdown bbox
3. subtract phase-1 cones from full scan → `_phase1_temp/scene_minus_phase1.ply`
4. `_phase2_dioramas.py` — 4 quadrant cross-cut dioramas + `cameras.json` (single source of truth for the 4 diorama cameras + room bounds)
5. `_phase2_detect.py` — Qwen per quadrant, label-blacklist + min-area + 90% same-label IoU dedup
6. `_phase3_extract_one.py` × M — visual hull from full scan via diorama camera + bbox + xz-quadrant filter; auto-suffix on slug collision (e.g. `02_black_speaker`, `_2`); QA render through the SAME diorama camera
7. `write_manifest` — `scene_manifest.json` summarizing all extracted objects

---

## Scene prep (once per scan, in `<scene>/`)

| # | Script | Step(s) | Output |
|---|---|---|---|
| 1 | `orient.py` | 1–3 | `step6_hough_corrected.ply` |
| 2 | `cardinal_pick.py` | 1–3 | `step7_cardinal_aligned.ply` |
| 3 | `slice.py` | 1–7 | `step7_sliced.ply` |

(`inventory.py` runs as part of `run_all.py --step 1`.)

---

## Per-object pipeline (in `<scene>/02_<slug>/`)

`procedure_dispatch.py` routes each object to a procedure based on its label:
- **general** (default — most furniture/decor) → the 4-stage SAM chain below
- **tv** (label matches `tv|television|monitor|screen|flat[- ]?screen` AND not `stand|console|cabinet|unit|table`) → `tv_carve.py` (pitch-sweep SAM, no floor_drop)
- **bookshelf** (label matches `bookshelf|bookcase|shelving`) → ⬜ STUB (TODO tomorrow)
- **skip** → no extraction; object falls through to leftover

### General procedure — 4 segmentation stages + 1 metadata step

| # | Stage | Script | Status | Output at root |
|---|---|---|---|---|
| 1 | visual_hull | `extract_one.py` (phase 1) or `_phase3_extract_one.py` (phase 3) | ✅ built | `1_visual_hull.ply` |
| 2 | sam_wide | `sam_carve.py --step 1..4` | ✅ built | `2_sam_wide.ply` |
| 3 | floor_drop | `floor_drop.py` | ✅ built | `3_floor_drop.ply` |
| 4 | sam_tight | `sam_tight.py` | ✅ built — **FINAL PLY** | `4_sam_tight.ply` |
| 5 | info | `info.py` | ✅ built — **FINAL METADATA** | `info.json` |

### TV procedure — `tv_carve.py`

Pitch-sweep SAM at fixed front yaw (derived from source diorama camera), pitches `[0, -15, -30, -45]` (camera above looking down), vote-frac 0.5. Skips floor_drop (TVs aren't on the floor). Writes `4_sam_tight.ply` directly.

### Bookshelf procedure — TODO

Stub returns "not_implemented". Bookshelves have deep clutter and a tall narrow profile that breaks the orbit-yaw SAM chain. Design tomorrow.

**`4_sam_tight.ply` is the shippable per-object PLY. `info.json` is its descriptive metadata** (object_type with color baked in, sub_objects, materials, style, object_ply path) — Qwen reads the 4 canonical yaws of `renders/4_sam_tight/` and emits the JSON.

**Dropped/parked from the stage-5 slot:**
- `bottom_cleanup.py` — aniso/scale filter on bottom band, wasn't pulling its weight (deleted).
- `thin_long.py` — same band-filtered AND-of-aniso-and-length, dropped only 0.4% on armchair, no visible improvement (deleted).
- `rama_cluster.py` — RAMA multicut clustering. Fragments the chair into pieces by material/normal/color and creates a per-cluster keep/drop decision burden Qwen can't reliably handle for single-object cleanup. Documented in `RAMA_NOTES.md`. Use `pair_separation.py` (in the plugin) for two-objects-merged-into-one cases instead.

Previously also dropped: aabb_filter, floor_band (sam_tight at 0.05m hard / 0.10m fabric pad covers them).

### sam_carve.py sub-steps (stage 2)

| Step | What | Output |
|---|---|---|
| 1 | Render 25 views (12 yaws × 2 pitches + topdown) | `diagnostics/sam_wide/input_*.png` + `cameras.json` |
| 2 | Qwen multi-image → pipe-union SAM prompt | `diagnostics/sam_wide/sam_prompt.txt` |
| 3 | SAM each view, retry up to 3 times if main hits <3 | `diagnostics/sam_wide/mask_*.png`, `mask_padded_*.png`, `report.json` |
| 4 | Project hull splats, vote 60%, render 5 canonical | `sam_wide.ply` + `renders/sam_wide/` |

---

## Discovery loop (later)

Iterative: extract all topdown items → subtract their hulls from the leftover detection PLY → re-inventory → repeat until 0 new items.

**KEY:**
- **Detection PLY** changes per iteration (sliced minus extracted hulls).
- **Extraction PLY** is ALWAYS `step7_sliced.ply` (full data, never the leftover).

After topdown loop converges: side cameras catch tall/thin items (lamps, etc.) that topdown structurally misses. Side detections dedup against already-extracted via 3D AABB IoU.

---

## Scene finalize (once per scene at the end)

| # | Stage | Script | Status |
|---|---|---|---|
| 1 | walls/ceiling/floor split | TBD | ⬜ |
| 2 | leftover capture (everything not extracted) | TBD | ⬜ |
| 3 | PLY → .splat conversion for web viewer | TBD | ⬜ |

---

## File layout per scene

```
<scene>/
├── raw_*.ply
├── step1_ydown.ply, step3_tilt_corrected.ply, step4_yaligned.ply,
│   step6_hough_corrected.ply, step7_cardinal_aligned.ply, step7_sliced.ply
├── orient_status.json, floor_plane.json
├── _cardinal_temp/                    (cardinal_pick artifacts)
├── _slice_temp/                       (slice artifacts)
├── _inventory_temp/                   (inventory + camera + items.json)
└── 02_<slug>/                         (one per extracted object)
    ├── 1_visual_hull.ply
    ├── 1_visual_hull_meta.json
    ├── 1_visual_hull_topdown.png      (legacy from extract_one — quick check)
    ├── 2_sam_wide.ply
    ├── 3_floor_drop.ply
    ├── 4_sam_tight.ply
    ├── 5_export.ply                   (later)
    ├── renders/
    │   ├── 1_visual_hull/{y0,y90,y180,y270,topdown}.png
    │   ├── 2_sam_wide/, 3_floor_drop/, 4_sam_tight/, 5_export/
    └── diagnostics/
        ├── 2_sam_wide/                (input/mask/cameras/prompts/report)
        ├── 3_floor_drop/, 4_sam_tight/, 5_export/
```

Yaw tag format: `y0, y90, y180, y270` (no leading zeros). Pitch: `p-15, p-45`.

---

## Locked constants (do not change without explicit user approval)

| Param | Value | Where |
|---|---|---|
| Render resolution | 1920×1080 (1080p) | All renders |
| FOV | 70° | All cameras |
| SAM yaw set | [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] | sam_carve |
| SAM pitches | [-15, -45] + topdown -89 | sam_carve |
| SAM threshold | 0.4 | sam_carve |
| sam_wide pad | 0.2m (was 0.5 — too wide; bumped 2026-05-03) | sam_carve |
| min_views_frac | 0.6 | sam_carve step 4 |
| Qwen retry attempts | 3 | sam_carve step 3 |
| Min main-prompt hits | 3 views | sam_carve step 3 |
| Inventory passes | 3 | inventory.py |
| Inventory bbox pad | 15% per side | extract_one.py |
| Render margin (single source) | 1.55 | sam_carve.RENDER_MARGIN + extract_one.RENDER_MARGIN (also imported by floor_drop) |
| floor_drop INITIAL_THRESH_M | **0.10** | floor_drop.py — start of RANSAC sweep (was 0.05) |
| floor_drop UPPER_THRESH_M | **0.18** | floor_drop.py — cap on how far above floor we drop (protects chair body/legs) |
| floor_drop NORMAL_COS_THRESH | **0.3** | floor_drop.py — looser (was 0.5) → drops more in-band splats within ~73° of vertical |
| floor_drop MAX_ITERATIONS | 5 | floor_drop.py — Qwen-driven loop, picks best at end |
| sam_tight SAM_PAD_HARD_M | 0.035 | sam_tight.py — tight pad on hard prompts (legs, frames) |
| sam_tight SAM_PAD_FABRIC_M | 0.10 | sam_tight.py — wider pad on soft prompts (body, pillow, throw) |
| sam_tight MIN_VIEWS_FRAC | **0.7** | sam_tight.py (see "Locked pair" note below) |

### 🔒 Locked pair: aggressive floor_drop ↔ sam_tight at 0.7

`floor_drop` and `sam_tight` are tuned **together**:

- **floor_drop is aggressive** (`INITIAL_THRESH_M=0.10` start, `NORMAL_COS_THRESH=0.3` looser, `UPPER_THRESH_M=0.18` cap). It removes most floor scatter — including band-of-floor near legs and skirt — while normal-protection spares the legs themselves (their normals aren't aligned with the floor plane).
- **sam_tight votes at 0.7**, not 0.8. 0.8 was tried and destroys bodies whenever SAM under-segments (label color mismatch, low contrast, busy fabric pattern) — the body fails the strict vote, only the geometrically-distinctive legs survive (22+/24 views), and you get a "floating-legs-only" PLY. 0.7 keeps the body intact across these cases at the cost of a small residual floor halo around the legs (since `floor_drop` already removed most of the floor, the halo is small but visible).

**Why not adaptive?** A retry-cascade (try 0.8, fall back if retention drops) was tried and removed. The cliff between "body kept" and "halo leaks" is sharp on real objects (0.78→63% retention with body ripped on one side, 0.7→99% retention with halo). Locking at 0.7 is the simpler honest answer.

**Pair invariant:** if `floor_drop` is made less aggressive, the residual halo at 0.7 grows. If `floor_drop` is made more aggressive, you can push 0.7 toward 0.75. They move together.

### Other locked constants

| Param | Value | Where |
|---|---|---|
| Extent calc (canonical renders) | full min/max from median center | sam_carve.render_canonical_5 |
| Slice TOP_PCTL | 9 | slice.py |
| Slice FLOOR_BUFFER | 0.35m | slice.py |
| Aniso/density/SH bands | top 10% of y | slice.py |
| Phase 2 detect MIN_AREA_NORM | 600 (in 0-1000² norm) | _phase2_detect.py |
| Phase 2 detect DEDUP_IOU | 0.9 (same-label only) | _phase2_detect.py |
| Phase 2 dioramas BACK | 2.0m | _phase2_dioramas.py |
| Phase 3 hull pad-pct | 0.05 (5% per side) | _phase3_extract_one.py |
| TV pitch sweep | [0, -15, -30, -45] (camera above looking down) | tv_carve.py |
| TV vote-frac | 0.5 | tv_carve.py |
| Group XZ_PAD | 0.05m | group.py |
| Group Y_STACK_TOL | 0.10m | group.py |
| Subtract child AABB pad | 0.02m (p2/p98) | subtract.py |

---

## Roadmap

See `TODO_TOMORROW.md` for active tasks. Roadmap items:

1. **Bookshelf procedure** — `procedure_dispatch.run_bookshelf` is a stub
2. **In-process Qwen** — replace HTTP API calls to vLLM with embedded loading
3. **Dockerize** — single image with CUDA + Qwen weights + pipeline scripts

---

## Update rule

Whenever a script is added or a stage status flips, update THIS file FIRST. Then write code.
