# TODO — pick up next session

Stopped 2026-05-06 late. Bookshelf procedure validated end-to-end on
`02_bookshelf` — chain works, params decided. Still need to wire it into
`procedure_dispatch.py` and test on the other 3 bookshelves.

Read `PIPELINE_ORDER.md` first every session — it's the authoritative source
for stage order, locked constants, and the matched-pair note.

---

## Where we are (end of 2026-05-05)

**Orchestrator:** `run_all.py` with three idempotent steps. Each stage skips
if its primary output marker exists; re-running on a finished scene is a
no-op.

| Step | What | Status |
|---|---|---|
| `--step 1` | Empty scene → all rough bbox hulls (phase 1 topdown + phase 2/3 quadrant dioramas) | ✅ working end-to-end |
| `--step 2` | Per-object dispatch (general / tv / bookshelf / skip) | ⚠️  general+tv work; bookshelf is stub |
| `--step 3` | Group + subtract (parent-child via 3D AABB) | ✅ working |

**Validated procedures:**
- `general` — `sam_carve` → `floor_drop` → `sam_tight` (vote-frac 0.7) → `info`. Confirmed on 02_armchair, 02_grey_armchair, 02_wooden_tv_stand, 02_black_speaker, others.
- `tv` (`tv_carve.py`) — pitch-sweep SAM at fixed yaw, pitches `[0,-15,-30,-45]` (camera above, looking down), vote-frac 0.5. Confirmed on 02_black_flat_screen_tv.

**Locked params (pair invariant — don't break):**
- `floor_drop`: INITIAL_THRESH_M=0.10, UPPER_THRESH_M=0.18, NORMAL_COS_THRESH=0.3, MAX_ITERATIONS=5
- `sam_tight`: SAM_PAD_HARD_M=0.035, SAM_PAD_FABRIC_M=0.10, **MIN_VIEWS_FRAC=0.7**

---

## Tomorrow's main work

### 1. Wire bookshelf procedure into dispatcher (priority)

The bookshelf chain is **validated** on `02_bookshelf` — what's missing is
wiring it into `procedure_dispatch.run_bookshelf` so it dispatches per-object
automatically and rolling out across the other 3 bookshelves.

**Validated chain (works on 02_bookshelf):**

```
sam_carve --step 1   (default) — render 25 input views
sam_carve --step 2   (default) — Qwen builds belt+suspenders prompt:
                     "wooden bookshelf {hard} | decorative vases {hard} |
                      framed photos {hard} | stacked books {hard} |
                      woven basket {hard} | green throw blanket {soft} |
                      metal frame legs {hard}"
sam_carve --step 3   (default) — SAM each view (17/25 had usable masks
                     for 02_bookshelf — front-facing yaws hit, edge-on skip)
sam_carve --step 4   (default 0.6 vote-frac) — produces 2_sam_wide.ply
floor_drop           (default) — produces 3_floor_drop.ply
sam_tight            --min-views-frac 0.5 --sam-pad-hard-m 0.05
                     --sam-pad-fabric-m 0.15 → 4_sam_tight.ply
                     (default 0.7 / 0.035 / 0.10 nukes the body — only
                     top shelf survived)
bookshelf_faceon     (NEW script, takes 4_sam_tight.ply as source) —
                     auto-picks depth axis = shortest of x/z; renders
                     face-on from both directions; picks more-non-white
                     direction; computes tight 2D bbox of body; crops
                     splats by bbox + depth cap → 5_faceon_hull.ply
info                 (default) → info.json
```

**To do tomorrow:**

1. Replace `run_bookshelf` stub in `procedure_dispatch.py` with the chain
   above. Pass the lower-frac/larger-pad CLI args to sam_tight. Then call
   `bookshelf_faceon.py` for stage 5. Then `info.py`.

2. Test on the 3 other bookshelf folders that still have only stage-1 hulls:
   `02_bookshelf_2`, `02_low_bookshelf`, `02_open_shelving_unit`. Verify
   the same params work or whether per-shelf tuning is needed.

3. Front-detection in `bookshelf_faceon.py` uses non-white pixel count.
   On 02_bookshelf the +/- direction scores were nearly tied (485k vs 492k).
   Add a Qwen tiebreaker if the score gap is < 5% — show both rendered
   views to Qwen, ask which shows the bookshelf face with shelves visible.

**Reference Qwen description we got tonight on `02_bookshelf`** (saved
verbatim in case we want to reuse it for design validation):

**What's hard about bookshelves:**
- Tall narrow profile — orbit-yaw views from far away show a thin slab; SAM
  may not latch on the cluttered shelf interior.
- Deep clutter — shelves full of books / decor; SAM "bookshelf" prompt may
  miss the back wall, or grab the books separately, or include adjacent
  furniture.
- Standard `sam_tight` 0.8 vote (we now use 0.7) may still be too strict if
  SAM only finds the bookshelf in a subset of views.

**Design notes (validated 2026-05-06):** Belt-and-suspenders SAM prompt
worked — `sam_carve --step 2` on a bookshelf naturally produces a pipe-
union covering the structure + per-item prompts. SAM masks 17/25 views
on 02_bookshelf (front-facing yaws hit, edge-on skip). At default 0.6
vote-frac in step 4, sam_wide kept 76% (full bookshelf visible). Then
floor_drop → sam_tight with **lowered frac (0.5)** + **larger pads
(0.05/0.15)** kept 95.5%. Default 0.7 + 0.035/0.10 tight params nuked
the body.

**New stage 5 (`bookshelf_faceon.py`):** takes `4_sam_tight.ply` as
source, doesn't redo SAM. Picks depth axis = shortest horizontal extent.
Renders face-on from both directions, picks more-non-white. Crops to
2D bbox of projected body splats + depth cap.

**Reference Qwen description we got tonight on `02_bookshelf`** (saved
verbatim in case we want to reuse it for design validation):

```
1. Structure: light wood shelves, dark metal frame, 4 main shelves + top,
   open frame with vertical metal posts, four metal legs to floor.
2. Top shelf: wooden sculpture, framed picture (light mat), pink ceramic
   vase, dark ceramic vase, tall patterned ceramic vase.
   Second shelf: dark patterned vase, framed picture, blue-green vase,
   tall textured vase.
   Third shelf: framed picture, green ceramic pitcher, stack of books,
   light rectangular box.
   Bottom shelf: woven basket with folded fabric, light rectangular box.
3. On top: green leafy plant.
4. No integrated back panel; outlet on wall is NOT part of the unit.
```

Sample bookshelf folders to test: `02_bookshelf` (89k splats),
`02_bookshelf_2`, `02_low_bookshelf`, `02_open_shelving_unit` — all 4 are
phase-1 hulls, only `02_bookshelf` has step-1 SAM views rendered (today).

### 2. Audit "succeeded" extractions

Earlier today's full run reported "57/62 succeeded" but we know many of those
were degenerate (sam_tight produced near-empty PLYs that didn't error out).
Walk the manifest, flag any 02_<slug>/ where 4_sam_tight.ply has < 1000 splats
or whose label is in the "shouldn't have been extracted" set (yellow_bananas,
blue_radio, white_bowl, etc.). Mark those as `skip` in the manifest, don't
ship them.

---

## Roadmap (post-tomorrow)

### Stable Qwen — switch from API to in-process loading

Currently every Qwen call goes through `OpenAI` client → vLLM HTTP server
on port 8000. Pros: separation of concerns. Cons: vLLM server can crash and
takes ~3 min to reload; HTTP layer is overhead; harder to dockerize cleanly.

Switch to loading Qwen3.6-AWQ directly in-process via `transformers` (or
embed via `vllm.LLM(...)`). Single Python process holds the model; pipeline
scripts call into it directly.

Plan:
- Pick a stable backend (transformers vs vllm-as-library).
- Wrap in `pipeline/qwen_client.py` with the same call signature current
  scripts use, so existing code switches with one import change.
- Benchmark vs current HTTP setup (latency per call, GPU memory).
- Migrate scripts one-by-one; keep HTTP fallback during transition.

### Dockerize

Containerize the pipeline so a fresh machine can run `docker run ...
run_all.py <scene>` without manual conda + vLLM setup. Required:
- Base image with CUDA + PyTorch + gsplat + SAM3 + Qwen weights
- Single `pip install` of pipeline scripts
- Volume mount for scene data
- One command starts the model + runs the pipeline

This pairs naturally with the in-process Qwen migration — once Qwen is
embedded, the docker image is one process not two.

---

## What's already locked (do not change without explicit approval)

See `PIPELINE_ORDER.md` for the authoritative list. Highlights:

- Render resolution 1920×1080
- FOV 70° everywhere
- 25-view SAM camera set (12 yaws × 2 pitches + topdown)
- Camera params written by `_phase2_dioramas.py` to `cameras.json` are the
  single source for diorama cameras — never re-derive them
- `procedure_dispatch.py` is the only entry point per object — don't bypass
- `floor_drop ↔ sam_tight` matched pair (see PIPELINE_ORDER.md note)
