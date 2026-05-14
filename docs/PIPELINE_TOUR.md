# Pipeline tour — room splat segmentation

A walkthrough of every stage in the pipeline and the variations we
landed on, written so a teammate joining a week from now can pick up
the code without spelunking through commits. Reads top-to-bottom.

---

## 1. What the pipeline does

**Input:** one `.ply` Gaussian splat scan of a room — typically 3 M
splats, a couple-hundred-MB file.

**Output:** per-object `.splat` "puzzle pieces" (sofa, sideboard,
coffee table, paintings, plants…) + a `_background.splat` (walls,
floor, ceiling), plus a reassembled QC `.ply` so you can verify the
parts add back up to the original.

The drivers:

- `pipeline/run_all.py` — orchestrator, three numbered steps
- `run_pipeline.sh` — single-scene single-GPU wrapper (boots Docker,
  waits for vLLM + sam_server, runs steps 1-3 + final outputs)
- `run_parallel.sh` — N scenes on N GPUs round-robin

The services baked into the Docker image:

- **vLLM** serving Qwen 3.6-30B-A3B AWQ at `:8000`. Used for
  every visual-reasoning step (room-type detection, inventory,
  per-view bounding boxes, QC verdicts).
- **`sam_server.py`** (FastAPI/uvicorn) holding facebook/sam3 in
  GPU memory at `:8001`. Eliminates the per-subprocess weight reload
  that used to cost ~12 min per scene.

Both URLs come from env vars (`QWEN_URL`, `SAM_URL`) so a parallel
run can point each scene at its own container.

---

## 2. Foundation alignment chain

A scan straight off a phone is usually z-up, slightly tilted, and
its walls aren't aligned to the world axes. Eight little steps fix
that, in order:

| step | script | what it does |
|------|--------|---|
| 1 | `rotate_zup_to_ydown.py` | swap z-up → y-down via `(x,y,z) → (x,-z,y)` + matching SH rotation |
| 2 | `remove_shell.py` (preliminary) | strip outer dome / capture noise. Yields `floor_plane.json`. |
| 3 | `tilt_correct.py` | pitch/roll until floor normal is exactly `(0,1,0)` |
| 4 | `y_axis_align.py` | PCA-based yaw to bring walls toward world x/z |
| 5 | render axis-aligned topdown | with `up=(0,0,1)` (avoids gimbal lock) |
| 6 | Hough lines on the topdown | get residual yaw error (PCA leaves ~5-10°) |
| 7 | `y_axis_align.py --angle-deg=<resid>` | corrective pass |
| 8 | `qwen_verify.py` gate | "Are the wood planks parallel to image edges?" CONFIRM or iterate. |

By the time this chain finishes you have `step7_cardinal_aligned.ply`
— the original scan, untouched data-wise, but rotated so floor is
y-down and walls are world-aligned. Everything downstream assumes
this frame.

The 9-step chain matters a lot. A 5° leftover tilt cascades into
trapezoidal Qwen bboxes, off-axis SAM masks, diagonal floor cuts in
`finalize_scene`. We learned this the hard way and now treat it as a
hard gate.

---

## 3. Phase 1 — Inventory (Qwen)

Goal: get a list of `(label, pixel-bbox-on-the-topdown)` for every
extractable item in the room.

Step 1 in `run_all.py`:

1. **Room-type detection** — ask Qwen "kitchen / living / bed /
   office" so the inventory pass list is room-appropriate.
2. **Topdown render** — `step7_sliced.ply` (top/bottom trimmed) at
   FOV 70°, 3840×2160. Camera looks straight down with
   `up=(0,0,-1)`.
3. **Multi-pass Qwen** with disjoint category prompts:
   - `pass1_seating` (sofas, chairs, ottomans, benches)
   - `pass2_storage` (cabinets, sideboards, bookshelves)
   - `pass3_appliances` (stoves, fridges, ovens, hoods)
   - `pass4_tables` (coffee, side, dining, end tables, nightstands)
   - `pass5_lighting_decor` (lamps, plants, baskets)
   - `pass6_wall_art` (paintings, framed prints) — runs from
     per-quadrant dioramas, not the topdown
4. **Cross-pass IoU dedup** — if pass2 and pass4 both return a bbox
   for the same physical object, keep the larger. IoU threshold 0.5.

Output: `_inventory_temp/qwen_items.json` with N items + the topdown
camera (so downstream stages can back-project bboxes to 3D).

For wall items, a parallel **phase 2** runs the same Qwen calls on
4 quadrant "dioramas" — slices of the room rendered from the center
toward each corner, with the opposite quadrant excluded. This finds
wall art and TV-mounted items that the topdown misses.

---

## 4. Phase 3 — Per-object extraction (the five stages)

Run per inventory item. Each stage saves a `.ply` so you can resume
or inspect mid-stream.

### Stage 1 — Visual hull (`extract_one.py` / `sam_carve.py` step 1)

1. **Back-project** the item's topdown pixel bbox into world space:
   project the 4 corners through the topdown camera onto a 3D ray,
   then build a thin AABB cone in xz.
2. **Top-extend** the cone upward by 2 m to include items resting on
   top (lamps, plants, vases).
3. **Carve** `step7_sliced.ply` to splats inside the cone.
4. Save `1_visual_hull.ply`.

Tuning knobs we explored: `top-extend-m` (default 2.0; bumped from
1.5 in v8 after lamps got cut), `pad-pct` for the source bbox (6%
on each side after starting at 15% and clipping legs on small
objects). `TOP_PCTL` was bumped 3 → 9 → 10 to keep more headroom
without including ceiling residue.

### Stage 2 — SAM wide (`sam_carve.py` step 2-3)

1. Render `1_visual_hull.ply` from **25 cameras**: yaws every 45°,
   plus pitches `[0, -15, -30, -45]` and a topdown, with optional
   ±5° yaw nudges around the front view.
2. **Wall-skip**: skip cameras whose eye sits on the wall-side of
   the hull's back face. (Without this, cameras inside a wall
   render through the wall and SAM latches onto the wrong thing.)
3. **Per-view SAM3** with the (Claude-crafted) parent prompt:
   for a sofa, that's a pipe-union like
   `"chesterfield sofa|sofa back|sofa armrest|sofa skirt|pillow"`.
4. Vote: keep splats whose mask hit count clears `MIN_VIEWS_FRAC`
   (currently 0.7, was 0.8 — too strict, killed bodies).
5. Save `2_sam_wide.ply`.

### Stage 3 — Floor drop (`floor_drop.py`)

1. RANSAC the floor plane from a strict-floor slab.
2. Score a threshold sweep on (band thickness, normal-cos angle
   to floor normal) by asking Qwen "is this view of just the
   object floating cleanly, or are there still floor remnants?"
3. Apply the winning threshold to drop floor-band splats.
4. Save `3_floor_drop.ply`.

Class-specific defaults: soft (upholstery) gets `floor-thresh 0.40
--enable-floor-band-drop`; hard surfaces stay at the default 0.05.
Soft chesterfields need `--normal-cos-thresh 0.5` because their
loose skirt fabric has noisy normals.

### Stage 4 — SAM tight (`sam_tight.py`)

Same 25-view setup as stage 2, but with three tighter knobs:
`sam_pad` smaller, mask-vote stricter, and — **the thing that
caused the v13→v14 coffee-table regression** — a per-view Qwen
**crop**.

The crop was added to constrain SAM to pixels inside the parent
object's per-view bbox (so SAM can't latch onto neighboring
furniture). It works for most objects. For **tables**, the per-view
Qwen bbox is a tight rectangle around the top surface, which cuts
splayed legs and apron off.

The fix (this session): a `NO_CROP_TOKENS` list in `sam_tight.py`
that skips the crop entirely when the parent label contains
`"coffee table"`, `"side table"`, `"end table"`, `"dining table"`,
`"accent table"`, `"console table"`, `"nightstand"`, or `"bench"`.

Validated: living-room coffee table went 20,603 → 20,059 splats in
v12 (no crop), versus 20,805 → 10,242 in v14 (with 3 % crop). After
the bypass, tables stay full again.

### Stage 5 — Sweep fallback (`sweep_fallback.py` or
`bookshelf_sweep.py`)

When Qwen QC rejects stage 4 (`qc_reject.py` verdict: "scattered
blurry shapes, doesn't look like the labeled item"), stage 5 takes
over: a **360° / 10° dense yaw sweep**, asking Qwen for a per-view
bbox at each yaw, then back-projecting and voting across the views
that returned non-empty results. `MIN_VIEW_HITS=5` to reject
hallucinations.

Bookshelves run their own sweep variant that crops by the per-view
Qwen bbox and includes ±5° around the front yaw. The TV/monitor
class has its own `tv_carve.py` (pitch sweep at fixed front yaw —
yaw orbit fragments TVs).

If stage 5 also gets rejected, the object directory is moved to
`rejects/` with a `qc_reject.json` explaining why.

---

## 5. Class-specific procedures

The router (`procedure_dispatch.py`) picks one of these based on the
detected object type:

| class | procedure | why |
|---|---|---|
| Sofa, armchair, cabinet, sideboard (general) | 5-stage chain above | most furniture |
| Wall art (painting, print) | `_phase4_art_extract.py` — 8-yaw + 8-pitch Qwen vote on a face-on hull | tight, near-flat objects need a wall-art-shaped carve |
| Wall mirror, mounted speaker, sconce | `wall_art_sam_refine.py` cascade | 3D wall fixtures with silhouettes |
| TV / monitor on furniture | `tv_carve.py` pitch sweep at fixed front yaw | yaw orbit either fragments the screen or contaminates with the stand |
| Bookshelf, open display shelf | `bookshelf_sweep.py` 360° dense sweep | the standard SAM tight loses the front face from inside-room cameras |
| Rug | `rug_extract.py` topdown carve, FOV 25° narrow, on-top items subtracted first | wide FOV distorts the rug's footprint enough to mis-frame it |

There are two cross-cutting helpers worth knowing about:

- **`companion_search.py`** — after a parent (TV stand, bookshelf,
  sideboard) extracts, re-prompt Qwen on its front view for small
  items sitting on it (remote, vase, speaker, books) and extract
  each as a sub-object.
- **`split_children.py`** — for TV-stand-shaped parents specifically,
  keep the cabinet whole and add the TV/speakers as DUPLICATE child
  splats rather than subtracting from the cabinet. Non-TV-stand
  parents fall through and use the regular subtract path.

---

## 6. Background, reassembly, packaging

After every per-object extraction succeeds:

1. **`extract_background.py`** — KDTree-match each
   `02_<obj>/object.ply` against `step7_cardinal_aligned.ply` and
   drop matched splats from the scene. The survivors become
   `scene_background.ply`.
2. **`finalize_scene.py`** — split the background into walls.ply +
   ceiling.ply + floor.ply + a combined `room.ply` empty-room
   background. Drops a dome on both sides (above-ceiling sky-dome
   and below-floor capture noise).
3. **`extract_final_outputs.py`** — convert every per-object PLY
   and the background to `.splat` for the web viewer. Writes
   `_manifest.json` describing each piece.
4. **`merge_scene.py`** — reassemble every output back into a single
   PLY and render canonical views, so you can visually QC that the
   pieces add up to the original scan.

Output layout:

```
<scene>/
├── step7_cardinal_aligned.ply       # axis-aligned source
├── 02_<slug>/                       # one per object
│   ├── 1_visual_hull.ply
│   ├── 2_sam_wide.ply
│   ├── 3_floor_drop.ply
│   ├── 4_sam_tight.ply
│   ├── 5_sweep_fallback.ply (or 5_subtracted.ply)
│   └── renders/<stage>/{y0,y90,y180,y270,topdown}.png
├── rejects/02_<slug>/               # QC-rejected objects + reason
├── scene_background.ply
├── scene_reassembled.ply
└── final_outputs/
    ├── <slug>.splat
    └── _background.splat
```

---

## 7. Variations we landed on this session

The work below is in the most recent commits.

### sam_tight NO_CROP_TOKENS bypass

The crop-before-SAM feature was the right call for most parents but
the wrong call for table-like ones. Patched in `pipeline/sam_tight.py`
with an explicit token-match bypass:

```python
NO_CROP_TOKENS = ("coffee table", "side table", "end table",
                  "dining table", "accent table", "console table",
                  "nightstand", "bench")
```

When the parent label matches, the per-view Qwen bbox call is
skipped entirely and SAM sees the full render (the v12 behavior).

### SAM3 HTTP service

`pipeline/sam_server.py` — FastAPI app loading `facebook/sam3` once
at container startup. Endpoints: `GET /health`, `POST /segment`
(returns base64 PNG mask + scores). Listens on
`$SAM_SERVER_PORT` (default 8001).

Every script that used to do `from transformers import …; load()` now
hits `SAM_URL/segment` over HTTP. Net win: a 36-object scene saves
~12 min on weight reloads, and parallel runs share the model across
all per-object subprocesses for free.

### Env-routed URLs (multi-GPU)

Every script reads `QWEN_URL`, `QWEN_MODEL`, and `SAM_URL` from
env. Defaults to localhost when unset. `run_pipeline.sh` forwards
both env vars into `docker exec`, and `run_parallel.sh` assigns
unique ports per slot (`vLLM 8000+i`, `sam 9000+i`) and launches
N containers pinned to N GPUs.

### Docker double-entrypoint fix

`run_pipeline.sh`'s `docker run … bash -c "/usr/local/bin/entrypoint.sh
sleep infinity"` was triggering entrypoint twice (Dockerfile
`ENTRYPOINT` ran it once, then the bash command ran it again,
hitting "port 8001 already in use" on the second attempt). Stripped
to `… sleep infinity`. Same fix in `run_parallel.sh` + README.

### Showcase tooling

- **`pipeline/render_wipe_pairs.py`** — given a scene + an object
  slug + an explicit eye/target (or a computed front-on camera),
  render the object PLY and the background PLY with the **same**
  camera so a website can wipe-cross-dissolve between them.
- **`pipeline/build_showcase.py`** — read
  `docs/showcase/wipe/cameras.json`, produce labeled side-by-sides
  per object (`<slug>_pair.png`) plus a single 4-row grid
  (`_grid.png`) for embedding in the README.

### Coffee-table PLY swap (workaround)

v14 ran with the crop bug present, so its coffee table only had
10 K splats. As a one-time patch we copied
`v12/02_coffee_table_2/5_sweep_fallback.ply` (20 K splats, same
world frame) into `v14/02_wooden_coffee_table/5_sweep_fallback.ply`.
`cameras.json` notes the source-PLY override for that one entry.
The proper fix (NO_CROP_TOKENS) is in the pipeline now so next
runs won't need this.

### README + LICENSE

- Embedded HF links for `Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ` and
  `facebook/sam3` with a `huggingface-cli download` snippet.
- Noted that the AWQ choice is for a 48-GB single-GPU fit, not a
  hard requirement.
- Added the matplotlib `_grid.png` plus links to per-object pairs
  and `cameras.json`.
- MIT.

---

## 8. Showcase camera tour (what it took)

Locking each of the four wipe pairs was iterative. The script
exposes a per-object spec string `slug:yaw,orbit,pan,lift,margin`
on top of the auto-computed front-on camera, plus a fully explicit
`--eye/--target` path for cameras coming straight from the web
viewer.

Notes from the iterations:

- **Armchair** — scan rig is parked right next to it (white-draped
  tripod, floor lamp). Locking required an explicit eye/target from
  the web viewer; FOV 50; the chair sits centered with the
  bookshelf behind and a sliver of the drape on the far left.
- **Sofa** — biggest object, widest framing. The last clean
  z-coordinate is `eye_z=2.87`; past `~2.9` the camera punches
  through the back wall and the room renders behind opaque
  scan-rig splats (the dark-frame failure mode).
- **Coffee table** — the v12 PLY swap was the only way to get
  20 K splats at a low-angle close camera; v14's 10 K rendered as
  smeared streaks.
- **Sideboard** — cleanest extraction in the scene, easiest lock.
  FOV 50, eye looking back at the alcove.

Final spec for each is in `docs/showcase/wipe/cameras.json`.

---

## 9. Useful paths

```
pipeline/run_all.py                     orchestrator
pipeline/extract_one.py                 stage 1 (visual hull)
pipeline/sam_carve.py                   stage 2 (wide SAM)
pipeline/floor_drop.py                  stage 3
pipeline/sam_tight.py                   stage 4 (with NO_CROP_TOKENS)
pipeline/sweep_fallback.py              stage 5 generic
pipeline/bookshelf_sweep.py             stage 5 for shelves
pipeline/tv_carve.py                    TV pitch sweep
pipeline/_phase4_art_extract.py         wall art yaw+pitch vote
pipeline/companion_search.py            items-on-parent re-prompt
pipeline/split_children.py              TV-stand keep-whole rule
pipeline/sam_server.py                  SAM3 FastAPI service
pipeline/render_wipe_pairs.py           one-off camera renders
pipeline/build_showcase.py              matplotlib grid

docker/Dockerfile                       image with vLLM + sam_server
docker/entrypoint.sh                    launches both, exports URLs
run_pipeline.sh                         single-GPU driver
run_parallel.sh                         multi-GPU round-robin driver

docs/showcase/wipe/cameras.json         locked camera specs
docs/showcase/wipe/_grid.png            README hero image
docs/showcase/wipe/<slug>_pair.png      labeled side-by-sides
docs/PIPELINE_TOUR.md                   this file
```
