# Output file structure — authoritative

This file is THE layout. If anything (script docs, my memory, our chat) disagrees with this file, this file wins.

Read this BEFORE writing or editing any script that produces output.

Last updated: 2026-05-02

---

## Top-level scene directory

```
<scene>/
├── raw_*.ply                          (raw input scan)
├── step1_ydown.ply                    (rotate_zup_to_ydown output)
├── step3_tilt_corrected.ply           (tilt_correct output)
├── step4_yaligned.ply                 (y_axis_align PCA pass)
├── step6_hough_corrected.ply          (orient.py final)
├── step7_cardinal_aligned.ply         (cardinal_pick.py final)
├── step7_sliced.ply                   (slice.py final — input to inventory + extract)
├── orient_status.json                 (chain log from orient.py)
├── floor_plane.json                   (RANSAC floor plane)
├── _cardinal_temp/                    (cardinal_pick artifacts; safe to delete after step 3 locks)
│   ├── candidates/                    (coarse + fine sweep PNGs)
│   ├── temp_strip.ply                 (throwaway tight-cropped temp PLY)
│   ├── temp_strip_topdown.png
│   ├── cardinal_choice.json           (winner + reasoning)
│   └── step3_full_ply_topdown.png     (verification render after rotation applied)
├── _slice_temp/                       (slice.py per-step JSONs + final topdown)
│   ├── slice_step1.json … slice_step7.json
│   └── step7_sliced_topdown.png
├── _inventory_temp/                   (inventory.py outputs)
│   ├── topdown_for_qwen.png           (4K topdown sent to Qwen)
│   ├── qwen_pass1_large_furniture_raw.txt
│   ├── qwen_pass2_rugs_raw.txt
│   ├── qwen_pass3_other_raw.txt
│   ├── qwen_items.json                (items + camera + image_size)
│   └── qwen_overlay.png               (bboxes drawn on topdown)
└── 02_<slug>/                         (one per extracted object — see below)
```

`<slug>` examples: `02_armchair`, `02_sofa`, `02_dining_table`, `02_coffee_table_2`. Built from inventory label via slugify. If two items share a label, append `_2`, `_3`, etc.

---

## Per-object directory `02_<slug>/`

All stage outputs are prefixed with the stage number for sort order.

```
02_<slug>/
├── 1_visual_hull.ply                  (Stage 1 — rough cone back-projection)
├── 1_visual_hull_meta.json            (Stage 1 meta: bbox, camera, source, counts)
├── 1_visual_hull_topdown.png          (Stage 1 quick verification render)
│
├── 2_sam_wide.ply                     (Stage 2 — multi-view SAM, wide pad, conservative vote)
├── 3_floor_drop.ply                   (Stage 3 — RANSAC floor + normal-aware drop)
├── 4_sam_tight.ply                    (Stage 4 — multi-view SAM, tight pad, strict vote — FINAL PLY)
├── info.json                          (Stage 5 — Qwen description: object_type, sub_objects, materials, style, object_ply)
│
├── renders/
│   ├── 1_visual_hull/                 ★ 5 canonical views of 1_visual_hull.ply
│   │   ├── y0.png                     (front, pitch -20°)
│   │   ├── y90.png                    (left)
│   │   ├── y180.png                   (back)
│   │   ├── y270.png                   (right)
│   │   └── topdown.png                (pitch -89°)
│   ├── 2_sam_wide/                    (5 views of 2_sam_wide.ply — same file names)
│   ├── 3_floor_drop/                  (5 views of 3_floor_drop.ply)
│   └── 4_sam_tight/                   (5 views of 4_sam_tight.ply — FINAL)
│
└── diagnostics/
    ├── 2_sam_wide/                    (Stage 2 working files — sam_carve.py)
    │   ├── input_y0_p-15.png          (24 oblique inputs — yaw{0..330}/30 × pitch{-15,-45})
    │   ├── input_y0_p-45.png
    │   ├── ... (24 total oblique)
    │   ├── input_topdown.png
    │   ├── cameras.json               (V/K + eye/target per view, 25 entries)
    │   ├── sam_prompt.txt             (final accepted pipe-union prompt)
    │   ├── sam_prompt_raw.txt         (Qwen's first raw response)
    │   ├── sam_prompt_attempt2.txt    (only present if retry happened)
    │   ├── sam_prompt_attempt3.txt
    │   ├── sam_prompt_history.json    (all attempts + verdicts)
    │   ├── mask_y0_p-15.png           (raw mask after morph cleanup)
    │   ├── mask_padded_y0_p-15.png    (dilated by sam_pad m, used in vote)
    │   └── report.json                (per-view scores, hits, dilation_px, vote results)
    │
    ├── 3_floor_drop/                  (Stage 3 — Qwen-loop iterations + report)
    │   ├── iter0/, iter1/, iter2/, iter3/, iter4/   (per-iter renders)
    │   ├── iter0.ply, ..., iter4.ply   (per-iter PLYs)
    │   ├── qwen_iter{0..4}_raw.txt
    │   ├── qwen_pick_best_raw.txt
    │   ├── history.json
    │   └── report.json
    │
    └── 4_sam_tight/                   (Stage 4 — re-rendered views + tight masks)
        ├── input_<tag>.png × 25
        ├── cameras.json
        ├── mask_<tag>.png + mask_padded_<tag>.png
        └── report.json
```

---

## Naming rules

### Yaw tags
- **No leading zeros**: `y0`, `y30`, `y60`, `y90`, `y180`, `y270`, `y330` — NOT `y000`, `y090`.

### Pitch tags
- **Signed integer with explicit minus**: `p-15`, `p-45` — NOT `p+15`, `p15`, `p_-15`.

### Combined
- `y0_p-15`, `y90_p-45`, `topdown` (no yaw/pitch tags on topdown).

### Diagnostics file prefix
- Render INPUTS to SAM: `input_<tag>.png`
- Raw SAM masks (post-morph): `mask_<tag>.png`
- Dilated masks (used in vote): `mask_padded_<tag>.png`
- Camera JSON: `cameras.json`
- Per-stage report: `report.json`

### Per-stage isolation
- **Stage N writes ONLY to `diagnostics/<stage_name>/` and `renders/<stage_name>/`.**
- **Stage N's output PLY lives at `02_<slug>/<stage_name>.ply`** (root of object dir).
- **Stage N never edits files outside its own diagnostic + render folders or its own PLY.**
- **Stage N reads from prior stages' PLYs at root.**

Example: `sam_carve.py` reads `visual_hull.ply`, writes only to:
- `02_armchair/sam_wide.ply`
- `02_armchair/diagnostics/sam_wide/`
- `02_armchair/renders/sam_wide/`

It does NOT touch `visual_hull.ply` or `renders/visual_hull/` or `diagnostics/visual_hull/`.

---

## Render specs (locked)

| Param | Value |
|---|---|
| Resolution | 1920×1080 (1080p) |
| FOV | 70° |
| Background | white (1, 1, 1) |
| Y convention | y-down |
| Yaw 0 face direction | +z (front of object faces camera at y0) |
| Pitch sign | negative = camera tilts down toward floor |
| Distance | `extent × MARGIN / (2 × tan(fov/2))`; MARGIN see PIPELINE_ORDER.md |
| Topdown pitch | -89° (avoids gimbal lock at -90°) |
| Topdown up vector | `(0, 0, -1)` |
| Side-view up vector | `(0, -1, 0)` |

---

## SAM diagnostic specs (Stage 2 only)

| Param | Value |
|---|---|
| Resolution | 1920×1080 |
| Yaw set | `[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]` |
| Pitch set | `[-15°, -45°]` |
| + topdown | pitch -89° |
| Total views | **25** (12 yaws × 2 pitches + 1 topdown) |

---

## Things this file is NOT

- A description of the pipeline order — see `PIPELINE_ORDER.md`.
- A description of what each stage does algorithmically — that lives in each script's module docstring.
- A list of locked constants — see `PIPELINE_ORDER.md`.

---

## Update rule

If a script changes file paths, update this file FIRST. Then write code. Then re-run end-to-end on the armchair to validate the layout still matches.
