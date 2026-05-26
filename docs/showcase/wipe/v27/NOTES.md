# v27 wipe-pair renders — work-in-progress notes

**Date paused:** 2026-05-23
**Scene:** `Kitchen_living_dining_v27`
**Backgrounds:** `step7_cardinal_aligned.ply` (3,000,000 splats, untouched rotated scan — object STILL in frame in the "before" image)
**Resolution:** 1080×1080 (1:1 square) — switched from 16:9 mid-session, will need website CSS update on deploy.
**FOV:** 50°, y-down convention, up = `(0, -1, 0)`.

---

## Locked cameras (per object)

All eye/target stored in `cameras.json`. Re-render with:

```bash
/home/ubuntu/anaconda3/envs/claude_seg/bin/python \
  /home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py \
  <object_ply> <out_png> \
  --eye=X,Y,Z --target=X,Y,Z --y-down --fov 50 --width 1080 --height 1080
```

### ✅ Armchair — LOCKED
- PLY: `02_beige_upholstered_armchair_with_wooden_legs/7_final.ply` (34,991 splats)
- eye: `(-1.94494, 0.41473, 0.23318)`
- target: `(-1.92972, 0.45000, 0.32792)`
- History: started at v26-locked → panned 0.2m left → dollied 0.2m forward (in hindsight too aggressive, distance-to-target only 0.10m so tiny tilts swept huge angles) → wild tilt iterations → reverted target_y → small tilt to 0.450 → small pan 0.1m left → DONE.

### ✅ Bookshelf — LOCKED
- PLY: `02_light_wood_bookshelf/7_final.ply` (75,474 splats)
- eye: `(-1.58786, 0.02322, -0.19613)`
- target: `(-1.46103, 0.65000, 2.68589)`
- History: long aim-y iteration (started at chair eye aim-up, ended much lower) → eye raised 1m → panned 0.5m right → multiple lower/raise iterations → 2× pan 0.2m left → target_y settled at 0.65 ("ok that works").

### ✅ Coffee table — LOCKED
- PLY: `02_light_oak_coffee_table_with_black_metal_frame/7_final.ply` (19,994 splats)
- eye: `(-4.80930, 0.11880, 2.06820)`
- target: `(-3.93010, 1.40404, 0.96220)`
- History: tried 30° orbit, undone to 15°, then switched to pure pan: 0.3m left + 0.1m right. No tilt or orbit in final.

### ⏭ Sideboard — SKIPPED for v27
- Not iterated. Being swapped OUT of the website slider in favor of the bookshelf.
- v26-locked render still exists in the v27 folder for completeness but is NOT a deploy target.

---

## Deploy plan (not yet executed)

When all 4 are locked:

1. Convert each `*_object.png` + `*_background.png` to `.webp` (lossy ~85, matching existing format).
2. Copy webps + pngs into `/home/ubuntu/splat_viewer/website/highlights/`, backing up existing as `.pre_v27_20260522` (matches user's existing backup convention).
3. Edit `/home/ubuntu/splat_viewer/website/pipeline-demo.html`:
   - CSS: change `.wipe-stage { aspect-ratio: 16/9; }` → `aspect-ratio: 1/1;`
   - Lines 134-152: replace the 3rd `<figure class="wipe-card">` (currently sideboard) with the bookshelf — caption `Light-wood bookshelf`, src paths `highlights/light_wood_bookshelf_*.webp`.
   - Final 3 sliders: Armchair, Coffee table, Bookshelf (sideboard dropped).
4. No git push yet — user decides timing.

---

## Earlier-session idea (parked, not built)

**Multi-state progressive wipe:** one slider that walks through 4 states at the bookshelf camera —
1. Full scan (chair + bookshelf both visible)
2. Scan minus chair
3. Scan minus chair AND bookshelf
4. Just background

Would need 4 renders (the 2 intermediates require running `extract_background.py` with custom object lists) and a 4-layer CSS clip-path slider mechanic. User said "yes that makes sense" but we pivoted to per-object camera tuning. Pick up if/when they want it.

---

## Files

- `cameras.json` — all 4 locked eye/target + common block
- `scripts/render_locked.py` — driver that reads cameras.json, swaps PLY paths to a target scene, re-renders all 4
- `scripts/render_wipe_pairs.py` — auto-compute camera version (used for bookshelf's initial front-on)
- `scripts/build_showcase.py` — matplotlib side-by-side `_pair.png` + `_grid.png` (we said "no grid" mid-session — script still works but the grid output is not the deliverable)
- `*_object.png` + `*_background.png` — the actual deliverables (4 × 2 = 8 PNGs)
- `*_pair.png` — debug side-by-sides (for visual review only, not for deploy)

---

## Conventions remember

- `--y-down` flag is REQUIRED. Without it, scenes render upside-down.
- "right" in world space ≈ `+x` for these cameras (forward is mostly `+z`).
- target_y SMALLER = aim UP (in y-down). target_y LARGER = aim DOWN.
- eye_y SMALLER = camera HIGHER in real space. eye_y LARGER = camera LOWER.
- Camera-orbit math: rotate `(eye - target).xz` around y by θ, recompute eye = target + rotated_offset.
