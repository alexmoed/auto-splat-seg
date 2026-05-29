# Open concerns — bookshelf/shelving fix session (2026-05-29)

Branch: `cleanup/codebase-review`. Latest relevant commit: **`a0ac3ef`** (bookshelf
front-arc imaging + capped sweep). Image `splat-pipeline:cleanup-test` rebuilt with
the fix baked (`ab4ea04071ce`); old image preserved as
`splat-pipeline:cleanup-test-pre-frontarc` (`42a1e9326ebb`).

This file tracks what is NOT done / still fragile. It is a living list — update it
as items are closed or new risks appear.

---

## Context: what was just fixed (so the concerns read in context)

The no-RANSAC bookshelf was eating its bottom shelf. Root cause was a **two-part
regression** from the "Major chain rework", NOT the picker prompt and NOT RANSAC:

1. **Full 360° orbit.** A bookshelf is categorically against a wall, so its
   back-hemisphere views see only wall and must be skipped (front-arc). v32 only
   escaped this because the per-object Qwen wall-adjacency check *happened* to write
   `wall_adjacent:true`. That check fails **silently** (empty `wall_adjacent.json`) →
   full 360 → SAM mask count ~doubles (v32: 26 → misfire: 58) → extra back/wall masks
   drag the bottom-shelf splats below the `inside_outside` cutoff → bottom shelf eaten.
   - **Fix:** `run_bookshelf` now ASSERTS `wall_adjacent:true` before the SAM chain
     (`_assert_wall_adjacent` in `procedure_dispatch.py`), so the proven geometric
     `compute_wall_skip` fires. It self-no-ops when the unit is freestanding (>2.5m
     from every wall), so asserting it is safe. Tagged `DO NOT REMOVE` in code.

2. **Sweep crept up to 0.75/0.85.** With front-arc, `inside_outside` @ 0.60 already
   keeps the whole unit (~92% = v32 bar); 0.75 is a cliff (−20%, eats the bottom).
   The prefer-higher picker grabs 0.75 if offered.
   - **Fix:** rigid shelving caps the `inside_outside` sweep at 0.60 (the v32 range);
     soft furniture keeps the full ladder. Reverted the 0.15 collapse-guard band-aid
     back to the uniform 0.22.

Verified (light wood bookshelf, from raw, `--procedure bookshelf`): wall-skip fires
(wall 0.18m), masks 58→30, picks 0.60 → 83,116 (92.2%) ≈ v32 84,400/96.6%, 8_final
80,362 vs v32 83,080. qwen-verifier PASS 4/4. AUTO path verified: object is mislabeled
`'wooden cabinet'` → `decide_procedure`=general, but `route_double_check` (Qwen on the
coarse hull) reads OPEN SHELVING → re-routes to bookshelf → front-arc → no RANSAC.

---

## CONCERN 1 — Branch not merged / image not promoted  *(your call)*

- `cleanup/codebase-review` (~16 commits incl. `a0ac3ef`) is **NOT merged to `main`**.
- `splat-pipeline:latest` is **unchanged** — still the pre-cleanup image. Only the
  test tag `cleanup-test` has the fix.
- **Action when ready:** merge `cleanup/codebase-review` → `main`, then rebuild +
  retag `splat-pipeline:latest` from the merged tree.
- **Risk if forgotten:** any run that uses `:latest` (not `cleanup-test`) does NOT
  have the front-arc / capped-sweep fix and will regress the bookshelf.

## CONCERN 2 — Cabinets have the SAME latent 360° bug  *(not fixed — stayed focused on shelving)*

- Cabinets/sideboards route to the **general** procedure (they keep RANSAC, by design).
  But the general route does **NOT** assert wall-adjacency — it still trusts the same
  flaky per-object Qwen `wall_adjacent.json` check that silently misfired on the
  bookshelf.
- A cabinet IS categorically wall-adjacent (back face flush against wall). `sam_tight`'s
  own comment notes wall-skip matters for cabinets ("backside views see
  wall-through-body and the vote drops the cabinet body"). So the identical
  silent-360 failure can eat a cabinet's body / back.
- **Suggested fix:** assert wall-adjacency for storage furniture on the general route
  too (e.g. when the label is a closed cabinet / sideboard / credenza / hutch). This is
  orthogonal to RANSAC — cabinets keep RANSAC; they just also get front-arc imaging.
  Mirror `_assert_wall_adjacent`, gated on the storage-label set.
- **Status:** deliberately deferred this session (user said stay on the shelf). Surfaced.

## CONCERN 3 — Detection mislabels open shelving as "cabinet"

- The inventory/detection step labelled this open shelving unit `'wooden cabinet'`.
  Today the **only** thing that saves the AUTO route is `route_double_check` (a Qwen
  open-vs-closed call on the coarse hull). It is a **backstop**, not a fix.
- **Risk:** if `route_double_check` is ever unsure / errors / the hull renders are
  missing, the object stays on the general route (RANSAC) and the bottom shelf gets
  eaten — exactly the failure we just fixed, re-entered through the label.
- **Real fix:** improve the detection/inventory prompt so open shelving is labelled
  shelving (not cabinet) in the first place. Then routing doesn't depend on the
  Qwen backstop firing.

## CONCERN 4 — The fragility PATTERN (the "why do we do this over and over")

- The deliberate design rule **"bookshelves don't orbit 360° — they're against a wall"**
  was encoded only in (a) a flaky runtime Qwen guess and (b) memory notes. So every
  chain rework silently re-rolled it, and when the guess misfired we re-derived the same
  lesson from scratch.
- **Principle going forward:** categorical, class-level design decisions belong baked
  into the **route as a hard assertion** (with a `DO NOT REMOVE` comment), not in a
  per-object runtime check that can fail quietly. The front-arc fix now follows this;
  Concern 2 (cabinets) is the next place to apply it.
- **Secondary:** the wall-adjacency check fails **silently** (empty/false json → no-op,
  keep all cameras). A silent "keep everything" is the dangerous default for a
  wall-flush object. Consider making the empty/unreadable case at least LOUD, or
  route-asserted per class (preferred).

---

## Minor / cosmetic

- **Object-dir rename churn:** the verified bookshelf ended up at
  `02_wooden_shelving_unit_3` (the deferred rename collided with `_2`/itself). Cosmetic,
  not a correctness issue.
- **Display shelf (`02_wooden_shelving_unit`)** was verified clean earlier at 0.60 and
  was NOT re-run with the new front-arc + capped-sweep code. The fix can only make it
  cleaner or equal (cap ≤0.60, front-arc reduces masks), so no regression expected — but
  it has not been re-verified post-fix.
