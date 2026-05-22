# Bookshelf route — RETIRED from auto-routing 2026-05-22

The bookshelf procedure was disconnected from `procedure_dispatch.decide_procedure`
on 2026-05-22. The general route extracts bookshelves at least as cleanly
(validated on the light-wood bookshelf and the tall metal-frame bookshelf —
the general route was marginally crisper with less haze), so the dedicated
route was retired to simplify the pipeline.

## What's still live
- `bookshelf_sweep.py` remains in `pipeline/` (this is an archived copy).
- `run_bookshelf` + `STAGE_BOOKSHELF_SWEEP` / `STAGE_BOOKSHELF_SWEEP_LOW` /
  `STAGE_SAM_TIGHT_BOOKSHELF` are still defined in `procedure_dispatch.py`.
- Still reachable manually: `procedure_dispatch.py <scene> <obj> --procedure bookshelf`
  — kept for website renders (the bookshelf route gives a tighter result).

## What changed
Only the auto-routing branch in `decide_procedure` was removed — bookshelf /
shelving labels now fall through to the `general` procedure.
