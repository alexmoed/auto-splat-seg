# qc_reject.py — RETIRED 2026-05-22

The standalone mid-chain QC reject pass is retired. It used to run inside
`procedure_dispatch._post_extract_qc` right after sam_tight, ask Qwen
PASS/REJECT on the 4_sam_tight renders, and on REJECT move the whole
folder to `rejects/`. Problem: it judged a mid-chain stage — an upstream
bug (e.g. the floor-lamp framing clip) could bury a genuinely good object
before stage_pick ever ran.

## Replacement
The keep/reject verdict is now folded into `info.py`'s final Qwen call as
`info.json["condition"]` ("good" | "reject"), assessed on the picked
`7_final` result. `rename_to_qwen.py` moves any reject-flagged folder to
`rejects/` at the very end; everything else is renamed normally.

`qc_reject.py` is left in `pipeline/` (unused) for reference; the dispatch
no longer calls it. `_run_qc` / `_move_to_rejects` / `STAGE_QC_REJECT` in
procedure_dispatch.py are now dead code.
