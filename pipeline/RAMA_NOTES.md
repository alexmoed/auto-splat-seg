# RAMA notes (parked 2026-05-03)

We tried RAMA multicut clustering as a Stage 5 cleanup pass and parked it.
Keeping these notes so a future me doesn't re-attempt the same path
without first reading why it failed.

## What is RAMA

RAMA = Rapid Multicut Algorithm. A graph-clustering solver that takes a
weighted graph (nodes + edges) and partitions nodes into clusters by
deciding which edges to "cut". Each edge has a SIGNED cost in logit
space:

  signed_cost = log(w / (1 - w)) - bias

  - positive cost  → "keep edge in same cluster"
  - negative cost  → "cut edge, separate clusters"

Output: a cluster id per node. RAMA picks the natural cluster count from
the cost landscape — no `--n-clusters` argument.

Lives in conda env `claude_seg_rama` as the `rama_py` python wrapper.
The `rama_py.rama_cuda(i_list, j_list, signed_costs, opts)` call returns
the cluster id array.

## How we wired it (this attempt — `rama_cluster.py`, deleted)

Input: `4_sam_tight.ply`.

1. Load splats → xyz, normals (quaternion + smallest-scale axis), SH
   degree-zero color (`f_dc_*`).
2. kNN graph (k=16) over xyz → ~360k undirected edges.
3. Edge weight = spatial-proximity × |normal-alignment| × color-similarity
   (each Gaussian on adaptive sigma).
4. RAMA multicut on the signed costs.

Pattern was lifted from the plugin's `pair_separation.py`
(`~/.claude/skills/pointcloud-segmentation/scripts/pair_separation.py`),
which uses RAMA to split TWO classes from one merged hull (chair tucked
under desk).

Per-cluster outputs:
  - cluster_<NN>.ply — splats from that cluster, full SH preserved
  - cluster_<NN>/{y0,y90,y180,y270}.png — 4 yaws rendered with the
    LOCKED CAMERA from the parent (4_sam_tight.ply) so every cluster
    shows in its actual position/scale within the chair

## What happened on the armchair

bias=0:    15,331 clusters total, largest 29 splats (0.1% of input)
bias=-3:   3,653 clusters, largest 1,375 splats (3.8%)
bias=-6:   1,033 clusters total, 13 ≥500 splats, largest 4,451 (12.2%)

At bias=-6 the top-13 clusters covered ~69% of the chair. Per-cluster
SH renders showed:

  - 0008 (4451): brown blanket region + back fragments
  - 0000 (4063): tiny dark wisp floating where chair top should be (NOISE)
  - 0002 (2814): beige right armrest only
  - 0007 (2610): blanket detail + chair leg + pillow edge
  - 0001 (2212): left armrest with reaching wisp
  - 0005 (1827): striped pillow (mostly dark stripe)
  - …7 more smaller fragments

## Why we parked it

The clusters do NOT cleanly separate "chair" from "noise/halo". They
fragment the chair body itself into pieces by material/normal/color
similarity. Telling Qwen "is this cluster part of the chair" then asks
Qwen to make 13+ yes/no decisions on visually similar fabric panels —
each one a chance for Qwen to drop a real chair piece.

Failure modes:
  - bias too high (less negative) → 1000s of tiny clusters. Useless.
  - bias too low (more negative) → chair fuses with halo into one cluster.
    Defeats the purpose.
  - sweet spot fragments the chair. No bias setting produces "1 chair
    cluster + N noise clusters".

This is fundamentally because RAMA is class-separation tooling
(pair_separation.py's job: chair-vs-desk where the materials differ
strongly). Cleaning up halo from a SINGLE object is a different
problem — the halo splats sit on the same surface continuum as the
real material; there's no edge cost that separates "chair" from "5cm
of fuzzy halo around the chair".

## When to revive RAMA

Use it for the original `pair_separation.py` use case: TWO objects
visually merged into one extracted hull (chair tucked under desk, lamp
pressed against table base, two adjacent dining chairs). NOT for
single-object cleanup.

## What to NOT try again on a single object

- Bias sweeps with the kNN+normal+SH weighting — we did 0, -3, -6,
  none of them gave us "1 big cluster".
- Asking Qwen to keep/drop per-cluster on 13+ similar fabric panels.
- Largest-connected-component as Stage 5 — we didn't run this, but
  almost any halo-splat is connected to the chair via bridge splats,
  so the LCC = the whole input, useless.

## Files removed when we parked

- `pipeline/rama_cluster.py` (deleted)
- `02_armchair/diagnostics/5_rama/` (deleted)
- The "Stage 5" slot in PIPELINE_ORDER.md / OUTPUT_STRUCTURE.md is now
  "sam_tight is final".

## Final pipeline as of 2026-05-03

  1. visual_hull   — extract_one.py
  2. sam_wide      — sam_carve.py
  3. floor_drop    — floor_drop.py
  4. sam_tight     — sam_tight.py   ← FINAL output
