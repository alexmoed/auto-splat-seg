#!/usr/bin/env python3
"""Stage 6 — inside/outside refinement.

A final cleanup pass after sam_tight / sweep_fallback. Pools the
4_sam_tight SAM masks ONLY (tight masks — they already include the
extra low camera rings), projects every splat through each saved
camera, and computes per-splat INSIDENESS = fraction of mask-views the
splat lands inside. Splats below the keep threshold are dropped.

Tight masks only by design: sam_wide masks are deliberately loose and
would relax the vote. Uses the raw `mask_<tag>.png` (un-padded) masks.

THRESHOLD SELECTION
  --keep-thresh X   fixed threshold (default 0.5)
  --auto            Qwen picks the threshold: a coarse sweep is rendered
                    and shown to Qwen, then a fine sweep around its pick.
                    Qwen's ABSOLUTE PRIORITY is preserving the object's
                    legs / feet / supports — any candidate that erodes a
                    leg/support is disqualified; among leg-safe candidates
                    it picks the cleanest underside.

    python inside_outside.py <obj_dir> [--auto | --keep-thresh 0.5]
                             [--label "..."] [--in-ply P] [--mask-dir D]
                             [--out-dir D]
"""
import argparse
import base64
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from view import load_gsplat_ply, render_splat, rotation_matrix_from_yaw_pitch  # noqa
from extract_one import viewmat_look_at, build_K, RENDER_MARGIN  # noqa
from floor_drop import render_canonical_5  # noqa

KEEP_THRESH = 0.5
MIN_MASK_PX = 2000
# 3-threshold Qwen-pick sweep (locked 2026-05-20). Was a 4-coarse +
# 3-fine two-stage sweep; replaced by a single one-shot sweep at three
# spread-out thresholds + one Qwen call. Validated on sofa: Qwen picked
# 0.60 with the new prompt. 0.30 is just above the prior fixed default
# (0.25) — light end. 0.60 is the aggressive end. 0.45 is the middle.
SWEEP_3 = [0.30, 0.45, 0.60, 0.75, 0.85]  # expanded 2026-05-27 — bookshelf wanted higher than 0.60; let Qwen pick across the full ladder including the high-strength end.
THR_MIN, THR_MAX = 0.1, 0.7
FOV, W, H = 70.0, 1920, 1080
# Candidate sweep renders shown to Qwen are framed WIDE (object well
# inside the frame with margin) so Qwen can actually see the legs/feet
# and any junk under/around the object. NOT the tight RENDER_MARGIN.
QWEN_VIEW_MARGIN = 2.05
QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

# Bookshelves / open shelving are EXEMPT: their structure is too open
# (shelves, gaps, items inside) for a silhouette inside/outside carve —
# the test would wrongly cut interior contents and see-through gaps.
SHELVING_KEYWORDS = ("bookshelf", "bookcase", "book shelf", "shelving",
                     "shelf", "etagere", "etagère", "étagère")


def is_shelving(label):
    lo = (label or "").lower()
    return any(k in lo for k in SHELVING_KEYWORDS)


# ----------------------------------------------------------------------
def insideness(xyz, masks):
    """Per-splat fraction of mask-views the splat projects inside.
    Projection matches sam_tight.vote_carve (-z OpenGL convention)."""
    n = len(xyz)
    Xh = np.concatenate([xyz, np.ones((n, 1))], axis=1)
    num = np.zeros(n)
    den = np.zeros(n)
    for V, K, m in masks:
        Hh, Ww = m.shape
        cam = (Xh @ V.T)[:, :3]
        zc = -cam[:, 2]
        front = zc > 0.01
        xs = K[0, 0] * cam[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]
        ys = K[1, 1] * cam[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
        xi = xs.astype(np.int32)
        yi = ys.astype(np.int32)
        inb = front & (xi >= 0) & (xi < Ww) & (yi >= 0) & (yi < Hh)
        gi = np.where(inb)[0]
        hit = np.zeros(n, dtype=bool)
        hit[gi] = m[yi[gi].clip(0, Hh - 1), xi[gi].clip(0, Ww - 1)]
        num += hit
        den += inb
    return np.where(den > 0, num / np.maximum(den, 1), 0.5)


def build_cam(center, yaw, pitch, distance):
    center = np.asarray(center, dtype=np.float32)
    base_eye = center + np.array([0, 0, distance], dtype=np.float32)
    R = rotation_matrix_from_yaw_pitch(yaw, pitch)
    eye = center + R.T @ (base_eye - center)
    up = np.array([0, -1, 0], dtype=np.float32)
    return viewmat_look_at(eye, center, up), build_K(FOV, W, H)


def render_y0_y180(ply_path, out_dir, center, distance):
    """Render y0 (front) + y180 (back — best for checking legs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = load_gsplat_ply(str(ply_path))
    paths = []
    for yaw in (0, 180):
        V, K = build_cam(center, yaw, -20.0, distance)
        img = render_splat(scene, V, K, W, H, bg=(1.0, 1.0, 1.0))
        p = out_dir / f"y{yaw}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def encode_b64(p, max_dim=1024):
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def qwen_pick(candidates, label, pipe_union=""):
    """candidates: list of dicts {thresh, frac_kept, renders:[y0,y180]}.
    pipe_union: the sam_prompt.txt pipe-union string (main + sub-items).
    Returns the index Qwen picks.

    Prompt is CLASS-SPECIFIC (2026-05-29):
      - RIGID storage (cabinet / bookshelf / shelving) → the OLD v32 prompt
        ("disqualify eroded, then PREFER the HIGHER strength"), which landed
        the clean 0.60 on the v32 bookshelf + display shelf.
      - SOFT / other furniture → conservative completeness-first prompt (the
        aggressive bias over-carved soft bodies — armchair @ 0.75).
    The mechanical collapse-guard in main() removes the body-collapsing rungs
    for BOTH before this pick, so 'prefer higher' can't run off the cliff."""
    content = []
    for i, c in enumerate(candidates):
        content.append({"type": "text",
                         "text": f"\n--- CANDIDATE {i+1} "
                                 f"(strength {c['thresh']:.2f}, "
                                 f"keeps {100*c['frac_kept']:.1f}%) ---"})
        for rp in c["renders"]:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_b64(rp)}"}})
    # ── Class-specific picker prompt ──────────────────────────────────────
    # RIGID storage (cabinets / bookshelves / shelving) has hard edges and
    # delicate shelf clutter: the OLD v32 prompt ("disqualify eroded, then
    # PREFER the HIGHER strength") reliably landed the clean 0.60 there. SOFT
    # / other furniture keeps the conservative completeness-first prompt — the
    # aggressive bias over-carves soft upholstered bodies (armchair @ 0.75).
    # The mechanical collapse-guard in main() removes the body-collapsing rungs
    # for BOTH before this pick, so "prefer higher" can't run off the cliff.
    lo = (label or "").lower()
    is_rigid_shelving = any(k in lo for k in (
        "cabinet", "cupboard", "bookshelf", "bookcase", "book shelf",
        "shelf", "shelves", "shelving", "etagere", "etagère", "étagère",
        "sideboard", "credenza", "hutch"))
    if is_rigid_shelving:
        # OLD v32 prompt — validated on the v32 bookshelf + display shelf
        # (both picked 0.60 cleanly).
        instruction = (
            f"Each numbered candidate shows the SAME extracted '{label}', "
            f"cleaned at an increasing strength. The pipe-union the upstream "
            f"SAM step used was:\n\n  {pipe_union}\n\n"
            f"The MAIN object is '{label}'. The SUB-ITEMS listed above (books, "
            f"vases, plants, baskets, decor, hardware, etc.) are also part of "
            f"this extraction and must be preserved.\n\n"
            f"Anything else in the renders — floor halo / smear under or "
            f"behind the object, wisps, neighbouring furniture, walls, capture "
            f"noise — is contamination this step is removing. Higher strength "
            f"removes more contamination but eventually starts eroding the "
            f"frame, shelves, or shelf items.\n\n"
            f"**DISQUALIFY** any candidate where:\n"
            f"- The frame / shelves / body of the '{label}' is eroded, broken, "
            f"or has chunks missing\n"
            f"- Any shelf item (book, vase, plant, decor, hardware) is gone, "
            f"eroded, or thinned\n\n"
            f"Among candidates that PASS the disqualification rules, **PREFER "
            f"THE HIGHER-STRENGTH (more aggressive) CANDIDATE** — it removes "
            f"the most floor/background haze — as long as the frame, shelves, "
            f"and every shelf item are clearly intact. Only fall back to a "
            f"lower strength if the higher one shows clear damage to the "
            f"object or a shelf item.\n\n"
            f"Reply with ONLY the candidate number (1 to {len(candidates)}). "
            f"No other text.")
    else:
        # Conservative completeness-first prompt for soft / other furniture.
        instruction = (
            f"Each numbered candidate shows the SAME extracted '{label}', "
            f"cleaned at an increasing strength (higher strength removes more "
            f"surrounding contamination but eventually starts eating the "
            f"object itself). The pipe-union the upstream SAM step used was:"
            f"\n\n  {pipe_union}\n\n"
            f"The MAIN object is '{label}'. The SUB-ITEMS listed above "
            f"(pillows, throws, blankets, lamps, decor, hardware, etc.) are "
            f"also part of this extraction and must be preserved.\n\n"
            f"RULE #1 — NEVER CHOOSE A CANDIDATE THAT DESTROYS THE OBJECT. The "
            f"complete '{label}' must be present and SOLID: its body / seat / "
            f"back / arms / frame and every sub-item must be fully there — not "
            f"hollowed out, fragmented, reduced to a thin shell, or partly "
            f"missing. A candidate that looks 'cleaner' but is missing the "
            f"body or large chunks of the object is DESTROYED and is "
            f"DISQUALIFIED no matter how little floor/halo it has.\n\n"
            f"RULE #2 — Among ONLY the candidates where the object and all "
            f"sub-items are FULLY INTACT and solid, pick the one that removes "
            f"the most contamination (floor halo / smear under the object / "
            f"wisps / neighbouring furniture / walls / capture noise). Slight "
            f"thinning of legs/feet is acceptable; a missing, broken, or "
            f"floating-with-a-gap leg is not.\n\n"
            f"RULE #3 — Prefer the HIGHEST strength at which the object is "
            f"STILL COMPLETELY INTACT (removes the most halo). Step down only "
            f"if the higher one loses ANY of the object's body or sub-items.\n\n"
            f"Reply with ONLY the candidate number (1 to {len(candidates)}). "
            f"No other text.")
    content.append({"type": "text", "text": instruction})

    payload = json.dumps({
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200, "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(QWEN_URL + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    # parse first integer in range
    pick = None
    for tok in raw.replace(",", " ").replace(".", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(candidates):
            pick = int(tok) - 1
            break
    if pick is None:
        print(f"[qwen] could not parse pick from '{raw}' — defaulting to middle")
        pick = len(candidates) // 2
    print(f"[qwen] reply='{raw}' -> candidate {pick+1} "
          f"(thresh {candidates[pick]['thresh']:.2f})")
    return pick


def carve(raw, s, thr, y_band_bottom_pct=None, y=None):
    """True = keep splat.

    If y_band_bottom_pct is set, ONLY splats in the bottom N% of the
    object's y-range (real-world bottom = larger y in y-down) are subject
    to the insideness threshold. Splats in the upper (100 - N)% are
    always kept. Use this when the upper structure of the object has
    sparse interior content (open shelving, ladder racks) that the
    insideness vote would wrongly carve.
    """
    keep_base = s >= thr
    if y_band_bottom_pct is None or y is None:
        return keep_base
    y_min, y_max = float(np.min(y)), float(np.max(y))
    # y-down: bottom in real space = larger y. Bottom N% of y-range
    # starts at y_min + (1 - N/100) * (y_max - y_min).
    y_cut = y_min + (1.0 - y_band_bottom_pct / 100.0) * (y_max - y_min)
    in_upper = y < y_cut          # smaller y = higher in real space = upper part
    return keep_base | in_upper


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--keep-thresh", type=float, default=KEEP_THRESH)
    ap.add_argument("--auto", action="store_true",
                    help="Qwen picks the threshold (coarse sweep -> fine sweep)")
    ap.add_argument("--label", default=None,
                    help="object label (Qwen prompt + bookshelf exemption). "
                         "If omitted, read from 1_visual_hull_meta.json.")
    ap.add_argument("--in-ply", type=Path, default=None)
    ap.add_argument("--mask-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--y-band-bottom-pct", type=float, default=None,
                    help="If set, only carve splats in the bottom N% of "
                         "the y-range (real-world bottom). The upper "
                         "(100-N)% is preserved unconditionally. Useful "
                         "for shelving where the insideness vote eats "
                         "interior contents.")
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    out_root = (args.out_dir.resolve() if args.out_dir else obj)
    out_root.mkdir(parents=True, exist_ok=True)

    in_ply = args.in_ply
    if in_ply is None:
        # same final-PLY priority as extract_final_outputs (minus stage 6)
        for cand in ("5_subtracted.ply", "5_bookshelf_sweep.ply", "4_rug.ply",
                     "5_sweep_fallback.ply", "4c_sam_tight_high.ply",
                     "4b_sam_tight_low.ply", "4_sam_tight.ply"):
            if (obj / cand).exists():
                in_ply = obj / cand
                break
    if in_ply is None or not Path(in_ply).exists():
        print(f"[inside_outside] SKIPPED — no input PLY in {obj}")
        return

    # resolve label: CLI override, else object meta, else generic
    label = args.label
    if label is None:
        meta = obj / "1_visual_hull_meta.json"
        if meta.exists():
            try:
                label = json.load(open(meta)).get("label") or None
            except Exception:
                label = None
    label = label or "furniture object"

    # SHELVING EXEMPTION RETIRED 2026-05-27 — bookshelves now run through
    # inside_outside with the expanded SWEEP_3 ladder. Validated on
    # light_wood_bookshelf: Qwen picked 0.75 → clean carve, ~5% removed,
    # no shelf contents eroded. (Previously skipped because the original
    # 3-threshold sweep [0.30, 0.45, 0.60] couldn't reach high enough.)

    # Pool masks from BOTH SAM passes — Pass A high cameras
    # (diagnostics/4_sam_tight) and Pass B low cameras
    # (diagnostics/4b_sam_tight_low). The low cameras silhouette
    # under-object smear from below; pooling both gives the insideness
    # test full angular coverage. An explicit --mask-dir overrides.
    if args.mask_dir:
        mask_dirs = [args.mask_dir.resolve()]
    else:
        mask_dirs = [d for d in (obj / "diagnostics" / "4_sam_tight",
                                 obj / "diagnostics" / "4b_sam_tight_low",
                                 obj / "diagnostics" / "4c_sam_tight_high")
                     if (d / "cameras.json").exists()]
    if not mask_dirs:
        print(f"[inside_outside] SKIPPED — no sam mask dirs under {obj} "
              f"(object didn't go through sam_tight); nothing to refine.")
        return

    print(f"[inside_outside] input PLY : {in_ply}")
    print(f"[inside_outside] mask dirs : {[d.name for d in mask_dirs]}")

    # --- load tight masks (raw, un-padded) from all pooled dirs ---
    masks = []
    n_skip = 0
    for md in mask_dirs:
        cams = json.load(open(md / "cameras.json"))["cameras"]
        for cm in cams:
            mp = md / f"mask_{cm['tag']}.png"
            if not mp.exists():
                continue
            m = np.asarray(Image.open(mp).convert("L")) > 127
            if int(m.sum()) < MIN_MASK_PX:
                n_skip += 1
                continue
            masks.append((np.asarray(cm["V"], np.float64),
                          np.asarray(cm["K"], np.float64), m))
    if not masks:
        sys.exit("[fatal] no usable tight masks")
    print(f"[inside_outside] {len(masks)} tight masks pooled from "
          f"{len(mask_dirs)} pass(es) ({n_skip} empty skipped)")

    # --- per-splat insideness (computed once) ---
    pd = PlyData.read(str(in_ply))
    v = pd["vertex"]
    raw = v.data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    n_in = len(xyz)
    s = insideness(xyz, masks)
    lo, hi = xyz.min(0), xyz.max(0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    import math
    # wide framing for the Qwen candidate renders (render_y0_y180) so the
    # whole object + legs + surrounding area are visible for judging
    distance = (float((hi - lo).max()) * QWEN_VIEW_MARGIN) / (2 * math.tan(math.radians(FOV) / 2))

    # --- choose threshold ---
    if args.auto:
        sweep_root = out_root / "diagnostics" / "6_inside_outside" / "sweep"

        def evaluate(thr_list, tag):
            cands = []
            for t in thr_list:
                keep = carve(raw, s, t, args.y_band_bottom_pct, xyz[:, 1])
                d = sweep_root / f"{tag}_{t:.2f}"
                d.mkdir(parents=True, exist_ok=True)
                pth = d / "cand.ply"
                PlyData([PlyElement.describe(raw[keep], "vertex")],
                        text=False).write(str(pth))
                renders = render_y0_y180(pth, d, center, distance)
                cands.append({"thresh": t, "frac_kept": keep.mean(),
                              "renders": renders})
                print(f"  [{tag}] thr={t:.2f}: kept {100*keep.mean():.1f}%")
            return cands

        # Read pipe-union from sam_prompt.txt so Qwen sees the names of
        # the main object + every sub-item the chain has been preserving.
        # If missing (rare — sam_carve step 2 should always have run),
        # pass empty string and the prompt still works.
        sam_prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
        pipe_union = (sam_prompt_path.read_text().strip()
                       if sam_prompt_path.exists() else "")

        print(f"[auto] threshold sweep: {SWEEP_3}")
        cands = evaluate(SWEEP_3, "sweep")
        # Collapse-guard (2026-05-29): a threshold whose kept-fraction falls
        # off a CLIFF vs the previous (lower) threshold has carved away the
        # object BODY, not just the floor halo — never offer it to the picker.
        # Drop the first collapsing candidate and every stricter one.
        # Regression context: 478a866 expanded SWEEP_3 with 0.75/0.85 for
        # bookshelves; on the grey armchair 0.60 kept 89% (clean, = the v32
        # bar) but 0.75 kept only 50% (body gone) and Qwen picked it. A clean
        # carve step drops only a few %; a structural collapse drops tens of %.
        # Rigid objects (bookshelf) don't collapse at 0.75 (~5% drop) so they
        # keep the high rungs.
        COLLAPSE_DROP = 0.22   # relative kept-fraction drop signalling collapse
        kept_cands = [cands[0]]
        for prev, cur in zip(cands, cands[1:]):
            pf = prev["frac_kept"]
            rel_drop = (pf - cur["frac_kept"]) / pf if pf > 0 else 1.0
            if rel_drop > COLLAPSE_DROP:
                print(f"[auto] collapse-guard: thr {cur['thresh']:.2f} drops "
                      f"{100*rel_drop:.0f}% vs {prev['thresh']:.2f} "
                      f"(body collapse) — excluding it + all stricter thresholds")
                break
            kept_cands.append(cur)
        if len(kept_cands) < len(cands):
            print(f"[auto] picker sees safe thresholds only: "
                  f"{[c['thresh'] for c in kept_cands]}")
        ci = qwen_pick(kept_cands, label, pipe_union=pipe_union)
        thresh = kept_cands[ci]["thresh"]
        print(f"[auto] Qwen-chosen threshold = {thresh:.2f}")
    else:
        thresh = args.keep_thresh

    # --- final carve ---
    keep = carve(raw, s, thresh, args.y_band_bottom_pct, xyz[:, 1])
    n_kept = int(keep.sum())
    print(f"[inside_outside] insideness: <0.3={int((s<0.3).sum())}  "
          f"0.3-0.7={int(((s>=0.3)&(s<=0.7)).sum())}  >0.7={int((s>0.7).sum())}")
    print(f"[inside_outside] keep>={thresh:.2f}: "
          f"{n_kept}/{n_in} ({100*n_kept/n_in:.1f}%)")

    out_ply = out_root / "6_inside_outside.ply"
    PlyData([PlyElement.describe(raw[keep], "vertex")], text=False).write(str(out_ply))
    render_canonical_5(out_ply, out_root / "renders" / "6_inside_outside")
    diag = out_root / "diagnostics" / "6_inside_outside"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "report.json").write_text(json.dumps({
        "stage": "inside_outside",
        "input_ply": str(in_ply),
        "mask_dirs": [str(d) for d in mask_dirs],
        "n_masks_used": len(masks),
        "threshold": thresh,
        "auto": bool(args.auto),
        "n_in": n_in, "n_kept": n_kept,
        "frac_kept": round(n_kept / n_in, 4),
        "output_ply": str(out_ply),
    }, indent=2))
    print(f"[done] {out_ply}")


if __name__ == "__main__":
    main()
