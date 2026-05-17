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
COARSE = [0.3, 0.4, 0.5, 0.6]
FINE_STEP = 0.05
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


def qwen_pick(candidates, label):
    """candidates: list of dicts {thresh, frac_kept, renders:[y0,y180]}.
    Returns the index Qwen picks (legs/supports first, then cleanest)."""
    content = []
    for i, c in enumerate(candidates):
        content.append({"type": "text",
                         "text": f"\n--- CANDIDATE {i+1} "
                                 f"(strength {c['thresh']:.2f}, "
                                 f"keeps {100*c['frac_kept']:.1f}%) ---"})
        for rp in c["renders"]:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_b64(rp)}"}})
    content.append({"type": "text", "text":
        f"Each numbered candidate is the SAME extracted {label}, cleaned at "
        f"an increasing strength. Higher strength removes more floor smear / "
        f"junk from underneath the object, but too high starts eroding the "
        f"object itself.\n\n"
        f"ABSOLUTE PRIORITY — LEGS / FEET / SUPPORTS MUST STAY INTACT.\n"
        f"For every candidate, look at the y180 (back) and y0 (front) views "
        f"and trace each leg / foot / support / pedestal / base from the "
        f"body of the {label} down to where it meets the floor. If a "
        f"candidate has ANY leg or support that is missing, shortened, "
        f"thinned, or floating (a gap appeared under it), that candidate is "
        f"DISQUALIFIED — never pick it, no matter how clean it looks.\n\n"
        f"Among ONLY the candidates whose legs and supports are all fully "
        f"intact, pick the one with the cleanest area under and around the "
        f"{label} (least leftover floor smear, wisps, or junk).\n\n"
        f"If in doubt, prefer the lower-strength (earlier) candidate — "
        f"keeping the object whole beats removing a little extra junk.\n\n"
        f"Reply with ONLY the candidate number (1 to "
        f"{len(candidates)}). No other text."})

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


def carve(raw, s, thr):
    return s >= thr


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
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    out_root = (args.out_dir.resolve() if args.out_dir else obj)
    out_root.mkdir(parents=True, exist_ok=True)

    in_ply = args.in_ply
    if in_ply is None:
        # same final-PLY priority as extract_final_outputs (minus stage 6)
        for cand in ("5_subtracted.ply", "5_bookshelf_sweep.ply", "4_rug.ply",
                     "5_sweep_fallback.ply", "4_sam_tight.ply"):
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

    # EXEMPTION — bookshelves / open shelving skip this step entirely.
    # Their open structure (shelves, gaps, contents) is too complex for a
    # silhouette inside/outside carve; pass the input through unchanged.
    if is_shelving(label):
        print(f"[inside_outside] SKIPPED — '{label}' is a shelving/bookshelf "
              f"class, exempt from inside/outside refinement (too "
              f"structurally open). Input passed through unchanged.")
        out_ply = out_root / "6_inside_outside.ply"
        shutil.copy(str(in_ply), str(out_ply))
        render_canonical_5(out_ply, out_root / "renders" / "6_inside_outside")
        diag = out_root / "diagnostics" / "6_inside_outside"
        diag.mkdir(parents=True, exist_ok=True)
        (diag / "report.json").write_text(json.dumps({
            "stage": "inside_outside", "skipped": True,
            "reason": "shelving/bookshelf class exempt",
            "label": label, "input_ply": str(in_ply),
            "output_ply": str(out_ply),
        }, indent=2))
        print(f"[done] {out_ply} (passthrough — exempt)")
        return

    mask_dir = (args.mask_dir or (obj / "diagnostics" / "4_sam_tight")).resolve()
    cam_json = mask_dir / "cameras.json"
    if not cam_json.exists():
        print(f"[inside_outside] SKIPPED — no sam_tight masks at {cam_json} "
              f"(object didn't go through sam_tight); nothing to refine.")
        return

    print(f"[inside_outside] input PLY : {in_ply}")
    print(f"[inside_outside] mask dir  : {mask_dir}")

    # --- load tight masks (raw, un-padded) ---
    cams = json.load(open(cam_json))["cameras"]
    masks = []
    n_skip = 0
    for cm in cams:
        mp = mask_dir / f"mask_{cm['tag']}.png"
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
    print(f"[inside_outside] {len(masks)} tight masks ({n_skip} empty skipped)")

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
                keep = carve(raw, s, t)
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

        print("[auto] coarse sweep:", COARSE)
        coarse = evaluate(COARSE, "coarse")
        ci = qwen_pick(coarse, label)
        t0 = coarse[ci]["thresh"]

        fine_list = sorted({round(max(THR_MIN, min(THR_MAX, t0 + d)), 2)
                            for d in (-FINE_STEP, 0.0, FINE_STEP)})
        print(f"[auto] fine sweep around {t0:.2f}:", fine_list)
        fine = evaluate(fine_list, "fine")
        fi = qwen_pick(fine, label)
        thresh = fine[fi]["thresh"]
        print(f"[auto] Qwen-chosen threshold = {thresh:.2f}")
    else:
        thresh = args.keep_thresh

    # --- final carve ---
    keep = carve(raw, s, thresh)
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
        "mask_dir": str(mask_dir),
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
