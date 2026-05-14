#!/usr/bin/env python3
"""split_children.py — split per-item children out of an extracted parent.

Reads the pipe-union prompt that Qwen already derived in sam_carve step 2
(saved at <obj>/diagnostics/2_sam_wide/sam_prompt.txt) and, for each
NON-PARENT prompt term, re-runs SAM3 against the kept views to produce
a tight per-item mask set, then votes splats from the source PLY
(default 5_sweep_fallback.ply, fall back to 4_sam_tight.ply) into
per-item child PLYs.

Output layout (inside the parent object dir):

    <obj>/children/00_<parent_slug>/object.ply   # parent minus all children
    <obj>/children/01_<child_slug>/object.ply
    <obj>/children/02_<child_slug>/object.ply
    ...
    <obj>/children/<NN>_<slug>/diagnostics/      # per-view masks
    <obj>/children/<NN>_<slug>/renders/          # canonical 5 views

Usage:
    python split_children.py <scene_dir> <obj_dir>
        [--source-stage 5_sweep_fallback]
        [--vote-frac 0.5]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ITER = Path(__file__).resolve().parent
sys.path.insert(0, str(ITER))

from sam_carve import (  # noqa: E402
    sam_segment, dilate_mask, morph_clean,
    parse_tagged_prompts,
    SAM_THRESHOLD, MIN_PROMPT_PX,
    render_canonical_5,
)

VOTE_FRAC = 0.5
MIN_VIEW_HITS = 5  # need at least this many SAM-mask views before voting
CHILD_SAM_PAD_HARD_M = 0.03
CHILD_SAM_PAD_FABRIC_M = 0.08

# TV-stand-class parents: when the parent's refined Qwen label contains
# any of these tokens, the cabinet should stay as ONE UNIT (closed
# cabinet, doors / contents not splittable). Only items literally ON
# TOP of it should split out as children — TV, speakers, soundbar,
# remote. Anything else Qwen names (books, headphones, electronic
# device, etc.) is either inside-cabinet noise or a hallucination and
# stays bundled with the cabinet body.
_TV_STAND_PARENT_TOKENS = (
    "tv stand", "media console", "media unit", "tv unit",
    "media stand", "media center", "entertainment center",
    "entertainment unit",
)
_TV_STAND_ALLOWED_CHILD_TOKENS = (
    "tv", "television", "screen", "monitor", "flat screen",
    "speaker", "soundbar", "remote",
)


def _is_tv_stand(label: str) -> bool:
    lo = label.lower()
    return any(tok in lo for tok in _TV_STAND_PARENT_TOKENS)


def _is_tv_stand_allowed_child(label: str) -> bool:
    """Match allowed TV-stand child tokens with plural tolerance: the
    token must start at a word boundary but can be followed by 's' (or
    not). So 'speaker' matches both 'speaker' and 'speakers'."""
    lo = label.lower()
    return any(re.search(r"\b" + re.escape(tok) + r"s?\b", lo)
                for tok in _TV_STAND_ALLOWED_CHILD_TOKENS)

# Structural-suffix tokens — when these appear AFTER the parent noun
# (e.g., 'wooden sideboard legs', 'wooden coffee table base'), the
# prompt names the parent's own structural piece and shouldn't split
# as a separate child. Matched only after the parent label, not in
# isolation — so 'picture frame' (no parent noun) stays a child.
_STRUCT_SUFFIXES = (
    "legs", "leg", "base", "pedestal", "frame", "skirt", "plinth",
    "caster", "casters", "spindle", "spindles", "stretcher", "stretchers",
)


def slugify(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_")
    return s or "item"


def project_and_count(xyz, V, K, W, H):
    """Project Nx3 world points through camera (V, K), return:
      xi, yi: int pixel coords (Nx int)
      in_img: bool mask, in-front-of-cam AND in image rect
    """
    hp = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    cam_xyz = (hp @ V.T)[:, :3]
    zc = -cam_xyz[:, 2]
    in_front = zc > 0.01
    xs = K[0, 0] * cam_xyz[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]
    ys = K[1, 1] * cam_xyz[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
    xi = xs.astype(np.int32)
    yi = ys.astype(np.int32)
    in_img = in_front & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return xi, yi, in_img


def vote_keep(xyz, masks_info, vote_frac):
    """For each splat in xyz, count how many masks_info masks include it.
    Return bool keep array with ≥ ceil(vote_frac * n_views) votes.
    """
    n_views = len(masks_info)
    if n_views == 0:
        return np.zeros(len(xyz), dtype=bool)
    votes = np.zeros(len(xyz), dtype=np.int32)
    valid = np.zeros(len(xyz), dtype=np.int32)
    for mv in masks_info:
        V, K = mv["V"], mv["K"]
        mask = mv["mask"]
        W, H = mv["W"], mv["H"]
        xi, yi, in_img = project_and_count(xyz, V, K, W, H)
        good = np.where(in_img)[0]
        valid[good] += 1
        vals = mask[yi[good].clip(0, H - 1), xi[good].clip(0, W - 1)]
        votes[good[vals > 0]] += 1
    required = int(math.ceil(vote_frac * n_views))
    return (valid >= required) & (votes >= required)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--source-stage", default="5_sweep_fallback",
                    help="which stage's PLY to source children from")
    ap.add_argument("--vote-frac", type=float, default=VOTE_FRAC)
    ap.add_argument("--min-view-hits", type=int, default=MIN_VIEW_HITS,
                     help="skip a prompt if SAM only hit on fewer than N views")
    ap.add_argument("--sam-pad-hard-m", type=float, default=CHILD_SAM_PAD_HARD_M)
    ap.add_argument("--sam-pad-fabric-m", type=float, default=CHILD_SAM_PAD_FABRIC_M)
    args = ap.parse_args()

    obj = args.obj_dir.resolve()

    src_ply = obj / f"{args.source_stage}.ply"
    if not src_ply.exists():
        # fall back to next stage up
        for alt in ("5_sweep_fallback", "4_sam_tight", "3_floor_drop",
                     "2_sam_wide"):
            cand = obj / f"{alt}.ply"
            if cand.exists():
                src_ply = cand
                print(f"[note] {args.source_stage}.ply missing, "
                      f"falling back to {alt}.ply")
                break
        else:
            sys.exit(f"[fatal] no source PLY in {obj}")

    prompt_path = obj / "diagnostics" / "2_sam_wide" / "sam_prompt.txt"
    if not prompt_path.exists():
        sys.exit(f"[fatal] missing {prompt_path}")
    pipe = prompt_path.read_text().strip()
    tagged = parse_tagged_prompts(pipe)
    if len(tagged) < 2:
        print(f"[skip] only {len(tagged)} prompt term(s) — no children to split")
        return 0

    parent_label = tagged[0][0]
    parent_slug = slugify(parent_label)
    print(f"[parent] '{parent_label}' (slug={parent_slug})")

    # Parent noun(s) for structural detection. Strip qualifiers like
    # 'wooden', 'green', 'tufted' so that 'wooden sideboard legs' is
    # caught as structural via the bare noun 'sideboard'. Take the
    # LAST noun-ish token from the parent label.
    parent_tokens = [t for t in re.split(r"\s+", parent_label.lower()) if t]
    parent_noun = parent_tokens[-1] if parent_tokens else parent_label.lower()

    def is_structural(text: str) -> bool:
        """True if this prompt names the parent's own structural piece —
        i.e., contains the parent's noun followed by a structural suffix
        (e.g., 'sideboard legs', 'coffee table base'). 'picture frame'
        is NOT structural because it doesn't reference the parent."""
        lo = text.lower()
        if parent_noun not in lo:
            return False
        for suf in _STRUCT_SUFFIXES:
            if re.search(r"\b" + re.escape(suf) + r"\b", lo):
                return True
        return False

    # Split tagged into children vs skip (parent body, legs/base/etc).
    parent_is_tv_stand = _is_tv_stand(parent_label)
    children_prompts = []
    skipped_prompts = []
    for i, (text, cls) in enumerate(tagged):
        if i == 0:
            skipped_prompts.append((text, cls, "parent_body"))
            continue
        if is_structural(text):
            skipped_prompts.append((text, cls, "structural"))
            continue
        # TV-stand-class parents: only TV / speakers / soundbar split
        # out. Everything else stays bundled with the cabinet body.
        if parent_is_tv_stand and not _is_tv_stand_allowed_child(text):
            skipped_prompts.append((text, cls, "tv_stand_keep_in_body"))
            continue
        children_prompts.append((text, cls))

    print(f"[children] {len(children_prompts)} sub-items to split:")
    for text, cls in children_prompts:
        print(f"  - '{text}' ({cls})")
    if skipped_prompts:
        print(f"[skipped] {len(skipped_prompts)} prompts (parent / structural):")
        for text, cls, why in skipped_prompts:
            print(f"  - '{text}' ({cls}) — {why}")

    if not children_prompts:
        print("[done] no children to split")
        return 0

    # Load cameras from sam_carve's kept views (already wall-skipped).
    cam_json = obj / "diagnostics" / "2_sam_wide" / "cameras.json"
    if not cam_json.exists():
        sys.exit(f"[fatal] missing {cam_json}")
    cam_data = json.load(open(cam_json))
    cameras = cam_data["cameras"]
    W = int(cam_data["width"])
    H = int(cam_data["height"])
    print(f"[cameras] {len(cameras)} kept views")

    # Load source splats.
    pl = PlyData.read(str(src_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n_total = len(xyz)
    print(f"[source] {src_ply.name}  {n_total:,} splats")

    children_root = obj / "children"
    children_root.mkdir(exist_ok=True)

    # For each child prompt: SAM each view → vote → write child PLY.
    all_child_keep = np.zeros(n_total, dtype=bool)
    written = []
    for idx, (text, cls) in enumerate(children_prompts, start=1):
        slug = slugify(text)
        child_dir = children_root / f"{idx:02d}_{slug}"
        child_diag = child_dir / "diagnostics"
        child_diag.mkdir(parents=True, exist_ok=True)
        pad_m = (args.sam_pad_fabric_m if cls == "soft"
                  else args.sam_pad_hard_m)

        masks_info = []
        per_view_hits = 0
        for cam in cameras:
            tag = cam["tag"]
            img_path = Path(cam["png"])
            if not img_path.exists():
                continue
            V = np.array(cam["V"], dtype=np.float64)
            K = np.array(cam["K"], dtype=np.float64)
            eye = np.array(cam["eye"])
            target = np.array(cam["target"])
            depth = float(np.linalg.norm(eye - target))
            f_px = float(K[0, 0])
            m, scores = sam_segment(img_path, text)
            if not scores or m is None or (m > 0).sum() < MIN_PROMPT_PX:
                continue
            per_view_hits += 1
            m_clean = morph_clean(m)
            rpx = pad_m * f_px / max(depth, 0.1)
            m_dilated = dilate_mask(m_clean, rpx)
            Image.fromarray(m_clean, mode="L").save(
                child_diag / f"mask_{tag}.png")
            Image.fromarray(m_dilated, mode="L").save(
                child_diag / f"mask_padded_{tag}.png")
            masks_info.append({
                "tag": tag, "V": V, "K": K,
                "mask": m_dilated, "W": W, "H": H,
            })
        print(f"  [{slug}] SAM hit {per_view_hits}/{len(cameras)} views")
        if not masks_info:
            print(f"  [{slug}] no usable SAM masks — skipping")
            continue
        if len(masks_info) < args.min_view_hits:
            print(f"  [{slug}] only {len(masks_info)} views with mask "
                  f"(< min {args.min_view_hits}) — likely hallucinated, skipping")
            continue

        keep = vote_keep(xyz, masks_info, args.vote_frac)
        n_kept = int(keep.sum())
        if n_kept == 0:
            print(f"  [{slug}] 0 splats after vote — skipping")
            continue
        print(f"  [{slug}] kept {n_kept:,} splats")
        all_child_keep |= keep

        out_ply = child_dir / "object.ply"
        PlyData([PlyElement.describe(v.data[keep], "vertex")],
                text=False).write(str(out_ply))
        render_canonical_5(out_ply, child_dir / "renders")
        meta = {
            "parent_dir": str(obj),
            "parent_label": parent_label,
            "child_label": text,
            "child_class": cls,
            "source_ply": str(src_ply.relative_to(obj)),
            "n_splats": n_kept,
            "n_views_used": len(masks_info),
            "vote_frac": args.vote_frac,
            "sam_pad_m": pad_m,
        }
        (child_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        written.append({"slug": slug, "label": text, "n_splats": n_kept,
                         "path": str(out_ply.relative_to(obj))})

    # Parent-alone splat selection. Normally: source MINUS all child
    # splats (so cabinet body has holes where the children were carved
    # out). For TV-stand parents the user wants the cabinet to stay
    # whole as one unit — child PLYs sit alongside as duplicates, not
    # subtractions. Same source splats appear in both parent_alone and
    # in each child.
    if parent_is_tv_stand:
        parent_alone = np.ones(n_total, dtype=bool)
        print("[parent_alone] TV-stand parent kept WHOLE (children "
              "duplicate, not subtracted)")
    else:
        parent_alone = ~all_child_keep
    parent_dir = children_root / f"00_{parent_slug}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    out_ply = parent_dir / "object.ply"
    PlyData([PlyElement.describe(v.data[parent_alone], "vertex")],
            text=False).write(str(out_ply))
    render_canonical_5(out_ply, parent_dir / "renders")
    (parent_dir / "meta.json").write_text(json.dumps({
        "parent_dir": str(obj),
        "parent_label": parent_label,
        "role": "parent_alone",
        "source_ply": str(src_ply.relative_to(obj)),
        "n_splats": int(parent_alone.sum()),
        "subtracted_children": [w["slug"] for w in written],
    }, indent=2))

    manifest = {
        "parent_dir": str(obj),
        "parent_label": parent_label,
        "parent_slug": parent_slug,
        "source_stage": args.source_stage,
        "source_ply": str(src_ply.relative_to(obj)),
        "vote_frac": args.vote_frac,
        "n_total_splats": n_total,
        "n_parent_alone_splats": int(parent_alone.sum()),
        "children": written,
    }
    (children_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[done] {len(written)} children written + parent-alone")
    print(f"  manifest: {children_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
