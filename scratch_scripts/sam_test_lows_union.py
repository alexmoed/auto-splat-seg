#!/usr/bin/env python3
"""sam_test_lows_union.py — SAM3 each prompt term in the pipe-union
separately on every LOW preview render, OR-combine the masks, show per-
term hit + combined mask. Mirrors what sam_low_refine does in production.

Usage:
    python sam_test_lows_union.py <preview_dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/pipeline")
from sam_carve import sam_segment  # noqa: E402

PROMPTS = ["grey armchair", "striped pillow",
           "brown throw blanket", "wooden chair legs"]


def overlay(img_rgb, mask, alpha=0.5):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preview_dir", type=Path)
    args = ap.parse_args()

    preview = args.preview_dir.resolve()
    out_dir = preview / "sam_test_union"
    out_dir.mkdir(exist_ok=True)
    imgs = sorted(p for p in preview.glob("*.png") if "sam_test" not in p.parts)
    print(f"[load] {len(imgs)} preview images, prompts={PROMPTS}")
    print()

    per_view_summary = []
    for ip in imgs:
        combined = None
        per_prompt_hits = []
        for pr in PROMPTS:
            mask, scores = sam_segment(str(ip), pr)
            n_px = int((mask > 0).sum())
            mx = max(scores) if scores else 0.0
            per_prompt_hits.append((pr, n_px, mx))
            if combined is None:
                combined = mask.copy()
            else:
                combined = np.maximum(combined, mask)
        usable_terms = sum(1 for _, n, _ in per_prompt_hits if n > 500)
        n_combined = int((combined > 0).sum()) if combined is not None else 0
        flag = "OK" if n_combined > 1000 else "MISS"
        print(f"[{flag}] {ip.name:25s}  union={n_combined:>8,}  "
              f"({usable_terms}/{len(PROMPTS)} terms)")
        for pr, n_px, mx in per_prompt_hits:
            mark = "+" if n_px > 500 else " "
            print(f"        {mark} {pr:25s}  px={n_px:>7,}  max={mx:.2f}")
        per_view_summary.append((ip.name, n_combined, usable_terms))
        # save overlay
        img = np.array(Image.open(ip).convert("RGB"))
        ov = overlay(img, combined)
        Image.fromarray(ov).save(out_dir / f"overlay_{ip.name}")
        Image.fromarray(combined).save(out_dir / f"mask_{ip.name}")
        print()
    print("[summary]")
    for n, c, t in per_view_summary:
        flag = "OK" if c > 1000 else "MISS"
        print(f"  [{flag}] {n:25s}  union_px={c:>8,}  terms={t}/4")


if __name__ == "__main__":
    main()
