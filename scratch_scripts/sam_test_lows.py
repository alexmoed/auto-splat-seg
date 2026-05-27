#!/usr/bin/env python3
"""sam_test_lows.py — run SAM3 on the LOW ring preview renders, save
mask overlays, report hit rate + pixel count per view.

Usage:
    python sam_test_lows.py <preview_dir> "<prompt>"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/pipeline")
from sam_carve import sam_segment  # noqa: E402


def overlay(img_rgb, mask, alpha=0.5):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preview_dir", type=Path)
    ap.add_argument("prompt", type=str)
    args = ap.parse_args()

    preview = args.preview_dir.resolve()
    out_dir = preview / "sam_test"
    out_dir.mkdir(exist_ok=True)
    imgs = sorted(p for p in preview.glob("*.png") if "sam_test" not in p.parts)
    print(f"[load] {len(imgs)} preview images, prompt='{args.prompt}'")
    hits = 0
    for ip in imgs:
        mask, scores = sam_segment(str(ip), args.prompt)
        n_px = int((mask > 0).sum())
        score_str = (f"max={max(scores):.2f}" if scores else "no_scores")
        usable = n_px > 1000
        flag = "OK" if usable else "MISS"
        print(f"  [{flag}] {ip.name:25s}  mask_px={n_px:>8,}  {score_str}")
        if usable:
            hits += 1
        # Save mask overlay
        img = np.array(Image.open(ip).convert("RGB"))
        ov = overlay(img, mask)
        Image.fromarray(ov).save(out_dir / f"overlay_{ip.name}")
        Image.fromarray(mask).save(out_dir / f"mask_{ip.name}")
    print(f"\n[summary] {hits}/{len(imgs)} usable")


if __name__ == "__main__":
    main()
