#!/usr/bin/env python3
"""Build labeled side-by-side showcase images from the wipe-pair renders.

Reads docs/showcase/wipe/cameras.json and produces:
  - docs/showcase/wipe/<slug>_pair.png   side-by-side per object (background | object)
  - docs/showcase/wipe/_grid.png         2x2 grid of all 4 pairs
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DISPLAY_LABEL = {
    "grey_armchair": "Armchair",
    "beige_sectional_sofa": "Sectional sofa",
    "wooden_coffee_table": "Coffee table",
    "wooden_sideboard": "Sideboard",
}


def build_pair(out_dir: Path, slug: str, label: str):
    bg = out_dir / f"{slug}_background.png"
    obj = out_dir / f"{slug}_object.png"
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.625), dpi=120)
    axes[0].imshow(mpimg.imread(str(bg)))
    axes[0].set_title("Original scan", fontsize=18, pad=12)
    axes[0].axis("off")
    axes[1].imshow(mpimg.imread(str(obj)))
    axes[1].set_title("Extracted .splat", fontsize=18, pad=12)
    axes[1].axis("off")
    fig.suptitle(label, fontsize=24, y=1.02, weight="bold")
    fig.tight_layout()
    out_png = out_dir / f"{slug}_pair.png"
    fig.savefig(str(out_png), bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    print(f"  → {out_png}")


def build_grid(out_dir: Path, slugs_labels):
    n = len(slugs_labels)
    fig, axes = plt.subplots(n, 2,
                             figsize=(20, 5.625 * n),
                             dpi=100,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    for row, (slug, label) in enumerate(slugs_labels):
        bg = mpimg.imread(str(out_dir / f"{slug}_background.png"))
        obj = mpimg.imread(str(out_dir / f"{slug}_object.png"))
        axes[row, 0].imshow(bg)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(obj)
        axes[row, 1].axis("off")
        # Row label as text in upper-left of the bg panel
        axes[row, 0].text(0.02, 0.96, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=26, weight="bold",
                          color="white", va="top", ha="left",
                          bbox=dict(facecolor="black", alpha=0.65,
                                    boxstyle="round,pad=0.4",
                                    edgecolor="none"))
        if row == 0:
            axes[row, 0].set_title("Original scan",
                                    fontsize=22, pad=12, weight="bold")
            axes[row, 1].set_title("Extracted .splat",
                                    fontsize=22, pad=12, weight="bold")
    fig.tight_layout()
    out_png = out_dir / "_grid.png"
    fig.savefig(str(out_png), bbox_inches="tight", pad_inches=0.2,
                facecolor="white")
    plt.close(fig)
    print(f"  → {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/showcase/wipe"))
    args = ap.parse_args()

    cameras = json.loads((args.out_dir / "cameras.json").read_text())
    slugs = list(cameras["objects"].keys())

    print("[per-object pairs]")
    for slug in slugs:
        label = DISPLAY_LABEL.get(slug, slug.replace("_", " ").title())
        build_pair(args.out_dir, slug, label)

    print("[grid]")
    slugs_labels = [(s, DISPLAY_LABEL.get(s, s.replace("_", " ").title()))
                    for s in slugs]
    build_grid(args.out_dir, slugs_labels)


if __name__ == "__main__":
    main()
