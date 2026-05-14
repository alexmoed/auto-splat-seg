#!/usr/bin/env python3
"""companion_search.py — find supporting items near an extracted parent
object (TV → soundbar/remote/set-top box, etc.) and extract each as a
separate child object.

Inputs:
  <scene>/02_<parent>/   already-extracted parent (e.g. 02_black_flat_screen_tv/)
  <scene>/_phase2_temp/cameras.json  (diorama cameras)
  <scene>/step7_cardinal_aligned.ply (rotated raw scene)

Outputs:
  <scene>/02_<companion_slug>/1_visual_hull.ply  (per companion item)
  <scene>/02_<parent>/companions.json            (list of companions found)
  Renders + meta per companion

Flow:
  1. Find which diorama view contains the parent (its meta has the
     source_quadrant from phase 3) OR use all 4 dioramas.
  2. Render the parent's face-on view (canonical face_on.png).
  3. Ask Qwen to enumerate supporting items in the same scene with bboxes.
  4. For each item: bbox cone hull from the SAME diorama camera,
     6% pad, same as phase 3 extraction.

Usage:
    python companion_search.py <scene_dir> <parent_dir>
"""
import os
import argparse
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels, slugify
from sam_carve import render_canonical_5

QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-awq")
HULL_PAD_PCT = 0.06


# Class-specific companion prompts. Auto-detected from parent label
# (matches any keyword → uses that preset). Falls back to generic.
COMPANION_PRESETS = {
    "tv": {
        "keywords": ["tv", "monitor", "television", "screen"],
        "items": ("Soundbars / speakers / center channels, Remote controls, "
                   "Set-top boxes / cable boxes / streaming devices / "
                   "gaming consoles, Routers / modems, Visible cables / "
                   "power cords (only if clearly bundled), Decor sitting "
                   "on the same surface (small lamps, plants, figurines, books)"),
    },
    "bookshelf": {
        "keywords": ["bookshelf", "bookcase", "shelving", "shelf",
                      "etagere", "étagère"],
        "items": ("Books (groups of books on the same shelf count as ONE "
                   "item per shelf, not per book), Vases, Picture frames, "
                   "Baskets, Small potted plants, Sculptures / figurines, "
                   "Decor objects, Small boxes / storage bins, "
                   "Candles, Bowls, Small lamps sitting on shelves"),
    },
    "sideboard": {
        "keywords": ["sideboard", "credenza", "cabinet", "buffet",
                      "console", "media"],
        "items": ("Decor objects (vases, candles, sculptures, picture "
                   "frames), Small lamps, Plants, Books, Trays, "
                   "Bowls / decorative dishes"),
    },
    "table": {
        "keywords": ["coffee table", "side table", "end table",
                      "console table", "accent table"],
        "items": ("Decor objects (books, candles, vases, sculptures), "
                   "Coasters / trays, Plants, Small lamps, Remote controls, "
                   "Magazines"),
    },
    "desk": {
        "keywords": ["desk", "workspace"],
        "items": ("Monitor / laptop, Keyboard, Mouse, Lamp, Books / "
                   "notebooks, Pen holder / stationery, Plants, "
                   "Speakers / headphones, Cables"),
    },
}

GENERIC_ITEMS = (
    "Any accessory / supporting / decor objects sitting on, next to, or "
    "associated with the parent object")


def pick_preset(parent_label: str) -> str:
    lo = parent_label.lower()
    for name, preset in COMPANION_PRESETS.items():
        for kw in preset["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", lo):
                return preset["items"], name
    return GENERIC_ITEMS, "generic"


def encode_b64(p: Path, max_dim: int = 1280) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        new_size = (int(img.size[0] * s), int(img.size[1] * s))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def find_companions(parent_label: str, diorama_path: Path) -> list:
    """Ask Qwen for supporting items near the parent object in this view."""
    img = Image.open(diorama_path)
    img_w, img_h = img.size
    items_list, preset_name = pick_preset(parent_label)
    prompt = (
        f"This image shows a {parent_label} in its room context. "
        f"Find every SUPPORTING / ACCESSORY object that sits near, on, "
        f"or with the {parent_label}. Look for:\n"
        f"  {items_list}\n\n"
        f"DO NOT list the {parent_label} itself. DO NOT list the surface "
        f"(table/cabinet/wall/floor) the {parent_label} sits on.\n\n"
        f"Output JSON only:\n"
        f"{{\n"
        f'  "items": [\n'
        f'    {{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short label>"}}\n'
        f"  ]\n"
        f"}}\n\n"
        f"Coordinates in 0-1000 normalized. If no supporting items, "
        f'return {{"items": []}}.'
    )
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{encode_b64(diorama_path)}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=800, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    if s == -1 or e <= s:
        return [], raw
    try:
        data = json.loads(cleaned[s:e])
        items = data.get("items", [])
        for it in items:
            b = it.get("bbox_2d")
            if b and len(b) == 4:
                it["bbox_pixels"] = [int(b[0] * img_w / 1000),
                                       int(b[1] * img_h / 1000),
                                       int(b[2] * img_w / 1000),
                                       int(b[3] * img_h / 1000)]
        return items, raw
    except Exception:
        return [], raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("parent_dir", type=Path,
                    help="path to the already-extracted parent (02_<slug>/)")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    parent = args.parent_dir.resolve()
    parent_name = parent.name

    # Load parent meta to find source quadrant + label
    meta_path = parent / "1_visual_hull_meta.json"
    if not meta_path.exists():
        sys.exit(f"[fatal] missing {meta_path}")
    pmeta = json.load(open(meta_path))
    parent_label = pmeta.get("label", parent_name)
    quadrant = pmeta.get("quadrant", "SW")
    print(f"[parent] {parent_name}  label='{parent_label}'  quadrant={quadrant}")

    # Use the parent's source diorama as the search image
    diorama_path = scene / "_phase2_temp" / f"quad_{quadrant}.png"
    if not diorama_path.exists():
        sys.exit(f"[fatal] missing {diorama_path}")

    # Ask Qwen for companions
    print(f"[search] querying Qwen with {diorama_path.name}")
    items, raw = find_companions(parent_label, diorama_path)
    diag_dir = parent / "diagnostics" / "companion_search"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "qwen_raw.txt").write_text(raw)
    print(f"[search] found {len(items)} companion items")
    for i, it in enumerate(items):
        print(f"  [{i}] {it.get('label')}  bbox={it.get('bbox_pixels')}")

    if not items:
        (parent / "companions.json").write_text(json.dumps(
            {"parent": parent_name, "companions": []}, indent=2))
        print("[done] no companions")
        return

    # Load source PLY + diorama camera
    src_ply = scene / "step7_cardinal_aligned.ply"
    cams = json.load(open(scene / "_phase2_temp" / "cameras.json"))
    cam = cams[quadrant]

    pl = PlyData.read(str(src_ply))
    vdata = pl["vertex"]
    xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
    V = viewmat_look_at(cam["eye"], cam["target"], cam["up"])
    K = build_K(cam["fov"], cam["width"], cam["height"])

    companions = []
    for i, it in enumerate(items):
        bbox = it.get("bbox_pixels")
        if not bbox:
            continue
        bw = bbox[2] - bbox[0]; bh = bbox[3] - bbox[1]
        px = bw * HULL_PAD_PCT
        py = bh * HULL_PAD_PCT
        padded = [
            max(0, int(bbox[0] - px)),
            max(0, int(bbox[1] - py)),
            min(cam["width"], int(bbox[2] + px)),
            min(cam["height"], int(bbox[3] + py)),
        ]
        u, v_img, in_front = project_to_pixels(xyz, V, K)
        inside = ((u >= padded[0]) & (u <= padded[2]) &
                  (v_img >= padded[1]) & (v_img <= padded[3]))
        keep = in_front & inside
        n_kept = int(keep.sum())
        if n_kept < 100:
            print(f"  [{i}] '{it['label']}' SKIP — only {n_kept} splats")
            continue

        slug = slugify(it["label"])
        out_dir = scene / f"02_{slug}"
        n = 2
        while out_dir.exists():
            out_dir = scene / f"02_{slug}_{n}"
            n += 1
        out_dir.mkdir(parents=True)
        out_ply = out_dir / "1_visual_hull.ply"
        PlyData([PlyElement.describe(vdata.data[keep], "vertex")],
                text=False).write(str(out_ply))
        render_canonical_5(out_ply, out_dir / "renders" / "1_visual_hull")
        (out_dir / "1_visual_hull_meta.json").write_text(json.dumps({
            "phase": 5,
            "kind": "companion",
            "parent": parent_name,
            "label": it["label"],
            "source_quadrant": quadrant,
            "bbox_pixels_tight": bbox,
            "bbox_pixels_padded": padded,
            "hull_pad_pct_per_side": HULL_PAD_PCT,
            "n_splats_kept": n_kept,
        }, indent=2))
        companions.append({"slug": out_dir.name, "label": it["label"],
                            "n_splats": n_kept})
        print(f"  [{i}] '{it['label']}' → {out_dir.name}  ({n_kept:,} splats)")

    (parent / "companions.json").write_text(json.dumps({
        "parent": parent_name,
        "companions": companions,
    }, indent=2))
    print(f"\n[done] {len(companions)} companions extracted")


if __name__ == "__main__":
    main()
