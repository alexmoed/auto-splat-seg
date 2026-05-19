#!/usr/bin/env python3
"""Phase 2 Qwen detection on the 4 quadrant diorama renders.

For each quadrant render in <scene>/_phase2_temp/quad_<NE|NW|SE|SW>.png,
ask Qwen for objects in the room (furniture / props / appliances).

Exclusions (user-locked 2026-05-05):
  - film / scan equipment, tripods, draped sheets, light stands, cables
  - artwork, paintings, posters, framed prints
  - architectural: walls, doors, windows, radiators, vents, ceilings, floors
  - hanging lamps / pendants / chandeliers / sconces

Outputs per quadrant:
  <scene>/_phase2_temp/qwen_<quad>_raw.txt
  <scene>/_phase2_temp/qwen_<quad>_overlay.png
And combined:
  <scene>/_phase2_temp/qwen_phase2_items.json

Usage:
    python _phase2_detect.py <scene_dir>
"""
import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

QUADS = ["NE", "NW", "SE", "SW"]


def load_room_extras(scene_dir: Path) -> tuple:
    """Read room_type.json (set by detect_room_type.py / run_all.py) and
    return (extra_keep, extra_skip) lines to inject into the prompt.
    Falls back to mixed config if file missing or import fails."""
    try:
        from room_config import get_room_config   # noqa: WPS433
    except ImportError:
        return [], []
    rt = "mixed"
    rt_file = scene_dir / "_inventory_temp" / "room_type.json"
    if rt_file.exists():
        try:
            rt = json.loads(rt_file.read_text()).get("room_type", "mixed")
        except Exception:
            pass
    cfg = get_room_config(rt)
    return (cfg.get("phase2_keep_extras", []),
            cfg.get("phase2_skip_extras", []))


def build_prompt(extra_keep: list, extra_skip: list) -> str:
    keep_lines = "".join(f"  • {x}\n" for x in extra_keep)
    skip_lines = "".join(f"  - {x}\n" for x in extra_skip)
    # Plain string replace — str.format would barf on the JSON braces
    # in the template body.
    out = PROMPT_TEMPLATE
    out = out.replace("{extra_keep}", keep_lines)
    out = out.replace("{extra_skip}", skip_lines)
    return out


PROMPT_TEMPLATE = (
    "This image is a side view across one quadrant of an interior room. "
    "The opposite walls have been removed so you can see into this "
    "quadrant from across the cut.\n\n"
    "Inventory has ALREADY identified large floor furniture (sofas, "
    "armchairs, beds, dressers, dining tables, desks, cabinets, "
    "wardrobes, ottomans, dining chairs — whichever apply to this room "
    "type). Those splats have been REMOVED from this view, so you may "
    "see blurry / smeared / hollow regions where they used to be — "
    "IGNORE those regions entirely.\n\n"
    "Your job is to find every distinct OBJECT inventory missed. "
    "Look for:\n"
    "  (Bookshelves, bookcases, and open shelving were detected in the "
    "phase 1 topdown inventory and have been subtracted from this view "
    "— do NOT list them again, and do NOT list books / vases / picture "
    "frames / baskets / decor on their shelves.)\n"
    "  • Floor-standing lamps (tall floor lamps with their own base on "
    "the floor — NOT table lamps sitting on furniture).\n"
    "  • Floor-standing potted plants — list EACH distinct plant "
    "separately (a tall plant in the corner and a smaller plant by the "
    "window are TWO entries, not one).\n"
    "  • Pet beds / dog beds / cat beds resting on the floor.\n"
    "  • TVs / monitors / mirrors mounted on walls or sitting on "
    "furniture.\n"
    "{extra_keep}\n"
    "For each, return a 2D bounding box.\n\n"
    "WHAT MAKES AN OBJECT LISTABLE:\n"
    "  • Has a clear physical footprint that occupies at least its own "
    "small region of the image.\n"
    "  • Is a self-contained item, not part of another item's surface.\n"
    "  • Would make sense to extract as a standalone 3D object (e.g. "
    "you could lift it up and move it to another room).\n\n"
    "WHAT MAKES IT NOT LISTABLE:\n"
    "  • It's small clutter sitting on top of something else — books on "
    "a shelf, jars on a counter, vases on a cabinet, cups on a table. "
    "These are part of the parent furniture, not separate objects.\n"
    "  • It's part of the building (walls, doors, windows, ceiling, "
    "floor, baseboards, moldings, radiators, vents, beams, fireplace).\n"
    "  • It's photography or scan equipment in the room (tripod with a "
    "camera, light stand, draped white sheet over equipment, c-stand, "
    "softbox, bicycle, motorcycle, cardboard box, bag, cable, wire). "
    "If you see a tall thin object in the middle of the room with a "
    "draped sheet on it, that is a TRIPOD — not a TV, not a lamp.\n"
    "  • It's a wall-hung painting, poster, framed print, or canvas — "
    "those are extracted by a separate wall-art pass.\n"
    "  • It's a hanging fixture (pendant light, chandelier, ceiling "
    "fan, ceiling lamp, hanging speaker).\n"
    "{extra_skip}"
    "  • It's a dried/dead plant arrangement that looks like splat "
    "noise rather than a real living plant.\n"
    "  • It's a splat-render artifact (blurry scatter, dark fringe "
    "noise, white rectangular capture artifacts, ghostly stretching "
    "near the diorama cut edge).\n\n"
    "WHEN UNSURE, USE SIZE: if the thing you're considering is smaller "
    "than roughly 1/20 of the image's smaller dimension, it's almost "
    "certainly clutter — skip it.\n\n"
    "Output valid JSON only, no markdown, no commentary:\n"
    "{\n"
    '    "items": [\n'
    '        {"bbox_2d": [x_min, y_min, x_max, y_max], '
    '"label": "<short natural label, color baked in>", '
    '"confidence": "high|medium|low"}\n'
    "    ]\n"
    "}\n\n"
    "Rules:\n"
    "- Coordinates are integers in 0-1000 (normalized, NOT pixels).\n"
    "- Label uses natural words and BAKES THE COLOR IN "
    "(e.g. 'beige armchair', 'wooden side table', 'black flat screen tv').\n"
    "- BANNED in labels: NO size words (small, big, large, tall, short, low, "
    "high, mini, huge, tiny, oversized, narrow, wide, slim) and NO shape "
    "words (rectangular, square, round, oval, l-shaped, u-shaped, curved). "
    "Just say 'armchair' not 'small armchair', 'sofa' not 'l-shaped "
    "sectional sofa', 'coffee table' not 'rectangular coffee table'. "
    "Color and material ARE allowed.\n"
    "- Each visible instance is its own entry.\n"
    "- confidence ∈ {\"high\", \"medium\", \"low\"}.\n"
    "- If 0 objects, return {\"items\": []}.\n\n"
    "IGNORE: cabinets, kitchen cabinets, kitchen built-ins, "
    "black boxes, cardboard boxes, equipment cases, bags, fabric bags, "
    "tote bags, recorder bags, cables, wires, power cords, "
    "dried plants, dried plant arrangements, dead plants, "
    "hanging plants, hanging fixtures, pendant lamps, chandeliers, "
    "sconces, wall lamps, paintings, framed prints, posters, "
    "tripods, cameras on stands, light stands, draped sheets."
)


def encode_b64(p: Path, max_dim: int = 1280) -> str:
    """Downscale large images so they fit in vLLM's 4096 context.
    Bbox returned by Qwen is in 0-1000 normalized space — downscaling
    doesn't affect bbox precision when scaled back to image_size."""
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        new_size = (int(img.size[0] * s), int(img.size[1] * s))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_qwen(img_path: Path, prompt: str) -> str:
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{encode_b64(img_path)}"}},
        {"type": "text", "text": prompt},
    ]
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=2000, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content.strip()


def parse_items(raw: str) -> list:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    if s != -1 and e > s:
        try:
            data = json.loads(cleaned[s:e])
            return data.get("items", [])
        except json.JSONDecodeError as ex:
            print(f"  [parse] JSON error ({ex}) — falling back to regex")
    items = []
    pat = re.compile(
        r'(?:"bbox_2d"|"bbox_pixels"|"bbox")\s*:\s*\[\s*'
        r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\][^"]*'
        r'"label"\s*:\s*"([^"]+)"',
        re.DOTALL,
    )
    for m in pat.finditer(cleaned):
        bbox = [int(m.group(i)) for i in (1, 2, 3, 4)]
        label = m.group(5)
        items.append({"bbox_2d": bbox, "label": label, "confidence": "unknown"})
    return items


def scale_bbox_to_pixels(bbox_2d: list, img_w: int, img_h: int) -> list:
    x0, y0, x1, y1 = bbox_2d
    return [int(x0 * img_w / 1000),
            int(y0 * img_h / 1000),
            int(x1 * img_w / 1000),
            int(y1 * img_h / 1000)]


def draw_overlay(img_path: Path, items: list, out: Path, color):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    for i, it in enumerate(items):
        bbox = it.get("bbox_pixels")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(c) for c in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
        label = f"{i+1}. {it.get('label', '?')}"
        tx = max(2, x0)
        ty = max(2, y0 - 32)
        draw.rectangle([tx, ty, tx + 14 * len(label), ty + 30], fill=color)
        draw.text((tx + 4, ty + 1), label, fill=(255, 255, 255), font=font)
    img.save(out)


QUAD_COLORS = {
    "NE": (60, 120, 255),
    "NW": (60, 200, 100),
    "SE": (255, 100, 60),
    "SW": (200, 60, 200),
}

MIN_AREA_NORM = 2500      # 0-1000² normalized → 50×50 px in normalized
                          # space. ~0.25% of image. Tiny clutter (jars,
                          # cups, books on shelf) is filtered by size.
                          # Real countertop appliances and meaningful
                          # objects sit comfortably above this threshold.
                          # Tune up/down if too aggressive/permissive.


# Collapse synonyms so dedup catches them. ORDER MATTERS — multi-word
# synonyms must come BEFORE shorter overlapping ones (e.g. 'range hood'
# before 'range', so 'silver range hood' doesn't get rewritten to
# 'silver stove' via the 'range' alias for stove).
SYNONYMS = {
    "range hood": ["range hood", "hood vent", "extractor hood"],
    "bookshelf": ["bookshelf", "bookcase", "open shelving unit",
                   "open shelving", "open shelf", "open bookshelf",
                   "open bookcase", "tall storage shelves",
                   "tall storage shelf", "tall shelves", "tall shelf",
                   "tall shelving", "ladder shelves", "ladder shelf",
                   "display shelves", "display shelf", "shelving units",
                   "shelving unit", "wooden shelving", "metal shelving",
                   "wooden shelves", "etagere", "étagère"],
    "floor lamp": ["floor lamp", "tall lamp", "standing lamp"],
    "potted plant": ["potted plant", "house plant", "houseplant",
                      "indoor plant", "plant in pot", "tall plant",
                      "large plant"],
    "pet bed": ["pet bed", "dog bed", "cat bed", "animal bed"],
    "tv": ["flat screen tv", "flat-screen tv", "flat screen television",
            "flat-screen television", "television", "tv"],
    "mirror": ["wall mirror", "mirror"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "stove": ["kitchen stove", "stove", "oven", "cooktop", "range"],
}


def canonical_label(label: str) -> str:
    """Collapse a Qwen label to a canonical class name. Word-boundary
    match so 'range' in 'range hood' doesn't trigger the 'stove' alias
    (because 'range hood' is matched FIRST in dict order). Preserves
    color/material modifiers."""
    lo = (label or "").lower().strip()
    for canon, syns in SYNONYMS.items():
        for s in syns:
            # Word-boundary regex
            m = re.search(r"\b" + re.escape(s) + r"\b", lo)
            if m:
                prefix = lo[:m.start()].strip()
                if prefix:
                    return f"{prefix} {canon}".strip()
                return canon
    return lo


DEDUP_IOU = 0.9   # only drop a duplicate if bbox overlap >= 90% AND label matches.
                  # Two real "potted plants" at different spots → IoU 0 → both kept.
                  # Same plant Qwen returned twice → IoU ~1 → drop second.


def bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    b_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def filter_items(items: list) -> list:
    """Two filters only: (1) bbox area must clear MIN_AREA_NORM (drops
    tiny countertop clutter, jars, cups, books). (2) within a single
    quadrant, drop items with same canonical_label and >=90% IoU
    (definitely a Qwen self-duplicate, not two real objects)."""
    kept = []
    for it in items:
        b = it.get("bbox_2d")
        if not b or len(b) != 4:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < MIN_AREA_NORM:
            it["_dropped"] = f"too_small({area})"
            continue
        canon = canonical_label(it.get("label", ""))
        it["canonical_label"] = canon
        is_dup = False
        for k in kept:
            if k.get("canonical_label") != canon:
                continue
            if bbox_iou(b, k["bbox_2d"]) >= DEDUP_IOU:
                is_dup = True
                break
        if is_dup:
            it["_dropped"] = "dup_same_label_90pct"
            continue
        kept.append(it)
    return kept


# Items that are GUARANTEED singular per scene — only one fridge, one
# range hood, etc. For these we can safely dedup cross-quadrant.
# Everything else (plants, lamps, bookshelves, tvs) can have multiple
# instances and must NOT be cross-quadrant deduped — we'll dedup at
# extraction time via 3D bbox overlap of the produced PLYs.
SINGULAR_CANONICAL_LABELS = {
    "refrigerator", "range hood", "stove", "kitchen stove",
}


def cross_quadrant_dedup(by_quad: dict) -> list:
    """Only dedup classes where the room can have at most one instance
    (fridge, stove, range hood). Multi-instance classes (bookshelves,
    plants, lamps, TVs, mirrors) are kept as-is — same physical object
    in 2 dioramas will produce 2 slightly-different PLYs that we can
    merge later via 3D bbox overlap.
    """
    flat = []
    for q, items in by_quad.items():
        for it in items:
            it = dict(it)
            it["_quad"] = q
            it["_area"] = ((it["bbox_2d"][2] - it["bbox_2d"][0]) *
                            (it["bbox_2d"][3] - it["bbox_2d"][1]))
            flat.append(it)

    from collections import defaultdict
    singular_groups = defaultdict(list)
    multi_items = []
    for it in flat:
        canon = it.get("canonical_label", "?")
        # Strip color/material prefix to test against SINGULAR set
        # (so "silver range hood" → "range hood")
        canon_core = canon.split()[-2:] if len(canon.split()) >= 2 else canon.split()
        canon_core = " ".join(canon_core).strip()
        is_singular = (canon in SINGULAR_CANONICAL_LABELS or
                       canon_core in SINGULAR_CANONICAL_LABELS or
                       canon.split()[-1] in SINGULAR_CANONICAL_LABELS)
        if is_singular:
            singular_groups[canon].append(it)
        else:
            multi_items.append(it)

    deduped = list(multi_items)
    for canon, group in singular_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        group.sort(key=lambda x: -x["_area"])
        kept = group[0]
        kept["_dup_dropped"] = [
            f"{g['_quad']}({g['_area']})" for g in group[1:]
        ]
        deduped.append(kept)
        for g in group[1:]:
            g["_dropped_cross_quadrant"] = (
                f"singular_smaller_than_{kept['_quad']}({kept['_area']})"
            )

    return deduped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    pdir = scene / "_phase2_temp"
    if not pdir.exists():
        sys.exit(f"[fatal] missing {pdir} — run _phase2_dioramas.py first")

    extra_keep, extra_skip = load_room_extras(scene)
    PROMPT = build_prompt(extra_keep, extra_skip)
    print(f"[room-extras] keep+={len(extra_keep)}  skip+={len(extra_skip)}")
    (pdir / "phase2_prompt_used.txt").write_text(PROMPT)

    all_by_quad = {}
    for q in QUADS:
        img_path = pdir / f"quad_{q}.png"
        if not img_path.exists():
            print(f"[warn] missing {img_path}, skipping")
            continue
        img = Image.open(img_path)
        img_w, img_h = img.size
        print(f"\n[{q}] {img_path.name}  {img_w}×{img_h}")
        raw = call_qwen(img_path, PROMPT)
        (pdir / f"qwen_{q}_raw.txt").write_text(raw)
        raw_items = parse_items(raw)
        for it in raw_items:
            b = it.get("bbox_2d")
            if b and len(b) == 4:
                it["bbox_pixels"] = scale_bbox_to_pixels(b, img_w, img_h)
            it["quadrant"] = q
        items = filter_items(raw_items)
        dropped = [x for x in raw_items if x.get("_dropped")]
        print(f"  raw {len(raw_items)}  kept {len(items)}  dropped {len(dropped)}")
        if dropped:
            for d in dropped:
                print(f"    [drop {d['_dropped']:20s}] {d.get('label')}")
        for it in items:
            print(f"    - {it.get('label')}  conf={it.get('confidence', '?')}")
        overlay_path = pdir / f"qwen_{q}_overlay.png"
        draw_overlay(img_path, items, overlay_path, QUAD_COLORS[q])
        print(f"  overlay: {overlay_path}")
        all_by_quad[q] = items

    raw_flat = []
    for q, items in all_by_quad.items():
        raw_flat.extend(items)

    # Cross-quadrant dedup — drop the same physical object if it
    # got listed from multiple dioramas (keeps the largest area instance).
    deduped = cross_quadrant_dedup(all_by_quad)
    n_dropped_cross = len(raw_flat) - len(deduped)
    if n_dropped_cross > 0:
        print(f"\n[cross-quad-dedup] dropped {n_dropped_cross} duplicate "
              f"detections across quadrants")

    out_json = pdir / "qwen_phase2_items.json"
    out_json.write_text(json.dumps({
        "items": deduped,
        "by_quadrant": all_by_quad,
        "cross_quadrant_dropped": n_dropped_cross,
    }, indent=2))
    print(f"\n[total] {len(deduped)} items (raw {len(raw_flat)}) "
          f"across {len(all_by_quad)} quadrants")
    print(f"  json: {out_json}")


if __name__ == "__main__":
    main()
