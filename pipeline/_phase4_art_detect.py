#!/usr/bin/env python3
"""_phase4_art_detect.py — find wall-mounted art on the 4 quadrant
dioramas (NE / NW / SE / SW) using Qwen. Same camera renders that
phase 2 detect uses for furniture / appliances.

Wall art = paintings, framed prints, posters, canvases hung on walls.
Mirrors are NOT included — they go through a separate mirror flow
because their content is a reflection rather than a painted surface.

Output: <scene>/_phase4_temp/qwen_art_items.json with per-quadrant
bboxes + overlay PNGs.

Usage:
    python _phase4_art_detect.py <scene_dir>
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


PROMPT = (
    "This image is a side view across one quadrant of an interior room. "
    "The opposite walls have been removed so you can see across the cut.\n\n"
    "Find every distinct piece of WALL ART hanging on the visible walls.\n\n"
    "Wall art includes:\n"
    "  • Paintings (oil, acrylic, watercolor, abstract, figurative)\n"
    "  • Framed prints / framed photographs\n"
    "  • Posters\n"
    "  • Canvases (framed or unframed)\n"
    "  • Wall hangings (tapestries, fabric art)\n\n"
    "DO NOT list:\n"
    "  - Mirrors (handled separately).\n"
    "  - Wall-mounted TVs / monitors (handled in furniture pass).\n"
    "  - Wall-mounted shelves / floating shelves (those are furniture).\n"
    "  - Wall sconces / wall lamps.\n"
    "  - Clocks.\n"
    "  - Windows / doors.\n"
    "  - Splat-render artifacts (blurry stretching near the cut edge, "
    "white capture noise).\n\n"
    "For each art piece, return a tight 2D bounding box around the "
    "frame (or canvas edge if no frame).\n\n"
    "Output valid JSON only, no markdown, no commentary:\n"
    "{\n"
    '    "items": [\n'
    '        {"bbox_2d": [x_min, y_min, x_max, y_max], '
    '"label": "<short natural label, color/style baked in>", '
    '"confidence": "high|medium|low"}\n'
    "    ]\n"
    "}\n\n"
    "Rules:\n"
    "- Coordinates are integers in 0-1000 (normalized, NOT pixels).\n"
    "- Label examples: 'orange yellow abstract painting', 'black and "
    "white photograph', 'blue circle painting', 'framed print', "
    "'red abstract canvas'. NO size words. NO shape words.\n"
    "- Each visible art piece is its own entry.\n"
    "- If 0 art pieces, return {\"items\": []}."
)


def encode_b64(p: Path, max_dim: int = 1280) -> str:
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
        max_tokens=1500, temperature=0.1,
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
        except json.JSONDecodeError:
            pass
    items = []
    pat = re.compile(
        r'(?:"bbox_2d"|"bbox")\s*:\s*\[\s*'
        r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\][^"]*'
        r'"label"\s*:\s*"([^"]+)"',
        re.DOTALL,
    )
    for m in pat.finditer(raw):
        bbox = [int(m.group(i)) for i in range(1, 5)]
        items.append({"bbox_2d": bbox, "label": m.group(5),
                       "confidence": "unknown"})
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
    "NE": (255, 200, 0),    # gold
    "NW": (255, 0, 200),    # magenta
    "SE": (0, 255, 200),    # cyan
    "SW": (200, 255, 0),    # yellow-green
}

MIN_AREA_NORM = 1000        # smaller threshold than furniture phase 2 —
                            # wall art can legitimately be smaller in the
                            # diorama, especially on far walls.


def filter_items(items: list) -> list:
    kept = []
    for it in items:
        b = it.get("bbox_2d")
        if not b or len(b) != 4:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < MIN_AREA_NORM:
            it["_dropped"] = f"too_small({area})"
            continue
        kept.append(it)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    pdir = scene / "_phase2_temp"
    if not pdir.exists():
        sys.exit(f"[fatal] missing {pdir} — phase 2 dioramas must run first")
    out_dir = scene / "_phase4_temp"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        (out_dir / f"qwen_{q}_raw.txt").write_text(raw)
        raw_items = parse_items(raw)
        for it in raw_items:
            b = it.get("bbox_2d")
            if b and len(b) == 4:
                it["bbox_pixels"] = scale_bbox_to_pixels(b, img_w, img_h)
            it["quadrant"] = q
        items = filter_items(raw_items)
        dropped = [x for x in raw_items if x.get("_dropped")]
        print(f"  raw {len(raw_items)}  kept {len(items)}  dropped {len(dropped)}")
        for d in dropped:
            print(f"    [drop {d['_dropped']:20s}] {d.get('label')}")
        for it in items:
            print(f"    - {it.get('label')}  conf={it.get('confidence', '?')}")
        overlay_path = out_dir / f"qwen_{q}_overlay.png"
        draw_overlay(img_path, items, overlay_path, QUAD_COLORS[q])
        print(f"  overlay: {overlay_path}")
        all_by_quad[q] = items

    flat = []
    for q, items in all_by_quad.items():
        flat.extend(items)
    out_json = out_dir / "qwen_art_items.json"
    out_json.write_text(json.dumps({
        "items": flat,
        "by_quadrant": all_by_quad,
    }, indent=2))
    print(f"\n[total] {len(flat)} wall-art pieces across {len(all_by_quad)} quadrants")
    print(f"  json: {out_json}")


if __name__ == "__main__":
    main()
