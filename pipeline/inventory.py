#!/usr/bin/env python3
"""inventory.py — Multi-pass Qwen object inventory on the sliced PLY topdown.

Runs N separate Qwen calls, each focused on a DIFFERENT category set.
Passes can return overlapping items (no dedup).

PASSES:
  Pass 1: seating (sofa, armchair, dining chair, stool, ottoman)
  Pass 2: storage (cabinet, bookshelf)
  Pass 3: tables (coffee, dining, side)

Rugs are intentionally NOT inventoried — see PASSES list below.

Renders at 4K (3840×2160) for higher per-item precision. Camera params
saved to qwen_items.json so downstream visual-hull / back-projection
can use the bboxes directly (no image resize, no crop, no trickery —
the rendered image and the saved camera define a coherent frustum).

Plugin conventions (qwen_bbox.py):
  - Explicit "TOP-DOWN (bird's eye) view" framing
  - bbox_2d in 0-1000 NORMALIZED coords (scaled to image pixels after)
  - JSON with confidence field

Outputs:
  <scene>/_inventory_temp/topdown_for_qwen.png   (4K, real-camera)
  <scene>/_inventory_temp/qwen_<pass>_raw.txt    (per-pass raw)
  <scene>/_inventory_temp/qwen_items.json        (items + camera + image_size)
  <scene>/_inventory_temp/qwen_overlay.png       (combined overlay)

Usage:
    python inventory.py <scene_dir>
"""
import os
import argparse
import base64
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from plyfile import PlyData

VIEW_PY = "/home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py"
QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-awq")

FOV = 70.0
W, H = 3840, 2160          # 4K render — kept honest for visual-hull back-projection
TOPDOWN_MARGIN = 3.0

# Default passes are loaded from room_config.py per detected/specified
# room type. The legacy LR/kitchen+dining list is kept here as a
# fallback if the room_config import fails.
_DEFAULT_PASSES = [
    {"name": "pass1_seating",
     "categories": ["sofa", "armchair", "dining chair", "stool", "ottoman"]},
    {"name": "pass2_storage",
     "categories": ["cabinet"]},
    {"name": "pass3_tables",
     "categories": ["coffee table", "dining table", "side table"]},
]


def load_passes_for_room(room_type: str | None):
    """Read inventory passes from room_config.py for the given room type.
    Falls back to legacy LR-style passes if the config can't be imported."""
    if not room_type:
        return _DEFAULT_PASSES
    try:
        from room_config import get_room_config   # noqa: WPS433
    except ImportError:
        return _DEFAULT_PASSES
    cfg = get_room_config(room_type)
    return [{"name": name, "categories": cats}
            for name, cats in cfg.get("inventory_passes", [])]


def render_topdown(ply: Path, out: Path) -> dict:
    """Render topdown at 4K. Returns camera params dict so downstream
    visual-hull / back-projection can reconstruct the same frustum."""
    pl = PlyData.read(str(ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    x_lo, z_lo = np.percentile(xyz[:, [0, 2]], 15, axis=0)
    x_hi, z_hi = np.percentile(xyz[:, [0, 2]], 85, axis=0)
    cx, cz = (x_lo + x_hi) / 2, (z_lo + z_hi) / 2
    xe, ze = float(x_hi - x_lo), float(z_hi - z_lo)
    yf = float(np.percentile(xyz[:, 1], 85))
    aspect = W / H
    tan_h = math.tan(math.radians(FOV / 2))
    dist = max((xe * TOPDOWN_MARGIN) / (2 * tan_h * aspect),
               (ze * TOPDOWN_MARGIN) / (2 * tan_h))
    eye = (float(cx), float(yf - dist), float(cz))
    target = (float(cx + 0.001), float(yf), float(cz))
    up = (0.0, 0.0, -1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, VIEW_PY, str(ply), str(out),
           f"--eye={eye[0]:.4f},{eye[1]:.4f},{eye[2]:.4f}",
           f"--target={target[0]:.4f},{target[1]:.4f},{target[2]:.4f}",
           f"--up={up[0]},{up[1]},{up[2]}", "--y-down",
           "--fov", str(FOV), "--width", str(W), "--height", str(H)]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"eye": list(eye), "target": list(target), "up": list(up),
            "fov": FOV, "width": W, "height": H, "y_down": True}


def encode_b64(p: Path) -> str:
    """Read image at full resolution, encode as PNG base64."""
    return base64.b64encode(p.read_bytes()).decode()


def build_pass_prompt(categories: list) -> str:
    cat_list = ", ".join(categories)
    return (
        "This image is a TOP-DOWN (bird's eye) view of an interior scene.\n\n"
        f"Find every object that fits any of these CATEGORIES: {cat_list}.\n\n"
        "These are category types, not strict labels. If you see something "
        "that fits one of these categories (e.g. a 'console table' fits "
        "the 'table' category, a 'love seat' fits 'sofa', a 'pendant lamp' "
        "fits 'lamp'), include it. Use a NATURAL label that describes what "
        "you actually see — you do NOT have to copy the category words.\n\n"
        "BANNED in labels — DO NOT use any of these descriptors:\n"
        "  - SIZE words: small, big, large, tall, short, low, high, mini,\n"
        "    huge, tiny, oversized, narrow, wide, deep, slim.\n"
        "  - SHAPE words: rectangular, square, round, oval, l-shaped,\n"
        "    u-shaped, curved, straight.\n"
        "  Use only the noun phrase: 'armchair' not 'small armchair',\n"
        "  'sofa' not 'L-shaped sectional sofa', 'cabinet' not 'low cabinet',\n"
        "  'coffee table' not 'rectangular coffee table'. Color and material\n"
        "  ARE allowed (e.g. 'wooden cabinet', 'beige sofa').\n\n"
        "Do NOT list objects that fall outside these categories.\n\n"
        "For each instance, return a 2D bounding box around its "
        "floor footprint.\n\n"
        "IGNORE — DO NOT LIST these:\n"
        "  - Hardwood floor / wood planks (the surface itself).\n"
        "  - Pillows, cushions on furniture.\n"
        "  - Kitchen cabinets / kitchen counters / kitchen islands / "
        "    kitchen base cabinets / kitchen wall cabinets / any built-in "
        "    cabinetry running along a kitchen wall. SKIP the entire "
        "    kitchen area. Only list furniture in the living / dining "
        "    parts of the room.\n"
        "  - Items sitting on top of furniture (decor, books, vases, "
        "    plates, candles).\n"
        "  - Splat-rendering errors, blurry artifacts, dark scatter "
        "    around the room edges, capture noise.\n\n"
        "Output valid JSON only, no commentary, no markdown fences:\n"
        "{\n"
        '    "items": [\n'
        '        {"bbox_2d": [x_min, y_min, x_max, y_max], '
        '"label": "<your natural label for the object>", '
        '"category": "<which requested category it fits>", '
        '"confidence": "high|medium|low"}\n'
        "    ]\n"
        "}\n\n"
        "Rules:\n"
        "- Coordinates must be integers in the range 0-1000 (normalized, "
        "NOT pixels).\n"
        "- bbox_2d encloses the object's floor footprint.\n"
        "- Each visible instance is its own entry.\n"
        f"- category must be one of: {cat_list}.\n"
        "- label can be more specific than the category if useful.\n"
        "- confidence ∈ {\"high\", \"medium\", \"low\"}.\n"
        "- If 0 instances visible, return {\"items\": []}."
    )


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
    if not items:
        pat2 = re.compile(
            r'"label"\s*:\s*"([^"]+)"[^"]*'
            r'(?:"bbox_2d"|"bbox_pixels"|"bbox")[^\[]*\[\s*'
            r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            re.DOTALL,
        )
        for m in pat2.finditer(cleaned):
            label = m.group(1)
            bbox = [int(m.group(i)) for i in (2, 3, 4, 5)]
            items.append({"bbox_2d": bbox, "label": label,
                          "confidence": "unknown"})
    return items


def scale_bbox_to_pixels(bbox_2d: list, img_w: int, img_h: int) -> list:
    x0, y0, x1, y1 = bbox_2d
    return [int(x0 * img_w / 1000),
            int(y0 * img_h / 1000),
            int(x1 * img_w / 1000),
            int(y1 * img_h / 1000)]


def draw_overlay(img_path: Path, items: list, out: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    # Color per pass
    pass_colors = {
        "pass1_large_furniture": (60, 120, 255),   # blue
        "pass2_rugs":            (200, 60, 200),   # purple
        "pass3_other":           (255, 60, 60),    # red
    }
    for i, it in enumerate(items):
        bbox = it.get("bbox_pixels")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(c) for c in bbox]
        color = pass_colors.get(it.get("pass", ""), (200, 60, 200))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = f"{i+1}. {it.get('label', '?')}"
        tx = max(2, x0)
        ty = max(2, y0 - 26)
        draw.rectangle([tx, ty, tx + 10 * len(label), ty + 24], fill=color)
        draw.text((tx + 4, ty + 1), label, fill=(255, 255, 255), font=font)
    img.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--room-type", default=None,
                    help="Override auto-detect: living_room | dining_room | "
                          "kitchen | bedroom | office | bathroom | hallway | "
                          "mixed. Defaults to reading scene_dir/_inventory_temp/"
                          "room_type.json (set by detect_room_type.py) or "
                          "'mixed' if missing.")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()

    in_ply = scene / "step7_sliced.ply"
    if not in_ply.exists():
        sys.exit(f"[fatal] missing input: {in_ply}\n  run slice.py first")

    out_dir = scene / "_inventory_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "topdown_for_qwen.png"
    print(f"[render] topdown 4K → {img_path}")
    camera = render_topdown(in_ply, img_path)

    img = Image.open(img_path)
    img_w, img_h = img.size
    assert (img_w, img_h) == (camera["width"], camera["height"]), (
        f"image size {img_w}×{img_h} != camera {camera['width']}×{camera['height']}")
    print(f"[image] {img_w}×{img_h}")
    print(f"[camera] eye={camera['eye']} target={camera['target']} "
          f"up={camera['up']} fov={camera['fov']}")

    # Resolve room type: CLI override > detect output > 'mixed' default
    room_type = args.room_type
    if not room_type:
        rt_file = out_dir / "room_type.json"
        if rt_file.exists():
            try:
                room_type = json.loads(rt_file.read_text()).get("room_type")
            except Exception:
                pass
    if not room_type:
        room_type = "mixed"
    PASSES = load_passes_for_room(room_type)
    print(f"[room-type] {room_type} → {len(PASSES)} inventory passes "
          f"({[p['name'] for p in PASSES]})")

    all_items = []
    for p in PASSES:
        pname = p["name"]
        cats = p["categories"]
        print(f"\n[{pname}] categories: {cats}")
        prompt = build_pass_prompt(cats)
        raw = call_qwen(img_path, prompt)
        (out_dir / f"qwen_{pname}_raw.txt").write_text(raw)
        items = parse_items(raw)
        for it in items:
            b = it.get("bbox_2d")
            if b and len(b) == 4:
                it["bbox_pixels"] = scale_bbox_to_pixels(b, img_w, img_h)
            it["pass"] = pname
        print(f"  got {len(items)} items")
        for it in items:
            print(f"    - {it.get('label')}  bbox_2d={it.get('bbox_2d')}  "
                  f"conf={it.get('confidence', '?')}")
        all_items.extend(items)

    print(f"\n[total] {len(all_items)} raw items across {len(PASSES)} passes")

    # Cross-pass IoU dedup. Phase 1 passes can return the SAME physical
    # object under different categories (e.g. "open shelving unit" vs
    # "wooden cabinet"). When two bboxes overlap heavily, keep the
    # LARGER one and drop the smaller. IoU 0.5 is looser than phase 2's
    # 0.9 because we DON'T require label match here.
    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
    deduped = []
    dropped = []
    DEDUP_IOU = 0.5
    # Sort by bbox area descending so the larger item is seen first.
    sorted_items = sorted(
        all_items,
        key=lambda it: -((it["bbox_pixels"][2] - it["bbox_pixels"][0]) *
                          (it["bbox_pixels"][3] - it["bbox_pixels"][1])),
    )
    for it in sorted_items:
        bb = it["bbox_pixels"]
        clash = None
        for kept in deduped:
            if _iou(bb, kept["bbox_pixels"]) >= DEDUP_IOU:
                clash = kept
                break
        if clash is None:
            deduped.append(it)
        else:
            dropped.append((it, clash))
    if dropped:
        print(f"[dedup] dropped {len(dropped)} duplicates (IoU >= {DEDUP_IOU}):")
        for d, k in dropped:
            print(f"  '{d['label']}' (kept '{k['label']}')")
    all_items = deduped
    print(f"[total] {len(all_items)} items after dedup")

    (out_dir / "qwen_items.json").write_text(
        json.dumps({
            "items": all_items,
            "image_size": [img_w, img_h],
            "image_path": str(img_path),
            "camera": camera,                    # for visual-hull back-projection
            "passes": [p["name"] for p in PASSES],
        }, indent=2))
    overlay_path = out_dir / "qwen_overlay.png"
    draw_overlay(img_path, all_items, overlay_path)

    print(f"\n[done] STOP")
    print(f"  topdown:  {img_path}")
    print(f"  items:    {out_dir / 'qwen_items.json'}")
    print(f"  overlay:  {overlay_path}")


if __name__ == "__main__":
    main()
