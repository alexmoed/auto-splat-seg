#!/usr/bin/env python3
"""detect_room_type.py — ask Qwen what kind of room is shown in a render
(or set of renders). Returns one of the room-types defined in
room_config.VALID_ROOM_TYPES.

Usage:
    python detect_room_type.py <topdown.png> [extra1.png extra2.png ...]
    python detect_room_type.py <scene_dir>   # auto-locates topdown

Prints the chosen room type to stdout. Writes JSON detail to
<scene_dir>/_inventory_temp/room_type.json if a scene_dir is provided.
"""
import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image
from openai import OpenAI

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))

from room_config import VALID_ROOM_TYPES   # noqa: E402

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"


def encode_b64(p: Path, max_dim: int = 1280) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        new_size = (int(img.size[0] * s), int(img.size[1] * s))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


PROMPT = (
    "Look at this rendered view of an interior space. Decide what kind of "
    "room it is. Choose EXACTLY ONE label from this list:\n\n"
    "  - living_room   (sofa, coffee table, TV, lounge seating)\n"
    "  - dining_room   (dining table with chairs around it)\n"
    "  - kitchen       (counters, cabinets, stove/oven, fridge)\n"
    "  - bedroom       (bed, dresser, nightstand)\n"
    "  - office        (desk, office chair, bookshelves, filing cabinet)\n"
    "  - bathroom      (toilet, vanity, bathtub, shower)\n"
    "  - hallway       (corridor, console, narrow space, no main seating)\n"
    "  - mixed         (open-plan combination of two or more of the above, "
    "e.g. kitchen+living+dining in one big space)\n\n"
    "Return ONLY a JSON object with a single field 'room_type'. No prose, "
    "no markdown.\n\n"
    "Example outputs:\n"
    '  {\"room_type\": \"living_room\"}\n'
    '  {\"room_type\": \"mixed\"}\n'
)


def detect(image_paths: list, debug: bool = False) -> str:
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"},
        })
    content.append({"type": "text", "text": PROMPT})

    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=200, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    if debug:
        print(f"[debug] raw qwen: {raw!r}", file=sys.stderr)

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    rt = None
    if s != -1 and e > s:
        try:
            data = json.loads(cleaned[s:e])
            rt = data.get("room_type")
        except json.JSONDecodeError:
            pass
    if not rt:
        m = re.search(r'"room_type"\s*:\s*"([^"]+)"', raw)
        if m:
            rt = m.group(1)

    if not rt:
        # Last-ditch: scan for any of the valid labels in raw output
        for cand in VALID_ROOM_TYPES:
            if cand in raw.lower().replace(" ", "_"):
                rt = cand
                break

    if not rt or rt not in VALID_ROOM_TYPES:
        print(f"[warn] uncertain — Qwen returned {rt!r}, falling back to "
              f"'mixed'", file=sys.stderr)
        rt = "mixed"

    return rt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="topdown image OR scene directory")
    ap.add_argument("extras", nargs="*", type=Path,
                    help="extra context images (corner views etc.)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    scene_dir = None
    images = []
    if args.input.is_dir():
        scene_dir = args.input.resolve()
        topdown = scene_dir / "_inventory_temp" / "topdown_for_qwen.png"
        if not topdown.exists():
            sys.exit(f"[fatal] topdown not found at {topdown} — run "
                      f"inventory.py first or pass an image directly")
        images = [topdown]
        # Single-image only — vLLM context window (4096) doesn't fit
        # multiple 1280px renders. Topdown alone is enough to classify
        # the room in nearly all cases (open-plan kitchens visible from
        # above, beds visible from above, desks visible from above).
    else:
        images = [args.input] + list(args.extras)

    if not images:
        sys.exit("[fatal] no input images")

    print(f"[detect] inputs: {[str(p) for p in images]}", file=sys.stderr)
    rt = detect(images, debug=args.debug)
    print(rt)

    if scene_dir:
        out = scene_dir / "_inventory_temp" / "room_type.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "room_type": rt,
            "n_images_used": len(images),
            "input_images": [str(p) for p in images],
        }, indent=2))
        print(f"[save] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
