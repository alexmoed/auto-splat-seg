#!/usr/bin/env python3
"""info.py — Final descriptive metadata for an extracted object.

Sends the 4 canonical yaws of `4_sam_tight.ply` (renders/4_sam_tight/
y{0,90,180,270}.png) to Qwen as a multi-image request and writes a
structured JSON description to `<obj>/info.json`.

Color is baked into each name string (e.g. "beige armchair", "brown
throw blanket"), no separate colors[] array.

Reads:
  <obj>/renders/4_sam_tight/{y0,y90,y180,y270}.png
  <obj>/1_visual_hull_meta.json   (for the inventory label)

Writes:
  <obj>/info.json

Usage:
    python info.py <scene_dir> 02_<slug>/
"""
import os
import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image

QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-awq")

VIEWS = ["y0", "y90", "y180", "y270", "topdown"]


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_last_json(raw: str) -> dict | None:
    """Try direct parse first; on failure, slice from first '{' to last '}'."""
    s = raw.strip()
    if s.startswith("```"):
        # strip markdown fences
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path,
                    help="path to 02_<slug>/")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()

    # Prefer the latest stage's output: 5_sweep_fallback (bbox-sweep
    # recovery) > 5_bookshelf_sweep > 4_sam_tight > 3_floor_drop.
    # 5_sweep_fallback wins because its presence means an earlier
    # stage was rejected.
    # 5_subtracted (parent with children carved out) > class-specific
    # finals (5_bookshelf_sweep, 4_rug) > sam_tight > sweep_fallback
    # (safety net) > earlier stages. The dispatcher renames
    # 4_sam_tight.ply → 4_sam_tight_rejected.ply on qc REJECT, which
    # makes 5_sweep_fallback the highest-ranked existing stage on the
    # recovery pass.
    stage_candidates = ["5_subtracted", "5_bookshelf_sweep", "4_rug",
                         "4_sam_tight", "5_sweep_fallback", "3_floor_drop"]
    stage = next((s for s in stage_candidates if (obj / f"{s}.ply").exists()), None)
    if stage is None:
        sys.exit(f"[fatal] no stage PLY found in {obj} (looked for: "
                 f"{', '.join(s + '.ply' for s in stage_candidates)})")
    in_ply = obj / f"{stage}.ply"
    render_dir = obj / "renders" / stage
    images = []
    for tag in VIEWS:
        p = render_dir / f"{tag}.png"
        if not p.exists():
            sys.exit(f"[fatal] missing {p}")
        images.append((tag, p))

    meta_path = obj / "1_visual_hull_meta.json"
    label = "object"
    if meta_path.exists():
        try:
            label = json.load(open(meta_path)).get("label", "object")
        except Exception:
            pass

    print(f"[info] label='{label}', sending {len(images)} views to Qwen")

    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag, p in images:
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    content.append({"type": "text", "text":
        f"You are looking at 5 views (4 yaws + topdown) of a single object. "
        f"The inventory label is: '{label}'.\n\n"
        f"Describe it in JSON with these exact fields:\n"
        f"  - object_type: short name with color baked in "
        f'(e.g. "beige lounge armchair", "wooden coffee table", '
        f'"stack of colorful books", "wooden picture frame with photograph"). '
        f"One short string.\n"
        f"  - sub_objects: list of items resting on / draped over / placed "
        f"on the main object — each name with color baked in. Empty list "
        f"if nothing is on top.\n"
        f"  - materials: list of material types you can see "
        f'(e.g. ["upholstered fabric", "wood", "metal", "leather", "glass", '
        f'"ceramic", "paper", "plastic", "foliage"]).\n'
        f"  - style: short style descriptor (one short string).\n\n"
        f"Output ONLY a single JSON object with these four fields. No "
        f"prose, no markdown fences, no commentary."})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1500, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    parsed = extract_last_json(raw)
    if parsed is None:
        print(f"\n[qwen raw]\n{raw}\n")
        sys.exit("[fatal] could not parse JSON from Qwen response")

    rel_ply = in_ply.relative_to(scene)
    info = {
        "object_ply": str(rel_ply),
        "label": label,
        "object_type": parsed.get("object_type", ""),
        "confirmed_inventory": parsed.get("confirmed_inventory", None),
        "sub_objects": parsed.get("sub_objects", []),
        "materials": parsed.get("materials", []),
        "style": parsed.get("style", ""),
    }
    out = obj / "info.json"
    out.write_text(json.dumps(info, indent=2))

    print(f"\n[info]")
    print(json.dumps(info, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
