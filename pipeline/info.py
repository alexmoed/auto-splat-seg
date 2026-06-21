#!/usr/bin/env python3
"""info.py — Final descriptive metadata for an extracted object.

Renders 13 fresh views (8 yaws + 4 oblique + topdown) of the object's
picked final stage (see stage_preference.py), sends them to Qwen as a
multi-image request, and writes a structured JSON description to
`<obj>/info.json`.

Color is baked into each name string (e.g. "beige armchair", "brown
throw blanket"), no separate colors[] array.

Reads:
  <obj>/<final-stage>.ply          (most-refined stage; see stage_preference)
  <obj>/1_visual_hull_meta.json    (for the inventory label)

Writes:
  <obj>/info.json

Usage:
    python info.py <scene_dir> 02_<slug>/
"""
import argparse
import base64
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_preference import pick_stage, STAGE_PREFERENCE  # noqa: E402

sys.path.insert(0, "/home/ubuntu/room_pipeline_v002/pipeline")
from sam_carve import build_camera  # noqa: E402

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

# Independent review: render the final PLY at every angle that matters.
# 8 yaws at slight downward tilt + 4 oblique yaws (steeper, see tops) +
# topdown = 13 views.
REVIEW_YAWS_FLAT = [0, 45, 90, 135, 180, 225, 270, 315]
REVIEW_YAWS_OBLIQUE = [0, 90, 180, 270]
REVIEW_PITCH_FLAT = -20.0
REVIEW_PITCH_OBLIQUE = -50.0
REVIEW_TOPDOWN_PITCH = -89.0
REVIEW_FOV = 50.0
REVIEW_W = 1920
REVIEW_H = 1080
REVIEW_MARGIN = 1.55  # extent * margin / (2 tan(fov/2)) = camera distance


def render_review_views(ply_path: Path, out_dir: Path) -> list:
    """Render 13 fresh views of the picked PLY. Returns [(tag, png_path)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    scene = load_gsplat_ply(str(ply_path))
    means = scene["means"].detach().cpu().numpy()
    lo = np.percentile(means, 2, axis=0)
    hi = np.percentile(means, 98, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = max(float((hi - lo).max()), 0.15)
    tan_half = math.tan(math.radians(REVIEW_FOV) / 2)
    distance = (extent * REVIEW_MARGIN) / (2 * tan_half) + 0.5

    views = []
    for yaw in REVIEW_YAWS_FLAT:
        V, K, _ = build_camera(center, yaw, REVIEW_PITCH_FLAT, distance,
                                REVIEW_FOV, REVIEW_W, REVIEW_H, y_down=True)
        img = render_splat(scene, V, K, REVIEW_W, REVIEW_H, bg=(1.0, 1.0, 1.0))
        tag = f"y{yaw}"
        p = out_dir / f"{tag}.png"
        Image.fromarray(img).save(p)
        views.append((tag, p))
    for yaw in REVIEW_YAWS_OBLIQUE:
        V, K, _ = build_camera(center, yaw, REVIEW_PITCH_OBLIQUE, distance,
                                REVIEW_FOV, REVIEW_W, REVIEW_H, y_down=True)
        img = render_splat(scene, V, K, REVIEW_W, REVIEW_H, bg=(1.0, 1.0, 1.0))
        tag = f"oblique_y{yaw}"
        p = out_dir / f"{tag}.png"
        Image.fromarray(img).save(p)
        views.append((tag, p))
    V, K, _ = build_camera(center, 0, REVIEW_TOPDOWN_PITCH, distance,
                            REVIEW_FOV, REVIEW_W, REVIEW_H, y_down=True)
    img = render_splat(scene, V, K, REVIEW_W, REVIEW_H, bg=(1.0, 1.0, 1.0))
    p = out_dir / "topdown.png"
    Image.fromarray(img).save(p)
    views.append(("topdown", p))
    return views


VIEWS = ["y0", "y90", "y180", "y270", "topdown"]  # legacy


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

    # Pick the object's final stage via the shared canonical preference
    # (stage_preference.py — single source of truth). 8_final = stage_pick's
    # picked + destreaked deliverable and sits at the top of the list.
    stage, in_ply = pick_stage(obj)
    if stage is None:
        sys.exit(f"[fatal] no stage PLY found in {obj} (looked for: "
                 f"{', '.join(s + '.ply' for s in STAGE_PREFERENCE)})")

    # Render 13 fresh views (8 yaws + 4 oblique + topdown) of the picked
    # PLY for an independent review. Saved to renders/info_review/.
    review_dir = obj / "renders" / "info_review"
    print(f"[info] rendering 13 review views of {stage}.ply → {review_dir}")
    images = render_review_views(in_ply, review_dir)

    print(f"[info] independent review — sending {len(images)} views to Qwen "
          f"(no upstream label passed; describe fresh from images only)")

    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag, p in images:
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    content.append({"type": "text", "text":
        f"You are looking at {len(images)} views of a single extracted "
        f"object on a white background (8 yaws at slight downward tilt, "
        f"4 oblique yaws at steeper pitch showing top surfaces, and a "
        f"topdown). No prior label is given — describe ONLY what you see.\n\n"
        f"Output a single JSON object with these fields, IN THIS ORDER:\n\n"
        f"  1. name: NAME THE OBJECT. Short specific noun phrase for the "
        f"main piece, color baked in. Examples: \"beige tufted lounge "
        f"armchair\", \"round walnut coffee table\", \"low wooden media "
        f"cabinet\", \"black metal floor lamp\".\n\n"
        f"  2. description: 2-4 sentences describing the main object — "
        f"its shape, proportions, structural features (legs, base, "
        f"backrest, doors, drawers, top), how it presents visually.\n\n"
        f"  3. sub_objects: list of EVERY distinct item resting on / "
        f"draped over / placed on / mounted to the main object. Each "
        f"entry is a SHORT name with color + material baked in. List "
        f"pillows, throws, vases, plants, lamps, books, remotes, "
        f"electronics, decor, hardware — every visible item. Empty list "
        f"only if nothing is on/with the main object.\n\n"
        f"  4. style: descriptive style phrase covering era + aesthetic "
        f'(e.g. "mid-century modern", "contemporary minimalist", '
        f'"traditional shaker", "industrial loft", "scandinavian", '
        f'"art deco"). 1-4 words.\n\n'
        f"  5. materials: list of distinct material types visible "
        f'(e.g. ["upholstered fabric", "stained oak wood", "brushed '
        f'steel", "leather", "tempered glass", "glazed ceramic", "matte '
        f'plastic", "live foliage", "woven rattan"]). Be specific where '
        f"you can tell.\n\n"
        f"  6. colors: list of the dominant colors visible on the main "
        f"object and prominent sub-objects (e.g. [\"warm beige\", "
        f'"walnut brown", "matte black", "muted sage green"]). Use '
        f"specific descriptive color names, not just \"brown\".\n\n"
        f"Output ONLY a single JSON object with these six fields. No "
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
        "name": parsed.get("name", ""),
        "description": parsed.get("description", ""),
        "sub_objects": parsed.get("sub_objects", []),
        "style": parsed.get("style", ""),
        "materials": parsed.get("materials", []),
        "colors": parsed.get("colors", []),
    }
    out = obj / "info.json"
    out.write_text(json.dumps(info, indent=2))

    print(f"\n[info]")
    print(json.dumps(info, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
