#!/usr/bin/env python3
"""qc_reject.py — Final QC pass: ask Qwen if the extracted object is
recognizable. If it's clearly noise / heavily damaged / unidentifiable,
move the whole folder to <scene>/rejects/.

Lenient threshold: imperfections, missing chunks, halo bleed, partial
fragments are all PASS. Reject ONLY when Qwen "cannot make out the
object at all" across all 5 canonical views.

Sends all 5 canonical renders (y0, y90, y180, y270, topdown) of the
object's most-refined stage (the shared stage_preference order, led by
stage_pick's 8_final) to Qwen as a multi-image request. Picks the latest
stage that actually has all 5 canonical renders on disk.

Reads:
  <obj>/renders/<stage>/{y0,y90,y180,y270,topdown}.png
  <obj>/1_visual_hull_meta.json   (for the inventory label)

Writes (always):
  <obj>/qc_reject.json   verdict + reason

On REJECT:
  Moves <scene>/02_<slug>/ → <scene>/rejects/02_<slug>/
  qc_reject.json travels with it.

Usage:
    python qc_reject.py <scene_dir> 02_<slug>/
"""
import argparse
import base64
import io
import json
import re
import shutil
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

VIEWS = ["y0", "y90", "y180", "y270", "topdown"]

# Stage preference order is the shared canonical list (stage_preference.py).
# This gate runs AFTER stage_pick, so 8_final (its picked + destreaked output)
# is at the top and is what we judge. qc_reject picks the latest stage that has
# all 5 canonical RENDERS on disk (not just the .ply) — see the loop below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_preference import STAGE_PREFERENCE  # noqa: E402


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_last_json(raw: str) -> dict | None:
    s = raw.strip()
    if s.startswith("```"):
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


def pick_render_dir(obj: Path) -> tuple[str, Path] | None:
    for stage in STAGE_PREFERENCE:
        d = obj / "renders" / stage
        if all((d / f"{v}.png").exists() for v in VIEWS):
            return stage, d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path, help="path to 02_<slug>/")
    ap.add_argument("--no-move", action="store_true",
                    help="write the verdict but DO NOT move rejects/. "
                         "Used by the dispatcher when it wants to try a "
                         "fallback before committing to the move.")
    args = ap.parse_args()
    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()

    picked = pick_render_dir(obj)
    if picked is None:
        sys.exit(f"[fatal] no stage in {obj}/renders/ has all 5 canonical "
                 f"views — checked: {STAGE_PREFERENCE}")
    stage, render_dir = picked

    meta_path = obj / "1_visual_hull_meta.json"
    label = "object"
    if meta_path.exists():
        try:
            label = json.load(open(meta_path)).get("label", "object")
        except Exception:
            pass

    print(f"[qc_reject] obj={obj.name}  label='{label}'  stage={stage}")

    images = [(v, render_dir / f"{v}.png") for v in VIEWS]

    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = []
    for tag, p in images:
        content.append({"type": "text", "text": f"\nView {tag}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_b64(p)}"}})
    content.append({"type": "text", "text":
        f"You are looking at 5 canonical renders (y0, y90, y180, y270, "
        f"topdown) of an extracted gaussian-splat object. The inventory "
        f"label says this should be: '{label}'.\n\n"
        f"Decide if the extraction is keepable: PASS or REJECT.\n\n"
        f"Be LENIENT — the goal is to catch ONLY absolute trash. It "
        f"will not be perfect every time, and that is fine: a rough or "
        f"imperfect extraction that is still recognizable should be "
        f"KEPT. Imperfections, missing chunks, halo bleed, partial "
        f"fragments, blurry surfaces, asymmetric leaves, minor noise, "
        f"and SMALL CLIPPED PARTS (a cut-off leg, a missing corner, a "
        f"clipped edge) are ALL FINE — return PASS for those as long "
        f"as you can still tell it's a '{label}'. Thin objects (lamps, "
        f"poles) look like slivers from the side and a dot from above "
        f"— that is normal, judge by the views where the object reads "
        f"clearly. When uncertain, return PASS.\n\n"
        f"Return REJECT ONLY when the extraction is genuinely unusable:\n"
        f"  1. NOISE / EMPTY: mostly white space with a few scattered "
        f"specks or smears, no recognizable object across the 5 views "
        f"combined.\n"
        f"  2. WRONG OBJECT: the views clearly show a different object "
        f"CLASS than '{label}'. Only reject for an obviously-wrong "
        f"class — never for stylistic or colour mismatches.\n\n"
        f"Reply in JSON with these exact fields:\n"
        f'  - verdict: "PASS" or "REJECT"\n'
        f"  - reason: one short sentence explaining the call (cite which "
        f"views you used and what you do/do not see)\n"
        f"  - object_visible: true/false (can you see ANY coherent "
        f"object across the 5 views — even if it's the wrong class?)\n"
        f"  - matches_label: true/false (does what you see plausibly "
        f"match the label '{label}'?)\n\n"
        f"Output ONLY the JSON object. No prose, no markdown fences."})
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=400, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    parsed = extract_last_json(raw)
    if parsed is None:
        print(f"\n[qwen raw]\n{raw}\n")
        sys.exit("[fatal] could not parse JSON from Qwen response")

    verdict = str(parsed.get("verdict", "")).strip().upper()
    reason = parsed.get("reason", "")
    object_visible = parsed.get("object_visible", None)
    matches_label = parsed.get("matches_label", None)

    out = {
        "stage_inspected": stage,
        "label": label,
        "verdict": verdict,
        "reason": reason,
        "object_visible": object_visible,
        "matches_label": matches_label,
        "qwen_raw": raw,
    }
    (obj / "qc_reject.json").write_text(json.dumps(out, indent=2))

    print(f"\n[qc_reject] verdict={verdict}  visible={object_visible}  "
          f"matches_label={matches_label}")
    print(f"[qc_reject] reason: {reason}")

    if verdict == "REJECT" and not args.no_move:
        rejects_root = scene / "rejects"
        rejects_root.mkdir(exist_ok=True)
        dest = rejects_root / obj.name
        if dest.exists():
            i = 2
            while (rejects_root / f"{obj.name}_{i}").exists():
                i += 1
            dest = rejects_root / f"{obj.name}_{i}"
        print(f"[qc_reject] moving {obj.name} → rejects/{dest.name}")
        shutil.move(str(obj), str(dest))
        print(f"[qc_reject] DONE — rejected to {dest}")
    elif verdict == "REJECT":
        print(f"[qc_reject] DONE — verdict=REJECT but --no-move; folder kept "
              f"in place for caller to handle")
    else:
        print(f"[qc_reject] DONE — verdict=PASS, kept")


if __name__ == "__main__":
    main()
