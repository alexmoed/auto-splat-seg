#!/usr/bin/env python3
"""qc_reject.py — Final QC pass: ask Qwen if the extracted object is
recognizable. If it's clearly noise / heavily damaged / unidentifiable,
move the whole folder to <scene>/rejects/.

Lenient threshold: imperfections, missing chunks, halo bleed, partial
fragments are all PASS. Reject ONLY when Qwen "cannot make out the
object at all" across all 5 canonical views.

Sends all 5 canonical renders (y0, y90, y180, y270, topdown) of the
latest stage's output (5_bookshelf_sweep > 4_sam_tight > 3_floor_drop >
2_sam_wide > 1_visual_hull, in that preference order) to Qwen as a
multi-image request.

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

# Stage preference order — pick the latest stage that has all 5 renders.
STAGE_PREFERENCE = [
    # 5_subtracted = parent with children carved out (group/subtract).
    # 5_bookshelf_sweep / 4_rug = class-specific finals.
    # 4_sam_tight = canonical SAM-vote output (preferred over sweep_fallback
    # which is the safety net).
    # 5_sweep_fallback exists for every object as automatic safety; it's
    # only inspected here if sam_tight has been renamed to
    # 4_sam_tight_rejected.ply by the dispatcher's REJECT-recovery path.
    "5_subtracted",
    "5_bookshelf_sweep",
    "4_rug",
    "4_sam_tight",
    "5_sweep_fallback",
    "3_floor_drop",
    "2_sam_wide",
    "1_visual_hull",
]


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
        f"Decide if the extraction is keepable.\n\n"
        f"Be LENIENT on quality. Imperfections, missing chunks, halo "
        f"bleed, partial fragments, blurry surfaces, asymmetric leaves, "
        f"and minor noise are all FINE — return PASS for those as long "
        f"as you can still tell it's a '{label}'.\n\n"
        f"Reject in either of these cases:\n"
        f"  1. NOISE / EMPTY: views are mostly white space with only a "
        f"few scattered specks or no recognizable shape at all across "
        f"the 5 views combined.\n"
        f"  2. WRONG OBJECT: the views show a clearly different object "
        f"than '{label}' (e.g. label says 'potted plant' but the views "
        f"show a fabric drape, a cushion, a chair arm, etc.). The shape "
        f"and colors must be plausibly consistent with a '{label}'. "
        f"Don't reject for minor stylistic mismatches — only reject if "
        f"the object class is obviously wrong.\n\n"
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
        print(f"[qc_reject] DONE — kept")


if __name__ == "__main__":
    main()
