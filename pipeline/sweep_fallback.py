#!/usr/bin/env python3
"""sweep_fallback.py — Final fallback when sam_tight produces nothing.

Same multi-view Qwen-bbox vote as bookshelf_sweep, but:
- Sources from 3_floor_drop.ply (the last clean stage before sam_tight)
- Uses the inventory label (not hard-coded "bookshelf") in the prompt
- 5% bbox padding (vs 2% for bookshelf — more tolerant for general
  furniture where Qwen bbox tightness varies)
- Output: 5_sweep_fallback.ply + renders/5_sweep_fallback/

Why this exists: sam_tight (especially the default 0.7-vote variant)
sometimes carves the entire object away — leaving no 4_sam_tight.ply.
The downstream chain (info, qc_reject) then has nothing to inspect.
This stage rebuilds the object from 3_floor_drop using a multi-view
bbox cone, which is more permissive than SAM mask voting.

Usage:
    python sweep_fallback.py <scene_dir> <obj_dir>

Reads:
    <obj>/3_floor_drop.ply
    <obj>/1_visual_hull_meta.json    (for the inventory label)

Writes:
    <obj>/5_sweep_fallback.ply
    <obj>/renders/5_sweep_fallback/{y0,y90,y180,y270,topdown}.png
    <obj>/diagnostics/5_sweep_fallback/{input_*,overlay_*}.png + report.json
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
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402

from sam_carve import build_camera, render_canonical_5, compute_wall_skip  # noqa: E402

YAWS_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
SWEEP_PITCH = -15.0
TOPDOWN_PITCH = -89.0

FOV = 70.0
W, H = 1920, 1080
RENDER_MARGIN = 2.0
Y_DOWN = True

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

VOTE_FRAC = 0.6   # slightly looser than bookshelf (0.7) — fallback runs
                  # when sam_tight already failed; better to keep extra
                  # halo than nuke the body again.
BBOX_PAD_PCT = 0.05
TOP_PAD_PCT = 0.12  # extra upward pad on y_min so items-on-top of the
                     # parent (lamp on cabinet, plant on sideboard) stay
                     # inside the bbox even when Qwen drew it tight to
                     # the parent's wooden body.


def qwen_prompt(label: str) -> str:
    return (
        f"You are looking at one view of a single piece of furniture and "
        f"possibly its surroundings.\n\n"
        f"TASK: return a bounding box around THE {label.upper()} plus "
        f"ONLY the items RESTING ON it, INSIDE it, or ATTACHED to it.\n\n"
        f"INCLUDE inside the bbox:\n"
        f"  • The {label}'s body, frame, legs, base.\n"
        f"  • Items RESTING ON TOP of the {label} (lamps, plants, "
        f"vases, picture frames, decor).\n"
        f"  • Items INSIDE the {label} (books on shelves, baskets, "
        f"items in drawers, decor on shelves).\n"
        f"  • Items ATTACHED to the {label} (hardware, knobs).\n\n"
        f"EXCLUDE — DO NOT extend the bbox to cover any of these even if "
        f"they visually overlap the {label} in this view:\n"
        f"  • Chairs, sofas, tables, stools, or any other furniture "
        f"sitting IN FRONT OF the {label} (between camera and {label}).\n"
        f"  • Furniture or objects BESIDE, BEHIND, or LEANING AGAINST "
        f"the {label} — they are not resting ON or INSIDE it.\n"
        f"  • Paintings / posters / framed prints on the wall behind it.\n"
        f"  • Items sitting on the floor next to the {label}.\n"
        f"  • Plants, lamps, or decor that stand on the floor near but "
        f"not on the {label}.\n\n"
        f"Test for each candidate item: would it stay where it is if "
        f"you lifted the {label} off the floor? If YES → it's resting "
        f"on / inside / attached to the {label} → include. If NO → "
        f"exclude.\n\n"
        f"If the {label} is not clearly visible in this view (e.g. you "
        f"are looking at it edge-on and only see a thin slab, or it is "
        f"off-frame or you cannot identify it), return "
        f'{{"found": false}}.\n\n'
        f"Otherwise return JSON in this EXACT format (coordinates in 0-1000 "
        f"normalized image space, x0<x1, y0<y1):\n"
        f'{{"found": true, "bbox_2d": [x0, y0, x1, y1]}}\n\n'
        f"Output ONLY the JSON object. No commentary, no markdown."
    )


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_qwen_bbox(img_path: Path, label: str) -> dict:
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{encode_b64(img_path)}"}},
        {"type": "text", "text": qwen_prompt(label)},
    ]
    r = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=400, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = r.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}") + 1
    out = {"found": False, "bbox_2d": None, "raw": raw}
    if s == -1 or e <= s:
        return out
    try:
        data = json.loads(cleaned[s:e])
    except json.JSONDecodeError:
        m = re.search(
            r'(?:"bbox_2d"|"bbox_pixels"|"bbox")\s*:\s*\[\s*'
            r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
            cleaned)
        if m:
            data = {"found": True,
                    "bbox_2d": [int(m.group(i)) for i in (1, 2, 3, 4)]}
        else:
            return out
    if not data.get("found", False):
        return out
    bbox = data.get("bbox_2d") or data.get("bbox_pixels") or data.get("bbox")
    if not bbox or len(bbox) != 4:
        return out
    out["found"] = True
    out["bbox_2d"] = [int(v) for v in bbox]
    return out


def norm_to_pixels(bbox_2d: list, w: int, h: int) -> list:
    x0, y0, x1, y1 = bbox_2d
    return [int(x0 * w / 1000),
            int(y0 * h / 1000),
            int(x1 * w / 1000),
            int(y1 * h / 1000)]


def draw_bbox_overlay(img_path: Path, bbox_px: list, out_path: Path,
                      tag: str, found: bool):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    color = (60, 200, 80) if found else (200, 60, 60)
    label_str = f"{tag}: {'BBOX' if found else 'NOT FOUND'}"
    if found and bbox_px:
        x0, y0, x1, y1 = bbox_px
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
    draw.rectangle([0, 0, 14 * len(label_str) + 8, 36], fill=color)
    draw.text((4, 2), label_str, fill=(255, 255, 255), font=font)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--vote-frac", type=float, default=VOTE_FRAC)
    ap.add_argument("--bbox-pad-pct", type=float, default=BBOX_PAD_PCT)
    ap.add_argument("--source-stage", default="3_floor_drop",
                    help="which earlier stage to source from (default 3_floor_drop)")
    args = ap.parse_args()

    obj = args.obj_dir.resolve()
    src_ply = obj / f"{args.source_stage}.ply"
    if not src_ply.exists():
        sys.exit(f"[fatal] no {args.source_stage}.ply in {obj}")

    label = "object"
    meta_path = obj / "1_visual_hull_meta.json"
    if meta_path.exists():
        try:
            label = json.load(open(meta_path)).get("label", "object")
        except Exception:
            pass

    diag = obj / "diagnostics" / "5_sweep_fallback"
    diag.mkdir(parents=True, exist_ok=True)
    for f in diag.glob("*.png"):
        f.unlink()

    pl = PlyData.read(str(src_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n_total = len(xyz)
    if n_total == 0:
        sys.exit("[fatal] source PLY is empty")
    print(f"[load] {src_ply.name}  {n_total:,} splats  label='{label}'")

    means_f32 = xyz.astype(np.float32)
    center = np.median(means_f32, axis=0)
    p5 = np.percentile(means_f32, 5, axis=0)
    p95 = np.percentile(means_f32, 95, axis=0)
    extent = float((p95 - p5).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m margin={RENDER_MARGIN}")

    scene = load_gsplat_ply(str(src_ply))
    _, _, eye_behind_object = compute_wall_skip(
        args.scene_dir.resolve(), means_f32)
    views = []

    for yaw_deg in YAWS_DEG:
        tag = f"y{int(round(yaw_deg))}_p{int(round(SWEEP_PITCH))}"
        V, K, eye = build_camera(center, yaw_deg, SWEEP_PITCH, distance,
                                  FOV, W, H, y_down=Y_DOWN)
        if eye_behind_object(eye):
            print(f"  [{tag}] SKIP — eye behind object "
                  f"({eye[0]:.2f},{eye[2]:.2f})")
            continue
        img = render_splat(scene, V.astype(np.float32),
                           K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
        png = diag / f"input_{tag}.png"
        Image.fromarray(img).save(png)
        views.append({"tag": tag, "V": V, "K": K,
                      "eye": eye.tolist(), "png": str(png)})
        print(f"  [{tag}] rendered")

    V, K, eye = build_camera(center, 0.0, TOPDOWN_PITCH, distance,
                              FOV, W, H, y_down=Y_DOWN)
    img = render_splat(scene, V.astype(np.float32),
                       K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
    png = diag / "input_topdown.png"
    Image.fromarray(img).save(png)
    views.append({"tag": "topdown", "V": V, "K": K,
                  "eye": eye.tolist(), "png": str(png)})
    print(f"  [topdown] rendered")

    print(f"\n[qwen] asking for '{label}' bbox in {len(views)} views")
    for vw in views:
        result = call_qwen_bbox(Path(vw["png"]), label)
        vw["found"] = result["found"]
        vw["raw"] = result["raw"]
        if result["found"]:
            bbox_px = norm_to_pixels(result["bbox_2d"], W, H)
            x0, y0, x1, y1 = bbox_px
            bw, bh = x1 - x0, y1 - y0
            px = bw * args.bbox_pad_pct
            py = bh * args.bbox_pad_pct
            top_extra = bh * TOP_PAD_PCT  # asymmetric upward pad
            bbox_padded = [
                max(0, int(x0 - px)),
                max(0, int(y0 - py - top_extra)),
                min(W, int(x1 + px)),
                min(H, int(y1 + py)),
            ]
            vw["bbox_norm"] = result["bbox_2d"]
            vw["bbox_px"] = bbox_padded
            vw["bbox_px_tight"] = bbox_px
            print(f"  [{vw['tag']}] bbox={bbox_px} (pad+{int(args.bbox_pad_pct*100)}%→{bbox_padded})")
        else:
            vw["bbox_norm"] = None
            vw["bbox_px"] = None
            print(f"  [{vw['tag']}] NOT FOUND ({result['raw'][:80]})")

        overlay = diag / f"overlay_{vw['tag']}.png"
        draw_bbox_overlay(Path(vw["png"]), vw.get("bbox_px"),
                          overlay, vw["tag"], vw["found"])

    valid_views = [vw for vw in views if vw["found"]]
    n_valid = len(valid_views)
    print(f"\n[qwen] {n_valid}/{len(views)} views returned a bbox")
    if n_valid == 0:
        sys.exit(f"[fatal] Qwen returned 0 valid bboxes — '{label}' wasn't "
                 f"recognized in any view. Check renders/diagnostics.")

    votes = np.zeros(n_total, dtype=np.int32)
    visible_count = np.zeros(n_total, dtype=np.int32)
    for vw in valid_views:
        u, v_img, in_front = project_to_pixels(xyz, vw["V"], vw["K"])
        in_img = (in_front & (u >= 0) & (u < W) &
                  (v_img >= 0) & (v_img < H))
        visible_count += in_img.astype(np.int32)
        bx0, by0, bx1, by1 = vw["bbox_px"]
        in_bbox = ((u >= bx0) & (u <= bx1) &
                   (v_img >= by0) & (v_img <= by1))
        votes += (in_img & in_bbox).astype(np.int32)

    required = int(math.ceil(args.vote_frac * n_valid))
    keep = (votes >= required) & (visible_count >= required)
    n_kept = int(keep.sum())
    print(f"\n[vote] required ≥{required}/{n_valid} views (vote-frac={args.vote_frac})")
    print(f"[vote] kept {n_kept:,}/{n_total:,} ({100*n_kept/n_total:.1f}%)")

    if n_kept == 0:
        sys.exit("[fatal] 0 splats survived voting — bbox or pad too tight?")

    out_ply = obj / "5_sweep_fallback.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    renders_dir = obj / "renders" / "5_sweep_fallback"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    (diag / "report.json").write_text(json.dumps({
        "stage": "5_sweep_fallback",
        "label": label,
        "src_ply": str(src_ply),
        "source_stage": args.source_stage,
        "n_total": n_total,
        "n_kept": n_kept,
        "kept_pct": round(100 * n_kept / n_total, 2),
        "n_views_total": len(views),
        "n_views_with_bbox": n_valid,
        "vote_frac": args.vote_frac,
        "required_votes": required,
        "bbox_pad_pct": args.bbox_pad_pct,
        "frame": {"center": center.tolist(),
                  "extent": extent, "distance": distance,
                  "fov": FOV, "width": W, "height": H,
                  "render_margin": RENDER_MARGIN},
        "views": [
            {"tag": vw["tag"], "found": vw["found"],
             "bbox_norm": vw.get("bbox_norm"),
             "bbox_px": vw.get("bbox_px"),
             "bbox_px_tight": vw.get("bbox_px_tight"),
             "qwen_raw": vw.get("raw", "")[:200]}
            for vw in views
        ],
    }, indent=2))
    print(f"[report] {diag / 'report.json'}")


if __name__ == "__main__":
    main()
