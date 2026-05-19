#!/usr/bin/env python3
"""bookshelf_sweep.py — Stage 5 for bookshelves.

Replaces bookshelf_faceon.py (single face-on bbox crop) with a multi-view
visual-hull sweep: render 4_sam_tight.ply at N yaws + topdown, ask Qwen
for a tight bbox of "the bookshelf" in each, project all splats through
each camera, vote-keep splats inside ≥K of N valid bboxes.

Removes co-planar items SAM kept (e.g. wall painting next to bookshelf)
because the painting falls outside the bookshelf bbox at every yaw — even
if it overlaps the bookshelf in face-on view, the side yaws separate them
in image space.

Pipeline:
  1. Load 4_sam_tight.ply (cleaned bookshelf + neighbours SAM kept)
  2. Render 12 yaws @ pitch -15° + 1 topdown → 13 input views
  3. Per view: Qwen "the bookshelf" bbox. Skip if not_found / edge-on
  4. Per splat: count #views where it's inside-bbox-and-visible
  5. Keep if count ≥ VOTE_FRAC × N_valid_views
  6. Save 5_bookshelf_sweep.ply + canonical 5 renders + bbox overlays

Source: 4_sam_tight.ply (NOT 1_visual_hull — we trust SAM's silhouette
and only need to remove off-axis neighbours that share the parent prompt).

Usage:
    python bookshelf_sweep.py <scene_dir> <obj_dir>
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

# Sweep cameras: 12 yaws × 1 pitch + topdown = 13 views.
# Single pitch (not 2) — adding p=-45 doubles Qwen calls without
# adding angular separation in xz; sweep is about xz separation
# of bookshelf vs neighbour. Dense 10° steps so the vote has lots
# of good-angle data; views where Qwen says found=false are skipped.
YAWS_DEG = list(range(0, 360, 10))  # 36 yaws at 10° steps
SWEEP_PITCH = -15.0
TOPDOWN_PITCH = -89.0

FOV = 70.0
W, H = 1920, 1080
RENDER_MARGIN = 2.0  # match sam_carve
Y_DOWN = True

# Qwen
QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"
QWEN_PROMPT = (
    "You are looking at one view of a single piece of furniture and "
    "possibly its surroundings.\n\n"
    "TASK: return a bounding box around THE BOOKSHELF plus ONLY the "
    "items RESTING ON it, INSIDE it, or ATTACHED to it.\n\n"
    "INCLUDE inside the bbox:\n"
    "  • The bookshelf's wooden body, frame, legs, base.\n"
    "  • Items RESTING ON TOP of the bookshelf (plants, vases, "
    "sculptures, decor).\n"
    "  • Items INSIDE the bookshelf (books on shelves, baskets, "
    "picture frames on shelves, decor objects ON the shelves).\n"
    "  • Items ATTACHED to the bookshelf (hardware, knobs).\n\n"
    "EXCLUDE — DO NOT extend the bbox to cover any of these even if "
    "they visually overlap the bookshelf in this view:\n"
    "  • Chairs, sofas, tables, stools, or any other furniture sitting "
    "IN FRONT OF the bookshelf (between the camera and the bookshelf).\n"
    "  • Furniture or objects BESIDE, BEHIND, or LEANING AGAINST the "
    "bookshelf — they are not resting ON or INSIDE it.\n"
    "  • Paintings / posters / framed prints on the wall behind or "
    "next to the bookshelf.\n"
    "  • Items sitting on the floor next to the bookshelf.\n"
    "  • Plants, lamps, or decor that stand on the floor near but not "
    "on the bookshelf.\n\n"
    "Test for each candidate item: would it stay where it is if you "
    "lifted the bookshelf off the floor? If YES → it's resting on / "
    "inside / attached to the bookshelf → include. If NO → exclude.\n\n"
    "If the bookshelf is not clearly visible in this view (e.g. you "
    "are looking at it edge-on and only see a thin slab, or it is "
    'off-frame), return {"found": false}.\n\n'
    "Otherwise return JSON in this EXACT format (coordinates in 0-1000 "
    "normalized image space, x0<x1, y0<y1):\n"
    '{"found": true, "bbox_2d": [x0, y0, x1, y1]}\n\n'
    "Output ONLY the JSON object. No commentary, no markdown."
)

# Vote
VOTE_FRAC = 0.7  # keep splat if inside ≥70% of valid bboxes
                  # AND visible in same. Mirrors sam_tight's 0.7.

# Bbox pad before voting. Side/bottom pad is OFF — we keep Qwen's bbox
# tight on those edges so neighbours adjacent to the bookshelf don't
# sneak in. Only the TOP edge is padded (asymmetric) so a plant / vase
# sitting on top of the unit isn't clipped when Qwen draws the bbox
# tight to the wooden frame.
BBOX_PAD_PCT = 0.0
TOP_PAD_PCT = 0.12


def encode_b64(p: Path, max_dim: int = 1024) -> str:
    img = Image.open(p).convert("RGB")
    s = max_dim / max(img.size)
    if s < 1.0:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_qwen_bbox(img_path: Path) -> dict:
    """Returns {"found": bool, "bbox_2d": [x0,y0,x1,y1] or None}.

    bbox_2d is in 0-1000 normalized space (Qwen's convention).
    """
    client = OpenAI(base_url=QWEN_URL, api_key="sk-x")
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{encode_b64(img_path)}"}},
        {"type": "text", "text": QWEN_PROMPT},
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
        # Regex fallback — same pattern _phase2_detect uses
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
    label = f"{tag}: {'BBOX' if found else 'NOT FOUND'}"
    if found and bbox_px:
        x0, y0, x1, y1 = bbox_px
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
    draw.rectangle([0, 0, 14 * len(label) + 8, 36], fill=color)
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--vote-frac", type=float, default=VOTE_FRAC,
                    help=f"keep splat if inside ≥this × N_valid views (default {VOTE_FRAC})")
    args = ap.parse_args()

    obj = args.obj_dir.resolve()
    src_ply = obj / "4_sam_tight.ply"
    if not src_ply.exists():
        sys.exit(f"[fatal] no 4_sam_tight.ply in {obj} — run sam_tight first")

    diag = obj / "diagnostics" / "5_bookshelf_sweep"
    diag.mkdir(parents=True, exist_ok=True)
    for f in diag.glob("*.png"):
        f.unlink()

    # Load source PLY
    pl = PlyData.read(str(src_ply))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    n_total = len(xyz)
    if n_total == 0:
        sys.exit("[fatal] source PLY is empty")
    print(f"[load] {src_ply.name}  {n_total:,} splats")

    # Frame: median centre, p5/p95 extent (same as sam_carve step1)
    means_f32 = xyz.astype(np.float32)
    center = np.median(means_f32, axis=0)
    p5 = np.percentile(means_f32, 5, axis=0)
    p95 = np.percentile(means_f32, 95, axis=0)
    extent = float((p95 - p5).max())
    tan_half = math.tan(math.radians(FOV) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m margin={RENDER_MARGIN}")

    # Render all sweep views. NO wall-skip — Qwen's found/not-found
    # response is a finer filter than geometric wall-skip and lets us
    # keep any view where the bookshelf is actually visible.
    scene = load_gsplat_ply(str(src_ply))
    views = []  # list of (tag, V, K, png_path)

    for yaw_deg in YAWS_DEG:
        tag = f"y{int(round(yaw_deg))}_p{int(round(SWEEP_PITCH))}"
        V, K, eye = build_camera(center, yaw_deg, SWEEP_PITCH, distance,
                                  FOV, W, H, y_down=Y_DOWN)
        img = render_splat(scene, V.astype(np.float32),
                           K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
        png = diag / f"input_{tag}.png"
        Image.fromarray(img).save(png)
        views.append({"tag": tag, "V": V, "K": K,
                      "eye": eye.tolist(), "png": str(png)})

    # Topdown
    V, K, eye = build_camera(center, 0.0, TOPDOWN_PITCH, distance,
                              FOV, W, H, y_down=Y_DOWN)
    img = render_splat(scene, V.astype(np.float32),
                       K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
    png = diag / "input_topdown.png"
    Image.fromarray(img).save(png)
    views.append({"tag": "topdown", "V": V, "K": K,
                  "eye": eye.tolist(), "png": str(png)})
    print(f"  [topdown] rendered")

    # Qwen bbox per view
    print(f"\n[qwen] asking for 'the bookshelf' bbox in {len(views)} views")
    for vw in views:
        result = call_qwen_bbox(Path(vw["png"]))
        vw["found"] = result["found"]
        vw["raw"] = result["raw"]
        if result["found"]:
            bbox_px = norm_to_pixels(result["bbox_2d"], W, H)
            # pad
            x0, y0, x1, y1 = bbox_px
            bw, bh = x1 - x0, y1 - y0
            px = bw * BBOX_PAD_PCT
            py = bh * BBOX_PAD_PCT
            # Extra upward pad on y_min to keep items-on-top (plant on a
            # bookshelf, lamp on a cabinet) inside the bbox even when
            # Qwen drew it tight to the wooden frame.
            top_extra = bh * TOP_PAD_PCT
            bbox_padded = [
                max(0, int(x0 - px)),
                max(0, int(y0 - py - top_extra)),
                min(W, int(x1 + px)),
                min(H, int(y1 + py)),
            ]
            vw["bbox_norm"] = result["bbox_2d"]
            vw["bbox_px"] = bbox_padded
            vw["bbox_px_tight"] = bbox_px
            print(f"  [{vw['tag']}] bbox={bbox_px} (pad+{int(BBOX_PAD_PCT*100)}%→{bbox_padded})")
        else:
            vw["bbox_norm"] = None
            vw["bbox_px"] = None
            print(f"  [{vw['tag']}] NOT FOUND ({result['raw'][:80]})")

        # Overlay
        overlay = diag / f"overlay_{vw['tag']}.png"
        draw_bbox_overlay(Path(vw["png"]), vw.get("bbox_px"),
                          overlay, vw["tag"], vw["found"])

    valid_views = [vw for vw in views if vw["found"]]
    n_valid = len(valid_views)
    print(f"\n[qwen] {n_valid}/{len(views)} views returned a bbox")
    if n_valid == 0:
        sys.exit("[fatal] Qwen returned 0 valid bboxes — the bookshelf wasn't "
                 "recognized in any view. Check renders/diagnostics.")

    # Vote: per splat, count #views where it falls inside bbox AND is visible
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
    # Splat must be (a) inside the bbox in ≥required views AND
    # (b) visible (in front of camera + on screen) in ≥required views.
    # Splats projected behind the camera at most yaws would otherwise sneak
    # in if their few visible projections happened to land in-bbox.
    keep = (votes >= required) & (visible_count >= required)
    n_kept = int(keep.sum())
    print(f"\n[vote] required ≥{required}/{n_valid} views (vote-frac={args.vote_frac})")
    print(f"[vote] kept {n_kept:,}/{n_total:,} ({100*n_kept/n_total:.1f}%)")

    if n_kept == 0:
        sys.exit("[fatal] 0 splats survived voting — bbox or pad too tight?")

    # Save
    out_ply = obj / "5_bookshelf_sweep.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    # Canonical 5
    renders_dir = obj / "renders" / "5_bookshelf_sweep"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    # Report
    (diag / "report.json").write_text(json.dumps({
        "stage": "5_bookshelf_sweep",
        "src_ply": str(src_ply),
        "n_total": n_total,
        "n_kept": n_kept,
        "kept_pct": round(100 * n_kept / n_total, 2),
        "n_views_total": len(views),
        "n_views_with_bbox": n_valid,
        "vote_frac": args.vote_frac,
        "required_votes": required,
        "bbox_pad_pct": BBOX_PAD_PCT,
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
