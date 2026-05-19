#!/usr/bin/env python3
"""rug_extract.py — Stage for rugs / area rugs / mats.

Rugs are FLAT — sam_carve's 25-yaw silhouette voting fails because a
rug has no silhouette in any side view. So this stage uses a single
TOPDOWN view + xz-bbox cone + y-band cut + on-top-item subtraction.

Pipeline:
  1. Load 1_visual_hull.ply (the bbox-cropped scene region)
  2. Render topdown at FOV=25 (narrow — wide FOV distorts rug edges)
  3. Qwen tight bbox of "the rug" / inventory label
  4. Project all splats through topdown camera; keep splats inside
     the (padded) bbox.
  5. Y-band cut: keep only splats within rug_band_m of the inferred
     floor plane (95th-percentile y in y-down). Drops coffee tables /
     furniture sitting above the rug.
  6. Auto-subtract: for each sibling 02_*/<latest_stage>.ply whose
     xz centroid falls inside the rug's xz bbox, KDTree-radius drop
     any rug splat within radius_m of those object splats. Removes
     coffee-table legs / floor lamps / sub-items resting on the rug.
  7. Save 4_rug.ply + renders/4_rug/{y0,y90,y180,y270,topdown}.png

Usage:
    python rug_extract.py <scene_dir> <obj_dir>
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
from scipy.spatial import cKDTree

ITERATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ITERATION_DIR))
from extract_one import viewmat_look_at, build_K, project_to_pixels  # noqa: E402

sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
from view import load_gsplat_ply, render_splat  # noqa: E402

from sam_carve import build_camera, render_canonical_5  # noqa: E402

FOV_TOPDOWN = 25.0   # narrow — wide FOV warps rug edges
W, H = 1920, 1080
RENDER_MARGIN = 1.5  # tighter framing for topdown
TOPDOWN_PITCH = -89.0
Y_DOWN = True

QWEN_URL = "http://127.0.0.1:8000/v1"
QWEN_MODEL = "qwen36-awq"

BBOX_PAD_PCT = 0.05            # 5% — matches the lr2 rug recipe
RUG_BAND_M = 0.10              # keep splats within 10cm above the rug surface
SUBTRACT_RADIUS_M = 0.05       # KDTree radius for on-top-item subtraction


def qwen_prompt(label: str) -> str:
    return (
        f"You are looking at a TOPDOWN view (camera looking straight "
        f"down at the floor). The inventory label is: '{label}'.\n\n"
        f"TASK: return a TIGHT pixel bounding box around the rug/mat "
        f"covering the floor. Items resting ON the rug (coffee tables, "
        f"chairs, cushions, etc.) count as on top of the rug — so the "
        f"bbox should still cover the full rug rectangle including the "
        f"areas hidden under those items.\n\n"
        f"If you cannot identify a rug in this view, return "
        f'{{"found": false}}.\n\n'
        f"Otherwise return JSON in this EXACT format (coordinates in "
        f"0-1000 normalized image space, x0<x1, y0<y1):\n"
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


def find_latest_stage_ply(obj_dir: Path) -> Path | None:
    """Pick the latest PLY a sibling object has produced. Prefer
    later stages (more refined)."""
    for stage in ("5_sweep_fallback", "5_bookshelf_sweep",
                   "4_sam_tight", "4_rug",
                   "3_floor_drop", "2_sam_wide"):
        p = obj_dir / f"{stage}.ply"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--source-stage", default="1_visual_hull",
                    help="which stage PLY to source from (default 1_visual_hull)")
    ap.add_argument("--bbox-pad-pct", type=float, default=BBOX_PAD_PCT)
    ap.add_argument("--rug-band-m", type=float, default=RUG_BAND_M)
    ap.add_argument("--subtract-radius-m", type=float, default=SUBTRACT_RADIUS_M)
    ap.add_argument("--no-subtract", action="store_true",
                    help="skip on-top-item subtraction")
    args = ap.parse_args()

    scene = args.scene_dir.resolve()
    obj = args.obj_dir.resolve()
    src_ply = obj / f"{args.source_stage}.ply"
    if not src_ply.exists():
        sys.exit(f"[fatal] no {args.source_stage}.ply in {obj}")

    label = "rug"
    meta_path = obj / "1_visual_hull_meta.json"
    if meta_path.exists():
        try:
            label = json.load(open(meta_path)).get("label", "rug")
        except Exception:
            pass

    diag = obj / "diagnostics" / "4_rug"
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
    tan_half = math.tan(math.radians(FOV_TOPDOWN) / 2)
    distance = (extent * RENDER_MARGIN) / (2 * tan_half)
    print(f"[frame] center={center.tolist()} extent={extent:.2f}m "
          f"dist={distance:.2f}m fov={FOV_TOPDOWN}° margin={RENDER_MARGIN}")

    # 1) Topdown render at FOV=25
    scene_render = load_gsplat_ply(str(src_ply))
    V, K, eye = build_camera(center, 0.0, TOPDOWN_PITCH, distance,
                              FOV_TOPDOWN, W, H, y_down=Y_DOWN)
    img = render_splat(scene_render, V.astype(np.float32),
                        K.astype(np.float32), W, H, bg=(1.0, 1.0, 1.0))
    topdown_png = diag / "input_topdown.png"
    Image.fromarray(img).save(topdown_png)
    print(f"[render] topdown FOV={FOV_TOPDOWN}° → {topdown_png.name}")

    # 2) Qwen bbox
    print(f"\n[qwen] asking for '{label}' bbox in topdown")
    result = call_qwen_bbox(topdown_png, label)
    if not result["found"]:
        sys.exit(f"[fatal] Qwen didn't find a rug in topdown: "
                 f"{result['raw'][:120]}")

    bbox_px = norm_to_pixels(result["bbox_2d"], W, H)
    x0, y0, x1, y1 = bbox_px
    bw, bh = x1 - x0, y1 - y0
    px = bw * args.bbox_pad_pct
    py = bh * args.bbox_pad_pct
    bbox_padded = [
        max(0, int(x0 - px)),
        max(0, int(y0 - py)),
        min(W, int(x1 + px)),
        min(H, int(y1 + py)),
    ]
    print(f"[qwen] bbox={bbox_px} (pad+{int(args.bbox_pad_pct*100)}%→{bbox_padded})")

    # Overlay
    img_pil = Image.open(topdown_png).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle(bbox_px, outline=(80, 220, 100), width=4)
    draw.rectangle(bbox_padded, outline=(220, 220, 60), width=2)
    draw.text((20, 20), f"rug bbox (green=tight, yellow=padded)",
              fill=(40, 40, 40), font=font)
    (diag / "overlay_topdown.png").write_bytes(b"")
    img_pil.save(diag / "overlay_topdown.png")

    # 3) Project all splats through topdown camera
    u, v_img, in_front = project_to_pixels(xyz, V, K)
    in_img = (in_front & (u >= 0) & (u < W) &
              (v_img >= 0) & (v_img < H))
    bx0, by0, bx1, by1 = bbox_padded
    in_bbox = ((u >= bx0) & (u <= bx1) &
               (v_img >= by0) & (v_img <= by1))
    keep_bbox = in_img & in_bbox
    n_in_bbox = int(keep_bbox.sum())
    print(f"\n[bbox-cone] {n_in_bbox:,}/{n_total:,} splats in bbox cone")

    if n_in_bbox == 0:
        sys.exit("[fatal] 0 splats inside topdown bbox — bbox or pad too tight")

    # 4) Y-band cut (rug surface ≈ 95th percentile of y in y-down)
    bbox_y = xyz[keep_bbox, 1]
    floor_y = np.percentile(bbox_y, 95)
    y_min_keep = floor_y - args.rug_band_m   # higher in image (above rug surface)
    y_max_keep = floor_y + 0.05              # below rug — small tolerance
    keep_y = (xyz[:, 1] >= y_min_keep) & (xyz[:, 1] <= y_max_keep)
    keep_band = keep_bbox & keep_y
    n_in_band = int(keep_band.sum())
    print(f"[y-band] floor_y={floor_y:.3f}, band=[{y_min_keep:.3f}, "
          f"{y_max_keep:.3f}]  →  {n_in_band:,} splats")

    if n_in_band == 0:
        sys.exit("[fatal] 0 splats survived y-band cut")

    keep = keep_band

    # 5) Auto-subtract on-top items
    n_subtracted = 0
    siblings_subtracted = []
    if not args.no_subtract:
        # Rug xz bbox from kept splats
        rug_xz = xyz[keep][:, [0, 2]] if keep.any() else None
        rug_xz_min = rug_xz.min(axis=0) if rug_xz is not None else None
        rug_xz_max = rug_xz.max(axis=0) if rug_xz is not None else None
        if rug_xz_min is not None:
            print(f"\n[subtract] rug xz bbox: x∈[{rug_xz_min[0]:.2f},"
                  f"{rug_xz_max[0]:.2f}]  z∈[{rug_xz_min[1]:.2f},"
                  f"{rug_xz_max[1]:.2f}]")
            kept_xyz = xyz[keep]
            tree = cKDTree(kept_xyz)
            for sib_dir in sorted(scene.glob("02_*")):
                if sib_dir.resolve() == obj:
                    continue
                sib_ply = find_latest_stage_ply(sib_dir)
                if sib_ply is None:
                    continue
                try:
                    sp = PlyData.read(str(sib_ply))
                    sv = sp["vertex"]
                    sxyz = np.stack([sv["x"], sv["y"], sv["z"]],
                                     axis=1).astype(np.float64)
                except Exception:
                    continue
                # Centroid xz check
                cxz = sxyz[:, [0, 2]].mean(axis=0)
                if not (rug_xz_min[0] <= cxz[0] <= rug_xz_max[0] and
                        rug_xz_min[1] <= cxz[1] <= rug_xz_max[1]):
                    continue
                # Subtract: for each kept_xyz point, find if any sibling
                # splat is within radius
                d, _ = cKDTree(sxyz).query(kept_xyz, k=1,
                                            distance_upper_bound=args.subtract_radius_m)
                hits = d < args.subtract_radius_m
                if hits.any():
                    n_hit = int(hits.sum())
                    n_subtracted += n_hit
                    siblings_subtracted.append({
                        "sib": sib_dir.name,
                        "ply": sib_ply.name,
                        "n_subtracted": n_hit,
                    })
                    print(f"  [subtract] {sib_dir.name}/{sib_ply.name}: "
                          f"removed {n_hit:,} rug splats within "
                          f"{args.subtract_radius_m}m")
                    # Update keep mask
                    kept_idx = np.where(keep)[0]
                    keep[kept_idx[hits]] = False
                    kept_xyz = xyz[keep]
                    if not keep.any():
                        break
                    tree = cKDTree(kept_xyz)

    n_kept = int(keep.sum())
    print(f"\n[final] kept {n_kept:,}/{n_total:,} splats "
          f"({100*n_kept/n_total:.1f}%)  subtracted={n_subtracted:,}")
    if n_kept == 0:
        sys.exit("[fatal] 0 splats after subtraction — something's off")

    # 6) Save
    out_ply = obj / "4_rug.ply"
    PlyData([PlyElement.describe(v.data[keep], "vertex")],
            text=False).write(str(out_ply))
    print(f"[save] {out_ply}")

    renders_dir = obj / "renders" / "4_rug"
    render_canonical_5(out_ply, renders_dir)
    print(f"[render] canonical 5 → {renders_dir}")

    # 7) Report
    (diag / "report.json").write_text(json.dumps({
        "stage": "4_rug",
        "label": label,
        "src_ply": str(src_ply),
        "n_total": n_total,
        "n_in_bbox": n_in_bbox,
        "n_in_band": n_in_band,
        "n_subtracted": n_subtracted,
        "n_kept": n_kept,
        "kept_pct": round(100 * n_kept / n_total, 2),
        "topdown_fov": FOV_TOPDOWN,
        "bbox_norm": result["bbox_2d"],
        "bbox_px_tight": bbox_px,
        "bbox_px_padded": bbox_padded,
        "bbox_pad_pct": args.bbox_pad_pct,
        "rug_band_m": args.rug_band_m,
        "floor_y": float(floor_y),
        "y_band": [float(y_min_keep), float(y_max_keep)],
        "subtract_radius_m": args.subtract_radius_m,
        "siblings_subtracted": siblings_subtracted,
        "frame": {"center": center.tolist(),
                  "extent": extent, "distance": distance,
                  "fov": FOV_TOPDOWN, "width": W, "height": H,
                  "render_margin": RENDER_MARGIN},
    }, indent=2))
    print(f"[report] {diag / 'report.json'}")


if __name__ == "__main__":
    main()
