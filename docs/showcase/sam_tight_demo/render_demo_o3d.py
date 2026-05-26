#!/usr/bin/env python3
"""render_demo_o3d.py — high-res offscreen preview of the sam_tight demo.

matplotlib can't texture a 3D plane crisply; Open3D's OffscreenRenderer
rasterizes the FULL-RESOLUTION SAM mask PNGs as real GPU textures.

Scene:
  - the armchair point cloud (3_floor_drop.ply)
  - each selected camera's projection cone (LineSet) + eye marker
  - each camera's original SAM mask as a full-res RGBA-textured quad
    near the camera apex (hue silhouette, transparent elsewhere)

Usage:
    python render_demo_o3d.py <obj_dir> [--padded] [--frac 0.27]
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import load_point_cloud, camera_geometry  # noqa: E402

TEX_W = 1280            # mask texture width (downscaled from 1920, LANCZOS)


def hue_rgb(i, n):
    return colorsys.hls_to_rgb(i / n, 0.62, 0.80)


def mask_quad(cen, right, up, hw, hh, mask_png, rgb):
    """A camera-facing quad textured with the SAM mask (full res, soft
    alpha). Returns (TriangleMesh, MaterialRecord)."""
    c = [cen - hw * right - hh * up,    # 0 bottom-left
         cen + hw * right - hh * up,    # 1 bottom-right
         cen + hw * right + hh * up,    # 2 top-right
         cen - hw * right + hh * up]    # 3 top-left
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(c))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3]]))
    # uv (0,0) = image bottom-left; per-triangle-vertex (6 entries)
    uv = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.array([uv[0], uv[1], uv[2], uv[0], uv[2], uv[3]], dtype=float))
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])

    m = Image.open(mask_png).convert("L")
    th = round(TEX_W * m.height / m.width)
    m = m.resize((TEX_W, th), Image.LANCZOS)
    alpha = np.array(m, dtype=np.uint8)
    rgba = np.zeros((th, TEX_W, 4), dtype=np.uint8)
    rgba[..., 0] = int(rgb[0] * 255)
    rgba[..., 1] = int(rgb[1] * 255)
    rgba[..., 2] = int(rgb[2] * 255)
    rgba[..., 3] = alpha
    mesh.textures = [o3d.geometry.Image(rgba)]
    mesh.compute_vertex_normals()

    # defaultUnlit honours a 4-channel albedo's alpha as a cutout — the
    # transparent area shows what's behind. (defaultUnlitTransparency
    # panics on a missing "srgbColor" uniform in this Open3D 0.19 build.)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.albedo_img = o3d.geometry.Image(rgba)
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    return mesh, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--padded", action="store_true")
    ap.add_argument("--frac", type=float, default=0.27)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1440)
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    diag = obj / "diagnostics" / "4_sam_tight"
    out_dir = Path(__file__).parent / "preview"
    out_dir.mkdir(exist_ok=True)
    frac = args.frac
    kind = "padded" if args.padded else "raw"

    # ---- point cloud ----------------------------------------------------
    xyz, rgb = load_point_cloud(obj / "3_floor_drop.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    # ---- cameras (selected views only) ----------------------------------
    cam_data = json.load(open(diag / "cameras.json"))
    rep = diag / "report.json"
    selected = ({v["tag"] for v in json.load(open(rep)).get("views", [])}
                if rep.exists() else
                {c["tag"] for c in cam_data["cameras"]})
    # drop the OUTER ring (pitch -15deg cameras); keep inner ring + topdown
    kept = [c for c in cam_data["cameras"]
            if c["tag"] in selected and not c["tag"].endswith("p-15")]
    n = len(kept)
    print(f"[select] {n}/{len(cam_data['cameras'])} views "
          f"(outer pitch-15 ring removed)")

    mtag = "mask_padded" if args.padded else "mask"
    line_pts, line_idx, line_col = [], [], []
    quads = []          # (name, mesh, mat)
    # Textbook frustum per camera: viewpoint + near clip plane + far
    # clip plane. The SAM mask is BIG on the far clip plane so it's
    # clearly visible; masks are alpha-composited so overlaps + the
    # chair show through.
    NEAR_FRAC = 0.12    # near clip plane — fraction of eye->chair distance
    FAR_FRAC = 0.42     # far clip plane — the wide end, carries the mask
    NARROW = 0.45       # fake a tighter FOV — frustum LINES only, not masks
    for i, c in enumerate(kept):
        g = camera_geometry(c)
        col = hue_rgb(i, n)
        eye = np.array(g["eye"])
        fwd = np.array(g["forward"])
        right = np.array(g["right"]); up = np.array(g["up"])
        depth = g["depth"]
        fhw, fhh = g["half_w"] * NARROW, g["half_h"] * NARROW  # frustum — narrowed
        hw, hh = g["half_w"], g["half_h"]                       # mask — full size

        def rect(t, eye=eye, fwd=fwd, right=right, up=up,
                 depth=depth, hw=fhw, hh=fhh):
            """Frustum cross-section corners at distance t*depth."""
            cen = eye + t * depth * fwd
            sw, sh = hw * t, hh * t
            return [cen + sx * sw * right + sy * sh * up
                    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]

        near, far = rect(NEAR_FRAC), rect(FAR_FRAC)
        # textbook viewing frustum: viewpoint + near clip plane + far
        # clip plane + 4 side edges from the viewpoint through to far.
        base = len(line_pts)
        line_pts.append(eye)                 # 0   viewpoint
        line_pts.extend(near)                # 1-4 near clip plane
        line_pts.extend(far)                 # 5-8 far clip plane
        for k in range(4):                   # side edges viewpoint->far
            line_idx.append([base, base + 5 + k]); line_col.append(col)
        for k in range(4):                   # near clip plane rectangle
            line_idx.append([base + 1 + k, base + 1 + (k + 1) % 4])
            line_col.append(col)
        for k in range(4):                   # far clip plane rectangle
            line_idx.append([base + 5 + k, base + 5 + (k + 1) % 4])
            line_col.append(col)
        # SAM mask textured BIG on the FAR clip plane (the wide end)
        cen_f = eye + FAR_FRAC * depth * fwd
        mesh, mat = mask_quad(cen_f, right, up, hw * FAR_FRAC,
                              hh * FAR_FRAC,
                              diag / f"{mtag}_{c['tag']}.png", col)
        quads.append((f"mask_{i}", mesh, mat))

    cones = o3d.geometry.LineSet()
    cones.points = o3d.utility.Vector3dVector(np.array(line_pts))
    cones.lines = o3d.utility.Vector2iVector(np.array(line_idx))
    cones.colors = o3d.utility.Vector3dVector(np.array(line_col))

    # ---- offscreen renderer --------------------------------------------
    # The masks are opaque (defaultUnlitTransparency panics in this
    # Open3D build), so render TWO passes and alpha-composite:
    #   A — point cloud + frustum lines         (the scene)
    #   B — only the mask quads on pure black   (keyed for compositing)
    # then blend B over A so the masks read as semi-transparent.
    renderer = rendering.OffscreenRenderer(args.width, args.height)
    scene = renderer.scene
    scene.view.set_post_processing(False)

    ctr = xyz.mean(axis=0).astype(np.float64)
    ext = float(np.ptp(xyz, axis=0).max())
    up_w = np.array([0.0, -1.0, 0.0])          # scene is y-down
    shots = {
        "iso":  ctr + np.array([2.3, -1.5, 2.7]) * ext,
        "high": ctr + np.array([1.1, -3.1, 1.5]) * ext,
        "front": ctr + np.array([0.2, -0.8, 3.2]) * ext,
    }
    MASK_OPACITY = 0.62

    def render_shots():
        out = {}
        for name, eye in shots.items():
            renderer.setup_camera(50.0, ctr, eye, up_w)
            out[name] = np.asarray(renderer.render_to_image()).astype(np.float32)
        return out

    # pass A — scene
    scene.set_background([0.051, 0.059, 0.071, 1.0])
    pc_mat = rendering.MaterialRecord()
    pc_mat.shader = "defaultUnlit"
    pc_mat.point_size = 3.5
    scene.add_geometry("cloud", pcd, pc_mat)
    ln_mat = rendering.MaterialRecord()
    ln_mat.shader = "unlitLine"
    ln_mat.line_width = 1.2
    scene.add_geometry("cones", cones, ln_mat)
    base = render_shots()

    # pass B — masks only, on pure black
    scene.clear_geometry()
    scene.set_background([0.0, 0.0, 0.0, 1.0])
    for name, mesh, mat in quads:
        scene.add_geometry(name, mesh, mat)
    masks = render_shots()

    # composite
    for name in shots:
        b = base[name]
        m = masks[name]
        a = (m.max(axis=2) > 14).astype(np.float32)[..., None] * MASK_OPACITY
        comp = np.clip(b * (1 - a) + m * a, 0, 255).astype(np.uint8)
        out = out_dir / f"o3d_{name}_{kind}.png"
        Image.fromarray(comp).save(out)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
