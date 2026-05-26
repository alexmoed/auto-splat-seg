#!/usr/bin/env python3
"""render_demo_preview.py — sam_tight demo: per-camera frustum to its SAM mask.

For every camera that HAS a SAM mask:
  - the mask is drawn as a textured quad in front of the camera
  - a frustum is drawn as 4 lines from the camera eye to the 4 CORNERS
    of that mask — the frustum ends exactly at the mask edge.
A camera with no mask is not drawn at all.
Plus the chair point cloud.

8 cameras (the outer pitch-15 ring is excluded).

Usage:
    python render_demo_preview.py <obj_dir> [--padded]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import load_point_cloud, camera_geometry  # noqa: E402

MASK_FRAC = 0.36     # mask sits this fraction of the eye->chair distance out
CAM_PULLBACK = 1.6   # push cameras back along their ray (longer frustum)
FRUSTUM_LEN = 1.5    # apex fraction of mask->eye (>1.0 = camera farther back)


def to_plot(p):
    """Scene is y-down; map to an upright matplotlib frame (Z = -y)."""
    p = np.asarray(p)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--padded", action="store_true")
    args = ap.parse_args()
    obj = args.obj_dir.resolve()
    diag = obj / "diagnostics" / "4_sam_tight"
    out_dir = Path(__file__).parent / "preview"
    out_dir.mkdir(exist_ok=True)
    kind = "padded" if args.padded else "raw"
    mtag = "mask_padded" if args.padded else "mask"

    # ---- point cloud ----------------------------------------------------
    xyz, rgb = load_point_cloud(obj / "3_floor_drop.ply")
    if len(xyz) > 11000:
        idx = np.random.default_rng(0).choice(len(xyz), 11000, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    pc = to_plot(xyz)
    pc_col = rgb.astype(np.float32) / 255.0

    # ---- cameras: keep only those WITH a mask (outer pitch-15 ring out) --
    cam_data = json.load(open(diag / "cameras.json"))
    cams = []
    for c in cam_data["cameras"]:
        mp = diag / f"{mtag}_{c['tag']}.png"
        if not mp.exists():                    # no mask -> don't show camera
            continue
        if np.array(Image.open(mp).convert("L")).max() < 80:   # empty mask
            continue
        g = camera_geometry(c)
        # pull the camera back along its view ray: longer frustum, larger
        # mask, bigger gap to the chair (half-extents scale with distance)
        eye0 = np.array(g["eye"]); fwd0 = np.array(g["forward"])
        tgt = eye0 + g["depth"] * fwd0
        g["eye"] = (tgt - g["depth"] * CAM_PULLBACK * fwd0).tolist()
        g["depth"] *= CAM_PULLBACK
        g["half_w"] *= CAM_PULLBACK
        g["half_h"] *= CAM_PULLBACK
        g["mask_png"] = mp
        g["input_png"] = diag / f"input_{c['tag']}.png"   # camera's render
        # tight bbox of the silhouette (white pixels) — frame coords:
        #   bbox_px = (x0,x1,y0,y1) pixels; bbox = (u0,u1,v0,v1) in [-1,1]
        mimg = np.array(Image.open(mp).convert("L"))
        H, W = mimg.shape
        ys, xs = np.where(mimg > 80)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        g["bbox_px"] = (x0, x1, y0, y1)
        g["bbox"] = (2 * x0 / W - 1, 2 * x1 / W - 1,
                     1 - 2 * y1 / H, 1 - 2 * y0 / H)
        cams.append(g)
    n = len(cams)
    cmap = plt.get_cmap("hsv")
    for i, g in enumerate(cams):
        g["color"] = cmap(i / n)
    print(f"[cams] {n} cameras with a mask")

    def mask_corners(g):
        """Eye + the 4 world-space corners of this camera's mask quad."""
        eye = np.array(g["eye"]); fwd = np.array(g["forward"])
        right = np.array(g["right"]); up = np.array(g["up"])
        cen = eye + MASK_FRAC * g["depth"] * fwd
        hw = g["half_w"] * MASK_FRAC
        hh = g["half_h"] * MASK_FRAC
        u0, u1, v0, v1 = g["bbox"]            # hug the silhouette bbox
        cor = [cen + u * hw * right + v * hh * up
               for u, v in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]]
        return eye, cen, cor

    def mask_surface(g, tw=300):
        """The SAM mask as a textured quad on the mask plane."""
        eye = np.array(g["eye"]); fwd = np.array(g["forward"])
        right = np.array(g["right"]); up = np.array(g["up"])
        cen = eye + MASK_FRAC * g["depth"] * fwd
        hw = g["half_w"] * MASK_FRAC
        hh = g["half_h"] * MASK_FRAC
        x0, x1, y0, y1 = g["bbox_px"]         # crop to the silhouette bbox
        m = np.array(Image.open(g["mask_png"]).convert("L"))[y0:y1 + 1,
                                                             x0:x1 + 1]
        th = max(2, round(tw * m.shape[0] / m.shape[1]))
        m = np.array(Image.fromarray(m).resize((tw, th)))
        u0, u1, v0, v1 = g["bbox"]
        uu, vv = np.meshgrid(np.linspace(u0, u1, tw),
                             np.linspace(v1, v0, th))
        P = (cen[None, None, :]
             + (uu * hw)[..., None] * right[None, None, :]
             + (vv * hh)[..., None] * up[None, None, :])
        Pp = to_plot(P)
        fc = np.zeros((th, tw, 4), dtype=np.float32)
        ip = g.get("input_png")
        if ip and ip.exists():
            # the camera's actual render, cut out by the SAM mask
            im = np.array(Image.open(ip).convert("RGB"))[y0:y1 + 1,
                                                         x0:x1 + 1]
            fc[..., :3] = np.asarray(
                Image.fromarray(im).resize((tw, th)),
                dtype=np.float32) / 255.0
        else:
            fc[..., :3] = np.array(g["color"][:3], dtype=np.float32)
        fc[..., 3] = np.where(m > 80, 0.85, 0.0)   # mostly opaque
        return Pp[..., 0], Pp[..., 1], Pp[..., 2], fc

    def camera_icon(g, s, eye=None):
        """Wireframe camera at `eye` (defaults to g['eye']): a rectangular
        box body + a cone lens with the WIDE end facing the mask. Returns
        line segments (pairs of points) in plot coords."""
        if eye is None:
            eye = np.array(g["eye"])
        fwd = np.array(g["forward"])
        right = np.array(g["right"]); up = np.array(g["up"])
        wx, wy, wz = s * 1.15, s * 0.85, s * 1.0   # rectangular (wider than tall)
        C = {(sx, sy, sz): eye + sx * wx * right + sy * wy * up + sz * wz * fwd
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
        segs = []
        for sy in (-1, 1):
            for sz in (-1, 1):
                segs.append((C[(-1, sy, sz)], C[(1, sy, sz)]))
        for sx in (-1, 1):
            for sz in (-1, 1):
                segs.append((C[(sx, -1, sz)], C[(sx, 1, sz)]))
        for sx in (-1, 1):
            for sy in (-1, 1):
                segs.append((C[(sx, sy, -1)], C[(sx, sy, 1)]))
        # cone lens — narrow apex at the body, WIDE ring out toward the mask
        apex = eye + wz * fwd
        ring_c = eye + (wz + s * 1.7) * fwd
        R = s * 1.25
        ring = [ring_c + R * (np.cos(a) * right + np.sin(a) * up)
                for a in np.linspace(0, 2 * np.pi, 13)[:-1]]
        for k in range(len(ring)):
            segs.append((apex, ring[k]))
            segs.append((ring[k], ring[(k + 1) % len(ring)]))
        return [(to_plot(a), to_plot(b)) for a, b in segs]

    # ---- bounds ---------------------------------------------------------
    allp = [pc] + [to_plot(np.array(g["eye"]))[None, :] for g in cams]
    allp = np.vstack(allp)
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    span = (hi - lo).max() / 2 * 1.05

    def draw(ax, zoom=1.0):
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc_col, s=2.0,
                   depthshade=False, linewidths=0)
        for g in cams:
            col = g["color"]
            eye, cen, cor = mask_corners(g)
            apex = cen + (eye - cen) * FRUSTUM_LEN     # shorter frustum
            e = to_plot(apex)
            # frustum: 4 lines apex -> mask corners
            for k in range(4):
                b = to_plot(cor[k])
                ax.plot([e[0], b[0]], [e[1], b[1]], [e[2], b[2]],
                        color=col, lw=1.0, alpha=0.75)
            # mask edge rectangle (the far end of the frustum)
            for k in range(4):
                a = to_plot(cor[k]); b = to_plot(cor[(k + 1) % 4])
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color=col, lw=1.0, alpha=0.75)
            # little wireframe camera icon at the apex
            for a, b in camera_icon(g, span * 0.026, eye=apex):
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color=col, lw=0.9, alpha=0.95)
            # the mask itself
            X, Y, Z, fc = mask_surface(g)
            ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                            shade=False, linewidth=0, antialiased=False)
        s = span * zoom
        ax.set_xlim(ctr[0] - s, ctr[0] + s)
        ax.set_ylim(ctr[1] - s, ctr[1] + s)
        ax.set_zlim(ctr[2] - s, ctr[2] + s)
        ax.set_box_aspect((1, 1, 1))
        ax.set_facecolor("#0d0f12")
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor("#0d0f12")
            a.pane.set_edgecolor("#2c313a")
            a.line.set_color("#2c313a")
            a.set_tick_params(colors="#5a626e")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    for name, elev, azim in [("all_cameras", 20, -62),
                             ("all_cameras_high", 42, -72)]:
        fig = plt.figure(figsize=(11, 10), facecolor="#0d0f12")
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        draw(ax, zoom=0.98)
        ax.view_init(elev=elev, azim=azim)
        fig.text(0.5, 0.965,
                 f"sam_tight · grey armchair · {n} cameras · {kind}",
                 color="#dfe3e8", fontsize=13, ha="center")
        out = out_dir / f"{name}_{kind}.png"
        fig.savefig(out, dpi=120, facecolor="#0d0f12")
        plt.close(fig)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
