#!/usr/bin/env python3
"""build_demo.py — self-contained Three.js web demo of Stage 4 (sam_tight).

Matches render_demo_preview.py exactly:
  - 15 cameras, each pulled back along its ray (CAM_PULLBACK)
  - each camera's SAM mask shown as the IMAGE CUT OUT by the mask
    (the camera's real render, clipped to the silhouette), translucent
  - mask quad + frustum hug the silhouette's tight bbox (no dead margin)
  - frustum apex pushed back past the eye (FRUSTUM_LEN); frustum runs
    apex -> 4 mask-bbox corners
  - a little WIREFRAME camera icon at each apex (rectangular box body +
    cone lens, wide end facing the mask)
  - the chair point cloud

Everything (point cloud, geometry, mask cut-outs) is base64-embedded
into one index.html. Three.js loads from a CDN.

Usage:
    python build_demo.py <obj_dir> [--out <dir>] [--ply 3_floor_drop.ply]
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

SH_C0 = 0.28209479177387814
MAX_POINTS = 140_000

MASK_FRAC = 0.36       # mask sits this fraction of the eye->chair distance
CAM_PULLBACK = 1.6     # push cameras back along their ray
FRUSTUM_LEN = 1.5      # apex fraction of mask->eye (>1 = camera farther back)
MASK_ALPHA = 0.85      # mask cut-out opacity
TEX_W = 360            # cut-out texture width


def load_point_cloud(ply_path: Path):
    """Return (xyz float32 [N,3], rgb uint8 [N,3]) from a gsplat PLY."""
    pl = PlyData.read(str(ply_path))
    v = pl["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    names = set(v.data.dtype.names)
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    elif {"red", "green", "blue"} <= names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    else:
        rgb = np.full((len(xyz), 3), 0.7, dtype=np.float32)
    rgb = (rgb * 255.0).astype(np.uint8)
    if len(xyz) > MAX_POINTS:
        idx = np.random.default_rng(42).choice(len(xyz), MAX_POINTS,
                                                replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def camera_geometry(cam: dict):
    """From a cameras.json entry, derive the world-space camera basis."""
    V = np.array(cam["V"], dtype=np.float64)
    K = np.array(cam["K"], dtype=np.float64)
    eye = np.array(cam["eye"], dtype=np.float64)
    target = np.array(cam["target"], dtype=np.float64)
    W, H = int(cam["width"]), int(cam["height"])

    R = V[:3, :3]
    right = R[0, :]
    up = -R[1, :]
    fwd = target - eye
    depth = float(np.linalg.norm(fwd))
    fwd = fwd / max(depth, 1e-9)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    half_w = (W / 2.0) * depth / fx
    half_h = (H / 2.0) * depth / fy
    off_r = (cx - W / 2.0) * depth / fx
    off_u = -(cy - H / 2.0) * depth / fy
    center = eye + depth * fwd + off_r * right + off_u * up

    return {
        "tag": cam["tag"],
        "eye": eye.tolist(),
        "forward": fwd.tolist(),
        "center": center.tolist(),
        "right": right.tolist(),
        "up": up.tolist(),
        "half_w": half_w,
        "half_h": half_h,
        "depth": depth,
    }


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def hue_rgb(i, n):
    import colorsys
    return list(colorsys.hls_to_rgb(i / n, 0.6, 0.85))


def camera_icon_segs(eye, fwd, right, up, s):
    """Wireframe camera at `eye`: rectangular box body + cone lens, wide
    end facing `fwd`. Returns a flat list of points (pairs = segments)."""
    wx, wy, wz = s * 1.15, s * 0.85, s * 1.0
    C = {(sx, sy, sz): eye + sx * wx * right + sy * wy * up + sz * wz * fwd
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
    segs = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            segs += [C[(-1, sy, sz)], C[(1, sy, sz)]]
    for sx in (-1, 1):
        for sz in (-1, 1):
            segs += [C[(sx, -1, sz)], C[(sx, 1, sz)]]
    for sx in (-1, 1):
        for sy in (-1, 1):
            segs += [C[(sx, sy, -1)], C[(sx, sy, 1)]]
    apex = eye + wz * fwd
    ring_c = eye + (wz + s * 1.7) * fwd
    R = s * 1.25
    ring = [ring_c + R * (np.cos(a) * right + np.sin(a) * up)
            for a in np.linspace(0, 2 * np.pi, 13)[:-1]]
    for k in range(len(ring)):
        segs += [apex, ring[k]]
        segs += [ring[k], ring[(k + 1) % len(ring)]]
    return [p.tolist() for p in segs]


def cutout_data_uri(input_png: Path, mask_png: Path, bbox_px):
    """The camera render cropped to the silhouette bbox, with the SAM
    mask as its alpha channel. Returns a PNG data URI."""
    x0, x1, y0, y1 = bbox_px
    im = np.array(Image.open(input_png).convert("RGB"))[y0:y1 + 1, x0:x1 + 1]
    m = np.array(Image.open(mask_png).convert("L"))[y0:y1 + 1, x0:x1 + 1]
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = im
    rgba[..., 3] = np.where(m > 80, 255, 0)
    img = Image.fromarray(rgba, mode="RGBA")
    tw = TEX_W
    th = max(2, round(tw * h / w))
    img = img.resize((tw, th), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + b64(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ply", default="3_floor_drop.ply")
    args = ap.parse_args()

    obj = args.obj_dir.resolve()
    out_dir = (args.out or Path(__file__).parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = obj / args.ply
    diag = obj / "diagnostics" / "4_sam_tight"
    cam_json = diag / "cameras.json"
    for p in (ply_path, cam_json):
        if not p.exists():
            sys.exit(f"[fatal] missing {p}")

    print(f"[load] point cloud {ply_path}")
    xyz, rgb = load_point_cloud(ply_path)
    print(f"[load] {len(xyz):,} points")

    cam_data = json.load(open(cam_json))
    extent = float(cam_data.get("extent", 1.3))
    icon_s = extent * 0.05

    # cameras that have a non-empty mask
    raw = []
    for cam in cam_data["cameras"]:
        tag = cam["tag"]
        mp = diag / f"mask_{tag}.png"
        ip = diag / f"input_{tag}.png"
        if not mp.exists() or not ip.exists():
            continue
        if np.array(Image.open(mp).convert("L")).max() < 80:
            continue
        raw.append((cam, mp, ip))
    n = len(raw)
    print(f"[load] {n} cameras with a mask")

    cams = []
    for i, (cam, mp, ip) in enumerate(raw):
        g = camera_geometry(cam)
        eye0 = np.array(g["eye"]); fwd = np.array(g["forward"])
        right = np.array(g["right"]); up = np.array(g["up"])
        depth = g["depth"]
        # pull camera back along its ray
        tgt = eye0 + depth * fwd
        eye = tgt - depth * CAM_PULLBACK * fwd
        depth *= CAM_PULLBACK
        half_w = g["half_w"] * CAM_PULLBACK
        half_h = g["half_h"] * CAM_PULLBACK
        # silhouette bbox
        m = np.array(Image.open(mp).convert("L"))
        H, W = m.shape
        ys, xs = np.where(m > 80)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        u0, u1 = 2 * x0 / W - 1, 2 * x1 / W - 1
        v0, v1 = 1 - 2 * y1 / H, 1 - 2 * y0 / H
        # mask quad: frame centre at MASK_FRAC, corners at the bbox
        cen = eye + MASK_FRAC * depth * fwd
        hw, hh = half_w * MASK_FRAC, half_h * MASK_FRAC
        corner = lambda u, v: (cen + u * hw * right + v * hh * up)
        c = [corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)]
        # frustum apex (camera pushed back past the eye)
        apex = cen + (eye - cen) * FRUSTUM_LEN
        cams.append({
            "color": hue_rgb(i, n),
            "apex": apex.tolist(),
            "corners": [p.tolist() for p in c],
            "icon": camera_icon_segs(apex, fwd, right, up, icon_s),
            "cutout": cutout_data_uri(ip, mp, (x0, x1, y0, y1)),
        })

    payload = {
        "object": obj.name,
        "n_points": len(xyz),
        "scene_center": cam_data["center"],
        "extent": extent,
        "mask_alpha": MASK_ALPHA,
        "positions": b64(xyz.tobytes()),
        "colors": b64(np.ascontiguousarray(rgb).tobytes()),
        "cameras": cams,
    }

    html = HTML_TEMPLATE.replace("/*__PAYLOAD__*/",
                                 json.dumps(payload, separators=(",", ":")))
    out_html = out_dir / "index.html"
    out_html.write_text(html)
    print(f"[done] {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>sam_tight — cameras + SAM masks</title>
<style>
  html,body{margin:0;height:100%;background:#0d0f12;overflow:hidden;
    font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#dfe3e8}
  #c{display:block}
  #panel{position:fixed;top:12px;left:12px;background:rgba(20,23,28,.88);
    border:1px solid #2c313a;border-radius:10px;padding:12px 14px;
    font-size:13px;line-height:1.5;max-width:300px;backdrop-filter:blur(6px)}
  #panel h1{font-size:14px;margin:0 0 6px;font-weight:600}
  #panel .dim{color:#8b929c}
  .row{display:flex;align-items:center;gap:7px;margin-top:8px}
  .row input{accent-color:#46b3ff}
  label{cursor:pointer;user-select:none}
  kbd{background:#2c313a;border-radius:4px;padding:1px 5px;font-size:11px}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="panel">
  <h1>sam_tight — <span id="objname"></span></h1>
  <div class="dim" id="stats"></div>
  <div class="row"><input type="checkbox" id="cloud" checked>
    <label for="cloud">point cloud</label></div>
  <div class="row"><input type="checkbox" id="frusta" checked>
    <label for="frusta">frustums</label></div>
  <div class="row"><input type="checkbox" id="masks" checked>
    <label for="masks">SAM mask cut-outs</label></div>
  <div class="row"><input type="checkbox" id="cams" checked>
    <label for="cams">camera icons</label></div>
  <div class="dim" style="margin-top:9px">
    drag rotate · scroll zoom · <kbd>right-drag</kbd> pan</div>
</div>

<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const DATA = /*__PAYLOAD__*/;

function b64bytes(s){
  const bin = atob(s); const u = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  return u;
}
const V = a => new THREE.Vector3(a[0],a[1],a[2]);

const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),
  antialias:true});
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f12);

const camera = new THREE.PerspectiveCamera(55,1,0.01,200);
camera.up.set(0,-1,0);          // scene is y-down: -y is up
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping = true;
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;   // full top-to-bottom orbit

const ctr = V(DATA.scene_center);
const ext = DATA.extent;
// initial view matches the render_demo_preview.py preview
// (matplotlib elev=20, azim=-62; converted to this y-down scene frame)
camera.position.copy(ctr).add(new THREE.Vector3(ext*1.765,-ext*1.368,-ext*3.319));
controls.target.copy(ctr);

// ---- point cloud --------------------------------------------------------
const posArr = new Float32Array(b64bytes(DATA.positions).buffer);
const colRaw = b64bytes(DATA.colors);
const colArr = new Float32Array(colRaw.length);
for(let i=0;i<colRaw.length;i++) colArr[i]=colRaw[i]/255;
const cg = new THREE.BufferGeometry();
cg.setAttribute('position',new THREE.BufferAttribute(posArr,3));
cg.setAttribute('color',new THREE.BufferAttribute(colArr,3));
const cloud = new THREE.Points(cg,new THREE.PointsMaterial({
  size:0.012,vertexColors:true,sizeAttenuation:true}));
scene.add(cloud);

// ---- per camera: cut-out mask + frustum + wireframe camera icon ---------
const frustaGroup = new THREE.Group();
const maskGroup   = new THREE.Group();
const iconGroup   = new THREE.Group();
scene.add(frustaGroup,maskGroup,iconGroup);
const loader = new THREE.TextureLoader();

for(const cam of DATA.cameras){
  const col = new THREE.Color(cam.color[0],cam.color[1],cam.color[2]);
  const apex = V(cam.apex);
  const c = cam.corners.map(V);   // c00,c10,c11,c01

  // mask cut-out — textured quad over the 4 bbox corners
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([
    ...c[0].toArray(), ...c[1].toArray(), ...c[2].toArray(),
    ...c[0].toArray(), ...c[2].toArray(), ...c[3].toArray()], 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute([
    0,0, 1,0, 1,1,  0,0, 1,1, 0,1], 2));
  const tex = loader.load(cam.cutout);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({map:tex,transparent:true,
    opacity:DATA.mask_alpha,side:THREE.DoubleSide,depthWrite:false});
  maskGroup.add(new THREE.Mesh(geo,mat));

  // frustum: apex -> 4 corners + the bbox rectangle
  const fseg=[];
  for(const k of [0,1,2,3]) fseg.push(apex,c[k]);
  for(const k of [0,1,2,3]) fseg.push(c[k],c[(k+1)%4]);
  frustaGroup.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(fseg),
    new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.75})));

  // wireframe camera icon
  iconGroup.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(cam.icon.map(V)),
    new THREE.LineBasicMaterial({color:col})));
}

// ---- ui -----------------------------------------------------------------
document.getElementById('objname').textContent=DATA.object;
document.getElementById('stats').textContent=
  `${DATA.n_points.toLocaleString()} points · ${DATA.cameras.length} cameras`;
document.getElementById('cloud').onchange=e=>cloud.visible=e.target.checked;
document.getElementById('frusta').onchange=e=>frustaGroup.visible=e.target.checked;
document.getElementById('masks').onchange=e=>maskGroup.visible=e.target.checked;
document.getElementById('cams').onchange=e=>iconGroup.visible=e.target.checked;

// ---- resize + loop ------------------------------------------------------
function resize(){
  renderer.setSize(innerWidth,innerHeight,false);
  camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();
}
addEventListener('resize',resize);resize();
(function loop(){
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene,camera);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
