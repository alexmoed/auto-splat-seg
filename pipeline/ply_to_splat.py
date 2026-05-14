#!/usr/bin/env python3
"""Convert a gsplat .ply to the .splat binary format used by web viewers.

The .splat format is a simple packed struct per splat, 32 bytes each:
  position  : 3 × float32   (x, y, z)
  scale     : 3 × float32   (sx, sy, sz)  — linear, not log
  color     : 4 × uint8     (r, g, b, a)
  rotation  : 4 × uint8     (normalized quaternion as (w,x,y,z) rescaled
                             to [0, 255] with 128 = 0)

Input ply fields expected (standard gsplat export):
  x, y, z                     : float
  scale_0, scale_1, scale_2   : float (log-scale — we expm it)
  f_dc_0, f_dc_1, f_dc_2      : float (SH DC coefficient — we convert to sRGB)
  opacity                     : float (sigmoid inverse — we sigmoid it)
  rot_0, rot_1, rot_2, rot_3  : float (wxyz quaternion)

Derived per-splat output:
  pos = (x, y, z)
  scale_linear = exp(scale_i)
  rgb = sigmoid(SH_DC) * 255   (linear RGB → bytes)
  alpha = sigmoid(opacity) * 255
  quat_byte_i = round(rot_i / ||rot||  * 128 + 128)

Usage:
    python ply_to_splat.py input.ply output.splat
    python ply_to_splat.py <dir_of_plys> <output_dir>
"""
import argparse
import os
import sys

import numpy as np
from plyfile import PlyData


SH_C0 = 0.28209479177387814  # Y_0^0 sphere harmonic constant


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def convert_one(ply_path, out_path):
    p = PlyData.read(ply_path)
    v = p["vertex"]
    n = len(v)
    names = v.data.dtype.names

    x = np.asarray(v["x"], dtype=np.float32)
    y = np.asarray(v["y"], dtype=np.float32)
    z = np.asarray(v["z"], dtype=np.float32)

    # Scale is log-space in gsplat PLY → exponentiate
    if "scale_0" in names:
        sx = np.exp(np.asarray(v["scale_0"], dtype=np.float32))
        sy = np.exp(np.asarray(v["scale_1"], dtype=np.float32))
        sz = np.exp(np.asarray(v["scale_2"], dtype=np.float32))
    else:
        sx = sy = sz = np.full(n, 0.01, dtype=np.float32)

    # Color: DC of SH (f_dc_0/1/2) → linear RGB via Y_0^0 constant → then
    # scale to [0,255]. The viewer expects sRGB-ish linear values in the
    # uint8 range so the standard antimatter15/.splat convention is:
    #     rgb = 0.5 + SH_C0 * dc       (clamp 0..1)  then * 255
    if "f_dc_0" in names:
        r = 0.5 + SH_C0 * np.asarray(v["f_dc_0"], dtype=np.float32)
        g = 0.5 + SH_C0 * np.asarray(v["f_dc_1"], dtype=np.float32)
        b = 0.5 + SH_C0 * np.asarray(v["f_dc_2"], dtype=np.float32)
    else:
        r = g = b = np.full(n, 0.5, dtype=np.float32)

    # Alpha: sigmoid(opacity) → [0,1] → *255
    if "opacity" in names:
        a = sigmoid(np.asarray(v["opacity"], dtype=np.float32))
    else:
        a = np.full(n, 1.0, dtype=np.float32)

    # Rotation: normalize wxyz to unit quaternion, then encode in [0,255]
    if "rot_0" in names:
        rw = np.asarray(v["rot_0"], dtype=np.float32)
        rx = np.asarray(v["rot_1"], dtype=np.float32)
        ry = np.asarray(v["rot_2"], dtype=np.float32)
        rz = np.asarray(v["rot_3"], dtype=np.float32)
        qn = np.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + 1e-8
        rw /= qn; rx /= qn; ry /= qn; rz /= qn
    else:
        rw = np.ones(n, dtype=np.float32)
        rx = np.zeros(n, dtype=np.float32)
        ry = np.zeros(n, dtype=np.float32)
        rz = np.zeros(n, dtype=np.float32)

    # Build packed output: 32 bytes per splat
    buf = np.zeros(n * 32, dtype=np.uint8)

    # Position (12 bytes)
    pos = np.stack([x, y, z], axis=1).astype(np.float32).tobytes()
    # Scale (12 bytes)
    scale = np.stack([sx, sy, sz], axis=1).astype(np.float32).tobytes()

    # Splat format expects interleaved: for splat i bytes [i*32:(i+1)*32]
    # = pos(12) + scale(12) + rgba(4) + quat(4)
    pos_bytes = np.frombuffer(pos, dtype=np.uint8).reshape(n, 12)
    scale_bytes = np.frombuffer(scale, dtype=np.uint8).reshape(n, 12)

    rgba = np.stack([
        np.clip(r * 255, 0, 255).astype(np.uint8),
        np.clip(g * 255, 0, 255).astype(np.uint8),
        np.clip(b * 255, 0, 255).astype(np.uint8),
        np.clip(a * 255, 0, 255).astype(np.uint8),
    ], axis=1)

    quat = np.stack([
        np.clip(rw * 128 + 128, 0, 255).astype(np.uint8),
        np.clip(rx * 128 + 128, 0, 255).astype(np.uint8),
        np.clip(ry * 128 + 128, 0, 255).astype(np.uint8),
        np.clip(rz * 128 + 128, 0, 255).astype(np.uint8),
    ], axis=1)

    # Pack
    out = np.zeros((n, 32), dtype=np.uint8)
    out[:, 0:12] = pos_bytes
    out[:, 12:24] = scale_bytes
    out[:, 24:28] = rgba
    out[:, 28:32] = quat

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(out.tobytes())

    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".ply file or a directory of .ply files")
    ap.add_argument("output", help="output .splat path OR directory")
    ap.add_argument("--recursive", action="store_true",
                    help="when input is a dir, recurse into subdirs")
    args = ap.parse_args()

    if os.path.isfile(args.input):
        if not args.input.endswith(".ply"):
            sys.exit("[splat] input must be .ply")
        n = convert_one(args.input, args.output)
        sz = os.path.getsize(args.output)
        print(f"[splat] {args.input}  →  {args.output}  ({n:,} splats, {sz/1024:.1f} KB)")
        return

    if not os.path.isdir(args.input):
        sys.exit(f"[splat] input not found: {args.input}")

    # dir → dir mode
    os.makedirs(args.output, exist_ok=True)
    if args.recursive:
        plys = []
        for root, _, files in os.walk(args.input):
            for f in files:
                if f.endswith(".ply"):
                    plys.append(os.path.join(root, f))
    else:
        import glob as _g
        plys = sorted(_g.glob(os.path.join(args.input, "*.ply")))

    if not plys:
        sys.exit(f"[splat] no .ply files in {args.input}")

    total = 0
    for ply in plys:
        rel = os.path.relpath(ply, args.input) if args.recursive else os.path.basename(ply)
        splat = os.path.join(args.output, rel[:-4] + ".splat")
        n = convert_one(ply, splat)
        sz = os.path.getsize(splat)
        total += n
        print(f"[splat] {rel}  →  {n:,} splats, {sz/1024:.1f} KB")
    print(f"[splat] DONE — {len(plys)} files, {total:,} total splats")


if __name__ == "__main__":
    main()
