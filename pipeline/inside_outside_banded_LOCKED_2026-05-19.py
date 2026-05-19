#!/usr/bin/env python3
"""inside_outside_banded.py — inside_outside, but the carve is applied
ONLY within the bottom band (default 15%) of the object's height.
Above-band splats are force-kept regardless of insideness.

Usage:
    python3 inside_outside_banded.py <obj_dir> [thresh=0.90] [band_frac=0.15]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

for _p in ("/workspace/pipeline", "/home/ubuntu/room_pipeline_v002/pipeline"):
    if Path(_p).exists():
        sys.path.insert(0, _p)
        break

from inside_outside import insideness          # noqa: E402
from sam_tight import render_canonical_5       # noqa: E402
from sam_carve import Y_DOWN                   # noqa: E402

obj = Path(sys.argv[1]).resolve()
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
band_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

in_ply = obj / "4_sam_tight.ply"
mask_dir = obj / "diagnostics" / "4_sam_tight"
cam_json = mask_dir / "cameras.json"
print(f"[load] in_ply={in_ply}")
print(f"[load] mask_dir={mask_dir}  thresh={thresh}  band_frac={band_frac}")

cams = json.load(open(cam_json))["cameras"]
masks = []
for cm in cams:
    mp = mask_dir / f"mask_{cm['tag']}.png"
    if not mp.exists():
        continue
    m = np.asarray(Image.open(mp).convert("L")) > 127
    if int(m.sum()) < 2000:
        continue
    masks.append((np.array(cm["V"], np.float64),
                  np.array(cm["K"], np.float64), m))
print(f"[load] {len(masks)} usable tight masks")

pd = PlyData.read(str(in_ply))
v = pd["vertex"]
raw = v.data
xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
n = len(xyz)

s = insideness(xyz, masks)
print(f"[insideness] <0.3={int((s<0.3).sum())}  "
      f"0.3-0.7={int(((s>=0.3)&(s<=0.7)).sum())}  >0.7={int((s>0.7).sum())}")

y = xyz[:, 1]
y_lo, y_hi = float(y.min()), float(y.max())
h = y_hi - y_lo
# Y_DOWN True → floor side = max y; band = top of y range = near-floor
if Y_DOWN:
    band_edge = y_hi - band_frac * h
    in_band = y >= band_edge
else:
    band_edge = y_lo + band_frac * h
    in_band = y <= band_edge
print(f"[band] y=[{y_lo:.3f},{y_hi:.3f}]  h={h:.3f}m  "
      f"band={band_frac*100:.0f}% = {band_frac*h:.3f}m  "
      f"band_edge={band_edge:.3f}  in_band={int(in_band.sum()):,}/{n:,}")

# carve only inside the band
keep_naive = s >= thresh
keep = keep_naive | (~in_band)
carved_in_band = int(in_band.sum() - (keep & in_band).sum())
print(f"[carve] thresh={thresh}: would carve {n-int(keep_naive.sum()):,} "
      f"globally → BANDED carves {carved_in_band:,} (in-band only)")
print(f"[carve] keep {int(keep.sum()):,}/{n:,} "
      f"({100*keep.sum()/n:.1f}%)")

out = obj / "6_io_banded.ply"
PlyData([PlyElement.describe(raw[keep], "vertex")],
        text=False).write(str(out))
render_canonical_5(out, obj / "renders" / "6_io_banded")
print(f"[done] {out}")
