#!/usr/bin/env python3
"""sam_server.py — persistent SAM3 segmentation service.

Loads facebook/sam3 ONCE at startup, then serves segmentation requests
over HTTP. Mirrors the vLLM-for-Qwen pattern so the per-object pipeline
subprocesses don't each reload the SAM weights (each load is ~3-5s
× ~144 calls per scene = 8-12 min wall-clock saved).

Endpoint:
    POST /segment   {image_path: str, prompt: str, threshold: float (opt)}
    →               {mask_b64_png: str, scores: [float, ...], h: int, w: int}

The image_path is the same filesystem path the client sees — the server
and the scripts both run inside the container and share /workspace.

Boot:
    uvicorn sam_server:app --host 127.0.0.1 --port 8001

The launch is wired into the container's entrypoint.sh alongside vLLM,
both started in the background before the user's CMD runs.
"""
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

import torch
torch.backends.cudnn.enabled = False  # vLLM/cuDNN coexistence
from transformers import Sam3Model, Sam3Processor

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[sam-server] loading facebook/sam3 on {_DEVICE}...", flush=True)
_PROC = Sam3Processor.from_pretrained("facebook/sam3")
_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(_DEVICE).eval()
print("[sam-server] ready", flush=True)


class SegmentRequest(BaseModel):
    image_path: str
    prompt: str
    threshold: float = 0.4


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "device": _DEVICE}


@app.post("/segment")
def segment(req: SegmentRequest):
    img = Image.open(req.image_path).convert("RGB")
    inputs = _PROC(images=img, text=req.prompt, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        out = _MODEL(**inputs)
    results = _PROC.post_process_instance_segmentation(
        out, threshold=req.threshold, target_sizes=[(img.height, img.width)]
    )[0]
    scores = results.get("scores", torch.tensor([]))
    masks = results.get("masks", torch.tensor([]))
    h, w = img.height, img.width
    if len(masks) == 0:
        m = np.zeros((h, w), dtype=np.uint8)
    else:
        m = (masks.cpu().numpy() > 0).any(axis=0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG")
    return {
        "mask_b64_png": base64.b64encode(buf.getvalue()).decode(),
        "scores": scores.cpu().tolist(),
        "h": h,
        "w": w,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAM_SERVER_PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
