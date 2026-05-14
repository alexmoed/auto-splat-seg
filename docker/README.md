# Splat segmentation — Docker setup

Single-container deploy of the pipeline pipeline. vLLM (Qwen 3.6 35B
A3B AWQ) launches in the background inside the same container; the
pipeline scripts call it over `http://127.0.0.1:8000/v1` like they
already do on the host.

**Non-destructive**: this folder lives next to `pipeline/` and
doesn't modify any pipeline scripts. The image copies them at build
time.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | Image definition (vllm/vllm-openai base + gsplat/SAM3/scripts) |
| `requirements.txt` | Pipeline-specific deps installed on top of vLLM's stack |
| `entrypoint.sh` | Launches vLLM in background, waits for readiness, runs CMD |
| `docker-compose.yml` | One-service compose with volume mounts |
| `.env.example` | Template for host-specific paths |

## Prereqs

- Docker 24+ with the `nvidia-container-toolkit` installed and `nvidia` runtime registered.
- A 48GB+ NVIDIA GPU (L40S / A6000-48GB / A40 / RTX 6000 Ada or bigger).
- The Qwen 3.6 35B-A3B-AWQ weights downloaded somewhere on the host (e.g. `~/models/Qwen3.6-35B-A3B-AWQ`).

## Build

```bash
cd /home/ubuntu/room_pipeline_v002
docker build -t splat-pipeline -f docker/Dockerfile .
```

The build context is the repo root (one level up from `docker/`) so
the `COPY pipeline/` step works. Image is ~12-15 GB (vLLM base ~8 GB
+ pipeline deps + scripts).

The Dockerfile vendors `view.py` from `~/.claude/skills/gsplat-viewer/scripts/`.
If that path doesn't exist on your build host, override the build arg:

```bash
docker build -t splat-pipeline \
  --build-arg GSPLAT_VIEWER_VIEW_PY=path/to/view.py \
  -f docker/Dockerfile .
```

## Run

**Recommended pattern: long-running container + `docker exec` per scene.**
Start the container once (vLLM loads Qwen — costs 60-90s once), then
fire off `docker exec` for every scene. The model stays warm across
runs — no need for an HTTP API.

```bash
# Start once — model loads, container stays alive
docker run -d --gpus all --name splat-pipe \
  -v ~/models/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro \
  -v ~/scenes:/workspace/scenes \
  splat-pipeline tail -f /dev/null

# Wait ~90s for vLLM to load (`docker logs splat-pipe` to watch)

# Run scenes — vLLM already warm, no startup cost
docker exec splat-pipe python /workspace/pipeline/run_all.py /workspace/scenes/scene1 --step 2
docker exec splat-pipe python /workspace/pipeline/run_all.py /workspace/scenes/scene2 --step 2

# Stop when done
docker stop splat-pipe && docker rm splat-pipe
```

### Interactive shell (debugging)
```bash
docker run --rm -it --gpus all \
  -v ~/models/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro \
  -v ~/path/to/scene_dir:/workspace/scene \
  splat-pipeline bash
# vLLM starts in the background, then you get a shell. Run any
# pipeline command, e.g.:
#   python /workspace/pipeline/run_all.py /workspace/scene --step 2
```

### One-shot via docker-compose
For when you just want to process one scene and exit (model loads + runs + exits):

```bash
cd docker/
cp .env.example .env
# edit .env with QWEN_WEIGHTS_HOST_PATH and SCENE_HOST_PATH
docker compose run --rm pipeline \
  python /workspace/pipeline/run_all.py /workspace/scene --step 2
```

This pays the 60-90s vLLM startup cost every run — fine for occasional
use, wasteful for batches. Use the long-running pattern above for
multiple scenes.

## Memory tuning (48 GB GPU)

Defaults baked into the image (in `Dockerfile` ENV + `entrypoint.sh`):
- `QWEN_GPU_MEMORY_UTILIZATION=0.65` (≈30 GB for vLLM)
- `QWEN_MAX_MODEL_LEN=4096` (smaller KV cache; our prompts are short)

Breakdown of the ~30 GB vLLM allocation:
- Model weights: 22 GB
- KV cache: ~5 GB
- torch.compile + CUDA graphs overhead: ~3 GB

This leaves ~16 GB for SAM3 + gsplat. If you OOM on a particularly
large scene, drop utilization:

```bash
docker run --rm --gpus all \
  -e QWEN_GPU_MEMORY_UTILIZATION=0.55 \
  ... splat-pipeline bash
```

If you have an 80 GB card, bump utilization to 0.55 (~44 GB) and
max_model_len to 8192 for faster batched inference. Earlier defaults
of 0.55/8192 were too tight on a 48 GB card — KV cache had only 0.15
GB and vLLM refused to start.

## Skipping the in-container vLLM

If you'd rather run vLLM in a separate container (e.g. for a multi-GPU
setup where one GPU is dedicated to Qwen), set `SKIP_VLLM=1` and point
`QWEN_URL` at the external service:

```bash
docker run --rm -it --gpus all \
  -e SKIP_VLLM=1 \
  -e QWEN_URL=http://other-host:8000/v1 \
  -v ~/path/to/scene_dir:/workspace/scene \
  splat-pipeline bash
```

## What the entrypoint does

1. If `SKIP_VLLM=0` (default): launches `vllm serve …` in the background, redirecting its log to `/tmp/vllm.log`.
2. Polls `http://127.0.0.1:8000/v1/models` every 5s, up to 5 min.
3. If vLLM dies during startup, prints the last 50 log lines and exits 1.
4. Once ready, exec's the `docker run` command (default `bash`).

## Common issues

| Symptom | Fix |
|---|---|
| `nvidia-container-cli: requirement error` | Install `nvidia-container-toolkit`, register `nvidia` runtime, restart Docker |
| `OSError: Background loop has errored already` | vLLM ran out of GPU memory; lower `QWEN_GPU_MEMORY_UTILIZATION` |
| `cv2 ImportError: libGL.so.1: cannot open shared object` | Apt deps in Dockerfile didn't install — rebuild with `--no-cache` |
| `gsplat compile error during pip install` | The vLLM base image's torch ABI changed; pin `VLLM_TAG=v0.19.1` (known good) and rebuild |
| Pipeline scripts can't find `view.py` | Ensure the `COPY ${GSPLAT_VIEWER_VIEW_PY}` step succeeded; check the build arg |

## Why no HTTP API?

The "API server" model (long-running process serving REST/gRPC) and
the "long-running container + exec" model give the same warm-Qwen
benefit. No reason to add an HTTP wrapper when `docker exec` already
gives you what you need. Skip the FastAPI layer until/unless a remote
client or non-engineer user actually needs it.

## Future cleanups (deferred)

Non-destructive today. If/when you want to tighten further:

1. Centralize the OpenAI client setup in `pipeline/qwen_client.py` (replace duplicated boilerplate in 7+ scripts).
2. Optionally swap the HTTP client for vLLM's offline Python API (`from vllm import LLM, …`) — eliminates the second process inside the container.
3. Make `QWEN_URL` etc. configurable via env vars in the scripts directly (currently hardcoded constants).

Each of these is invasive — defer until the Docker build is validated end-to-end.
