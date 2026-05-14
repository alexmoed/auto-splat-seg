# syntax=docker/dockerfile:1.6
#
# Single-container splat-segmentation pipeline.
#
# Layout:
#   - Base: vllm/vllm-openai (CUDA 12.1 + torch + vLLM pre-installed)
#   - Adds: gsplat, SAM3 deps, plyfile, scipy, OpenCV
#   - Adds: pipeline/ scripts copied to /workspace/pipeline
#   - Adds: gsplat-viewer/scripts/view.py vendored to /workspace/pipeline
#           (so we don't need the host's ~/.claude/skills/ dir)
#
# Entrypoint launches vLLM serving Qwen 3.6 35B AWQ in the background,
# waits for :8000 to be ready, then runs whatever command the user
# passed (or `bash` if none).
#
# Build:
#   docker build -t splat-pipeline -f docker/Dockerfile .
#
# Run (interactive shell, scene mounted in):
#   docker run --rm -it --gpus all \
#     -v /path/to/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro \
#     -v /path/to/scene_dir:/workspace/scene \
#     splat-pipeline bash
#
# Run pipeline directly:
#   docker run --rm --gpus all \
#     -v /path/to/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro \
#     -v /path/to/scene_dir:/workspace/scene \
#     splat-pipeline \
#     python /workspace/pipeline/run_all.py /workspace/scene --step 2

ARG VLLM_TAG=v0.19.1
FROM vllm/vllm-openai:${VLLM_TAG}

# We deliberately stay non-root inside vllm/vllm-openai (it ships with
# 'vllm' user). Switch to root only for installs.
USER root

# System deps for OpenCV runtime + diagnostics. libgl1 / libglib are
# what get missing-symbol errors when cv2 imports without them.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        wget curl ca-certificates jq \
 && rm -rf /var/lib/apt/lists/*

# Pipeline deps on top of vLLM's torch+transformers stack.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline scripts into the image.
WORKDIR /workspace
COPY pipeline/ /workspace/pipeline/

# Vendor view.py from the gsplat-viewer skill so scripts that do
#     sys.path.insert(0, "/home/ubuntu/.claude/skills/gsplat-viewer/scripts")
# resolve inside the container. The file is copied from docker/view.py
# in the build context (vendored alongside the Dockerfile so it doesn't
# need to escape the context), and we create the hardcoded path as a
# symlink so non-modified scripts keep working.
COPY docker/view.py /workspace/pipeline/_vendored_view.py
RUN mkdir -p /home/ubuntu/.claude/skills/gsplat-viewer/scripts \
 && ln -sf /workspace/pipeline/_vendored_view.py \
       /home/ubuntu/.claude/skills/gsplat-viewer/scripts/view.py

# Entrypoint: start vLLM in background, wait for readiness, exec CMD.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Default env vars — override at run time.
ENV QWEN_URL=http://127.0.0.1:8000/v1 \
    QWEN_MODEL=qwen36-awq \
    QWEN_MODEL_PATH=/models/qwen36-awq \
    QWEN_GPU_MEMORY_UTILIZATION=0.6 \
    QWEN_MAX_MODEL_LEN=16384 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
