#!/usr/bin/env bash
#
# Serve the local Qwen3.5-4B that writes the hourly summary.
#
# Two models now run side by side, and they do different jobs:
#
#   Qwen2.5-VL (qwen25vl-lite serve, port 8081)  -- looks at frames, describes the scene
#   Qwen3.5-4B (this script, port 8082)          -- reads an hour of memory, writes the broadcast
#
# The VL model already holds most of the 8 GB card, so the 4B runs on CPU.  That is not a
# compromise: a summary is a once-an-hour, few-hundred-token job, and CPU costs no VRAM at all.
# llama.cpp's `llama-server` speaks the same OpenAI API the agent already talks to, so the only
# agent-side change is `llm.text_url`.
#
# The weights are GGUF rather than safetensors: `llama-server` loads GGUF directly, and a Q4_K_M
# quant is ~2.7 GB against ~9.3 GB for the bf16 checkpoint -- which matters on a network that
# delivers a few hundred KB/s.
#
# Usage:
#   scripts/serve_qwen35.sh                  # CPU, port 8082
#   NGLAYERS=99 scripts/serve_qwen35.sh      # offload to GPU (only if the VRAM is actually free)
#   PORT=9000 scripts/serve_qwen35.sh        # a different port
#
# Environment overrides:
#   LLAMA_BIN  directory holding llama-server and libggml*.so   (default: the bonsai build)
#   MODEL      path to the Qwen3.5-4B GGUF
#   PORT/HOST  where to listen                (default: 127.0.0.1:8082)
#   CTX        context length                 (default: 8192)
#   THREADS    CPU threads                    (default: all cores minus two)
#   NGLAYERS   GPU layers to offload          (default: 0, i.e. CPU only)
#   ALIAS      the model name the API reports (default: Qwen3.5-4B)
#
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/workspace/bonsai2-27b/bin}"
MODEL="${MODEL:-$HOME/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf}"
PORT="${PORT:-8082}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
NGLAYERS="${NGLAYERS:-0}"
ALIAS="${ALIAS:-Qwen3.5-4B}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -x "${LLAMA_BIN}/llama-server" ]] || die "no llama-server in ${LLAMA_BIN} (set LLAMA_BIN)"
[[ -f "${MODEL}" ]] || die "no GGUF at ${MODEL} (set MODEL; download it with
       modelscope download --model unsloth/Qwen3.5-4B-GGUF \\
         --local-dir \"$(dirname "${MODEL}")\" Qwen3.5-4B-Q4_K_M.gguf)"

printf 'serving %s\n  on http://%s:%s/v1 (alias %s)\n  ctx %s, %s threads, %s GPU layers\n' \
    "$(basename "${MODEL}")" "${HOST}" "${PORT}" "${ALIAS}" "${CTX}" "${THREADS}" "${NGLAYERS}"

# The build ships libggml*.so next to the binary, so they must be on the loader path.
exec env LD_LIBRARY_PATH="${LLAMA_BIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${LLAMA_BIN}/llama-server" \
    --model "${MODEL}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --ctx-size "${CTX}" \
    --threads "${THREADS}" \
    --n-gpu-layers "${NGLAYERS}" \
    --jinja \
    --reasoning-budget 0 \
    --chat-template-kwargs '{"enable_thinking":false}'
