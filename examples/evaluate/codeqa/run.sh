#!/bin/bash
# CodeQA Trajectory Generation — Portable Runner
# Works on any machine with GPUs, conda, and vLLM installed.
#
# Usage:
#   ./run.sh              # uses .env in same directory
#   ./run.sh /path/to/.env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Load .env ──
ENV_FILE="${1:-$SCRIPT_DIR/.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.template to .env and update paths:"
    echo "  cp $SCRIPT_DIR/.env.template $SCRIPT_DIR/.env"
    exit 1
fi
echo "Loading config from: $ENV_FILE"
set -a; source "$ENV_FILE"; set +a

# ── Resolve relative paths against SCRIPT_DIR ──
resolve_path() {
    local p="$1"
    if [[ "$p" != /* ]]; then
        echo "$SCRIPT_DIR/$p"
    else
        echo "$p"
    fi
}

DATA_DIR="$(resolve_path "${DATA_DIR}")"
DATASET_PATH="$(resolve_path "${DATASET_PATH}")"
IMAGES_DIR="$(resolve_path "${IMAGES_DIR}")"
CONTAMINATED_PATH="$(resolve_path "${CONTAMINATED_PATH}")"
NONTRUNCATED_PATH="$(resolve_path "${NONTRUNCATED_PATH}")"
ROLLOUT_DIR="$(resolve_path "${ROLLOUT_DIR}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"

# Re-export resolved paths
export DATA_DIR DATASET_PATH IMAGES_DIR CONTAMINATED_PATH NONTRUNCATED_PATH
export ROLLOUT_DIR LOG_DIR

# ── Validate data exists ──
for f in "$DATASET_PATH" "$CONTAMINATED_PATH" "$NONTRUNCATED_PATH"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        echo "Run setup_data.py first to download and prepare the dataset."
        exit 1
    fi
done
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    echo "Run setup_data.py first to download and prepare the dataset."
    exit 1
fi

# ── Activate conda (skip if already active) ──
CONDA_ENV="${CONDA_ENV:-vagen}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "$CONDA_ENV"
    elif command -v mamba &>/dev/null; then
        eval "$(mamba shell.bash hook 2>/dev/null)"
        mamba activate "$CONDA_ENV"
    else
        echo "WARNING: conda/mamba not found. Assuming environment is already set up."
    fi
fi

# ── Set environment ──
export PYTHONPATH="${VAGEN_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── Generate config from template ──
CONFIG_TEMPLATE="$SCRIPT_DIR/config_trajectories.yaml.template"
CONFIG_PATH="$SCRIPT_DIR/config_trajectories_generated.yaml"

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "ERROR: Config template not found: $CONFIG_TEMPLATE"
    exit 1
fi

envsubst < "$CONFIG_TEMPLATE" > "$CONFIG_PATH"
echo "Generated config: $CONFIG_PATH"

# ── vLLM settings ──
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-50000}"
MAX_IMAGES="${MAX_IMAGES_PER_PROMPT:-30}"

mkdir -p "$LOG_DIR" "$ROLLOUT_DIR"

echo "=========================================="
echo "CodeQA Trajectory Generation (6x125=750)"
echo "=========================================="
echo "Time: $(date)"
echo "Model: $MODEL"
echo "Max model len: $MAX_MODEL_LEN"
echo "Max images/prompt: $MAX_IMAGES"
echo "Rollout dir: $ROLLOUT_DIR"
echo "=========================================="

# ── Start vLLM server ──
echo "Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --disable-log-requests \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
    --disable-mm-preprocessor-cache \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# ── Wait for server ──
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start after ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready after ${WAITED}s"

# ── Run evaluation ──
echo ""
echo "Running trajectory generation..."
echo "  Config: $CONFIG_PATH"
echo "  750 episodes (125 samples x 6 trajectories)"
echo ""

python -m vagen.evaluate.run_eval --config "$CONFIG_PATH"
EXIT_CODE=$?

# ── Cleanup ──
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Job Complete!"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
