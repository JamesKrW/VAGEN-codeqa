#!/bin/bash
#SBATCH --account=p32992
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=codeqa_traj
#SBATCH --output=/projects/p32992/VAGEN/examples/evaluate/codeqa/logs/codeqa_traj_%j.out
#SBATCH --error=/projects/p32992/VAGEN/examples/evaluate/codeqa/logs/codeqa_traj_%j.err

echo "=========================================="
echo "CodeQA Trajectory Generation (6x125=750)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=========================================="

# ── Environment ──
module load mamba/24.3.0
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate agentocr

export PYTHONPATH="/projects/p32992/VAGEN:$PYTHONPATH"
export VLLM_USE_V1=1
export HF_HOME="/projects/p32992/.cache/huggingface"

# ── Config (absolute path) ──
CONFIG_PATH="/projects/p32992/VAGEN/examples/evaluate/codeqa/config_trajectories.yaml"

# ── vLLM server settings ──
MODEL="Qwen/Qwen3-VL-8B-Instruct"
PORT=8000
MAX_MODEL_LEN=50000
TP=4
MAX_IMAGES=30

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Max model len: $MAX_MODEL_LEN (reduced for ~20K token budget)"
echo "  Max images per prompt: $MAX_IMAGES"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $PORT \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --disable-log-requests \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
    --disable-mm-preprocessor-cache \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server..."
MAX_WAIT=60000
WAITED=0
while ! curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; do
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
