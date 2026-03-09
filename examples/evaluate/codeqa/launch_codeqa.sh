#!/bin/bash
#SBATCH --account=p32992
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --job-name=codeqa_vagen
#SBATCH --output=logs/codeqa_vagen_%j.out
#SBATCH --error=logs/codeqa_vagen_%j.err

set -e
mkdir -p logs

echo "=========================================="
echo "CodeQA VAGEN Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "=========================================="

# ---- Environment setup ----
module purge
module load mamba/24.3.0

CONDA_ENV="/home/aib4675/.conda/envs/agentocr"
eval "$('/hpc/software/mamba/24.3.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
source "/hpc/software/mamba/24.3.0/etc/profile.d/mamba.sh"
mamba activate ${CONDA_ENV}

# Caches
export HF_HOME="/projects/p32992/.cache/huggingface"
export VLLM_CACHE_ROOT="/projects/p32992/.cache/vllm"
export XDG_CACHE_HOME="/projects/p32992/.cache"
export VLLM_USE_V1=1
mkdir -p $HF_HOME $VLLM_CACHE_ROOT

# Add VAGEN to PYTHONPATH
export PYTHONPATH="/projects/p32992/VAGEN:${PYTHONPATH}"

# ---- Phase 1: Start vLLM OpenAI-compatible server ----
echo ""
echo "Starting vLLM server..."

# Max images needed: 113 (python-jsonschema_jsonschema at 8pt)
MAX_IMAGES=120

python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 235520 \
    --gpu-memory-utilization 0.70 \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
    --disable-mm-preprocessor-cache \
    --host 127.0.0.1 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
MAX_WAIT=600
WAITED=0
while ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done
echo "vLLM server is ready!"

# ---- Phase 2: Run VAGEN evaluation ----
echo ""
echo "Running VAGEN CodeQA evaluation..."

CONFIG_PATH="/projects/p32992/VAGEN/examples/evaluate/codeqa/config.yaml"

python -m vagen.evaluate.run_eval \
    --config "${CONFIG_PATH}"

EXIT_CODE=$?

# ---- Cleanup ----
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "Job Complete!"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
