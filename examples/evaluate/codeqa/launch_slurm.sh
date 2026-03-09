#!/bin/bash
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=codeqa_traj
#SBATCH --output=logs/codeqa_traj_%j.out
#SBATCH --error=logs/codeqa_traj_%j.err

# CodeQA Trajectory Generation — SLURM Wrapper
#
# Usage:
#   sbatch launch_slurm.sh                # uses default .env
#   sbatch --account=YOUR_ACCOUNT launch_slurm.sh
#
# Before submitting:
#   1. Copy .env.template to .env and update paths
#   2. Update --account above for your allocation
#   3. Adjust --partition and --gres for your cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load module system if available (e.g., Quest HPC)
if command -v module &>/dev/null; then
    module load mamba/24.3.0 2>/dev/null || module load conda 2>/dev/null || true
fi

mkdir -p "$SCRIPT_DIR/logs"

# Delegate to the portable run.sh
exec "$SCRIPT_DIR/run.sh" "$@"
