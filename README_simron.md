# CodeQA: Two-Turn VLM Code Understanding Evaluation

## Overview

This adds a **CodeQA** Gymnasium environment to VAGEN for evaluating VLM (Vision Language Model) code understanding via a two-turn interaction:

1. **Turn 1 (OCR)**: The model receives code rendered as images and transcribes it
2. **Turn 2 (QA)**: The model receives a multiple-choice question about the code and answers A/B/C/D

Only the final QA answer is scored (reward = 1.0 correct, 0.0 incorrect).

## Research Question

Can a VLM (Qwen3-VL-8B) read code from images and answer questions about it? This extends the VTI (Visual Token Interface) experiments by using VAGEN's multi-turn agent framework instead of single-shot inference.

## Data

- **Benchmark**: LongCodeBench QA -- 443 MCQ questions across 98 Python repositories
- **Subset used**: Clean+NonTruncated (125 samples) -- non-contaminated questions from repos with 100% image coverage
- **Images**: Code rendered at 8pt font (`code_images_8pt/`), avg 545x1275px per image, up to 113 images per repo
- **Model**: Qwen3-VL-8B-Instruct via vLLM (4xA100, tensor-parallel)

### Data Paths

| Data | Path |
|------|------|
| Dataset JSONL | `AgentOCR/.../data/longcodebench_qa/prepared_dataset_full/all_repos_dataset.jsonl` |
| Code Images | `AgentOCR/.../data/longcodebench_qa/code_images_8pt/{repo_dir}/chunk_*.png` |
| Contaminated IDs | `AgentOCR/.../data/longcodebench_qa/contaminated_questions_v3.json` |
| Non-truncated Repos | `AgentOCR/.../data/longcodebench_qa/nontruncated_repos.json` |

## Architecture

### Two-Turn Gymnasium Protocol

```
                     VAGEN Workflow (max_turns=2)
                     ============================

system_prompt()  -->  "You are an expert code analyst..."

reset(seed=N)    -->  Turn 1 Observation:
                      - Code images (up to 113 PNGs per repo)
                      - "Read and transcribe the code..."

                      Model responds with OCR transcription

step(ocr_text)   -->  Turn 2 Observation:
                      - MCQ question (text only, no images)
                      - "Answer A, B, C, or D"
                      - reward=0.0, done=False

                      Model responds with answer

step(qa_answer)  -->  Score the answer
                      - Extract letter via regex cascade
                      - Compare to ground truth
                      - reward=1.0/0.0, done=True
                      - info["success"] = True/False
```

### Concat Mode

VAGEN uses growing context (concat mode), so the model sees Turn 1 images in its conversation history when answering the Turn 2 QA question. The full context is:

```
[system] [code images + OCR instruction] [OCR response] [QA question] [QA answer]
```

### Seed-to-Sample Mapping

- 125 Clean+NT samples sorted deterministically by ID
- `seed % 125` maps each seed to a specific sample
- Config uses `seed: [0, 124, 1]` for 1:1 mapping

## File Structure

```
VAGEN/
  vagen/envs/codeqa/
    __init__.py
    codeqa_env.py               # Main CodeQA(GymImageEnv) class
    utils/
      __init__.py
      answer_extraction.py      # extract_answer_letter() -- 9-priority regex cascade
      data_loader.py            # Dataset loading, filtering, image loading
      prompt.py                 # System, OCR, QA prompt templates
  vagen/configs/
    env_registry.yaml           # Added: CodeQA entry
  examples/evaluate/codeqa/
    config.yaml                 # Evaluation config (vLLM, 125 samples, 2 turns)
    launch_codeqa.sh            # SLURM script (4xA100, 12hr)
    test_codeqa_env.py          # CPU test suite
```

## How to Run

### 1. CPU Test (no GPU needed)

Verify the environment works correctly:

```bash
cd /projects/p32992/VAGEN
PYTHONPATH=. python examples/evaluate/codeqa/test_codeqa_env.py
```

Expected output: all 4 test suites pass (answer_extraction, registry, lifecycle, seed_mapping).

### 2. Full GPU Evaluation

Submit the SLURM job:

```bash
cd /projects/p32992/VAGEN/examples/evaluate/codeqa
sbatch launch_codeqa.sh
```

This will:
1. Start a vLLM server with Qwen3-VL-8B (4xA100, tp=4)
2. Wait for server readiness
3. Run `python -m vagen.evaluate.run_eval --config config.yaml`
4. Process all 125 Clean+NT samples with 2 turns each
5. Dump results to `rollouts/codeqa_clean_nt_8pt/`

### 3. Small-Scale Test (3 samples)

For a quick GPU test, override the config:

```bash
python -m vagen.evaluate.run_eval \
    --config examples/evaluate/codeqa/config.yaml \
    envs.0.n_envs=3 \
    envs.0.seed=[0,2,1]
```

## Results Location

After evaluation, results are saved to:

```
rollouts/codeqa_clean_nt_8pt/
  tag_codeqa_clean_nt_8pt/
    {rollout_id}/
      messages.json          # Full conversation (images redacted)
      assistant_texts.json   # Model responses (OCR + QA)
      images/                # Saved input images per turn
      transcript.txt         # Human-readable conversation
      metrics.json           # Episode metrics (success, rewards, etc.)
    summary.json             # Aggregate summary across all episodes
```

### Key Metrics in `metrics.json`

- `success`: Whether the QA answer was correct
- `finish_reason`: "done" (normal), "max_turns", "model_error", "env_error"
- `cumulative_reward`: 1.0 (correct) or 0.0 (incorrect)
- `num_turns`: Should be 2 for normal episodes
- `infos[-1].predicted_letter`: Model's extracted answer
- `infos[-1].correct_letter`: Ground truth

## Expected Baseline Performance

From previous VTI V3 experiments on the same 125 Clean+NT samples:
- **VLM single-shot (v3_baseline)**: 48.8% accuracy (61/125)
- **LLM text baseline**: 42.4% accuracy (53/125)

The two-turn VAGEN evaluation may differ because:
- Turn 1 OCR is capped at 4096 output tokens (model can't transcribe all code)
- The model has a "transcription step" that may improve or hurt QA accuracy
- Temperature=0 for deterministic outputs

## Configuration Reference

### Environment Config (`env_config`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | str | "" | Path to all_repos_dataset.jsonl |
| `images_dir` | str | "" | Path to code_images_8pt/ directory |
| `contaminated_path` | str | "" | Path to contaminated_questions_v3.json |
| `nontruncated_path` | str | "" | Path to nontruncated_repos.json |
| `subset` | str | "clean_nt" | Filter: "all", "clean", "non_truncated", "clean_nt" |
| `max_vision_tokens` | int | 230000 | Vision token budget for image loading |
| `correct_reward` | float | 1.0 | Reward for correct QA answer |

### SLURM Resources

- 4x A100 GPUs (tensor-parallel vLLM)
- 128GB RAM
- 12 hours wall time
- Partition: gengpu, Account: p32992
