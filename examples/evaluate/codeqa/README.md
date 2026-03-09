# CodeQA: Vision-Language Code Understanding with VAGEN

This module extends [VAGEN](https://github.com/RAGEN-AI/VAGEN) to evaluate and train Vision Language Models (VLMs) on code understanding tasks. The model reads source code rendered as images and answers multiple-choice questions about it.

## Dataset

The dataset is derived from [LongCodeBench](https://github.com/patidarrahul/LongCodeBench), a benchmark of 443 MCQ questions across 98 Python repositories. We adapted it for vision-language evaluation:

### Modifications from LongCodeBench

1. **Code as images**: Repository source code is rendered as syntax-highlighted PNG images (8pt font, 712px wide, 125 lines per image) using Pygments + WeasyPrint, rather than passed as raw text.

2. **Contamination filtering**: We identified 229/443 questions where the model could answer correctly *without* seeing the code (the questions name specific libraries). These contaminated questions are excluded, leaving 214 clean questions.

3. **Non-truncated filtering**: Of the 214 clean questions, 125 come from repos whose full source code fits within the model's context window without truncation. We use these 125 "clean + non-truncated" (Clean+NT) samples.

4. **Stratified image subsampling**: Large repos can have 50-113 code images (up to 119K vision tokens). To fit within a ~20K token budget for training, we use stratified subsampling that proportionally selects images across file categories (source, docs, config, test files). Small repos (already under 20K tokens) are included in full.

### Dataset Statistics

| Split | Rollouts | Questions | Repos | Sampled Repos | Unsampled Repos | Avg Accuracy |
|-------|----------|-----------|-------|---------------|-----------------|--------------|
| Train | 450      | 75        | 24    | 18            | 6               | 50.7%        |
| Val   | 156      | 26        | 15    | 10            | 5               | 50.0%        |
| Test  | 144      | 24        | 14    | 10            | 4               | 45.8%        |
| **Total** | **750** | **125** | **53** | **38**       | **15**          | **49.6%**    |

- **Sampled repos**: Repos with >20K vision tokens — images are subsampled to fit the token budget
- **Unsampled repos**: Repos with <=20K vision tokens — all images are included
- **6 trajectories per question**: Generated with `temperature=0.4` for output diversity (same images across trajectories)
- **Model**: Qwen3-VL-8B-Instruct via vLLM
- **Repo-level splitting**: No repository appears in multiple splits (prevents data leakage through shared code images)

### Two-Turn Conversation Format

Each rollout is a 2-turn conversation:

1. **Turn 1 (OCR)**: The model receives code images and is asked to transcribe/understand the code
2. **Turn 2 (QA)**: The model receives an MCQ question about the code and selects A/B/C/D

### HuggingFace Dataset

The dataset is hosted at: [`spatel-learn/codeqa-vagen-trajectories`](https://huggingface.co/datasets/spatel-learn/codeqa-vagen-trajectories)

Each row contains:
| Column | Type | Description |
|--------|------|-------------|
| `trajectory_id` | int | Trajectory index (0-5) |
| `seed` | int | Sample index |
| `repo` | string | Repository name (e.g., `aio-libs/yarl`) |
| `sample_id` | string | Question identifier |
| `images` | list[Image] | Code screenshot PNGs (avg ~20 per sample) |
| `system_prompt` | string | System prompt text |
| `ocr_instruction` | string | Turn 1 instruction text |
| `ocr_response` | string | Model's code transcription |
| `qa_question` | string | Turn 2 MCQ question |
| `qa_response` | string | Model's answer and reasoning |
| `success` | bool | Whether the model answered correctly |
| `predicted_letter` | string | Model's answer (A/B/C/D) |
| `correct_letter` | string | Ground truth answer |
| `vision_tokens_used` | int | Number of vision tokens in the prompt |
| `cumulative_reward` | float | Reward (1.0 if correct, 0.0 otherwise) |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/spatel418/VAGEN.git
cd VAGEN
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate vagen
pip install -e .
```

### 3. Download the dataset

```bash
cd examples/evaluate/codeqa
python setup_data.py --data_dir ./data
```

This downloads from HuggingFace and generates a `.env` file with the correct paths. Review `.env` and adjust settings if needed (GPU count, model, etc.).

### 4. Run evaluation

**Without SLURM** (any machine with 4x GPUs):
```bash
./run.sh
```

**With SLURM** (HPC cluster):
```bash
# Edit launch_slurm.sh to set your --account
sbatch launch_slurm.sh
```

The script will:
1. Start a vLLM server with the Qwen3-VL-8B model
2. Run 750 episodes (125 samples x 6 trajectories)
3. Save rollouts to the configured output directory
4. Shut down the vLLM server

### 5. Build train/val/test splits

After rollouts complete:
```bash
python build_datasets.py \
    --rollout_dir ./data/rollouts \
    --output_dir ./data/splits
```

### 6. Convert to HuggingFace format (optional)

```bash
python build_hf_dataset.py \
    --splits_dir ./data/splits \
    --output_dir ./data/hf_dataset
```

## Project Structure

```
examples/evaluate/codeqa/
├── README.md                           # This file
├── .env.template                       # Configuration template
├── config_trajectories.yaml.template   # YAML config template (paths filled by run.sh)
├── setup_data.py                       # Download data from HuggingFace
├── run.sh                              # Portable runner (any machine)
├── launch_slurm.sh                     # SLURM wrapper
├── build_datasets.py                   # Build train/val/test JSONL splits
└── build_hf_dataset.py                 # Convert to HuggingFace dataset format

vagen/envs/codeqa/
├── __init__.py
├── codeqa_env.py                       # CodeQA gymnasium environment
└── utils/
    ├── data_loader.py                  # Dataset loading and image loading
    ├── prompt.py                       # System/user prompts
    ├── answer_extraction.py            # Extract A/B/C/D from model output
    ├── stratified_sampler.py           # Stratified image subsampling
    └── split_definition.json           # Repo-level train/val/test split
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | VLM model to use |
| `TENSOR_PARALLEL_SIZE` | `4` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | `50000` | Maximum sequence length for vLLM |
| `MAX_IMAGES_PER_PROMPT` | `30` | Maximum images per prompt |
| `CONDA_ENV` | `vagen` | Conda environment name |

Key settings in `config_trajectories.yaml.template`:

| Setting | Value | Description |
|---------|-------|-------------|
| `token_budget` | `20000` | Vision token budget per sample |
| `temperature` | `0.4` | Sampling temperature for diversity |
| `max_tokens` | `4096` | Maximum output tokens |
| `n_envs` | `125` | Number of samples (Clean+NT subset) |
| Trajectories | `6` | Number of rollouts per sample |

## Hardware Requirements

- **GPUs**: 4x A100 (80GB) or equivalent (for tensor-parallel vLLM serving)
- **RAM**: 128GB recommended
- **Storage**: ~2GB for dataset + rollouts
- **Time**: ~4 hours for 750 rollouts on 4x A100

## Next Steps: Training with GRPO

The trajectory rollouts are designed for Group Relative Policy Optimization (GRPO) training:

1. Each sample has 6 trajectories with the **same input images** but **varied model outputs** (temperature=0.4)
2. GRPO ranks trajectories within each group by reward (correct=1.0, incorrect=0.0)
3. The model learns to prefer trajectories that lead to correct answers
4. No separate critic/value model needed — rewards come directly from answer correctness

To train, use the VAGEN training pipeline with the generated train/val/test splits.
