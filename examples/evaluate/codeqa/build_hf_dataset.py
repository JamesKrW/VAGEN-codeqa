"""
Convert CodeQA rollout splits to HuggingFace datasets format.

Reads the JSONL splits (train/val/test), references images by file path,
and saves as a HF DatasetDict. Processes one split at a time to limit memory.

Usage:
    python build_hf_dataset.py \
        --splits_dir /gpfs/projects/p32992/VAGEN/data/codeqa_splits_v2 \
        --output_dir /gpfs/projects/p32992/VAGEN/data/codeqa_hf_dataset

    # Then push to HuggingFace Hub:
    # python build_hf_dataset.py --push_to_hub --hub_repo your-username/codeqa-trajectories
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value


def load_split(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_image_paths(rollout_path: str) -> list:
    """Get sorted list of image file paths from a rollout directory."""
    img_dir = os.path.join(rollout_path, "images")
    if not os.path.exists(img_dir):
        return []
    paths = []
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.endswith(".png"):
            paths.append(os.path.join(img_dir, img_name))
    return paths


def extract_text_from_content(content) -> str:
    """Extract text from a message content field (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)
    return ""


def convert_records(records: list) -> dict:
    """Convert rollout records to columnar format for HF Dataset."""
    columns = {
        "trajectory_id": [],
        "rollout_id": [],
        "seed": [],
        "repo": [],
        "sample_id": [],
        "num_images": [],
        "total_images_available": [],
        "vision_tokens_used": [],
        "success": [],
        "predicted_letter": [],
        "correct_letter": [],
        "is_not_in_context": [],
        "cumulative_reward": [],
        "system_prompt": [],
        "ocr_instruction": [],
        "ocr_response": [],
        "qa_question": [],
        "qa_response": [],
        "images": [],
    }

    for i, r in enumerate(records):
        if i % 50 == 0:
            print(f"  Processing record {i}/{len(records)}...")

        columns["trajectory_id"].append(r["trajectory_id"])
        columns["rollout_id"].append(r["rollout_id"])
        columns["seed"].append(r["seed"])
        columns["repo"].append(r["repo"])
        columns["sample_id"].append(r["sample_id"])
        columns["num_images"].append(r["num_images"])
        columns["total_images_available"].append(r["total_images_available"])
        columns["vision_tokens_used"].append(r["vision_tokens_used"])
        columns["success"].append(r["success"])
        columns["predicted_letter"].append(r["predicted_letter"])
        columns["correct_letter"].append(r["correct_letter"])
        columns["is_not_in_context"].append(r["is_not_in_context"])
        columns["cumulative_reward"].append(r["cumulative_reward"])

        # Extract conversation text
        msgs = r.get("messages", [])
        columns["system_prompt"].append(
            extract_text_from_content(msgs[0]["content"]) if len(msgs) > 0 else ""
        )
        columns["ocr_instruction"].append(
            extract_text_from_content(msgs[1]["content"]) if len(msgs) > 1 else ""
        )
        columns["ocr_response"].append(
            extract_text_from_content(msgs[2]["content"]) if len(msgs) > 2 else ""
        )
        columns["qa_question"].append(
            extract_text_from_content(msgs[3]["content"]) if len(msgs) > 3 else ""
        )
        columns["qa_response"].append(
            extract_text_from_content(msgs[4]["content"]) if len(msgs) > 4 else ""
        )

        # Store image file paths (HF datasets will encode them lazily)
        columns["images"].append(get_image_paths(r["rollout_path"]))

    return columns


def process_split(split_name: str, jsonl_path: str, output_dir: str, features):
    """Process and save one split at a time to limit memory usage."""
    print(f"\nProcessing {split_name}...")
    records = load_split(jsonl_path)
    print(f"  Loaded {len(records)} records")

    columns = convert_records(records)

    # Create dataset — HF will read images from paths lazily
    ds = Dataset.from_dict(columns, features=features)
    print(f"  Created dataset with {len(ds)} rows")

    # Save this split
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    ds.save_to_disk(split_dir)
    print(f"  Saved to {split_dir}")

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Convert CodeQA splits to HuggingFace dataset"
    )
    parser.add_argument(
        "--splits_dir",
        default="/gpfs/projects/p32992/VAGEN/data/codeqa_splits_v2",
    )
    parser.add_argument(
        "--output_dir",
        default="/gpfs/projects/p32992/VAGEN/data/codeqa_hf_dataset",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", type=str, default="")
    args = parser.parse_args()

    features = Features(
        {
            "trajectory_id": Value("int32"),
            "rollout_id": Value("string"),
            "seed": Value("int32"),
            "repo": Value("string"),
            "sample_id": Value("string"),
            "num_images": Value("int32"),
            "total_images_available": Value("int32"),
            "vision_tokens_used": Value("int32"),
            "success": Value("bool"),
            "predicted_letter": Value("string"),
            "correct_letter": Value("string"),
            "is_not_in_context": Value("bool"),
            "cumulative_reward": Value("float32"),
            "system_prompt": Value("string"),
            "ocr_instruction": Value("string"),
            "ocr_response": Value("string"),
            "qa_question": Value("string"),
            "qa_response": Value("string"),
            "images": Sequence(Image()),
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)

    splits = {}
    for split_name in ["train", "val", "test"]:
        jsonl_path = os.path.join(args.splits_dir, f"{split_name}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Skipping {split_name} (not found)")
            continue
        ds = process_split(split_name, jsonl_path, args.output_dir, features)
        splits[split_name] = ds

    if args.push_to_hub and args.hub_repo:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        dataset_dict = DatasetDict(splits)
        dataset_dict.push_to_hub(args.hub_repo)
        print("Done!")
    elif args.push_to_hub:
        print("\nWARNING: --push_to_hub set but no --hub_repo provided")

    print("\nComplete!")
    for name, ds in splits.items():
        print(f"  {name}: {len(ds)} rows")


if __name__ == "__main__":
    main()
