"""
Build train/val/test datasets from CodeQA trajectory rollouts.

Reads the 6 trajectory rollout folders (traj_0 through traj_5),
splits samples by repo according to split_definition.json,
and writes train/val/test JSONL files.

Usage:
    python build_datasets.py \
        --rollout_dir /gpfs/projects/p32992/VAGEN/rollouts/codeqa_trajectories \
        --output_dir /gpfs/projects/p32992/VAGEN/data/codeqa_splits \
        --split_file /projects/p32992/VAGEN/vagen/envs/codeqa/utils/split_definition.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_split_definition(split_file: str) -> dict:
    with open(split_file) as f:
        return json.load(f)


def load_rollouts(rollout_dir: str, num_trajectories: int = 6) -> list:
    """Load all rollouts across all trajectory tags."""
    all_rollouts = []

    for traj_id in range(num_trajectories):
        tag_dir = Path(rollout_dir) / f"tag_traj_{traj_id}"
        if not tag_dir.exists():
            print(f"WARNING: {tag_dir} not found, skipping trajectory {traj_id}")
            continue

        # Load summary for this trajectory
        summary_path = tag_dir / "summary.json"
        if not summary_path.exists():
            print(f"WARNING: {summary_path} not found, scanning individual rollouts")
            # Fallback: scan individual rollout dirs
            for rollout_dir_entry in sorted(tag_dir.iterdir()):
                if not rollout_dir_entry.is_dir():
                    continue
                metrics_path = rollout_dir_entry / "metrics.json"
                messages_path = rollout_dir_entry / "messages.json"
                if not metrics_path.exists() or not messages_path.exists():
                    continue
                with open(metrics_path) as f:
                    metrics = json.load(f)
                with open(messages_path) as f:
                    messages = json.load(f)
                rollout = {
                    "trajectory_id": traj_id,
                    "rollout_id": rollout_dir_entry.name,
                    "rollout_path": str(rollout_dir_entry),
                    "metrics": metrics,
                    "messages": messages,
                }
                all_rollouts.append(rollout)
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        for ep in summary.get("episodes", []):
            rid = ep["rollout_id"]
            rollout_path = tag_dir / rid

            # Load full messages
            messages_path = rollout_path / "messages.json"
            if not messages_path.exists():
                print(f"WARNING: {messages_path} not found, skipping")
                continue
            with open(messages_path) as f:
                messages = json.load(f)

            # Extract key info from episode data
            turn0_info = ep["per_turn"][0]["info"]
            last_turn_info = ep["per_turn"][-1]["info"]

            rollout = {
                "trajectory_id": traj_id,
                "rollout_id": rid,
                "rollout_path": str(rollout_path),
                "seed": ep["seed"],
                "repo": turn0_info.get("repo", ""),
                "sample_id": turn0_info.get("sample_id", ""),
                "num_images": turn0_info.get("num_images", 0),
                "total_images_available": turn0_info.get("total_images_available", 0),
                "vision_tokens_used": turn0_info.get("vision_tokens_used", 0),
                "success": last_turn_info.get("success", False),
                "predicted_letter": last_turn_info.get("predicted_letter", ""),
                "correct_letter": last_turn_info.get("correct_letter", ""),
                "is_not_in_context": last_turn_info.get("is_not_in_context", False),
                "cumulative_reward": ep.get("cumulative_reward", 0.0),
                "messages": messages,
            }
            all_rollouts.append(rollout)

    return all_rollouts


def assign_splits(rollouts: list, split_def: dict) -> dict:
    """Assign each rollout to train/val/test based on repo."""
    train_repos = set(split_def["train"])
    val_repos = set(split_def["val"])
    test_repos = set(split_def["test"])

    splits = {"train": [], "val": [], "test": [], "unknown": []}

    for r in rollouts:
        repo = r["repo"]
        if repo in train_repos:
            splits["train"].append(r)
        elif repo in val_repos:
            splits["val"].append(r)
        elif repo in test_repos:
            splits["test"].append(r)
        else:
            splits["unknown"].append(r)

    return splits


def write_split(rollouts: list, output_path: str):
    """Write rollouts as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")


def print_stats(splits: dict):
    """Print split statistics."""
    print("\n" + "=" * 70)
    print("DATASET SPLIT STATISTICS")
    print("=" * 70)

    for split_name in ["train", "val", "test"]:
        rollouts = splits[split_name]
        if not rollouts:
            print(f"\n{split_name.upper()}: 0 rollouts")
            continue

        repos = defaultdict(list)
        for r in rollouts:
            repos[r["repo"]].append(r)

        correct = sum(1 for r in rollouts if r.get("success", False))
        total = len(rollouts)
        unique_questions = len(set(r["sample_id"] for r in rollouts))
        trajectories_per_q = total / unique_questions if unique_questions > 0 else 0

        print(f"\n{split_name.upper()}:")
        print(f"  Rollouts: {total}")
        print(f"  Unique questions: {unique_questions}")
        print(f"  Trajectories per question: {trajectories_per_q:.1f}")
        print(f"  Unique repos: {len(repos)}")
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"  Avg vision tokens: {sum(r.get('vision_tokens_used', 0) for r in rollouts) / total:,.0f}")
        print(f"  Avg images: {sum(r.get('num_images', 0) for r in rollouts) / total:.1f}")

        # Per-repo breakdown
        sampled = sum(1 for repo, rs in repos.items()
                      if rs[0].get("num_images", 0) < rs[0].get("total_images_available", 0))
        unsampled = len(repos) - sampled
        print(f"  Sampled repos: {sampled}, Unsampled repos: {unsampled}")

    if splits["unknown"]:
        print(f"\nWARNING: {len(splits['unknown'])} rollouts from unknown repos!")
        unknown_repos = set(r["repo"] for r in splits["unknown"])
        for repo in unknown_repos:
            print(f"  - {repo}")


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test datasets from CodeQA rollouts")
    parser.add_argument("--rollout_dir",
                        default="/gpfs/projects/p32992/VAGEN/rollouts/codeqa_trajectories",
                        help="Directory containing tag_traj_* subdirectories")
    parser.add_argument("--output_dir",
                        default="/gpfs/projects/p32992/VAGEN/data/codeqa_splits",
                        help="Output directory for split JSONL files")
    parser.add_argument("--split_file",
                        default="/projects/p32992/VAGEN/vagen/envs/codeqa/utils/split_definition.json",
                        help="Path to split definition JSON")
    parser.add_argument("--num_trajectories", type=int, default=6,
                        help="Number of trajectory tags to load (default: 6)")
    args = parser.parse_args()

    # Load split definition
    print(f"Loading split definition from {args.split_file}")
    split_def = load_split_definition(args.split_file)
    print(f"  Train repos: {len(split_def['train'])}")
    print(f"  Val repos: {len(split_def['val'])}")
    print(f"  Test repos: {len(split_def['test'])}")

    # Load all rollouts
    print(f"\nLoading rollouts from {args.rollout_dir}")
    rollouts = load_rollouts(args.rollout_dir, args.num_trajectories)
    print(f"  Total rollouts loaded: {len(rollouts)}")

    if not rollouts:
        print("ERROR: No rollouts found!")
        return

    # Assign to splits
    splits = assign_splits(rollouts, split_def)

    # Print stats
    print_stats(splits)

    # Write output files
    os.makedirs(args.output_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        if not splits[split_name]:
            continue
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_split(splits[split_name], output_path)
        print(f"\nWrote {len(splits[split_name])} rollouts to {output_path}")

    # Write a metadata file
    meta = {
        "split_definition": args.split_file,
        "rollout_dir": args.rollout_dir,
        "num_trajectories": args.num_trajectories,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
