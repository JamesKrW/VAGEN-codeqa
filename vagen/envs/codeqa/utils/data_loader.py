from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from vagen.envs.codeqa.utils.stratified_sampler import stratified_subsample

# Module-level caches to avoid reloading for every env instance
_DATASET_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CONTAMINATED_CACHE: Dict[str, Set[str]] = {}
_NONTRUNCATED_CACHE: Dict[str, Dict[str, Any]] = {}


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset, cached by path."""
    if dataset_path not in _DATASET_CACHE:
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        _DATASET_CACHE[dataset_path] = data
    return _DATASET_CACHE[dataset_path]


def load_contaminated_ids(contaminated_path: str) -> Set[str]:
    """Load contaminated question IDs, cached by path."""
    if contaminated_path not in _CONTAMINATED_CACHE:
        if not os.path.exists(contaminated_path):
            _CONTAMINATED_CACHE[contaminated_path] = set()
        else:
            with open(contaminated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _CONTAMINATED_CACHE[contaminated_path] = set(data.get("question_ids", []))
    return _CONTAMINATED_CACHE[contaminated_path]


def load_nontruncated_info(nontruncated_path: str) -> Dict[str, Any]:
    """Load non-truncated repo info, cached by path."""
    if nontruncated_path not in _NONTRUNCATED_CACHE:
        if not os.path.exists(nontruncated_path):
            _NONTRUNCATED_CACHE[nontruncated_path] = {"repos": [], "sample_ids": []}
        else:
            with open(nontruncated_path, "r", encoding="utf-8") as f:
                _NONTRUNCATED_CACHE[nontruncated_path] = json.load(f)
    return _NONTRUNCATED_CACHE[nontruncated_path]


def filter_dataset(
    data: List[Dict[str, Any]],
    subset: str,
    contaminated_ids: Set[str],
    nontruncated_sample_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Filter dataset based on subset type.

    Subsets:
      "all"             -> all 443 samples
      "clean"           -> non-contaminated only (203)
      "non_truncated"   -> non-truncated repos only (264)
      "clean_nt"        -> both non-contaminated AND non-truncated (125)
    """
    if subset == "all":
        filtered = list(data)
    elif subset == "clean":
        filtered = [d for d in data if d["id"] not in contaminated_ids]
    elif subset == "non_truncated":
        filtered = [d for d in data if d["id"] in nontruncated_sample_ids]
    elif subset == "clean_nt":
        filtered = [
            d
            for d in data
            if d["id"] not in contaminated_ids and d["id"] in nontruncated_sample_ids
        ]
    else:
        raise ValueError(
            f"Unknown subset: {subset}. Use: all, clean, non_truncated, clean_nt"
        )

    # Sort deterministically by ID for consistent seed-to-sample mapping
    filtered.sort(key=lambda d: d["id"])
    return filtered


def get_repo_dir_name(repo: str) -> str:
    """Convert repo name (owner/name) to directory name (owner_name)."""
    return repo.replace("/", "_")


def load_repo_images(
    repo: str,
    images_dir: str,
    max_vision_tokens: int = 230000,
) -> Tuple[List[Image.Image], int, int]:
    """
    Load PIL images for a repository, respecting vision token budget.

    Returns: (images_list, total_available, tokens_used)
    """
    repo_dir_name = get_repo_dir_name(repo)
    repo_dir = Path(images_dir) / repo_dir_name

    if not repo_dir.exists():
        return [], 0, 0

    mapping_file = repo_dir / "image_mapping.json"
    if not mapping_file.exists():
        # Fallback: load all chunk_*.png files sorted by name
        image_files = sorted(repo_dir.glob("chunk_*.png"))
        images = []
        for f in image_files:
            img = Image.open(f)
            images.append(img.copy())
            img.close()
        # Estimate ~1500 tokens per image when no metadata available
        return images, len(image_files), len(images) * 1500

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    total_available = mapping.get("num_images", 0)
    images_info = mapping.get("images", [])

    selected_images: List[Image.Image] = []
    tokens_used = 0

    for img_info in images_info:
        img_tokens = img_info.get("vision_tokens", 1500)
        if tokens_used + img_tokens > max_vision_tokens:
            break
        img_path = repo_dir / img_info["filename"]
        if img_path.exists():
            img = Image.open(img_path)
            selected_images.append(img.copy())
            img.close()
            tokens_used += img_tokens

    return selected_images, total_available, tokens_used


def load_repo_images_stratified(
    repo: str,
    images_dir: str,
    context: str,
    token_budget: int = 20000,
    seed: Optional[int] = None,
) -> Tuple[List[Image.Image], int, int]:
    """
    Load a stratified subsample of images for a repo within a token budget.

    For repos under budget, loads all images (same as load_repo_images).
    For larger repos, uses stratified sampling across file categories.

    Returns: (images_list, total_available, tokens_used)
    """
    repo_dir_name = get_repo_dir_name(repo)
    repo_dir = Path(images_dir) / repo_dir_name

    if not repo_dir.exists():
        return [], 0, 0

    mapping_file = repo_dir / "image_mapping.json"
    if not mapping_file.exists():
        # Fallback to loading all images
        return load_repo_images(repo, images_dir, token_budget)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    total_available = mapping.get("num_images", 0)

    # Get stratified chunk indices
    selected_indices = stratified_subsample(
        context=context,
        image_mapping=mapping,
        token_budget=token_budget,
        seed=seed,
    )

    images_info = mapping.get("images", [])
    selected_images: List[Image.Image] = []
    tokens_used = 0

    for idx in selected_indices:
        if idx >= len(images_info):
            continue
        img_info = images_info[idx]
        img_path = repo_dir / img_info["filename"]
        if img_path.exists():
            img = Image.open(img_path)
            selected_images.append(img.copy())
            img.close()
            tokens_used += img_info.get("vision_tokens", 0)

    return selected_images, total_available, tokens_used
