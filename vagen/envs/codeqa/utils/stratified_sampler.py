"""
Stratified image subsampler for CodeQA.

Subsamples code images from a repo to fit within a token budget (~20K),
preserving proportional representation of file categories (source, docs,
config, test, etc.).
"""

import re
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def categorize_file(fname: str) -> str:
    """Categorize a file by type based on its name/extension."""
    if fname.endswith('.py'):
        if 'test' in fname.lower():
            return 'test'
        elif '__init__' in fname:
            return 'init'
        elif fname.startswith(('setup', 'conf')):
            return 'config_py'
        else:
            return 'source'
    elif any(fname.endswith(ext) for ext in ['.toml', '.cfg', '.ini', '.yaml', '.yml']):
        return 'config'
    elif any(fname.endswith(ext) for ext in ['.md', '.rst', '.txt']):
        return 'docs'
    else:
        return 'other'


def map_chunks_to_categories(
    context: str,
    lines_per_chunk: int = 125,
) -> Dict[int, str]:
    """
    Map each chunk index to its dominant file category.

    Parses [start of <file>] markers in the context to determine
    which files each chunk contains, then assigns the dominant category.

    Returns dict of {chunk_index: category_name}.
    """
    lines = context.split('\n')
    num_chunks = (len(lines) + lines_per_chunk - 1) // lines_per_chunk

    # Track current file for each line
    current_cat = None
    line_categories = []
    for line in lines:
        m = re.match(r'\[start of (.+?)\]', line)
        if m:
            current_cat = categorize_file(m.group(1))
        line_categories.append(current_cat)

    # Assign dominant category per chunk
    chunk_to_cat = {}
    for c in range(num_chunks):
        start = c * lines_per_chunk
        end = min(start + lines_per_chunk, len(lines))
        cats = [line_categories[i] for i in range(start, end) if line_categories[i]]
        if cats:
            chunk_to_cat[c] = Counter(cats).most_common(1)[0][0]
        else:
            chunk_to_cat[c] = 'other'

    return chunk_to_cat


def stratified_subsample(
    context: str,
    image_mapping: Dict,
    token_budget: int = 20000,
    seed: Optional[int] = None,
    lines_per_chunk: int = 125,
) -> List[int]:
    """
    Select a stratified subset of chunk indices that fits within token_budget.

    For repos already under budget, returns all chunk indices.
    For larger repos, selects chunks proportionally from each file category.

    Parameters
    ----------
    context : str
        Full repo context text with [start of <file>] markers.
    image_mapping : dict
        Image mapping JSON with per-image vision_tokens info.
    token_budget : int
        Target token budget (default: 20000).
    seed : int, optional
        Random seed for reproducible subsampling. If None, random.
    lines_per_chunk : int
        Lines per image chunk (default: 125).

    Returns
    -------
    list of int
        Sorted list of selected chunk indices.
    """
    images = image_mapping['images']
    total_tokens = image_mapping['vision_tokens']['total']
    num_images = image_mapping['num_images']

    # If already under budget, return all
    if total_tokens <= token_budget:
        return list(range(num_images))

    # Map chunks to categories
    chunk_to_cat = map_chunks_to_categories(context, lines_per_chunk)

    # Group chunks by category with their token costs
    cat_chunks = defaultdict(list)
    for idx in range(min(num_images, len(chunk_to_cat))):
        cat = chunk_to_cat.get(idx, 'other')
        tokens = images[idx]['vision_tokens'] if idx < len(images) else 0
        cat_chunks[cat].append({'index': idx, 'tokens': tokens})

    # Calculate proportional budget per category
    cat_tokens = {}
    for cat, chunks in cat_chunks.items():
        cat_tokens[cat] = sum(c['tokens'] for c in chunks)

    rng = random.Random(seed)
    selected = []
    remaining_budget = token_budget

    # First pass: allocate proportionally, shuffle within category for diversity
    for cat, chunks in cat_chunks.items():
        proportion = cat_tokens[cat] / total_tokens
        cat_budget = int(token_budget * proportion)

        # Shuffle chunks within category for variety across seeds
        shuffled = list(chunks)
        rng.shuffle(shuffled)

        cat_selected = []
        cat_used = 0
        for c in shuffled:
            if cat_used + c['tokens'] <= cat_budget:
                cat_selected.append(c['index'])
                cat_used += c['tokens']

        selected.extend(cat_selected)
        remaining_budget -= cat_used

    # Second pass: fill remaining budget with unselected chunks
    selected_set = set(selected)
    unselected = [
        {'index': idx, 'tokens': images[idx]['vision_tokens']}
        for idx in range(num_images)
        if idx not in selected_set
    ]
    rng.shuffle(unselected)

    for c in unselected:
        if remaining_budget >= c['tokens']:
            selected.append(c['index'])
            remaining_budget -= c['tokens']

    return sorted(selected)
