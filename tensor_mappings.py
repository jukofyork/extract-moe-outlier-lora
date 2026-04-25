"""Tensor name mappings using regex patterns.

A permutation matrix P sits between source tensors and target tensors:
- Source tensors get P applied to columns
- Target tensors get P^T applied to rows

Patterns use explicit block numbers and alternation for variants:
  blk\.0\.ffn_down\.(weight|bias) blk\.0\.ffn_(up|gate|gate_up)\.(weight|bias)
"""

import re
from typing import List, Tuple, Optional


# (source_pattern, target_pattern_with_alternation)
TENSOR_RELATIONSHIPS = [
    (r"ffn_down\.(weight|bias)", r"ffn_(up|gate|gate_up)\.(weight|bias)"),
    (r"ffn_down_shexp\.(weight|bias)", r"ffn_(up|gate|gate_up)_shexp\.(weight|bias)"),
    (r"ffn_down_exps\.(weight|bias)", r"ffn_(up|gate|gate_up)_exps\.(weight|bias)"),
]


def get_column_source_patterns() -> List[str]:
    """Get source substrings for finding column sources in imatrix.

    Returns simple substring patterns (not regex) for initial matching.
    Extracts just the base tensor name (e.g., "ffn_down" from the pattern).
    """
    patterns = []
    for source_pat, _ in TENSOR_RELATIONSHIPS:
        # Extract base name before ".(weight|bias)"
        # e.g., "ffn_down\.(weight|bias)" -> "ffn_down"
        base = source_pat.replace(r"\.(weight|bias)", "")
        patterns.append(base)
    return patterns


def classify_tensor(name: str) -> Optional[Tuple[str, str, bool]]:
    """Classify tensor and return (source_regex, target_regex, is_expert).

    For expert tensors (ffn_*_exps), the source and target patterns match
    the 3D model tensor names (without expert index in the name).

    Args:
        name: Tensor name (may include .in_sum2 or .counts suffix)

    Returns:
        Tuple of (source_regex, target_regex, is_expert) ready for output file,
        where is_expert is True for expert tensors (ffn_*_exps), False otherwise,
        or None if tensor doesn't match any known relationship.
    """
    # Strip suffixes
    base_name = name.replace(".in_sum2", "").replace(".counts", "")

    # Try to match against each relationship
    for source_pat, target_pat in TENSOR_RELATIONSHIPS:
        # Check if source pattern matches anywhere in the name
        match = re.search(source_pat, base_name)
        if match:
            # Extract block number (e.g., "blk.0" from "blk.0.ffn_down.weight")
            block = _extract_block(base_name)
            if block:
                # Build full patterns with block prefix
                source_regex = f"{re.escape(block)}\\.{source_pat}"
                target_regex = f"{re.escape(block)}\\.{target_pat}"
                # Check if this is an expert tensor
                is_expert = "_exps" in source_pat
                return (source_regex, target_regex, is_expert)

    return None


def _extract_block(name: str) -> Optional[str]:
    """Extract block identifier from tensor name.

    Examples:
        "blk.0.ffn_down.weight" -> "blk.0"
        "blk.10.attn_output.weight" -> "blk.10"
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.startswith("blk") and i + 1 < len(parts):
            # Handle both "blk.0" and "blk.10" (multi-digit)
            block_num = parts[i + 1]
            if block_num.isdigit():
                return f"{part}.{block_num}"
    return None
