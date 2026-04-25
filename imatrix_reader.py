"""Read importance matrix data from GGUF or legacy .dat format."""

import re
import struct
from typing import List, Tuple, Dict, Set
import numpy as np

from tensor_mappings import get_column_source_patterns


def compute_importance(in_sum2: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Compute importance as sqrt(in_sum2 / count) - RMS value."""
    count = np.maximum(count, 1)
    # Handle broadcasting - if count is shorter, repeat it
    if len(count) < len(in_sum2) and len(in_sum2) % len(count) == 0:
        repeat_factor = len(in_sum2) // len(count)
        count = np.repeat(count, repeat_factor)
    return np.sqrt(in_sum2 / count)


def _is_gguf_file(filepath: str) -> bool:
    """Check if file is GGUF format by reading magic bytes."""
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            return magic == b"GGUF"
    except Exception:
        return False


def _read_legacy_imatrix(filepath: str) -> Dict[str, np.ndarray]:
    """Read legacy .dat imatrix file."""
    tensors = {}
    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError(f"File too small: {filepath}")

    offset = 0
    n_entries = struct.unpack("i", data[offset : offset + 4])[0]
    offset += 4

    for _ in range(n_entries):
        # Read name
        name_len = struct.unpack("i", data[offset : offset + 4])[0]
        offset += 4
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len

        # Read ncall and nval
        ncall = struct.unpack("i", data[offset : offset + 4])[0]
        offset += 4
        nval = struct.unpack("i", data[offset : offset + 4])[0]
        offset += 4

        # Read values
        values = np.frombuffer(
            data[offset : offset + nval * 4], dtype=np.float32
        ).astype(np.float64)
        offset += nval * 4

        # Convert to importance
        mean_squared = np.maximum(values, 0) / max(ncall, 1)
        tensors[name] = np.sqrt(mean_squared)

    return tensors


def _extract_block_idx(base_name: str) -> int:
    """Extract block index from tensor name."""
    parts = base_name.split(".")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def _read_tensor_data(
    tensor_map: Dict, base_name: str
) -> List[Tuple[np.ndarray, np.ndarray, str, int]]:
    """Read tensor data from GGUF.

    Returns:
        List of (in_sum2, count, name, expert_idx) tuples.
        expert_idx is -1 for non-expert tensors, 0-n for expert tensors.
    """
    in_sum2_tensor = tensor_map[base_name + ".in_sum2"]
    count_tensor = tensor_map[base_name + ".counts"]
    in_sum2 = in_sum2_tensor.data
    count = count_tensor.data

    # Check if this is an expert tensor (has _exps in name)
    is_expert = "_exps" in base_name

    if is_expert and in_sum2.ndim == 2:
        # Expert tensor: shape is [num_experts, features]
        # e.g., in_sum2: (256, 512) = [num_experts, intermediate_dim]
        #       count: (256, 1)
        num_experts = in_sum2.shape[0]
        results = []
        for i in range(num_experts):
            # Use base_name directly (no fake expert index in name)
            # Pass expert_idx separately
            in_sum2_exp = in_sum2[i, :]
            # Count is per-expert, shape is (num_experts, 1)
            count_exp = (
                np.array([count[i, 0]]) if count.ndim == 2 else np.array([count[i]])
            )
            results.append((in_sum2_exp, count_exp, base_name, i))
        return results
    elif in_sum2.ndim == 2:
        # 2D layout: flatten to 1D
        in_sum2 = in_sum2.reshape(-1)
        count = count.reshape(-1)
        return [(in_sum2, count, base_name, -1)]
    else:
        # Simple 1D layout
        in_sum2 = in_sum2.reshape(-1)
        count = count.reshape(-1)
        return [(in_sum2, count, base_name, -1)]


class ImatrixData:
    """Container for imatrix data with column tensors."""

    def __init__(self):
        # List of (name, values, expert_idx) tuples for column source tensors
        # expert_idx is -1 for non-expert tensors, 0-n for expert tensors
        self.column_tensors: List[Tuple[str, List[float], int]] = []
        # Set of all tensor names available in the imatrix
        self.available_tensors: Set[str] = set()

    def add_column_tensor(self, name: str, values: List[float], expert_idx: int = -1):
        """Add a column tensor.

        Args:
            name: Tensor name (without expert index embedded)
            values: Importance values
            expert_idx: Expert index (-1 for non-expert, 0-n for expert tensors)
        """
        self.column_tensors.append((name, values, expert_idx))


def _is_column_source(tensor_name: str) -> bool:
    """Check if tensor name contains any column source pattern."""
    base_name = tensor_name.replace(".in_sum2", "").replace(".counts", "")
    for pattern in get_column_source_patterns():
        if pattern in base_name:
            return True
    return False


def read_imatrix(filepath: str) -> ImatrixData:
    """Read tensor data from GGUF or legacy .dat imatrix file."""
    result = ImatrixData()

    if _is_gguf_file(filepath):
        # Read GGUF format
        from gguf.gguf_reader import GGUFReader

        reader = GGUFReader(filepath)
        tensor_map = {tensor.name: tensor for tensor in reader.tensors}
        result.available_tensors = set(tensor_map.keys())

        # Find all column source tensors
        for name in tensor_map.keys():
            base_name = name.replace(".in_sum2", "").replace(".counts", "")
            if not name.endswith(".in_sum2"):
                continue

            if _is_column_source(name):
                # Read and process the column tensor data
                for in_sum2, count, output_name, expert_idx in _read_tensor_data(
                    tensor_map, base_name
                ):
                    importance = compute_importance(in_sum2, count)
                    result.add_column_tensor(
                        output_name, importance.tolist(), expert_idx
                    )

        return result
    else:
        # Read legacy .dat format
        tensors = _read_legacy_imatrix(filepath)
        result.available_tensors = set(tensors.keys())

        # Process all column tensors from legacy format
        for name, values in tensors.items():
            if _is_column_source(name):
                result.add_column_tensor(name, values.tolist(), -1)

        return result
