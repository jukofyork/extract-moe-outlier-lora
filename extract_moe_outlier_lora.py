"""
Extract outlier-compensation LoRA adapters from MoE models.

This tool identifies outlier columns in MoE MLP down-projection weights
(either via heuristic or sampled imatrix importances) and extracts them
as rank-N LoRA adapters. The residual model has these outliers subtracted
out, making the remaining weights easier to quantize.

Two-pass approach:
1. Pass 1: Generate all LoRA tensors (load tensors, compute, unload immediately)
2. Pass 2: Apply subtractions to each safetensors file and save residuals

Usage:
    python extract_moe_outlier_lora.py \
        --input models/Your-MoE-Model \
        --output models/my_model \
        --quant-type F32
"""

import os
import json
import re
import shutil
from argparse import ArgumentParser
from typing import Dict, Optional, Set, Tuple

import torch
import numpy as np
from safetensors.torch import load_file, save_file

import gguf


def parse_expert_info(weight_name: str) -> Tuple[int, int, str]:
    """
    Parse layer number, expert number, and projection type from weight name.

    Returns:
        (layer_num, expert_num, proj_type) or None if not an expert weight
    """
    match = re.match(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
        weight_name,
    )
    if match:
        layer_num = int(match.group(1))
        expert_num = int(match.group(2))
        proj_type = match.group(3)
        return layer_num, expert_num, proj_type
    return None


def extract_gate_up_lora(weight_f32: torch.Tensor, routing_f32: torch.Tensor):
    """
    Extract rank-1 LoRA for gate_proj or up_proj using routing vector.

    Args:
        weight_f32: [output_dim, input_dim] - expert weight in F32
        routing_f32: [input_dim] - routing vector in F32

    Returns:
        lora_a: [input_dim] - normalized routing vector (F32)
        lora_b: [output_dim] - alignment scores (F32)
    """
    # Normalize routing vector
    r_norm = torch.norm(routing_f32)
    if r_norm < 1e-10:
        r = routing_f32.clone()
    else:
        r = routing_f32 / r_norm

    # Compute alignment
    b = weight_f32 @ r

    return r, b


def extract_down_lora_from_importance(
    down_f32: torch.Tensor, importance: torch.Tensor, rank: int
):
    """
    Extract rank-N LoRA for down_proj using importance values.

    Args:
        down_f32: [output_dim, input_dim] - down_proj weight in F32
        importance: [input_dim] - importance values for each input feature
        rank: Rank for the LoRA

    Returns:
        lora_a: [rank, input_dim] with 1.0 at selected indices
        lora_b: [output_dim, rank] with columns at selected indices
        selected_indices: list of selected indices
    """
    if rank > importance.numel():
        raise ValueError(
            f"down_rank ({rank}) exceeds input dimension ({importance.numel()})"
        )

    # Find top-N most important indices
    topk = torch.topk(importance, k=rank)
    selected_indices = topk.indices.tolist()

    # Build lora_a as [rank, input_dim] with 1.0 at selected indices
    lora_a = torch.zeros(rank, down_f32.shape[1], dtype=torch.float32)
    # Build lora_b as [output_dim, rank] with columns at selected indices
    lora_b = torch.zeros(down_f32.shape[0], rank, dtype=torch.float32)

    for i, idx in enumerate(selected_indices):
        lora_a[i, idx] = 1.0
        lora_b[:, i] = down_f32[:, idx]

    return lora_a, lora_b, selected_indices


def extract_down_lora(
    down_f32: torch.Tensor, b_gate: torch.Tensor, b_up: torch.Tensor, rank: int = 1
):
    """
    Extract rank-N LoRA for down_proj using heuristic (b_gate * b_up).

    Args:
        down_f32: [output_dim, input_dim] - down_proj weight in F32
        b_gate: [input_dim] - alignment from gate_proj
        b_up: [input_dim] - alignment from up_proj
        rank: Rank for the LoRA (1 for single outlier, >1 for top-N)

    Returns:
        lora_a: [input_dim] for rank=1 or [rank, input_dim] for rank>1
        lora_b: [output_dim] for rank=1 or [output_dim, rank] for rank>1
        outlier_indices: list of selected indices
    """
    combined = b_gate * b_up
    return extract_down_lora_from_importance(down_f32, combined, rank)


def load_tensor(weight_map: Dict[str, str], input_folder: str, weight_name: str):
    """Load a single tensor from its file."""
    filename = weight_map[weight_name]
    filepath = os.path.join(input_folder, filename)
    state_dict = load_file(filepath, device="cpu")
    return state_dict[weight_name]


def discover_moe_architecture(
    weight_map: Dict[str, str],
) -> Tuple[Set[int], Dict[int, Set[int]]]:
    """
    Discover MoE layers and their experts from the weight map.

    Returns:
        (moe_layers, experts_in_layer) where:
        - moe_layers: set of layer numbers with MoE
        - experts_in_layer: dict mapping layer_num -> set of expert indices
    """
    moe_layers = set()
    experts_in_layer: Dict[int, Set[int]] = {}

    for weight_name in weight_map.keys():
        if ".mlp.gate.weight" in weight_name and "experts" not in weight_name:
            layer_num = int(weight_name.split(".layers.")[1].split(".")[0])
            moe_layers.add(layer_num)

        info = parse_expert_info(weight_name)
        if info:
            layer_num, expert_num, _ = info
            if layer_num not in experts_in_layer:
                experts_in_layer[layer_num] = set()
            experts_in_layer[layer_num].add(expert_num)

    return moe_layers, experts_in_layer


def _pass1_heuristic(
    input_folder: str,
    weight_map: Dict[str, str],
    moe_layers: Set[int],
    experts_in_layer: Dict[int, Set[int]],
    skip_up_gate: bool,
    down_rank: int,
) -> Dict:
    """
    Pass 1 (heuristic mode): Generate LoRA tensors using b_gate * b_up.

    For each MoE layer:
      - Load gate.weight
      - For each expert: load gate_proj, up_proj, compute and store b vectors
      - For each expert: load down_proj, compute LoRA using stored b vectors
      - Immediately unload weights after use
    """
    print("\n" + "=" * 60)
    print("PASS 1: Generating LoRA tensors (heuristic mode)")
    print("=" * 60)

    lora_map = {}

    print(f"Found {len(moe_layers)} MoE layers: {sorted(moe_layers)}")

    for layer_num in sorted(moe_layers):
        print(f"\nProcessing layer {layer_num}...")

        # Load gate.weight for this layer
        gate_weight_name = f"model.layers.{layer_num}.mlp.gate.weight"
        gate_weight = load_tensor(weight_map, input_folder, gate_weight_name)

        # gate.weight shape: [n_experts, hidden_size]
        n_experts = gate_weight.shape[0]
        print(f"  - {n_experts} experts")

        layer_experts = experts_in_layer.get(layer_num, set())

        # Process gate_proj and up_proj for all experts
        for expert_num in sorted(layer_experts):
            # Get routing vector for this expert
            routing_vector = gate_weight[expert_num, :].to(torch.float32)

            # Process gate_proj (always needed for b_gate)
            gate_proj_name = (
                f"model.layers.{layer_num}.mlp.experts.{expert_num}.gate_proj.weight"
            )
            gate_proj = load_tensor(weight_map, input_folder, gate_proj_name).to(
                torch.float32
            )
            r, b_gate = extract_gate_up_lora(gate_proj, routing_vector)

            # Store LoRA for gate_proj only if not skipping
            if not skip_up_gate:
                lora_map[(layer_num, expert_num, "gate_proj", "a")] = r
                lora_map[(layer_num, expert_num, "gate_proj", "b")] = b_gate

            del gate_proj  # Free memory

            # Process up_proj (always needed for b_up)
            up_proj_name = (
                f"model.layers.{layer_num}.mlp.experts.{expert_num}.up_proj.weight"
            )
            up_proj = load_tensor(weight_map, input_folder, up_proj_name).to(
                torch.float32
            )
            r, b_up = extract_gate_up_lora(up_proj, routing_vector)

            # Store LoRA for up_proj only if not skipping
            if not skip_up_gate:
                lora_map[(layer_num, expert_num, "up_proj", "a")] = r
                lora_map[(layer_num, expert_num, "up_proj", "b")] = b_up

            # Store b vectors for down_proj computation (always needed)
            lora_map[(layer_num, expert_num, "b_gate")] = b_gate
            lora_map[(layer_num, expert_num, "b_up")] = b_up

            del up_proj  # Free memory

        # Process down_proj for all experts
        for expert_num in sorted(layer_experts):
            # Get stored b vectors
            b_gate = lora_map[(layer_num, expert_num, "b_gate")]
            b_up = lora_map[(layer_num, expert_num, "b_up")]

            if down_rank > 0:
                # Process down_proj
                down_proj_name = f"model.layers.{layer_num}.mlp.experts.{expert_num}.down_proj.weight"
                down_proj = load_tensor(weight_map, input_folder, down_proj_name).to(
                    torch.float32
                )
                lora_a, lora_b, outlier_indices = extract_down_lora(
                    down_proj, b_gate, b_up, rank=down_rank
                )

                # Store LoRA for down_proj
                lora_map[(layer_num, expert_num, "down_proj", "a")] = lora_a
                lora_map[(layer_num, expert_num, "down_proj", "b")] = lora_b

                del down_proj  # Free memory

            # Clean up temporary b vectors
            del lora_map[(layer_num, expert_num, "b_gate")]
            del lora_map[(layer_num, expert_num, "b_up")]

        del gate_weight  # Free memory
        print(f"  - Complete")

    print(f"\nPass 1 complete. Generated {len(lora_map)} LoRA tensors.")
    return lora_map


def _pass1_imatrix(
    input_folder: str,
    weight_map: Dict[str, str],
    moe_layers: Set[int],
    experts_in_layer: Dict[int, Set[int]],
    down_rank: int,
    imatrix_data,
) -> Dict:
    """
    Pass 1 (imatrix mode): Generate down_proj LoRA tensors from sampled importances.

    Skips all gate/up processing. Uses real activation importances from imatrix
    to select the most important features for each expert's down_proj.
    """
    print("\n" + "=" * 60)
    print("PASS 1: Generating LoRA tensors (imatrix mode)")
    print("=" * 60)

    # Build lookup: (layer, expert) -> importance tensor
    imatrix_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
    for name, values, expert_idx in imatrix_data.column_tensors:
        if "ffn_down_exps" not in name:
            continue
        # Parse layer number from GGUF name, e.g., "blk.1.ffn_down_exps.weight"
        match = re.match(r"blk\.(\d+)\.ffn_down_exps\.weight", name)
        if not match:
            continue
        layer_num = int(match.group(1))
        if expert_idx < 0:
            continue
        importance = torch.tensor(values, dtype=torch.float32)
        imatrix_lookup[(layer_num, expert_idx)] = importance

    # Validate full coverage
    missing = []
    for layer_num in moe_layers:
        for expert_num in sorted(experts_in_layer.get(layer_num, [])):
            if (layer_num, expert_num) not in imatrix_lookup:
                missing.append((layer_num, expert_num))
    if missing:
        sample = missing[:5]
        raise ValueError(
            f"Imatrix missing data for {len(missing)} expert(s). "
            f"First few missing: {sample}"
        )

    lora_map = {}

    print(f"Found {len(moe_layers)} MoE layers: {sorted(moe_layers)}")

    for layer_num in sorted(moe_layers):
        print(f"\nProcessing layer {layer_num}...")
        layer_experts = experts_in_layer.get(layer_num, set())
        n_experts = len(layer_experts)
        print(f"  - {n_experts} experts")

        for expert_num in sorted(layer_experts):
            importance = imatrix_lookup[(layer_num, expert_num)]

            down_proj_name = (
                f"model.layers.{layer_num}.mlp.experts.{expert_num}.down_proj.weight"
            )
            down_proj = load_tensor(weight_map, input_folder, down_proj_name).to(
                torch.float32
            )

            lora_a, lora_b, _ = extract_down_lora_from_importance(
                down_proj, importance, down_rank
            )

            lora_map[(layer_num, expert_num, "down_proj", "a")] = lora_a
            lora_map[(layer_num, expert_num, "down_proj", "b")] = lora_b

            del down_proj

        print(f"  - Complete")

    print(f"\nPass 1 complete. Generated {len(lora_map)} LoRA tensors.")
    return lora_map


def pass1_generate_loras(
    input_folder: str,
    weight_map: Dict[str, str],
    skip_up_gate: bool,
    down_rank: int,
    imatrix_data=None,
) -> Dict:
    """
    Pass 1: Generate all LoRA tensors.

    Dispatches to heuristic mode or imatrix mode based on whether imatrix_data
    is provided.

    Returns:
        lora_map: Dict with keys (layer, expert, proj, 'a'/'b') -> tensor
    """
    moe_layers, experts_in_layer = discover_moe_architecture(weight_map)

    if imatrix_data is not None:
        return _pass1_imatrix(
            input_folder,
            weight_map,
            moe_layers,
            experts_in_layer,
            down_rank,
            imatrix_data,
        )
    else:
        return _pass1_heuristic(
            input_folder,
            weight_map,
            moe_layers,
            experts_in_layer,
            skip_up_gate,
            down_rank,
        )


def pass2_apply_subtractions(
    input_folder: str,
    output_folder: str,
    weight_map: Dict[str, str],
    lora_map: Dict,
    skip_up_gate: bool,
    down_rank: int,
):
    """
    Pass 2: Apply subtractions and save residual files.

    For each safetensors file:
      - Load file
      - For each expert weight: subtract LoRA reconstruction (if enabled)
      - Save modified file to output folder
    """
    print("\n" + "=" * 60)
    print("PASS 2: Applying subtractions and saving residuals")
    print("=" * 60)

    # Get unique filenames
    filenames = sorted(set(weight_map.values()))

    os.makedirs(output_folder, exist_ok=True)

    for filename in filenames:
        print(f"\nProcessing {filename}...")

        # Load file
        filepath = os.path.join(input_folder, filename)
        state_dict = load_file(filepath, device="cpu")

        new_state_dict = {}
        modified_count = 0

        for weight_name, weight in state_dict.items():
            # Check if this is an expert weight we need to modify
            info = parse_expert_info(weight_name)

            if info:
                layer_num, expert_num, proj_type = info

                # Skip subtraction for disabled projection types
                if proj_type in ("gate_proj", "up_proj") and skip_up_gate:
                    new_state_dict[weight_name] = weight
                    continue
                if proj_type == "down_proj" and down_rank == 0:
                    new_state_dict[weight_name] = weight
                    continue

                # Get LoRA tensors
                key_a = (layer_num, expert_num, proj_type, "a")
                key_b = (layer_num, expert_num, proj_type, "b")

                if key_a in lora_map and key_b in lora_map:
                    lora_a = lora_map[key_a]
                    lora_b = lora_map[key_b]

                    # Convert weight to F32 for computation
                    weight_f32 = weight.to(torch.float32)

                    # Compute reconstruction and subtract
                    if lora_a.dim() == 1:
                        # Rank-1: outer product
                        reconstruction = torch.outer(lora_b, lora_a)
                    else:
                        # Rank-N: lora_b @ lora_a
                        # lora_a is [rank, in_dim], lora_b is [out_dim, rank]
                        reconstruction = lora_b @ lora_a

                    residual_f32 = weight_f32 - reconstruction

                    # Convert back to original dtype (BF16)
                    residual = residual_f32.to(weight.dtype)

                    new_state_dict[weight_name] = residual
                    modified_count += 1
                else:
                    new_state_dict[weight_name] = weight
            else:
                # Non-expert weight, keep as-is
                new_state_dict[weight_name] = weight

        # Save modified file
        output_path = os.path.join(output_folder, filename)
        save_file(new_state_dict, output_path)
        print(f"  - Saved {output_path} ({modified_count} tensors modified)")

        # Free memory
        del state_dict, new_state_dict


def prepare_gguf_tensors(lora_map: Dict, proj_types: list) -> list:
    """
    Prepare LoRA tensors for GGUF export for specific projection types.

    Stacks tensors by layer and projection type.
    """
    print("\n" + "=" * 60)
    print(f"Preparing GGUF tensors for: {', '.join(proj_types)}")
    print("=" * 60)

    gguf_tensors = []

    # Find all layers in lora_map for the specified projection types
    layers = set()
    experts = {}  # layer -> set of experts

    for key in lora_map.keys():
        if len(key) == 4:  # (layer, expert, proj, 'a'/'b')
            layer, expert, proj, ab = key
            if proj in proj_types:
                layers.add(layer)
                if layer not in experts:
                    experts[layer] = set()
                experts[layer].add(expert)

    # Map projection types to GGUF names
    gguf_name_map = {
        "gate_proj": "ffn_gate_exps",
        "up_proj": "ffn_up_exps",
        "down_proj": "ffn_down_exps",
    }

    for layer_num in sorted(layers):
        expert_list = sorted(experts[layer_num])
        n_experts = len(expert_list)

        print(f"Layer {layer_num}: {n_experts} experts")

        for proj_type in proj_types:
            expert_tensors_a = []
            expert_tensors_b = []

            for expert_num in expert_list:
                key_a = (layer_num, expert_num, proj_type, "a")
                key_b = (layer_num, expert_num, proj_type, "b")

                tensor_a = lora_map.get(key_a)
                tensor_b = lora_map.get(key_b)

                if tensor_a is None or tensor_b is None:
                    print(
                        f"  Warning: Missing LoRA for expert {expert_num}, {proj_type}"
                    )
                    continue

                if tensor_a.dim() == 1:
                    expert_tensors_a.append(tensor_a.contiguous().unsqueeze(0))
                else:
                    expert_tensors_a.append(tensor_a.contiguous())
                if tensor_b.dim() == 1:
                    expert_tensors_b.append(tensor_b.contiguous().unsqueeze(-1))
                else:
                    expert_tensors_b.append(tensor_b.contiguous())

            if not expert_tensors_a:
                continue

            # Stack all experts - use stack to preserve rank dimension
            stacked_a = torch.stack(expert_tensors_a, dim=0)
            stacked_b = torch.stack(expert_tensors_b, dim=0)

            print(f"  {proj_type}: A {stacked_a.shape}, B {stacked_b.shape}")

            # Add to GGUF tensors
            gguf_name = f"blk.{layer_num}.{gguf_name_map[proj_type]}.weight"
            gguf_tensors.append((f"{gguf_name}.lora_a", stacked_a))
            gguf_tensors.append((f"{gguf_name}.lora_b", stacked_b))

    return gguf_tensors


def export_lora_gguf(
    path: str, tensors: list, alpha: int = 1, quant_type=gguf.GGMLQuantizationType.F32
):
    """Export LoRA tensors to GGUF format."""
    print(f"\nExporting GGUF to: {path}")

    writer = gguf.GGUFWriter(path, "deepseek2")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", alpha)

    for name, tensor in tensors:
        print(f"  - {name}: {tensor.shape}")

        if quant_type in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]:
            dtype = (
                np.float32
                if quant_type == gguf.GGMLQuantizationType.F32
                else np.float16
            )
            writer.add_tensor(name, tensor.numpy().astype(dtype))
        else:
            quant_tensor = gguf.quants.quantize(tensor.numpy(), quant_type)
            writer.add_tensor(
                name, quant_tensor, raw_shape=quant_tensor.shape, raw_dtype=quant_type
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("GGUF export complete")


def copy_non_safetensors(input_folder: str, output_folder: str):
    """Copy all files except .safetensors from input to output folder recursively."""
    for root, dirs, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        out_dir = (
            os.path.join(output_folder, rel_path) if rel_path != "." else output_folder
        )
        os.makedirs(out_dir, exist_ok=True)
        for file in files:
            if not file.endswith(".safetensors"):
                src = os.path.join(root, file)
                dst = os.path.join(out_dir, file)
                shutil.copy2(src, dst)


def main():
    parser = ArgumentParser(
        description="Extract routing-based LoRA from DeepSeek MoE models"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Folder containing model safetensors and index.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base path for outputs (residual folder + GGUF files)",
    )
    parser.add_argument(
        "--skip-up-gate",
        action="store_true",
        help="Skip extracting LoRA for gate_proj and up_proj",
    )
    parser.add_argument(
        "--down-rank",
        type=int,
        default=1,
        help="Rank for down_proj LoRA (0 to skip, default: 1)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="F32",
        choices=["F32", "F16", "BF16", "Q8_0"],
        help="Quantization type for GGUF (default: F32)",
    )
    parser.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Path to imatrix file (GGUF or .dat). Uses sampled activation importances for down_proj instead of heuristic.",
    )

    args = parser.parse_args()

    # When using imatrix, gate/up LoRAs are skipped automatically
    if args.imatrix:
        args.skip_up_gate = True

    # Validate that we're not doing a null operation
    if args.skip_up_gate and args.down_rank == 0:
        parser.error(
            "Both --skip-up-gate and --down-rank 0 would result in a null operation."
        )

    if args.down_rank < 0:
        parser.error("--down-rank must be >= 0")

    # Load weight map from index
    index_path = os.path.join(args.input, "model.safetensors.index.json")
    print(f"Loading index from: {index_path}")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    print(f"Found {len(weight_map)} tensors in index")

    # Map string to GGUF quantization type
    quant_type_map = {
        "F32": gguf.GGMLQuantizationType.F32,
        "F16": gguf.GGMLQuantizationType.F16,
        "BF16": gguf.GGMLQuantizationType.BF16,
        "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    }
    quant_type = quant_type_map[args.quant_type]

    # Setup residual output folder
    residual_folder = f"{args.output}_residual"
    os.makedirs(residual_folder, exist_ok=True)

    # Load imatrix if provided
    imatrix_data = None
    if args.imatrix:
        print(f"\nLoading imatrix from: {args.imatrix}")
        from imatrix_reader import read_imatrix

        imatrix_data = read_imatrix(args.imatrix)
        print(f"Loaded imatrix with {len(imatrix_data.column_tensors)} column tensors")

    # Pass 1: Generate LoRA tensors
    lora_map = pass1_generate_loras(
        args.input, weight_map, args.skip_up_gate, args.down_rank, imatrix_data
    )

    # Pass 2: Apply subtractions
    pass2_apply_subtractions(
        args.input,
        residual_folder,
        weight_map,
        lora_map,
        args.skip_up_gate,
        args.down_rank,
    )

    # Export up/gate LoRA GGUF
    if not args.skip_up_gate:
        up_gate_tensors = prepare_gguf_tensors(lora_map, ["gate_proj", "up_proj"])
        if up_gate_tensors:
            up_gate_path = f"{args.output}_up_gate_lora.gguf"
            # Alpha = rank = 1 gives scale factor 1.0 in standard LoRA (alpha/rank)
            export_lora_gguf(
                up_gate_path, up_gate_tensors, alpha=1, quant_type=quant_type
            )

    # Export down LoRA GGUF
    if args.down_rank > 0:
        down_tensors = prepare_gguf_tensors(lora_map, ["down_proj"])
        if down_tensors:
            down_path = f"{args.output}_down_lora.gguf"
            # Alpha = rank gives scale factor 1.0 in standard LoRA (alpha/rank)
            export_lora_gguf(
                down_path, down_tensors, alpha=args.down_rank, quant_type=quant_type
            )

    # Copy all non-safetensors files from input to output
    copy_non_safetensors(args.input, residual_folder)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Residual model: {residual_folder}")
    if args.imatrix:
        print(f"Mode:           imatrix (sampled importances)")
    elif not args.skip_up_gate:
        print(f"Up/Gate LoRA:   {args.output}_up_gate_lora.gguf")
    if args.down_rank > 0:
        print(f"Down LoRA:      {args.output}_down_lora.gguf")


if __name__ == "__main__":
    main()
