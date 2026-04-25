# Extract MoE Outlier LoRA

Extract outlier-compensation LoRA adapters from Mixture-of-Experts (MoE) language models.

This tool identifies outlier columns in MoE MLP down-projection weights and extracts them as rank-N LoRA adapters. Outliers can be selected either by a fast heuristic (`b_gate * b_up`) or by real sampled activation importances from an imatrix. The residual model has these outliers subtracted out, making the remaining weights easier to quantize while the LoRA adapter restores them at inference time.

## How It Works

The script performs a two-pass extraction:

1. **Pass 1** — For each MoE layer and expert, identify outlier columns and compute LoRA tensors:
   - **Gate / Up projections** (heuristic mode only): Rank-1 LoRA using the normalized routing vector
   - **Down projection**: Rank-N LoRA using the top-N outlier columns (selected by upstream activity heuristic or sampled imatrix importances)

2. **Pass 2** — Subtract the LoRA reconstructions from the original expert weights and save the residual model.

The extracted LoRAs are exported in GGUF format with `alpha = rank`, ensuring a scale factor of **1.0** (`alpha / rank = 1.0`) so the LoRA exactly reconstructs what was removed.

## Installation

```bash
pip install torch numpy safetensors gguf
```

## Usage

```bash
python extract_moe_outlier_lora.py \
    --input models/Your-MoE-Model \
    --output models/my_model \
    --quant-type F32
```

### Options

| Option | Description |
|--------|-------------|
| `--input` | Path to the input model folder containing safetensors and `model.safetensors.index.json` |
| `--output` | Base path for outputs. Creates `{output}_residual/` for the model and `{output}_up_gate_lora.gguf` / `{output}_down_lora.gguf` for the LoRAs |
| `--skip-up-gate` | Skip extracting the gate/up LoRA. Gate and up weights are left unmodified in the residual model |
| `--down-rank N` | Set the rank for the down projection LoRA. Use `0` to skip it. Default: `1` |
| `--quant-type` | GGUF quantization type: `F32`, `F16`, `BF16`, or `Q8_0`. Default: `F32` |
| `--imatrix PATH` | Path to an imatrix file (GGUF or `.dat`). Uses sampled activation importances for `down_proj` instead of the `b_gate * b_up` heuristic. Implies `--skip-up-gate` |

**Note**: `--skip-up-gate` combined with `--down-rank 0` is rejected as a null operation. Same for `--imatrix` with `--down-rank 0`.

## Output Files

Given `--output models/my_model`, the following are created:

| File / Folder | Contents |
|---------------|----------|
| `models/my_model_residual/` | Residual model (original weights minus LoRA reconstructions) with all non-safetensors files copied from the input |
| `models/my_model_up_gate_lora.gguf` | Rank-1 LoRA for `gate_proj` and `up_proj` (unless `--skip-up-gate` or `--imatrix`) |
| `models/my_model_down_lora.gguf` | Rank-N LoRA for `down_proj` (unless `--down-rank 0`) |

## Examples

**Extract both LoRAs with default rank-1 down projection:**
```bash
python extract_moe_outlier_lora.py --input ./Your-MoE-Model --output ./my_model
```

**Skip gate/up LoRA, extract rank-4 down LoRA:**
```bash
python extract_moe_outlier_lora.py \
    --input ./Your-MoE-Model \
    --output ./my_model \
    --skip-up-gate \
    --down-rank 4
```

**Use FP16 quantization:**
```bash
python extract_moe_outlier_lora.py \
    --input ./Your-MoE-Model \
    --output ./my_model \
    --quant-type F16
```

**Use sampled activation importances (imatrix):**
```bash
python extract_moe_outlier_lora.py \
    --input ./Your-MoE-Model \
    --output ./my_model \
    --down-rank 4 \
    --imatrix ./Your-MoE-Model-BF16-imatrix.gguf
```

### Heuristic vs. Imatrix Mode

The script supports two ways of selecting features for the `down_proj` LoRA:

- **Heuristic mode** (default): Computes `b_gate * b_up` — the elementwise product of alignment scores from the gate and up projections. This is fast and requires no extra data, but is an approximation.
- **Imatrix mode** (`--imatrix`): Uses real sampled activation importances computed by llama.cpp's `imatrix` tool. The imatrix records which input features to `down_proj` are most active across real data, giving a more accurate selection. When `--imatrix` is provided, gate/up LoRA extraction is skipped automatically.

## End-to-End Examples (llama.cpp)

### Heuristic Mode

```bash
# 1. Extract rank-1 up/gate LoRA and rank-2 down LoRA
python3 ./extract_moe_outlier_lora.py \
    --input ~/models/Your-MoE-Model \
    --output Your-MoE-Model \
    --down-rank 2

# 2. Convert residual model to GGUF
~/llama.cpp/convert_hf_to_gguf.py \
    Your-MoE-Model_residual \
    --outtype auto \
    --outfile Your-MoE-Model_residual-BF16.gguf

# 3. Quantize expert tensors to Q4_0
~/llama.cpp/build/bin/llama-quantize \
    --tensor-type "_exps=q4_0" \
    Your-MoE-Model_residual-BF16.gguf \
    Your-MoE-Model_residual-Q4_0_X.gguf \
    Q8_0 16

# 4. Run inference with both LoRAs applied
~/llama.cpp/build/bin/llama-perplexity \
    -f wiki.test.raw \
    -m Your-MoE-Model_residual-Q4_0_X.gguf \
    --lora Your-MoE-Model_up_gate_lora.gguf \
    --lora Your-MoE-Model_down_lora.gguf
```

### Imatrix Mode

```bash
# 1. Generate imatrix from the base GGUF model (run once)
~/llama.cpp/build/bin/llama-imatrix \
    -m Your-MoE-Model-BF16.gguf \
    -f calibration.txt \
    -o Your-MoE-Model-BF16-imatrix.gguf \
    --output-frequency 10

# 2. Extract rank-4 down LoRA using sampled importances
python3 ./extract_moe_outlier_lora.py \
    --input ~/models/Your-MoE-Model \
    --output Your-MoE-Model-imatrix \
    --down-rank 4 \
    --imatrix Your-MoE-Model-BF16-imatrix.gguf

# 3. Convert residual model to GGUF
~/llama.cpp/convert_hf_to_gguf.py \
    Your-MoE-Model-imatrix_residual \
    --outtype auto \
    --outfile Your-MoE-Model-imatrix_residual-BF16.gguf

# 4. Quantize expert tensors
~/llama.cpp/build/bin/llama-quantize \
    --tensor-type "_exps=q4_0" \
    Your-MoE-Model-imatrix_residual-BF16.gguf \
    Your-MoE-Model-imatrix_residual-Q4_0_X.gguf \
    Q8_0 16

# 5. Run inference with only the down LoRA applied
~/llama.cpp/build/bin/llama-perplexity \
    -f wiki.test.raw \
    -m Your-MoE-Model-imatrix_residual-Q4_0_X.gguf \
    --lora Your-MoE-Model-imatrix_down_lora.gguf
```

## Math Notes

The reconstruction subtracted from each weight is:

- **Rank-1**: `W_residual = W_original - outer(b, a)`
- **Rank-N**: `W_residual = W_original - B @ A`

With `alpha = rank`, standard LoRA inference computes `W_base + (alpha / rank) * B @ A`, which simplifies to `W_base + B @ A` — exactly restoring the subtracted component.

## License

Apache 2.0 License - See [LICENSE](LICENSE) for details
