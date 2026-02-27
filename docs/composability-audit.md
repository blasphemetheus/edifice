# Composability Audit: Shared Block Extraction

Audit of duplicated code patterns across Edifice modules and their consolidation status.

## Extracted Shared Blocks

### 1. `Edifice.Blocks.SinusoidalPE2D`

2D sinusoidal positional encoding for flattened spatial grids. Encodes x/y positions independently using sin/cos frequency bands, then concatenates.

**Previous duplicates** (3 copies, ~30 lines each):
- `Edifice.Detection.DETR` — `build_2d_sinusoidal_pe/2`
- `Edifice.Detection.RTDETR` — `build_2d_sinusoidal_pe/2`
- `Edifice.Detection.SAM2` — `build_2d_sinusoidal_pe/2`

**Consolidated to**: `Edifice.Blocks.SinusoidalPE2D.build_table/2`

### 2. `Edifice.Blocks.Upsample2x`

Nearest-neighbor 2x spatial upsample for channels-last tensors via reshape/broadcast.

**Previous duplicates** (2 copies, ~14 lines each):
- `Edifice.Detection.RTDETR` — `upsample_2x/2`
- `Edifice.Detection.SAM2` — `upsample_2x/2`

**Consolidated to**: `Edifice.Blocks.Upsample2x.layer/2`

### 3. `Edifice.Blocks.SDPA`

Scaled Dot-Product Attention: reshape to heads, compute QK^T/sqrt(d), optional mask, fused softmax, apply to values, reshape back.

**Previous duplicates** (6 copies, ~20 lines each):
- `Edifice.Detection.DETR` — `compute_attention/5`
- `Edifice.Detection.RTDETR` — `compute_attention/5`
- `Edifice.Detection.SAM2` — `compute_attention/5`
- `Edifice.Robotics.ACT` — `compute_attention/5` (manual softmax variant)
- `Edifice.Audio.Whisper` — `compute_mha/6` (with optional mask)
- `Edifice.Blocks.CrossAttention` — `cross_attention_impl/5`

**Consolidated to**: `Edifice.Blocks.SDPA.compute/5` and `Edifice.Blocks.SDPA.compute/6` (with mask)

**Note**: ACT previously used a manual numerically-stable softmax; now uses `FusedOps.fused_softmax` for consistency (functionally equivalent, same numerical stability via FP32 computation).

### 4. `Edifice.Blocks.BBoxHead`

3-layer bounding box regression MLP: dense(hidden) -> ReLU -> dense(hidden) -> ReLU -> dense(4) -> sigmoid. Outputs normalized (cx, cy, w, h) coordinates.

**Previous duplicates** (2 copies):
- `Edifice.Detection.RTDETR` — `bbox_mlp/3`
- `Edifice.Detection.DETR` — inline bbox MLP in `build/1`

**Consolidated to**: `Edifice.Blocks.BBoxHead.layer/3`

### 5. CausalMask Migration

Private `create_causal_mask/1` functions that duplicate `CausalMask.causal/1` + reshape.

**Previous duplicates** (2 copies, ~5 lines each):
- `Edifice.Audio.VALLE` — `create_causal_mask/1`
- `Edifice.Recurrent.XLSTM` — `create_causal_mask/1`

**Consolidated to**: `Edifice.Blocks.CausalMask.causal/1` (existing) + `Nx.reshape({1, 1, seq_len, seq_len})`

### 6. TopK Sparsify Dedup

Identical `top_k_sparsify/2` defnp in two interpretability modules.

**Previous duplicates** (2 copies, ~5 lines each):
- `Edifice.Interpretability.SparseAutoencoder` — `top_k_sparsify/2`
- `Edifice.Interpretability.Transcoder` — `top_k_sparsify/2`

**Consolidated to**: `Edifice.Interpretability.SparseAutoencoder.top_k_sparsify/2` (promoted to `defn`, called from Transcoder)

## Not Consolidated (Different Semantics)

- **CogVideoX/VAR/BLT upsample** — 5D/variable/1D patterns differ from spatial 2x
- **GRPO/DPO/VAR causal mask** — inverted convention (0=attend, 1=mask)
- **DiT/DiTv2/SiT SDPA** — being modified separately (AdaptiveNorm refactor)
- **VALLE softmax** — uses manual stable softmax inside an `Axon.nx` closure (not a separate function); distinct pattern from SDPA compute

## Summary

| Block | Lines Removed | Files Affected |
|-------|--------------|----------------|
| SinusoidalPE2D | ~90 | 3 |
| Upsample2x | ~28 | 2 |
| SDPA | ~120 | 6 |
| BBoxHead | ~18 | 2 |
| CausalMask | ~10 | 2 |
| TopK Sparsify | ~5 | 2 |
| **Total** | **~271** | **12** |
