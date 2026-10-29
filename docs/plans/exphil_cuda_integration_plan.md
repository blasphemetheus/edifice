# ExPhil + Edifice CUDA Integration Plan

## Overview

This document covers two objectives:
1. **Architecture Gap Analysis** — Edifice architectures ExPhil should adopt but hasn't yet
2. **Fused Kernel Status** — Complete status of all 48 CUDA kernels with EXLA dual-linkage

### Current State

**ExPhil** has 48 valid backbones in `@valid_backbones` (config.ex).
**Edifice** has 234 registered architectures across 26 families.

ExPhil's 60 FPS target = **<16ms inference** at batch=1, embed=256, seq_len=32, layers=2.

**All 48 fused CUDA kernels are complete** — NIF + EXLA custom call (dual-linkage) with
f32/bf16 support, forward + backward passes, and block scan variants for key architectures.

---

## Part 1: Edifice Architectures ExPhil Should Add

### Currently in ExPhil (48 backbones)

```
mlp, lstm, gru, mamba, mamba_nif, mamba_cumsum, mamba_hillis_steele,
mamba_ssd, gated_ssm, attention, sliding_window, lstm_hybrid, hybrid,
jamba, zamba, griffin, hawk, xlstm, xlstm_slstm, xlstm_mlstm, retnet,
rwkv, gla, hgrn, s5, s4, s4d, h3, performer, deltanet, fnet, perceiver,
ttt, hopfield, ntm, reservoir, snn, bayesian, decision_transformer,
liquid, kan, min_gru, min_lstm, tcn, mamba3, hyena, titans, gated_deltanet
```

### Tier 1: Should Add Now (Sequence Models — Directly Relevant)

These are architectures Edifice has that process sequences and are realistic ExPhil
backbone candidates. They either show competitive inference speed or have interesting
quality/speed tradeoffs.

| Architecture | Family | Why Add | Expected Latency | Notes |
|---|---|---|---|---|
| `native_recurrence` | recurrent | 3 GRU variants (elu_gru, real_gru, diag_linear) from NativeRecurrence paper. Simple, fast. | ~15-25ms (est.) | Similar to min_gru, worth benchmarking |
| `longhorn` | ssm | Drop-in Mamba replacement, no forget gate, from closed-form online recall | ~20-30ms (est.) | Uses same parallel scan as Mamba |
| `samba` | ssm | Hybrid Mamba+SWA+MLP, beats Transformers on short+long context | ~25-40ms (est.) | First hybrid to beat transformers across context lengths |
| `hymba` | ssm | Hymba hybrid architecture | ~25-40ms (est.) | SSM+attention hybrid |
| `gss` | ssm | Gated State Space model | ~15-25ms (est.) | Simpler SSM variant |
| `delta_product` | recurrent | Multi-step DeltaNet via Householder products | ~30-60ms (est.) | Extends existing DeltaNet |
| `huginn` | recurrent | Depth-recurrent transformer with adaptive iteration | ~30-50ms (est.) | Latent reasoning capability |
| `deep_res_lstm` | recurrent | Deep residual LSTM | ~200-400ms (est.) | Same LSTM bottleneck but worth having |
| `gla_v2` | attention | Gated Linear Attention v2 | ~20-35ms (est.) | Improved GLA |
| `hgrn_v2` | attention | HGRN v2 | ~20-35ms (est.) | Improved HGRN |
| `ttt_e2e` | recurrent | End-to-end TTT | ~150-300ms (est.) | May benefit from kernel fusion |
| `gsa` | attention | Gated Slot Attention — linear time, fixed slots | ~20-35ms (est.) | Linear complexity |
| `rla` | attention | Residual Linear Attention | ~20-35ms (est.) | Corrects linear attention errors |
| `nha` | attention | Native Hybrid Attention | ~20-35ms (est.) | Per-layer linear vs full selection |
| `fox` | attention | Forgetting Transformer — learnable forget on softmax | ~18-30ms (est.) | Bounded memory in transformers |
| `log_linear` | attention | O(log T) space attention | ~18-30ms (est.) | Memory-quality tradeoff |
| `laser` | attention | exp(V) attention for larger gradients | ~18-30ms (est.) | Low complexity |
| `moba` | attention | Mixture of Block Attention | ~18-30ms (est.) | Production-proven (Kimi) |
| `tnn` | attention | Toeplitz Neural Network | ~15-25ms (est.) | O(n log n), good extrapolation |
| `coconut` | meta | Continuous chain of thought (latent reasoning) | ~20-35ms (est.) | BFS reasoning without text gen |
| `miras` | recurrent | Memory variants (Moneta, Yaad, Memora) | ~30-60ms (est.) | Multiple memory mechanisms |
| `mixture_of_mamba` | ssm | Modality-aware Mamba sparsity | ~25-40ms (est.) | Per-modality SSM routing |

**Total: 22 architectures to add to ExPhil's `@valid_backbones`.**

### Tier 2: Consider Later (Interesting but Lower Priority)

| Architecture | Family | Why Wait |
|---|---|---|
| `decoder_only` | transformer | Standard transformer, unlikely to hit 16ms at seq=32 |
| `conformer` | attention | Audio-oriented, may not suit Melee state |
| `mega` | attention | Older architecture, superseded by newer variants |
| `based` | attention | Worth testing but less proven |
| `bimamba` | ssm | Bidirectional — not ideal for causal inference |
| `striped_hyena` | ssm | Complex hybrid, may be slow |
| `ss_transformer` | ssm | SSM-Transformer hybrid |
| `hyena_v2` | attention | Hyena improvement, test if faster |
| `xlstm_v2` | recurrent | If xlstm works, try v2 |
| `mixture_of_recursions` | meta | Depth routing — interesting for adaptive compute |
| `mixture_of_expert_depths` | meta | Depth routing variant |
| `medusa` | inference | Speculative decoding — useful post-training |

### Tier 3: Not Applicable to ExPhil

Vision, graph, audio, detection, generative, interpretability, scientific, robotics
families are not backbone candidates for the Melee sequence modeling task.

---

## Part 2: Fused CUDA Kernel Status

### Implementation Strategy (All Phases Complete)

**Phase A: NIF Bridge — DONE**
- CUDA kernels compiled as shared libraries (`libedifice_cuda_kernels.so`)
- Called via Erlang NIF from Elixir
- Device pointer extraction from EXLA tensors → kernel launch → wrap result back
- GC-tracked memory management

**Phase B: EXLA Custom Call (XLA FFI) — DONE**
- Forked `elixir-nx/nx`, added GPU custom calls via XLA FFI
- Kernels compiled directly into EXLA's NIF .so with dual-linkage (`-DEXLA_FFI`)
- Zero-copy: XLA manages all memory, kernels run on XLA's CUDA stream
- Generic dispatch in `defn.ex:749-755` handles all custom calls automatically

**Phase C: Stretch Goals — DONE**
- bf16 kernel variants via `precision.cuh` (`io_type`/`IO_LOAD`/`IO_STORE` macros)
- Backward pass kernels for all scan architectures
- Block scan variants for MinGRU, MinLSTM, LSTM, GRU, Linear

### Scalar Operand Workarounds

XLA's FFI mechanism segfaults on 0-dimensional (scalar) buffer operands. Two patterns used:

1. **Tensor packing** (preferred): Pack scalar into a 1-element `{1}` tensor of the same
   type as the model tensors. Unpack on CUDA side with `cudaMemcpyAsync` D→H.
   Used for: Titans momentum, MIRAS momentum, Flash/LASER causal flag.

2. **Combined tensor column** (for values intrinsic to the data): Pack as an extra column
   in an existing input tensor. Used for: Titans scan (momentum as extra column in combined tensor).

### Complete Kernel Inventory (48 production kernels)

#### Forward Scan Kernels (19)

| # | Kernel File | Architecture(s) | Status |
|---|---|---|---|
| 1 | `fused_mingru_scan.cu` | MinGRU | Done (NIF + EXLA f32/bf16) |
| 2 | `fused_minlstm_scan.cu` | MinLSTM | Done (NIF + EXLA f32/bf16) |
| 3 | `fused_native_rec_scan.cu` | NativeRecurrence (elu_gru, real_gru, diag_linear) | Done (NIF + EXLA f32/bf16) |
| 4 | `fused_liquid_scan.cu` | Liquid / LiquidS4 | Done (NIF + EXLA f32/bf16) |
| 5 | `fused_selective_scan.cu` | Mamba, Mamba variants | Done (NIF + EXLA f32/bf16) |
| 6 | `fused_delta_rule_scan.cu` | DeltaNet, GatedDeltaNet | Done (NIF + EXLA f32/bf16) |
| 7 | `fused_delta_product_scan.cu` | DeltaProduct | Done (NIF + EXLA f32/bf16) |
| 8 | `fused_linear_scan.cu` | Linear scan (diag_linear base) | Done (NIF + EXLA f32/bf16) |
| 9 | `fused_slstm_scan.cu` | sLSTM (xLSTM) | Done (NIF + EXLA f32/bf16) |
| 10 | `fused_ttt_scan.cu` | TTT | Done (NIF + EXLA f32/bf16) |
| 11 | `fused_kda_scan.cu` | KDA | Done (NIF + EXLA f32/bf16) |
| 12 | `fused_rla_scan.cu` | Residual Linear Attention | Done (NIF + EXLA f32/bf16) |
| 13 | `fused_gru_scan.cu` | GRU | Done (NIF + EXLA f32/bf16) |
| 14 | `fused_lstm_scan.cu` | LSTM | Done (NIF + EXLA f32/bf16) |
| 15 | `fused_titans_scan.cu` | Titans | Done (NIF + EXLA f32/bf16) |
| 16 | `fused_miras_scan.cu` | MIRAS (Moneta, Yaad, Memora) | Done (NIF + EXLA f32/bf16) |
| 17 | `fused_gsa_scan.cu` | Gated Slot Attention | Done (NIF + EXLA f32/bf16) |
| 18 | `fused_reservoir_scan.cu` | Reservoir | Done (NIF + EXLA f32/bf16) |
| 19 | `fused_fox_attention.cu` | Fox (Forgetting Transformer) | Done (NIF + EXLA f32/bf16) |

#### Block-Fused Scan Kernels (5)

| # | Kernel File | Architecture | Status |
|---|---|---|---|
| 20 | `fused_mingru_block_scan.cu` | MinGRU (block scan) | Done (NIF + EXLA f32/bf16) |
| 21 | `fused_minlstm_block_scan.cu` | MinLSTM (block scan) | Done (NIF + EXLA f32/bf16) |
| 22 | `fused_linear_block_scan.cu` | Linear (block scan) | Done (NIF + EXLA f32/bf16) |
| 23 | `fused_lstm_block_scan.cu` | LSTM (block scan) | Done (NIF + EXLA f32/bf16) |
| 24 | `fused_gru_block_scan.cu` | GRU (block scan) | Done (NIF + EXLA f32/bf16) |

#### Backward Scan Kernels (16)

| # | Kernel File | Architecture | Status |
|---|---|---|---|
| 25 | `fused_mingru_scan_backward.cu` | MinGRU backward | Done (NIF + EXLA f32/bf16) |
| 26 | `fused_minlstm_scan_backward.cu` | MinLSTM backward | Done (NIF + EXLA f32/bf16) |
| 27 | `fused_liquid_scan_backward.cu` | Liquid backward | Done (NIF + EXLA f32/bf16) |
| 28 | `fused_linear_scan_backward.cu` | Linear scan backward | Done (NIF + EXLA f32/bf16) |
| 29 | `fused_delta_rule_scan_backward.cu` | DeltaNet backward | Done (NIF + EXLA f32/bf16) |
| 30 | `fused_selective_scan_backward.cu` | Mamba backward | Done (NIF + EXLA f32/bf16) |
| 31 | `fused_delta_product_scan_backward.cu` | DeltaProduct backward | Done (NIF + EXLA f32/bf16) |
| 32 | `fused_slstm_scan_backward.cu` | sLSTM backward | Done (NIF + EXLA f32/bf16) |
| 33 | `fused_ttt_scan_backward.cu` | TTT backward | Done (NIF + EXLA f32/bf16) |
| 34 | `fused_kda_scan_backward.cu` | KDA backward | Done (NIF + EXLA f32/bf16) |
| 35 | `fused_rla_scan_backward.cu` | RLA backward | Done (NIF + EXLA f32/bf16) |
| 36 | `fused_elu_gru_scan_backward.cu` | NativeRec elu_gru backward | Done (NIF + EXLA f32/bf16) |
| 37 | `fused_real_gru_scan_backward.cu` | NativeRec real_gru backward | Done (NIF + EXLA f32/bf16) |
| 38 | `fused_diag_linear_scan_backward.cu` | NativeRec diag_linear backward | Done (NIF + EXLA f32/bf16) |
| 39 | `fused_gru_scan_backward.cu` | GRU backward | Done (NIF + EXLA f32/bf16) |
| 40 | `fused_lstm_scan_backward.cu` | LSTM backward | Done (NIF + EXLA f32/bf16) |

#### Attention Kernels (6)

| # | Kernel File | Architecture | Status |
|---|---|---|---|
| 41 | `fused_flash_attention.cu` | Flash Attention V2 forward | Done (NIF + EXLA f32/bf16) |
| 42 | `fused_flash_attention_backward.cu` | Flash Attention V2 backward | Done (NIF + EXLA f32/bf16) |
| 43 | `fused_laser_attention.cu` | LASER Attention forward | Done (NIF + EXLA f32/bf16) |
| 44 | `fused_laser_attention_backward.cu` | LASER Attention backward | Done (NIF + EXLA f32/bf16) |
| 45 | `fused_fox_attention.cu` | Fox Attention forward | Done (NIF + EXLA f32/bf16) |
| 46 | `fused_fox_attention_backward.cu` | Fox Attention backward | Done (NIF + EXLA f32/bf16) |

#### Utility Files (2)

| # | File | Purpose |
|---|---|---|
| 47 | `test_kernels.cu` | Standalone GPU correctness tests |
| 48 | `bench_kernels.cu` | Standalone GPU performance benchmarks |

---

## Part 3: Implementation Phases (Historical)

All phases are complete. This section preserved for historical reference.

### Phase A: NIF Bridge — Complete

Extended the NIF bridge pattern to all architectures.

```
edifice/
├── native/cuda/
│   ├── fused_mingru_scan.cu              ✓ Done
│   ├── fused_minlstm_scan.cu            ✓ Done
│   ├── fused_native_rec_scan.cu          ✓ Done (3 variants)
│   ├── fused_liquid_scan.cu              ✓ Done
│   ├── fused_selective_scan.cu           ✓ Done (Mamba)
│   ├── fused_delta_rule_scan.cu          ✓ Done
│   ├── fused_delta_product_scan.cu       ✓ Done
│   ├── ... (all 48 kernels)             ✓ Done
│   ├── test_kernels.cu                   ✓ Done
│   └── bench_kernels.cu                  ✓ Done
├── c_src/
│   └── edifice_cuda_nif.c               ✓ All NIF functions added
└── lib/edifice/cuda/
    ├── nif.ex                            ✓ All NIF bindings
    └── fused_scan.ex                     ✓ All dispatch + 3-tier fallback
```

### Phase B: EXLA Fork — Complete

All NIF kernels ported to EXLA with dual-linkage compilation.

- Each `.cu` file compiles in two modes: standalone (NIF) and `-DEXLA_FFI` (EXLA custom call)
- `precision.cuh` provides `io_type`/`IO_LOAD`/`IO_STORE` macros for f32/bf16
- Generic dispatch in EXLA `defn.ex` passes all `Nx.Shared.optional` args through to
  `Value.custom_call_fused` automatically — no per-kernel EXLA changes needed
- Handler registration: `XLA_FFI_REGISTER_HANDLER` with `"CUDA"` platform + `_f32`/`_bf16` suffix

### Phase C: Stretch Goals — Complete

1. **bf16 kernel variants** — Done via `precision.cuh` dual compilation
2. **Backward pass kernels** — Done for all scan architectures (16 backward kernels)
3. **Block scan variants** — Done for MinGRU, MinLSTM, Linear, LSTM, GRU (5 block kernels)
4. **Triton-style autotune** — Not pursued (manual tile sizing sufficient)
5. **PR to upstream EXLA** — Not pursued (fork approach working well)

---

## Part 4: Current Next Steps

From TODO.md — the CUDA kernel work is complete. Focus has shifted to:

### Phase 1 — Trust & Usability (Priority: High)

**Numerical Correctness Suite** — Expand PyTorch reference validation beyond ViT/Whisper
to 10 key architectures (LSTM, Mamba, GQA, MinGRU, DeltaNet, DETR, DiT, ResNet, GAT, ConvNeXt).

**Applied Task Benchmarks** (`bench/tasks/` suite):
- Sequence classification (length-generalization)
- Image classification (MNIST/FashionMNIST)
- Graph classification (community detection)
- Autoregressive generation (char-level Shakespeare)
- Copy/recall (synthetic)

### Phase 2 — Production Path (Priority: Medium)

**Inference Serving Layer** (`Edifice.Serving`):
- Batched inference server, autoregressive generation loop, speculative decoding, streaming

**Training Recipes** (`Edifice.Recipes`):
- Classification, sequence modeling, contrastive, fine-tuning recipes

### Phase 3 — Discovery & Polish (Priority: Low-Medium)

- Interactive Model Explorer (Livebook Smart Cell or Phoenix LiveView)
- Architecture Recommender (`Edifice.AutoML.recommend/2`)

---

## Appendix: Architecture → Kernel Mapping

Quick reference for every ExPhil-relevant architecture and its kernel status.

| Architecture | Scan Type | Kernel File | Status |
|---|---|---|---|
| min_gru | Sequential, register | `fused_mingru_scan.cu` | Done (NIF + EXLA f32/bf16 + backward + block) |
| min_lstm | Sequential, register | `fused_minlstm_scan.cu` | Done (NIF + EXLA f32/bf16 + backward + block) |
| native_recurrence | Sequential, register | `fused_native_rec_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| liquid | Sequential, register | `fused_liquid_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| mamba | Parallel + selective | `fused_selective_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| delta_net | Sequential, matrix state | `fused_delta_rule_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| gated_delta_net | Sequential, matrix state | `fused_delta_rule_scan.cu` | Done (shared kernel with delta_net) |
| delta_product | Sequential, Householder | `fused_delta_product_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| lstm | Sequential, matmul | `fused_lstm_scan.cu` | Done (NIF + EXLA f32/bf16 + backward + block) |
| gru | Sequential, matmul | `fused_gru_scan.cu` | Done (NIF + EXLA f32/bf16 + backward + block) |
| slstm | Sequential, log-domain | `fused_slstm_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| ttt | Sequential, matrix W | `fused_ttt_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| titans | Recurrent + attention | `fused_titans_scan.cu` | Done (NIF + EXLA f32/bf16, momentum packed) |
| miras | Memory variants | `fused_miras_scan.cu` | Done (NIF + EXLA f32/bf16, momentum packed) |
| rla | Linear attention | `fused_rla_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| gsa | Gated slot attention | `fused_gsa_scan.cu` | Done (NIF + EXLA f32/bf16) |
| kda | KDA | `fused_kda_scan.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| reservoir | Fixed weights | `fused_reservoir_scan.cu` | Done (NIF + EXLA f32/bf16) |
| attention / gqa | Matmul | `fused_flash_attention.cu` | Done (NIF + EXLA f32/bf16 + backward, causal packed) |
| laser | exp(V) attention | `fused_laser_attention.cu` | Done (NIF + EXLA f32/bf16 + backward, causal packed) |
| fox | Forgetting attention | `fused_fox_attention.cu` | Done (NIF + EXLA f32/bf16 + backward) |
| linear scan | Diagonal linear | `fused_linear_scan.cu` | Done (NIF + EXLA f32/bf16 + backward + block) |
| gated_ssm | Element-wise | Not needed | Already fast (12.96ms) |
| fnet | FFT | Not needed | Already fast (14.56ms, cuFFT) |
| s4 / s4d / s5 / h3 | FFT conv | Not needed | cuFFT handles it |
| hyena | FFT conv | Not needed | cuFFT handles it |
| mlp / kan | None | Not needed | No temporal processing |
| snn | Spike-based | Not needed | Different optimization domain |
| huginn | Iterative depth | Not needed | Not sequential scan, use EXLA |
| coconut | Iterative | Not needed | Not sequential scan, use EXLA |
