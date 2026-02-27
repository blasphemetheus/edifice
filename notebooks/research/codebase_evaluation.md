# Edifice Codebase Evaluation — 2026-02-27

Comprehensive evaluation of composability, maintainability, test coverage, and
open-source quality metrics. Compared against PyTorch, JAX/Flax, Keras, tinygrad,
MLX, and Elixir ecosystem standards (Nx, Axon, Phoenix, Ecto).

## Codebase Snapshot

| Metric | Value |
|--------|-------|
| Source files | 223 |
| Source LOC | ~78,700 |
| Test files | 200 |
| Test LOC | ~45,700 |
| Total tests | 2,582 |
| Registered architectures | 196 |
| Architecture families | 26 |
| Shared blocks | 20 |
| Test-to-source ratio | 0.58:1 |
| Tests per module (average) | 11.6 |

---

## 1. Composability

### 1.1 Shared Block Adoption

20 shared blocks exist in `lib/edifice/blocks/`. Adoption across 223 modules:

| Block | Modules Using It | Adoption % |
|-------|-----------------|------------|
| FFN.layer | 28 | 12.6% |
| SinusoidalPE | 11 | 4.9% |
| RoPE | 8 | 3.6% |
| TransformerBlock | 8 | 3.6% |
| CausalMask | 5 | 2.2% |
| CrossAttention | 3 | 1.3% |
| SDPA | ~6 | 2.7% |
| RMSNorm | ~4 | 1.8% |
| SwiGLU | ~3 | 1.3% |
| ModelBuilder | ~5 | 2.2% |

**~88% of modules implement patterns that exist in shared blocks.** This is partly
by design (unique architecture needs) and partly because most architectures were
implemented before blocks were extracted. The composability audit (commit `3a5bb44`
et al.) already migrated several modules, but many remain.

### 1.2 Encoder-Decoder Gap (Addressed)

TransformerBlock was extended with `layer/3` and `stack/4` for 3-sublayer decoder
blocks (self-attn + cross-attn + FFN). DETR, RT-DETR, ACT, and Whisper now use
it. See `docs/composability-audit.md` for details.

### 1.3 Duplicate Patterns Still Present

Despite the extraction audit, many modules still inline their own:

- **Norm + residual wiring** — ~120 modules (54%) implement their own LayerNorm
  placement. This is lightweight (3-5 lines) but inconsistent.
- **Dense-activation-dense FFN** — Modules that don't use `FFN.layer` repeat
  the same 3-line pattern. Not harmful per se, but means FFN bug fixes don't
  propagate.
- **Multi-head attention reshape** — Several modules outside the attention family
  implement Q/K/V projection + head reshape + scaled dot product inline instead
  of calling SDPA or CrossAttention.

### 1.4 API Consistency

**Excellent.** All 223 modules follow the `build/1` pattern:

```elixir
@spec build([build_opt()]) :: Axon.t()  # or {Axon.t(), Axon.t()} for encoder-decoder
def build(opts \\ []) do
  # Options via Keyword.get with defaults
  # Return Axon graph
end
```

Registry access is consistent: `Edifice.build(:name, opts)`, `list_architectures/0`,
`list_families/0`. Typed option unions (`@type build_opt`) with zero `term()`
fallbacks across all modules.

### 1.5 Cross-Module Coupling

- **Average internal aliases per module:** 0.69
- **Modules with 0 internal deps:** ~60% (fully self-contained)
- **Modules with 1-2 deps:** ~30% (loosely coupled)
- **Modules with 3+ deps:** ~10% (SSM and meta families)
- **No circular dependencies detected**

### 1.6 Vision Composability Gap

Vision modules (ViT, Swin, DeiT, ConvNeXt, etc.) lack a standardized backbone
interface. Each has different output conventions, making it hard to swap vision
encoders. Attention-based architectures are much more composable due to
TransformerBlock's callback pattern.

---

## 2. Maintainability

### 2.1 Static Analysis

| Tool | Result |
|------|--------|
| `mix format --check-formatted` | Pass (CI-enforced) |
| Credo strict | 0 issues across 259 files |
| Dialyzer | 0 errors |
| Doctor | 100% moduledoc, doc, spec coverage |
| Compile warnings-as-errors | Yes (CI) |
| Unused deps check | Yes (CI) |
| Dependency audit | Yes (CI) |

This exceeds most Hex packages. The CI pipeline runs 10 quality checks, which is
more comprehensive than many top-tier Elixir libraries.

### 2.2 Documentation

| Metric | Value |
|--------|-------|
| Modules with `@moduledoc` | 223/223 (100%) |
| Modules with ASCII arch diagrams | ~180+ |
| Public functions with `@doc` | 1,174 |
| Public functions with `@spec` | 1,188 |
| Type definitions (`@type`) | 208 |
| Doctests | **0** (gap) |
| Guides | 18 (6,366 lines) |
| Examples (.exs) | 5 |
| README | 27KB |
| CHANGELOG | 12KB |

**Doctests are the single biggest documentation gap.** Phoenix, Ecto, and Nx all
use doctests extensively. They serve triple duty: documentation (rendered on
HexDocs), regression tests (run by `mix test`), and usage examples. For Edifice,
adding doctests to shared blocks and the registry module would be the highest-value
documentation improvement.

### 2.3 Module Size Distribution

```
Small   (<100 lines):     10 modules (4.5%)
Medium  (100-300 lines):  92 modules (41.3%)
Large   (>300 lines):    121 modules (54.3%)
```

**Modules exceeding 800 lines:**

| Module | Lines | Functions | Issue |
|--------|-------|-----------|-------|
| multi_head.ex | 1,152 | 22 | 5 distinct attention algorithms in one file |
| trellis.ex | 1,011 | 22 | Multi-stage pipeline (encoder + decoder + diffusion) |
| cogvideox.ex | 944 | 33 | Tightly coupled video generation |
| gated_ssm.ex | 930 | 19 | Multiple scan implementations |
| openvla.ex | 851 | 27 | Robotics model with inline encoding/decoding |
| gaussian_splat.ex | 851 | 18 | 3D rendering pipeline |

`multi_head.ex` is the clearest decomposition candidate. Its 5 algorithms
(standard, chunked, sliding window, memory-efficient, online softmax) are largely
independent and could become separate modules while sharing a common interface.

### 2.4 Dead Code & Deprecated Patterns

None detected. No unused aliases, no commented-out code blocks, no deprecated
function calls. Codebase is clean.

---

## 3. Test Coverage

### 3.1 File-Level Coverage

- **116 of 223 source files (52%)** have a dedicated test file
- **107 source files (48%)** have no dedicated test — tested only indirectly
  through registry sweep or parent module tests

### 3.2 Per-Family Coverage

| Family | Files With Tests | Total Files | Coverage |
|--------|-----------------|-------------|----------|
| Audio | 4/4 | 4 | **100%** |
| Convolutional | 6/6 | 6 | **100%** |
| Feedforward | 5/5 | 5 | **100%** |
| Transformer | 5/5 | 5 | **100%** |
| Energy | 3/3 | 3 | **100%** |
| Neuromorphic | 2/2 | 2 | **100%** |
| Attention | 5/34 | 34 | 15% |
| Recurrent | 4/17 | 17 | 24% |
| SSM | 2/21 | 21 | 10% |
| Meta | 1/23 | 23 | 4% |
| Graph | 0/10 | 10 | **0%** |
| Contrastive | 0/8 | 8 | **0%** |
| Vision | 0/15 | 15 | **0%** |
| Blocks | 1/20 | 20 | **5%** |
| Generative | ~1/21 | 21 | 5% |
| Detection | 0/3 | 3 | **0%** |
| Robotics | 0/2 | 2 | **0%** |
| Multimodal | 0/1 | 1 | **0%** |

**Critical gap: shared blocks at 5% dedicated coverage.** The 20 shared blocks
(TransformerBlock, FFN, CrossAttention, CausalMask, RoPE, SinusoidalPE,
ModelBuilder, RMSNorm, SwiGLU, SDPA, etc.) are the foundation for composability
improvements. They're tested indirectly through architecture tests, but this makes
refactoring risky because changes can break consumers in non-obvious ways.

### 3.3 Test Depth Patterns

Typical test file structure:

```elixir
describe "Module.build/1" do
  test "produces correct struct type"        # Always present
  test "forward pass produces correct shape" # Almost always
  test "output values are finite"            # ~27% of test files
  test "minimal configuration"               # ~40% of test files
  test "Edifice.build/2 integration"         # ~14% of test files
end
```

What's tested well:
- Build → init → predict → shape assertion (consistent pattern)
- The registry sweep test exercises all 196 registered architectures

What's under-tested:
- **Batch=1 edge cases** — only 27% of test files check batch_size=1
- **Option validation / error handling** — almost no tests for invalid opts
- **Gradient flow** — no gradient checks in any test file
- **output_size/1** — tested in some modules, not all
- **Loss functions** — tested in isolation (smoke tests), not per-module

### 3.4 Test Infrastructure

- `TestHelpers` module centralizes common assertions
- Registry sweep test (`registry_sweep_test.exs`) covers all architectures
- Input robustness test (`input_robustness_test.exs`) catches numerical instability
- Property-based testing via StreamData (limited usage)
- **No shared fixtures** for common input patterns (each test creates its own)
- **No test coverage tracking** (ExCoveralls available but not configured)

### 3.5 Comparison to Industry

| Metric | PyTorch | Keras | Nx/Axon | Edifice |
|--------|---------|-------|---------|---------|
| Test-to-source ratio | ~0.8:1 | ~0.5:1 | ~0.6:1 | 0.58:1 |
| Coverage tracking | Coveralls | codecov | None | None |
| Gradient checks | Yes | Yes | No | No |
| Multi-backend tests | CUDA/CPU/MPS | TF/JAX | BinaryBackend/EXLA | BinaryBackend only |
| CI matrix (versions) | Python 3.8-3.12 | Python 3.9+ | Elixir/OTP matrix | **Single version** |
| Benchmark regression | TorchBench | No | No | No |

---

## 4. Open Source Quality Metrics

Evaluated against CHAOSS (Community Health Analytics for Open Source Software),
GitHub OSPO standards, and Elixir ecosystem conventions.

### 4.1 Code Quality Tooling — 5/5

Exceeds most Hex packages:
- 10 CI quality checks (format, compile-warnings-as-errors, unused deps, Credo,
  deps.audit, Dialyzer, hex.build, docs gen, Doctor, tests)
- Zero issues across all linters
- 100% type coverage per Doctor

### 4.2 Documentation — 4/5

Strong but missing doctests:
- 100% moduledoc/spec/doc coverage
- 18 guides + 5 examples + comprehensive README/CHANGELOG
- Zero doctests (significant gap for Elixir ecosystem)
- No CODE_OF_CONDUCT.md (GitHub OSPO standard)

### 4.3 API Design — 5/5

- Unified `build/1` pattern across all 223 modules
- Registry pattern with `build/2`, `list_architectures/0`, `list_families/0`
- Typed option unions with zero `term()` fallbacks
- Progressive disclosure (sensible defaults, advanced opts)

### 4.4 CI/CD — 4/5

- 10 quality checks in CI
- Missing: multi-version matrix, coverage reporting, benchmark regression
- Release practices: SemVer, CHANGELOG, Hex published

### 4.5 ML-Specific Quality — 2/5

| Dimension | Industry Standard | Edifice |
|-----------|------------------|---------|
| Pretrained weights | Model zoo (HuggingFace/timm) | None |
| Interoperability | ONNX, SafeTensors | None (axon_onnx exists separately) |
| Benchmark infra | TorchBench, MLPerf | Benchee exists, not in CI |
| Gradient testing | Standard in PyTorch/JAX | None |
| Architecture viz | TensorBoard, Netron | None |
| Hardware backends | CUDA/ROCm/Metal/TPU | Via EXLA (inherited, correct) |
| Distributed training | Native support | Via Nx (inherited, correct) |

### 4.6 Community Health — 2/5 (expected for age)

| Metric | Value |
|--------|-------|
| Contributors | 1 |
| GitHub stars | 15 |
| Hex all-time downloads | 38 |
| Hex weekly downloads | 17 |
| Open issues | 0 |
| Open PRs | 0 |
| ElixirForum thread | Yes (1 post) |
| Blog posts | 0 |
| Conference talks | 0 |

Bus factor of 1 is the primary risk flag per CHAOSS metrics. This is normal for
a young solo project but is the single biggest concern for potential adopters.

### 4.7 Ecosystem Compliance

**Elixir Library Guidelines Checklist:**

| Requirement | Status |
|-------------|--------|
| `mix new` scaffold | Yes |
| `snake_case` naming | Yes |
| Semantic versioning | Yes |
| License (MIT/Apache) | Yes (MIT) |
| Formatter enforced | Yes |
| ExUnit tests | Yes (2,582) |
| Doctests | **No** |
| Complete API docs | Yes (100% Doctor) |
| Dev/test dep scoping | Yes |
| Optional dep declaration | Yes (`:exla`) |
| `mix.lock` in VCS | Yes |

**Hex Package Quality Signals:**

| Signal | Status |
|--------|--------|
| Published on Hex | Yes (v0.1.1, v0.2.0) |
| HexDocs generated | Yes |
| README with badges | Yes |
| CHANGELOG | Yes |
| LICENSE | Yes |
| CONTRIBUTING.md | Yes |
| CODE_OF_CONDUCT.md | **No** |
| CI badge | Yes |

---

## 5. Comparison to Other ML Libraries

### 5.1 Edifice vs PyTorch (torch.nn)

| Dimension | PyTorch | Edifice | Notes |
|-----------|---------|---------|-------|
| Architecture count | ~200 in nn | 196 | Comparable breadth |
| API consistency | Mixed (Module + functional) | Unified build/1 | Edifice is more consistent |
| Documentation | Extensive + tutorials | Excellent docs, fewer tutorials | Need more notebooks/tutorials |
| Testing | Comprehensive + gradient | Shape + finiteness only | Missing gradient checks |
| Pretrained weights | Thousands (Hub) | None | Not applicable yet |
| Community | Massive | Solo | Expected for age/ecosystem |
| Interoperability | ONNX, TorchScript, etc. | None | Biggest functional gap |
| Performance tooling | Profiler, TorchBench | Benchee (not in CI) | Need benchmark infra |

### 5.2 Edifice vs tinygrad

| Dimension | tinygrad | Edifice | Notes |
|-----------|----------|---------|-------|
| Philosophy | Minimal core (~14.5K LOC) | Comprehensive library (~78.7K) | Different goals |
| Architecture breadth | Runtime-defined | 196 pre-built | Edifice wins on breadth |
| Code quality | Minimal docs/tests | Strict tooling, 100% docs | Edifice wins on quality |
| Community | 28K stars, 80 contributors | 15 stars, 1 contributor | tinygrad has momentum |

### 5.3 Edifice vs Keras/Flax

| Dimension | Keras | Flax | Edifice |
|-----------|-------|------|---------|
| Scope | Full framework | Framework | Architecture library |
| API style | Sequential/Functional | Module-based | build/1 + registry |
| Tutorials | 100+ guides | Good | 18 guides, 0 notebooks |
| Ecosystem integration | TF/JAX backends | JAX only | Nx/Axon (EXLA/Torchx) |

### 5.4 Key Differentiators

What Edifice does **better** than most:
1. **Architecture breadth in pure Elixir** — 196 architectures, no Python dependency
2. **API consistency** — single unified pattern across all architectures
3. **Type safety** — typed option unions with zero `term()` fallbacks
4. **Static analysis** — Credo + Dialyzer + Doctor all passing clean
5. **Documentation density** — every module has architecture diagrams and references

What Edifice needs to **match** industry:
1. Pretrained weight loading (even 2-3 reference models)
2. Interoperability documentation (ONNX via axon_onnx)
3. Interactive notebooks (Livebook is Elixir's killer feature)
4. Gradient/numerical testing
5. Multi-version CI matrix
6. Coverage tracking

---

## 6. 107 Untested Source Files

Complete list of source files without dedicated test files, by family:

### Attention (29 files)
based, conformer, diff_transformer, dual_chunk, flash_linear_attention, fnet,
gated_attention, gqa, griffin, hawk, hgrn, hgrn_v2, infini_attention, kda,
lightning_attention, linear_transformer, mega, megalodon, mla, nsa, nystromformer,
perceiver, performer, retnet_v2, ring_attention, rnope_swa, sigmoid_attention,
tmrope, yarn

### Blocks (19 files)
adaptive_norm, alibi, bbox_head, causal_mask, cross_attention, depthwise_conv,
ffn, kv_cache, model_builder, patch_embed, rms_norm, rope, sdpa, sinusoidal_pe,
sinusoidal_pe_2d, softpick, ssmax, swiglu, transformer_block, upsample_2x

### SSM (19 files)
bimamba, common, gated_ssm, gss, h3, hybrid_builder, hybrid, hyena, hyena_v2,
hymba, mamba3, mamba_cumsum, mamba_hillis_steele, mamba_ssd, s4d, s4, s5,
ss_transformer, striped_hyena, zamba

### Meta (22 files)
adapter, capsule, distillation_head, dora, dpo, grpo, hybrid_builder,
hypernetwork, kto, lora, mixture_of_agents, mixture_of_depths,
mixture_of_tokenizers, moe, moe_v2, qat, remoe, rlhf_head, soft_moe,
speculative_decoding, speculative_head, switch_moe, test_time_compute

### Vision (15 files)
convnext, deit, dino_v2, efficient_vit, focalnet, gaussian_splat, mamba_vision,
metaformer, mlp_mixer, nerf, poolformer, swin, unet, vit

### Recurrent (13 files)
delta_net, gated_delta_net, min_gru, min_lstm, native_recurrence, recurrent,
reservoir, slstm, titans, transformer_like, ttt_e2e, ttt, xlstm, xlstm_v2

### Graph (10 files)
egnn, gat, gcn, gin, gin_v2, graph_sage, graph_transformer, message_passing,
pna, schnet

### Contrastive (8 files)
barlow_twins, byol, jepa, mae, siglip, simclr, temporal_jepa, vicreg

### Generative (20 files)
cogvideox, consistency_model, ddim, diffusion, dit, dit_v2, flow_matching, gan,
latent_diffusion, linear_dit, mar, mdlm, mmdit, normalizing_flow, rectified_flow,
score_sde, sit, soflow, transfusion, trellis

### Other families (18 files)
convolutional: conv, densenet, efficientnet, mobilenet
detection: detr, rt_detr, sam2
energy: ebm, hopfield, neural_ode
feedforward: bitnet, kan, kat, mlp, tabnet
export: gguf
inference: medusa
interpretability: sparse_autoencoder, transcoder
liquid: liquid
memory: engram, memory_network, ntm
multimodal: fusion
neuromorphic: ann2snn, snn
probabilistic: bayesian, evidential, mc_dropout
recurrent: (see above)
rl: decision_transformer, environment, cart_pole, grid_world, gae, policy_value, ppo_trainer
robotics: act, openvla
scientific: fno
sets: deep_sets, pointnet
transformer: byte_latent_transformer, decoder_only, multi_token_prediction, nemotron_h
utils: common, fused_ops, ode_solver, quantization
vision: (see above)
world_model: world_model

---

## 7. Scorecard Summary

| Category | Score (1-5) | Key Gap |
|----------|-------------|---------|
| Code quality tooling | 5 | — |
| API design | 5 | — |
| Documentation depth | 5 | — |
| Documentation conventions | 3 | Zero doctests |
| Test breadth | 3 | 48% modules untested individually |
| Shared block adoption | 3 | 88% of modules roll their own |
| Module size discipline | 3 | 5 modules >850 lines |
| ML-specific maturity | 2 | No weights, benchmarks, interop |
| Community health | 2 | Bus factor of 1 |

**Overall: 31/45**

Exceptional engineering quality for a solo project. Gaps are addressable and
fall into "breadth" categories (tests, notebooks, community) rather than "depth"
issues. The codebase is clean, well-typed, well-documented, and architecturally
sound.
