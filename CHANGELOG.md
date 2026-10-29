# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **AI-generated release notes:** per-architecture validation labels and a
  guide separating independent references, pretrained comparisons, local
  behavior checks and unreviewed evidence. Scope, backend limits and the
  accompanying **AI-generated poem** are in `guides/validation.md`.

- Kernel tier benchmark (`bench/kernel_tier_bench.exs`, results and method in
  `docs/kernel_tier_benchmarks.md` and `bench/results/kernel_tiers_2026-10-26/`):
  GRU scan and Mamba selective scan, op-level and inside the 2-layer models,
  forward and gradient, as the XLA-compiled fallback (default and exact-f32
  precision), the FFI custom call, the NIF bridge and op-by-op eager
  evaluation, with every tier's dispatch arm asserted through the profiler.
  On an RTX 5090 with EXLA 1.0 the GRU kernel is slower than XLA's compiled
  fallback at every size (3-10x forward, 10-50x backward), the selective-scan
  kernel wins only at batch (3-5x inside Mamba at batch 32, slower at batch
  1), and the NIF bridge is always slower than the same kernel as a custom
  call. The one universal win of the custom calls is compile time (seconds
  instead of tens of seconds per shape).

- GPU parity matrix for the fused CUDA kernels
  (`test/edifice/cuda/kernel_matrix_gpu_test.exs`, `--only cuda_kernels`,
  needs `EDIFICE_TEST_CLIENT=cuda`): the selective-scan custom-call handlers
  forward and backward against the pure-Nx reference with the tier forced
  off, and two-layer GRU and Mamba models at batch 4 at exact precision.
  `dispatch_gating_test.exs` pins the tier-selection rules on CPU.
  `scripts/check_cuda_kernels.sh` builds the FFI library and syntax-checks
  every `.cu` (`--test` adds the matrix); `.github/workflows/cuda.yml` runs
  it on a self-hosted GPU runner.

- `Edifice.verified_architectures/0` and `Edifice.verification/1`: which
  architectures have evidence beyond the unit tests, by level (deployed as an
  ExPhil policy backbone, trained with tuned defaults in ExPhil, pretrained
  checkpoint parity, random-weight reference parity). The names are checked
  against the registry at compile time, and the "Verified in use" table in
  `guides/architecture_index.md` is checked against the map by the doc-count
  test.

- `Edifice.Stateful.jit_replay_rows/3` compiles selected-row replay with fused
  gather/scatter and preserves padded or unaffected rows. Tuple histories in
  `jit_replay/3` fuse history assembly into the executable. Includes CPU/CUDA
  resync profiling and regression coverage for nested carry and row selection.

- `Edifice.Stateful.jit_replay/3` compiles chunk-to-carry replay as a bounded
  loop for cold-start padding and resynchronization, avoiding per-frame host
  dispatch and graph unrolling. Includes parity checks and CPU replay benchmarks.

- Stateful single-frame inference for GatedDeltaNet (`:gated_delta_net`) and
  Mamba-2/SSD (`:mamba_ssd`), including causal convolution history, serializable
  rollback state and structured-mask state for SSD. Prefix outputs and parameter
  gradients are checked against full prediction.

- Reproducible recurrent inference benchmark for ExPhil-used backbones, with
  CPU/GPU results, eager/JIT/precompiled comparisons, prefix parity and rollback
  checks, and documented stateful integration gaps.

- `DecoderOnly.Cache` and `Generate.build_cache/1` provide fixed-capacity GQA
  KV decoding with trained weights, chunked prefill, RoPE offsets and rollback.
  Generation resets and prefills at capacity to preserve sliding-window output
  parity. Includes a host EXLA latency benchmark and an architecture completion
  audit focused on Melee inference and pretrained compatibility.

- `mix edifice.cuda.build` builds a separate EXLA FFI library for f32 GRU and
  Mamba selective-scan forward/backward kernels. Opt-in runtime loading registers
  handlers before compilation, without modifying or rebuilding EXLA itself.

- **JIT-compiled stateful stepping**: `Edifice.step/5` and
  `Edifice.init_state/3` accept `compiler:` (e.g. `EXLA`) — the step path
  compiles to a single executable per architecture (cached in
  `persistent_term`), 17–25x faster than eager dispatch on EXLA host CPU
  (0.05–0.09 ms/step at Melee dims). State remains a plain Nx container;
  rollback guarantees are re-pinned through the jitted path.
- **`output: :container` on all SAE/transcoder builds** — exposes
  `%{reconstruction, hidden, pre_acts}` so losses are finally wireable
  (default single-tensor output unchanged).
- **`Edifice.Interpretability.SAETrainer`**: first fit path for the SAE
  family — jitted momentum-SGD with decoder unit-norm renorm each step,
  AuxK dead-feature loss, and inference-threshold tracking.
- **BatchTopKSAE production inference**: `inference_threshold:` build
  option makes feature firing batch-composition-independent; pinned by a
  planted-dictionary recovery test.

- **Model manifest (self-describing checkpoints)**: `Edifice.Spec`,
  `Edifice.build_with_spec/3`, `Checkpoint.save(..., spec:)`,
  `Checkpoint.fetch_spec/1`, `Checkpoint.load_model/2` (rebuild from the
  embedded spec + parameter shape validation). Kills the
  silently-rebuilt-with-default-shapes checkpoint failure class.
- **`Edifice.Stateful` step contract** for O(1) per-frame inference:
  `init_state/2` + `step/3` with plain-Nx-container state
  (snapshot/serialize/restore = netplay rollback primitives). Implemented
  for MinGRU, Mamba, GatedSSM (`scan_mode: :causal`), GRU, and LSTM — each
  pinned by step-equals-full-forward equivalence tests at every prefix
  length and bitwise rollback replay tests.
- **`Edifice.Profile` `mode: :step`**: batch=1 per-step p50/p95 latency +
  recurrent-state memory footprint across stateful architectures;
  `bench/step_latency.exs` one-command comparison.
- **`Edifice.Interpretability.Probe`** (ported from exphil): linear probing
  with balanced accuracy, class-weighted CE, train-only standardization,
  label masking, degeneracy guards, shuffled-label control.
- **`Edifice.Interpretability.Attribution`** (ported from exphil):
  gradient×input saliency + dimension-group shares.

### Removed

- **The CUDA NIF tier.** `Edifice.CUDA.NIF`, the C bridge in `c_src/`, the
  f32 and bf16 kernel libraries (`priv/cuda/libedifice_cuda_kernels*.so`) and
  45 of the 47 kernel sources in `native/cuda/` are gone, together with the
  NIF arm of every `Edifice.CUDA.FusedScan` dispatcher and the standalone
  test and bench harnesses. The tier ran kernels on eager EXLA tensors only;
  the kernel tier benchmark (`docs/kernel_tier_benchmarks.md`) measured it
  1.3-3x slower than the same kernel as an XLA custom call and 3-10x slower
  than the XLA-compiled fallback, and gradients never reached it. The
  sources are preserved on the `archive/cuda-nif-kernels` branch.
  `scripts/check_cuda_kernels.sh` and the GPU workflow now build and test
  only the FFI library; `make -C native/cuda` has `ffi`, `test-compile`,
  `clean` and `info` targets.

- `bench/nif_overhead_profile.exs` (measured the removed bridge).

### Changed

- Default Nx/EXLA dependencies are published Hex 1.0 packages, so the release
  package can be built. `EDIFICE_UPSTREAM_NX=1` retains the validated Git revision
  with XLA 0.11. Published EXLA 1.0 / XLA 0.10 has the documented CPU erf/GELU
  defect; its unchanged regression is explicitly run with `--include upstream_fix`
  on the fixed profile. Native-call layout metadata supports both EXLA APIs.

- Documentation review before 0.3.0. Session handoffs, audits and superseded
  benchmark notes moved from the repository root and `docs/` into
  `docs/history/` (indexed there); the open plans into `docs/plans/`. The
  README family table, install line and Requirements, and the counts quoted
  in the guides now match the registry (268 architectures, 26 families) and
  the Nx 1.0 dependencies; `test/edifice/doc_counts_test.exs` now checks
  every guide, not four files.

- **GRU/LSTM parameter layout follows the EXLA target, not the NIF.**
  `Edifice.Recurrent.fused_rnn_available?/1` used "NIF library loaded" to
  decide between the fused layout (`<name>_input_proj` +
  `<name>_fused_scan.<name>_recurrent_kernel`) and `Axon.gru`/`Axon.lstm`,
  which in practice meant GPU machines got the fused layout and CPU machines
  did not. It now uses "EXLA built for CUDA" (`EXLA_TARGET=cuda` or
  `XLA_TARGET=cuda*`), so existing checkpoints keep loading on the machines
  that produced them. `EDIFICE_FUSED_RNN_LAYOUT=axon|fused` forces either
  layout. The scan runs the pure-Nx recurrence compiled in-graph either way.

- **GRU scan no longer has a CUDA custom-call handler.** The kernel tier
  benchmark (`docs/kernel_tier_benchmarks.md`) showed the fused GRU kernel
  3-10x slower forward and 10-50x slower backward than XLA's compiled
  fallback on an RTX 5090 at every size, so `mix edifice.cuda.build` now
  builds only the Mamba selective-scan handlers and
  `Edifice.Block.native_ops/0` lists only those two ops. The GRU scan
  compiles its pure-Nx recurrence in-graph whether or not
  `EDIFICE_FUSED_CUSTOM_CALL=1` is set. Parameter layouts are unchanged.

- **Nx 1.0 ecosystem.** Edifice now requires `nx ~> 1.0`, `axon ~> 0.9`,
  `polaris ~> 0.2`, and (optionally) `exla ~> 1.0`, `scholar ~> 0.5`,
  `safetensors ~> 0.2`. Consumers on the Nx 0.13 line should stay on Edifice
  0.2.x. Two upstream changes surface in Edifice's API surface:
  - `Axon.Loop.run/4` now returns the full `%Axon.Loop.State{}` instead of
    the model state (Axon dropped the loop output transform). Extract it with
    `%Axon.Loop.State{step_state: %{model_state: model_state}} = ...`; the
    `Recipes`, `Checkpoint` and notebook examples were updated accordingly.
  - Passing a plain `%{}` as the initial state to `Axon.Loop.run/4` or an
    `init_fn` logs a deprecation; use `Axon.ModelState.empty()`.
  `Edifice.Display.as_table/2` counts static-shape parameters that Axon 0.9
  stores as template tensors, and `Edifice.CUDA.Library` loads the FFI
  library through the public `EXLA.load_dylib/1`.
- `Edifice.Utils.Quantization.gptq/3` now requires actual per-layer input
  activations keyed by full parameter paths. A list remains valid for a single
  kernel; missing calibration raises. Dense kernels now use full Hessian error
  compensation, static group scales, and blocked updates instead of RTN.

- `Edifice.SSM.GatedSSM` gains `scan_mode: :causal | :legacy` (default
  `:legacy`, semantics unchanged). The causal mode is a true EMA recurrence
  and is required for stateful stepping.
- All interpretability `loss/4` functions now default `l1_coeff` properly
  (`keyword!`) and cast to f32 at entry per the numerical precision policy;
  `Crosscoder.loss/4` accepts lists or stacked tensors (was non-runnable).

### Fixed

- **Custom-call operand and result layouts.** EXLA emits
  `stablehlo.custom_call` without layout constraints, so XLA's layout
  assignment was free to hand the FFI handlers a non-row-major buffer. Inside
  the 2-layer GRU model it chose a time-major layout for the scan result (its
  consumers are a reduce over the hidden axis and a last-step slice), and the
  model's output was wrong by about 7 at batch 32 while batch 1, and every
  parameter-fed op-level test, were exact. The `Edifice.Block.FusedOp`
  custom-call implementation now pins row-major `operand_layouts` and
  `result_layouts` on all four native ops; the GPU matrix gained model-level
  GRU and Mamba cases at batch 4 and the CPU library test asserts the
  attributes.

- **Fused CUDA kernel dispatch.** Three gating bugs meant the tiers in
  the fused-scan dispatcher (`lib/edifice/cuda/fused_scan.ex`) were not the
  ones actually running:
  - The custom-call tier was reported available for every op whenever
    `EDIFICE_FUSED_CUSTOM_CALL=1` was set, although only the ops in
    `Edifice.Block.native_ops/0` (GRU scan, selective scan, forward and
    backward) have FFI handlers. The other 25 ops were routed into `Nx.block`,
    compiled the pure fallback and shadowed the NIF kernels; AutoTune then
    timed fallback against fallback. `custom_call_available?/1` is now per op.
  - "NIF loaded" was probed with `function_exported?/3` against stubs that
    always exist, so it was always true; the NIF module's new `loaded?/0`
    reports the real `:erlang.load_nif/2` outcome.
  - `Recurrent` tested `fused_lstm_scan/7` against an arity-8 stub, so the
    fused LSTM/GRU parameter layout could never be selected through the NIF.
  The NIF arm additionally requires the tensor to live on the `:cuda` client
  (host-client EXLA tensors were handed through as device pointers) and uses
  the input's device id instead of device 0.
- **NIF kernels read inputs before EXLA had finished writing them.** EXLA
  enqueues host->device copies from a thread pool and `Nx.to_pointer/2`
  does not wait for the buffer's definition event; the NIF launched on the
  legacy stream and could read a buffer whose copy was still pending or not
  yet queued. Seen as run-to-run NaN/garbage (GRU at H=512, flash attention)
  on an RTX 5090. Every NIF now synchronizes the device before launching and
  the dispatcher blocks on each input's readiness before taking its pointer.
- `FusedScan.reservoir_scan/3` built its initial state on the BinaryBackend
  on the NIF path and crashed taking its pointer.
- `PointNet` T-Net transforms used an unbatched contraction, pairing every
  sample's points with every sample's transform (`[batch, n, batch, k]`);
  with `use_t_net`/`use_feature_t_net` the classifier output was
  `{batch, batch, classes}`. The contraction is now batched.
- The test suite pins EXLA to its host client (`EDIFICE_TEST_CLIENT=cuda` to
  opt in to the GPU). JIT-compiled gradient checks were silently running on
  the GPU when one was present and produced intermittent NaN gradients
  under load.
- `MixtureOfAgents` honours `:hidden_size` for the aggregator instead of
  silently using the 256-wide default (`:aggregator_hidden_size` still wins).
- `MambaVision` accepts any number of stages; the stage dims, the hybrid-stage
  boundary and the downsample schedule are derived from `:depths` instead of
  a hardcoded four, and mismatched `:num_heads` lengths raise `ArgumentError`.
- `RTDETR` rejects an `:image_size` that is not a multiple of 32 up front
  (the stride-8/16/32 feature levels could not be fused otherwise).
- `Training.Adaptive`, `Training.Monitor` and `Display.Heatmap` detect NaN
  losses again: `Nx.to_number/1` returns `:nan`, so the old `x != x` check never
  fired and spikes/NaN streaks went unreported.
- `Recipes` epoch logging passed the event and the formatter to `Axon.Loop.log/3`
  in the wrong order; per-epoch metrics now print.
- CI is green again: formatting, Credo, Dialyzer (test env, `ex_unit` in the
  PLT), `mix docs`, `deps.audit`, the hex build, the benchmark-regression gate
  (now normalised by a hardware scale factor) and all test shards pass.
  `rustler_precompiled` 0.10 fixes the tokenizers `header_too_long` download
  failure; CPU-infeasible tests are tagged `:exla_only`; test jobs cap
  `--max-cases` at `nproc`.
- `mix test` runs warning-free (29 compiler warnings in tests removed).
- NSA and MoBA now gather selected blocks before token attention and preserve
  causal routing, including incomplete final blocks. NSA uses compressed
  probabilities for actual top-k retrieval and gathers its local window; MoBA
  forces the current block and enforces exact top-k cardinality on ties.
  NSA's pooling/scalar-gate approximation is explicitly documented. Parameter
  layouts remain compatible, but corrected outputs require checkpoint revalidation.
  Includes output/gradient references and CPU/CUDA sparse-execution benchmarks.

- Load OTP crypto before globally loading CUDA handlers, preventing lazy crypto
  NIF failures during fused Mamba Agent evaluation. Includes a cold-VM regression.

- Mamba-2/SSD matmul scan now excludes the source timestep's decay from each
  input contribution, matching the sequential recurrence across chunk sizes.
  This changes predictions from the affected matmul path in older checkpoints;
  revalidate those policies before deployment. State stepping also supports
  pointwise convolution without zero-sized history tensors.

- PPO now bootstraps from the final next observation, preserves environments
  across rollouts, updates singleton minibatches, clips global gradient norms
  before Adam, and compiles the full update when a compiler is supplied.
  Stateless policy collection and differentiation use consistent inference mode.
- Pretrained numerical tests explicitly select EXLA host execution. ResNet passes;
  ViT and Whisper layout failures and the DETR mismatch remain documented failures.

- Mamba's Common sequence and step discretization now retain each channel's
  timestep in the input contribution, matching the fused scan. SSM analysis
  uses the same recurrence. Existing checkpoints run through the old Common
  path can change outputs; validate them before replacing deployed models.
- RLA/RDN backward fallback differentiates the full recurrence, including key
  gradients and both state-retrieval paths, instead of returning partial gradients.
- The development shell prefers the installed NixOS CUDA driver over a stale
  compatibility library that can prevent XLA from finding the GPU.
- Training helper documentation no longer promises rematerialization or measured
  memory savings for the current pass-through checkpoint APIs.

- Fused generation retains prompt and generated context on every prediction,
  including sliding-window overflow, one-token requests, and per-row stopping.
- TRELLIS sparse encoding selects occupied cells; dense decoding scatters
  features to voxel coordinates instead of broadcasting a global mean.
- SSM analysis exposes signed scan coefficients validated against Mamba probe
  output, excludes future frames, and computes decay summaries without log-exp
  underflow.
- SSM analysis resolves discretization constants outside `defn` tracing.
- EXLA custom-call registration is conditional, so consumers can compile and
  run Edifice without EXLA. CI checks an isolated CPU-only consumer.
- Sharding tests skip only device-dependent cases on single-device hosts;
  CI also executes them with two virtual CPU devices.
- ReMoE gradient smoke input covers both signs and uses seeded initialization
  to avoid random all-inactive routing.

### Deprecated

- `Edifice.SSM.GatedSSM.step/4` and `init_cache/1` — never numerically
  matched the forward; use the `Edifice.Stateful` contract with
  `scan_mode: :causal`.

### Documentation

- Status warnings on GatedSAE, MatryoshkaSAE, CrossLayerTranscoder, and
  DASProbe per INTERP_AUDIT_2026-07-15 (known-incorrect implementations;
  see each moduledoc).

## [0.2.0] - 2026-02-25

### Added

186 registered architectures across 25 families (up from 92 across 16 in v0.1.0). 94 new architectures grouped by family:

- **Attention** (35 total, +20): Hawk, RetNet v2, Megalodon, Lightning Attention, GLA v2, HGRN v2, Flash Linear Attention, KDA (Kernelized Deformable Attention), Gated Attention, SSMax (Scalable-Softmax), Softpick (non-saturating sparse normalization), RNoPE-SWA (sliding window without positional encoding), YaRN (context window extension via frequency-scaled RoPE), NSA (Native Sparse Attention from DeepSeek-V3/V4), TMRoPE (Time-aligned Multimodal RoPE), Dual Chunk Attention, Based (Taylor expansion linear attention), InfiniAttention (compressive memory + local attention), Conformer (conv + transformer for audio), Mega (EMA + single-head gated attention), RingAttention (chunked ring-distributed), MLA (Multi-Head Latent Attention), DiffTransformer (dual softmax noise-cancelling)
- **Audio** (3, NEW family): SoundStorm (parallel audio generation via masked prediction), EnCodec (neural audio codec), VALL-E (zero-shot TTS via neural codec language modeling)
- **Contrastive** (8, +3): JEPA (Joint Embedding Predictive Architecture), Temporal JEPA, SigLIP (sigmoid contrastive loss for language-image pretraining)
- **Generative** (22, +11): MMDiT (multi-modal DiT), SoFlow, VAR (Visual Autoregressive Modeling, NeurIPS 2024 Best Paper), Linear DiT/SANA (DiT with linear attention), SiT (Scalable Interpolant Transformer), Transfusion (unified AR text + diffusion image), MAR (Masked Autoregressive Generation), CogVideoX (text-to-video diffusion with 3D causal VAE), TRELLIS (structured 3D latents with sparse transformer + rectified flow), DiT v2, Consistency Model
- **Graph** (9, +2): GIN v2 (GIN with edge features), EGNN (E(n)-equivariant graph neural network)
- **Inference** (1, NEW family): Medusa (multi-head speculative decoding for 2-3x speedup)
- **Interpretability** (2, NEW family): Sparse Autoencoder, Transcoder
- **Memory** (3, +1): Engram (O(1) hash-based associative memory via locality-sensitive hashing)
- **Meta** (22, +11): DPO (Direct Preference Optimization), KTO (Kahneman-Tversky Optimization), GRPO (Group Relative Policy Optimization), MoE v2 (aux-loss-free load balancing), DoRA, Speculative Decoding, Test-Time Compute, Mixture of Tokenizers, Speculative Head, Distillation Head, QAT (Quantization-Aware Training), Hybrid Builder (flexible hybrid architecture composition), MixtureOfDepths, MixtureOfAgents, RLHFHead
- **Multimodal** (1, NEW family): Multimodal Fusion
- **Recurrent** (15, +7): sLSTM, xLSTM v2, Gated DeltaNet, TTT-E2E (end-to-end test-time training), Native Recurrence, plus previously added recurrent variants
- **RL** (1, NEW family): PolicyValue
- **Robotics** (2, NEW family): ACT (Action Chunking Transformer for robot imitation learning), OpenVLA (Vision-Language-Action model)
- **Scientific** (1, NEW family): FNO (Fourier Neural Operator)
- **SSM** (19, +5): StripedHyena (gated conv + Hyena hybrid), Mamba-3 (complex state dynamics, trapezoidal discretization, MIMO rank-r), GSS (Gated State Spaces), Hyena v2, Hymba, SS Transformer
- **Transformer** (4, NEW family): Decoder-Only (GPT-style with GQA, RoPE, SwiGLU, RMSNorm), Multi-Token Prediction, Byte Latent Transformer, Nemotron-H (NVIDIA's hybrid Mamba-Transformer)
- **Vision** (15, +9): FocalNet (focal modulation), PoolFormer (pooling-based MetaFormer), NeRF (positional encoding + MLP for radiance fields), Gaussian Splatting (real-time differentiable radiance field rendering), MambaVision, DINOv2 (self-supervised vision backbone via self-distillation), MetaFormer + CAFormer (pluggable token mixer framework), EfficientViT (O(n) linear attention with cascaded group attention)
- **World Model** (1, NEW family): World Model
- **Feedforward**: KAT (Kolmogorov-Arnold Transformer), BitNet (ternary/binary weight quantization)
- **Blocks**: CausalMask (unified mask creation), DepthwiseConv (1D depthwise separable convolution)

Infrastructure and tooling:

- GGUF export for decoder-only models
- KV cache for inference
- Quantization toolkit (QAT module)
- `shell.nix` for reproducible Erlang 27 + Elixir 1.18 + CUDA dev environment
- `livebook.sh` script for attached-mode Livebook with EXLA/CUDA
- `ARCHITECTURE_ROADMAP.md` tracking remaining architectures by priority tier
- `.credo.exs` configuration and `CONTRIBUTING.md` with architecture addition guide

Notebooks (12 Livebook notebooks):

- Architecture zoo guided tour
- Architecture comparison (decision boundaries)
- Sequence modeling (RNN vs SSM vs Transformer)
- MLP training end-to-end walkthrough
- Graph classification (GCN vs GAT vs GIN)
- Generative models (VAE)
- Small language model (Transformer + Mamba char-level LM)
- Liquid neural networks
- LM architecture shootout
- Softmax shootout (Softmax vs SSMax vs Softpick)
- Guided tour demo with detailed ML explanations
- Notebook index with descriptions and categories

Documentation:

- 18 conceptual guide documents (up from 12) covering architecture taxonomy, ML foundations, learning path, meta-learning, and reading Edifice source
- Architecture landscape survey and research docs
- 100% moduledoc coverage across all 211 modules
- 100% `@spec` coverage on all public functions
- Typed `@type build_opt` for all `build/1` modules

Benchmarks:

- Full architecture sweep benchmark covering all families
- Training throughput and memory profile benchmarks
- GPU runtime warmup phases for accurate measurements

Testing:

- 2822+ tests (up from ~1160 in v0.1.0)
- Gradient smoke tests with JIT-wrapped `value_and_grad`
- Parameter sensitivity tests and EXLA.Backend variants for conv models
- Dialyzer added to CI, zero warnings enforced

### Enhanced

- Decoder-Only transformer: added `:attention_type` option, iRoPE (interleaved RoPE) support
- MultiHead and GQA attention: added `:rope` option for built-in RoPE integration
- TTT (Test-Time Training): added `:variant` option for `:linear` and `:mlp` inner models
- TransformerBlock: added `:custom_ffn` callback for non-standard feed-forward networks
- xLSTM: added `:mlstm` registry alias (`Edifice.build(:mlstm, opts)`)
- sLSTM: log-domain stabilization (m_t state), recurrent connections (R*h_{t-1}), proper normalization (max(|n_t|, 1))
- MoE v2: aux-loss-free load balancing via bias mode
- DiffTransformer: simplified V2 with scalar lambda and RMSNorm only
- Liquid Neural Networks: exact analytical ODE solver added, set as default
- API option names normalized across all modules for consistency

### Changed

- Removed unnecessary `require Axon` from 104 modules (Axon has zero macros)
- BitNet `bitlinear_impl` comment clarified: STE is implicit via Axon's param/callback architecture
- Removed broken `sliding_window` registry alias
- Dependency constraints tightened to match tested versions
- All 72 Credo warnings resolved across 41 files
- All Dialyzer errors resolved; strict formatting enforced
- Notebooks default to 10 epochs with EXLA optional and dual setup cells (standalone / attached mode)

### Fixed

- **EnCodec**: channels-first bug fixed across all conv/conv_transpose layers
- **Gaussian Splatting**: render pipeline rewritten for JIT/EXLA compatibility; `render_layer` arity mismatch resolved
- **Gradient smoke tests**: JIT-wrapped `value_and_grad` for conv model gradients; `put_nested` no longer destroys sibling params
- **MessagePassing**: aggregate batch axes added to `Nx.dot` for correct batched matrix multiplication; `global_pool` refactored for 100% coverage
- **RetNet**: corrected `recurrent_retention_step` batching
- **TTT**: paper-faithful initialization for numerical stability
- **FNet**: replaced `Nx.fft` with real DFT matrix multiply for compatibility; `Nx.real` taken after each FFT to avoid complex intermediates
- **RWKV**: fixed seq_len=1 compile failure; silenced Range warnings in parallel scans
- **sLSTM**: log-domain stabilization for numerical stability
- **MoE routing**: top-k uses `Nx.top_k` with one-hot mask; hash routing properly selects expert; Switch MoE uses straight-through top-1 selection
- Paper-faithfulness corrections across 8 architecture modules
- 5 GPU test failures resolved in capsule and conv gradient tests
- VAE training fixed (single Axon graph); graph viz range bug resolved
- FocalNet bench spec corrected to match flat-architecture API

## [0.1.1] - 2026-02-14

### Fixed

- **MoE top-k routing**: `top_k_forward` now uses `Nx.top_k` indices with one-hot mask for correct expert selection (was ignoring indices and averaging first k experts)
- **MoE hash routing**: `hash_forward` now properly selects expert by hash (was always returning first expert)
- **SwitchMoE routing**: Replaced soft weighted average with hard top-1 selection via straight-through estimator, restoring the sparsity that defines Switch Transformer
- **SchNet filter generation**: Added learned 2-layer filter-generating network (RBF -> Dense -> SiLU -> Dense) replacing naive mean aggregation
- **ConvNeXt layer scale**: Changed from frozen constant to learnable `Axon.param`, matching Liu et al. 2022
- **MessagePassing aggregate**: Added batch axes to `Nx.dot` for correct batched matrix multiplication
- **SNN docstring**: Corrected reset mechanism description from hard reset to soft reset (subtract threshold)

### Changed

- **KAN default basis**: Changed from `:sine` (Fourier features) to `:bspline` (cubic B-spline via Cox-de Boor), faithful to Liu et al. 2024. Previous bases (`:sine`, `:chebyshev`, `:fourier`, `:rbf`) remain available as options
- **TTT W_0 initialization**: Changed from `0.01 * Identity` to `:glorot_uniform` per Sun et al. 2024
- **TTT output RMS norm**: Made optional via `:output_rms_norm` option (default: `false`), was unconditionally applied

### Removed

- Unused `_x` and `_dt` parameters from Liquid `integrate_ode`

## [0.1.0] - 2026-02-14

### Added

- 92 registered architectures across 16 families
- Unified interface: `Edifice.build(:name, opts)` and `Edifice.list_architectures()`
- **Feedforward**: MLP, KAN (Kolmogorov-Arnold Networks), TabNet
- **Convolutional**: Conv1D/2D, ResNet, DenseNet, TCN, MobileNet, EfficientNet
- **Recurrent**: LSTM, GRU, xLSTM, MinGRU, MinLSTM, DeltaNet, TTT, Titans, Reservoir (ESN)
- **State Space Models**: Mamba (parallel scan), Mamba-2 (SSD), MambaCumsum, MambaHillisSteele, S4, S4D, S5, H3, Hyena, BiMamba, GatedSSM, Jamba, Zamba
- **Attention**: Multi-Head (sliding window, hybrid), GQA, Perceiver, FNet, LinearTransformer, Nystromformer, Performer, RetNet, RWKV-7, GLA, HGRN-2, Griffin/Hawk
- **Vision**: ViT, DeiT, Swin Transformer, U-Net, ConvNeXt, MLP-Mixer
- **Generative**: VAE, VQ-VAE, GAN (WGAN-GP), DDPM Diffusion, DDIM, DiT, Latent Diffusion, Consistency Model, Score SDE, Flow Matching, Normalizing Flows
- **Contrastive**: SimCLR, BYOL, Barlow Twins, MAE (Masked Autoencoder), VICReg
- **Graph**: GCN, GAT, GIN, GraphSAGE, Graph Transformer, PNA, SchNet
- **Sets**: DeepSets, PointNet
- **Energy**: EBM (contrastive divergence), Modern Hopfield Networks, Neural ODE
- **Probabilistic**: Bayesian Neural Networks, MC Dropout, Evidential Neural Networks
- **Memory**: Neural Turing Machine, Memory Networks
- **Meta**: MoE, Switch MoE, Soft MoE, LoRA, Adapter, Hypernetworks, Capsule Networks
- **Liquid**: Liquid Neural Networks (continuous-time ODE)
- **Neuromorphic**: SNN (LIF neurons), ANN2SNN conversion
- **Building Blocks**: RMSNorm, SwiGLU, FFN, RoPE, ALiBi, PatchEmbed, SinusoidalPE, AdaptiveNorm, CrossAttention
- 12 conceptual guide documents covering theory, evolution, and decision tables for all families
- `CONTRIBUTING.md` with architecture addition guide, test patterns, and Nx/Axon gotchas
- ~1160 tests covering all architecture families
