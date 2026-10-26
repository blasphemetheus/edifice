# Architecture completion audit for ExPhil and Melee

Audited October 24, 2026. Scope: source inspection of approximation markers,
bot-relevant stateful and generation paths, small cross-framework fixtures, and
the existing pretrained numerical suite. This is not a paper-by-paper certification
of every architecture. The subsequent recurrent follow-up enables MinGRU
stepping in ExPhil (see the linked benchmark report). Real activation
checkpointing remains deferred until the upstream Nx checkpoint work merges.

## Completed in this pass

- **PPO correctness:** inference-mode policy evaluation for both collection and
  differentiation; correct next-observation bootstrapping; environment continuity
  across rollouts; singleton minibatch updates; global gradient clipping before
  Adam. Prediction and the complete optimizer update can compile with EXLA.
  This trainer supports stateless categorical PolicyValue models. ExPhil's own
  recurrent training loop still needs a separate integration review.
- **GQA KV decoding:** `DecoderOnly.Cache` reuses trained backbone parameters and
  projects only new frames. Supports RoPE, NoPE, interleaved RoPE, input projection,
  optional final normalization, chunked prefill and rollback.
  `Generate.build_cache/1` integrates token embedding and the LM head in the same
  executable. Sampling and context-overflow behavior match full prediction.
- **Numerical validation executed:** eight small random-weight fixtures pass.
  Full pretrained tests now report four passes and one remaining Whisper
  precision failure, with no skips and unchanged tolerances. See
  [results and commands](plans/pretrained-validation-fixes.md).

## Next priorities

The recurrent microbenchmark is now complete; see
[measured CPU/GPU timings and ExPhil integration gaps](recurrent_inference_benchmarks.md).
Actual-checkpoint MinGRU/Mamba complete-policy latency and state/controller parity
were validated October 25 in ExPhil. Full-update Mamba batch-8 CUDA medians are
1.056 ms ordinary and 5.456 ms for one-row resync; see ExPhil's
`eval_runs/recurrent_resync_profile_20261025/RESULTS.md`.
The follow-up adds GatedDeltaNet and Mamba-2 step paths in Edifice and enables
MinGRU's adapter in ExPhil. SSD's matmul source-decay bug was corrected; old
checkpoints using that path need revalidation (details in the benchmark report).

| Priority | Finding and evidence | Useful next action |
|---|---|---|
| 1 | Actual-checkpoint MinGRU/Mamba replay and host embedding now pass CUDA baseline parity | Extend full-policy profiling to other deployed architectures and evaluate closed-loop quality separately |
| Completed October 25 | NSA/MoBA causal routing, actual selected-block execution and partial lengths fixed; local NSA attention uses gathers | [CPU/CUDA measurements and remaining paper-fidelity limits](../bench/results/sparse_attention_2026-10-25/README.md). Short contexts need not be faster; no fused sparse CUDA kernel yet |
| 3 | [RetNetV2](../lib/edifice/attention/retnet_v2.ex) uses LayerNorm at the layer named `group_norm` | Implement grouped normalization and compare a reference fixture before relying on retention fidelity |
| 3 | [Reservoir](../lib/edifice/recurrent/reservoir.ex) estimates spectral radius using a Frobenius-norm proxy | Measure the resulting radius and correct scaling before interpreting requested reservoir timescales |
| Completed October 26 | ViT attention heads, position table and pretrained layout corrected; ViT and ConvNeXt pass original numerical bounds | Keep numerical and focused regressions |
| Completed October 26 | DETR positional encoding, projection order, backbone padding and query-weight loading corrected | Class logits and boxes pass original numerical bounds |
| Conditional on audio | Whisper encoder loads and runs, but maximum numerical error remains `2.59e-4` against a `1e-4` bound | Use the Rust/Elixir intermediate comparison to isolate accumulated differences; decoder parity is not certified |

## Other approximation findings

- **GatedSSM** is its own gated/cumulative approximation, not interchangeable with
  Mamba. **SSTransformer** uses mean-dt gating and cumulative averaging rather
  than a full selective state recurrence. Label and benchmark these as distinct
  models. Mamba's per-channel dt fix and stateful parity tests do not establish
  compatibility with every external Mamba checkpoint.
- **Hymba** actually calls SSM discretization and a Blelloch scan despite a
  “simplified SSM” comment. That comment alone does not prove a placeholder;
  complete topology fidelity still needs reference validation.
- **MAE** uses an MLP-mixer-style encoder without transformer self-attention.
  **JEPA** operates on flat vectors with single-token attention. **EfficientNet**
  explicitly implements a dense 1D variant. None should be treated as a drop-in
  canonical image/video backbone solely from its name.
- **TRELLIS** now has a real voxel density codec, but its attention still derives
  Q/K/V from the same input without independent learned projections.
  **CogVideoX** uses pooling/averaging substitutions in its convolutional path.
  **GaussianSplat** uses only the DC spherical-harmonic component.
- **VAR** uses sinusoidal positions despite a learnable-position comment;
  **FreeTransformer** substitutes a dense latent encoder for bidirectional
  cross-attention; **DPO** constructs deterministic embedding values despite a
  learned-embedding comment. These are lower priorities for frame-vector bots.

Approximation is not automatically a defect: MinGRU/MinLSTM simplification and
S4D's diagonal structure are part of their definitions. Performer/Nystrom and
similar approximate attention methods need algorithm-specific reference checks.
Keyword searches alone cannot establish missing functionality.

## Using the cache for frame-vector policies

```elixir
opts = [embed_dim: 64, hidden_size: 64, num_heads: 4, num_kv_heads: 2,
        num_layers: 2, seq_len: 60, dropout: 0.0]
model = Edifice.Transformer.DecoderOnly.build(opts)
# Train/load params using this model's normal Axon parameter structure.
cache = Edifice.Transformer.DecoderOnly.Cache.build(opts ++ [compiler: EXLA])
state = cache.init.(1)
{features, state} = cache.predict.(params, frames, state)
# frames: {1, new_frame_count, 64}; features: {1, new_frame_count, 64}
saved = Edifice.Stateful.snapshot(state)
# Restore saved to replay from the same position after rollback.
```

For token generation, build the LM normally, then pass
`cache: Edifice.Serving.Generate.build_cache(model_opts ++ [compiler: EXLA])`
to `Generate.generate/3` with the same `seq_len`. Streaming APIs do not yet use
this cache. Validation currently covers f32 GQA without YaRN; other attention
types and YaRN are rejected.

The cache has a fixed capacity. Raw frame callers must reset and prefill the
retained window when it fills; generation does this automatically. Dropping only
old K/V entries would retain higher-layer activations derived from evicted
context and would not match a freshly recomputed sliding window. Consequently,
the exact sliding-window path returns to full prefill on every step after
capacity. Indefinite fast attention inference requires an explicit changed
context contract or a different architecture, not unnoticed cache eviction.

## Validation and benchmark

Targeted RL, generation, decoder, RoPE and stateful tests, including EXLA tests:
**152 tests and 2 properties, zero failures**. These cover terminal and truncated
rollout bootstrapping, parameter updates, rollback, chunk/token parity, sampling,
external embeddings, stop handling and context overflow.

```sh
devenv shell -- mix run scripts/bench_decoder_cache.exs
```

The CPU benchmark checks parity and measures a warmed single-frame decode against
the full last-timestep output at batch 1, window 60, hidden 64, two layers, EXLA
host. It includes output synchronization and the cache bounds check, but excludes
compilation and prefill. It is not a GPU or end-to-end ExPhil measurement.

A standalone run measured full prediction at **340 µs median / 420 µs p95**
versus cached decode at **78 µs median / 85 µs p95** (4.36× median speedup).
Earlier runs varied with host load; measure on the deployment machine.
