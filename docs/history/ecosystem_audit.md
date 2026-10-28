# Elixir-Nx Ecosystem Audit — Edifice

**Date**: 2026-03-22

## Current Dependencies

| Library | Version | Role | Usage Grade |
|---------|---------|------|-------------|
| **Nx** | ~> 0.11 (bleeding-edge fork) | Core | A+ |
| **Axon** | ~> 0.8 | Core | A |
| **Polaris** | ~> 0.1 | Core | A |
| **EXLA** | ~> 0.11 (optional) | GPU backend | A+ |
| **Safetensors** | ~> 0.1.3 (optional) | Weight loading | A |
| **Req** | ~> 0.5 (optional) | HuggingFace Hub | A |
| **Kino** | ~> 0.14 (dev) | Livebook integration | B |
| **Kino.VegaLite** | ~> 0.1 (dev) | Charts | **Unused** |

## Libraries NOT Used (Should They Be?)

| Library | Verdict | Reason |
|---------|---------|--------|
| **Bumblebee** | Skip | Edifice has its own pretrained loading (key maps + Safetensors), covers 5x more architectures |
| **Scholar** | Consider | Has metrics (F1, precision, recall, confusion matrix) that could enhance Recipes |
| **Tokenizers** | Consider | Text preprocessing for LM recipes — currently Edifice expects pre-tokenized input |
| **Explorer** | Skip | Dataframes aren't a fit for tensor-first architecture library |
| **Ortex** | Skip | ONNX runtime, different execution model |
| **StbImage** | Skip | Image loading — out of scope, user handles preprocessing |

---

## Axon — Grade: A

### Well Used
- **Axon.Loop** — deeply integrated. All 5 recipes use trainer/metric/validate/early_stop/checkpoint/handle_event. Custom handlers for Monitor, Adaptive, MemoryTracker, AutoTuneProfiler, Checkpoint.
- **Axon.MixedPrecision** — wrapped by `Edifice.MixedPrecision` with architecture-aware norm exclusions.
- **Axon.ModelState** — freeze/unfreeze for fine-tuning strategies (head_only, LoRA, full).
- **Axon.layer/Axon.nx** — custom layers throughout (RMSNorm, RoPE, causal mask, etc.).

### Underused
- **Axon.Losses** — only `categorical_cross_entropy` used directly. Edifice reimplements Huber, InfoNCE, DINO, PPO losses manually. These are domain-specific so this is acceptable, but `Axon.Losses.huber/2` exists.
- **Axon.Display** — replaced by `Edifice.Display` (better: adds Mermaid, works without Kino). Fine.
- **Axon.Initializers** — never imported. Uses atoms (`:glorot_uniform`) or inline closures. Fine.

### Not Used (Correctly Skipped)
- **Axon.Compiler** — direct EXLA JIT preferred for control.
- **Axon.Schedules** — Polaris.Schedules used instead (same functionality).

---

## Polaris — Grade: A

### Well Used
- `Polaris.Optimizers.adamw/1` — all 5 recipes
- `Polaris.Optimizers.adam/1` — RL PPO trainer
- `Polaris.Schedules.cosine_decay/1` — standard schedule
- `Polaris.Updates.clip_by_global_norm/1` — language model gradient clipping
- `Polaris.Updates.compose/2` — composing clipping + optimizer

### Not Used
- SGD, AdaGrad, RMSProp, LAMB, etc. — AdamW is the standard, fine.
- Linear/exponential/polynomial schedules — cosine dominates modern ML, fine.

---

## Kino/VegaLite — Grade: B-

### Used
- **Kino.SmartCell** — `Edifice.SmartCell.ModelExplorer` for interactive model building in Livebook
- **Kino.JS.Live** — dynamic UI in smart cell
- Guard: `if Code.ensure_loaded?(Kino.SmartCell)` — proper conditional compilation

### Not Used
- **Kino.VegaLite** — in mix.exs but never imported or used in lib/
- No training curve visualization
- No loss/metric plotting in notebooks

---

## Safetensors + Pretrained — Grade: A

- `Edifice.Pretrained.Loader` — loads SafeTensors format with parameter name mapping
- `Edifice.Pretrained.Hub` — downloads from HuggingFace, parses config.json
- Key maps for ViT, ResNet, ConvNeXt, DETR, Whisper
- Well-designed: separates format (SafeTensors) from mapping (key maps) from transforms

---

## Actionable Improvements

### 1. Use VegaLite for Training Curves (Low Effort, High UX)

The dependency is already in mix.exs but unused. Add a training visualization module:

```elixir
# Edifice.Display.TrainingPlot
def loss_curve(metrics_history) do
  VegaLite.new(width: 600, height: 300)
  |> VegaLite.data_from_values(metrics_history)
  |> VegaLite.mark(:line)
  |> VegaLite.encode_field(:x, "step", type: :quantitative)
  |> VegaLite.encode_field(:y, "loss", type: :quantitative)
end
```

### 2. Add Scholar Metrics to Recipes (Low Effort, Medium Value)

Scholar provides production metrics that Recipes currently lack:

```elixir
# In Recipes.classify, add:
Scholar.Metrics.Classification.f1_score(y_true, y_pred, num_classes: n)
Scholar.Metrics.Classification.confusion_matrix(y_true, y_pred, num_classes: n)
```

Would require adding `{:scholar, "~> 0.3"}` to deps.

### 3. Wire LoRA into Fine-Tuning Recipe (Medium Effort, High Value)

`Edifice.Meta.LoRA` exists but `Recipes.fine_tune` doesn't use it:

```elixir
# Recipes.fine_tune with strategy: :lora already unfreezes lora params,
# but doesn't INSERT LoRA adapters into the model. It only unfreezes
# params matching "lora" in the name. Need to actually apply LoRA
# layers before building the loop.
```

### 4. Tokenizers Integration (Medium Effort, High Value)

For text-based recipes (language_model, fine_tune), a tokenizer bridge would
let users pass raw strings instead of pre-tokenized tensor IDs:

```elixir
# Optional dep: {:tokenizers, "~> 0.4"}
Edifice.Tokenizers.from_pretrained("meta-llama/Llama-3-8B")
```

### 5. Distributed Training Documentation (Low Effort)

EXLA supports multi-node via Nx.Serving distribution and shard_jit.
Edifice.Sharding wraps single-host multi-GPU. Document the path to
multi-node with `:pg` process groups.

---

## Summary

Edifice uses the elixir-nx ecosystem well. The core stack (Nx + Axon + Polaris + EXLA)
is deeply integrated. The main gaps are:

1. **VegaLite** — already a dependency, just needs wiring for training visualization
2. **Scholar metrics** — would enhance Recipes with F1/precision/recall
3. **LoRA in Recipes** — architecture exists, recipe doesn't use it yet
4. **Tokenizers** — would complete the text pipeline story

None of these are critical — Edifice works well without them. They're quality-of-life
improvements for users who want batteries-included training workflows.
