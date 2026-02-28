# Edifice Testing Guide

Edifice has 263 test files covering 231 registered architecture implementations across
26 families. The full suite takes ~9 minutes on BinaryBackend (CPU), so targeted test
runs are essential.

## Quick Reference

```bash
# Fast tests only (default - excludes slow/integration/exla_only/known_issue)
mix test

# Run only tests affected by recent changes (RECOMMENDED after editing code)
mix test --stale

# Run only tests that failed last time
mix test --failed

# Run a specific architecture's tests
mix test test/edifice/recurrent/deep_res_lstm_test.exs

# Run a specific test by line number
mix test test/edifice/recurrent/deep_res_lstm_test.exs:42

# Run by domain tag (architecture family)
mix test --only recurrent
mix test --only ssm
mix test --only attention

# Smoke test (one test per family, ~60s)
mix test.smoke

# Run with verbose output
mix test --trace

# Run with specific seed (reproducibility)
mix test --seed 12345
```

## Test Targeting: "I Changed X, Run Y"

**Do NOT run the full test suite after editing one architecture.** It takes ~9 minutes.

| Changed File(s) | Run |
|-----------------|-----|
| `lib/edifice/recurrent/<arch>.ex` | `mix test test/edifice/recurrent/<arch>_test.exs` |
| `lib/edifice/ssm/<arch>.ex` | `mix test test/edifice/ssm/<arch>_test.exs` |
| `lib/edifice/attention/<arch>.ex` | `mix test test/edifice/attention/<arch>_test.exs` |
| `lib/edifice/<family>/<arch>.ex` | `mix test test/edifice/<family>/<arch>_test.exs` |
| `lib/edifice/blocks/*.ex` | `mix test test/edifice/blocks/` then `mix test --stale` |
| `lib/edifice.ex` (registry) | `mix test test/edifice/registry_integrity_test.exs` then the specific arch test |
| Multiple files / unsure | `mix test --stale` |
| Pre-commit / full validation | `mix test` |

### Using `--stale` (Recommended)

`mix test --stale` tracks module dependencies transitively. After editing
`lib/edifice/recurrent/deep_res_lstm.ex`, it only reruns `deep_res_lstm_test.exs`
and any tests that depend on it (e.g., registry tests if they import the module).

```bash
# After editing one architecture:
mix test --stale    # Runs ~1-2 test files instead of 258

# After editing a shared block (e.g., Blocks.FFN):
mix test --stale    # Runs all tests that use FFN — more files, but still not all 258
```

First run builds a manifest and runs everything. Subsequent runs are fast.

## Mix Aliases

| Alias | Equivalent | Description |
|-------|-----------|-------------|
| `mix test` | *(default)* | Fast tests, excludes slow/integration/exla_only |
| `mix test.changed` | `mix test --stale` | Only tests affected by changes |
| `mix test.smoke` | *(curated subset)* | One test per family (~60s) |
| `mix test.recurrent` | `mix test --only recurrent` | Recurrent architecture tests |
| `mix test.ssm` | `mix test --only ssm` | State space model tests |
| `mix test.attention` | `mix test --only attention` | Attention architecture tests |
| `mix test.vision` | `mix test --only vision` | Vision architecture tests |
| `mix test.generative` | `mix test --only generative` | Generative model tests |

## Tags

### Speed/Environment Tags

| Tag | Description | Default |
|-----|-------------|---------|
| `:slow` | Tests taking >1 second (conv-heavy on BinaryBackend) | Excluded |
| `:integration` | Integration tests | Excluded |
| `:external` | External dependency tests | Excluded |
| `:exla_only` | Tests requiring EXLA backend | Excluded |
| `:known_issue` | Known failing tests | Excluded |
| (none) | Fast unit tests | Included |

### Domain Tags (Architecture Family)

Use `--only <tag>` to run a specific family:

| Tag | Directory | File Count |
|-----|-----------|------------|
| `:attention` | `test/edifice/attention/` | 41 |
| `:blocks` | `test/edifice/blocks/` | 25 |
| `:meta` | `test/edifice/meta/` | 24 |
| `:generative` | `test/edifice/generative/` | 24 |
| `:ssm` | `test/edifice/ssm/` | 19 |
| `:vision` | `test/edifice/vision/` | 14 |
| `:recurrent` | `test/edifice/recurrent/` | 13 |
| `:interpretability` | `test/edifice/interpretability/` | 12 |
| `:graph` | `test/edifice/graph/` | 10 |
| `:convolutional` | `test/edifice/convolutional/` | 6 |
| `:audio` | `test/edifice/audio/` | 6 |
| `:feedforward` | `test/edifice/feedforward/` | 5 |
| `:contrastive` | `test/edifice/contrastive/` | 5 |
| `:transformer` | `test/edifice/transformer/` | 4 |
| `:energy` | `test/edifice/energy/` | 4 |
| `:sets` | `test/edifice/sets/` | 3 |
| `:rl` | `test/edifice/rl/` | 3 |
| `:memory` | `test/edifice/memory/` | 3 |
| `:detection` | `test/edifice/detection/` | 3 |
| `:scientific` | `test/edifice/scientific/` | 2 |
| `:robotics` | `test/edifice/robotics/` | 2 |
| `:neuromorphic` | `test/edifice/neuromorphic/` | 2 |

```bash
# Run only recurrent architecture tests
mix test --only recurrent

# Run SSM + attention tests
mix test --only ssm --include attention

# Run a family including slow tests
mix test --only recurrent --include slow
```

## Test Structure

```
test/
├── test_helper.exs              # ExUnit configuration (1 line)
├── support/
│   └── test_helpers.ex          # Shared: assert_finite!, build_and_init, random_tensor
├── edifice/
│   ├── attention/               # MultiHead, RetNet, RWKV, Griffin, MLA, NSA, ... (41)
│   ├── blocks/                  # FFN, TransformerBlock, RMSNorm, SDPA, ... (25)
│   ├── meta/                    # MoE, LoRA, Adapter, EAGLE-3, mHC, ... (24)
│   ├── generative/              # Diffusion, VAE, GAN, FlowMatching, VAR, ... (24)
│   ├── ssm/                     # Mamba, S4, S4D, H3, Hyena, Mamba3, ... (19)
│   ├── vision/                  # ViT, Swin, UNet, ConvNeXt, DINOv2, ... (14)
│   ├── recurrent/               # LSTM, GRU, xLSTM, DeltaNet, TTT, ... (13)
│   ├── interpretability/        # SAE, Transcoder, Crosscoder, LEACE, ... (12)
│   ├── graph/                   # GCN, GAT, GraphSAGE, DimeNet, SE3, ... (10)
│   ├── convolutional/           # ResNet, DenseNet, TCN, MobileNet, ... (6)
│   ├── audio/                   # Whisper, EnCodec, VALL-E, SoundStorm, ... (6)
│   ├── feedforward/             # MLP, KAN, TabNet, BitNet, ... (5)
│   ├── contrastive/             # SimCLR, BYOL, JEPA, ... (5)
│   ├── transformer/             # DecoderOnly, NemotronH, ... (4)
│   ├── energy/                  # EBM, Hopfield, NeuralODE (4)
│   ├── sets/                    # DeepSets, PointNet, PointNet++ (3)
│   ├── rl/                      # PolicyValue, DecisionTransformer (3)
│   ├── memory/                  # NTM, MemoryNetwork, Engram (3)
│   ├── detection/               # DETR, RT-DETR, SAM2 (3)
│   ├── scientific/              # FNO, DeepONet (2)
│   ├── robotics/                # ACT, OpenVLA (2)
│   ├── neuromorphic/            # SNN, ANN2SNN (2)
│   ├── world_model/             # WorldModel (1)
│   ├── inference/               # Medusa (1)
│   ├── export/                  # GGUF export (1)
│   ├── registry_sweep_test.exs  # Forward-pass tests for all architectures (batch=1,4)
│   ├── registry_integrity_test.exs  # Build-only tests for all 202 architectures
│   ├── gradient_smoke_test.exs  # Gradient flow for all architectures
│   ├── output_size_sweep_test.exs  # output_size/1 for all modules
│   ├── coverage_batch_*.exs     # Coverage sweep tests (a-f batches)
│   ├── input_robustness_test.exs  # Edge-case inputs
│   └── property_test.exs        # Property-based tests
└── mix/tasks/                   # Mix task tests
```

### Cross-Cutting Test Files

These test files in `test/edifice/` (not in a family subdirectory) validate
properties across ALL architectures:

| File | Purpose | Speed |
|------|---------|-------|
| `registry_sweep_test.exs` | Forward pass at batch=1,4 for all architectures | Slow (~5 min) |
| `registry_integrity_test.exs` | Build-only test for all 202 registered architectures | Fast (~30s) |
| `gradient_smoke_test.exs` | Gradient flows through all 26 families | Moderate (~2 min) |
| `output_size_sweep_test.exs` | `output_size/1` returns correct values for all modules | Fast (~10s) |
| `coverage_batch_*.exs` | Coverage sweep in batches (a-f) | Slow |
| `input_robustness_test.exs` | Handles edge-case inputs | Moderate |
| `property_test.exs` | Property-based invariant checks | Slow |
| `audit_correctness_test.exs` | Correctness audits across architectures | Moderate |
| `architecture_doctest_test.exs` | Representative architecture doctests (MLP, LSTM, etc.) | Fast |
| `display_test.exs` | `mix edifice.viz` display module tests | Fast |

## Writing Tests for a New Architecture

Follow this pattern (example: `DeepResLSTM`):

### 1. Create the test file

Mirror the source path: `lib/edifice/recurrent/deep_res_lstm.ex`
becomes `test/edifice/recurrent/deep_res_lstm_test.exs`.

### 2. Use the standard structure

```elixir
defmodule Edifice.Recurrent.DeepResLSTMTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.DeepResLSTM

  # Use small dimensions for fast tests
  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 64

  # Helper: build model, init params, run forward pass
  defp build_and_run(opts) do
    model = DeepResLSTM.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  defp base_opts do
    [embed_dim: @embed_dim, hidden_size: @hidden_size, num_layers: 2,
     seq_len: @seq_len, dropout: 0.0]
  end

  # 1. Shape tests — verify output dimensions
  describe "build/1 shape tests" do
    test "returns an Axon model" do
      assert %Axon{} = DeepResLSTM.build(base_opts())
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      # ...
    end
  end

  # 2. Numerical stability — no NaN/Inf
  describe "numerical stability" do
    test "output is finite for random input" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      # ...
    end
  end

  # 3. output_size/1 utility
  describe "output_size/1" do
    test "returns hidden_size" do
      assert DeepResLSTM.output_size(hidden_size: 256) == 256
    end
  end

  # 4. Registry integration
  describe "registry integration" do
    test "Edifice.build(:deep_res_lstm, ...) works" do
      model = Edifice.build(:deep_res_lstm,
        embed_dim: @embed_dim, hidden_size: @hidden_size,
        num_layers: 1, seq_len: @seq_len)
      assert %Axon{} = model
    end
  end
end
```

### 3. Test checklist for new architectures

Every architecture test should cover:

- [ ] **Returns Axon model** — `assert %Axon{} = Module.build(opts)`
- [ ] **Correct output shape** — build, init, predict, check shape
- [ ] **Different dimensions** — embed != hidden, embed == hidden
- [ ] **Layer/config variation** — different num_layers, heads, etc.
- [ ] **Numerical stability** — no NaN/Inf for random and zero input
- [ ] **`output_size/1`** — returns expected value, default works
- [ ] **Registry integration** — `Edifice.build(:name, opts)` works

### 4. Keep test dimensions small

Use `@embed_dim 32`, `@hidden_size 64`, `@seq_len 8`, `@batch 2`.
Large dimensions slow tests on BinaryBackend. The goal is to verify
correctness, not benchmark performance.

## Test Helpers

`test/support/test_helpers.ex` provides shared utilities:

| Helper | Description |
|--------|-------------|
| `assert_finite!(tensor)` | Raise if tensor has NaN or Inf |
| `assert_finite!(container)` | Works with map containers (VAE outputs) |
| `build_and_init(model, input_map)` | Build model, init params, return `{predict_fn, params}` |
| `random_tensor(shape, seed)` | Generate deterministic random tensor |
| `any_nonzero_params?(params)` | Check params aren't all zeros |
| `flatten_params(params)` | Flatten nested param map to `[{path, tensor}]` |

```elixir
import Edifice.TestHelpers

# Assert output is finite (no NaN/Inf)
output |> assert_finite!("mamba_output")

# Build and init a model
{predict_fn, params} = build_and_init(model, %{"input" => input_tensor})

# Generate deterministic random input
input = random_tensor({2, 8, 32}, 42)
```

## Debugging Tests

```bash
# Run single test by file and line
mix test test/edifice/recurrent/deep_res_lstm_test.exs:42

# Show test names as they run
mix test test/edifice/recurrent/ --trace

# Run with same seed as a failed run
mix test --seed 12345

# Rerun only failed tests
mix test --failed

# Find flaky tests
mix test test/edifice/recurrent/deep_res_lstm_test.exs --repeat-until-failure 50
```

## Performance Notes

- **Full suite**: ~9 minutes on BinaryBackend (CPU)
- **Single family** (e.g., recurrent): ~30-60 seconds
- **Single architecture**: ~5-30 seconds
- **Conv-heavy architectures** (ResNet, UNet, ConvNeXt, Swin): slowest on BinaryBackend
- **Never run the full suite reflexively** — use `--stale` or target specific files
- **Vision tests** are slowest due to 2D convolutions on BinaryBackend; these are
  typically tagged `:slow` or have extended timeouts

## CI Configuration

```yaml
jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - run: mix test  # Default excludes, ~2-3 min

  full-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        partition: [1, 2, 3, 4]
    env:
      MIX_TEST_PARTITION: ${{ matrix.partition }}
    steps:
      - run: mix test --include slow --partitions 4
```
