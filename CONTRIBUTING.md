# Contributing to Edifice

Thanks for your interest in contributing! This guide covers how to add new architectures, fix bugs, and maintain consistency across the library.

## Setup

```bash
git clone https://github.com/blasphemetheus/edifice.git
cd edifice
mix deps.get
mix test
```

Requires Elixir 1.18+ and OTP 27+. EXLA is optional (for GPU benchmarks only).

## Adding a New Architecture

### 1. Create the module

Place it in the appropriate family directory:

```
lib/edifice/<family>/<architecture>.ex
```

Families: `attention`, `convolutional`, `contrastive`, `energy`, `feedforward`, `generative`, `graph`, `liquid`, `memory`, `meta`, `neuromorphic`, `probabilistic`, `recurrent`, `sets`, `ssm`, `vision`.

### 2. Follow the module structure

Every architecture module must:

- Have a `@moduledoc` with overview, architecture diagram, options, usage example, and references
- Export `build/1` accepting a keyword list and returning `Axon.t()` (or `{Axon.t(), Axon.t()}` for encoder/decoder models)
- Include `@spec build(keyword()) :: Axon.t()`
- Export `output_size/1` returning the output dimension

```elixir
defmodule Edifice.Family.MyArch do
  @moduledoc """
  Short description.

  ## Architecture
  (ASCII diagram)

  ## Options
  - `:hidden_size` - ...

  ## Usage
      model = Edifice.Family.MyArch.build(hidden_size: 256)

  ## References
  - "Paper Title" (Author et al., Year)
  """

  require Axon

  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    # ...
  end

  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    # ...
  end
end
```

### 3. Register it

Add entries to `lib/edifice.ex`:

- In `@architecture_registry`: `my_arch: Edifice.Family.MyArch`
- In `list_families/0`: add `:my_arch` to the appropriate family list

### 4. Add to `mix.exs`

Add the module to the appropriate group in `groups_for_modules` inside `docs/0`.

### 5. Write tests

Create `test/edifice/<family>/<architecture>_test.exs` following this pattern:

```elixir
defmodule Edifice.Family.MyArchTest do
  use ExUnit.Case, async: true

  alias Edifice.Family.MyArch

  @opts [hidden_size: 64, ...]

  describe "build/1" do
    test "builds and runs" do
      model = MyArch.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      input = Nx.random_uniform({2, ...})
      params = init_fn.(Nx.template({2, ...}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, ...}
      assert Nx.all(Nx.is_finite(output)) |> Nx.to_number() == 1
    end
  end
end
```

Key assertions:
- Output shape matches expected dimensions
- All output values are finite (no NaN/Inf)
- Use small dimensions (hidden=64, batch=2) for fast tests

### 6. Update README

Add a row to the appropriate table in `README.md`.

## Code Style

- Run `mix format` before committing (enforced in CI)
- Run `mix credo` to check for issues
- `mix compile --warnings-as-errors` must pass

### Nx/Axon gotchas

- **Never use Elixir arithmetic on tensors** outside `defn` — use `Nx.multiply`, `Nx.add`, etc.
- **Batched matrix multiply** requires batch axes: `Nx.dot(a, [contract], [batch], b, [contract], [batch])`
- **Axon uses `:silu` not `:swish`** (same function: `x * sigmoid(x)`)
- **Use composable blocks** from `Edifice.Blocks` (FFN, SwiGLU, RoPE, etc.) instead of duplicating patterns

## Running Tests

```bash
# Full suite
mix test

# Single file
mix test test/edifice/ssm/mamba_test.exs

# Exclude slow tests (default in CI)
mix test --exclude slow
```

## Benchmarks

```bash
# Requires EXLA
mix run bench/architecture_profile.exs
mix run bench/scaling_profile.exs
```

## Pull Requests

- One architecture per PR (unless closely related, e.g., MinGRU + MinLSTM)
- Include the paper reference in the PR description
- Tests must pass: `mix test && mix format --check-formatted && mix credo`

## Releasing

Checklist for publishing a new version to Hex:

1. **Bump version** in `mix.exs` (`@version`)
2. **Update CHANGELOG.md** with a new section for the version
3. **Update README.md** install version (`{:edifice, "~> X.Y"}`)
4. **Update mix.exs description** if architecture count changed
5. **Run CI checks**:
   ```bash
   mix test
   mix format --check-formatted
   mix credo --strict
   mix dialyzer
   mix hex.build
   mix docs
   mix deps.unlock --check-unused
   ```
6. **Commit and tag**:
   ```bash
   git tag v0.X.0
   git push origin main --tags
   ```
7. **Publish**: `mix hex.publish`
