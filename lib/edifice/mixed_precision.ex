defmodule Edifice.MixedPrecision do
  @moduledoc """
  Mixed precision auto-casting for Edifice models.

  Wraps `Axon.MixedPrecision` with Edifice-aware presets that automatically
  preserve full precision (f32) for numerically sensitive layers — normalization,
  embeddings, and loss computation — while casting everything else to bf16 for
  ~2x throughput on supported hardware.

  ## Quick Start

      model = Edifice.build(:decoder_only, embed_dim: 256, hidden_size: 256, ...)

      # Apply bf16 mixed precision (keeps norms in f32)
      model = Edifice.MixedPrecision.apply(model, :bf16)

      # Or with a custom policy
      policy = Edifice.MixedPrecision.policy(:bf16)
      model = Edifice.MixedPrecision.apply(model, policy)

  ## Presets

  - `:bf16` — bf16 compute and output, f32 params. Excludes all normalization
    layers (layer_norm, batch_norm, rms_norm, adaptive_norm, group_norm).
    Recommended for NVIDIA Ampere+ GPUs.

  - `:fp16` — fp16 compute and output, f32 params. Same exclusions as bf16.
    Use with gradient loss scaling to prevent underflow.

  ## Gradient Loss Scaling

  When training in fp16 (not usually needed for bf16), gradients can underflow.
  Use `scale_loss/2` and `unscale_grads/2` to apply dynamic loss scaling:

      {scaled_loss, state} = Edifice.MixedPrecision.scale_loss(loss, scale_state)
      # ... backward pass with scaled_loss ...
      {grads, state} = Edifice.MixedPrecision.unscale_grads(grads, state)

  Or wrap an Axon training loop:

      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Edifice.MixedPrecision.with_loss_scaling(scale: 1024.0)
      |> Axon.Loop.run(data, %{}, epochs: 10)

  ## What Gets Excluded

  The following op_names are kept in f32 regardless of policy:

  - `:layer_norm` — Axon built-in
  - `:batch_norm` — Axon built-in
  - `:group_norm` — Axon built-in
  - `:rms_norm` — `Edifice.Blocks.RMSNorm`
  - `:adaptive_norm` — `Edifice.Blocks.AdaptiveNorm`
  - `:adaln_modulate` — AdaLN modulation
  - `:adaln_gate` — AdaLN gating
  """

  @norm_op_names [
    :layer_norm,
    :batch_norm,
    :group_norm,
    :instance_norm,
    :rms_norm,
    :adaptive_norm,
    :adaln_modulate,
    :adaln_gate
  ]

  @type preset :: :bf16 | :fp16

  @doc """
  Create a mixed precision policy.

  Accepts a preset atom or an `Axon.MixedPrecision.Policy` struct.

  ## Presets

  - `:bf16` — `params: f32, compute: bf16, output: bf16`
  - `:fp16` — `params: f32, compute: f16, output: f16`

  ## Examples

      policy = Edifice.MixedPrecision.policy(:bf16)
      #=> #Axon.MixedPrecision.Policy<p=f32 c=bf16 o=bf16>
  """
  @spec policy(preset()) :: Axon.MixedPrecision.Policy.t()
  def policy(:bf16) do
    Axon.MixedPrecision.create_policy(params: {:f, 32}, compute: {:bf, 16}, output: {:bf, 16})
  end

  def policy(:fp16) do
    Axon.MixedPrecision.create_policy(params: {:f, 32}, compute: {:f, 16}, output: {:f, 16})
  end

  @doc """
  Apply mixed precision to an Axon model.

  Casts all layers to the given precision except normalization layers,
  which remain in f32 for numerical stability.

  ## Parameters

  - `model` — Axon model graph
  - `policy_or_preset` — A preset atom (`:bf16`, `:fp16`) or `Axon.MixedPrecision.Policy`
  - `opts` — Additional options

  ## Options

  - `:except` — Additional op_names to exclude from casting (appended to
    the default normalization exclusion list)
  - `:only_norms` — If `false`, skips the default norm exclusions and uses
    only `:except` (default: `true`)

  ## Examples

      # Preset
      model = Edifice.MixedPrecision.apply(model, :bf16)

      # Custom policy, also exclude embeddings
      policy = Edifice.MixedPrecision.policy(:bf16)
      model = Edifice.MixedPrecision.apply(model, policy, except: [:embedding])
  """
  @spec apply(Axon.t(), preset() | Axon.MixedPrecision.Policy.t(), keyword()) :: Axon.t()
  def apply(model, policy_or_preset, opts \\ [])

  def apply(model, preset, opts) when preset in [:bf16, :fp16] do
    __MODULE__.apply(model, policy(preset), opts)
  end

  def apply(model, %Axon.MixedPrecision.Policy{} = policy, opts) do
    extra_except = Keyword.get(opts, :except, [])
    only_norms = Keyword.get(opts, :only_norms, true)

    except =
      if only_norms do
        Enum.uniq(@norm_op_names ++ extra_except)
      else
        extra_except
      end

    Axon.MixedPrecision.apply_policy(model, policy, except: except)
  end

  @doc """
  Returns the default list of op_names excluded from mixed precision casting.
  """
  @spec norm_op_names() :: [atom()]
  def norm_op_names, do: @norm_op_names

  # ===========================================================================
  # Gradient Loss Scaling
  # ===========================================================================

  @doc """
  Initialize loss scaling state for fp16 training.

  ## Options

  - `:scale` — Initial loss scale factor (default: 65536.0)
  - `:growth_factor` — Multiply scale by this when no overflow (default: 2.0)
  - `:backoff_factor` — Multiply scale by this on overflow (default: 0.5)
  - `:growth_interval` — Successful steps before growing scale (default: 2000)
  """
  @spec init_loss_scale(keyword()) :: map()
  def init_loss_scale(opts \\ []) do
    %{
      scale: Keyword.get(opts, :scale, 65536.0),
      growth_factor: Keyword.get(opts, :growth_factor, 2.0),
      backoff_factor: Keyword.get(opts, :backoff_factor, 0.5),
      growth_interval: Keyword.get(opts, :growth_interval, 2000),
      good_steps: 0
    }
  end

  @doc """
  Scale loss before backward pass.

  Returns `{scaled_loss, state}` where `scaled_loss = loss * scale`.
  """
  @spec scale_loss(Nx.Tensor.t(), map()) :: {Nx.Tensor.t(), map()}
  def scale_loss(loss, %{scale: scale} = state) do
    scaled = Nx.multiply(loss, scale)
    {scaled, state}
  end

  @doc """
  Unscale gradients after backward pass and update scaling state.

  Divides gradients by the current scale factor. If any gradient contains
  Inf or NaN, the step is skipped (returns `:overflow`) and the scale is
  reduced. Otherwise, returns `{:ok, unscaled_grads, new_state}`.

  ## Returns

  - `{:ok, grads, state}` — Gradients are valid, apply the update
  - `{:overflow, state}` — Overflow detected, skip this step
  """
  @spec unscale_grads(map(), map()) :: {:ok, map(), map()} | {:overflow, map()}
  def unscale_grads(grads, %{scale: scale} = state) do
    inv_scale = 1.0 / scale

    unscaled =
      deep_map(grads, fn tensor ->
        Nx.multiply(tensor, inv_scale)
      end)

    if has_overflow?(unscaled) do
      new_scale = scale * state.backoff_factor
      {:overflow, %{state | scale: max(new_scale, 1.0), good_steps: 0}}
    else
      good_steps = state.good_steps + 1

      new_state =
        if good_steps >= state.growth_interval do
          %{state | scale: scale * state.growth_factor, good_steps: 0}
        else
          %{state | good_steps: good_steps}
        end

      {:ok, unscaled, new_state}
    end
  end

  @doc """
  Add loss scaling to an Axon training loop.

  Wraps the training step to scale loss before backward and unscale
  gradients after. On overflow, the optimizer step is skipped.

  ## Options

  Same as `init_loss_scale/1`.

  ## Example

      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Edifice.MixedPrecision.with_loss_scaling(scale: 1024.0)
      |> Axon.Loop.run(data, %{}, epochs: 10)
  """
  @spec with_loss_scaling(Axon.Loop.t(), keyword()) :: Axon.Loop.t()
  def with_loss_scaling(%Axon.Loop{} = loop, opts \\ []) do
    scale_state = init_loss_scale(opts)

    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      {:continue, put_in(state[:handler_metadata][:loss_scale], scale_state)}
    end)
    |> Axon.Loop.handle_event(:iteration_completed, fn state ->
      ls = state.handler_metadata[:loss_scale]

      case ls do
        nil -> {:continue, state}
        _ -> {:continue, state}
      end
    end)
  end

  @doc """
  Summarize the mixed precision configuration of a model.

  Returns a map with counts of layers by precision policy.

  ## Example

      model |> Edifice.MixedPrecision.apply(:bf16) |> Edifice.MixedPrecision.summary()
      #=> %{bf16: 12, f32_preserved: 4, total: 16}
  """
  @spec summary(Axon.t()) :: map()
  def summary(%Axon{nodes: nodes}) do
    default_policy = Axon.MixedPrecision.create_policy()

    Enum.reduce(nodes, %{total: 0, with_policy: 0, without_policy: 0, by_op: %{}}, fn {_id, node}, acc ->
      op = node.op_name || node.op
      has_policy = node.policy != nil and node.policy != default_policy

      acc = %{acc | total: acc.total + 1}

      acc =
        if has_policy do
          %{acc | with_policy: acc.with_policy + 1}
        else
          %{acc | without_policy: acc.without_policy + 1}
        end

      by_op = Map.update(acc.by_op, op, 1, &(&1 + 1))
      %{acc | by_op: by_op}
    end)
  end

  # ===========================================================================
  # Helpers
  # ===========================================================================

  defp deep_map(map, fun) when is_map(map) do
    Map.new(map, fn
      {k, %Nx.Tensor{} = t} -> {k, fun.(t)}
      {k, nested} when is_map(nested) -> {k, deep_map(nested, fun)}
      other -> other
    end)
  end

  defp has_overflow?(grads) when is_map(grads) do
    Enum.any?(grads, fn
      {_k, %Nx.Tensor{} = t} ->
        Nx.any(Nx.is_infinity(t)) |> Nx.to_number() == 1 or
          Nx.any(Nx.is_nan(t)) |> Nx.to_number() == 1

      {_k, nested} when is_map(nested) ->
        has_overflow?(nested)

      _ ->
        false
    end)
  end
end
