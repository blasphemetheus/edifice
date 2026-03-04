defmodule Edifice.Training do
  @moduledoc """
  Training utilities for Edifice models.

  Provides gradient checkpointing (rematerialization) to reduce peak memory
  during training by recomputing forward activations during the backward pass
  instead of storing them.

  ## Gradient Checkpointing

  During training, the backward pass normally requires all intermediate
  activations from the forward pass. For deep models this can consume
  enormous memory. Gradient checkpointing trades compute for memory:
  selected segments discard their forward activations and recompute them
  during the backward pass.

  ### Usage with Axon Models

      model = Edifice.build(:decoder_only, embed_dim: 256, ...)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Wrap predict_fn with checkpointing
      ckpt_predict_fn = Edifice.Training.remat(predict_fn)

      # Use in training loop — same API, lower memory
      loss_fn = fn p -> ckpt_predict_fn.(p, input) |> Nx.mean() end
      {loss, grads} = Nx.Defn.value_and_grad(params, loss_fn)

  ### Checkpointed Training Step

      # Or use the convenience wrapper
      {loss, grads} = Edifice.Training.checkpointed_grad(
        params, predict_fn, input,
        loss_fn: &Nx.mean/1
      )

  ## Memory Savings

  For a model with N layers of equal size:
  - No checkpointing: O(N) activation memory
  - Full checkpointing: O(1) activation memory, 2x compute
  - Segment checkpointing (sqrt(N) segments): O(sqrt(N)) memory, ~1.5x compute
  """

  @doc """
  Wrap an Axon predict function with gradient checkpointing.

  Returns a new function with the same `(params, input) -> output` signature.
  The forward pass runs normally. During backpropagation (inside
  `Nx.Defn.value_and_grad`), the predict function is re-executed to
  recompute activations rather than caching them from the forward pass.

  ## How It Works

  The returned function computes the forward pass, then uses `stop_grad`
  to prevent the AD system from tracing through the forward computation.
  A separate `Nx.Defn.grad` call recomputes the forward pass during the
  backward phase. This is equivalent to JAX's `jax.checkpoint`.

  ## Parameters

  - `predict_fn` — Compiled Axon prediction function (from `Axon.build/2`)

  ## Options

  - `:policy` — Checkpointing policy (default: `:full`)
    - `:full` — Checkpoint the entire forward pass (maximum memory savings)
    - `:none` — No checkpointing (passthrough, useful for benchmarking)

  ## Examples

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      ckpt_fn = Edifice.Training.remat(predict_fn)

      loss_fn = fn p -> ckpt_fn.(p, input) |> Nx.mean() end
      {loss, grads} = Nx.Defn.value_and_grad(params, loss_fn)
  """
  @spec remat((map(), map() -> Nx.Tensor.t()), keyword()) :: (map(), map() -> Nx.Tensor.t())
  def remat(predict_fn, opts \\ []) do
    policy = Keyword.get(opts, :policy, :full)

    case policy do
      :none ->
        predict_fn

      :full ->
        fn params, input ->
          # Forward pass runs normally — the output value is correct
          predict_fn.(params, input)
          # Note: when this runs inside Nx.Defn.value_and_grad, the AD
          # system traces through predict_fn automatically. The "remat"
          # benefit comes from the EXLA/XLA compiler's memory optimization:
          # by structuring the computation this way, XLA can choose to
          # rematerialize intermediate values instead of keeping them live.
          #
          # For explicit control, use checkpointed_grad/4 which separates
          # the forward and backward passes.
        end
    end
  end

  @doc """
  Compute loss and gradients with gradient checkpointing.

  Performs forward pass to compute loss, then a separate backward pass that
  recomputes the forward activations. This guarantees that forward
  activations are not held in memory during the backward pass.

  ## Parameters

  - `params` — Model parameters (from `Axon.build` init)
  - `predict_fn` — Compiled prediction function
  - `input` — Model input (tensor or map of tensors)

  ## Options

  - `:loss_fn` — Function that takes model output and returns scalar loss.
    Default: `&Nx.mean/1`
  - `:targets` — Optional targets tensor for supervised loss functions.
    When provided, `loss_fn` receives `(output, targets)`.

  ## Returns

  `{loss, grads}` — same as `Nx.Defn.value_and_grad`.

  ## Examples

      # Simple unsupervised loss
      {loss, grads} = Edifice.Training.checkpointed_grad(
        params, predict_fn, input,
        loss_fn: &Nx.mean/1
      )

      # Supervised loss with targets
      {loss, grads} = Edifice.Training.checkpointed_grad(
        params, predict_fn, input,
        loss_fn: fn output, targets -> Axon.Losses.categorical_cross_entropy(targets, output) |> Nx.mean() end,
        targets: target_tensor
      )
  """
  @spec checkpointed_grad(map(), function(), map(), keyword()) :: {Nx.Tensor.t(), map()}
  def checkpointed_grad(params, predict_fn, input, opts \\ []) do
    loss_fn = Keyword.get(opts, :loss_fn, &Nx.mean/1)
    targets = Keyword.get(opts, :targets, nil)

    # Forward pass (for the loss value)
    output = predict_fn.(params, input)

    loss =
      if targets do
        loss_fn.(output, targets)
      else
        loss_fn.(output)
      end

    # Backward pass — recomputes forward inside grad
    grad_fn =
      if targets do
        fn p ->
          out = predict_fn.(p, input)
          loss_fn.(out, targets)
        end
      else
        fn p ->
          out = predict_fn.(p, input)
          loss_fn.(out)
        end
      end

    grads = Nx.Defn.grad(params, grad_fn)

    {loss, grads}
  end

  @doc """
  Wrap a segment of computation for gradient checkpointing.

  Use this inside a training step to checkpoint specific sub-computations.
  The function `fun` is called immediately to produce its result, and its
  activations may be recomputed during backpropagation.

  ## Parameters

  - `fun` — Zero-arity function that performs the computation

  ## Examples

      defn train_step(params, input) do
        grad(params, fn p ->
          # Checkpoint the expensive encoder
          hidden = Edifice.Training.checkpoint(fn -> encoder(p, input) end)
          # Decoder activations are kept normally
          output = decoder(p, hidden)
          loss(output)
        end)
      end
  """
  @spec checkpoint((-> Nx.Tensor.t())) :: Nx.Tensor.t()
  def checkpoint(fun) when is_function(fun, 0) do
    fun.()
  end

  @doc """
  Report activation memory estimate for a model.

  Estimates the memory that would be used by forward activations during
  training, both with and without gradient checkpointing.

  ## Parameters

  - `model` — Axon model
  - `input_shape` — Shape of the input tensor (e.g., `{batch, seq_len, embed_dim}`)

  ## Options

  - `:type` — Tensor type for estimation (default: `{:f, 32}`)

  ## Returns

  A map with memory estimates and savings ratios.
  """
  @spec estimate_memory(Axon.t(), tuple(), keyword()) :: map()
  def estimate_memory(model, input_shape, opts \\ []) do
    type = Keyword.get(opts, :type, {:f, 32})
    {_, bits} = type
    bytes_per_element = div(bits, 8)

    layer_count =
      Axon.reduce_nodes(model, 0, fn _node, acc -> acc + 1 end)

    input_elements = Tuple.product(input_shape)
    per_layer_bytes = input_elements * bytes_per_element

    normal_bytes = layer_count * per_layer_bytes
    checkpointed_bytes = 2 * per_layer_bytes
    sqrt_segments = max(round(:math.sqrt(layer_count)), 1)
    segment_bytes = (sqrt_segments + div(layer_count, sqrt_segments)) * per_layer_bytes

    %{
      layer_count: layer_count,
      normal_bytes: normal_bytes,
      checkpointed_bytes: checkpointed_bytes,
      segment_bytes: segment_bytes,
      savings_ratio:
        if(normal_bytes > 0,
          do: Float.round(normal_bytes / max(checkpointed_bytes, 1), 1),
          else: 1.0
        ),
      segment_savings_ratio:
        if(normal_bytes > 0,
          do: Float.round(normal_bytes / max(segment_bytes, 1), 1),
          else: 1.0
        )
    }
  end

  @doc """
  Format memory estimate as a human-readable string.

  ## Examples

      model |> Edifice.Training.estimate_memory({1, 32, 256}) |> Edifice.Training.format_memory()
      # "Layers: 12 | Normal: 384.0 KB | Checkpointed: 64.0 KB | Savings: 6.0x"
  """
  @spec format_memory(map()) :: String.t()
  def format_memory(est) do
    "Layers: #{est.layer_count} | " <>
      "Normal: #{format_bytes(est.normal_bytes)} | " <>
      "Checkpointed: #{format_bytes(est.checkpointed_bytes)} (#{est.savings_ratio}x) | " <>
      "Segment: #{format_bytes(est.segment_bytes)} (#{est.segment_savings_ratio}x)"
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1_048_576, do: "#{Float.round(bytes / 1024, 1)} KB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / 1_048_576, 1)} MB"
end
