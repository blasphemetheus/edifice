defmodule Edifice.Training do
  @moduledoc """
  Training utilities for Edifice models.

  The current checkpoint helpers do not implement activation rematerialization.
  `remat/2` and `checkpoint/1` call their functions directly;
  `checkpointed_grad/4` evaluates the loss separately from ordinary gradient
  computation. None of these helpers guarantees lower peak memory.

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

      # Compatibility wrapper; currently the same computation as predict_fn
      ckpt_predict_fn = Edifice.Training.remat(predict_fn)

      # Use in a training loop
      loss_fn = fn p -> ckpt_predict_fn.(p, input) |> Nx.mean() end
      {loss, grads} = Nx.Defn.value_and_grad(params, loss_fn)

  ### Checkpointed Training Step

      # Or use the convenience wrapper
      {loss, grads} = Edifice.Training.checkpointed_grad(
        params, predict_fn, input,
        loss_fn: &Nx.mean/1
      )

  ## Memory Savings

  Actual checkpointing requires compiler support and suitable region boundaries.
  Wrapping an entire model does not guarantee a reduction in peak memory.
  `estimate_memory/3` is a shape-based illustration of hypothetical strategies,
  not a measurement of these helpers or of a compiled training step.
  """

  @doc """
  Wrap an Axon predict function without changing its computation.

  Returns a new function with the same `(params, input) -> output` signature.
  Both policies currently call `predict_fn` directly. This wrapper does not
  insert a recomputation boundary during differentiation.

  ## Parameters

  - `predict_fn` — Compiled Axon prediction function (from `Axon.build/2`)

  ## Options

  - `:policy` — Checkpointing policy (default: `:full`)
    - `:full` — Compatibility wrapper around the prediction function
    - `:none` — Return the prediction function unchanged

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
          predict_fn.(params, input)
        end
    end
  end

  @doc """
  Compute the loss separately from its gradients.

  Evaluates the prediction and loss, then invokes ordinary `Nx.Defn.grad/2`.
  The gradient computation can still retain intermediate activations. This
  helper does not implement rematerialization or guarantee memory savings.
  For a combined value and gradient computation, use `Nx.Defn.value_and_grad/2`.

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
  Call a zero-arity function directly.

  This compatibility helper creates no checkpoint boundary. The function
  `fun` is called immediately to produce its result.

  ## Parameters

  - `fun` — Zero-arity function that performs the computation
  """
  @spec checkpoint((-> Nx.Tensor.t())) :: Nx.Tensor.t()
  def checkpoint(fun) when is_function(fun, 0) do
    fun.()
  end

  @doc """
  Illustrate activation sizes for hypothetical checkpointing strategies.

  Assumes every graph node produces a tensor with `input_shape`. It does not
  account for actual intermediate shapes, compiler fusion, buffer reuse,
  parameters, optimizer state, or native workspaces. The checkpoint estimates
  do not describe the current pass-through helpers and must not be used as
  an out-of-memory preflight check.

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
