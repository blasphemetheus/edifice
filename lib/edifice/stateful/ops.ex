defmodule Edifice.Stateful.Ops do
  @moduledoc """
  Shared plain-Nx forward primitives for `Edifice.Stateful` step
  implementations.

  Step code runs eagerly on concrete tensors (one frame at a time), reading
  learned weights out of the trained param map by layer name. Wherever a
  primitive has an Axon equivalent, these helpers delegate to `Axon.Layers`
  so the step math matches the full-sequence forward by construction —
  hand-rolled reimplementations are how step-vs-forward drift starts.
  """

  @doc """
  Dense layer forward using Axon's convention: `x · kernel + bias`.

  `layer_params` is the param submap for one Axon dense layer
  (`%{"kernel" => ..., "bias" => ...}`; bias optional).
  """
  @spec dense(Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def dense(x, %{"kernel" => kernel} = layer_params) do
    case layer_params do
      %{"bias" => bias} -> Axon.Layers.dense(x, kernel, bias)
      _ -> Axon.Layers.dense(x, kernel, Nx.tensor(0.0))
    end
  end

  @doc """
  Layer norm over the last axis using Axon's convention.

  Pass the epsilon the *builder* used — `Axon.layer_norm/2` defaults to
  `1.0e-5`, but e.g. `Edifice.Recurrent` builds its norms with `1.0e-6`.
  """
  @spec layer_norm(Nx.Tensor.t(), map(), float()) :: Nx.Tensor.t()
  def layer_norm(x, %{"gamma" => gamma, "beta" => beta}, epsilon \\ 1.0e-5) do
    Axon.Layers.layer_norm(x, gamma, beta, epsilon: epsilon)
  end

  @doc "SiLU activation (matches `Axon.Activations.silu/1`)."
  @spec silu(Nx.Tensor.t()) :: Nx.Tensor.t()
  def silu(x), do: Axon.Activations.silu(x)

  @doc "Softplus activation (matches `Axon.Activations.softplus/1`)."
  @spec softplus(Nx.Tensor.t()) :: Nx.Tensor.t()
  def softplus(x), do: Axon.Activations.softplus(x)

  @doc """
  One step of a learned depthwise causal 1-D convolution via ring buffer.

  Reproduces `Axon.conv` with `padding: [{k - 1, 0}]` and
  `feature_group_size: channels` (one kernel column per channel) exactly:
  a zero-initialized buffer of the last `k - 1` inputs equals the left
  zero-padding at `t < k`.

    * `frame` - `[batch, channels]` current input
    * `buffer` - `[batch, k - 1, channels]` previous inputs (oldest first)
    * `layer_params` - `%{"kernel" => [k, 1, channels], "bias" => [channels]}`
      (Axon depthwise conv1d layout)

  Returns `{out [batch, channels], new_buffer}`.
  """
  @spec conv1d_step(Nx.Tensor.t(), Nx.Tensor.t(), map()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def conv1d_step(frame, buffer, %{"kernel" => kernel} = layer_params) do
    # window: [batch, k, channels] — buffer ++ current frame
    window = Nx.concatenate([buffer, Nx.new_axis(frame, 1)], axis: 1)

    # kernel [k, 1, channels] -> [k, channels]; out[b, c] = sum_k w[k, c] * window[b, k, c]
    weights = Nx.squeeze(kernel, axes: [1])

    out =
      window
      |> Nx.multiply(Nx.new_axis(weights, 0))
      |> Nx.sum(axes: [1])

    out =
      case layer_params do
        %{"bias" => bias} -> Nx.add(out, bias)
        _ -> out
      end

    # Slide: drop the oldest frame, append the current one
    new_buffer = Nx.slice_along_axis(window, 1, Nx.axis_size(buffer, 1), axis: 1)

    {out, new_buffer}
  end

  @doc """
  Unwrap an `Axon.ModelState` to its plain param data map (identity for
  plain maps).
  """
  @spec unwrap_params(Axon.ModelState.t() | map()) :: map()
  def unwrap_params(%Axon.ModelState{data: data}), do: data
  def unwrap_params(%{} = map), do: map

  @doc """
  Fetch a named layer's params, raising a descriptive error if absent.
  """
  @spec layer_params!(map(), String.t()) :: map()
  def layer_params!(params, name) do
    params = unwrap_params(params)

    case Map.fetch(params, name) do
      {:ok, layer} ->
        layer

      :error ->
        raise ArgumentError,
              "no layer #{inspect(name)} in params (available: " <>
                "#{params |> Map.keys() |> Enum.sort() |> Enum.join(", ")}). " <>
                "Were these params trained with the same build options?"
    end
  end

  @doc """
  Zero state tensor helper: `[batch | dims]` of f32 zeros.
  """
  @spec zeros(pos_integer(), [pos_integer()]) :: Nx.Tensor.t()
  def zeros(batch_size, dims) do
    Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), List.to_tuple([batch_size | dims]))
  end
end
