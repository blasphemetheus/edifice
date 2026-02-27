defmodule Edifice.Blocks.AdaptiveNorm do
  @moduledoc """
  Adaptive Layer Normalization (AdaLN / AdaLN-Zero).

  Conditional normalization where scale and shift parameters are predicted
  from a conditioning signal (e.g., timestep embedding, class label). Used
  in Diffusion Transformers (DiT) and class-conditional generation.

  ## API Levels

  **All-in-one** — `layer/3` bundles norm + projection + modulation + gating:

      output = AdaptiveNorm.layer(input, condition, hidden_size: 256, mode: :adaln_zero)

  **Composable primitives** — `modulate/3` and `gate/3` for DiT-family
  architectures where a single Dense projection produces parameters shared
  across sublayers (attn + MLP):

      # Caller projects condition to 6 params (shift, scale, gate × 2 sublayers)
      params = Axon.dense(condition, hidden_size * 6, name: "adaln_proj")

      # Attention sub-layer
      x_mod = AdaptiveNorm.modulate(LayerNorm(x), params, hidden_size: h, offset: 0)
      attn_out = self_attention(x_mod)
      attn_out = AdaptiveNorm.gate(attn_out, params, hidden_size: h, gate_index: 2)

      # MLP sub-layer
      x_mod2 = AdaptiveNorm.modulate(LayerNorm(x), params, hidden_size: h, offset: 3)
      mlp_out = mlp(x_mod2)
      mlp_out = AdaptiveNorm.gate(mlp_out, params, hidden_size: h, gate_index: 5)

  ## Formula

      AdaLN(x, c) = (1 + scale(c)) * LayerNorm(x) + shift(c)
      AdaLN-Zero(x, c) = gate(c) * ((1 + scale(c)) * LayerNorm(x) + shift(c))

  ## References
  - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
  - https://arxiv.org/abs/2212.09748
  """

  @doc """
  Build an AdaLN / AdaLN-Zero layer (all-in-one).

  Bundles LayerNorm + parameter projection + modulation (+ optional gating)
  into a single layer. For DiT-family architectures that need separate control
  over projection, modulation, and gating, use `modulate/3` and `gate/3`.

  ## Parameters
    - `input` - Input tensor Axon node [batch, ..., hidden_size]
    - `condition` - Conditioning signal Axon node [batch, cond_dim]

  ## Options
    - `:hidden_size` - Feature dimension (required)
    - `:mode` - :adaln or :adaln_zero (default: :adaln_zero)
    - `:name` - Layer name prefix (default: "adaptive_norm")
  """
  @spec layer(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(input, condition, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    mode = Keyword.get(opts, :mode, :adaln_zero)
    name = Keyword.get(opts, :name, "adaptive_norm")

    # Apply LayerNorm first
    normed = Axon.layer_norm(input, name: "#{name}_ln")

    # Branch at graph construction time (not runtime) to avoid passing
    # atom opts to Axon.layer, which breaks Axon's graph traversal
    if mode == :adaln_zero do
      params = Axon.dense(condition, hidden_size * 3, name: "#{name}_params")

      Axon.layer(
        &adaln_zero_impl/3,
        [normed, params],
        name: name,
        hidden_size: hidden_size,
        op_name: :adaptive_norm
      )
    else
      params = Axon.dense(condition, hidden_size * 2, name: "#{name}_params")

      Axon.layer(
        &adaln_impl/3,
        [normed, params],
        name: name,
        hidden_size: hidden_size,
        op_name: :adaptive_norm
      )
    end
  end

  @doc """
  Apply shift-scale modulation from pre-computed parameters.

  Composable primitive for DiT-family architectures where a single Dense
  projection produces parameters for multiple sublayers (e.g., 6 params
  for attn + MLP). The caller handles normalization and parameter projection;
  this function applies `(1 + scale) * x + shift`.

  ## Parameters
    - `input` - Pre-normalized input Axon node [batch, ..., hidden_size]
    - `params` - Pre-computed modulation params [batch, N * hidden_size]

  ## Options
    - `:hidden_size` - Feature dimension (required)
    - `:offset` - Parameter offset in hidden_size units (default: 0).
      shift = params[offset * h : (offset+1) * h],
      scale = params[(offset+1) * h : (offset+2) * h]
    - `:name` - Layer name prefix (default: "adaln_modulate")
  """
  @spec modulate(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def modulate(input, params, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    offset = Keyword.get(opts, :offset, 0)
    name = Keyword.get(opts, :name, "adaln_modulate")

    Axon.layer(
      &modulate_impl/3,
      [input, params],
      name: name,
      hidden_size: hidden_size,
      offset: offset,
      op_name: :adaln_modulate
    )
  end

  @doc """
  Apply gating from pre-computed parameters.

  Extracts a gate vector at the given parameter index and multiplies:
  `gate * x`. Handles broadcasting for both 2D and 3D inputs.

  ## Parameters
    - `input` - Sublayer output Axon node [batch, ..., hidden_size]
    - `params` - Pre-computed modulation params [batch, N * hidden_size]

  ## Options
    - `:hidden_size` - Feature dimension (required)
    - `:gate_index` - Which parameter slot is the gate (default: 2)
    - `:name` - Layer name prefix (default: "adaln_gate")
  """
  @spec gate(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def gate(input, params, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    gate_index = Keyword.get(opts, :gate_index, 2)
    name = Keyword.get(opts, :name, "adaln_gate")

    Axon.layer(
      &gate_impl/3,
      [input, params],
      name: name,
      hidden_size: hidden_size,
      gate_index: gate_index,
      op_name: :adaln_gate
    )
  end

  # Shift-scale modulation: (1 + scale) * x + shift
  defp modulate_impl(x, params, opts) do
    hidden_size = opts[:hidden_size]
    offset = opts[:offset] || 0

    shift = Nx.slice_along_axis(params, offset * hidden_size, hidden_size, axis: -1)
    scale = Nx.slice_along_axis(params, (offset + 1) * hidden_size, hidden_size, axis: -1)

    {shift, scale} =
      if tuple_size(Nx.shape(x)) == 3 do
        {Nx.new_axis(shift, 1), Nx.new_axis(scale, 1)}
      else
        {shift, scale}
      end

    Nx.add(Nx.multiply(Nx.add(1.0, scale), x), shift)
  end

  # Gate extraction and multiplication
  defp gate_impl(x, params, opts) do
    hidden_size = opts[:hidden_size]
    gate_index = opts[:gate_index] || 2

    g = Nx.slice_along_axis(params, gate_index * hidden_size, hidden_size, axis: -1)

    g =
      if tuple_size(Nx.shape(x)) == 3 do
        Nx.new_axis(g, 1)
      else
        g
      end

    Nx.multiply(g, x)
  end

  # AdaLN-Zero: gamma(c) * LN(x) + beta(c), gated by alpha(c)
  defp adaln_zero_impl(normed, params, opts) do
    hidden_size = opts[:hidden_size]

    gamma = Nx.slice_along_axis(params, 0, hidden_size, axis: -1)
    beta = Nx.slice_along_axis(params, hidden_size, hidden_size, axis: -1)
    alpha = Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: -1)

    {gamma, beta, alpha} =
      if tuple_size(Nx.shape(normed)) == 3 do
        {Nx.new_axis(gamma, 1), Nx.new_axis(beta, 1), Nx.new_axis(alpha, 1)}
      else
        {gamma, beta, alpha}
      end

    modulated = Nx.add(Nx.multiply(Nx.add(1.0, gamma), normed), beta)
    Nx.multiply(alpha, modulated)
  end

  # AdaLN: gamma(c) * LN(x) + beta(c)
  defp adaln_impl(normed, params, opts) do
    hidden_size = opts[:hidden_size]

    gamma = Nx.slice_along_axis(params, 0, hidden_size, axis: -1)
    beta = Nx.slice_along_axis(params, hidden_size, hidden_size, axis: -1)

    {gamma, beta} =
      if tuple_size(Nx.shape(normed)) == 3 do
        {Nx.new_axis(gamma, 1), Nx.new_axis(beta, 1)}
      else
        {gamma, beta}
      end

    Nx.add(Nx.multiply(Nx.add(1.0, gamma), normed), beta)
  end
end
