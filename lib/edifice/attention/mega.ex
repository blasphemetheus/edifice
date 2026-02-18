defmodule Edifice.Attention.Mega do
  @moduledoc """
  Mega: Moving Average Equipped Gated Attention.

  Implements the Mega architecture from "Mega: Moving Average Equipped Gated
  Attention" (Ma et al., ICLR 2023). Mega combines exponential moving averages
  (EMA) for local context with single-head gated attention for global context,
  achieving strong performance with sub-quadratic complexity.

  ## Key Innovation: EMA + Gated Attention

  Each Mega block has three sub-layers:
  1. **EMA sub-layer**: Multi-dimensional exponential moving average captures
     local temporal patterns with learnable decay rates per dimension
  2. **Gated attention**: Single-head attention with sigmoid gating provides
     selective global context aggregation
  3. **FFN**: Standard feed-forward network for feature transformation

  ```
  Mega Block:
    input -> LayerNorm -> EMA -> residual
          -> LayerNorm -> GatedAttn -> residual
          -> LayerNorm -> FFN -> residual
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-----------------------+
  | Input Projection      |
  +-----------------------+
        |
        v
  +-----------------------+
  | Mega Block x N        |
  |  EMA Sub-Layer        |
  |    alpha = sigmoid(a) |
  |    h_t = alpha*h_{t-1}|
  |        + (1-alpha)*x_t|
  |  Gated Attention      |
  |    Q, K, V projections|
  |    gate * attn_output |
  |  FFN                  |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Complexity

  | Operation | Standard Attention | Mega |
  |-----------|-------------------|------|
  | Local     | O(L^2)            | O(L * D_ema) via EMA |
  | Global    | O(L^2 * H)       | O(L^2) single-head |

  ## Usage

      model = Mega.build(
        embed_dim: 287,
        hidden_size: 256,
        ema_dim: 16,
        num_layers: 4
      )

  ## Reference

  - Paper: "Mega: Moving Average Equipped Gated Attention"
  - arXiv: https://arxiv.org/abs/2209.10655
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_ema_dim 16
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Mega model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:ema_dim` - Dimensionality of EMA expansion (default: 16)
    - `:num_layers` - Number of Mega blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:laplace_attention` - Use Laplace attention instead of softmax (default: false)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:ema_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:laplace_attention, boolean()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    ema_dim = Keyword.get(opts, :ema_dim, @default_ema_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    laplace_attention = Keyword.get(opts, :laplace_attention, false)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_mega_block(acc,
          hidden_size: hidden_size,
          ema_dim: ema_dim,
          dropout: dropout,
          laplace_attention: laplace_attention,
          name: "mega_block_#{layer_idx}"
        )
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single Mega block with EMA + gated attention + FFN.
  """
  @spec build_mega_block(Axon.t(), keyword()) :: Axon.t()
  def build_mega_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    ema_dim = Keyword.get(opts, :ema_dim, @default_ema_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    laplace_attention = Keyword.get(opts, :laplace_attention, false)
    name = Keyword.get(opts, :name, "mega_block")

    # 1. EMA sub-layer: project to ema_dim, apply EMA with learnable alpha, project back
    ema_normed = Axon.layer_norm(input, name: "#{name}_ema_norm")

    # Learnable parameters for EMA
    proj_w =
      Axon.param("#{name}_ema_proj_w", {hidden_size, ema_dim}, initializer: :glorot_uniform)

    proj_b = Axon.param("#{name}_ema_proj_b", {ema_dim}, initializer: :zeros)
    alpha_logit = Axon.param("#{name}_ema_alpha", {ema_dim}, initializer: :zeros)
    out_w = Axon.param("#{name}_ema_out_w", {ema_dim, hidden_size}, initializer: :glorot_uniform)
    out_b = Axon.param("#{name}_ema_out_b", {hidden_size}, initializer: :zeros)

    ema_out =
      Axon.layer(
        &ema_impl/7,
        [ema_normed, proj_w, proj_b, alpha_logit, out_w, out_b],
        name: "#{name}_ema",
        op_name: :ema
      )

    ema_out = maybe_dropout(ema_out, dropout, "#{name}_ema_drop")
    x = Axon.add(input, ema_out, name: "#{name}_ema_residual")

    # 2. Gated attention sub-layer
    attn_normed = Axon.layer_norm(x, name: "#{name}_attn_norm")

    attn_out =
      build_gated_attention(attn_normed,
        hidden_size: hidden_size,
        laplace_attention: laplace_attention,
        name: "#{name}_gattn"
      )

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(x, attn_out, name: "#{name}_attn_residual")

    # 3. FFN sub-layer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.gated_layer(ffn_normed,
        hidden_size: hidden_size,
        activation: :silu,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # EMA implementation: multi-dimensional exponential moving average
  # input: [batch, seq_len, hidden_size]
  # Projects to ema_dim, applies EMA with learnable alpha, projects back
  defp ema_impl(input, proj_w, proj_b, alpha_logit, out_w, out_b, _opts) do
    # Project input to ema space: [batch, seq_len, ema_dim]
    projected = Nx.dot(input, [2], proj_w, [0]) |> Nx.add(proj_b)

    # Learnable alpha (decay rate): sigmoid to keep in [0, 1]
    alpha = Nx.sigmoid(alpha_logit)

    # Sequential EMA scan: h_t = alpha * h_{t-1} + (1 - alpha) * x_t
    batch_size = Nx.axis_size(input, 0)
    ema_dim = Nx.axis_size(proj_w, 1)
    seq_len = Nx.axis_size(input, 1)
    one_minus_alpha = Nx.subtract(1.0, alpha)

    # Initialize hidden state
    h = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(input)), {batch_size, ema_dim})

    # Collect EMA outputs for each timestep
    {ema_outputs, _final_h} =
      Enum.reduce(0..(seq_len - 1), {[], h}, fn t, {outputs, h_prev} ->
        x_t = Nx.slice_along_axis(projected, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_new = Nx.add(Nx.multiply(alpha, h_prev), Nx.multiply(one_minus_alpha, x_t))
        {[Nx.new_axis(h_new, 1) | outputs], h_new}
      end)

    # Stack: [batch, seq_len, ema_dim]
    ema_seq = ema_outputs |> Enum.reverse() |> Nx.concatenate(axis: 1)

    # Project back: [batch, seq_len, ema_dim] @ [ema_dim, hidden_size]
    Nx.dot(ema_seq, [2], out_w, [0]) |> Nx.add(out_b)
  end

  # Build gated single-head attention
  defp build_gated_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    laplace_attention = Keyword.get(opts, :laplace_attention, false)
    name = Keyword.get(opts, :name, "gated_attn")

    # Q, K, V projections
    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    # Gate projection
    gate = Axon.dense(input, hidden_size, name: "#{name}_gate")
    gate = Axon.sigmoid(gate, name: "#{name}_gate_sigmoid")

    # Compute attention
    attn_out =
      Axon.layer(
        &gated_attention_impl/5,
        [q, k, v, gate],
        name: "#{name}_compute",
        laplace_attention: laplace_attention,
        op_name: :gated_attention
      )

    # Output projection
    Axon.dense(attn_out, hidden_size, name: "#{name}_out")
  end

  # Gated attention implementation: gate * attention(Q, K, V)
  defp gated_attention_impl(q, k, v, gate, opts) do
    laplace = opts[:laplace_attention] || false
    d_k = Nx.axis_size(k, 2)

    if laplace do
      # Laplace attention: 1 / (1 + |q_i - k_j|)
      q_exp = Nx.new_axis(q, 2)
      k_exp = Nx.new_axis(k, 1)
      diff = Nx.subtract(q_exp, k_exp)
      dist = Nx.sum(Nx.abs(diff), axes: [-1])
      weights = Nx.divide(1.0, Nx.add(1.0, dist))
      # Normalize
      weight_sum = Nx.sum(weights, axes: [-1], keep_axes: true)
      weights = Nx.divide(weights, Nx.add(weight_sum, 1.0e-8))
      attn_out = Nx.dot(weights, [2], [0], v, [1], [0])
      Nx.multiply(gate, attn_out)
    else
      # Standard softmax attention (single-head)
      scale = Nx.sqrt(Nx.tensor(d_k, type: Nx.type(q)))
      scores = Nx.dot(q, [2], [0], k, [2], [0])
      scores = Nx.divide(scores, scale)

      # Causal mask
      seq_len = Nx.axis_size(q, 1)
      rows = Nx.iota({seq_len, seq_len}, axis: 0)
      cols = Nx.iota({seq_len, seq_len}, axis: 1)
      mask = Nx.greater_equal(rows, cols)
      batch_size = Nx.axis_size(q, 0)

      mask =
        mask
        |> Nx.new_axis(0)
        |> Nx.broadcast({batch_size, seq_len, seq_len})

      scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

      weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))

      weights =
        Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8))

      attn_out = Nx.dot(weights, [2], [0], v, [1], [0])
      Nx.multiply(gate, attn_out)
    end
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  @doc """
  Get the output size of a Mega model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      ema_dim: 16,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1,
      laplace_attention: false
    ]
  end
end
