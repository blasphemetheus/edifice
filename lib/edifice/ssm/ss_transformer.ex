defmodule Edifice.SSM.SSTransformer do
  @moduledoc """
  State Space Transformer — parallel SSM + attention with learned gating per block.

  Combines a selective state space model (SSM) path with a multi-head causal
  attention path in every block, fused via a learned sigmoid gate. This allows
  the model to dynamically balance local/recurrent processing (SSM) with
  global attention at each layer.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Per block:
    Pre-norm -> SSM path (selective scan with gating)
             -> Attention path (multi-head causal)
             -> gate * ssm_out + (1-gate) * attn_out
             -> FFN + residual
        |
  Final norm -> last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = SSTransformer.build(
        embed_dim: 256,
        hidden_size: 256,
        state_size: 16,
        num_layers: 6,
        num_heads: 4
      )

  ## References

  - Dao & Gu, "Transformers are SSMs" (2024) — Mamba-2
  - NVIDIA, "Hymba: A Hybrid-head Architecture" (2024) — parallel gating
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_state_size 16
  @default_num_layers 6
  @default_num_heads 4
  @default_head_dim 64
  @default_expand_factor 2
  @default_conv_size 4
  @default_dropout 0.1

  @doc """
  Build a State Space Transformer model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension (default: 16)
    - `:num_layers` - Number of hybrid blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:expand_factor` - SSM expansion factor (default: 2)
    - `:conv_size` - Causal convolution kernel size (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_ss_block(acc, Keyword.merge(opts, layer_idx: layer_idx))

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "sst_dropout_#{layer_idx}")
        else
          block
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_actual - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_ss_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "sst_block_#{layer_idx}"

    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # SSM path
    ssm_out = build_ssm_path(normed, hidden_size, state_size, expand_factor, conv_size, name)

    # Attention path
    attn_out = build_attention_path(normed, hidden_size, num_heads, head_dim, name)

    # Learned gate: sigmoid gate per dimension
    gate =
      normed
      |> Axon.dense(hidden_size, name: "#{name}_gate_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_gate")

    # Fused: gate * ssm + (1 - gate) * attn
    fused =
      Axon.layer(
        fn ssm, attn, g, _opts ->
          Nx.add(
            Nx.multiply(g, ssm),
            Nx.multiply(Nx.subtract(1.0, g), attn)
          )
        end,
        [ssm_out, attn_out, gate],
        name: "#{name}_fusion",
        op_name: :gated_fusion
      )

    after_fusion = Axon.add(input, fused, name: "#{name}_residual")

    # FFN sub-layer
    ffn_normed = Axon.layer_norm(after_fusion, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        expansion_factor: 4,
        name: "#{name}_ffn"
      )

    Axon.add(after_fusion, ffn_out, name: "#{name}_ffn_residual")
  end

  # SSM path: expand -> split x/z -> conv -> silu -> selective scan -> gate -> project out
  defp build_ssm_path(input, hidden_size, state_size, expand_factor, conv_size, name) do
    inner_size = hidden_size * expand_factor

    xz = Axon.dense(input, inner_size * 2, name: "#{name}_ssm_in_proj")

    x_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, 0, inner_size, axis: 2) end,
        name: "#{name}_ssm_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2) end,
        name: "#{name}_ssm_z_split"
      )

    # Causal depthwise conv
    x_conv =
      Axon.conv(x_branch, inner_size,
        kernel_size: {conv_size},
        padding: [{conv_size - 1, 0}],
        feature_group_size: inner_size,
        name: "#{name}_ssm_conv"
      )

    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_ssm_silu")

    # Selective scan
    ssm_out = build_selective_scan(x_activated, inner_size, state_size, "#{name}_ssm")

    # Gate with z branch
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_ssm_gate")
    gated = Axon.multiply(ssm_out, z_activated, name: "#{name}_ssm_gated")

    Axon.dense(gated, hidden_size, name: "#{name}_ssm_out_proj")
  end

  defp build_selective_scan(input, hidden_size, state_size, name) do
    # B and C projections
    b_proj = Axon.dense(input, state_size, name: "#{name}_b_proj")
    c_proj = Axon.dense(input, state_size, name: "#{name}_c_proj")

    # Delta (dt) through low-rank bottleneck + softplus
    dt_proj =
      input
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.nx(fn t -> Nx.log(Nx.add(Nx.exp(t), 1.0)) end, name: "#{name}_dt_softplus")

    Axon.layer(
      &selective_scan_impl/5,
      [input, b_proj, c_proj, dt_proj],
      name: "#{name}_scan",
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :selective_scan
    )
  end

  defp selective_scan_impl(x, b, c, dt, opts) do
    _state_size = opts[:state_size]
    hidden_size = opts[:hidden_size]
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # Simplified gating-based SSM approximation
    # gate = sigmoid(mean(dt))
    gate = Nx.sigmoid(Nx.mean(dt, axes: [2], keep_axes: true))

    # bc_gate = sigmoid(sum(b * c))
    # b: [batch, seq, state_size], c: [batch, seq, state_size]
    bc_product = Nx.multiply(b, c)
    bc_gate = Nx.sigmoid(Nx.sum(bc_product, axes: [2], keep_axes: true))

    # Approximate recurrence with cumulative EMA
    ema_coeff = Nx.mean(gate, axes: [2], keep_axes: true)
    ema_coeff = Nx.broadcast(ema_coeff, {batch_size, seq_len, hidden_size})
    gated_broadcast = Nx.broadcast(Nx.multiply(gate, bc_gate), {batch_size, seq_len, hidden_size})
    output = Nx.multiply(gated_broadcast, x)

    # Cumulative weighted averaging
    cumulative_ema(output, ema_coeff)
  end

  defp cumulative_ema(x, alpha) do
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)
    hidden_size = Nx.axis_size(x, 2)

    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        a_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), Nx.multiply(Nx.subtract(1.0, a_t), x_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Multi-head causal attention path
  defp build_attention_path(input, hidden_size, num_heads, head_dim, name) do
    attn_dim = num_heads * head_dim

    q = Axon.dense(input, attn_dim, name: "#{name}_attn_q")
    k = Axon.dense(input, attn_dim, name: "#{name}_attn_k")
    v = Axon.dense(input, attn_dim, name: "#{name}_attn_v")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q, k, v],
        name: "#{name}_attn_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :causal_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_attn_out_proj")
  end

  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    q_heads =
      q
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k_heads =
      k
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v_heads =
      v
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q_heads, [3], [0, 1], k_heads, [3], [0, 1]) |> Nx.divide(scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols)

    mask =
      mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch_size, num_heads, seq_len, seq_len})

    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
    scores = Nx.select(mask, scores, neg_inf)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply to values
    output = Nx.dot(attn_weights, [3], [0, 1], v_heads, [2], [0, 1])

    # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
    Nx.transpose(output, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      num_layers: 6,
      num_heads: 4,
      head_dim: 64,
      expand_factor: 2,
      conv_size: 4,
      dropout: 0.1
    ]
  end
end
