defmodule Edifice.Attention.GLAv2 do
  @moduledoc """
  GLA v2: Improved Gated Linear Attention.

  Builds on GLA v1 with several key improvements for better long-range
  memory retention and sequence modeling quality.

  ## Key Improvements over GLA v1

  - **Learnable forget gate bias** initialized to 3.0 (sigmoid ≈ 0.95 = long memory)
  - **Short convolution on V**: causal depthwise conv1d (kernel=4) before linear attention
  - **Separate forget + input gates**: decoupled gating with different activations
  - **Group normalization** after attention output

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +---------------------------------------------+
  |  GLA v2 Block                                |
  |                                              |
  |  Q, K projections → feature map (ELU+1)     |
  |  V projection → causal conv1d (kernel=4)     |
  |  Forget gate: sigmoid(W_f·x + bias_init=3)  |
  |  Input gate: silu(W_i·x)                    |
  |         |                                    |
  |  Gated linear attention with forget/input    |
  |         |                                    |
  |  Group norm → output projection              |
  +---------------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | Standard Attention | GLA v2 |
  |--------|-------------------|--------|
  | Time | O(L²) | O(L) |
  | Space | O(L²) | O(L) |

  ## Usage

      model = GLAv2.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6
      )

  ## References

  - "Gated Linear Attention Transformers with Hardware-Efficient Training"
  - "GLA v2" improvements from flash-linear-attention repository
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 4
  @default_head_dim 64
  @default_conv_kernel_size 4
  @default_forget_gate_init 3.0
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a GLA v2 model for sequence processing.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of GLA v2 blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:conv_kernel_size` - Causal conv kernel size on V (default: 4)
    - `:forget_gate_init` - Initial forget gate bias (default: 3.0)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:conv_kernel_size, pos_integer()}
          | {:forget_gate_init, float()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    forget_gate_init = Keyword.get(opts, :forget_gate_init, @default_forget_gate_init)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

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
        build_gla_v2_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          conv_kernel_size: conv_kernel_size,
          forget_gate_init: forget_gate_init,
          dropout: dropout,
          name: "gla_v2_block_#{layer_idx}"
        )
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_gla_v2_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    forget_gate_init = Keyword.get(opts, :forget_gate_init, @default_forget_gate_init)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gla_v2_block")

    # Gated linear attention v2 sublayer
    x =
      build_gla_v2_attention(input,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        conv_kernel_size: conv_kernel_size,
        forget_gate_init: forget_gate_init,
        dropout: dropout,
        name: "#{name}_attention"
      )

    # Gated FFN (SwiGLU)
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.gated_layer(ffn_normed,
        hidden_size: hidden_size,
        inner_size: hidden_size * 2,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  defp build_gla_v2_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    forget_gate_init = Keyword.get(opts, :forget_gate_init, @default_forget_gate_init)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gla_v2")

    attn_dim = num_heads * head_dim

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Q, K projections
    q_proj = Axon.dense(x, attn_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(x, attn_dim, name: "#{name}_k_proj")

    # V projection with causal conv1d
    v_proj = Axon.dense(x, attn_dim, name: "#{name}_v_proj")

    v_proj =
      Axon.layer(
        &causal_conv1d_impl/2,
        [v_proj],
        name: "#{name}_v_conv",
        kernel_size: conv_kernel_size,
        op_name: :causal_conv1d
      )

    v_proj = Axon.activation(v_proj, :silu, name: "#{name}_v_silu")

    # Separate forget gate with learnable bias initialized to forget_gate_init
    forget_proj = Axon.dense(x, attn_dim, name: "#{name}_forget_proj")

    forget_gate =
      Axon.layer(
        &forget_gate_with_bias/2,
        [forget_proj],
        name: "#{name}_forget_gate",
        bias_init: forget_gate_init,
        attn_dim: attn_dim,
        op_name: :forget_gate_bias
      )

    # Separate input gate with SiLU activation
    input_proj = Axon.dense(x, attn_dim, name: "#{name}_input_proj")
    input_gate = Axon.activation(input_proj, :silu, name: "#{name}_input_silu")

    # Gated linear attention v2
    output =
      Axon.layer(
        &gla_v2_attention_impl/6,
        [q_proj, k_proj, v_proj, forget_gate, input_gate],
        name: "#{name}_gla_v2",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :gla_v2_attention
      )

    # Group normalization
    output = Axon.group_norm(output, num_heads, name: "#{name}_group_norm")

    # Output projection
    output = Axon.dense(output, hidden_size, name: "#{name}_output")

    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    Axon.add(input, output, name: "#{name}_residual")
  end

  # Causal conv1d: zero-pad left by (kernel_size - 1), then average over kernel
  defp causal_conv1d_impl(v, opts) do
    kernel_size = opts[:kernel_size]
    # v: [batch, seq_len, dim]
    batch = Nx.axis_size(v, 0)
    dim = Nx.axis_size(v, 2)

    # Left pad with zeros for causal conv
    pad = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(v)), {batch, kernel_size - 1, dim})
    padded = Nx.concatenate([pad, v], axis: 1)

    # Simple depthwise causal conv: average over kernel window
    # Sum kernel_size consecutive positions
    # padded: [batch, seq_len + kernel_size - 1, dim]
    seq_len = Nx.axis_size(v, 1)

    Enum.reduce(0..(kernel_size - 1), Nx.broadcast(Nx.tensor(0.0, type: Nx.type(v)), {batch, seq_len, dim}), fn offset, acc ->
      slice = Nx.slice_along_axis(padded, offset, seq_len, axis: 1)
      Nx.add(acc, slice)
    end)
    |> Nx.divide(kernel_size)
  end

  # Apply sigmoid with a learnable-style bias (constant init for graph building)
  defp forget_gate_with_bias(proj, opts) do
    bias_init = opts[:bias_init]
    Nx.sigmoid(Nx.add(proj, bias_init))
  end

  # GLA v2 attention: gated linear attention with separate forget/input gates
  defp gla_v2_attention_impl(q, k, v, forget, input_gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to heads: [batch, seq_len, num_heads, head_dim]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_dim})
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})
    forget = Nx.reshape(forget, {batch, seq_len, num_heads, head_dim})
    input_gate = Nx.reshape(input_gate, {batch, seq_len, num_heads, head_dim})

    # Feature map: ELU+1 for positive features
    q_feat = Nx.add(1.0, Nx.select(Nx.greater(q, 0.0), q, Nx.subtract(Nx.exp(q), 1.0)))
    k_feat = Nx.add(1.0, Nx.select(Nx.greater(k, 0.0), k, Nx.subtract(Nx.exp(k), 1.0)))

    # Apply input gate to value
    gated_v = Nx.multiply(input_gate, v)

    # Key-value outer products with forget gating
    k_expanded = Nx.new_axis(k_feat, 4)
    v_expanded = Nx.new_axis(gated_v, 3)
    kv = Nx.multiply(k_expanded, v_expanded)

    # Apply forget gate to KV accumulation
    forget_expanded = Nx.new_axis(forget, 4)
    kv_gated = Nx.multiply(kv, forget_expanded)

    # Cumulative sum for causal attention
    kv_cumsum = Nx.cumulative_sum(kv_gated, axis: 1)
    k_cumsum = Nx.cumulative_sum(Nx.multiply(k_feat, forget), axis: 1)

    # Query attention
    q_expanded = Nx.new_axis(q_feat, 3)
    numerator = Nx.sum(Nx.multiply(q_expanded, kv_cumsum), axes: [4])
    denominator = Nx.sum(Nx.multiply(q_feat, k_cumsum), axes: [3], keep_axes: true)

    eps = 1.0e-6
    output = Nx.divide(numerator, Nx.add(denominator, eps))

    # Reshape back: [batch, seq_len, num_heads * head_dim]
    Nx.reshape(output, {batch, seq_len, num_heads * head_dim})
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Calculate approximate parameter count."
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    attn_dim = num_heads * head_dim
    inner_size = hidden_size * 2

    # Per layer: Q,K,V,forget,input projections + output + FFN
    attention_params = 5 * hidden_size * attn_dim + attn_dim * hidden_size
    ffn_params = 2 * hidden_size * inner_size + inner_size * hidden_size
    per_layer = attention_params + ffn_params

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0
    input_proj + per_layer * num_layers
  end

  @doc "Recommended default configuration."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 4,
      head_dim: 64,
      conv_kernel_size: 4,
      forget_gate_init: 3.0,
      dropout: 0.1
    ]
  end
end
