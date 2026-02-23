defmodule Edifice.Attention.HGRNv2 do
  @moduledoc """
  HGRN v2: Multi-Resolution Hierarchical Gating with Outer Product State.

  Builds on HGRN v1 with richer state representation and data-dependent
  initialization for improved sequence modeling.

  ## Key Improvements over HGRN v1

  - **Explicit resolution schedule**: lower layers use per-element gating,
    upper layers use per-group gating (computed from layer index)
  - **Outer product state expansion**: value is split into `v_a, v_b`;
    state = `f·h + i·outer(v_a, v_b)` for a rank-1 update that creates
    a richer hidden state without full matrix multiplication
  - **Data-dependent forget init**: a small dense layer computes the
    forget gate bias from input content, adapting initialization per token

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +---------------------------------------------------+
  |  HGRN v2 Block                                     |
  |                                                     |
  |  +- Outer Product State Expansion ---------------+ |
  |  |  v_a, v_b = split(value)                      | |
  |  |  state_update = outer(v_a, v_b)               | |
  |  +-----------------------------------------------+ |
  |                                                     |
  |  +- Hierarchical Gating (resolution schedule) ---+ |
  |  |  forget = sigmoid(W_f·x + data_bias(x))       | |
  |  |  input = sigmoid(W_i·x)                       | |
  |  |  h = f·h + i·state_update                     | |
  |  +-----------------------------------------------+ |
  |                                                     |
  |  +- State Contraction ----------------------------+ |
  |  |  output = Linear(flatten(h), D)                | |
  |  +-----------------------------------------------+ |
  +---------------------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | Value |
  |--------|-------|
  | Training Time | O(L) |
  | Training Space | O(L) |
  | Inference Time | O(1) per step |
  | Inference Space | O(1) |

  ## Usage

      model = HGRNv2.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        state_expansion: 2,
        outer_product_dim: 16
      )

  ## References

  - "HGRN2: Gated Linear RNNs with State Expansion" (arXiv:2404.07904)
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_num_layers 6
  @default_state_expansion 2
  @default_outer_product_dim 16
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an HGRN v2 model for sequence processing.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:num_layers` - Number of HGRN v2 blocks (default: 6)
    - `:state_expansion` - State expansion factor E (default: 2)
    - `:outer_product_dim` - Dimension for outer product split (default: 16)
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
          | {:state_expansion, pos_integer()}
          | {:outer_product_dim, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    outer_product_dim = Keyword.get(opts, :outer_product_dim, @default_outer_product_dim)
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
        build_hgrn_v2_block(acc,
          hidden_size: hidden_size,
          state_expansion: state_expansion,
          outer_product_dim: outer_product_dim,
          dropout: dropout,
          layer_idx: layer_idx,
          num_layers: num_layers,
          name: "hgrn_v2_block_#{layer_idx}"
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

  defp build_hgrn_v2_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "hgrn_v2_block")

    # HGRN v2 recurrence layer
    x =
      build_hgrn_v2_layer(input,
        hidden_size: hidden_size,
        state_expansion: Keyword.get(opts, :state_expansion, @default_state_expansion),
        outer_product_dim: Keyword.get(opts, :outer_product_dim, @default_outer_product_dim),
        dropout: dropout,
        layer_idx: Keyword.get(opts, :layer_idx, 1),
        num_layers: Keyword.get(opts, :num_layers, @default_num_layers),
        name: "#{name}_rnn"
      )

    # FFN with residual
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        activation: :silu,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  defp build_hgrn_v2_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    outer_product_dim = Keyword.get(opts, :outer_product_dim, @default_outer_product_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    _num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    name = Keyword.get(opts, :name, "hgrn_v2")

    expanded_size = hidden_size * state_expansion

    # Resolution schedule: finer at lower layers, coarser at higher
    gate_dim = max(1, Bitwise.bsr(expanded_size, layer_idx - 1))
    group_size = div(expanded_size, gate_dim)

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Data-dependent forget bias: small dense layer computes per-token bias
    forget_bias = Axon.dense(x, gate_dim, name: "#{name}_forget_bias_proj")

    # Forget gate with data-dependent bias
    forget_proj = Axon.dense(x, gate_dim, name: "#{name}_forget_proj")

    forget_gate =
      Axon.layer(
        &data_dependent_forget/3,
        [forget_proj, forget_bias],
        name: "#{name}_forget_gate",
        op_name: :data_dependent_forget
      )

    # Input gate
    input_proj = Axon.dense(x, gate_dim, name: "#{name}_input_proj")
    input_gate = Axon.activation(input_proj, :sigmoid, name: "#{name}_input_sigmoid")

    # Outer product value: project to 2 * outer_product_dim, split into v_a, v_b
    # Then outer product gives expanded_size = outer_product_dim * outer_product_dim
    # But we need expanded_size to match, so project to expanded_size and reshape
    v_a_proj = Axon.dense(x, outer_product_dim, name: "#{name}_v_a_proj")
    v_a_proj = Axon.activation(v_a_proj, :silu, name: "#{name}_v_a_silu")

    v_b_proj = Axon.dense(x, div(expanded_size, outer_product_dim), name: "#{name}_v_b_proj")
    v_b_proj = Axon.activation(v_b_proj, :silu, name: "#{name}_v_b_silu")

    # Outer product and recurrence
    output =
      Axon.layer(
        &hgrn_v2_recurrence/5,
        [forget_gate, input_gate, v_a_proj, v_b_proj],
        name: "#{name}_recurrence",
        expanded_size: expanded_size,
        outer_product_dim: outer_product_dim,
        gate_dim: gate_dim,
        group_size: group_size,
        op_name: :hgrn_v2_recurrence
      )

    # Contract back to hidden_size
    output = Axon.dense(output, hidden_size, name: "#{name}_contract")

    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    Axon.add(input, output, name: "#{name}_residual")
  end

  # Data-dependent forget: sigmoid(proj + bias_from_content)
  defp data_dependent_forget(proj, bias, _opts) do
    Nx.sigmoid(Nx.add(proj, bias))
  end

  # HGRN v2 recurrence with outer product state expansion
  defp hgrn_v2_recurrence(forget, input_gate, v_a, v_b, opts) do
    group_size = opts[:group_size]
    expanded_size = opts[:expanded_size]
    outer_product_dim = opts[:outer_product_dim]
    v_b_dim = div(expanded_size, outer_product_dim)

    batch = Nx.axis_size(v_a, 0)
    seq_len = Nx.axis_size(v_a, 1)

    # Compute outer product: v_a [batch, seq, op_dim] x v_b [batch, seq, v_b_dim]
    # -> [batch, seq, op_dim * v_b_dim] = [batch, seq, expanded_size]
    v_a_exp = Nx.new_axis(v_a, 3)
    v_b_exp = Nx.new_axis(v_b, 2)
    outer = Nx.multiply(v_a_exp, v_b_exp)
    value = Nx.reshape(outer, {batch, seq_len, outer_product_dim * v_b_dim})

    # Broadcast coarse gates to full expanded_size
    forget_full =
      if group_size > 1 do
        gate_dim = Nx.axis_size(forget, 2)

        forget
        |> Nx.new_axis(3)
        |> Nx.broadcast({batch, seq_len, gate_dim, group_size})
        |> Nx.reshape({batch, seq_len, gate_dim * group_size})
      else
        forget
      end

    input_gate_full =
      if group_size > 1 do
        gate_dim = Nx.axis_size(input_gate, 2)

        input_gate
        |> Nx.new_axis(3)
        |> Nx.broadcast({batch, seq_len, gate_dim, group_size})
        |> Nx.reshape({batch, seq_len, gate_dim * group_size})
      else
        input_gate
      end

    # Gated recurrence with parallel scan (log-cumsum-exp trick)
    gated_value = Nx.multiply(input_gate_full, value)

    log_forget = Nx.log(Nx.add(forget_full, 1.0e-10))
    log_forget_cumsum = Nx.cumulative_sum(log_forget, axis: 1)
    forget_cumprod = Nx.exp(log_forget_cumsum)

    eps = 1.0e-10
    normalized_value = Nx.divide(gated_value, Nx.add(forget_cumprod, eps))
    value_cumsum = Nx.cumulative_sum(normalized_value, axis: 1)
    Nx.multiply(forget_cumprod, value_cumsum)
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
    state_expansion = Keyword.get(opts, :state_expansion, @default_state_expansion)
    outer_product_dim = Keyword.get(opts, :outer_product_dim, @default_outer_product_dim)

    expanded_size = hidden_size * state_expansion
    v_b_dim = div(expanded_size, outer_product_dim)
    inner_size = hidden_size * 4

    # Per layer: forget_bias + forget + input + v_a + v_b projections + contract + FFN
    hgrn_params =
      hidden_size * expanded_size +
        hidden_size * expanded_size +
        hidden_size * expanded_size +
        hidden_size * outer_product_dim +
        hidden_size * v_b_dim +
        expanded_size * hidden_size

    ffn_params = hidden_size * inner_size + inner_size * hidden_size
    per_layer = hgrn_params + ffn_params

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0
    input_proj + per_layer * num_layers
  end

  @doc "Recommended default configuration."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      state_expansion: 2,
      outer_product_dim: 16,
      dropout: 0.1
    ]
  end
end
