defmodule Edifice.SSM.HyenaV2 do
  @moduledoc """
  Hyena v2: Improved Implicit Long Convolution with Short Conv and Better Decay.

  Builds on the original Hyena Hierarchy with three key improvements for
  better long-range modeling and computational efficiency.

  ## Key Improvements over Hyena v1

  1. **Short depthwise conv before long conv**: Like Mamba, adds a short
     depthwise convolution that captures very local patterns before the
     long-range implicit convolution. This helps with local feature extraction.

  2. **Improved filter parameterization with exponential decay**: The implicit
     filter MLP outputs are multiplied by an exponential decay envelope
     `exp(-alpha * t)` where alpha is learnable. This gives the model an
     inductive bias toward recent tokens while still allowing long-range access.

  3. **Configurable striped pattern**: Built-in support for alternating
     short-conv-only and full-Hyena layers (striped pattern), reducing
     computational cost while maintaining quality.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  | Hyena v2 Block                       |
  |  LayerNorm                           |
  |  Short DW Conv (local features)      |
  |  Split: v, x1, x2                   |
  |  y = v                              |
  |  y = DecayConv(y, filter1) * x1     |
  |  y = DecayConv(y, filter2) * x2     |
  |  OutProj + Residual + FFN           |
  +-------------------------------------+
        |
        v
  [batch, hidden_size]
  ```

  ## Usage

      model = HyenaV2.build(
        embed_dim: 287,
        hidden_size: 256,
        order: 2,
        num_layers: 4
      )

  ## References

  - Poli et al., "Hyena Hierarchy" (ICML 2023)
  - Massaroli et al., "Laughing Hyena Distillery" improvements
  """

  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:order, pos_integer()}
          | {:filter_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}
          | {:striped, boolean()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default gating order"
  @spec default_order() :: pos_integer()
  def default_order, do: 2

  @doc "Default filter MLP hidden size"
  @spec default_filter_size() :: pos_integer()
  def default_filter_size, do: 64

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Hyena v2 model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:order` - Number of gating levels (default: 2)
    - `:filter_size` - Implicit filter MLP hidden size (default: 64)
    - `:num_layers` - Number of Hyena v2 blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:striped` - Use striped pattern (alternate short-conv-only layers) (default: false)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    striped = Keyword.get(opts, :striped, false)

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
        # Striped pattern: odd layers use short-conv only, even use full Hyena
        use_long_conv = if striped, do: rem(layer_idx, 2) == 0, else: true

        build_hyena_v2_block(acc,
          hidden_size: hidden_size,
          order: Keyword.get(opts, :order, default_order()),
          filter_size: Keyword.get(opts, :filter_size, default_filter_size()),
          dropout: dropout,
          seq_len: seq_len,
          use_long_conv: use_long_conv,
          name: "hyena_v2_block_#{layer_idx}"
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

  # ============================================================================
  # Hyena v2 Block
  # ============================================================================

  defp build_hyena_v2_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    order = Keyword.get(opts, :order, default_order())
    filter_size = Keyword.get(opts, :filter_size, default_filter_size())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    seq_len = Keyword.get(opts, :seq_len, 60)
    use_long_conv = Keyword.get(opts, :use_long_conv, true)
    name = Keyword.get(opts, :name, "hyena_v2_block")

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Short depthwise conv (new in v2 — captures local patterns)
    x =
      Axon.conv(x, hidden_size,
        kernel_size: {3},
        padding: [{2, 0}],
        feature_group_size: hidden_size,
        name: "#{name}_short_dw_conv"
      )

    if use_long_conv do
      # Full Hyena block with long convolution + decay
      num_projections = order + 1
      projections = Axon.dense(x, hidden_size * num_projections, name: "#{name}_proj")

      splits =
        for i <- 0..(num_projections - 1) do
          Axon.nx(
            projections,
            fn tensor ->
              Nx.slice_along_axis(tensor, i * hidden_size, hidden_size, axis: 2)
            end,
            name: "#{name}_split_#{i}"
          )
        end

      [v | gates] = splits

      positions =
        Nx.iota({1, seq_len, 1}, axis: 1, type: :f32)
        |> Nx.divide(max(seq_len - 1, 1))

      positions_node = Axon.constant(positions)

      y =
        Enum.with_index(gates)
        |> Enum.reduce(v, fn {gate_i, idx}, acc ->
          # Filter MLP with exponential decay (improvement over v1)
          filter =
            positions_node
            |> Axon.dense(filter_size, name: "#{name}_filter#{idx}_dense1")
            |> Axon.nx(&Nx.sin/1, name: "#{name}_filter#{idx}_sin1")
            |> Axon.dense(filter_size, name: "#{name}_filter#{idx}_dense2")
            |> Axon.nx(&Nx.sin/1, name: "#{name}_filter#{idx}_sin2")
            |> Axon.dense(hidden_size, name: "#{name}_filter#{idx}_dense3")

          # Apply exponential decay envelope (new in v2)
          decay_alpha =
            Axon.param("#{name}_decay_alpha_#{idx}", {1, 1, hidden_size},
              initializer: fn shape, _opts -> Nx.broadcast(Nx.tensor(0.1), shape) end
            )

          filter =
            Axon.layer(
              fn f, alpha, _opts ->
                seq = Nx.axis_size(f, 1)
                t = Nx.iota({1, seq, 1}, axis: 1, type: Nx.type(f))
                decay = Nx.exp(Nx.negate(Nx.multiply(Nx.abs(alpha), t)))
                Nx.multiply(f, decay)
              end,
              [filter, decay_alpha],
              name: "#{name}_decay_#{idx}",
              op_name: :decay_filter
            )

          # Causal long convolution
          conv_out =
            Axon.layer(
              &causal_long_conv_impl/3,
              [acc, filter],
              name: "#{name}_long_conv_#{idx}",
              hidden_size: hidden_size,
              op_name: :causal_long_conv
            )

          Axon.multiply(conv_out, gate_i, name: "#{name}_gate_#{idx}")
        end)

      out = Axon.dense(y, hidden_size, name: "#{name}_out_proj")

      out =
        if dropout > 0 do
          Axon.dropout(out, rate: dropout, name: "#{name}_drop")
        else
          out
        end

      x_res = Axon.add(input, out, name: "#{name}_residual")

      # FFN sub-layer
      ffn_normed = Axon.layer_norm(x_res, name: "#{name}_ffn_norm")

      ffn_out =
        FFN.gated_layer(ffn_normed,
          hidden_size: hidden_size,
          inner_size: hidden_size * 4,
          activation: :silu,
          dropout: dropout,
          name: "#{name}_ffn"
        )

      Axon.add(x_res, ffn_out, name: "#{name}_ffn_residual")
    else
      # Short-conv only layer (striped pattern)
      out = Axon.dense(x, hidden_size, name: "#{name}_short_proj")

      out =
        if dropout > 0 do
          Axon.dropout(out, rate: dropout, name: "#{name}_short_drop")
        else
          out
        end

      Axon.add(input, out, name: "#{name}_short_residual")
    end
  end

  # Causal long convolution (reused from Hyena v1 pattern)
  defp causal_long_conv_impl(signal, filter, _opts) do
    hidden_size = Nx.axis_size(signal, 2)
    seq_len = Nx.axis_size(signal, 1)

    signal_t = Nx.transpose(signal, axes: [0, 2, 1])
    h = Nx.squeeze(filter, axes: [0])
    h_rev = Nx.reverse(h, axes: [0])

    kernel =
      h_rev
      |> Nx.transpose(axes: [1, 0])
      |> Nx.reshape({hidden_size, 1, seq_len})

    result =
      Nx.conv(signal_t, kernel,
        padding: [{seq_len - 1, 0}],
        feature_group_size: hidden_size
      )

    Nx.transpose(result, axes: [0, 2, 1])
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a Hyena v2 model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      order: 2,
      filter_size: 64,
      num_layers: 4,
      window_size: 60,
      striped: false,
      dropout: 0.1
    ]
  end
end
