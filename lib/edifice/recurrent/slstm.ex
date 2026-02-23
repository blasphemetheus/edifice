defmodule Edifice.Recurrent.SLSTM do
  @moduledoc """
  sLSTM: Scalar LSTM with Exponential Gating.

  Standalone extraction of the sLSTM variant from xLSTM. The sLSTM extends
  traditional LSTM with exponential gating and log-domain stabilization,
  enabling stable training with very large gate values.

  ## Key Innovation: Exponential Gating with Log-Domain Stabilization

  Standard LSTM gates are bounded by sigmoid [0, 1]. sLSTM uses exponential
  gates that can take any positive value, with a stabilization trick to
  prevent overflow:

  ```
  Standard LSTM:  i_t = sigmoid(...)        ∈ [0, 1]
  sLSTM:          i_t = exp(log_i_t - m_t)  ∈ [0, ∞)
  ```

  The stabilizer `m_t = max(log_f_t + m_{t-1}, log_i_t)` keeps values
  numerically tractable while preserving the relative magnitudes.

  ## Equations

  Gate pre-activations (with recurrent connections):
  - `log_i_t = W_i x_t + R_i h_{t-1} + b_i`
  - `log_f_t = W_f x_t + R_f h_{t-1} + b_f`
  - `z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)`
  - `o_t = sigmoid(W_o x_t + R_o h_{t-1} + b_o)`

  Log-domain stabilization:
  - `m_t = max(log_f_t + m_{t-1}, log_i_t)`
  - `i_t' = exp(log_i_t - m_t)`
  - `f_t' = exp(log_f_t + m_{t-1} - m_t)`

  State updates:
  - `c_t = f_t' * c_{t-1} + i_t' * z_t`
  - `n_t = f_t' * n_{t-1} + i_t'`
  - `h_t = o_t * (c_t / max(|n_t|, 1))`

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |         sLSTM Block                  |
  |  LayerNorm -> sLSTM recurrence       |
  |  LayerNorm -> Feedforward            |
  |  Residual connections                |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = SLSTM.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4
      )

  ## References

  - Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024)
  - https://arxiv.org/abs/2405.04517
  """

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default feedforward expansion factor"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 2

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a standalone sLSTM model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of sLSTM blocks (default: 4)
    - `:expand_factor` - Feedforward expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
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
        block = build_slstm_block(acc, Keyword.merge(opts, layer_idx: layer_idx))

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "dropout_#{layer_idx}")
        else
          block
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # sLSTM Block
  # ============================================================================

  defp build_slstm_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "slstm_block_#{layer_idx}"

    # 1. Temporal mixing: LayerNorm -> sLSTM -> Residual
    temporal_normed = Axon.layer_norm(input, name: "#{name}_temporal_norm")
    temporal_out = build_slstm_layer(temporal_normed, hidden_size, "#{name}_slstm")
    after_temporal = Axon.add(input, temporal_out, name: "#{name}_temporal_residual")

    # 2. Feedforward: LayerNorm -> FFN -> Residual
    ff_normed = Axon.layer_norm(after_temporal, name: "#{name}_ff_norm")
    ff_out = build_feedforward(ff_normed, hidden_size, expand_factor, "#{name}_ff")
    Axon.add(after_temporal, ff_out, name: "#{name}_ff_residual")
  end

  # ============================================================================
  # sLSTM Layer (Scalar LSTM with Exponential Gating)
  # ============================================================================

  @doc """
  Build a standalone sLSTM layer for use in custom architectures.

  Returns an Axon node that applies sLSTM recurrence to the input sequence.
  """
  @spec build_slstm_layer(Axon.t(), pos_integer(), String.t()) :: Axon.t()
  def build_slstm_layer(input, hidden_size, name \\ "slstm") do
    # Input projection: W*x + b for all 4 gates
    input_gates = Axon.dense(input, hidden_size * 4, name: "#{name}_input_proj")

    # Recurrent weight: R [hidden_size, 4*hidden_size]
    recurrent_weight =
      Axon.param("#{name}_recurrent_weight", {hidden_size, hidden_size * 4},
        initializer: :glorot_uniform
      )

    Axon.layer(
      &slstm_impl/3,
      [input_gates, recurrent_weight],
      name: "#{name}_recurrence",
      hidden_size: hidden_size,
      op_name: :slstm
    )
  end

  # sLSTM with log-domain stabilization and recurrent connections
  defp slstm_impl(input_gates, recurrent_weight, opts) do
    hidden_size = opts[:hidden_size]
    batch_size = Nx.axis_size(input_gates, 0)
    seq_len = Nx.axis_size(input_gates, 1)

    type = Nx.type(input_gates)
    h_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})
    c_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})
    n_prev = Nx.broadcast(Nx.tensor(1.0, type: type), {batch_size, hidden_size})
    m_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(
        0..(seq_len - 1),
        {{h_prev, c_prev, n_prev, m_prev}, []},
        fn t, {{h_p, c_p, n_p, m_p}, acc} ->
          wx_t = Nx.slice_along_axis(input_gates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
          gates_t = Nx.add(wx_t, rh_t)

          log_i_t = Nx.slice_along_axis(gates_t, 0, hidden_size, axis: 1)
          log_f_t = Nx.slice_along_axis(gates_t, hidden_size, hidden_size, axis: 1)
          z_t = Nx.slice_along_axis(gates_t, hidden_size * 2, hidden_size, axis: 1) |> Nx.tanh()

          o_t =
            Nx.slice_along_axis(gates_t, hidden_size * 3, hidden_size, axis: 1) |> Nx.sigmoid()

          m_t = Nx.max(Nx.add(log_f_t, m_p), log_i_t)
          i_t = Nx.exp(Nx.subtract(log_i_t, m_t))
          f_t = Nx.exp(Nx.subtract(Nx.add(log_f_t, m_p), m_t))

          c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, z_t))
          n_t = Nx.add(Nx.multiply(f_t, n_p), i_t)

          safe_denom = Nx.max(Nx.abs(n_t), 1.0)
          h_t = Nx.multiply(o_t, Nx.divide(c_t, safe_denom))

          {{h_t, c_t, n_t, m_t}, [h_t | acc]}
        end
      )

    h_list
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Feedforward
  # ============================================================================

  defp build_feedforward(input, hidden_size, expand_factor, name) do
    inner_size = hidden_size * expand_factor

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of an sLSTM model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      expand_factor: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
