defmodule Edifice.Recurrent.MinLSTM do
  @moduledoc """
  Minimal LSTM (MinLSTM) - A simplified LSTM that is parallel-scannable.

  Implements the MinLSTM from "Were RNNs All We Needed?" (Feng et al., 2024).
  MinLSTM simplifies the LSTM by removing the output gate and hidden state
  nonlinearity, keeping only the forget and input gates with a normalization
  constraint f + i = 1.

  ## Key Innovations

  - **Normalized gates**: f_t + i_t = 1 (forget and input gates sum to 1)
  - **No output gate**: Cell state IS the hidden state
  - **No hidden-to-hidden in gates**: Gates depend only on input
  - **Parallel scannable**: The normalized gating admits parallel prefix scan

  ## Equations

  ```
  f_t = sigmoid(linear_f(x_t))           # Forget gate
  i_t = sigmoid(linear_i(x_t))           # Input gate
  f'_t = f_t / (f_t + i_t)               # Normalized forget
  i'_t = i_t / (f_t + i_t)               # Normalized input
  candidate_t = linear_h(x_t)            # Candidate value
  c_t = f'_t * c_{t-1} + i'_t * candidate_t  # Cell state = hidden state
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +---------------------------+
  |     MinLSTM Layer         |
  |  f = sigmoid(W_f * x)    |
  |  i = sigmoid(W_i * x)    |
  |  f', i' = normalize(f,i) |
  |  c = W_h * x             |
  |  h = f'*h + i'*c         |
  +---------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = MinLSTM.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2410.01201
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.1

  @doc "Normalization epsilon"
  @spec norm_eps() :: float()
  def norm_eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a MinLSTM model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of MinLSTM layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack MinLSTM layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_min_lstm_layer(acc, hidden_size, "min_lstm_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # MinLSTM Layer
  # ============================================================================

  defp build_min_lstm_layer(input, hidden_size, name) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Forget gate projection: f = sigmoid(W_f * x)
    forget_proj = Axon.dense(normed, hidden_size, name: "#{name}_forget")

    # Input gate projection: i = sigmoid(W_i * x)
    input_proj = Axon.dense(normed, hidden_size, name: "#{name}_input")

    # Candidate projection: candidate = W_h * x
    candidate_proj = Axon.dense(normed, hidden_size, name: "#{name}_candidate")

    # Concatenate all projections for the recurrence
    recurrence_input =
      Axon.concatenate([forget_proj, input_proj, candidate_proj],
        axis: 2,
        name: "#{name}_cat"
      )

    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined ->
          min_lstm_scan(combined, hidden_size)
        end,
        name: "#{name}_recurrence"
      )

    # Residual connection
    Axon.add(input, recurrence_output, name: "#{name}_residual")
  end

  defp min_lstm_scan(combined, hidden_size) do
    # combined: [batch, seq_len, hidden_size * 3]
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    # Split forget, input, and candidate
    f_pre = Nx.slice_along_axis(combined, 0, hidden_size, axis: 2)
    i_pre = Nx.slice_along_axis(combined, hidden_size, hidden_size, axis: 2)
    candidate = Nx.slice_along_axis(combined, hidden_size * 2, hidden_size, axis: 2)

    # Compute gates
    f_gate = Nx.sigmoid(f_pre)
    i_gate = Nx.sigmoid(i_pre)

    # Normalize: f' = f/(f+i), i' = i/(f+i) so f' + i' = 1
    gate_sum = Nx.add(f_gate, Nx.add(i_gate, norm_eps()))
    f_norm = Nx.divide(f_gate, gate_sum)
    i_norm = Nx.divide(i_gate, gate_sum)

    # Sequential scan: c_t = f'_t * c_{t-1} + i'_t * candidate_t
    c_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {c_init, []}, fn t, {c_prev, acc} ->
        f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        cand_t = Nx.slice_along_axis(candidate, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
        {c_t, [c_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a MinLSTM model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end
end
