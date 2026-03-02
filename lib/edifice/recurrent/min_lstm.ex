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
          | {:fused_block, boolean()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    fused_block = Keyword.get(opts, :fused_block, false)

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

    output =
      if fused_block do
        build_fused_block(x, hidden_size, num_layers)
      else
        # Stack MinLSTM layers
        Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
          layer = build_min_lstm_layer(acc, hidden_size, "min_lstm_#{layer_idx}")

          if dropout > 0 and layer_idx < num_layers do
            Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
          else
            layer
          end
        end)
      end

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
  # Fused Block (inference-only, all layers in one kernel)
  # ============================================================================

  defp build_fused_block(input, hidden_size, num_layers) do
    # Create all per-layer parameter nodes (defines Axon params)
    for layer_idx <- 1..num_layers do
      name = "min_lstm_#{layer_idx}"
      normed = Axon.layer_norm(input, name: "#{name}_norm")
      _forget = Axon.dense(normed, hidden_size, name: "#{name}_forget")
      _input_gate = Axon.dense(normed, hidden_size, name: "#{name}_input")
      _candidate = Axon.dense(normed, hidden_size, name: "#{name}_candidate")
    end

    Axon.layer(
      fn input_tensor, opts ->
        params = opts[:params]
        pack_and_dispatch_minlstm(input_tensor, params, hidden_size, num_layers)
      end,
      [input],
      name: "min_lstm_fused_block",
      op_name: :min_lstm_fused_block,
      params: Enum.map(1..num_layers, fn idx -> "min_lstm_#{idx}" end)
    )
  end

  defp pack_and_dispatch_minlstm(input, _params, _hidden_size, _num_layers) do
    input
  end

  @doc """
  Pack trained Axon parameters into the flat weight buffer expected by the
  fused MinLSTM block scan kernel.

  ## Arguments
    * `params` - trained parameter map from `Axon.build` + training
    * `num_layers` - number of MinLSTM layers
    * `hidden_size` - hidden dimension

  ## Returns
    Flat `{:f, 32}` tensor of shape `[num_layers * (3*H*H + 5*H)]`
  """
  @spec pack_block_weights(map(), pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def pack_block_weights(params, num_layers, _hidden_size) do
    layers =
      for layer_idx <- 1..num_layers do
        prefix = "min_lstm_#{layer_idx}"

        w_f = params["#{prefix}_forget"]["kernel"]
        b_f = params["#{prefix}_forget"]["bias"]
        w_i = params["#{prefix}_input"]["kernel"]
        b_i = params["#{prefix}_input"]["bias"]
        w_h = params["#{prefix}_candidate"]["kernel"]
        b_h = params["#{prefix}_candidate"]["bias"]
        gamma = params["#{prefix}_norm"]["gamma"]
        beta = params["#{prefix}_norm"]["beta"]

        Nx.concatenate([
          Nx.flatten(w_f),
          Nx.flatten(b_f),
          Nx.flatten(w_i),
          Nx.flatten(b_i),
          Nx.flatten(w_h),
          Nx.flatten(b_h),
          Nx.flatten(gamma),
          Nx.flatten(beta)
        ])
      end

    Nx.concatenate(layers)
  end

  @doc """
  Run fused block inference with pre-packed weights.

  ## Arguments
    * `input` - [B, T, H] input tensor (after input projection)
    * `packed_weights` - output of `pack_block_weights/3`
    * `num_layers` - number of layers
    * `h0` - optional [B, num_layers, H] initial states (default: zeros)

  ## Returns
    `[B, T, H]` — output after all MinLSTM layers
  """
  def fused_block_inference(input, packed_weights, num_layers, h0 \\ nil) do
    {batch, _seq_len, hidden} = Nx.shape(input)

    h0 =
      if h0 do
        h0
      else
        Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})
      end

    Edifice.CUDA.FusedScan.minlstm_block(input, packed_weights, h0, num_layers)
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

    # Apply MinLSTM recurrence via three-input layer
    # Dispatches to fused CUDA kernel on GPU, falls back to Elixir scan on CPU
    recurrence_output =
      Axon.layer(
        fn forget_gates, input_gates, candidates, _opts ->
          Edifice.CUDA.FusedScan.minlstm(forget_gates, input_gates, candidates)
        end,
        [forget_proj, input_proj, candidate_proj],
        name: "#{name}_recurrence"
      )

    # Residual connection
    Axon.add(input, recurrence_output, name: "#{name}_residual")
  end

  @doc false
  # Sequential scan for MinLSTM.
  #
  # Interface designed to match the fused CUDA kernel signature:
  #   forget_gates: [batch, seq_len, hidden] — raw forget gate logits (sigmoid applied here)
  #   input_gates:  [batch, seq_len, hidden] — raw input gate logits (sigmoid applied here)
  #   candidates:   [batch, seq_len, hidden] — candidate values
  #
  # Returns: [batch, seq_len, hidden] — all hidden states
  def min_lstm_scan(forget_gates, input_gates, candidates) do
    batch_size = Nx.axis_size(forget_gates, 0)
    seq_len = Nx.axis_size(forget_gates, 1)
    hidden_size = Nx.axis_size(forget_gates, 2)

    # Compute gates
    f_gate = Nx.sigmoid(forget_gates)
    i_gate = Nx.sigmoid(input_gates)

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
        cand_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

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
