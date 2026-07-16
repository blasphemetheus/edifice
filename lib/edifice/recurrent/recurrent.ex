defmodule Edifice.Recurrent do
  @moduledoc """
  Recurrent neural network layers for temporal sequence processing.

  Provides LSTM and GRU architectures for learning temporal dependencies
  in sequential data - essential for understanding:
  - Multi-step action sequences
  - Temporal patterns and trends
  - Long-range dependencies
  - Reactive decision sequences

  ## Architecture

  The recurrent backbone processes sequences of embedded states:

  ```
  Frame Sequence [batch, seq_len, embed_dim]
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │ ←─ hidden state (h, c for LSTM)
  │  Layer 1    │
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │  (optional stacked layers)
  │  Layer 2    │
  └─────────────┘
        │
        ▼
  Hidden Output [batch, hidden_size]
  ```

  ## Hidden State Management

  For real-time inference, hidden states must be carried between frames:

      # Initialize hidden state
      hidden = Recurrent.initial_hidden(model, batch_size)

      # Process frame, get new hidden
      {output, new_hidden} = Recurrent.forward_with_state(model, params, frame, hidden)

      # Use new_hidden for next frame
      ...

  ## Usage

      # Build recurrent backbone
      model = Recurrent.build(
        embed_dim: 1024,
        hidden_size: 256,
        num_layers: 2,
        cell_type: :lstm,
        dropout: 0.1
      )

      # Use as backbone in policy network
      input = Axon.input("state_sequence", shape: {nil, nil, 1024})
      backbone = Recurrent.build_backbone(input, hidden_size: 256, cell_type: :gru)
      policy_head = build_policy_head(backbone)

  """

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 1
  @default_cell_type :lstm
  @default_dropout 0.0
  # nil = full BPTT, integer = truncate to last N steps
  @default_truncate_bptt nil
  # @default_bidirectional false  # Future: bidirectional support

  @type cell_type :: :lstm | :gru
  @type hidden_state :: Nx.Tensor.t() | {Nx.Tensor.t(), Nx.Tensor.t()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a recurrent model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Size of recurrent hidden state (default: 256)
    - `:num_layers` - Number of stacked recurrent layers (default: 1)
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:dropout` - Dropout rate between layers (default: 0.0)
    - `:bidirectional` - Use bidirectional processing (default: false)
    - `:return_sequences` - Return all timesteps or just last (default: false)

  ## Returns
    An Axon model that processes sequences and outputs hidden representations.

  ## Examples

      iex> model = Edifice.Recurrent.build(embed_dim: 32, hidden_size: 16, cell_type: :lstm)
      iex> %Axon{} = model
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:cell_type, cell_type()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:fused_block, boolean()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:return_sequences, boolean()}
          | {:seq_len, pos_integer()}
          | {:truncate_bptt, pos_integer() | nil}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    return_sequences = Keyword.get(opts, :return_sequences, false)
    truncate_bptt = Keyword.get(opts, :truncate_bptt, @default_truncate_bptt)
    fused_block = Keyword.get(opts, :fused_block, false)
    # Use concrete seq_len for efficient JIT compilation (same fix as attention)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim] - use concrete seq_len when available
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    build_backbone(input,
      hidden_size: hidden_size,
      num_layers: num_layers,
      cell_type: cell_type,
      dropout: dropout,
      return_sequences: return_sequences,
      truncate_bptt: truncate_bptt,
      fused_block: fused_block
    )
  end

  @doc """
  Build the recurrent backbone from an existing input layer.

  Useful for integrating into larger networks (policy, value).

  ## Options
    - `:hidden_size` - Size of recurrent hidden state (default: 256)
    - `:num_layers` - Number of stacked recurrent layers (default: 1)
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:dropout` - Dropout rate between layers (default: 0.0)
    - `:return_sequences` - Return all timesteps or just last (default: false)
    - `:truncate_bptt` - Truncate gradients to last N steps (default: nil = full BPTT)
                         Set to e.g. 15-20 for 2-3x faster training with some accuracy loss
    - `:input_layer_norm` - Apply layer norm to input for stability (default: true)
    - `:use_layer_norm` - Apply layer norm after each RNN layer (default: true)
  """
  @spec build_backbone(Axon.t(), keyword()) :: Axon.t()
  def build_backbone(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    return_sequences = Keyword.get(opts, :return_sequences, false)
    truncate_bptt = Keyword.get(opts, :truncate_bptt, @default_truncate_bptt)
    input_layer_norm = Keyword.get(opts, :input_layer_norm, true)
    use_layer_norm = Keyword.get(opts, :use_layer_norm, true)
    fused_block = Keyword.get(opts, :fused_block, false)

    if fused_block and cell_type in [:lstm, :gru] do
      # Fused block path: kernel handles prenorm internally via packed weights.
      # Input is passed raw (no input_ln) — the block kernel applies layer 1's
      # prenorm using the packed input_ln weights.
      build_fused_block(input, hidden_size, num_layers, cell_type, return_sequences)
    else
      # Standard sequential path
      # Normalize input embeddings for stable gradient flow
      # This prevents large activation magnitudes from compounding through time
      normalized_input =
        if input_layer_norm do
          Axon.layer_norm(input, name: "input_ln", epsilon: 1.0e-6)
        else
          input
        end

      # Apply gradient truncation if configured
      # This stops gradients from flowing back beyond the last N timesteps
      processed_input =
        if truncate_bptt do
          apply_gradient_truncation(normalized_input, truncate_bptt)
        else
          normalized_input
        end

      # Build stacked recurrent layers
      output =
        Enum.reduce(1..num_layers, processed_input, fn layer_idx, acc ->
          is_last_layer = layer_idx == num_layers

          # Only return sequences for intermediate layers, or if explicitly requested
          layer_return_seq = not is_last_layer or return_sequences

          layer =
            build_recurrent_layer(acc, hidden_size, cell_type,
              name: "#{cell_type}_#{layer_idx}",
              return_sequences: layer_return_seq,
              use_layer_norm: use_layer_norm
            )

          # Add dropout between layers (not after last)
          if dropout > 0 and not is_last_layer do
            Axon.dropout(layer, rate: dropout, name: "recurrent_dropout_#{layer_idx}")
          else
            layer
          end
        end)

      output
    end
  end

  @doc """
  Build a single recurrent layer (LSTM or GRU).

  ## Options
    - `:name` - Layer name prefix
    - `:return_sequences` - Whether to return all timesteps or just the last (default: true)
    - `:use_layer_norm` - Add layer normalization after RNN for stability (default: true)
    - `:recurrent_initializer` - Initializer for recurrent weights (default: :glorot_uniform)

  ## Stability Notes

  RNNs are prone to gradient explosion/vanishing. This implementation uses:
  1. **Orthogonal initialization** for recurrent weights (preserves gradient magnitude)
  2. **Layer normalization** after each RNN layer (stabilizes hidden state magnitudes)
  3. Standard glorot for input weights (via Axon defaults)

  If training still diverges, reduce learning rate to 1e-5 and use gradient clipping 0.5.
  """
  @spec build_recurrent_layer(Axon.t(), non_neg_integer(), cell_type(), keyword()) :: Axon.t()
  def build_recurrent_layer(input, hidden_size, cell_type, opts \\ []) do
    name = Keyword.get(opts, :name, "recurrent")
    return_sequences = Keyword.get(opts, :return_sequences, true)
    use_layer_norm = Keyword.get(opts, :use_layer_norm, true)
    recurrent_init = Keyword.get(opts, :recurrent_initializer, :glorot_uniform)

    output_seq =
      if fused_rnn_available?(cell_type) do
        build_fused_recurrent(input, hidden_size, cell_type,
          name: name,
          recurrent_initializer: recurrent_init
        )
      else
        build_axon_recurrent(input, hidden_size, cell_type,
          name: name,
          recurrent_initializer: recurrent_init
        )
      end

    # Apply layer normalization for stability (normalizes across hidden dimension)
    # This helps prevent activation explosion in long sequences
    normalized_output =
      if use_layer_norm do
        Axon.layer_norm(output_seq, name: "#{name}_ln", epsilon: 1.0e-6)
      else
        output_seq
      end

    if return_sequences do
      normalized_output
    else
      # Take the last timestep: [batch, seq_len, hidden] -> [batch, hidden]
      Axon.nx(
        normalized_output,
        fn tensor ->
          seq_len = Nx.axis_size(tensor, 1)

          Nx.slice_along_axis(tensor, seq_len - 1, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "#{name}_last"
      )
    end
  end

  @doc """
  Build a raw RNN layer (LSTM or GRU) with fused CUDA kernel when available.

  Returns just the output sequence `[batch, seq_len, hidden_size]` — no LayerNorm,
  no last-timestep extraction. Use this when you need to integrate LSTM/GRU into
  a custom architecture (residual blocks, hybrid models, etc.).

  Falls back to `Axon.lstm`/`Axon.gru` when CUDA kernels are unavailable.

  ## Options
    - `:name` - Layer name prefix (default: "recurrent")
    - `:recurrent_initializer` - Initializer for recurrent weights (default: :glorot_uniform)
  """
  @spec build_raw_rnn(Axon.t(), non_neg_integer(), cell_type(), keyword()) :: Axon.t()
  def build_raw_rnn(input, hidden_size, cell_type, opts \\ []) do
    if fused_rnn_available?(cell_type) do
      build_fused_recurrent(input, hidden_size, cell_type, opts)
    else
      build_axon_recurrent(input, hidden_size, cell_type, opts)
    end
  end

  # Fused CUDA path: pre-compute W@x + bias, pass recurrent weight R to kernel
  defp build_fused_recurrent(input, hidden_size, cell_type, opts) do
    name = Keyword.get(opts, :name, "recurrent")
    recurrent_init = Keyword.get(opts, :recurrent_initializer, :glorot_uniform)

    num_gates = case cell_type do
      :lstm -> 4
      :gru -> 3
    end

    gate_size = num_gates * hidden_size

    # Input projection: W@x + bias → [B, T, G*H]
    wx = Axon.dense(input, gate_size,
      name: "#{name}_input_proj",
      kernel_initializer: :glorot_uniform,
      use_bias: true
    )

    # Recurrent weight R: [H, G*H] — passed directly to kernel
    r_param = Axon.param("#{name}_recurrent_kernel", {hidden_size, gate_size},
      initializer: recurrent_init
    )

    # Fused scan layer: kernel computes R@h internally
    # Axon.layer always appends an opts map as the last argument
    scan_fn = case cell_type do
      :lstm -> fn wx, r, _opts -> Edifice.CUDA.FusedScan.lstm_scan(wx, r) end
      :gru -> fn wx, r, _opts -> Edifice.CUDA.FusedScan.gru_scan(wx, r) end
    end

    Axon.layer(scan_fn, [wx, r_param],
      name: "#{name}_fused_scan",
      op_name: :"fused_#{cell_type}_scan"
    )
  end

  # Standard Axon path: uses Axon.lstm/gru with while_loop unrolling
  defp build_axon_recurrent(input, hidden_size, cell_type, opts) do
    name = Keyword.get(opts, :name, "recurrent")
    recurrent_init = Keyword.get(opts, :recurrent_initializer, :glorot_uniform)

    recurrent_opts = [
      name: name,
      recurrent_initializer: recurrent_init,
      use_bias: true
    ]

    {output_seq, _hidden} =
      case cell_type do
        :lstm -> Axon.lstm(input, hidden_size, recurrent_opts)
        :gru -> Axon.gru(input, hidden_size, recurrent_opts)
      end

    output_seq
  end

  @doc false
  def fused_rnn_available?(cell_type) when cell_type in [:lstm, :gru] do
    Edifice.CUDA.FusedScan.custom_call_available?() or nif_available?()
  end

  def fused_rnn_available?(_), do: false

  defp nif_available? do
    Code.ensure_loaded?(Edifice.CUDA.NIF) and
      function_exported?(Edifice.CUDA.NIF, :fused_lstm_scan, 7)
  end

  # ============================================================================
  # Fused Block Scan (multi-layer kernel dispatch)
  # ============================================================================

  # Fused block path: passthrough on CPU, actual dispatch via pack_block_weights + fused_block_inference
  defp build_fused_block(input, hidden_size, num_layers, cell_type, return_sequences) do
    num_gates = case cell_type do
      :lstm -> 4
      :gru -> 3
    end

    # Create per-layer parameter nodes (disconnected — ensures Axon tracks param shapes)
    for layer_idx <- 1..num_layers do
      name = "#{cell_type}_#{layer_idx}"
      normed = Axon.layer_norm(input, name: "#{name}_prenorm", epsilon: 1.0e-6)
      _wx = Axon.dense(normed, num_gates * hidden_size, name: "#{name}_input_proj")
    end

    # Passthrough layer — fused execution uses pack_block_weights/4 + fused_*_block_inference
    block_output = Axon.layer(
      fn input_tensor, _opts -> input_tensor end,
      [input],
      name: "#{cell_type}_fused_block",
      op_name: :"#{cell_type}_fused_block"
    )

    # Final post-norm (corresponds to the last layer's layer norm in sequential path)
    normed = Axon.layer_norm(block_output, name: "#{cell_type}_#{num_layers}_ln", epsilon: 1.0e-6)

    if return_sequences do
      normed
    else
      Axon.nx(normed, fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end, name: "#{cell_type}_fused_last")
    end
  end

  @doc """
  Pack trained Axon parameters into the flat weight buffer expected by the
  fused LSTM block scan kernel.

  Maps the sequential path's parameter names to the block kernel layout:
  layer 1's prenorm uses `input_ln`, layer k>1 uses `{cell_type}_{k-1}_ln`.

  ## Arguments
    * `params` - trained parameter map from `Axon.build` + training
    * `num_layers` - number of stacked recurrent layers
    * `hidden_size` - hidden dimension
    * `cell_type` - `:lstm` or `:gru`

  ## Returns
    Flat `{:f, 32}` tensor. LSTM stride: `8*H*H + 6*H` per layer.
    GRU stride: `6*H*H + 5*H` per layer.
  """
  @spec pack_block_weights(map(), pos_integer(), pos_integer(), cell_type()) :: Nx.Tensor.t()
  def pack_block_weights(params, num_layers, _hidden_size, :lstm) do
    layers =
      for layer_idx <- 1..num_layers do
        w_x = params["lstm_#{layer_idx}_input_proj"]["kernel"]
        b_x = params["lstm_#{layer_idx}_input_proj"]["bias"]
        r_w = params["lstm_#{layer_idx}_fused_scan"]["lstm_#{layer_idx}_recurrent_kernel"]

        # Prenorm mapping: block layer 1 uses input_ln, layer k>1 uses lstm_{k-1}_ln
        {gamma, beta} =
          if layer_idx == 1 do
            {params["input_ln"]["gamma"], params["input_ln"]["beta"]}
          else
            ln = "lstm_#{layer_idx - 1}_ln"
            {params[ln]["gamma"], params[ln]["beta"]}
          end

        Nx.concatenate([
          Nx.flatten(w_x),
          Nx.flatten(b_x),
          Nx.flatten(r_w),
          Nx.flatten(gamma),
          Nx.flatten(beta)
        ])
      end

    Nx.concatenate(layers)
  end

  def pack_block_weights(params, num_layers, _hidden_size, :gru) do
    layers =
      for layer_idx <- 1..num_layers do
        w_x = params["gru_#{layer_idx}_input_proj"]["kernel"]
        b_x = params["gru_#{layer_idx}_input_proj"]["bias"]
        r_w = params["gru_#{layer_idx}_fused_scan"]["gru_#{layer_idx}_recurrent_kernel"]

        # Prenorm mapping: block layer 1 uses input_ln, layer k>1 uses gru_{k-1}_ln
        {gamma, beta} =
          if layer_idx == 1 do
            {params["input_ln"]["gamma"], params["input_ln"]["beta"]}
          else
            ln = "gru_#{layer_idx - 1}_ln"
            {params[ln]["gamma"], params[ln]["beta"]}
          end

        Nx.concatenate([
          Nx.flatten(w_x),
          Nx.flatten(b_x),
          Nx.flatten(r_w),
          Nx.flatten(gamma),
          Nx.flatten(beta)
        ])
      end

    Nx.concatenate(layers)
  end

  @doc """
  Run fused LSTM block inference with pre-packed weights.

  Runs all LSTM layers in a single kernel call.

  ## Arguments
    * `input` - [B, T, H] input tensor
    * `packed_weights` - output of `pack_block_weights/4` with `:lstm`
    * `num_layers` - number of layers
    * `h0` - optional [B, num_layers, H] initial hidden states (default: zeros)
    * `c0` - optional [B, num_layers, H] initial cell states (default: zeros)

  ## Returns
    `[B, T, H]` — output after all LSTM layers
  """
  def fused_lstm_block_inference(input, packed_weights, num_layers, h0 \\ nil, c0 \\ nil) do
    {batch, _seq_len, hidden} = Nx.shape(input)

    h0 = h0 || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})
    c0 = c0 || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

    Edifice.CUDA.FusedScan.lstm_block(input, packed_weights, h0, c0, num_layers)
  end

  @doc """
  Run fused GRU block inference with pre-packed weights.

  Runs all GRU layers in a single kernel call.

  ## Arguments
    * `input` - [B, T, H] input tensor
    * `packed_weights` - output of `pack_block_weights/4` with `:gru`
    * `num_layers` - number of layers
    * `h0` - optional [B, num_layers, H] initial hidden states (default: zeros)

  ## Returns
    `[B, T, H]` — output after all GRU layers
  """
  def fused_gru_block_inference(input, packed_weights, num_layers, h0 \\ nil) do
    {batch, _seq_len, hidden} = Nx.shape(input)

    h0 = h0 || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

    Edifice.CUDA.FusedScan.gru_block(input, packed_weights, h0, num_layers)
  end

  # ============================================================================
  # Stateful Inference (for real-time use)
  # ============================================================================

  @doc """
  Build a stateful recurrent model that explicitly manages hidden state.

  This is essential for real-time inference where we process one frame at a time
  and need to carry hidden state between frames.

  Returns a simple model that processes single frames. Hidden state management
  is handled externally using `initial_hidden/2`.

  ## Options
    - `:embed_dim` - Size of input embedding (required)
    - `:hidden_size` - Size of hidden state (default: 256)
    - `:cell_type` - :lstm or :gru (default: :lstm)

  ## Returns
    An Axon model that takes single frames and outputs hidden representations.
  """
  @spec build_stateful(keyword()) :: Axon.t()
  def build_stateful(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)

    # Single frame input reshaped as sequence of 1: [batch, 1, embed_dim]
    frame_input = Axon.input("frame", shape: {nil, 1, embed_dim})

    # Build single recurrent layer
    # Axon.lstm/gru returns {output_sequence, hidden_state}
    {output_seq, _hidden} =
      case cell_type do
        :lstm ->
          Axon.lstm(frame_input, hidden_size,
            name: "lstm_stateful",
            recurrent_initializer: :glorot_uniform
          )

        :gru ->
          Axon.gru(frame_input, hidden_size,
            name: "gru_stateful",
            recurrent_initializer: :glorot_uniform
          )
      end

    # Squeeze the sequence dimension (seq_len=1)
    Axon.nx(
      output_seq,
      fn tensor ->
        Nx.squeeze(tensor, axes: [1])
      end,
      name: "stateful_output"
    )
  end

  @doc """
  Create initial hidden state for a given batch size.

  ## Options
    - `:hidden_size` - Size of hidden state (default: 256)
    - `:cell_type` - :lstm or :gru (default: :lstm)

  ## Returns
    For LSTM: `{h, c}` tuple of zero tensors
    For GRU: single zero tensor
  """
  @spec initial_hidden(non_neg_integer(), keyword()) :: hidden_state()
  def initial_hidden(batch_size, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)

    case cell_type do
      :lstm ->
        h = Nx.broadcast(0.0, {batch_size, hidden_size})
        c = Nx.broadcast(0.0, {batch_size, hidden_size})
        {h, c}

      :gru ->
        Nx.broadcast(0.0, {batch_size, hidden_size})
    end
  end

  # ============================================================================
  # Hybrid Architectures (Recurrent + MLP)
  # ============================================================================

  @doc """
  Build a hybrid recurrent-MLP backbone.

  Combines recurrent layers for temporal processing with MLP layers
  for non-linear transformation. This often works better than pure RNN.

  ```
  Sequence [batch, seq_len, embed_dim]
        │
        ▼
  ┌─────────────┐
  │  LSTM/GRU   │
  │  Layers     │
  └─────────────┘
        │
        ▼
  [batch, hidden_size]
        │
        ▼
  ┌─────────────┐
  │    MLP      │
  │  Layers     │
  └─────────────┘
        │
        ▼
  [batch, output_size]
  ```

  ## Options
    - `:embed_dim` - Size of input embedding (required)
    - `:recurrent_size` - Size of recurrent hidden (default: 256)
    - `:mlp_sizes` - List of MLP layer sizes (default: [256])
    - `:cell_type` - :lstm or :gru (default: :lstm)
    - `:num_recurrent_layers` - Number of RNN layers (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:activation` - MLP activation (default: :relu)
  """
  @spec build_hybrid(keyword()) :: Axon.t()
  def build_hybrid(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    recurrent_size = Keyword.get(opts, :recurrent_size, @default_hidden_size)
    mlp_sizes = Keyword.get(opts, :mlp_sizes, [256])
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)
    num_recurrent_layers = Keyword.get(opts, :num_recurrent_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, 0.1)
    activation = Keyword.get(opts, :activation, :relu)

    # Input sequence
    input = Axon.input("state_sequence", shape: {nil, nil, embed_dim})

    # Recurrent backbone (outputs last timestep)
    recurrent_output =
      build_backbone(input,
        hidden_size: recurrent_size,
        num_layers: num_recurrent_layers,
        cell_type: cell_type,
        dropout: dropout,
        return_sequences: false
      )

    # MLP layers on top
    mlp_sizes
    |> Enum.with_index()
    |> Enum.reduce(recurrent_output, fn {size, idx}, acc ->
      acc
      |> Axon.dense(size, name: "hybrid_mlp_#{idx}")
      |> Axon.activation(activation)
      |> Axon.dropout(rate: dropout)
    end)
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Calculate the output size of a recurrent backbone.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Create a sequence from individual frames for batch processing.

  Takes a list of embedded frames and stacks them into a sequence tensor.
  """
  @spec frames_to_sequence([Nx.Tensor.t()]) :: Nx.Tensor.t()
  def frames_to_sequence(frames) when is_list(frames) do
    # frames: list of [embed_dim] or [batch, embed_dim] tensors
    # output: [batch, seq_len, embed_dim]
    frames
    |> Enum.map(fn frame ->
      case Nx.shape(frame) do
        # Add batch dim
        {_embed} -> Nx.new_axis(frame, 0)
        {_batch, _embed} -> frame
      end
    end)
    |> Nx.stack(axis: 1)
  end

  @doc """
  Pad or truncate sequence to fixed length.

  Useful for batch processing sequences of different lengths.
  """
  @spec pad_sequence(Nx.Tensor.t(), non_neg_integer(), keyword()) :: Nx.Tensor.t()
  def pad_sequence(sequence, target_length, opts \\ []) do
    pad_value = Keyword.get(opts, :pad_value, 0.0)

    current_length = Nx.axis_size(sequence, 1)

    cond do
      current_length == target_length ->
        sequence

      current_length > target_length ->
        # Truncate (keep last target_length frames)
        start = current_length - target_length
        Nx.slice_along_axis(sequence, start, target_length, axis: 1)

      true ->
        # Pad at the beginning
        batch_size = Nx.axis_size(sequence, 0)
        embed_dim = Nx.axis_size(sequence, 2)
        padding_length = target_length - current_length

        padding = Nx.broadcast(pad_value, {batch_size, padding_length, embed_dim})
        Nx.concatenate([padding, sequence], axis: 1)
    end
  end

  @doc """
  Get supported cell types.
  """
  @spec cell_types() :: [cell_type()]
  def cell_types, do: [:lstm, :gru]

  # ============================================================================
  # Truncated BPTT
  # ============================================================================

  @doc """
  Apply gradient truncation to a sequence for truncated BPTT.

  This creates an Axon layer that stops gradients from flowing back through
  timesteps earlier than the last `keep_steps` frames.

  ## How it works
  For a sequence of 60 frames with truncate_bptt=15:
  - Forward pass: all 60 frames processed normally
  - Backward pass: gradients only flow through the last 15 frames
  - Earlier frames have their gradients stopped with Nx.stop_gradient

  ## Performance Impact
  - ~2-3x faster training (less gradient computation)
  - May reduce ability to learn very long-range dependencies
  - Recommended: start with window_size/2 or window_size/3
  """
  @spec apply_gradient_truncation(Axon.t(), pos_integer()) :: Axon.t()
  def apply_gradient_truncation(input, keep_steps)
      when is_integer(keep_steps) and keep_steps > 0 do
    Axon.nx(
      input,
      fn sequence ->
        # sequence shape: [batch, seq_len, embed_dim]
        seq_len = Nx.axis_size(sequence, 1)

        if seq_len <= keep_steps do
          # No truncation needed if sequence is shorter than keep_steps
          sequence
        else
          # Split into "frozen" (stop gradient) and "active" (keep gradient) parts
          frozen_len = seq_len - keep_steps

          # Slice the frozen part (early frames) and stop its gradient
          frozen_part = Nx.slice_along_axis(sequence, 0, frozen_len, axis: 1)
          frozen_part = Nx.Defn.Kernel.stop_grad(frozen_part)

          # Slice the active part (last keep_steps frames)
          active_part = Nx.slice_along_axis(sequence, frozen_len, keep_steps, axis: 1)

          # Concatenate back together
          Nx.concatenate([frozen_part, active_part], axis: 1)
        end
      end,
      name: "truncated_bptt_#{keep_steps}"
    )
  end

  # ============================================================================
  # Stateful step contract (Edifice.Stateful)
  # ============================================================================

  @behaviour Edifice.Stateful

  alias Edifice.Stateful.Ops

  @doc """
  Initial recurrent state for O(1) stepping.

  GRU: `%{h: [batch, num_layers, hidden]}`. LSTM additionally carries
  `%{c: [batch, num_layers, hidden]}`.

  Two build-time graph layouts exist and are detected from the params:

    * **Axon layout** (`Axon.gru`/`Axon.lstm`, the CPU/hex-deps path) —
      Axon's initial hidden state is NOT zeros: it is glorot-uniform sampled
      from a stored RNG key (`"gru_1_h_hidden_state" => %{"key" => ...}`).
      We replicate that exactly (read key, `Nx.Random.split`, sample), or
      equivalence fails from the very first frame.
    * **Fused layout** (`_input_proj` + `_fused_scan`, the CUDA-fork path) —
      zero initial states, matching the kernel and its CPU fallback.

  Options: `:batch_size` (default 1), `:hidden_size`, `:num_layers`,
  `:cell_type` (same defaults as `build/1`; the registry's `:gru`/`:lstm`
  entries merge `cell_type` in automatically via `Edifice.init_state/3`).
  """
  @impl Edifice.Stateful
  def init_state(params, opts \\ []) do
    params = Ops.unwrap_params(params)
    batch_size = Keyword.get(opts, :batch_size, 1)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    cell_type = Keyword.get(opts, :cell_type, @default_cell_type)

    layer_state = fn state_name ->
      1..num_layers
      |> Enum.map(fn i ->
        case rnn_layout(params, cell_type) do
          :axon ->
            rnn_initial_hidden(
              params,
              "#{cell_type}_#{i}_#{state_name}_hidden_state",
              batch_size,
              hidden_size
            )

          :fused ->
            Ops.zeros(batch_size, [hidden_size])
        end
      end)
      |> Nx.stack(axis: 1)
    end

    case cell_type do
      :lstm -> %{h: layer_state.("h"), c: layer_state.("c")}
      :gru -> %{h: layer_state.("h")}
    end
  end

  @doc """
  Advance one frame: `[batch, embed_dim]` in, `{[batch, hidden], state}` out.

  Cell type is inferred from the state (`:c` present = LSTM); the graph
  layout is inferred from the param keys. Matches the full-sequence forward
  exactly (pinned by `test/edifice/stateful/step_equivalence_test.exs`).
  """
  @impl Edifice.Stateful
  def step(params, %{h: h} = state, frame) do
    params = Ops.unwrap_params(params)
    cell_type = if Map.has_key?(state, :c), do: :lstm, else: :gru
    num_layers = Nx.axis_size(h, 1)
    layout = rnn_layout(params, cell_type)

    x =
      case params do
        %{"input_ln" => ln} -> Ops.layer_norm(frame, ln, 1.0e-6)
        _ -> frame
      end

    slice = fn tensor, i ->
      tensor |> Nx.slice_along_axis(i - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
    end

    {x, new_hs, new_cs} =
      Enum.reduce(1..num_layers, {x, [], []}, fn i, {acc, hs, cs} ->
        name = "#{cell_type}_#{i}"
        h_i = slice.(h, i)

        {out, h_new, c_new} =
          case {layout, cell_type} do
            {:axon, :gru} ->
              layer = Ops.layer_params!(params, name)

              {out, {h_new}} =
                Axon.Layers.gru_cell(
                  acc,
                  {h_i},
                  Nx.tensor(0),
                  layer["input_kernel"],
                  layer["hidden_kernel"],
                  layer["bias"]
                )

              {out, h_new, nil}

            {:axon, :lstm} ->
              layer = Ops.layer_params!(params, name)
              c_i = slice.(state.c, i)

              {out, {c_new, h_new}} =
                Axon.Layers.lstm_cell(
                  acc,
                  {c_i, h_i},
                  Nx.tensor(0),
                  layer["input_kernel"],
                  layer["hidden_kernel"],
                  layer["bias"]
                )

              {out, h_new, c_new}

            {:fused, :gru} ->
              wx = Ops.dense(acc, Ops.layer_params!(params, "#{name}_input_proj"))
              r_kernel = Ops.layer_params!(params, "#{name}_fused_scan")["#{name}_recurrent_kernel"]
              h_new = fused_gru_cell_step(wx, r_kernel, h_i)
              {h_new, h_new, nil}

            {:fused, :lstm} ->
              wx = Ops.dense(acc, Ops.layer_params!(params, "#{name}_input_proj"))
              r_kernel = Ops.layer_params!(params, "#{name}_fused_scan")["#{name}_recurrent_kernel"]
              c_i = slice.(state.c, i)
              {h_new, c_new} = fused_lstm_cell_step(wx, r_kernel, h_i, c_i)
              {h_new, h_new, c_new}
          end

        # Per-layer post-RNN layer norm (present unless use_layer_norm: false)
        out =
          case Map.fetch(params, "#{name}_ln") do
            {:ok, ln} -> Ops.layer_norm(out, ln, 1.0e-6)
            :error -> out
          end

        {out, [h_new | hs], [c_new | cs]}
      end)

    new_state = %{h: new_hs |> Enum.reverse() |> Nx.stack(axis: 1)}

    new_state =
      if cell_type == :lstm do
        Map.put(new_state, :c, new_cs |> Enum.reverse() |> Nx.stack(axis: 1))
      else
        new_state
      end

    {x, new_state}
  end

  # One step of the fused-layout GRU, transcribed from
  # Edifice.CUDA.FusedScan.gru_scan_fallback/3: gate order [r, z, n], reset
  # applied to the recurrent contribution only, no recurrent bias.
  defp fused_gru_cell_step(wx, r_kernel, h_prev) do
    hidden = Nx.axis_size(h_prev, 1)
    rh = Nx.dot(h_prev, [1], r_kernel, [0])

    r =
      Nx.sigmoid(
        Nx.add(
          Nx.slice_along_axis(wx, 0, hidden, axis: 1),
          Nx.slice_along_axis(rh, 0, hidden, axis: 1)
        )
      )

    z =
      Nx.sigmoid(
        Nx.add(
          Nx.slice_along_axis(wx, hidden, hidden, axis: 1),
          Nx.slice_along_axis(rh, hidden, hidden, axis: 1)
        )
      )

    n =
      Nx.tanh(
        Nx.add(
          Nx.slice_along_axis(wx, hidden * 2, hidden, axis: 1),
          Nx.multiply(r, Nx.slice_along_axis(rh, hidden * 2, hidden, axis: 1))
        )
      )

    Nx.add(Nx.multiply(Nx.subtract(1.0, z), n), Nx.multiply(z, h_prev))
  end

  # One step of the fused-layout LSTM, transcribed from
  # Edifice.CUDA.FusedScan.lstm_scan_fallback/4: gate order [i, f, g, o].
  defp fused_lstm_cell_step(wx, r_kernel, h_prev, c_prev) do
    hidden = Nx.axis_size(h_prev, 1)
    gates = Nx.add(wx, Nx.dot(h_prev, [1], r_kernel, [0]))

    i = Nx.sigmoid(Nx.slice_along_axis(gates, 0, hidden, axis: 1))
    f = Nx.sigmoid(Nx.slice_along_axis(gates, hidden, hidden, axis: 1))
    g = Nx.tanh(Nx.slice_along_axis(gates, hidden * 2, hidden, axis: 1))
    o = Nx.sigmoid(Nx.slice_along_axis(gates, hidden * 3, hidden, axis: 1))

    c_new = Nx.add(Nx.multiply(f, c_prev), Nx.multiply(i, g))
    h_new = Nx.multiply(o, Nx.tanh(c_new))

    {h_new, c_new}
  end

  defp rnn_layout(params, cell_type) do
    cond do
      Map.has_key?(params, "#{cell_type}_1") -> :axon
      Map.has_key?(params, "#{cell_type}_1_fused_scan") -> :fused
      true ->
        raise ArgumentError,
              "params contain neither #{inspect("#{cell_type}_1")} (Axon layout) nor " <>
                "#{inspect("#{cell_type}_1_fused_scan")} (fused layout) — were these " <>
                "params trained with cell_type: #{inspect(cell_type)}?"
    end
  end

  # Replicates Axon's rnn_state initialization (axon.ex rnn_state/7): the
  # initial hidden state is glorot-uniform sampled from the stored RNG key,
  # NOT zeros. keys[1] is what the forward consumes in inference mode.
  defp rnn_initial_hidden(params, state_layer_name, batch_size, hidden_size) do
    key = Ops.layer_params!(params, state_layer_name)["key"]
    keys = Nx.Random.split(key)
    Axon.Initializers.glorot_uniform().({batch_size, hidden_size}, {:f, 32}, keys[1])
  end
end
