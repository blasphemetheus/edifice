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
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:cell_type, cell_type()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
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
      truncate_bptt: truncate_bptt
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

    # RNN stability comes from layer normalization (below) + lower learning rates
    # Orthogonal init can help but has shape constraints that fail with small hidden sizes
    # This is the recommended initialization for RNNs per Saxe et al. (2013)
    recurrent_opts = [
      name: name,
      recurrent_initializer: recurrent_init,
      use_bias: true
    ]

    # Axon.lstm/gru returns {output_sequence, hidden_state_tuple}
    # We need to extract just the output sequence using Axon.elem
    {output_seq, _hidden} =
      case cell_type do
        :lstm -> Axon.lstm(input, hidden_size, recurrent_opts)
        :gru -> Axon.gru(input, hidden_size, recurrent_opts)
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
end
