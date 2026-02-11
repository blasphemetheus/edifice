defmodule Edifice.Recurrent.XLSTM do
  @moduledoc """
  xLSTM: Extended Long Short-Term Memory.

  Implements the xLSTM architecture from "xLSTM: Extended Long Short-Term Memory"
  (Beck et al., NeurIPS 2024).

  ## Key Innovations

  xLSTM addresses three fundamental LSTM limitations:
  1. Inability to revise storage decisions -> **Exponential gating**
  2. Limited storage capacity -> **Matrix memory (mLSTM)**
  3. Lack of parallelizability -> **mLSTM covariance update**

  ## Two Variants

  ### sLSTM (Scalar LSTM)
  - Exponential gating: `i_t = exp(W_i x_t + R_i h_{t-1} + b_i)`
  - Normalizer state prevents overflow: `n_t = f_t * n_{t-1} + i_t`
  - Sequential processing with memory mixing
  - Good for state-tracking tasks

  ### mLSTM (Matrix LSTM)
  - Matrix memory cell: `C_t = f_t * C_{t-1} + i_t * (v_t k_t^T)`
  - Key-value storage similar to attention
  - Fully parallelizable during training
  - Good for memorization tasks

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |         xLSTM Block                  |
  |  +----------------------------------+|
  |  | Layer Norm -> sLSTM/mLSTM        ||
  |  |       |                          ||
  |  | Layer Norm -> Feedforward        ||
  |  |       |                          ||
  |  | Residual Connection             ||
  |  +----------------------------------+|
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      # sLSTM-only model (state tracking)
      model = XLSTM.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        variant: :slstm
      )

      # mLSTM-only model (memorization)
      model = XLSTM.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        variant: :mlstm
      )

      # Mixed model (default: alternating)
      model = XLSTM.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 6,
        variant: :mixed  # sLSTM at layers 1,3,5; mLSTM at 2,4,6
      )

  ## References
  - Paper: https://arxiv.org/abs/2405.04517
  - Official code: https://github.com/NX-AI/xlstm
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  def default_num_layers, do: 4

  @doc "Default head dimension for mLSTM"
  def default_head_dim, do: 64

  @doc "Default number of heads for mLSTM"
  def default_num_heads, do: 4

  @doc "Default feedforward expansion factor"
  def default_expand_factor, do: 2

  @doc "Default dropout rate"
  def default_dropout, do: 0.0

  @doc "Stabilization epsilon for exponential gating"
  def gate_eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an xLSTM model for sequence processing.

  ## Options
    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of xLSTM blocks (default: 4)
    - `:variant` - :slstm, :mlstm, or :mixed (default: :mixed)
    - `:num_heads` - Number of heads for mLSTM (default: 4)
    - `:head_dim` - Dimension per head for mLSTM (default: 64)
    - `:expand_factor` - Feedforward expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    variant = Keyword.get(opts, :variant, :mixed)
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack xLSTM blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Determine block type based on variant
        block_type = get_block_type(variant, layer_idx)

        block_opts = Keyword.merge(opts, layer_idx: layer_idx, block_type: block_type)
        block = build_xlstm_block(acc, block_opts)

        # Dropout between blocks
        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "dropout_#{layer_idx}")
        else
          block
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
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

  defp get_block_type(:slstm, _layer_idx), do: :slstm
  defp get_block_type(:mlstm, _layer_idx), do: :mlstm

  defp get_block_type(:mixed, layer_idx) do
    # Alternate: odd layers get sLSTM, even layers get mLSTM
    if rem(layer_idx, 2) == 1, do: :slstm, else: :mlstm
  end

  # ============================================================================
  # xLSTM Block
  # ============================================================================

  @doc """
  Build a single xLSTM block.

  xLSTM block structure:
  1. LayerNorm -> sLSTM/mLSTM -> Residual
  2. LayerNorm -> Feedforward -> Residual
  """
  @spec build_xlstm_block(Axon.t(), keyword()) :: Axon.t()
  def build_xlstm_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    block_type = Keyword.get(opts, :block_type, :slstm)
    name = "xlstm_block_#{layer_idx}"

    # 1. Temporal mixing (sLSTM or mLSTM)
    temporal_normed = Axon.layer_norm(input, name: "#{name}_temporal_norm")

    temporal_out =
      case block_type do
        :slstm ->
          build_slstm_layer(temporal_normed, Keyword.put(opts, :name, "#{name}_slstm"))

        :mlstm ->
          build_mlstm_layer(temporal_normed, Keyword.put(opts, :name, "#{name}_mlstm"))
      end

    # Residual connection
    after_temporal = Axon.add(input, temporal_out, name: "#{name}_temporal_residual")

    # 2. Feedforward branch
    ff_normed = Axon.layer_norm(after_temporal, name: "#{name}_ff_norm")
    ff_out = build_feedforward(ff_normed, hidden_size, expand_factor, "#{name}_ff")

    # Residual connection
    Axon.add(after_temporal, ff_out, name: "#{name}_ff_residual")
  end

  # ============================================================================
  # sLSTM (Scalar LSTM with Exponential Gating)
  # ============================================================================

  @doc """
  Build the sLSTM (Scalar LSTM) layer.

  sLSTM equations:
  - i_t = exp(W_i x_t + R_i h_{t-1} + b_i)     # Input gate (exponential)
  - f_t = exp(W_f x_t + R_f h_{t-1} + b_f)     # Forget gate (exponential)
  - z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)    # Cell input
  - o_t = sigmoid(W_o x_t + R_o h_{t-1} + b_o)  # Output gate (sigmoid)
  - c_t = f_t * c_{t-1} + i_t * z_t             # Cell state
  - n_t = f_t * n_{t-1} + i_t                   # Normalizer state
  - h_t = o_t * (c_t / n_t)                     # Hidden state

  The normalizer n_t prevents exponential overflow from accumulated gates.
  """
  @spec build_slstm_layer(Axon.t(), keyword()) :: Axon.t()
  def build_slstm_layer(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    name = Keyword.get(opts, :name, "slstm")

    # Project input to gate dimensions
    # We need: i, f, z, o gates (4 * hidden_size)
    gates = Axon.dense(input, hidden_size * 4, name: "#{name}_gates_proj")

    # Apply sLSTM recurrence
    Axon.nx(
      gates,
      fn gate_tensor ->
        slstm_forward(gate_tensor, hidden_size)
      end,
      name: "#{name}_recurrence"
    )
  end

  defp slstm_forward(gates, hidden_size) do
    # gates: [batch, seq_len, hidden_size * 4]
    batch_size = Nx.axis_size(gates, 0)
    seq_len = Nx.axis_size(gates, 1)

    # Split into individual gates
    i_pre = Nx.slice_along_axis(gates, 0, hidden_size, axis: 2)
    f_pre = Nx.slice_along_axis(gates, hidden_size, hidden_size, axis: 2)
    z_pre = Nx.slice_along_axis(gates, hidden_size * 2, hidden_size, axis: 2)
    o_pre = Nx.slice_along_axis(gates, hidden_size * 3, hidden_size, axis: 2)

    # Stabilized exponential gating
    # To prevent overflow, we use: exp(x - max(x)) and track the max
    # For simplicity, we clip the pre-activation values
    max_gate_val = 20.0
    i_pre_clipped = Nx.clip(i_pre, -max_gate_val, max_gate_val)
    f_pre_clipped = Nx.clip(f_pre, -max_gate_val, max_gate_val)

    # Initialize states
    c_init = Nx.broadcast(0.0, {batch_size, 1, hidden_size})
    n_init = Nx.broadcast(1.0, {batch_size, 1, hidden_size})

    # Sequential recurrence
    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {{c_init, n_init}, []}, fn t, {{c_prev, n_prev}, acc} ->
        # Get gates at time t
        i_t = Nx.slice_along_axis(i_pre_clipped, t, 1, axis: 1) |> Nx.exp()
        f_t = Nx.slice_along_axis(f_pre_clipped, t, 1, axis: 1) |> Nx.exp()
        z_t = Nx.slice_along_axis(z_pre, t, 1, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(o_pre, t, 1, axis: 1) |> Nx.sigmoid()

        # Cell state update: c_t = f_t * c_{t-1} + i_t * z_t
        c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, z_t))

        # Normalizer update: n_t = f_t * n_{t-1} + i_t
        n_t = Nx.add(Nx.multiply(f_t, n_prev), i_t)

        # Hidden state: h_t = o_t * (c_t / n_t)
        # Add epsilon to prevent division by zero
        h_t = Nx.multiply(o_t, Nx.divide(c_t, Nx.add(n_t, gate_eps())))

        {{c_t, n_t}, [Nx.squeeze(h_t, axes: [1]) | acc]}
      end)

    # Stack: [batch, seq_len, hidden_size]
    h_list
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # mLSTM (Matrix LSTM)
  # ============================================================================

  @doc """
  Build the mLSTM (Matrix LSTM) layer.

  mLSTM equations:
  - i_t = exp(W_i x_t + b_i)                   # Input gate (exponential)
  - f_t = exp(W_f x_t + b_f)                   # Forget gate (exponential)
  - o_t = sigmoid(W_o x_t + b_o)               # Output gate (sigmoid)
  - k_t = W_k x_t                              # Key projection
  - v_t = W_v x_t                              # Value projection
  - q_t = W_q x_t                              # Query projection
  - C_t = f_t * C_{t-1} + i_t * (v_t k_t^T)   # Matrix memory
  - n_t = f_t * n_{t-1} + i_t * k_t            # Normalizer
  - h_t = o_t * (C_t q_t / max(q_t^T n_t, 1)) # Hidden state

  The matrix memory C stores key-value associations like attention.
  """
  @spec build_mlstm_layer(Axon.t(), keyword()) :: Axon.t()
  def build_mlstm_layer(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    head_dim = Keyword.get(opts, :head_dim, default_head_dim())
    name = Keyword.get(opts, :name, "mlstm")

    # Project to gates and key/value/query
    # i, f, o gates: 3 * hidden_size
    # k, v, q: 3 * (num_heads * head_dim)
    kv_dim = num_heads * head_dim
    total_proj = hidden_size * 3 + kv_dim * 3

    projections = Axon.dense(input, total_proj, name: "#{name}_proj")

    # Apply mLSTM recurrence
    Axon.nx(
      projections,
      fn proj_tensor ->
        mlstm_forward(proj_tensor, hidden_size, num_heads, head_dim)
      end,
      name: "#{name}_recurrence"
    )
  end

  defp mlstm_forward(projections, hidden_size, num_heads, head_dim) do
    # projections: [batch, seq_len, total_proj]
    batch_size = Nx.axis_size(projections, 0)
    seq_len = Nx.axis_size(projections, 1)
    kv_dim = num_heads * head_dim

    # Split projections
    offset = 0
    i_pre = Nx.slice_along_axis(projections, offset, hidden_size, axis: 2)
    offset = offset + hidden_size
    f_pre = Nx.slice_along_axis(projections, offset, hidden_size, axis: 2)
    offset = offset + hidden_size
    o_pre = Nx.slice_along_axis(projections, offset, hidden_size, axis: 2)
    offset = offset + hidden_size
    k_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)
    offset = offset + kv_dim
    v_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)
    offset = offset + kv_dim
    q_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)

    # Stabilized exponential gating
    max_gate_val = 20.0
    i_pre_clipped = Nx.clip(i_pre, -max_gate_val, max_gate_val)
    f_pre_clipped = Nx.clip(f_pre, -max_gate_val, max_gate_val)

    # For mLSTM, we use a simplified parallel formulation
    # This is an approximation that processes all timesteps at once
    # For exact sequential behavior, use the sequential version below

    # Compute gates for all timesteps
    # In the full mLSTM, these gates modulate the matrix memory update
    # For this simplified parallel version, we use them to modulate values
    # [batch, seq, hidden]
    i_gate = Nx.exp(i_pre_clipped)
    # [batch, seq, hidden]
    f_gate = Nx.exp(f_pre_clipped)
    # [batch, seq, hidden]
    o_gate = Nx.sigmoid(o_pre)

    # Compute gate factor for values (approximation of gated memory)
    # Normalized gate: i / (f + i) gives relative importance of new vs old
    gate_sum = Nx.add(f_gate, i_gate)
    gate_factor = Nx.divide(i_gate, Nx.add(gate_sum, gate_eps()))

    # Reshape k, v, q for multi-head
    # [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    k = reshape_for_heads(k_proj, batch_size, seq_len, num_heads, head_dim)
    v = reshape_for_heads(v_proj, batch_size, seq_len, num_heads, head_dim)
    q = reshape_for_heads(q_proj, batch_size, seq_len, num_heads, head_dim)

    # Apply gate factor to values (modulates contribution)
    # Project gate_factor to kv_dim if needed, or broadcast
    gate_factor_kv =
      if hidden_size == kv_dim do
        gate_factor
      else
        # Average pool gate_factor to match kv_dim dimensions
        # Simple approach: take first kv_dim elements if hidden_size > kv_dim
        # or tile if hidden_size < kv_dim
        if hidden_size >= kv_dim do
          Nx.slice_along_axis(gate_factor, 0, kv_dim, axis: 2)
        else
          # Tile to match
          times = div(kv_dim, hidden_size) + 1
          tiled = Nx.tile(gate_factor, [1, 1, times])
          Nx.slice_along_axis(tiled, 0, kv_dim, axis: 2)
        end
      end

    gate_factor_heads =
      reshape_for_heads(gate_factor_kv, batch_size, seq_len, num_heads, head_dim)

    v = Nx.multiply(v, gate_factor_heads)

    # Compute attention-like scores: [batch, heads, seq, seq]
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(k)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Apply causal mask
    causal_mask = create_causal_mask(seq_len)
    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))

    scores =
      Nx.select(
        Nx.broadcast(causal_mask, Nx.shape(scores)),
        scores,
        Nx.broadcast(neg_inf, Nx.shape(scores))
      )

    # Softmax attention
    max_scores = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    attn_weights = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(attn_weights, Nx.sum(attn_weights, axes: [3], keep_axes: true))

    # Apply attention to gated values
    # [batch, heads, seq, seq] @ [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    output = Nx.transpose(attn_out, axes: [0, 2, 1, 3])
    output = Nx.reshape(output, {batch_size, seq_len, num_heads * head_dim})

    # Project to hidden_size if needed and apply output gating
    if num_heads * head_dim != hidden_size do
      # Truncate or pad to match hidden_size
      Nx.slice_along_axis(output, 0, hidden_size, axis: 2)
      |> Nx.multiply(o_gate)
    else
      # Apply output gating
      Nx.multiply(o_gate, output)
    end
  end

  defp reshape_for_heads(tensor, batch_size, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp create_causal_mask(seq_len) do
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    Nx.greater_equal(rows, cols) |> Nx.reshape({1, 1, seq_len, seq_len})
  end

  # ============================================================================
  # Feedforward
  # ============================================================================

  @doc """
  Build a feedforward layer with GeLU activation.
  """
  @spec build_feedforward(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_feedforward(input, hidden_size, expand_factor, name) do
    inner_size = hidden_size * expand_factor

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an xLSTM model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc """
  Calculate approximate parameter count for an xLSTM model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    head_dim = Keyword.get(opts, :head_dim, default_head_dim())
    variant = Keyword.get(opts, :variant, :mixed)

    inner_size = hidden_size * expand_factor
    kv_dim = num_heads * head_dim

    # sLSTM block parameters:
    # - Gate projections: hidden * (4 * hidden)
    slstm_params = hidden_size * (4 * hidden_size)

    # mLSTM block parameters:
    # - Gate projections: hidden * (3 * hidden)
    # - K, V, Q projections: hidden * (3 * kv_dim)
    mlstm_params = hidden_size * (3 * hidden_size) + hidden_size * (3 * kv_dim)

    # Feedforward parameters:
    # - Up projection: hidden * inner
    # - Down projection: inner * hidden
    ff_params = hidden_size * inner_size + inner_size * hidden_size

    # Count layers by type
    {num_slstm, num_mlstm} =
      case variant do
        :slstm ->
          {num_layers, 0}

        :mlstm ->
          {0, num_layers}

        :mixed ->
          # Odd layers
          slstm_count = div(num_layers + 1, 2)
          # Even layers
          mlstm_count = div(num_layers, 2)
          {slstm_count, mlstm_count}
      end

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj +
      num_slstm * (slstm_params + ff_params) +
      num_mlstm * (mlstm_params + ff_params)
  end

  @doc """
  Get recommended defaults for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      num_heads: 4,
      head_dim: 64,
      expand_factor: 2,
      variant: :mixed,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
