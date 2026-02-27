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
  Input [batch, seq_len, embed_dim]
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
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        variant: :slstm
      )

      # mLSTM-only model (memorization)
      model = XLSTM.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        variant: :mlstm
      )

      # Mixed model (default: alternating)
      model = XLSTM.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        variant: :mixed  # sLSTM at layers 1,3,5; mLSTM at 2,4,6
      )

  ## References
  - Paper: https://arxiv.org/abs/2405.04517
  - Official code: https://github.com/NX-AI/xlstm
  """

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:variant, :slstm | :mlstm | :mixed}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
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

  @doc "Default head dimension for mLSTM"
  @spec default_head_dim() :: pos_integer()
  def default_head_dim, do: 64

  @doc "Default number of heads for mLSTM"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: 4

  @doc "Default feedforward expansion factor"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 2

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  @doc "Stabilization epsilon for exponential gating"
  @spec gate_eps() :: float()
  def gate_eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an xLSTM model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
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
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    variant = Keyword.get(opts, :variant, :mixed)
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
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

  sLSTM equations with log-domain stabilization:
  - log_i_t = W_i x_t + R_i h_{t-1} + b_i
  - log_f_t = W_f x_t + R_f h_{t-1} + b_f
  - z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)
  - o_t = sigmoid(W_o x_t + R_o h_{t-1} + b_o)

  Log-domain stabilization (prevents exponential overflow):
  - m_t = max(log_f_t + m_{t-1}, log_i_t)
  - i_t' = exp(log_i_t - m_t)
  - f_t' = exp(log_f_t + m_{t-1} - m_t)
  - c_t = f_t' * c_{t-1} + i_t' * z_t
  - n_t = f_t' * n_{t-1} + i_t'
  - h_t = o_t * (c_t / max(|n_t|, 1))

  The recurrent connections R_i, R_f, R_z, R_o enable memory mixing.
  """
  @spec build_slstm_layer(Axon.t(), keyword()) :: Axon.t()
  def build_slstm_layer(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    name = Keyword.get(opts, :name, "slstm")

    # Input projection: W*x + b for all 4 gates across all timesteps
    input_gates = Axon.dense(input, hidden_size * 4, name: "#{name}_input_proj")

    # Recurrent weight: R matrix [hidden_size, 4*hidden_size]
    # Projects h_{t-1} to gate contributions for i, f, z, o
    recurrent_weight =
      Axon.param("#{name}_recurrent_weight", {hidden_size, hidden_size * 4},
        initializer: :glorot_uniform
      )

    # Apply sLSTM recurrence with recurrent connections and log-domain stabilization
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
    # input_gates: [batch, seq_len, hidden_size * 4] (W*x + b, precomputed)
    # recurrent_weight: [hidden_size, hidden_size * 4]
    batch_size = Nx.axis_size(input_gates, 0)
    seq_len = Nx.axis_size(input_gates, 1)

    # Initialize states
    type = Nx.type(input_gates)
    h_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})
    c_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})
    n_prev = Nx.broadcast(Nx.tensor(1.0, type: type), {batch_size, hidden_size})
    m_prev = Nx.broadcast(Nx.tensor(0.0, type: type), {batch_size, hidden_size})

    # Sequential recurrence with log-domain stabilization
    {_, h_list} =
      Enum.reduce(
        0..(seq_len - 1),
        {{h_prev, c_prev, n_prev, m_prev}, []},
        fn t, {{h_p, c_p, n_p, m_p}, acc} ->
          # Input projection at time t: [batch, 4*hidden]
          wx_t = Nx.slice_along_axis(input_gates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

          # Recurrent projection: h_{t-1} @ R -> [batch, 4*hidden]
          rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])

          # Combined gate pre-activations
          gates_t = Nx.add(wx_t, rh_t)

          # Split into individual gates
          log_i_t = Nx.slice_along_axis(gates_t, 0, hidden_size, axis: 1)
          log_f_t = Nx.slice_along_axis(gates_t, hidden_size, hidden_size, axis: 1)
          z_t = Nx.slice_along_axis(gates_t, hidden_size * 2, hidden_size, axis: 1) |> Nx.tanh()

          o_t =
            Nx.slice_along_axis(gates_t, hidden_size * 3, hidden_size, axis: 1) |> Nx.sigmoid()

          # Log-domain stabilization
          # m_t = max(log_f_t + m_{t-1}, log_i_t)
          m_t = Nx.max(Nx.add(log_f_t, m_p), log_i_t)

          # Stabilized gates
          # i_t' = exp(log_i_t - m_t)
          i_t = Nx.exp(Nx.subtract(log_i_t, m_t))
          # f_t' = exp(log_f_t + m_{t-1} - m_t)
          f_t = Nx.exp(Nx.subtract(Nx.add(log_f_t, m_p), m_t))

          # Cell state update: c_t = f_t' * c_{t-1} + i_t' * z_t
          c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, z_t))

          # Normalizer update: n_t = f_t' * n_{t-1} + i_t'
          n_t = Nx.add(Nx.multiply(f_t, n_p), i_t)

          # Hidden state: h_t = o_t * (c_t / max(|n_t|, 1))
          safe_denom = Nx.max(Nx.abs(n_t), 1.0)
          h_t = Nx.multiply(o_t, Nx.divide(c_t, safe_denom))

          {{h_t, c_t, n_t, m_t}, [h_t | acc]}
        end
      )

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
    # i, f gates: scalar per head (num_heads each) — exponential gating
    # o gate: hidden_size — sigmoid gating
    # k, v, q: num_heads * head_dim each
    kv_dim = num_heads * head_dim
    total_proj = num_heads * 2 + hidden_size + kv_dim * 3

    projections = Axon.dense(input, total_proj, name: "#{name}_proj")

    # Apply mLSTM with matrix memory via D-matrix formulation
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

    # Split projections: i(num_heads), f(num_heads), o(hidden), k(kv), v(kv), q(kv)
    offset = 0
    i_pre = Nx.slice_along_axis(projections, offset, num_heads, axis: 2)
    offset = offset + num_heads
    f_pre = Nx.slice_along_axis(projections, offset, num_heads, axis: 2)
    offset = offset + num_heads
    o_pre = Nx.slice_along_axis(projections, offset, hidden_size, axis: 2)
    offset = offset + hidden_size
    k_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)
    offset = offset + kv_dim
    v_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)
    offset = offset + kv_dim
    q_proj = Nx.slice_along_axis(projections, offset, kv_dim, axis: 2)

    # Stabilized exponential gating (log space)
    max_gate_val = 20.0
    log_i = Nx.clip(i_pre, -max_gate_val, max_gate_val)
    log_f = Nx.clip(f_pre, -max_gate_val, max_gate_val)
    o_gate = Nx.sigmoid(o_pre)

    # Reshape gates to [batch, num_heads, seq] for head-wise operations
    # i_pre, f_pre: [batch, seq, num_heads] -> [batch, num_heads, seq]
    log_i_h = Nx.transpose(log_i, axes: [0, 2, 1])
    log_f_h = Nx.transpose(log_f, axes: [0, 2, 1])

    # Cumulative log forget: [batch, heads, seq]
    cum_log_f = Nx.cumulative_sum(log_f_h, axis: 2)

    # D matrix: log_D[t, i] = log_i[i] + cum_log_f[t] - cum_log_f[i]
    # This captures: alpha_{t,i} = i_i * prod_{j=i+1}^{t} f_j
    # Expand for broadcasting to [batch, heads, seq_t, seq_i]:
    log_i_expanded = Nx.reshape(log_i_h, {batch_size, num_heads, 1, seq_len})
    cum_f_t = Nx.reshape(cum_log_f, {batch_size, num_heads, seq_len, 1})
    cum_f_i = Nx.reshape(cum_log_f, {batch_size, num_heads, 1, seq_len})

    log_d = Nx.add(log_i_expanded, Nx.subtract(cum_f_t, cum_f_i))

    # Apply causal mask (t >= i only)
    causal_mask =
      Edifice.Blocks.CausalMask.causal(seq_len) |> Nx.reshape({1, 1, seq_len, seq_len})

    neg_inf = Nx.Constants.neg_infinity(Nx.type(log_d))
    log_d = Nx.select(Nx.broadcast(causal_mask, Nx.shape(log_d)), log_d, neg_inf)

    # Stabilize D: subtract max per query position
    max_log_d = Nx.reduce_max(log_d, axes: [3], keep_axes: true)
    max_log_d = Nx.clip(max_log_d, -1.0e10, 1.0e10)
    d_matrix = Nx.exp(Nx.subtract(log_d, max_log_d))

    # Reshape K, V, Q for heads: [batch, heads, seq, head_dim]
    k = reshape_for_heads(k_proj, batch_size, seq_len, num_heads, head_dim)
    v = reshape_for_heads(v_proj, batch_size, seq_len, num_heads, head_dim)
    q = reshape_for_heads(q_proj, batch_size, seq_len, num_heads, head_dim)

    # K-Q scores: [batch, heads, seq_t, seq_i]
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(k)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Combined weights: D * scores (gate-weighted attention)
    weights = Nx.multiply(d_matrix, scores)

    # Numerator: weighted sum of values
    # [batch, heads, seq_t, seq_i] @ [batch, heads, seq_i, head_dim]
    numerator = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Denominator: sum of weights per query position
    denominator = Nx.sum(weights, axes: [3], keep_axes: true)

    # Normalize: h = numerator / max(|denominator|, 1)
    # This is the key mLSTM normalization (NOT softmax!)
    safe_denom = Nx.max(Nx.abs(denominator), 1.0)
    h = Nx.divide(numerator, safe_denom)

    # Reshape back: [batch, seq, kv_dim]
    output = Nx.transpose(h, axes: [0, 2, 1, 3])
    output = Nx.reshape(output, {batch_size, seq_len, kv_dim})

    # Apply output gate
    if kv_dim != hidden_size do
      Nx.slice_along_axis(output, 0, hidden_size, axis: 2)
      |> Nx.multiply(o_gate)
    else
      Nx.multiply(o_gate, output)
    end
  end

  defp reshape_for_heads(tensor, batch_size, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
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
    embed_dim = Keyword.get(opts, :embed_dim, 287)
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
    # - i, f gate projections: hidden * (2 * num_heads) (scalar per head)
    # - o gate projection: hidden * hidden
    # - K, V, Q projections: hidden * (3 * kv_dim)
    mlstm_params = hidden_size * (2 * num_heads + hidden_size) + hidden_size * (3 * kv_dim)

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

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

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
