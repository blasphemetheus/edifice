defmodule Edifice.Attention.RetNet do
  @moduledoc """
  RetNet: Retentive Network - A Successor to Transformer.

  Implements the RetNet architecture from "Retentive Network: A Successor to
  Transformer for Large Language Models" (Sun et al., Microsoft 2023).

  ## Key Innovation: Retention Mechanism

  RetNet replaces attention with "retention" - a decay-based mechanism:

  ```
  Parallel:   Y = (Q . Theta) . D . (K . Theta)^T . V
  Recurrent:  s_n = gamma*s_{n-1} + K_n^T*V_n; o_n = Q_n*s_n
  ```

  Where D is a decay matrix: D[n,m] = gamma^(n-m) if n>=m, else 0.

  ## Triple Paradigm

  The same weights support three computation modes:
  - **Parallel**: Training mode, O(L^2) but GPU-parallel
  - **Recurrent**: Inference mode, O(1) per token
  - **Chunkwise**: Long sequences, O(L) with chunking

  ## Multi-Scale Retention (MSR)

  Different heads use different decay rates for multi-scale modeling:
  - gamma_h = 1 - 2^(-5-h) for head h
  - GroupNorm instead of LayerNorm (handles different head variances)
  - SiLU gating: Y = SiLU(X*W_G) . Retention(X)*W_O

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |       RetNet Block                   |
  |  LayerNorm -> MSR -> Residual        |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      # Build RetNet backbone
      model = RetNet.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 6,
        num_heads: 4
      )

  ## Comparison

  | Mode | Time | Memory | Best For |
  |------|------|--------|----------|
  | Parallel | O(L^2) | O(L^2) | Training |
  | Recurrent | O(1) | O(1) | Inference |
  | Chunkwise | O(L) | O(C) | Long sequences |

  ## References
  - Paper: https://arxiv.org/abs/2307.08621
  - Code: https://github.com/microsoft/unilm/tree/master/retnet
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  def default_num_layers, do: 6

  @doc "Default number of retention heads"
  def default_num_heads, do: 4

  @doc "Default feedforward expansion factor"
  def default_expand_factor, do: 2

  @doc "Default dropout rate"
  def default_dropout, do: 0.0

  @doc "Epsilon for numerical stability"
  def eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a RetNet model for sequence processing.

  ## Options
    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of RetNet blocks (default: 6)
    - `:num_heads` - Number of retention heads (default: 4)
    - `:expand_factor` - FFN expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)
    - `:mode` - Computation mode: :parallel, :recurrent, :chunkwise (default: :parallel)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
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

    # Stack RetNet blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block_opts = Keyword.merge(opts, layer_idx: layer_idx)
        block = build_retnet_block(acc, block_opts)

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

  # ============================================================================
  # RetNet Block
  # ============================================================================

  @doc """
  Build a single RetNet block.

  RetNet block structure:
  1. LayerNorm -> Multi-Scale Retention -> Residual
  2. LayerNorm -> FFN -> Residual
  """
  @spec build_retnet_block(Axon.t(), keyword()) :: Axon.t()
  def build_retnet_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "retnet_block_#{layer_idx}"

    # 1. Multi-Scale Retention branch
    retention_normed = Axon.layer_norm(input, name: "#{name}_retention_norm")

    retention_out =
      build_multi_scale_retention(retention_normed, Keyword.put(opts, :name, "#{name}_msr"))

    # Residual connection
    after_retention = Axon.add(input, retention_out, name: "#{name}_retention_residual")

    # 2. Feedforward branch
    ff_normed = Axon.layer_norm(after_retention, name: "#{name}_ff_norm")
    ff_out = build_ffn(ff_normed, hidden_size, expand_factor, "#{name}_ffn")

    # Residual connection
    Axon.add(after_retention, ff_out, name: "#{name}_ff_residual")
  end

  # ============================================================================
  # Multi-Scale Retention (MSR)
  # ============================================================================

  @doc """
  Build Multi-Scale Retention layer.

  MSR uses different decay rates (gamma) per head for multi-scale modeling:
  - gamma_h = 1 - 2^(-5-h) for head h
  - SiLU gating: Y = SiLU(X*W_G) . Retention(X)*W_O
  - GroupNorm for handling different head variances
  """
  @spec build_multi_scale_retention(Axon.t(), keyword()) :: Axon.t()
  def build_multi_scale_retention(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    name = Keyword.get(opts, :name, "msr")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Gating projection for SiLU/Swish gate
    gate = Axon.dense(input, hidden_size, name: "#{name}_gate_proj")
    gate = Axon.activation(gate, :silu, name: "#{name}_gate_silu")

    # Apply retention mechanism (parallel mode for training)
    retention_out =
      Axon.layer(
        &retention_parallel/4,
        [q, k, v],
        name: "#{name}_retention",
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :retention
      )

    # Apply GroupNorm (approximated with LayerNorm per head)
    retention_normed = Axon.layer_norm(retention_out, name: "#{name}_group_norm")

    # SiLU gating: Y = gate . retention
    gated = Axon.multiply(gate, retention_normed, name: "#{name}_gated")

    # Output projection
    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

  # Parallel retention computation (for training)
  defp retention_parallel(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = reshape_for_heads(q, batch_size, seq_len, num_heads, head_dim)
    k = reshape_for_heads(k, batch_size, seq_len, num_heads, head_dim)
    v = reshape_for_heads(v, batch_size, seq_len, num_heads, head_dim)

    # Compute decay rates for each head: gamma_h = 1 - 2^(-5-h)
    gammas = compute_head_gammas(num_heads)

    # Build decay matrix D for each head
    # D[n,m] = gamma^(n-m) if n >= m, else 0
    d_matrix = build_decay_matrix(seq_len, gammas)

    # Compute retention: (Q . K^T) . D . V
    # QK^T: [batch, heads, seq, seq]
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    qk = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Apply decay mask
    # d_matrix: [heads, seq, seq] -> broadcast to [batch, heads, seq, seq]
    qk_decayed = Nx.multiply(qk, d_matrix)

    # Apply to values: [batch, heads, seq, head_dim]
    retention_out = Nx.dot(qk_decayed, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    Nx.transpose(retention_out, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  defp reshape_for_heads(tensor, batch_size, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Compute decay rates for each head
  # gamma_h = 1 - 2^(-5-h) for head h in [0, num_heads)
  defp compute_head_gammas(num_heads) do
    Enum.map(0..(num_heads - 1), fn h ->
      1.0 - :math.pow(2, -5 - h)
    end)
    |> Nx.tensor(type: :f32)
  end

  # Build decay matrix D where D[n,m] = gamma^(n-m) if n >= m, else 0
  # Returns [heads, seq, seq]
  defp build_decay_matrix(seq_len, gammas) do
    num_heads = Nx.axis_size(gammas, 0)

    # Create position indices
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})

    # Compute distances: n - m
    distances = Nx.subtract(rows, cols)

    # Causal mask: n >= m
    causal_mask = Nx.greater_equal(rows, cols)

    # Compute decay: gamma^(n-m) for each head
    # gammas: [heads], distances: [seq, seq]
    # We need: [heads, seq, seq]
    gammas_expanded = Nx.reshape(gammas, {num_heads, 1, 1})
    distances_expanded = Nx.reshape(distances, {1, seq_len, seq_len})

    # gamma^distance (clamp distance to avoid negative exponents before mask)
    distances_clamped = Nx.max(distances_expanded, 0)
    decay = Nx.pow(gammas_expanded, distances_clamped)

    # Apply causal mask - broadcast to match decay shape
    causal_broadcast = Nx.broadcast(causal_mask, {num_heads, seq_len, seq_len})
    Nx.select(causal_broadcast, decay, Nx.tensor(0.0))
  end

  # ============================================================================
  # Feedforward Network
  # ============================================================================

  @doc """
  Build a feedforward network with GeLU activation.
  """
  @spec build_ffn(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_ffn(input, hidden_size, expand_factor, name) do
    inner_size = hidden_size * expand_factor

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Recurrent Mode (for inference)
  # ============================================================================

  @doc """
  Build recurrent retention state update.

  Recurrent formulation for O(1) inference:
  - s_n = gamma * s_{n-1} + K_n^T * V_n
  - o_n = Q_n * s_n

  This is used during inference when processing one token at a time.
  """
  @spec recurrent_retention_step(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def recurrent_retention_step(q, k, v, state, gamma) do
    # q, k, v: [batch, head_dim]
    # state: [batch, head_dim, head_dim]
    # gamma: scalar

    # Update state: s_n = gamma * s_{n-1} + k^T * v
    # k^T * v: [head_dim, 1] * [1, head_dim] = [head_dim, head_dim]
    kv_outer = Nx.dot(Nx.new_axis(k, 2), Nx.new_axis(v, 1))
    new_state = Nx.add(Nx.multiply(gamma, state), kv_outer)

    # Output: o_n = q * s_n
    # q: [batch, head_dim], s: [batch, head_dim, head_dim]
    output = Nx.dot(Nx.new_axis(q, 1), new_state) |> Nx.squeeze(axes: [1])

    {output, new_state}
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a RetNet model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc """
  Calculate approximate parameter count for a RetNet model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())

    inner_size = hidden_size * expand_factor

    # MSR parameters per layer:
    # - Q, K, V projections: 3 * hidden * hidden
    # - Gate projection: hidden * hidden
    # - Output projection: hidden * hidden
    msr_params = 5 * hidden_size * hidden_size

    # FFN parameters per layer:
    # - Up projection: hidden * inner
    # - Down projection: inner * hidden
    ffn_params = hidden_size * inner_size + inner_size * hidden_size

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + num_layers * (msr_params + ffn_params)
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 4,
      expand_factor: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end

  @doc """
  Initialize retention state for recurrent inference.

  Returns a zero-initialized state tensor of shape [batch, heads, head_dim, head_dim].
  """
  @spec init_retention_state(pos_integer(), pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def init_retention_state(batch_size, num_heads, head_dim) do
    Nx.broadcast(0.0, {batch_size, num_heads, head_dim, head_dim})
  end
end
