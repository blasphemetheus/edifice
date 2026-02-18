defmodule Edifice.Attention.Griffin do
  @moduledoc """
  Griffin: Hybrid RG-LRU + Local Attention Architecture.

  Implements the Griffin architecture from "Griffin: Mixing Gated Linear Recurrences
  with Local Attention for Efficient Language Models" (De et al., 2024).

  ## Key Innovation: Real-Gated Linear Recurrent Unit (RG-LRU)

  Unlike Mamba's selective SSM, Griffin uses a simpler gated recurrence:

  ```
  r_t = sigma(W_a x_t + b_a)           # Recurrence gate
  i_t = sigma(W_x x_t + b_x)           # Input gate
  a_t = a^(c * r_t)                    # Gated decay (a = sigma(Lambda), c = 8)
  h_t = a_t . h_{t-1} + sqrt(1-a_t^2) . (i_t . x_t)
  ```

  The `sqrt(1-a_t^2)` term ensures hidden state norm is preserved (like a rotation),
  enabling stable training at long sequences.

  ## Architecture Pattern

  Griffin alternates between RG-LRU and local attention blocks:
  - Pattern: 2 RG-LRU blocks -> 1 Local Attention block -> repeat
  - Local attention uses sliding window (default 1024 tokens)
  - Each block: RMSNorm -> temporal mix -> residual -> RMSNorm -> gated MLP -> residual

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |       Griffin Block (RG-LRU)        |
  |  RMSNorm -> RG-LRU -> Residual      |
  |  RMSNorm -> Gated MLP -> Residual   |
  +-------------------------------------+
        | (repeat 2x)
        v
  +-------------------------------------+
  |    Griffin Block (Local Attn)       |
  |  RMSNorm -> LocalAttn -> Residual   |
  |  RMSNorm -> Gated MLP -> Residual   |
  +-------------------------------------+
        |
        v (repeat pattern)
  ```

  ## Usage

      # Build Griffin backbone
      model = Griffin.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        window_size: 60,
        local_attn_window: 32
      )

  ## Compared to Mamba

  | Aspect | Mamba | Griffin |
  |--------|-------|---------|
  | Recurrence | SSM with A,B,C,Delta | Simple gated RNN |
  | Parallel scan | Required | Optional (can be sequential) |
  | Long-range | Pure recurrence | Hybrid with local attention |
  | Parameters | Higher (SSM projections) | Lower (just gates) |

  ## References
  - Paper: https://arxiv.org/abs/2402.19427
  - Hawk: RG-LRU only variant (no local attention)
  """

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:local_attn_window, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}
          | {:use_local_attention, boolean()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers (should be divisible by 3 for 2:1 pattern)"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 6

  @doc "Default MLP expansion factor"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 3

  @doc "Default local attention window size"
  @spec default_local_attn_window() :: pos_integer()
  def default_local_attn_window, do: 32

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  @doc "RG-LRU gate constant c (controls decay rate range)"
  @spec rg_lru_c() :: float()
  def rg_lru_c, do: 8.0

  @doc "Number of attention heads for local attention"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: 4

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Griffin model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of Griffin blocks (default: 6, divisible by 3)
    - `:expand_factor` - MLP expansion factor (default: 3)
    - `:local_attn_window` - Local attention window size (default: 32)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)
    - `:use_local_attention` - Include local attention blocks (default: true)
      Set to false for Hawk variant (pure RG-LRU)

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
    use_local_attention = Keyword.get(opts, :use_local_attention, true)

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

    # Stack Griffin blocks with 2:1 pattern (RG-LRU : Local Attention)
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Determine block type: every 3rd block is local attention (if enabled)
        block_type =
          if use_local_attention and rem(layer_idx, 3) == 0 do
            :local_attention
          else
            :rg_lru
          end

        block_opts = Keyword.merge(opts, layer_idx: layer_idx, block_type: block_type)
        block = build_griffin_block(acc, block_opts)

        # Residual connection is inside the block, but add dropout between blocks
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

  @doc """
  Build a Hawk model (Griffin without local attention).

  This is a pure RG-LRU model, simpler and faster than Griffin.
  """
  @spec build_hawk(keyword()) :: Axon.t()
  def build_hawk(opts \\ []) do
    build(Keyword.put(opts, :use_local_attention, false))
  end

  # ============================================================================
  # Griffin Block
  # ============================================================================

  @doc """
  Build a single Griffin block.

  Griffin block structure:
  1. RMSNorm -> Temporal mixing (RG-LRU or Local Attention) -> Residual
  2. RMSNorm -> Gated MLP -> Residual
  """
  @spec build_griffin_block(Axon.t(), keyword()) :: Axon.t()
  def build_griffin_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    block_type = Keyword.get(opts, :block_type, :rg_lru)
    name = "griffin_block_#{layer_idx}"

    # 1. Temporal mixing branch
    temporal_normed = Axon.layer_norm(input, name: "#{name}_temporal_norm")

    temporal_out =
      case block_type do
        :rg_lru ->
          build_rg_lru_layer(temporal_normed, Keyword.put(opts, :name, "#{name}_rg_lru"))

        :local_attention ->
          build_local_attention(temporal_normed, Keyword.put(opts, :name, "#{name}_local_attn"))
      end

    # Residual connection for temporal mixing
    after_temporal = Axon.add(input, temporal_out, name: "#{name}_temporal_residual")

    # 2. Gated MLP branch
    mlp_normed = Axon.layer_norm(after_temporal, name: "#{name}_mlp_norm")
    mlp_out = build_gated_mlp(mlp_normed, hidden_size, expand_factor, "#{name}_gated_mlp")

    # Residual connection for MLP
    Axon.add(after_temporal, mlp_out, name: "#{name}_mlp_residual")
  end

  # ============================================================================
  # RG-LRU (Real-Gated Linear Recurrent Unit)
  # ============================================================================

  @doc """
  Build the Real-Gated Linear Recurrent Unit layer.

  RG-LRU equations:
  - r_t = sigma(W_a x_t + b_a)           # Recurrence gate
  - i_t = sigma(W_x x_t + b_x)           # Input gate
  - a_t = a^(c * r_t)                    # Gated decay (a = sigma(Lambda), c = 8)
  - h_t = a_t . h_{t-1} + sqrt(1-a_t^2) . (i_t . x_t)

  The sqrt(1-a_t^2) normalization ensures the recurrence preserves
  hidden state magnitude (like a complex rotation).
  """
  @spec build_rg_lru_layer(Axon.t(), keyword()) :: Axon.t()
  def build_rg_lru_layer(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    name = Keyword.get(opts, :name, "rg_lru")

    # Recurrence gate: r_t = sigma(W_a x_t + b_a)
    recurrence_gate =
      input
      |> Axon.dense(hidden_size, name: "#{name}_recurrence_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_recurrence_gate")

    # Input gate: i_t = sigma(W_x x_t + b_x)
    input_gate =
      input
      |> Axon.dense(hidden_size, name: "#{name}_input_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_input_gate")

    # Gated input: i_t . x_t (after linear projection)
    x_proj = Axon.dense(input, hidden_size, name: "#{name}_x_proj")
    gated_input = Axon.multiply(input_gate, x_proj, name: "#{name}_gated_input")

    # Learnable lambda parameter for per-dimension decay: a = sigmoid(lambda)
    # Initialized so that a^c is uniformly in [0.9, 0.999]
    lambda_init = init_lambda({hidden_size})
    lambda_node = Axon.constant(lambda_init)

    # Apply the RG-LRU recurrence using Axon.layer for the scan
    Axon.layer(
      &rg_lru_layer_impl/4,
      [gated_input, recurrence_gate, lambda_node],
      name: "#{name}_scan",
      hidden_size: hidden_size,
      op_name: :rg_lru_scan
    )
  end

  # Axon.layer wrapper that unpacks opts and delegates to scan
  defp rg_lru_layer_impl(x_gated, recurrence_gate, lambda, opts) do
    rg_lru_scan_impl(x_gated, recurrence_gate, lambda, opts[:hidden_size])
  end

  # RG-LRU scan implementation
  # h_t = a_t . h_{t-1} + sqrt(1-a_t^2) . x_gated_t
  defp rg_lru_scan_impl(x_gated, recurrence_gate, lambda, _hidden_size) do
    c = rg_lru_c()

    # a = sigmoid(lambda) — learnable per-dimension base decay
    # lambda is initialized via init_lambda so that a^c is in [0.9, 0.999]
    a_base = Nx.sigmoid(lambda)

    # a_t = a^(c * r_t), gated decay
    # recurrence_gate: [batch, seq_len, hidden_size]
    a_t = Nx.pow(a_base, Nx.multiply(c, recurrence_gate))

    # sqrt(1 - a_t^2) for input scaling (preserves norm like rotation)
    # Add small epsilon to avoid sqrt(0) issues
    input_scale = Nx.sqrt(Nx.add(Nx.subtract(1.0, Nx.pow(a_t, 2)), 1.0e-8))

    # Scaled input contribution
    x_scaled = Nx.multiply(input_scale, x_gated)

    # Run the recurrence: h_t = a_t . h_{t-1} + x_scaled_t
    rg_lru_sequential_scan(a_t, x_scaled)
  end

  # Sequential scan for RG-LRU
  # This is simpler than Mamba's scan since a_t is directly the decay factor
  defp rg_lru_sequential_scan(a, x_scaled) do
    # a, x_scaled: [batch, seq_len, hidden_size]
    batch_size = Nx.axis_size(a, 0)
    seq_len = Nx.axis_size(a, 1)
    hidden_size = Nx.axis_size(a, 2)

    # Initialize h[-1] = 0
    h_init = Nx.broadcast(0.0, {batch_size, 1, hidden_size})

    # Sequential recurrence
    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)
        x_t = Nx.slice_along_axis(x_scaled, t, 1, axis: 1)

        # h_t = a_t * h_{t-1} + x_t
        h_t = Nx.add(Nx.multiply(a_t, h_prev), x_t)
        {h_t, [Nx.squeeze(h_t, axes: [1]) | acc]}
      end)

    # Stack: [batch, seq_len, hidden_size]
    h_list
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Local Attention
  # ============================================================================

  @doc """
  Build a local (sliding window) attention layer.

  Uses windowed attention for computational efficiency while
  still capturing short-range dependencies.
  """
  @spec build_local_attention(Axon.t(), keyword()) :: Axon.t()
  def build_local_attention(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    local_window = Keyword.get(opts, :local_attn_window, default_local_attn_window())
    name = Keyword.get(opts, :name, "local_attention")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Apply local attention
    attn_out =
      Axon.layer(
        &local_attention_impl/4,
        [q, k, v],
        name: name,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        local_window: local_window,
        op_name: :local_attention
      )

    # Output projection
    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # Local attention implementation with sliding window
  defp local_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    local_window = opts[:local_window]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    q = reshape_for_heads(q, batch_size, seq_len, num_heads, head_dim)
    k = reshape_for_heads(k, batch_size, seq_len, num_heads, head_dim)
    v = reshape_for_heads(v, batch_size, seq_len, num_heads, head_dim)

    # Compute attention with local window mask
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Q @ K^T: [batch, heads, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Create local attention mask and apply it
    # Mask: 1 where we attend, 0 where we mask out
    mask = create_local_attention_mask(seq_len, local_window)

    # Broadcast mask to [batch, heads, seq, seq] and apply
    # Where mask is 0, set scores to -inf
    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
    # Use where: mask == 1 ? scores : -inf
    scores =
      Nx.select(
        Nx.broadcast(mask, Nx.shape(scores)),
        scores,
        Nx.broadcast(neg_inf, Nx.shape(scores))
      )

    # Softmax (stable version)
    max_scores = Nx.reduce_max(scores, axes: [3], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [3], keep_axes: true))

    # Attention output: [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    Nx.transpose(attn_out, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  defp reshape_for_heads(tensor, batch_size, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp create_local_attention_mask(seq_len, window_size) do
    # Create position indices
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})

    # Causal mask (lower triangular): rows >= cols
    causal_mask = Nx.greater_equal(rows, cols)

    # Window mask (within window_size positions): (rows - cols) < window_size
    distance = Nx.subtract(rows, cols)
    window_mask = Nx.less(distance, window_size)

    # Combined: causal AND within window
    # Returns boolean tensor [seq_len, seq_len]
    combined = Nx.logical_and(causal_mask, window_mask)

    # Reshape to [1, 1, seq_len, seq_len] for broadcasting
    Nx.reshape(combined, {1, 1, seq_len, seq_len})
  end

  # ============================================================================
  # Gated MLP
  # ============================================================================

  @doc """
  Build a Gated MLP layer (used in Griffin blocks).

  Structure: Linear -> split -> (GeLU, Identity) -> multiply -> Linear
  """
  @spec build_gated_mlp(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_gated_mlp(input, hidden_size, expand_factor, name) do
    inner_size = hidden_size * expand_factor

    # Project to 2x inner size for gating
    up_proj = Axon.dense(input, inner_size * 2, name: "#{name}_up_proj")

    # Split into gate and value branches
    gate_branch =
      Axon.nx(
        up_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
        end,
        name: "#{name}_gate_split"
      )
      |> Axon.activation(:gelu, name: "#{name}_gate_gelu")

    value_branch =
      Axon.nx(
        up_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
        end,
        name: "#{name}_value_split"
      )

    # Gated multiplication
    gated = Axon.multiply(gate_branch, value_branch, name: "#{name}_gated")

    # Project back to hidden size
    Axon.dense(gated, hidden_size, name: "#{name}_down_proj")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Griffin model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc """
  Calculate approximate parameter count for a Griffin model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    use_local_attention = Keyword.get(opts, :use_local_attention, true)

    inner_size = hidden_size * expand_factor

    # RG-LRU block parameters:
    # - Recurrence projection: hidden * hidden
    # - Input projection: hidden * hidden
    # - X projection: hidden * hidden
    # - Lambda: hidden
    rg_lru_params = hidden_size * hidden_size * 3 + hidden_size

    # Local attention parameters:
    # - Q, K, V projections: 3 * hidden * hidden
    # - Output projection: hidden * hidden
    local_attn_params = 4 * hidden_size * hidden_size

    # Gated MLP parameters:
    # - Up projection: hidden * (2 * inner)
    # - Down projection: inner * hidden
    mlp_params = hidden_size * (2 * inner_size) + inner_size * hidden_size

    # Count layers by type
    num_rg_lru =
      if use_local_attention do
        # 2:1 pattern
        num_layers - div(num_layers, 3)
      else
        num_layers
      end

    num_local_attn =
      if use_local_attention do
        div(num_layers, 3)
      else
        0
      end

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj +
      num_rg_lru * (rg_lru_params + mlp_params) +
      num_local_attn * (local_attn_params + mlp_params)
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      expand_factor: 3,
      local_attn_window: 32,
      num_heads: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end

  @doc """
  Initialize the lambda parameter for RG-LRU.

  Lambda is initialized so that a^c is uniformly distributed in [0.9, 0.999].
  Since a = sigmoid(lambda) and a^c should be in [0.9, 0.999]:
  - a_min = 0.9^(1/c), a_max = 0.999^(1/c)
  - lambda = logit(uniform(a_min, a_max))
  """
  @spec init_lambda(tuple()) :: Nx.Tensor.t()
  def init_lambda(shape) do
    c = rg_lru_c()

    # Target range for a^c
    a_c_min = 0.9
    a_c_max = 0.999

    # Corresponding range for a
    a_min = :math.pow(a_c_min, 1.0 / c)
    a_max = :math.pow(a_c_max, 1.0 / c)

    # Sample uniform in [a_min, a_max]
    key = Nx.Random.key(System.system_time(:nanosecond))
    {uniform, _} = Nx.Random.uniform(key, a_min, a_max, shape: shape)

    # Convert to lambda via inverse sigmoid (logit)
    # logit(a) = log(a / (1 - a))
    Nx.log(Nx.divide(uniform, Nx.subtract(1.0, uniform)))
  end
end
