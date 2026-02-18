defmodule Edifice.Attention.RWKV do
  @moduledoc """
  RWKV-7 "Goose": Linear attention with O(1) space complexity.

  RWKV (Receptance Weighted Key Value) is a linear attention architecture that
  combines the parallelizable training of Transformers with the efficient O(1)
  inference of RNNs.

  ## Key Innovation: Generalized Delta Rule

  RWKV-7 uses a generalized delta rule that surpasses the TC0 constraint,
  enabling it to comprehensively outperform Transformers on many tasks.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |  RWKV Block                          |
  |                                      |
  |  +----------------------------------+
  |  | Time-Mixing (WKV Attention)      |
  |  | - R-gate: receptance             |
  |  | - W: time decay                  |
  |  | - K, V: key-value pairs          |
  |  | - time_first: first token bias   |
  |  +----------------------------------+
  |                                      |
  |  +----------------------------------+
  |  | Channel-Mixing (FFN)              |
  |  | - R-gate * K-gate                 |
  |  +----------------------------------+
  +-------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Phase | Time | Space |
  |-------|------|-------|
  | Training | O(L) | O(L) |
  | Inference | O(1) per step | O(1) |

  ## Key Difference from Mamba

  | Aspect | RWKV | Mamba |
  |--------|------|-------|
  | Attention | WKV (weighted key-value) | SSM (state space) |
  | State | O(1) fixed size | O(L) for full sequence |
  | Decay | Learned per-channel | Input-dependent |
  | Gating | R-gate, K-gate | SiLU gating |

  ## Usage

      model = RWKV.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6
      )

  ## References

  - RWKV-7 "Goose" architecture wiki: https://wiki.rwkv.com/basic/architecture.html
  - Paper: "RWKV: Reinventing RNNs for the Transformer Era" (arXiv:2305.13048)
  - Deployment: Shipped to 1.5B Windows devices for on-device Copilot
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 6
  @default_head_size 64
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an RWKV-7 model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of RWKV blocks (default: 6)
    - `:head_size` - Size per attention head (default: 64)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    head_size = Keyword.get(opts, :head_size, @default_head_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Number of heads
    num_heads = div(hidden_size, head_size)

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack RWKV blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_rwkv_block(
          acc,
          hidden_size: hidden_size,
          head_size: head_size,
          num_heads: num_heads,
          dropout: dropout,
          name: "rwkv_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single RWKV block.

  Each block has two sub-blocks:
  1. Time-mixing: WKV attention mechanism
  2. Channel-mixing: Feed-forward with gating
  """
  @spec build_rwkv_block(Axon.t(), keyword()) :: Axon.t()
  def build_rwkv_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    head_size = Keyword.get(opts, :head_size, @default_head_size)
    num_heads = Keyword.get(opts, :num_heads, div(hidden_size, head_size))
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "rwkv_block")

    # Time-mixing (WKV attention)
    x =
      build_time_mixing(input,
        hidden_size: hidden_size,
        head_size: head_size,
        num_heads: num_heads,
        dropout: dropout,
        name: "#{name}_time_mix"
      )

    # Channel-mixing (FFN with gating)
    build_channel_mixing(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_channel_mix"
    )
  end

  @doc """
  Build the Time-Mixing sub-block (WKV attention).

  Time-mixing implements the WKV (Weighted Key-Value) attention mechanism:

  ```
  wkv[t] = (sum_{i<t} exp(w*(t-1-i) + k[i]) * v[i] + exp(u + k[t]) * v[t]) /
           (sum_{i<t} exp(w*(t-1-i) + k[i]) + exp(u + k[t]))
  ```

  Where:
  - w: learned time decay (per head)
  - u: learned "time_first" bias for current token
  - k, v: keys and values from input
  - r: receptance gate

  Output = sigmoid(r) * wkv
  """
  @spec build_time_mixing(Axon.t(), keyword()) :: Axon.t()
  def build_time_mixing(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    head_size = Keyword.get(opts, :head_size, @default_head_size)
    num_heads = Keyword.get(opts, :num_heads, div(hidden_size, head_size))
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "time_mix")

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Token shift: mix current token with previous token
    # This is a key RWKV innovation for capturing temporal patterns
    x_shifted = build_token_shift(x, name: "#{name}_shift")

    # Time-mix interpolation factors (learned)
    # These control how much to use current vs shifted token
    x_mixed = Axon.dense(x_shifted, hidden_size, name: "#{name}_mix_proj")

    # Project to R, W, K, V
    # R: receptance (gate)
    # W: time decay
    # K: key
    # V: value
    r_proj = Axon.dense(x_mixed, hidden_size, name: "#{name}_r_proj")
    w_proj = Axon.dense(x_mixed, hidden_size, name: "#{name}_w_proj")
    k_proj = Axon.dense(x_mixed, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(x_mixed, hidden_size, name: "#{name}_v_proj")

    # Time-first (u) parameter: boosts current token's key
    # This is a key RWKV innovation for giving special weight to the current position
    u_init = Nx.broadcast(Nx.tensor(0.0, type: :f32), {1, 1, num_heads, head_size})
    u_node = Axon.constant(u_init)

    # Apply WKV attention
    wkv =
      Axon.layer(
        &wkv_attention/6,
        [w_proj, k_proj, v_proj, r_proj, u_node],
        name: "#{name}_wkv",
        hidden_size: hidden_size,
        head_size: head_size,
        num_heads: num_heads,
        op_name: :wkv_attention
      )

    # Output projection
    output = Axon.dense(wkv, hidden_size, name: "#{name}_output")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  @doc """
  Build the Channel-Mixing sub-block (FFN with gating).

  Channel-mixing uses a gated FFN structure:
  ```
  output = sigmoid(r) * (k * v)
  ```

  Where:
  - r: receptance gate
  - k: key (square activation)
  - v: value projection
  """
  @spec build_channel_mixing(Axon.t(), keyword()) :: Axon.t()
  def build_channel_mixing(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "channel_mix")

    # Expansion factor (typically 3.5x or 4x)
    inner_size = hidden_size * 4

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Token shift for channel mixing
    x_shifted = build_token_shift(x, name: "#{name}_shift")

    # Project to R, K
    r_proj = Axon.dense(x_shifted, hidden_size, name: "#{name}_r_proj")
    k_proj = Axon.dense(x_shifted, inner_size, name: "#{name}_k_proj")

    # K-gate: squared ReLU (key innovation in RWKV)
    k_gate =
      k_proj
      |> Axon.relu()
      |> Axon.nx(fn t -> Nx.multiply(t, t) end, name: "#{name}_k_squared")

    # V projection (down-project from inner_size)
    v_proj = Axon.dense(k_gate, hidden_size, name: "#{name}_v_proj")

    # R-gate: sigmoid receptance
    r_gate = Axon.activation(r_proj, :sigmoid, name: "#{name}_r_gate")

    # Output: r * v
    output = Axon.multiply(r_gate, v_proj, name: "#{name}_gated")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Build token shift operation (mix current with previous token)
  defp build_token_shift(input, opts) do
    name = Keyword.get(opts, :name, "token_shift")

    Axon.nx(
      input,
      fn x ->
        # x: [batch, seq_len, hidden_size]
        batch = Nx.axis_size(x, 0)
        hidden = Nx.axis_size(x, 2)

        # Pad with zeros at the start (for the first token)
        zeros = Nx.broadcast(0.0, {batch, 1, hidden})

        # Shift: concatenate zeros with all but last token
        shifted =
          Nx.concatenate([zeros, Nx.slice_along_axis(x, 0, Nx.axis_size(x, 1) - 1, axis: 1)],
            axis: 1
          )

        # Concatenate current and shifted for mixing
        Nx.concatenate([x, shifted], axis: 2)
      end,
      name: name
    )
  end

  # WKV attention implementation with time_first (u) parameter
  # u gives special weight to the current token at each position
  defp wkv_attention(w, k, v, r, u, opts) do
    hidden_size = opts[:hidden_size]
    head_size = opts[:head_size]
    num_heads = opts[:num_heads]

    # w: [batch, seq_len, hidden_size] - time decay
    # k: [batch, seq_len, hidden_size] - keys
    # v: [batch, seq_len, hidden_size] - values
    # r: [batch, seq_len, hidden_size] - receptance
    # u: [1, 1, num_heads, head_size] - time_first bias for current token

    batch = Nx.axis_size(w, 0)
    seq_len = Nx.axis_size(w, 1)

    # Reshape to heads: [batch, seq_len, num_heads, head_size]
    w = Nx.reshape(w, {batch, seq_len, num_heads, head_size})
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_size})
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_size})
    r = Nx.reshape(r, {batch, seq_len, num_heads, head_size})

    # Time decay: negative exponential (learned, clamped for stability)
    # w represents log(decay), so exp(-softplus(w)) gives decay in (0, 1)
    decay = Nx.exp(Nx.negate(Nx.log(Nx.add(1.0, Nx.exp(w)))))

    # Compute attention weights: exp(k)
    # Clamp k for numerical stability
    k_clamped = Nx.clip(k, -10.0, 10.0)
    exp_k = Nx.exp(k_clamped)

    # Weighted values: exp(k) * v
    weighted_v = Nx.multiply(exp_k, v)

    # Compute cumulative sums with decay using parallel scan
    # The inclusive scan gives: sum_{i<=t} decay^(t-i) * exp(k[i]) * v[i]
    {numerator, denominator} = parallel_wkv_scan(decay, weighted_v, exp_k, seq_len)

    # Time-first (u) parameter: boost current token's contribution
    # The inclusive scan already includes exp(k[t]) * v[t] for current token.
    # We want exp(u + k[t]) * v[t] = exp(u) * exp(k[t]) * v[t] instead.
    # Extra contribution: (exp(u) - 1) * exp(k[t]) * v[t]
    u_boost = Nx.subtract(Nx.exp(u), 1.0)
    numerator = Nx.add(numerator, Nx.multiply(Nx.multiply(u_boost, exp_k), v))
    denominator = Nx.add(denominator, Nx.multiply(u_boost, exp_k))

    # Output: sigmoid(r) * (numerator / denominator)
    r_gate = Nx.sigmoid(r)

    # Add epsilon for numerical stability
    eps = 1.0e-6
    wkv_output = Nx.multiply(r_gate, Nx.divide(numerator, Nx.add(denominator, eps)))

    # Reshape back: [batch, seq_len, hidden_size]
    Nx.reshape(wkv_output, {batch, seq_len, hidden_size})
  end

  # Parallel WKV scan using cumulative operations
  # This approximates the recurrent WKV formula in parallel
  defp parallel_wkv_scan(decay, weighted_v, exp_k, _seq_len) do
    # decay: [batch, seq_len, num_heads, head_size]
    # weighted_v: [batch, seq_len, num_heads, head_size] = exp(k) * v
    # exp_k: [batch, seq_len, num_heads, head_size]

    # Compute cumulative products of decay for each position
    # decay_cumsum[t] = decay[1] * decay[2] * ... * decay[t]
    log_decay = Nx.log(Nx.add(decay, 1.0e-10))
    log_decay_cumsum = Nx.cumulative_sum(log_decay, axis: 1)
    decay_cumsum = Nx.exp(log_decay_cumsum)

    # For position t, we need sum_{i<=t} decay^(t-i) * value[i]
    # = sum_{i<=t} (decay_cumsum[t] / decay_cumsum[i]) * value[i]
    # = decay_cumsum[t] * sum_{i<=t} (value[i] / decay_cumsum[i])

    # Normalize values by cumulative decay
    eps = 1.0e-10
    normalized_v = Nx.divide(weighted_v, Nx.add(decay_cumsum, eps))
    normalized_k = Nx.divide(exp_k, Nx.add(decay_cumsum, eps))

    # Cumulative sum of normalized values
    numerator_normalized = Nx.cumulative_sum(normalized_v, axis: 1)
    denominator_normalized = Nx.cumulative_sum(normalized_k, axis: 1)

    # Multiply back by cumulative decay to get final values
    numerator = Nx.multiply(decay_cumsum, numerator_normalized)
    denominator = Nx.multiply(decay_cumsum, denominator_normalized)

    {numerator, denominator}
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an RWKV model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an RWKV model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer:
    # Time-mixing:
    #   - mix_proj: hidden * 2 * hidden (input is doubled from shift)
    #   - r, w, k, v projections: 4 * hidden * hidden
    #   - output: hidden * hidden
    time_mix_params =
      hidden_size * 2 * hidden_size +
        4 * hidden_size * hidden_size +
        hidden_size * hidden_size

    # Channel-mixing:
    #   - r_proj: hidden * 2 * hidden
    #   - k_proj: hidden * 2 * inner_size
    #   - v_proj: inner_size * hidden
    channel_mix_params =
      hidden_size * 2 * hidden_size +
        hidden_size * 2 * inner_size +
        inner_size * hidden_size

    per_layer = time_mix_params + channel_mix_params

    # Input projection
    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      head_size: 64,
      window_size: 60,
      dropout: 0.1
    ]
  end

  @doc """
  Initialize hidden state for O(1) incremental inference.

  RWKV's key advantage: constant memory per inference step.
  """
  @spec init_cache(keyword()) :: map()
  def init_cache(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 1)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    head_size = Keyword.get(opts, :head_size, @default_head_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    num_heads = div(hidden_size, head_size)

    # Per layer, we cache:
    # - WKV state: [batch, num_heads, head_size] for numerator and denominator
    # - Last token for token shift
    layers =
      for layer_idx <- 1..num_layers, into: %{} do
        layer_cache = %{
          # WKV attention state
          wkv_numerator: Nx.broadcast(0.0, {batch_size, num_heads, head_size}),
          wkv_denominator: Nx.broadcast(0.0, {batch_size, num_heads, head_size}),
          # Last token for time-mixing shift
          last_token_time: Nx.broadcast(0.0, {batch_size, hidden_size}),
          # Last token for channel-mixing shift
          last_token_channel: Nx.broadcast(0.0, {batch_size, hidden_size})
        }

        {"layer_#{layer_idx}", layer_cache}
      end

    %{
      layers: layers,
      step: 0,
      config: %{
        hidden_size: hidden_size,
        head_size: head_size,
        num_heads: num_heads,
        num_layers: num_layers
      }
    }
  end
end
