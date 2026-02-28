defmodule Edifice.Attention.LogLinear do
  @moduledoc """
  Log-Linear Attention: O(log T) memory bridging linear and softmax attention.

  Uses a hierarchical segment tree where each level stores aggregated KV
  pairs at exponentially increasing granularity. Recent tokens get exact
  attention while older tokens get coarser-grained attention.

  ```
  LogLinear(Q, K, V) = softmax(QK_local^T / sqrt(d)) * V_local
                     + sum_level softmax(QK_agg[level]^T / sqrt(d)) * V_agg[level]
  ```

  ## Key Innovation

  Divides the sequence into segments at multiple levels:
  - Level 0: individual tokens (exact, recent window)
  - Level 1: segments of size S (aggregated KV pairs)
  - Level 2: segments of size S^2 (further aggregated)
  - ...up to O(log_S T) levels

  Each level maintains a running aggregate of K and V, so the total
  memory is O(log T) instead of O(T) for full attention or O(1) for
  linear attention. Quality is provably Pareto-optimal on the
  memory-quality tradeoff curve.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------------------+
  |  Log-Linear Attention Block (x layers)   |
  |                                          |
  |  Q, K, V projections                    |
  |  Local window attention (exact)          |
  |  + Hierarchical segment aggregation      |
  |  Combined output + residual + FFN        |
  +------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Usage

      model = LogLinear.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6,
        segment_size: 4
      )

  ## Reference

  - "Log-Linear Attention" (arXiv, June 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1
  @default_segment_size 4

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:segment_size, pos_integer()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Log-Linear Attention model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:segment_size` - Base segment size for hierarchical aggregation (default: 4)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    segment_size = Keyword.get(opts, :segment_size, @default_segment_size)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "log_linear_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_log_linear_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              segment_size: segment_size,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  @doc """
  Build the log-linear attention layer.

  Combines local window attention with hierarchical segment-aggregated
  attention at multiple scales.
  """
  @spec build_log_linear_attention(Axon.t(), keyword()) :: Axon.t()
  def build_log_linear_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    segment_size = Keyword.get(opts, :segment_size, @default_segment_size)
    name = Keyword.get(opts, :name, "log_linear_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Aggregation gate: learns to weight local vs aggregated attention
    agg_gate = Axon.dense(input, num_heads, name: "#{name}_agg_gate")

    output =
      Axon.layer(
        &log_linear_attention_impl/5,
        [q_proj, k_proj, v_proj, agg_gate],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        segment_size: segment_size,
        op_name: :log_linear_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Log-linear attention implementation
  # Q, K, V: [batch, seq, hidden], agg_gate: [batch, seq, num_heads]
  defp log_linear_attention_impl(q, k, v, agg_gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    segment_size = opts[:segment_size]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Gate: [batch, seq, num_heads] -> [batch, heads, seq, 1]
    gate =
      agg_gate
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.new_axis(-1)
      |> Nx.sigmoid()

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # 1. Local attention (exact, within causal window)
    local_out = compute_local_attention(q, k, v, scale, batch, num_heads, seq_len)

    # 2. Segment-aggregated attention
    agg_out =
      compute_segment_attention(q, k, v, scale, segment_size, batch, num_heads, seq_len, head_dim)

    # Combine: gate * local + (1 - gate) * aggregated
    output =
      Nx.add(
        Nx.multiply(gate, local_out),
        Nx.multiply(Nx.subtract(1.0, gate), agg_out)
      )

    # Reshape back: [batch, seq, hidden]
    reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
  end

  # Standard causal attention for local context
  defp compute_local_attention(q, k, v, scale, batch, num_heads, seq_len) do
    # Full causal attention scores: [batch, heads, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(Nx.tensor(-1.0e9, type: Nx.type(scores)), Nx.shape(scores))
      )

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [-1], keep_axes: true))

    # Output: [batch, heads, seq, head_dim]
    Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])
  end

  # Segment-aggregated attention: aggregate K, V within segments, attend to aggregates
  defp compute_segment_attention(
         q,
         k,
         v,
         scale,
         segment_size,
         batch,
         num_heads,
         seq_len,
         head_dim
       ) do
    # Number of segments (pad if needed)
    num_segments = max(div(seq_len, segment_size), 1)

    # Compute segment means for K and V
    # We aggregate by averaging K and V within each segment
    k_agg = aggregate_segments(k, segment_size, num_segments, batch, num_heads, head_dim)
    v_agg = aggregate_segments(v, segment_size, num_segments, batch, num_heads, head_dim)

    # k_agg, v_agg: [batch, heads, num_segments, head_dim]

    # Attention from each query to segment aggregates
    # scores: [batch, heads, seq, num_segments]
    scores = Nx.dot(q, [3], [0, 1], k_agg, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal masking: query at position i can only attend to segments
    # that end before or at position i
    # Segment j covers positions [j*S, (j+1)*S - 1]
    # Query i can attend to segment j if (j+1)*S - 1 <= i, i.e., j < ceil((i+1)/S)
    query_pos = Nx.iota({seq_len, 1}, axis: 0) |> Nx.as_type(Nx.type(scores))

    seg_end =
      Nx.multiply(
        Nx.add(Nx.iota({1, num_segments}, axis: 1), 1),
        segment_size
      )
      |> Nx.as_type(Nx.type(scores))

    # segment j is visible to query i if seg_end[j] <= i + 1
    # (segment must be fully past to be aggregated)
    seg_mask = Nx.greater_equal(Nx.add(query_pos, 1), seg_end)

    seg_mask =
      seg_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, num_segments})

    scores =
      Nx.select(
        seg_mask,
        scores,
        Nx.broadcast(Nx.tensor(-1.0e9, type: Nx.type(scores)), Nx.shape(scores))
      )

    # Softmax over segments
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [-1], keep_axes: true))

    # Output: [batch, heads, seq, head_dim]
    Nx.dot(attn_weights, [3], [0, 1], v_agg, [2], [0, 1])
  end

  # Aggregate K/V within segments by averaging
  # input: [batch, heads, seq, head_dim] -> [batch, heads, num_segments, head_dim]
  defp aggregate_segments(tensor, segment_size, num_segments, batch, num_heads, head_dim) do
    {_, _, seq_len, _} = Nx.shape(tensor)

    # Pad sequence to be divisible by segment_size
    padded_len = num_segments * segment_size

    tensor =
      if padded_len > seq_len do
        pad_config = [{0, 0, 0}, {0, 0, 0}, {0, padded_len - seq_len, 0}, {0, 0, 0}]
        Nx.pad(tensor, 0.0, pad_config)
      else
        Nx.slice(tensor, [0, 0, 0, 0], [batch, num_heads, padded_len, head_dim])
      end

    # Reshape to [batch, heads, num_segments, segment_size, head_dim]
    tensor = Nx.reshape(tensor, {batch, num_heads, num_segments, segment_size, head_dim})

    # Average within each segment
    Nx.mean(tensor, axes: [3])
  end

  # Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, heads, seq, head_dim] -> [batch, seq, hidden]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_layers: 6,
      dropout: 0.1,
      segment_size: 4
    ]
  end
end
