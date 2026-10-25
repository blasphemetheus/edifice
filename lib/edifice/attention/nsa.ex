defmodule Edifice.Attention.NSA do
  @moduledoc """
  NSA-inspired three-path sparse attention.

  This portable variant uses non-overlapping average pooling, per-head routing,
  and three learned scalar softmax gates. It is not a paper-exact NSA model:
  learned overlapping compression, grouped-head selection and query-dependent
  sigmoid gates are not implemented. Existing parameter names/shapes are retained,
  but corrected causal routing changes outputs from older checkpoints.

  Combines compressed context, selected-block retrieval, and local attention.
  The selected and local paths gather tokens before scoring; the compressed
  path retains O(T² / compression_ratio) scores. No fused sparse kernel is used.

  ## Key Innovation: Hardware-Aligned Sparse Attention

  Instead of standard full quadratic attention, NSA uses three complementary
  sparse attention patterns that can be computed efficiently on modern hardware:

  1. **Compressed Tokens**: Global context via pooled/compressed sequences
  2. **Top-k Blocks**: Fine-grained retrieval of most relevant key-value blocks
  3. **Sliding Window**: Local attention for recent context

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +----------------------------------------+
  |        Native Sparse Attention          |
  |                                         |
  |  Q, K, V = Linear(input)                |
  |                                         |
  |  +-----------+  +------------+  +------+|
  |  | Compress  |  | Top-k      |  | Slide||
  |  | (global)  |  | Blocks     |  | Wind ||
  |  |           |  | (retrieval)|  |(local)||
  |  +-----------+  +------------+  +------+|
  |        |              |            |    |
  |        v              v            v    |
  |      attn_c        attn_b       attn_w  |
  |        |              |            |    |
  |        +------+-------+-----+------+    |
  |               |             |           |
  |               v             v           |
  |         gate_weights (learnable)        |
  |               |                         |
  |               v                         |
  |      weighted_sum(attn_c, attn_b, attn_w)|
  +----------------------------------------+
        |
        v
  [batch, seq_len, embed_dim] or [batch, hidden_size]
  ```

  ## Three Paths

  ### 1. Compressed Tokens (Global Context)
  Each original query attends to completed average-pooled K/V blocks only.
  This branch uses O(n²/r) scores; early queries with no completed blocks return zero.

  ### 2. Top-k Blocks (Fine-grained Retrieval)
  - Divide K/V into blocks of block_size
  - Aggregate causal compressed-attention probabilities into block scores
  - Select up to num_selected_blocks blocks (including the current block)
  - Compute attention within selected blocks

  ### 3. Sliding Window (Local)
  Standard local attention over the last window_size tokens for recent context.

  ## Usage

      model = NSA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        head_dim: 32,
        window_size: 64,
        block_size: 16,
        num_selected_blocks: 8,
        compression_ratio: 4,
        num_layers: 4
      )

  ## References

  - Paper: "Native Sparse Attention: Hardware-Aligned and Natively Trainable"
  - Authors: DeepSeek-AI (2025)
  - https://arxiv.org/abs/2502.11089
  """

  alias Edifice.Attention.SparseAttention, as: Sparse
  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 8
  @default_head_dim 32
  @default_window_size 64
  @default_block_size 16
  @default_num_selected_blocks 8
  @default_compression_ratio 4
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an NSA model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:head_dim` - Dimension per head (default: 32)
    - `:window_size` - Sliding window size for local attention (default: 64)
    - `:block_size` - Block size for top-k selection (default: 16)
    - `:num_selected_blocks` - Number of blocks to select per query (default: 8)
    - `:compression_ratio` - Compression ratio for global path (default: 4)
    - `:num_layers` - Number of NSA blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 256)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:block_size, pos_integer()}
          | {:compression_ratio, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_selected_blocks, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, 256)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        seq_len: seq_len,
        block_builder: fn input, block_opts ->
          layer_idx = Keyword.get(block_opts, :layer_idx, 1)

          build_nsa_block(input,
            hidden_size: hidden_size,
            num_heads: Keyword.get(opts, :num_heads, @default_num_heads),
            head_dim: Keyword.get(opts, :head_dim, @default_head_dim),
            window_size: Keyword.get(opts, :window_size, @default_window_size),
            block_size: Keyword.get(opts, :block_size, @default_block_size),
            num_selected_blocks:
              Keyword.get(opts, :num_selected_blocks, @default_num_selected_blocks),
            compression_ratio: Keyword.get(opts, :compression_ratio, @default_compression_ratio),
            dropout: dropout,
            name: "nsa_block_#{layer_idx}"
          )
        end
      )
    )
  end

  @doc """
  Build a single NSA transformer block.
  """
  @spec build_nsa_block(Axon.t(), keyword()) :: Axon.t()
  def build_nsa_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "nsa_block")

    # 1. Attention sublayer: norm -> NSA -> residual
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out = build_nsa_attention(attn_normed, opts)

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer: norm -> FFN -> residual
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the NSA attention layer with three parallel sparse paths.
  """
  @spec build_nsa_attention(Axon.t(), keyword()) :: Axon.t()
  def build_nsa_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    num_selected_blocks = Keyword.get(opts, :num_selected_blocks, @default_num_selected_blocks)
    compression_ratio = Keyword.get(opts, :compression_ratio, @default_compression_ratio)
    name = Keyword.get(opts, :name, "nsa_attn")

    Sparse.validate_positive!(
      num_heads: num_heads,
      head_dim: head_dim,
      window_size: window_size,
      block_size: block_size,
      num_selected_blocks: num_selected_blocks,
      compression_ratio: compression_ratio
    )

    qkv_dim = num_heads * head_dim

    # Q, K, V projections
    q_proj = Axon.dense(input, qkv_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, qkv_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, qkv_dim, name: "#{name}_v_proj")

    # Learnable mixing weights for the three paths (will be softmaxed)
    gate_weights = Axon.param("#{name}_gate_weights", {3}, initializer: :zeros)

    # Compute NSA with three paths
    output =
      Axon.layer(
        &nsa_attention_impl/5,
        [q_proj, k_proj, v_proj, gate_weights],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        block_size: block_size,
        num_selected_blocks: num_selected_blocks,
        compression_ratio: compression_ratio,
        op_name: :nsa_attention
      )

    # Output projection
    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # NSA attention implementation with three parallel paths
  defp nsa_attention_impl(q, k, v, gate_weights, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    output = attention(q, k, v, gate_weights, opts)
    reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
  end

  @doc false
  def attention(q, k, v, gate_weights, opts) do
    {attn_compressed, probabilities} = Sparse.compressed(q, k, v, opts[:compression_ratio])

    scores =
      Sparse.compression_scores(
        probabilities,
        Nx.axis_size(q, 2),
        opts[:compression_ratio],
        opts[:block_size]
      )

    selected = Sparse.select_blocks(scores, opts[:block_size], opts[:num_selected_blocks], true)
    attn_blocks = Sparse.selected(q, k, v, selected, opts[:block_size], true)
    attn_window = Sparse.window(q, k, v, opts[:window_size])
    gates = Nx.exp(Nx.subtract(gate_weights, Nx.reduce_max(gate_weights)))
    gates = Nx.divide(gates, Nx.sum(gates))
    gate_c = Nx.slice(gates, [0], [1]) |> Nx.squeeze()
    gate_b = Nx.slice(gates, [1], [1]) |> Nx.squeeze()
    gate_w = Nx.slice(gates, [2], [1]) |> Nx.squeeze()

    # Weighted sum
    output =
      Nx.add(
        Nx.add(
          Nx.multiply(attn_compressed, gate_c),
          Nx.multiply(attn_blocks, gate_b)
        ),
        Nx.multiply(attn_window, gate_w)
      )

    output
  end

  # Reshape [batch, seq, num_heads * dim] -> [batch, num_heads, seq, dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, num_heads, seq, dim] -> [batch, seq, num_heads * dim]
  defp reshape_from_heads(x, batch, seq_len, num_heads, dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an NSA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an NSA model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    qkv_dim = num_heads * head_dim
    inner_size = hidden_size * 4

    # Per layer:
    # Attention: Q, K, V projections + output projection + gate weights
    attn_params = hidden_size * qkv_dim * 3 + qkv_dim * hidden_size + 3

    # FFN: up + down
    ffn_params = hidden_size * inner_size + inner_size * hidden_size

    per_layer = attn_params + ffn_params

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 8,
      head_dim: 32,
      window_size: 64,
      block_size: 16,
      num_selected_blocks: 8,
      compression_ratio: 4,
      num_layers: 4,
      dropout: 0.1
    ]
  end
end
