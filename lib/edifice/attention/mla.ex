defmodule Edifice.Attention.MLA do
  @moduledoc """
  Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.

  MLA compresses key-value representations into low-rank latent vectors,
  dramatically reducing KV cache memory while maintaining attention quality.
  It also uses decoupled Rotary Position Embedding (RoPE) to keep position
  information separate from compressed content.

  ## Key Innovations

  - **KV compression**: Instead of caching full K,V per head, compress to a
    low-rank latent `c_KV` and reconstruct K,V on-the-fly during attention
  - **Q compression**: Query is also compressed through a low-rank bottleneck
  - **Decoupled RoPE**: Position information is encoded via separate RoPE
    dimensions that are concatenated with content dimensions, not mixed

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +--------------------------+
  | MLA Block x N            |
  |  LayerNorm               |
  |  MLA Attention:          |
  |   h -> W_DKV -> c_KV     |  (KV latent)
  |   c_KV -> W_UK -> K_c    |  (content keys)
  |   c_KV -> W_UV -> V      |  (values)
  |   h -> W_DQ -> c_Q       |  (Q latent)
  |   c_Q -> W_UQ -> Q_c     |  (content queries)
  |   c_Q -> W_QR -> RoPE    |  (query rope)
  |   h -> W_KR -> RoPE      |  (key rope, shared)
  |   Q = [Q_c ; Q_r]        |
  |   K = [K_c ; K_r]        |
  |   score = softmax(QK^T/s) |
  |  Residual                |
  |  LayerNorm -> FFN        |
  |  Residual                |
  +--------------------------+
        |
        v
  [batch, hidden_size]       (last timestep)
  ```

  ## Usage

      model = MLA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        kv_latent_dim: 64,
        num_layers: 4
      )

  ## References

  - "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
    Language Model" (DeepSeek-AI, 2024)
  - arXiv: https://arxiv.org/abs/2405.04434
  """

  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 4
  @default_head_dim 64
  @default_rope_dim 32
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an MLA model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per head for content (default: 64)
    - `:kv_latent_dim` - Compressed KV latent dimension (default: hidden_size / 4)
    - `:q_latent_dim` - Compressed Q latent dimension (default: hidden_size * 3 / 4)
    - `:rope_dim` - Decoupled RoPE dimension per head (default: 32)
    - `:num_layers` - Number of MLA blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 60)
    - `:window_size` - Alias for seq_len (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:kv_latent_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:q_latent_dim, pos_integer()}
          | {:rope_dim, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    kv_latent_dim = Keyword.get(opts, :kv_latent_dim, div(hidden_size, 4))
    q_latent_dim = Keyword.get(opts, :q_latent_dim, div(hidden_size * 3, 4))
    rope_dim = Keyword.get(opts, :rope_dim, @default_rope_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      embed_dim: Keyword.fetch!(opts, :embed_dim),
      hidden_size: hidden_size,
      num_layers: Keyword.get(opts, :num_layers, @default_num_layers),
      seq_len: Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60)),
      dropout: dropout,
      block_builder: fn input, block_opts ->
        layer_idx = Keyword.get(block_opts, :layer_idx, 1)

        build_mla_block(input,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          kv_latent_dim: kv_latent_dim,
          q_latent_dim: q_latent_dim,
          rope_dim: rope_dim,
          dropout: dropout,
          name: "mla_block_#{layer_idx}"
        )
      end
    )
  end

  @doc """
  Build a single MLA transformer block.
  """
  @spec build_mla_block(Axon.t(), keyword()) :: Axon.t()
  def build_mla_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    kv_latent_dim = Keyword.get(opts, :kv_latent_dim, div(hidden_size, 4))
    q_latent_dim = Keyword.get(opts, :q_latent_dim, div(hidden_size * 3, 4))
    rope_dim = Keyword.get(opts, :rope_dim, @default_rope_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "mla_block")

    # 1. Attention sublayer: norm -> MLA attention -> residual
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out =
      build_mla_attention(attn_normed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        kv_latent_dim: kv_latent_dim,
        q_latent_dim: q_latent_dim,
        rope_dim: rope_dim,
        name: "#{name}_attn"
      )

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

  # Build the MLA attention mechanism with low-rank KV compression and decoupled RoPE
  defp build_mla_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    kv_latent_dim = Keyword.fetch!(opts, :kv_latent_dim)
    q_latent_dim = Keyword.fetch!(opts, :q_latent_dim)
    rope_dim = Keyword.fetch!(opts, :rope_dim)
    name = Keyword.get(opts, :name, "mla_attn")

    # KV compression: h -> W_DKV -> c_KV (low-rank latent)
    c_kv = Axon.dense(input, kv_latent_dim, name: "#{name}_kv_down")

    # Reconstruct K content and V from latent
    # K_content: [batch, seq, num_heads * head_dim]
    k_content = Axon.dense(c_kv, num_heads * head_dim, name: "#{name}_k_up")
    # V: [batch, seq, num_heads * head_dim]
    v = Axon.dense(c_kv, num_heads * head_dim, name: "#{name}_v_up")

    # Q compression: h -> W_DQ -> c_Q (low-rank latent)
    c_q = Axon.dense(input, q_latent_dim, name: "#{name}_q_down")
    # Q_content: [batch, seq, num_heads * head_dim]
    q_content = Axon.dense(c_q, num_heads * head_dim, name: "#{name}_q_up")

    # Decoupled RoPE projections
    # Q_rope: [batch, seq, num_heads * rope_dim] (per head)
    q_rope = Axon.dense(c_q, num_heads * rope_dim, name: "#{name}_q_rope_proj")
    # K_rope: [batch, seq, rope_dim] (shared across heads, will broadcast)
    k_rope = Axon.dense(input, rope_dim, name: "#{name}_k_rope_proj")

    # Compute attention with decoupled RoPE
    Axon.layer(
      &mla_attention_impl/6,
      [q_content, q_rope, k_content, k_rope, v],
      name: "#{name}_compute",
      num_heads: num_heads,
      head_dim: head_dim,
      rope_dim: rope_dim,
      op_name: :mla_attention
    )
    |> Axon.dense(hidden_size, name: "#{name}_out_proj")
  end

  # MLA attention implementation: concat content + RoPE dims, compute attention
  defp mla_attention_impl(q_content, q_rope, k_content, k_rope, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    rope_dim = opts[:rope_dim]
    full_head_dim = head_dim + rope_dim

    {batch, seq_len, _} = Nx.shape(q_content)

    # Reshape content to multi-head: [batch, num_heads, seq, head_dim]
    q_c = reshape_to_heads(q_content, batch, seq_len, num_heads, head_dim)
    k_c = reshape_to_heads(k_content, batch, seq_len, num_heads, head_dim)
    v_h = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Apply RoPE to rope dimensions
    # q_rope: [batch, seq, num_heads * rope_dim] -> [batch, num_heads, seq, rope_dim]
    q_r = reshape_to_heads(q_rope, batch, seq_len, num_heads, rope_dim)
    q_r = apply_rope(q_r, seq_len, rope_dim)

    # k_rope: [batch, seq, rope_dim] -> apply RoPE -> broadcast to all heads
    # [batch, 1, seq, rope_dim]
    k_r =
      k_rope
      |> Nx.reshape({batch, 1, seq_len, rope_dim})
      |> apply_rope(seq_len, rope_dim)
      |> Nx.broadcast({batch, num_heads, seq_len, rope_dim})

    # Concatenate content and RoPE: Q = [Q_c ; Q_r], K = [K_c ; K_r]
    # [batch, num_heads, seq, head_dim + rope_dim]
    q = Nx.concatenate([q_c, q_r], axis: 3)
    k = Nx.concatenate([k_c, k_r], axis: 3)

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(full_head_dim, type: Nx.type(q)))

    # QK^T: [batch, heads, seq_q, seq_k]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols)

    mask =
      mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    # Softmax
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))

    weights =
      Nx.divide(
        weights,
        Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8)
      )

    # Weighted values: [batch, heads, seq, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v_h, [2], [0, 1])

    # Reshape back: [batch, seq, num_heads * head_dim]
    reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
  end

  # Apply RoPE to tensor with shape [..., seq_len, dim]
  # Works on any leading dimensions (batch, heads, etc.)
  defp apply_rope(tensor, seq_len, dim) do
    half_dim = div(dim, 2)

    freqs =
      Nx.pow(
        10_000.0,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), dim))
      )
      |> Nx.as_type(Nx.type(tensor))

    positions = Nx.iota({seq_len}) |> Nx.as_type(Nx.type(tensor))
    angles = Nx.outer(positions, freqs)

    # Reshape for broadcasting: [1, 1, seq_len, half_dim]
    cos_t = Nx.cos(angles) |> Nx.reshape({1, 1, seq_len, half_dim})
    sin_t = Nx.sin(angles) |> Nx.reshape({1, 1, seq_len, half_dim})

    x1 = Nx.slice_along_axis(tensor, 0, half_dim, axis: 3)
    x2 = Nx.slice_along_axis(tensor, half_dim, half_dim, axis: 3)

    r1 = Nx.subtract(Nx.multiply(x1, cos_t), Nx.multiply(x2, sin_t))
    r2 = Nx.add(Nx.multiply(x1, sin_t), Nx.multiply(x2, cos_t))

    Nx.concatenate([r1, r2], axis: 3)
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

  @doc """
  Get the output size of an MLA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      head_dim: 64,
      kv_latent_dim: 64,
      q_latent_dim: 192,
      rope_dim: 32,
      num_layers: 4,
      dropout: 0.1
    ]
  end
end
