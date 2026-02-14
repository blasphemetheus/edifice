defmodule Edifice.Attention.GQA do
  @moduledoc """
  GQA: Grouped Query Attention.

  Grouped Query Attention is an interpolation between Multi-Head Attention (MHA)
  and Multi-Query Attention (MQA). Groups of query heads share key/value heads,
  reducing KV cache size while maintaining most of MHA's quality.

  ## Key Innovation: KV Head Sharing

  Instead of one KV head per query head (MHA) or one KV head total (MQA),
  GQA uses G groups where each group of Q heads shares one KV head:

  ```
  MHA:  Q1-K1-V1  Q2-K2-V2  Q3-K3-V3  Q4-K4-V4   (4 KV heads)
  GQA:  Q1-K1-V1  Q2-K1-V1  Q3-K2-V2  Q4-K2-V2   (2 KV heads)
  MQA:  Q1-K1-V1  Q2-K1-V1  Q3-K1-V1  Q4-K1-V1   (1 KV head)
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |       GQA Transformer Block          |
  |                                      |
  |  LayerNorm -> GQA Attention          |
  |    Q: num_heads projections          |
  |    K: num_kv_heads projections       |
  |    V: num_kv_heads projections       |
  |    K,V repeated for Q head groups    |
  |    -> scaled dot-product attention   |
  |    -> output projection              |
  |  -> Residual                         |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Variant | KV Cache | Quality |
  |---------|----------|---------|
  | MHA (G=H) | O(H*d) | Best |
  | GQA (1<G<H) | O(G*d) | Near-MHA |
  | MQA (G=1) | O(d) | Slightly lower |

  ## Usage

      model = GQA.build(
        embed_size: 287,
        hidden_size: 256,
        num_heads: 8,
        num_kv_heads: 2,
        num_layers: 4
      )

  ## References
  - Paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    (Ainslie et al., 2023)
  - Used in: LLaMA 2 70B, Mistral 7B, Gemma
  """

  require Axon

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Utils.FusedOps

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_heads 8
  @default_num_kv_heads 2
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build a GQA transformer model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of query heads (default: 8)
    - `:num_kv_heads` - Number of key/value heads (default: 2)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          name = "gqa_block_#{block_opts[:layer_idx]}"

          attn_fn = fn x, attn_name ->
            build_gqa_attention(x, Keyword.merge(opts, name: attn_name))
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
  Build the Grouped Query Attention layer.

  Projects Q into num_heads groups and K/V into num_kv_heads groups,
  repeats K/V to match Q head count, then applies scaled dot-product attention.
  """
  @spec build_gqa_attention(Axon.t(), keyword()) :: Axon.t()
  def build_gqa_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    name = Keyword.get(opts, :name, "gqa_attn")

    head_dim = div(hidden_size, num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Q projection: full num_heads
    q_proj = Axon.dense(input, q_dim, name: "#{name}_q_proj")

    # K, V projections: only num_kv_heads
    k_proj = Axon.dense(input, kv_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, kv_dim, name: "#{name}_v_proj")

    # Apply GQA attention
    output =
      Axon.layer(
        &gqa_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        head_dim: head_dim,
        op_name: :gqa_attention
      )

    # Output projection
    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # GQA attention implementation
  defp gqa_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Number of Q heads per KV head
    heads_per_group = div(num_heads, num_kv_heads)

    # Reshape Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    q =
      q
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Reshape K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
    k =
      k
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Repeat K, V for each group of Q heads
    # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
    # Reshape to [batch, num_kv_heads, 1, seq, head_dim], broadcast, then reshape
    k =
      k
      |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
      |> Nx.broadcast({batch, num_kv_heads, heads_per_group, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    v =
      v
      |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
      |> Nx.broadcast({batch, num_kv_heads, heads_per_group, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # scores: [batch, num_heads, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      Nx.broadcast(
        Nx.reshape(causal_mask, {1, 1, seq_len, seq_len}),
        {batch, num_heads, seq_len, seq_len}
      )

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Softmax over keys
    weights = FusedOps.fused_softmax(scores)

    # Weighted sum: [batch, num_heads, seq, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, num_heads * head_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a GQA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a GQA model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    head_dim = div(hidden_size, num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    inner_size = hidden_size * 4

    # Per layer:
    # Attention: Q proj + K proj + V proj + output proj
    attn_params =
      hidden_size * q_dim +
        hidden_size * kv_dim * 2 +
        q_dim * hidden_size

    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = attn_params + ffn_params

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 8,
      num_kv_heads: 2,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
