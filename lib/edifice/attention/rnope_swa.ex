defmodule Edifice.Attention.RNoPESWA do
  @moduledoc """
  RNoPE-SWA: Sliding Window Attention without positional encoding.

  A minimalist attention mechanism that combines:
  - **Sliding Window Attention**: Each position only attends to the last `window_size` positions
  - **No Positional Encoding**: Pure content-based attention without position bias

  ## Key Innovation

  By removing positional encoding, the model learns purely content-based attention patterns.
  Combined with sliding window, this creates an efficient local attention mechanism that:
  - Has O(L * W) complexity instead of O(L^2) where W = window_size
  - Generalizes perfectly to any sequence length at inference time
  - Forces the model to rely on content similarity, not position heuristics

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v (no positional encoding)
  +--------------------------------+
  |  Sliding Window Attention      |
  |                                |
  |  Each position attends to      |
  |  last W positions only         |
  |  Q, K, V projections           |
  |  Attention(Q, K, V)            |
  |  Output projection             |
  +--------------------------------+
        |
  [batch, seq_len, hidden_size]
  ```

  ## When to Use

  - Long sequences where full attention is too expensive
  - Tasks where local context is most important (e.g., language modeling)
  - When you want length generalization at inference time
  - When you want to ablate the effect of positional encoding

  ## Usage

      model = RNoPESWA.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        window_size: 128,
        num_layers: 6
      )

  ## Reference

  - "RoPE is Overrated: Positional Encoding Ablations" (2025)
  - "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Utils.FusedOps

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_window_size 128
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:window_size, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build an RNoPE-SWA model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:window_size` - Attention window size (default: 128)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "rnope_swa_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_sliding_window_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              window_size: window_size,
              rope: false,
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
  Build a sliding window attention layer without positional encoding.

  ## Options

    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:window_size` - Attention window size (default: 128)
    - `:rope` - Whether to use RoPE (default: false for RNoPE-SWA)
    - `:name` - Layer name prefix
  """
  @spec build_sliding_window_attention(Axon.t(), keyword()) :: Axon.t()
  def build_sliding_window_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    use_rope = Keyword.get(opts, :rope, false)
    name = Keyword.get(opts, :name, "rnope_swa")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    output =
      Axon.layer(
        &sliding_window_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        use_rope: use_rope,
        op_name: :rnope_swa
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Sliding window attention implementation (no RoPE by default)
  defp sliding_window_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    window_size = opts[:window_size]
    use_rope = opts[:use_rope]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Optionally apply RoPE (default: off for RNoPE-SWA)
    {q, k} =
      if use_rope do
        Edifice.Blocks.RoPE.apply_rotary_4d(q, k)
      else
        {q, k}
      end

    # Scale
    scale = Nx.sqrt(head_dim) |> Nx.as_type(Nx.type(q))

    # Attention scores
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Sliding window + causal mask
    # Valid if: col <= row (causal) AND col >= row - window_size + 1 (in window)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    causal_mask = Nx.greater_equal(rows, cols)
    window_mask = Nx.greater_equal(cols, Nx.subtract(rows, window_size - 1))
    combined_mask = Nx.logical_and(causal_mask, window_mask)

    combined_mask =
      combined_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(combined_mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    # Softmax
    attn_weights = FusedOps.fused_softmax(scores)

    # Attention output
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back
    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

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
      window_size: 128,
      dropout: 0.1
    ]
  end
end
