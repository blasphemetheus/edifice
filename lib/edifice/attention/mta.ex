defmodule Edifice.Attention.MTA do
  @moduledoc """
  MTA: Multi-Token Attention with 2D convolution on attention logits.

  Enhances standard multi-head attention by applying learnable 2D depthwise
  convolutions on the attention logit matrix before softmax, followed by
  optional 1D convolution across heads for head mixing. The 2D convolution
  allows each head to aggregate information from neighboring query-key
  positions, enabling richer attention patterns beyond pointwise scores.

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------------------+
  |  MTA Block (x num_layers)               |
  |                                          |
  |  Q, K, V projections                    |
  |  logits = QK^T / sqrt(d)               |
  |  logits *= binary_causal_mask           |
  |  logits = depthwise_conv2d(logits)      |
  |  logits += -inf causal mask             |
  |  weights = softmax(logits)              |
  |  weights = head_mixing_conv1d(weights)  |
  |  output = weights @ V                   |
  |  + Residual + FFN                       |
  +------------------------------------------+
        |
  [batch, hidden_size]
  ```

  The 2D convolution kernel has shape `{num_heads, 1, c_q, c_k}` and is
  initialized as an identity filter (center tap = 1.0), so the model starts
  equivalent to standard attention and learns to deviate.

  ## Key Properties

  - **Identity initialization**: Conv kernel starts as identity, so MTA = standard
    attention at init. Training learns which neighbor positions to aggregate.
  - **Double masking**: Binary mask zeroes future logits *before* conv (prevents
    -inf spreading through conv kernel), then -inf mask *after* conv for softmax.
  - **Selective application**: 2D KQ conv and head mixing can be applied every
    Nth layer via `:kq_conv_every` and `:head_conv_every` options.
  - **Depthwise conv**: Each head has its own independent 2D kernel, implemented
    via grouped convolution with `groups: num_heads`.

  ## Usage

      model = MTA.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6,
        c_q: 6,
        c_k: 11
      )

  ## Reference

  - Team et al., "Multi-Token Attention" (Meta, 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_c_q 6
  @default_c_k 11
  @default_c_h 2
  @default_kq_conv_every 4
  @default_head_conv_every 1
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:c_q, pos_integer()}
          | {:c_k, pos_integer()}
          | {:c_h, pos_integer()}
          | {:kq_conv_every, pos_integer()}
          | {:head_conv_every, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build an MTA (Multi-Token Attention) model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:c_q` - Query-axis conv kernel size (default: 6)
    - `:c_k` - Key-axis conv kernel size (default: 11)
    - `:c_h` - Head mixing conv kernel size (default: 2)
    - `:kq_conv_every` - Apply 2D KQ conv every N layers (default: 4)
    - `:head_conv_every` - Apply head mixing every N layers (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
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
    c_q = Keyword.get(opts, :c_q, @default_c_q)
    c_k = Keyword.get(opts, :c_k, @default_c_k)
    c_h = Keyword.get(opts, :c_h, @default_c_h)
    kq_conv_every = Keyword.get(opts, :kq_conv_every, @default_kq_conv_every)
    head_conv_every = Keyword.get(opts, :head_conv_every, @default_head_conv_every)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "mta_block_#{layer_idx}"

          apply_kq_conv = rem(layer_idx - 1, kq_conv_every) == 0
          apply_head_conv = rem(layer_idx - 1, head_conv_every) == 0

          attn_fn = fn x, attn_name ->
            build_mta_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              c_q: c_q,
              c_k: c_k,
              c_h: c_h,
              apply_kq_conv: apply_kq_conv,
              apply_head_conv: apply_head_conv,
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
  Build the MTA attention layer.

  Projects to Q, K, V, creates learnable 2D conv kernel and optional head
  mixing kernel, then computes attention with convolution on logits.
  """
  @spec build_mta_attention(Axon.t(), keyword()) :: Axon.t()
  def build_mta_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    c_q = Keyword.get(opts, :c_q, @default_c_q)
    c_k = Keyword.get(opts, :c_k, @default_c_k)
    c_h = Keyword.get(opts, :c_h, @default_c_h)
    apply_kq_conv = Keyword.get(opts, :apply_kq_conv, true)
    apply_head_conv = Keyword.get(opts, :apply_head_conv, true)
    name = Keyword.get(opts, :name, "mta_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    if apply_kq_conv do
      # Learnable 2D depthwise conv kernel: {num_heads, 1, c_q, c_k}
      # Identity initialization: center tap = 1.0
      kq_kernel =
        Axon.param("#{name}_kq_kernel", {num_heads, 1, c_q, c_k},
          initializer: fn shape, _type ->
            {h, _one, cq, ck} = shape
            base = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), shape)

            indices =
              Nx.tensor(for head <- 0..(h - 1), do: [head, 0, cq - 1, div(ck, 2)])

            Nx.indexed_put(base, indices, Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {h}))
          end
        )

      if apply_head_conv do
        # Head mixing 1D conv kernel: {num_heads, c_h, 1} with groups
        # Each group of c_h heads gets mixed together
        head_kernel =
          Axon.param("#{name}_head_kernel", {num_heads, c_h, 1},
            initializer: fn shape, _type ->
              {nh, ch, _one} = shape
              base = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), shape)
              # Identity: for each head, the center position in the c_h kernel = 1.0
              indices =
                Nx.tensor(for head <- 0..(nh - 1), do: [head, div(ch, 2), 0])

              Nx.indexed_put(
                base,
                indices,
                Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {nh})
              )
            end
          )

        output =
          Axon.layer(
            &mta_attention_with_head_mix/6,
            [q_proj, k_proj, v_proj, kq_kernel, head_kernel],
            name: "#{name}_compute",
            num_heads: num_heads,
            head_dim: head_dim,
            c_q: c_q,
            c_k: c_k,
            c_h: c_h,
            op_name: :mta_attention
          )

        Axon.dense(output, hidden_size, name: "#{name}_out_proj")
      else
        output =
          Axon.layer(
            &mta_attention_kq_only/5,
            [q_proj, k_proj, v_proj, kq_kernel],
            name: "#{name}_compute",
            num_heads: num_heads,
            head_dim: head_dim,
            c_q: c_q,
            c_k: c_k,
            op_name: :mta_attention
          )

        Axon.dense(output, hidden_size, name: "#{name}_out_proj")
      end
    else
      # Plain attention (no conv) — used on layers where conv is not applied
      output =
        Axon.layer(
          &plain_attention_impl/4,
          [q_proj, k_proj, v_proj],
          name: "#{name}_compute",
          num_heads: num_heads,
          head_dim: head_dim,
          op_name: :mta_attention
        )

      Axon.dense(output, hidden_size, name: "#{name}_out_proj")
    end
  end

  # MTA attention with both KQ conv and head mixing
  # Q, K, V: [batch, seq, hidden], kq_kernel: {H, 1, c_q, c_k}, head_kernel: {H, c_h, 1}
  defp mta_attention_with_head_mix(q, k, v, kq_kernel, head_kernel, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    c_q = opts[:c_q]
    c_k = opts[:c_k]
    c_h = opts[:c_h]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Compute logits: QK^T / sqrt(d) -> [batch, heads, seq, seq]
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    logits = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Pre-conv causal zeroing (binary mask, NOT -inf, because conv spreads -inf)
    causal_binary = build_binary_causal_mask(seq_len, batch, num_heads, Nx.type(logits))
    logits = Nx.multiply(logits, causal_binary)

    # 2D depthwise conv on attention logit matrix
    logits = apply_kq_conv(logits, kq_kernel, batch, num_heads, seq_len, c_q, c_k)

    # Post-conv causal mask: -inf for future positions before softmax
    causal_neg_inf = build_neg_inf_causal_mask(seq_len, batch, num_heads, Nx.type(logits))
    logits = Nx.add(logits, causal_neg_inf)

    # Softmax over key dimension
    attn_weights = softmax_last_axis(logits)

    # Head mixing: 1D conv across head dimension
    attn_weights =
      apply_head_mixing(attn_weights, head_kernel, batch, num_heads, seq_len, c_h)

    # V aggregation: [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  # MTA attention with KQ conv only (no head mixing)
  defp mta_attention_kq_only(q, k, v, kq_kernel, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    c_q = opts[:c_q]
    c_k = opts[:c_k]

    {batch, seq_len, _} = Nx.shape(q)

    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    logits = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    causal_binary = build_binary_causal_mask(seq_len, batch, num_heads, Nx.type(logits))
    logits = Nx.multiply(logits, causal_binary)

    logits = apply_kq_conv(logits, kq_kernel, batch, num_heads, seq_len, c_q, c_k)

    causal_neg_inf = build_neg_inf_causal_mask(seq_len, batch, num_heads, Nx.type(logits))
    logits = Nx.add(logits, causal_neg_inf)

    attn_weights = softmax_last_axis(logits)
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  # Plain attention (no conv on logits) for layers where conv is disabled
  defp plain_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    logits = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Standard -inf causal mask
    causal_neg_inf = build_neg_inf_causal_mask(seq_len, batch, num_heads, Nx.type(logits))
    logits = Nx.add(logits, causal_neg_inf)

    attn_weights = softmax_last_axis(logits)
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  # Apply 2D depthwise conv on attention logits
  # logits: [batch, heads, seq_q, seq_k] -> same shape
  # kq_kernel: {heads, 1, c_q, c_k}
  defp apply_kq_conv(logits, kq_kernel, batch, num_heads, seq_len, c_q, c_k) do
    # Nx.conv expects [batch, channels, spatial...] with kernel [out_ch, in_ch/groups, spatial...]
    # Input: [B, H, N, N] — B=batch, H=channels, N x N spatial
    # Kernel: [H, 1, c_q, c_k] — H output channels, 1 input channel per group (groups=H)
    # Causal padding on query axis: {c_q - 1, 0} (only look back)
    # Symmetric padding on key axis: {div(c_k, 2), div(c_k, 2)}
    #
    # However, Nx.conv requires the same batch dimension for all samples.
    # We process each batch element separately to avoid batching issues.
    pad_q = c_q - 1
    pad_k_left = div(c_k, 2)
    pad_k_right = div(c_k, 2)

    # Process batch elements via manual loop to avoid Nx.conv batch dimension issues
    # Each element: [H, N, N] -> add batch dim -> [1, H, N, N] -> conv -> [1, H, N, N]
    batch_results =
      for b <- 0..(batch - 1) do
        element = Nx.slice(logits, [b, 0, 0, 0], [1, num_heads, seq_len, seq_len])

        Nx.conv(element, kq_kernel,
          strides: [1, 1],
          padding: [{pad_q, 0}, {pad_k_left, pad_k_right}],
          input_dilation: [1, 1],
          kernel_dilation: [1, 1],
          feature_group_size: num_heads
        )
      end

    Nx.concatenate(batch_results, axis: 0)
  end

  # Apply head mixing across the head dimension via learnable per-group mixing matrices.
  # weights: [batch, heads, seq_q, seq_k]
  # head_kernel: {num_heads, c_h, 1} — reshaped to {num_groups, c_h, c_h} mixing matrices
  defp apply_head_mixing(weights, head_kernel, batch, num_heads, seq_len, c_h) do
    num_groups = div(num_heads, c_h)

    # Reshape: [B, H, N, N] -> [B, N, N, H] -> [B*N*N, num_groups, c_h]
    flat =
      weights
      |> Nx.transpose(axes: [0, 2, 3, 1])
      |> Nx.reshape({batch * seq_len * seq_len, num_groups, c_h})

    # Learnable per-group mixing matrix: {num_groups, c_h, c_h}
    mix_matrix = Nx.reshape(head_kernel, {num_groups, c_h, c_h})

    # Transpose to put groups at axis 0 for batched dot:
    # flat: [B*N*N, num_groups, c_h] -> [num_groups, B*N*N, c_h]
    flat_t = Nx.transpose(flat, axes: [1, 0, 2])

    # Batched matmul with groups as batch dim (axis 0):
    # [num_groups, B*N*N, c_h] x [num_groups, c_h, c_h] -> [num_groups, B*N*N, c_h]
    mixed = Nx.dot(flat_t, [2], [0], mix_matrix, [1], [0])

    # Transpose back: [num_groups, B*N*N, c_h] -> [B*N*N, num_groups, c_h]
    mixed = Nx.transpose(mixed, axes: [1, 0, 2])

    # Reshape back: [B*N*N, num_groups, c_h] -> [B*N*N, H] -> [B, N, N, H] -> [B, H, N, N]
    mixed
    |> Nx.reshape({batch * seq_len * seq_len, num_heads})
    |> Nx.reshape({batch, seq_len, seq_len, num_heads})
    |> Nx.transpose(axes: [0, 3, 1, 2])
  end

  # Binary causal mask (1.0 for valid, 0.0 for future)
  defp build_binary_causal_mask(seq_len, batch, num_heads, type) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    Nx.greater_equal(rows, cols)
    |> Nx.new_axis(0)
    |> Nx.new_axis(0)
    |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
    |> Nx.as_type(type)
  end

  # Neg-inf causal mask (0.0 for valid, -1e9 for future)
  defp build_neg_inf_causal_mask(seq_len, batch, num_heads, type) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    is_valid = Nx.greater_equal(rows, cols)

    mask =
      Nx.select(
        is_valid,
        Nx.tensor(0.0, type: type),
        Nx.tensor(-1.0e9, type: type)
      )

    mask
    |> Nx.new_axis(0)
    |> Nx.new_axis(0)
    |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
  end

  # Numerically stable softmax over last axis
  defp softmax_last_axis(tensor) do
    max_val = Nx.reduce_max(tensor, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(tensor, max_val)
    exp_vals = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_vals, axes: [-1], keep_axes: true)
    Nx.divide(exp_vals, sum_exp)
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
      c_q: 6,
      c_k: 11,
      c_h: 2,
      kq_conv_every: 4,
      head_conv_every: 1,
      dropout: 0.1
    ]
  end
end
