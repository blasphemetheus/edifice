defmodule Edifice.Generative.LLaDA do
  @moduledoc """
  LLaDA: Large Language Diffusion with Masking.

  Implements the denoising network from "Large Language Diffusion Models"
  (Nie et al., ICLR 2025). First 8B-scale discrete diffusion language model
  competitive with autoregressive LLMs (LLaMA3 8B).

  ## Key Innovation

  Unlike MDLM which uses AdaLN-Zero timestep conditioning, LLaDA uses **no
  timestep information at all**. The transformer receives only the masked
  token sequence and predicts clean tokens at masked positions. Prior work
  (RADD) proved this is sufficient because mask-based discrete diffusion
  is equivalent to any-order autoregressive modeling.

  ## Architecture

  A bidirectional LLaMA-style transformer:

  ```
  Inputs: tokens [batch, seq_len] (with [MASK] tokens)
        |
  +--------------------------------------------------+
  | Token Embedding (vocab_size × hidden_size)        |
  +--------------------------------------------------+
        |
  +--------------------------------------------------+
  | Bidirectional Transformer Block × num_layers      |
  |   Pre-RMSNorm                                     |
  |   GQA Self-Attention (no causal mask) + RoPE      |
  |   Residual                                        |
  |   Pre-RMSNorm                                     |
  |   SwiGLU FFN (gate_proj, up_proj, down_proj)      |
  |   Residual                                        |
  +--------------------------------------------------+
        |
  | RMSNorm + Linear → vocab_size                     |
        |
  Output: logits [batch, seq_len, vocab_size]
  ```

  ## Comparison with MDLM

  - **MDLM**: DiT encoder, AdaLN-Zero timestep conditioning, cosine schedule
  - **LLaDA**: LLaMA-style bidirectional, no timestep conditioning, linear masking
  - LLaDA is architecturally simpler but scales to 8B parameters

  ## Forward Process (Training)

  Each token is independently masked with probability `t ~ U(0,1)`:

      x_t[i] = [MASK]  if rand_i < t, else x_0[i]

  The model predicts original tokens only at masked positions.

  ## Usage

      model = LLaDA.build(
        vocab_size: 50_257,
        seq_len: 1024,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12
      )

  ## References

  - Nie et al., "Large Language Diffusion Models" (ICLR 2025)
    https://arxiv.org/abs/2502.09992
  """

  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 8
  @default_intermediate_size nil
  @default_rms_norm_eps 1.0e-5
  @default_rope_theta 10_000.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:vocab_size, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:intermediate_size, pos_integer()}
          | {:rms_norm_eps, float()}
          | {:rope_theta, float()}

  @doc """
  Build the LLaDA denoising transformer.

  ## Options

    - `:vocab_size` - Vocabulary size including mask token (required)
    - `:seq_len` - Maximum sequence length (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:num_heads` - Number of query attention heads (default: 8)
    - `:num_kv_heads` - Number of KV heads for GQA (default: num_heads)
    - `:intermediate_size` - SwiGLU FFN intermediate dim (default: hidden_size * 4)
    - `:rms_norm_eps` - RMSNorm epsilon (default: 1.0e-5)
    - `:rope_theta` - RoPE base frequency (default: 10000.0)

  ## Returns

    An Axon model taking `tokens` `[batch, seq_len]` and outputting
    logits `[batch, seq_len, vocab_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    seq_len = Keyword.fetch!(opts, :seq_len)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, num_heads)

    intermediate_size =
      Keyword.get(opts, :intermediate_size, @default_intermediate_size) || hidden_size * 4

    rms_norm_eps = Keyword.get(opts, :rms_norm_eps, @default_rms_norm_eps)
    rope_theta = Keyword.get(opts, :rope_theta, @default_rope_theta)

    tokens = Axon.input("tokens", shape: {nil, seq_len})

    # Token embedding
    x = build_token_embedding(tokens, vocab_size, hidden_size)

    # Transformer blocks (bidirectional, no causal mask)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_transformer_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          num_kv_heads: num_kv_heads,
          intermediate_size: intermediate_size,
          rms_norm_eps: rms_norm_eps,
          rope_theta: rope_theta,
          name: "layer_#{layer_idx}"
        )
      end)

    # Final RMSNorm + output projection
    x = build_rms_norm(x, hidden_size, rms_norm_eps, "final_norm")
    Axon.dense(x, vocab_size, name: "lm_head")
  end

  # ===========================================================================
  # Token Embedding
  # ===========================================================================

  defp build_token_embedding(tokens, vocab_size, hidden_size) do
    one_hot =
      Axon.nx(
        tokens,
        fn t ->
          Nx.equal(Nx.new_axis(t, -1), Nx.iota({vocab_size}))
          |> Nx.as_type(:f32)
        end,
        name: "token_one_hot"
      )

    Axon.dense(one_hot, hidden_size, name: "token_embed")
  end

  # ===========================================================================
  # Transformer Block (Pre-RMSNorm, GQA + RoPE, SwiGLU)
  # ===========================================================================

  defp build_transformer_block(input, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    intermediate_size = opts[:intermediate_size]
    rms_norm_eps = opts[:rms_norm_eps]
    rope_theta = opts[:rope_theta]
    name = opts[:name]

    # --- Attention sub-layer ---
    normed = build_rms_norm(input, hidden_size, rms_norm_eps, "#{name}_attn_norm")

    attn_out =
      build_gqa_rope_attention(normed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        rope_theta: rope_theta,
        name: "#{name}_attn"
      )

    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # --- MLP sub-layer ---
    normed2 = build_rms_norm(x, hidden_size, rms_norm_eps, "#{name}_mlp_norm")

    mlp_out =
      Edifice.Blocks.SwiGLU.layer(normed2,
        hidden_size: hidden_size,
        inner_size: intermediate_size,
        name: "#{name}_ffn"
      )

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # ===========================================================================
  # GQA Self-Attention with RoPE (Bidirectional)
  # ===========================================================================

  defp build_gqa_rope_attention(input, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    rope_theta = opts[:rope_theta]
    name = opts[:name]
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, num_heads * head_dim, name: "#{name}_q")
    k = Axon.dense(input, num_kv_heads * head_dim, name: "#{name}_k")
    v = Axon.dense(input, num_kv_heads * head_dim, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &gqa_rope_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        head_dim: head_dim,
        rope_theta: rope_theta,
        op_name: :gqa_rope_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp gqa_rope_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    head_dim = opts[:head_dim]
    rope_theta = opts[:rope_theta]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    k =
      k
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE
    {q, k} = Edifice.Blocks.RoPE.apply_rotary_4d(q, k, theta: rope_theta)

    # Repeat KV heads for GQA
    {k, v} = repeat_kv(k, v, num_heads, num_kv_heads)

    # Scaled dot-product attention (bidirectional — no causal mask)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    attn_out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp repeat_kv(k, v, num_heads, num_kv_heads) when num_heads == num_kv_heads, do: {k, v}

  defp repeat_kv(k, v, num_heads, num_kv_heads) do
    repeats = div(num_heads, num_kv_heads)
    # k: [batch, kv_heads, seq, head_dim] -> [batch, heads, seq, head_dim]
    batch = Nx.axis_size(k, 0)
    seq_len = Nx.axis_size(k, 2)
    head_dim = Nx.axis_size(k, 3)

    k =
      k
      |> Nx.new_axis(2)
      |> Nx.broadcast({batch, num_kv_heads, repeats, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    v =
      v
      |> Nx.new_axis(2)
      |> Nx.broadcast({batch, num_kv_heads, repeats, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    {k, v}
  end

  # ===========================================================================
  # RMSNorm
  # ===========================================================================

  defp build_rms_norm(input, hidden_size, eps, name) do
    Axon.layer(
      &rms_norm_impl/2,
      [input],
      name: name,
      hidden_size: hidden_size,
      eps: eps,
      op_name: :rms_norm
    )
  end

  defp rms_norm_impl(x, opts) do
    eps = opts[:eps]
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(variance, eps)))
  end

  # ===========================================================================
  # Utilities
  # ===========================================================================

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :vocab_size)
  end
end
