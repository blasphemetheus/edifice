defmodule Edifice.Generative.CaDDi do
  @moduledoc """
  CaDDi: Causal Discrete Diffusion.

  Implements CaDDi from "Non-Markovian Discrete Diffusion with Causal
  Language Models" (Zhang et al., NeurIPS 2025). Lifts the Markov
  assumption in discrete diffusion by conditioning each denoising step
  on the entire generative trajectory, using a standard causal transformer.

  ## Key Innovation

  Traditional discrete diffusion (MDLM, LLaDA) uses Markov transitions:
  each step sees only the current noisy state. CaDDi flattens the diffusion
  trajectory into a 1D sequence and processes it with a causal LM, so each
  step can see all previous (noisier) states. When T=1, CaDDi reduces
  exactly to a standard autoregressive LM.

  ## Two Variants

  - **`:block`** (default) — Block-causal attention: within each block of
    `seq_len` tokens, attention is bidirectional; across blocks, causal.
    Predicts each block conditioned on all previous diffusion steps.

  - **`:ar`** — Full token-level autoregressive. Each token in the
    flattened trajectory is predicted left-to-right. Compatible with
    pretrained LLM weights.

  ## Architecture

  ```
  Diffusion trajectory: T blocks of seq_len tokens each
        |
  Flatten: [x_T, x_{T-1}, ..., x_1] → length T * seq_len
        |
  +--------------------------------------------------+
  | Token Embedding + Position/Timestep Embedding     |
  +--------------------------------------------------+
        |
  +--------------------------------------------------+
  | Causal Transformer × num_layers                   |
  |   Block-causal mask (:block) or full causal (:ar) |
  |   Pre-RMSNorm + Self-Attention + RoPE             |
  |   Pre-RMSNorm + SwiGLU FFN                        |
  +--------------------------------------------------+
        |
  | RMSNorm + Linear → vocab_size                     |
        |
  Output: logits [batch, T * seq_len, vocab_size]
  ```

  ## Context Window

  For efficiency, CaDDi uses a sliding window of `context_window` diffusion
  steps rather than the full trajectory. Default W=4 means each step sees
  only the 4 most recent prior states.

  ## Usage

      model = CaDDi.build(
        vocab_size: 50_257,
        seq_len: 128,
        num_diffusion_steps: 16,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        variant: :block
      )

  ## References

  - Zhang et al., "Non-Markovian Discrete Diffusion with Causal Language
    Models" (NeurIPS 2025) — https://arxiv.org/abs/2502.09767
  """

  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 8
  @default_num_diffusion_steps 16
  @default_context_window 4
  @default_intermediate_size nil
  @default_rms_norm_eps 1.0e-5

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:context_window, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:intermediate_size, pos_integer()}
          | {:num_diffusion_steps, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:rms_norm_eps, float()}
          | {:seq_len, pos_integer()}
          | {:variant, :block | :ar}
          | {:vocab_size, pos_integer()}

  @doc """
  Build the CaDDi causal discrete diffusion model.

  ## Options

    - `:vocab_size` - Vocabulary size including mask token (required)
    - `:seq_len` - Tokens per diffusion block (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_layers` - Transformer blocks (default: 6)
    - `:num_heads` - Attention heads (default: 8)
    - `:num_diffusion_steps` - T, number of diffusion steps (default: 16)
    - `:context_window` - Sliding window over trajectory (default: 4)
    - `:variant` - `:block` (block-causal) or `:ar` (token-level) (default: `:block`)
    - `:intermediate_size` - FFN hidden size (default: hidden_size * 4)
    - `:rms_norm_eps` - RMSNorm epsilon (default: 1.0e-5)

  ## Returns

    An Axon model taking `trajectory` `[batch, context_window * seq_len]`
    and outputting logits `[batch, context_window * seq_len, vocab_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    seq_len = Keyword.fetch!(opts, :seq_len)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_diffusion_steps = Keyword.get(opts, :num_diffusion_steps, @default_num_diffusion_steps)
    context_window = Keyword.get(opts, :context_window, @default_context_window)
    variant = Keyword.get(opts, :variant, :block)

    intermediate_size =
      Keyword.get(opts, :intermediate_size, @default_intermediate_size) || hidden_size * 4

    rms_norm_eps = Keyword.get(opts, :rms_norm_eps, @default_rms_norm_eps)

    # Effective context: min(T, context_window) blocks of seq_len
    effective_steps = min(num_diffusion_steps, context_window)
    total_len = effective_steps * seq_len

    trajectory = Axon.input("trajectory", shape: {nil, total_len})

    # Token embedding
    x = build_token_embedding(trajectory, vocab_size, hidden_size)

    # Add step embedding (which diffusion block each position belongs to)
    x =
      Axon.layer(
        &add_step_embedding_impl/2,
        [x],
        name: "step_embed",
        seq_len: seq_len,
        effective_steps: effective_steps,
        hidden_size: hidden_size,
        op_name: :step_embed
      )

    # Transformer blocks with appropriate causal mask
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_causal_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          intermediate_size: intermediate_size,
          rms_norm_eps: rms_norm_eps,
          variant: variant,
          seq_len: seq_len,
          effective_steps: effective_steps,
          name: "layer_#{layer_idx}"
        )
      end)

    # Final norm + output projection
    x = build_rms_norm(x, rms_norm_eps, "final_norm")
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

  # Learned embedding indicating which diffusion step each position belongs to
  defp add_step_embedding_impl(x, opts) do
    seq_len = opts[:seq_len]
    effective_steps = opts[:effective_steps]
    hidden_size = opts[:hidden_size]

    batch = Nx.axis_size(x, 0)
    total_len = effective_steps * seq_len

    # Create step indices: [0,0,...,0, 1,1,...,1, ..., T-1,T-1,...,T-1]
    step_ids =
      Nx.iota({total_len})
      |> Nx.quotient(seq_len)

    # Simple sinusoidal step embedding
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim)
        )
      )

    step_float = Nx.as_type(step_ids, :f32) |> Nx.reshape({total_len, 1})
    angles = Nx.multiply(step_float, Nx.reshape(freqs, {1, half_dim}))
    step_embed = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)

    # Broadcast and add: [total_len, hidden] -> [1, total_len, hidden]
    step_embed = Nx.broadcast(Nx.new_axis(step_embed, 0), {batch, total_len, hidden_size})
    Nx.add(x, step_embed)
  end

  # ===========================================================================
  # Transformer Block with Causal Masking
  # ===========================================================================

  defp build_causal_block(input, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    intermediate_size = opts[:intermediate_size]
    rms_norm_eps = opts[:rms_norm_eps]
    variant = opts[:variant]
    seq_len = opts[:seq_len]
    effective_steps = opts[:effective_steps]
    name = opts[:name]

    # Attention sub-layer
    normed = build_rms_norm(input, rms_norm_eps, "#{name}_attn_norm")

    attn_out =
      build_masked_attention(normed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        variant: variant,
        seq_len: seq_len,
        effective_steps: effective_steps,
        name: "#{name}_attn"
      )

    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # MLP sub-layer
    normed2 = build_rms_norm(x, rms_norm_eps, "#{name}_mlp_norm")

    mlp_out =
      Edifice.Blocks.SwiGLU.layer(normed2,
        hidden_size: hidden_size,
        inner_size: intermediate_size,
        name: "#{name}_ffn"
      )

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # ===========================================================================
  # Masked Attention (block-causal or full causal)
  # ===========================================================================

  defp build_masked_attention(input, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    variant = opts[:variant]
    seq_len = opts[:seq_len]
    effective_steps = opts[:effective_steps]
    name = opts[:name]
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &masked_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        variant: variant,
        seq_len: seq_len,
        effective_steps: effective_steps,
        op_name: :masked_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp masked_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    variant = opts[:variant]
    seq_len = opts[:seq_len]
    effective_steps = opts[:effective_steps]

    batch = Nx.axis_size(q, 0)
    total_len = Nx.axis_size(q, 1)

    q =
      q |> Nx.reshape({batch, total_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    k =
      k |> Nx.reshape({batch, total_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v |> Nx.reshape({batch, total_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE
    {q, k} = Edifice.Blocks.RoPE.apply_rotary_4d(q, k)

    # Compute attention scores
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Apply causal mask [total_len, total_len] -> [1, 1, total_len, total_len]
    # Use additive masking: 0 where allowed, -1e9 where blocked
    bool_mask =
      build_mask(variant, total_len, seq_len, effective_steps)
      |> Nx.reshape({1, 1, total_len, total_len})

    additive_mask =
      bool_mask
      |> Nx.as_type(:f32)
      |> Nx.subtract(1.0)
      |> Nx.multiply(1.0e9)

    scores = Nx.add(scores, additive_mask)

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    attn_out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, total_len, num_heads * head_dim})
  end

  # Build the attention mask depending on variant
  defp build_mask(:ar, total_len, _seq_len, _effective_steps) do
    # Full causal: lower triangular
    rows = Nx.iota({total_len, 1})
    cols = Nx.iota({1, total_len})
    Nx.greater_equal(rows, cols)
  end

  defp build_mask(:block, total_len, seq_len, _effective_steps) do
    # Block-causal: within blocks = bidirectional, across blocks = causal
    # Position i is in block i // seq_len
    positions = Nx.iota({total_len})
    row_blocks = Nx.quotient(Nx.reshape(positions, {total_len, 1}), seq_len)
    col_blocks = Nx.quotient(Nx.reshape(positions, {1, total_len}), seq_len)

    # Allow attention if same block OR query block > key block
    Nx.greater_equal(row_blocks, col_blocks)
  end

  # ===========================================================================
  # RMSNorm
  # ===========================================================================

  defp build_rms_norm(input, eps, name) do
    Axon.layer(
      &rms_norm_impl/2,
      [input],
      name: name,
      eps: eps,
      op_name: :rms_norm
    )
  end

  defp rms_norm_impl(x, opts) do
    eps = opts[:eps]
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
