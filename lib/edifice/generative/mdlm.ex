defmodule Edifice.Generative.MDLM do
  @moduledoc """
  MDLM: Masked Diffusion Language Model — discrete diffusion for text.

  Implements the denoising network from "Simple and Effective Masked Diffusion
  Language Models" (Sahoo et al., NeurIPS 2024) and Mercury (Inception Labs,
  2025). Instead of autoregressive left-to-right generation, MDLM corrupts
  tokens by masking and trains a bidirectional transformer to predict the
  original tokens from the masked input.

  ## Key Concept

  Unlike continuous diffusion (add Gaussian noise, predict noise), discrete
  diffusion uses an **absorbing process**: tokens are stochastically replaced
  with a `[MASK]` token, and the network learns to predict the clean token
  distribution at each masked position.

  ```
  Forward (training):
    1. Sample clean tokens x₀ from data
    2. Sample timestep t ~ U(0, 1)
    3. Mask each token independently with probability α(t)
    4. xₜ = where(rand < α(t), [MASK], x₀)
    5. logits = network(xₜ, t)
    6. Loss: cross-entropy on masked positions between logits and x₀

  Reverse (inference):
    1. Start with all [MASK] tokens
    2. For t = 1 → 0 in num_steps:
       - logits = network(xₜ, t)
       - Sample tokens from softmax(logits) at masked positions
       - Unmask fraction of tokens based on noise schedule
    3. Return final tokens
  ```

  ## Architecture

  Uses a Diffusion Transformer (DiT) backbone adapted for text:

  ```
  Inputs: masked_tokens [batch, seq_len], timestep [batch]
        |
  +--------------------------------------------------+
  | Token Embedding (learnable, vocab_size × hidden)  |
  | + Timestep Embedding (sinusoidal → MLP)           |
  +--------------------------------------------------+
        |
  +--------------------------------------------------+
  | DDiT Block × num_layers                           |
  |   AdaLN-Zero modulation from timestep embedding  |
  |   (shift, scale, gate) × 2 (attn + MLP)          |
  |   Multi-Head Self-Attention + RoPE                |
  |   SwiGLU FFN (4× expansion)                      |
  |   Residual connections                            |
  +--------------------------------------------------+
        |
  | Final AdaLN + Linear → vocab_size                 |
        |
  Output: logits [batch, seq_len, vocab_size]
  ```

  ## Mercury vs MDLM

  Mercury (Inception Labs) uses the same masked diffusion framework but
  scales to larger models and uses a "coarse-to-fine" sampling strategy
  with fewer denoising steps (~32-64) for 10× speedup over autoregressive
  generation. The neural network architecture is identical.

  ## Noise Schedules

  Supports multiple noise schedules via helper functions:
  - **Log-linear**: `-log(1 - (1-ε)t)` — default, good general choice
  - **Cosine**: `-log(ε + (1-ε)cos(πt/2))` — smoother masking curve

  ## Usage

      model = MDLM.build(
        vocab_size: 50_257,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        seq_len: 1024
      )

  ## References

  - Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
    (NeurIPS 2024) — https://arxiv.org/abs/2406.07524
  - Inception Labs, Mercury — https://arxiv.org/abs/2506.17298
  - Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of
    the Data Distribution" (SEDD) — https://arxiv.org/abs/2310.16834
  """

  @default_hidden_size 768
  @default_num_layers 12
  @default_num_heads 12
  @default_mlp_ratio 4
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:vocab_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:mlp_ratio, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build the MDLM denoising transformer.

  ## Options

    - `:vocab_size` - Number of tokens in vocabulary (required)
    - `:hidden_size` - Transformer hidden dimension (default: 768)
    - `:num_layers` - Number of DDiT blocks (default: 12)
    - `:num_heads` - Number of attention heads (default: 12)
    - `:seq_len` - Maximum sequence length (default: 1024)
    - `:mlp_ratio` - FFN expansion ratio (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    An Axon model that takes `masked_tokens` [batch, seq_len] and
    `timestep` [batch] and outputs logits [batch, seq_len, vocab_size].
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    seq_len = Keyword.get(opts, :seq_len, 1024)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Inputs
    masked_tokens = Axon.input("masked_tokens", shape: {nil, seq_len})
    timestep = Axon.input("timestep", shape: {nil})

    # Token embedding: indices -> dense vectors
    x = build_token_embedding(masked_tokens, vocab_size, hidden_size)

    # Timestep conditioning: sinusoidal -> MLP
    cond = build_timestep_embedding(timestep, hidden_size)

    # DDiT blocks with AdaLN-Zero conditioning
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_ddit_block(acc, cond,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          dropout: dropout,
          name: "ddit_block_#{layer_idx}"
        )
      end)

    # Final layer: AdaLN + output projection to vocab logits
    build_final_layer(x, cond, hidden_size, vocab_size)
  end

  # ===========================================================================
  # Token Embedding
  # ===========================================================================

  defp build_token_embedding(tokens, vocab_size, hidden_size) do
    # One-hot encode then project (equivalent to nn.Embedding)
    one_hot =
      Axon.nx(
        tokens,
        fn t ->
          Nx.equal(
            Nx.new_axis(t, -1),
            Nx.iota({vocab_size})
          )
          |> Nx.as_type(:f32)
        end, name: "token_one_hot")

    Axon.dense(one_hot, hidden_size, name: "token_embed")
  end

  # ===========================================================================
  # Timestep Embedding (sinusoidal + MLP)
  # ===========================================================================

  defp build_timestep_embedding(timestep, hidden_size) do
    freq_dim = 256

    embed =
      Axon.layer(
        &sinusoidal_timestep_impl/2,
        [timestep],
        name: "time_sinusoidal",
        freq_dim: freq_dim,
        op_name: :sinusoidal_embed
      )

    embed
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp sinusoidal_timestep_impl(timestep, opts) do
    freq_dim = opts[:freq_dim]
    half_dim = div(freq_dim, 2)

    # Frequencies: exp(-log(10000) * i / half_dim)
    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim)
        )
      )

    # timestep: [batch] -> [batch, 1]
    t = Nx.reshape(timestep, {Nx.axis_size(timestep, 0), 1}) |> Nx.as_type(:f32)

    # angles: [batch, half_dim]
    angles = Nx.multiply(t, Nx.reshape(freqs, {1, half_dim}))

    # [batch, freq_dim]
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # ===========================================================================
  # DDiT Block (Diffusion Transformer Block with AdaLN-Zero)
  # ===========================================================================

  defp build_ddit_block(input, condition, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    mlp_ratio = opts[:mlp_ratio]
    dropout = opts[:dropout]
    name = opts[:name]
    mlp_dim = hidden_size * mlp_ratio

    # AdaLN modulation: 6 params (shift, scale, gate for attn and MLP)
    adaln_params =
      condition
      |> Axon.activation(:silu, name: "#{name}_adaln_silu")
      |> Axon.dense(hidden_size * 6, name: "#{name}_adaln_proj")

    # --- Attention sub-layer ---
    x_norm = build_layer_norm(input, "#{name}_attn_norm")

    # Modulate with shift1, scale1
    x_mod =
      Axon.layer(
        &adaln_modulate_impl/3,
        [x_norm, adaln_params],
        name: "#{name}_attn_mod",
        hidden_size: hidden_size,
        offset: 0,
        op_name: :adaln_modulate
      )

    # Self-attention with RoPE
    attn_out =
      build_rope_attention(x_mod, hidden_size, num_heads, "#{name}_attn")

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")

    # Gate with gate1: [batch, hidden] → [batch, 1, hidden] for seq broadcast
    gate_attn =
      Axon.nx(
        adaln_params,
        fn params ->
          Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: -1)
          |> Nx.new_axis(1)
        end, name: "#{name}_gate_attn")

    attn_out = Axon.multiply(attn_out, gate_attn, name: "#{name}_attn_gated")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # --- MLP sub-layer ---
    x_norm2 = build_layer_norm(x, "#{name}_mlp_norm")

    # Modulate with shift2, scale2
    x_mod2 =
      Axon.layer(
        &adaln_modulate_impl/3,
        [x_norm2, adaln_params],
        name: "#{name}_mlp_mod",
        hidden_size: hidden_size,
        offset: 3,
        op_name: :adaln_modulate
      )

    # SwiGLU FFN
    mlp_out = build_swiglu_ffn(x_mod2, hidden_size, mlp_dim, "#{name}_ffn")
    mlp_out = maybe_dropout(mlp_out, dropout, "#{name}_mlp_drop")

    # Gate with gate2: [batch, hidden] → [batch, 1, hidden] for seq broadcast
    gate_mlp =
      Axon.nx(
        adaln_params,
        fn params ->
          Nx.slice_along_axis(params, hidden_size * 5, hidden_size, axis: -1)
          |> Nx.new_axis(1)
        end, name: "#{name}_gate_mlp")

    mlp_out = Axon.multiply(mlp_out, gate_mlp, name: "#{name}_mlp_gated")
    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # AdaLN modulation: (1 + scale) * x + shift
  # offset=0 → attn (shift1, scale1), offset=3 → MLP (shift2, scale2)
  defp adaln_modulate_impl(x, adaln_params, opts) do
    hidden_size = opts[:hidden_size]
    offset = opts[:offset]

    shift = Nx.slice_along_axis(adaln_params, offset * hidden_size, hidden_size, axis: -1)
    scale = Nx.slice_along_axis(adaln_params, (offset + 1) * hidden_size, hidden_size, axis: -1)

    # Expand [batch, hidden] → [batch, 1, hidden] for seq dim broadcast
    shift = Nx.new_axis(shift, 1)
    scale = Nx.new_axis(scale, 1)

    Nx.add(Nx.multiply(Nx.add(1.0, scale), x), shift)
  end

  # ===========================================================================
  # Self-Attention with Rotary Position Embeddings
  # ===========================================================================

  defp build_rope_attention(input, hidden_size, num_heads, name) do
    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &rope_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: div(hidden_size, num_heads),
        op_name: :rope_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp rope_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE to Q and K
    q = apply_rope(q, head_dim, seq_len)
    k = apply_rope(k, head_dim, seq_len)

    # Scaled dot-product attention (bidirectional, no causal mask)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [-1], keep_axes: true))

    # Weighted values: [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    attn_out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Apply RoPE to tensor [batch, heads, seq, head_dim]
  defp apply_rope(x, head_dim, seq_len) do
    half_dim = div(head_dim, 2)

    # Frequencies: theta_i = 10000^(-2i/d)
    freqs =
      Nx.pow(
        10_000.0,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), head_dim))
      )
      |> Nx.as_type(Nx.type(x))

    # Positions × frequencies → angles [seq, half_dim]
    positions = Nx.iota({seq_len}) |> Nx.as_type(Nx.type(x))
    angles = Nx.outer(positions, freqs)

    # [1, 1, seq, half_dim] for broadcasting
    cos_t = Nx.cos(angles) |> Nx.reshape({1, 1, seq_len, half_dim})
    sin_t = Nx.sin(angles) |> Nx.reshape({1, 1, seq_len, half_dim})

    # Split last dim into two halves and rotate
    x1 = Nx.slice_along_axis(x, 0, half_dim, axis: 3)
    x2 = Nx.slice_along_axis(x, half_dim, half_dim, axis: 3)

    rotated1 = Nx.subtract(Nx.multiply(x1, cos_t), Nx.multiply(x2, sin_t))
    rotated2 = Nx.add(Nx.multiply(x1, sin_t), Nx.multiply(x2, cos_t))

    Nx.concatenate([rotated1, rotated2], axis: 3)
  end

  # ===========================================================================
  # SwiGLU FFN
  # ===========================================================================

  defp build_swiglu_ffn(input, hidden_size, mlp_dim, name) do
    # Gate and up projections
    gate = Axon.dense(input, mlp_dim, name: "#{name}_gate")
    gate = Axon.activation(gate, :silu, name: "#{name}_silu")
    up = Axon.dense(input, mlp_dim, name: "#{name}_up")

    # Gated output
    gated = Axon.multiply(gate, up, name: "#{name}_gated")
    Axon.dense(gated, hidden_size, name: "#{name}_down")
  end

  # ===========================================================================
  # Final Layer (AdaLN + output projection)
  # ===========================================================================

  defp build_final_layer(input, condition, hidden_size, vocab_size) do
    # Final AdaLN: 2 params (shift, scale)
    final_adaln =
      condition
      |> Axon.activation(:silu, name: "final_adaln_silu")
      |> Axon.dense(hidden_size * 2, name: "final_adaln_proj")

    x_norm = build_layer_norm(input, "final_norm")

    # Modulate
    x_mod =
      Axon.layer(
        &final_adaln_modulate_impl/3,
        [x_norm, final_adaln],
        name: "final_mod",
        hidden_size: hidden_size,
        op_name: :final_adaln_modulate
      )

    # Project to vocab logits: [batch, seq_len, vocab_size]
    Axon.dense(x_mod, vocab_size, name: "output_proj")
  end

  defp final_adaln_modulate_impl(x, adaln_params, opts) do
    hidden_size = opts[:hidden_size]

    shift = Nx.slice_along_axis(adaln_params, 0, hidden_size, axis: -1)
    scale = Nx.slice_along_axis(adaln_params, hidden_size, hidden_size, axis: -1)

    shift = Nx.new_axis(shift, 1)
    scale = Nx.new_axis(scale, 1)

    Nx.add(Nx.multiply(Nx.add(1.0, scale), x), shift)
  end

  # ===========================================================================
  # Utilities
  # ===========================================================================

  defp build_layer_norm(input, name) do
    Axon.layer_norm(input, name: name)
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input

  # ===========================================================================
  # Noise Schedule Helpers
  # ===========================================================================

  @doc """
  Log-linear noise schedule.

  Returns the mask probability at time `t` in [0, 1]:
  `alpha(t) = 1 - exp(-sigma(t))` where `sigma(t) = -log(1 - (1-eps)*t)`.

  At t=0: alpha ≈ 0 (no masking). At t=1: alpha ≈ 1 (full masking).
  """
  @spec log_linear_schedule(Nx.Tensor.t()) :: Nx.Tensor.t()
  def log_linear_schedule(t) do
    eps = 1.0e-5
    sigma = Nx.negate(Nx.log(Nx.subtract(1.0, Nx.multiply(1.0 - eps, t))))
    Nx.subtract(1.0, Nx.exp(Nx.negate(sigma)))
  end

  @doc """
  Cosine noise schedule.

  `alpha(t) = 1 - (eps + (1-eps)*cos(pi*t/2))`.
  Smoother transition than log-linear.
  """
  @spec cosine_schedule(Nx.Tensor.t()) :: Nx.Tensor.t()
  def cosine_schedule(t) do
    eps = 1.0e-5
    Nx.subtract(1.0, Nx.add(eps, Nx.multiply(1.0 - eps, Nx.cos(Nx.multiply(:math.pi() / 2, t)))))
  end

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :vocab_size)
  end

  @doc """
  Recommended default configuration (GPT-2 medium scale).
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      vocab_size: 50_257,
      hidden_size: 768,
      num_layers: 12,
      num_heads: 12,
      seq_len: 1024,
      mlp_ratio: 4,
      dropout: 0.0
    ]
  end
end
