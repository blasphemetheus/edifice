defmodule Edifice.Meta.Eagle3 do
  @moduledoc """
  EAGLE-3: Multi-Level Feature Fusion Draft Head for Speculative Decoding.

  Implements the EAGLE-3 draft head from "EAGLE-3: Scaling up Inference
  Acceleration of Large Language Models via Training-Time Test" (NeurIPS 2025).
  Achieves 3-6x decoding speedup by extracting low/mid/high features from
  the target model and fusing them into a lightweight single-layer draft head.

  ## Key Innovations over EAGLE-1/2

  - **Feature fusion**: Extracts hidden states from 3 levels (low/mid/high)
    of the target model instead of just the final layer.
  - **No feature prediction constraint**: Removes the requirement to match
    target model internal features; the draft head learns freely from data.
  - **Training-time test**: Simulates the speculative decoding process during
    training for better alignment between train and inference behavior.
  - **Vocabulary mapping**: Supports a smaller draft vocabulary (e.g., 32K)
    with a learned mapping to the target vocabulary (e.g., 128K).

  ## Architecture

  ```
  Target Model Internals:
    Layer L_low  → features_low   [batch, seq, hidden]
    Layer L_mid  → features_mid   [batch, seq, hidden]
    Layer L_high → features_high  [batch, seq, hidden]

  Draft Head:
    [features_low, features_mid, features_high]
              |
    Concatenate → [batch, seq, 3 * hidden]
              |
    FC (fusion) → [batch, seq, hidden]
              |
    RMSNorm
              |
    [fused, token_embedding] → Concatenate → [batch, seq, 2 * hidden]
              |
    FC (compress) → [batch, seq, hidden]
              |
    Decoder Layer (self-attention + MLP)
              |
    RMSNorm → LM Head → [batch, seq, vocab_size]
  ```

  ## Usage

      model = Eagle3.build(
        hidden_size: 4096,
        vocab_size: 32000,
        num_heads: 32,
        num_kv_heads: 8
      )

      # Input: token embeddings + 3 levels of target hidden states
      # Output: draft logits [batch, seq, vocab_size]

  ## References

  - "EAGLE-3: Scaling up Inference Acceleration of Large Language Models
    via Training-Time Test" (NeurIPS 2025) — https://arxiv.org/abs/2503.01840
  - EAGLE-1: "Speculative Sampling Requires Rethinking Feature Uncertainty" (ICML 2024)
  - EAGLE-2: "Faster Inference with Dynamic Draft Trees" (EMNLP 2024)
  """

  alias Edifice.Blocks.SwiGLU

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_intermediate_size 1024
  @default_num_feature_levels 3

  @doc """
  Build an EAGLE-3 draft head model.

  The model takes 4 inputs:
  - `"token_embeddings"` — embedded tokens `[batch, seq, embed_dim]`
  - `"features_low"` — low-level target hidden states `[batch, seq, hidden]`
  - `"features_mid"` — mid-level target hidden states `[batch, seq, hidden]`
  - `"features_high"` — high-level target hidden states `[batch, seq, hidden]`

  Returns logits `[batch, seq, vocab_size]`.

  ## Options

    - `:hidden_size` - Hidden dimension matching target model (default: 256)
    - `:vocab_size` - Draft vocabulary size (required)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_kv_heads` - Number of KV heads for GQA (default: 2)
    - `:intermediate_size` - MLP intermediate dimension (default: 1024)
    - `:num_feature_levels` - Number of extracted feature levels (default: 3)
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:intermediate_size, pos_integer()}
          | {:num_feature_levels, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:vocab_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    intermediate_size = Keyword.get(opts, :intermediate_size, @default_intermediate_size)
    num_feature_levels = Keyword.get(opts, :num_feature_levels, @default_num_feature_levels)

    head_dim = div(hidden_size, num_heads)

    # Inputs: token embeddings + multi-level features from target model
    token_emb = Axon.input("token_embeddings", shape: {nil, nil, hidden_size})
    features_low = Axon.input("features_low", shape: {nil, nil, hidden_size})
    features_mid = Axon.input("features_mid", shape: {nil, nil, hidden_size})
    features_high = Axon.input("features_high", shape: {nil, nil, hidden_size})

    # === Feature Fusion ===
    # Concatenate multi-level features: [batch, seq, num_levels * hidden_size]
    feature_inputs =
      case num_feature_levels do
        3 -> [features_low, features_mid, features_high]
        2 -> [features_low, features_high]
        1 -> [features_high]
      end

    fused =
      if num_feature_levels > 1 do
        concat_features(feature_inputs, num_feature_levels, hidden_size)
      else
        features_high
      end

    # FC fusion: [batch, seq, num_levels * hidden_size] -> [batch, seq, hidden_size]
    fused =
      if num_feature_levels > 1 do
        fused
        |> Axon.dense(hidden_size, name: "fusion_fc")
        |> Axon.layer_norm(name: "fusion_norm")
      else
        fused
      end

    # === Combine with Token Embedding ===
    # Concatenate: [fused, token_emb] -> [batch, seq, 2 * hidden_size]
    combined =
      Axon.layer(
        fn fused_t, emb_t, _opts ->
          Nx.concatenate([fused_t, emb_t], axis: 2)
        end,
        [fused, token_emb],
        name: "combine_features",
        op_name: :concatenate
      )

    # Compress: [batch, seq, 2 * hidden_size] -> [batch, seq, hidden_size]
    x = Axon.dense(combined, hidden_size, name: "compress_fc")

    # === Single Decoder Layer ===
    # Pre-norm self-attention
    normed = Axon.layer_norm(x, name: "decoder_norm1")

    # GQA self-attention with proper head expansion
    q = Axon.dense(normed, hidden_size, name: "decoder_q_proj")
    k = Axon.dense(normed, num_kv_heads * head_dim, name: "decoder_k_proj")
    v = Axon.dense(normed, num_kv_heads * head_dim, name: "decoder_v_proj")

    attended =
      Axon.layer(
        &gqa_attention_impl/4,
        [q, k, v],
        name: "decoder_attn",
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        head_dim: head_dim,
        op_name: :gqa_attention
      )

    attn_out = Axon.dense(attended, hidden_size, name: "decoder_out_proj")
    x = Axon.add(x, attn_out, name: "decoder_res1")

    # Pre-norm SwiGLU FFN
    normed2 = Axon.layer_norm(x, name: "decoder_norm2")

    ffn =
      SwiGLU.layer(normed2,
        hidden_size: hidden_size,
        inner_size: intermediate_size,
        name: "decoder_ffn"
      )

    x = Axon.add(x, ffn, name: "decoder_res2")

    # === LM Head ===
    x = Axon.layer_norm(x, name: "output_norm")
    Axon.dense(x, vocab_size, use_bias: false, name: "lm_head")
  end

  # Concatenate feature tensors along the last axis
  defp concat_features(inputs, num_levels, hidden_size) do
    case num_levels do
      2 ->
        [a, b] = inputs

        Axon.layer(
          fn a_t, b_t, _opts -> Nx.concatenate([a_t, b_t], axis: 2) end,
          [a, b],
          name: "concat_features",
          op_name: :concatenate
        )

      3 ->
        [a, b, c] = inputs

        Axon.layer(
          fn a_t, b_t, c_t, _opts -> Nx.concatenate([a_t, b_t, c_t], axis: 2) end,
          [a, b, c],
          name: "concat_features",
          expected_size: 3 * hidden_size,
          op_name: :concatenate
        )
    end
  end

  # GQA attention with KV head expansion for smaller draft heads
  defp gqa_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    q =
      q
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Reshape K, V: [batch, seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
    k =
      k
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Expand KV heads to match Q heads: repeat each KV head (num_heads / num_kv_heads) times
    groups = div(num_heads, num_kv_heads)

    {k, v} =
      if groups > 1 do
        # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        k_expanded =
          k
          |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
          |> Nx.broadcast({batch, num_kv_heads, groups, seq_len, head_dim})
          |> Nx.reshape({batch, num_heads, seq_len, head_dim})

        v_expanded =
          v
          |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
          |> Nx.broadcast({batch, num_kv_heads, groups, seq_len, head_dim})
          |> Nx.reshape({batch, num_heads, seq_len, head_dim})

        {k_expanded, v_expanded}
      else
        {k, v}
      end

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Vocabulary Mapping
  # ============================================================================

  @doc """
  Map draft token IDs to target vocabulary token IDs.

  EAGLE-3 can use a smaller draft vocabulary (e.g., 32K) for efficiency.
  The mapping tensor `d2t` maps each draft token index to its corresponding
  target vocabulary index.

  ## Parameters

    - `draft_tokens` - Draft token IDs `[batch, K]` or `[K]`
    - `d2t_map` - Mapping tensor `[draft_vocab_size]` → target token IDs

  ## Returns

    Target vocabulary token IDs with same shape as input.
  """
  @spec map_to_target_vocab(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def map_to_target_vocab(draft_tokens, d2t_map) do
    Nx.take(d2t_map, draft_tokens)
  end

  # ============================================================================
  # Accept-Reject
  # ============================================================================

  @doc """
  Accept-reject verification for speculative tokens.

  Compares draft tokens to verifier tokens and returns the count of
  consecutively accepted tokens from the start.

  ## Parameters

    - `draft_tokens` - Draft token IDs `[batch, K]` or `[K]`
    - `verifier_tokens` - Verifier token IDs `[batch, K]` or `[K]`

  ## Returns

    Integer tensor with accepted token count per batch element.
  """
  @spec accept_reject(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defdelegate accept_reject(draft_tokens, verifier_tokens),
    to: Edifice.Meta.SpeculativeDecoding

  @doc "Get the output size (vocab_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts) do
    Keyword.fetch!(opts, :vocab_size)
  end

  @doc """
  Get recommended layer indices for feature extraction from common models.

  Returns `{low, mid, high}` layer indices for extracting features
  from the target model.
  """
  @spec recommended_extract_layers(atom()) ::
          {non_neg_integer(), non_neg_integer(), non_neg_integer()}
  def recommended_extract_layers(model_size \\ :llama_8b) do
    case model_size do
      :llama_8b -> {8, 20, 31}
      :llama_70b -> {20, 50, 79}
      :llama_7b -> {8, 16, 31}
      :llama_13b -> {10, 25, 39}
      _ -> {8, 20, 31}
    end
  end
end
