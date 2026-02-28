defmodule Edifice.Meta.MixtureOfTransformers do
  @moduledoc """
  Mixture of Transformers (MoT): Modality-Sparse Multi-Modal Architecture.

  Implements MoT from "Mixture-of-Transformers: A Sparse and Scalable
  Architecture for Multi-Modal Foundation Models" (Liang et al., TMLR 2025).
  Decouples all non-embedding transformer parameters by modality — each
  modality gets its own attention projections (Q/K/V/O), layer norms, and
  FFN — while sharing only the core attention computation across modalities.

  ## Key Innovation

  In a standard transformer, FFN accounts for ~67% of parameters and attention
  projections for ~33%. MoT makes each token activate only 1/M of these
  parameters (where M = number of modalities), while the shared global
  self-attention lets all modalities attend to each other. This matches dense
  baseline quality at 55.8% FLOPs (text+image) or 37.2% (text+image+speech).

  Unlike Mixture-of-Experts, routing is deterministic (based on known modality
  of each token), not learned. No load balancing loss needed.

  ## Architecture

  ```
  Inputs: tokens [B, S, D], modality_mask [B, S, M]
        |
  +------------------------------------------------------------+
  | Per MoT Layer:                                              |
  |   For each modality m:                                      |
  |     LN_m(x) → Q_m, K_m, V_m (modality-specific projections)|
  |   Scatter per-modality QKV → full-sequence Q, K, V          |
  |   Global Self-Attention (shared, with RoPE)                 |
  |   Gather per-modality attention output                      |
  |   For each modality m:                                      |
  |     O_m projection → + residual                             |
  |     LN2_m(x) → FFN_m(x) → + residual                       |
  +------------------------------------------------------------+
        | (repeat num_layers)
  +------------------------------------------------------------+
  | Final RMSNorm → Linear → vocab_size                         |
  +------------------------------------------------------------+
        |
  Output: logits [B, S, vocab_size]
  ```

  ## Dense Path Implementation

  For Axon's static graph, this uses the "dense path": all modality-specific
  projections run on all tokens, then mask-multiply zeroes out wrong-modality
  outputs. This simplifies the graph at the cost of some redundant FLOPs,
  which is acceptable for the small test configurations and avoids dynamic
  gather/scatter complexity.

  ## Usage

      model = MixtureOfTransformers.build(
        vocab_size: 32000,
        hidden_size: 512,
        num_heads: 8,
        num_layers: 6,
        intermediate_size: 1024,
        num_modalities: 2
      )

  ## References

  - Liang et al., "Mixture-of-Transformers: A Sparse and Scalable Architecture
    for Multi-Modal Foundation Models"
    (TMLR 2025) — https://arxiv.org/abs/2411.04996
  """

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_layers 6
  @default_intermediate_size nil
  @default_num_modalities 2
  @default_rms_norm_eps 1.0e-5
  @default_seq_len 64

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:intermediate_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_modalities, pos_integer()}
          | {:rms_norm_eps, float()}
          | {:seq_len, pos_integer()}
          | {:vocab_size, pos_integer()}

  @doc """
  Build the MoT modality-sparse multi-modal transformer.

  ## Options

    - `:vocab_size` - Token vocabulary size (required)
    - `:hidden_size` - Model hidden dimension (default: 256)
    - `:num_heads` - Attention heads, shared across modalities (default: 8)
    - `:num_layers` - Number of MoT layers (default: 6)
    - `:intermediate_size` - FFN hidden size per modality (default: hidden_size * 4)
    - `:num_modalities` - Number of modalities (default: 2)
    - `:seq_len` - Maximum sequence length (default: 64)
    - `:rms_norm_eps` - RMSNorm epsilon (default: 1.0e-5)

  ## Returns

    An Axon model taking `tokens` `[batch, seq_len]` and `modality_mask`
    `[batch, seq_len, num_modalities]`, outputting logits
    `[batch, seq_len, vocab_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_modalities = Keyword.get(opts, :num_modalities, @default_num_modalities)
    seq_len = Keyword.get(opts, :seq_len, @default_seq_len)
    rms_norm_eps = Keyword.get(opts, :rms_norm_eps, @default_rms_norm_eps)

    intermediate_size =
      Keyword.get(opts, :intermediate_size, @default_intermediate_size) || hidden_size * 4

    # Inputs
    tokens = Axon.input("tokens", shape: {nil, seq_len})
    modality_mask = Axon.input("modality_mask", shape: {nil, seq_len, num_modalities})

    # Token embedding
    x = build_token_embedding(tokens, vocab_size, hidden_size)

    # MoT layers
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_mot_layer(acc, modality_mask,
          hidden_size: hidden_size,
          num_heads: num_heads,
          intermediate_size: intermediate_size,
          num_modalities: num_modalities,
          rms_norm_eps: rms_norm_eps,
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

  # ===========================================================================
  # MoT Layer
  # ===========================================================================

  defp build_mot_layer(input, modality_mask, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    num_modalities = opts[:num_modalities]
    rms_norm_eps = opts[:rms_norm_eps]
    intermediate_size = opts[:intermediate_size]
    name = opts[:name]

    # ---- Attention sub-layer with per-modality projections ----

    # Per-modality pre-attention norm + QKV projections (dense path)
    # Each modality has its own LN + Q/K/V projections
    # We compute all and mask-combine them
    {q_combined, k_combined, v_combined} =
      build_modality_qkv(input, modality_mask,
        hidden_size: hidden_size,
        num_modalities: num_modalities,
        rms_norm_eps: rms_norm_eps,
        name: "#{name}_attn"
      )

    # Shared global self-attention
    attn_out =
      build_shared_attention(q_combined, k_combined, v_combined,
        hidden_size: hidden_size,
        num_heads: num_heads,
        name: "#{name}_attn"
      )

    # Per-modality output projections
    attn_projected =
      build_modality_output_proj(attn_out, modality_mask,
        hidden_size: hidden_size,
        num_modalities: num_modalities,
        name: "#{name}_attn"
      )

    # Residual
    x = Axon.add(input, attn_projected, name: "#{name}_attn_residual")

    # ---- FFN sub-layer with per-modality FFNs ----

    ffn_out =
      build_modality_ffn(x, modality_mask,
        hidden_size: hidden_size,
        intermediate_size: intermediate_size,
        num_modalities: num_modalities,
        rms_norm_eps: rms_norm_eps,
        name: "#{name}_ffn"
      )

    # Residual
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ===========================================================================
  # Per-Modality QKV (Dense Path)
  # ===========================================================================

  # For each modality m, compute LN_m(x) → Q_m, K_m, V_m on ALL tokens,
  # then mask-multiply by modality_mask[:, :, m] to zero out wrong tokens,
  # then sum across modalities.
  defp build_modality_qkv(input, modality_mask, opts) do
    hidden_size = opts[:hidden_size]
    num_modalities = opts[:num_modalities]
    rms_norm_eps = opts[:rms_norm_eps]
    name = opts[:name]

    # Build per-modality Q, K, V
    qkv_per_modality =
      for m <- 0..(num_modalities - 1) do
        # Per-modality layer norm
        normed = build_rms_norm(input, rms_norm_eps, "#{name}_m#{m}_norm")

        # Per-modality Q, K, V projections
        q_m = Axon.dense(normed, hidden_size, name: "#{name}_m#{m}_q")
        k_m = Axon.dense(normed, hidden_size, name: "#{name}_m#{m}_k")
        v_m = Axon.dense(normed, hidden_size, name: "#{name}_m#{m}_v")

        {q_m, k_m, v_m}
      end

    # Extract lists
    q_list = Enum.map(qkv_per_modality, &elem(&1, 0))
    k_list = Enum.map(qkv_per_modality, &elem(&1, 1))
    v_list = Enum.map(qkv_per_modality, &elem(&1, 2))

    # Mask-combine: for each projection, sum(proj_m * mask[:,:,m:m+1])
    q_combined = mask_combine(q_list, modality_mask, num_modalities, "#{name}_q")
    k_combined = mask_combine(k_list, modality_mask, num_modalities, "#{name}_k")
    v_combined = mask_combine(v_list, modality_mask, num_modalities, "#{name}_v")

    {q_combined, k_combined, v_combined}
  end

  # Mask a single modality tensor by its mask slice, producing masked_m
  defp mask_modality(tensor, modality_mask, modality_idx, name) do
    Axon.layer(
      &mask_modality_impl/3,
      [tensor, modality_mask],
      name: name,
      modality_idx: modality_idx,
      op_name: :mask_modality
    )
  end

  defp mask_modality_impl(tensor, mask, opts) do
    m = opts[:modality_idx]
    # mask: [batch, seq, num_modalities] -> slice column m -> [batch, seq, 1]
    mask_m = Nx.slice_along_axis(mask, m, 1, axis: 2)
    Nx.multiply(tensor, mask_m)
  end

  # Combine per-modality tensors using mask weighting:
  # result = sum_m(tensor_m * mask[:, :, m:m+1])
  defp mask_combine(tensor_list, modality_mask, _num_modalities, name) do
    # Mask each modality's tensor separately, then sum
    masked_tensors =
      tensor_list
      |> Enum.with_index()
      |> Enum.map(fn {tensor, m} ->
        mask_modality(tensor, modality_mask, m, "#{name}_mask_m#{m}")
      end)

    # Sum all masked tensors
    Enum.reduce(masked_tensors, fn masked, acc ->
      Axon.add(masked, acc)
    end)
  end

  # ===========================================================================
  # Shared Global Self-Attention
  # ===========================================================================

  defp build_shared_attention(q, k, v, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    name = opts[:name]
    head_dim = div(hidden_size, num_heads)

    Axon.layer(
      &shared_attention_impl/4,
      [q, k, v],
      name: "#{name}_compute",
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :shared_attention
    )
  end

  defp shared_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q =
      q
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k =
      k
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Weighted sum
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ===========================================================================
  # Per-Modality Output Projection
  # ===========================================================================

  defp build_modality_output_proj(attn_out, modality_mask, opts) do
    hidden_size = opts[:hidden_size]
    num_modalities = opts[:num_modalities]
    name = opts[:name]

    # Each modality has its own O projection
    projected_list =
      for m <- 0..(num_modalities - 1) do
        Axon.dense(attn_out, hidden_size, name: "#{name}_m#{m}_o")
      end

    # Mask-combine
    mask_combine(projected_list, modality_mask, num_modalities, "#{name}_o")
  end

  # ===========================================================================
  # Per-Modality FFN
  # ===========================================================================

  defp build_modality_ffn(input, modality_mask, opts) do
    hidden_size = opts[:hidden_size]
    intermediate_size = opts[:intermediate_size]
    num_modalities = opts[:num_modalities]
    rms_norm_eps = opts[:rms_norm_eps]
    name = opts[:name]

    # Each modality has its own LN + FFN
    ffn_list =
      for m <- 0..(num_modalities - 1) do
        normed = build_rms_norm(input, rms_norm_eps, "#{name}_m#{m}_norm")

        normed
        |> Axon.dense(intermediate_size, name: "#{name}_m#{m}_up")
        |> Axon.activation(:silu, name: "#{name}_m#{m}_silu")
        |> Axon.dense(hidden_size, name: "#{name}_m#{m}_down")
      end

    # Mask-combine
    mask_combine(ffn_list, modality_mask, num_modalities, "#{name}")
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
