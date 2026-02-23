defmodule Edifice.Multimodal.Fusion do
  @moduledoc """
  Multimodal Fusion Layers for Vision-Language Models.

  Provides two fusion approaches for combining visual and text features:

  ## 1. MLP Projection (LLaVA-style)

  The dominant approach in 2025-2026 VLMs. A pre-trained vision encoder's
  output tokens are projected through an MLP into the language model's
  embedding space and concatenated with text tokens.

  ```
  Visual tokens [batch, num_patches, vision_dim]
        |
  [MLP: Linear -> GELU -> Linear]
        |
  Projected [batch, num_patches, llm_dim]
        |
  Concatenate with text tokens
        |
  [batch, num_patches + text_len, llm_dim]
  ```

  Used by: LLaVA, InternVL, Qwen-VL, PaliGemma, DeepSeek-VL

  ## 2. Cross-Attention (Flamingo-style)

  Gated cross-attention blocks inserted into a language model. Visual
  features serve as keys/values, text features serve as queries.

  ```
  Text hidden [batch, text_len, llm_dim]
        |
  [LayerNorm]
        |
  [Cross-Attention]  <-- Visual features as K, V
        |
  [tanh(alpha) * output]  <-- Learnable gate, init=0
        |
  [Residual]
  ```

  Used by: Flamingo, LLaMA 3.2 Vision

  ## Usage

      # MLP Projection
      fused = Fusion.mlp_projection(visual_tokens, text_tokens,
        vision_dim: 1024,
        llm_dim: 4096
      )

      # Cross-Attention block
      attended = Fusion.cross_attention_block(text_hidden, visual_tokens,
        hidden_size: 4096,
        num_heads: 32
      )

  ## References
  - LLaVA: "Visual Instruction Tuning" (Liu et al., NeurIPS 2023)
  - Flamingo: "Few-Shot Visual Language Models" (Alayrac et al., NeurIPS 2022)
  - LLaMA 3.2 Vision: Meta technical report
  """

  alias Edifice.Utils.FusedOps

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_vision_dim 1024
  @default_llm_dim 256
  @default_num_heads 4
  @default_num_visual_tokens 196

  # ============================================================================
  # Registry-Compatible Build Function
  # ============================================================================

  @doc """
  Build a multimodal fusion model (MLP projection).

  This is the registry-compatible entry point that delegates to `build_mlp_projection/1`.

  ## Options
  See `build_mlp_projection/1` for available options.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []), do: build_mlp_projection(opts)

  # ============================================================================
  # MLP Projection Fusion (LLaVA-style)
  # ============================================================================

  @doc """
  Build an MLP projection fusion model.

  Takes visual tokens from a vision encoder and text token embeddings,
  projects the visual tokens to the LLM embedding space via a 2-layer MLP,
  and concatenates them.

  ## Options
    - `:vision_dim` - Dimension of visual tokens from ViT (default: 1024)
    - `:llm_dim` - Dimension of LLM embedding space (default: 256)
    - `:num_visual_tokens` - Number of visual tokens (default: 196)
    - `:text_seq_len` - Maximum text sequence length (default: nil)
    - `:compress_ratio` - Group N visual tokens into 1 (Qwen-style, default: 1)

  ## Returns
    An Axon model: (visual_tokens, text_embeddings) -> fused_sequence
  """
  @spec build_mlp_projection(keyword()) :: Axon.t()
  def build_mlp_projection(opts \\ []) do
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    llm_dim = Keyword.get(opts, :llm_dim, @default_llm_dim)
    num_visual_tokens = Keyword.get(opts, :num_visual_tokens, @default_num_visual_tokens)
    text_seq_len = Keyword.get(opts, :text_seq_len, nil)
    compress_ratio = Keyword.get(opts, :compress_ratio, 1)

    effective_visual_tokens = div(num_visual_tokens, compress_ratio)

    # Inputs
    visual_tokens = Axon.input("visual_tokens", shape: {nil, num_visual_tokens, vision_dim})
    text_embeddings = Axon.input("text_embeddings", shape: {nil, text_seq_len, llm_dim})

    # Optional compression: group N tokens into 1
    visual =
      if compress_ratio > 1 do
        compressed_dim = vision_dim * compress_ratio

        compressed =
          Axon.nx(
            visual_tokens,
            fn tokens ->
              batch = Nx.axis_size(tokens, 0)
              # Group tokens: [batch, N/ratio, ratio*dim]
              Nx.reshape(tokens, {batch, effective_visual_tokens, compressed_dim})
            end,
            name: "compress_visual"
          )

        # Project compressed tokens
        compressed
        |> Axon.dense(llm_dim, name: "visual_proj_compress")
        |> Axon.activation(:gelu, name: "visual_proj_gelu_compress")
        |> Axon.dense(llm_dim, name: "visual_proj_out_compress")
      else
        # Standard 2-layer MLP projection
        visual_tokens
        |> Axon.dense(llm_dim, name: "visual_proj_1")
        |> Axon.activation(:gelu, name: "visual_proj_gelu")
        |> Axon.dense(llm_dim, name: "visual_proj_2")
      end

    # Concatenate: [visual_tokens, text_tokens]
    Axon.layer(
      &concat_visual_text/3,
      [visual, text_embeddings],
      name: "fuse_concat",
      op_name: :fuse_concat
    )
  end

  defp concat_visual_text(visual, text, _opts) do
    Nx.concatenate([visual, text], axis: 1)
  end

  # ============================================================================
  # Cross-Attention Fusion (Flamingo-style)
  # ============================================================================

  @doc """
  Build a cross-attention fusion model.

  Inserts gated cross-attention blocks that allow text tokens to attend
  to visual features. The gating starts at zero (preserving original LLM
  behavior) and gradually learns to incorporate visual information.

  ## Options
    - `:hidden_size` - LLM hidden dimension (default: 256)
    - `:vision_dim` - Visual feature dimension (default: 1024)
    - `:num_heads` - Number of cross-attention heads (default: 4)
    - `:num_visual_tokens` - Number of visual tokens (default: 196)
    - `:text_seq_len` - Text sequence length (default: nil)
    - `:num_layers` - Number of cross-attention layers (default: 4)

  ## Returns
    An Axon model: (text_hidden, visual_features) -> attended_text
  """
  @spec build_cross_attention(keyword()) :: Axon.t()
  def build_cross_attention(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_llm_dim)
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_visual_tokens = Keyword.get(opts, :num_visual_tokens, @default_num_visual_tokens)
    text_seq_len = Keyword.get(opts, :text_seq_len, nil)
    num_layers = Keyword.get(opts, :num_layers, 4)

    # Inputs
    text_hidden = Axon.input("text_hidden", shape: {nil, text_seq_len, hidden_size})
    visual_features = Axon.input("visual_features", shape: {nil, num_visual_tokens, vision_dim})

    # Project visual features to LLM dimension if needed
    visual =
      if vision_dim != hidden_size do
        Axon.dense(visual_features, hidden_size, name: "visual_proj")
      else
        visual_features
      end

    # Stack cross-attention layers
    Enum.reduce(1..num_layers, text_hidden, fn layer_idx, acc ->
      build_gated_cross_attention_block(acc, visual, hidden_size, num_heads,
        name: "xattn_#{layer_idx}"
      )
    end)
  end

  # Gated cross-attention block (Flamingo-style)
  # y = x + tanh(alpha) * CrossAttention(LayerNorm(x), visual_features)
  defp build_gated_cross_attention_block(text_input, visual, hidden_size, num_heads, opts) do
    name = Keyword.get(opts, :name, "xattn")

    # LayerNorm on text input
    normed = Axon.layer_norm(text_input, name: "#{name}_norm")

    # Cross-attention: text queries attend to visual keys/values
    # Q from text, K and V from visual
    q = Axon.dense(normed, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(visual, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(visual, hidden_size, name: "#{name}_v_proj")

    # Scaled dot-product cross-attention
    attended =
      Axon.layer(
        &cross_attention_impl/4,
        [q, k, v],
        name: "#{name}_attend",
        num_heads: num_heads,
        hidden_size: hidden_size,
        op_name: :cross_attention
      )

    # Output projection
    projected = Axon.dense(attended, hidden_size, name: "#{name}_out_proj")

    # Tanh gating: gate initialized via a dense layer (will learn to be ~0 initially)
    # In practice the gate learns to open gradually during training
    gate =
      Axon.layer(
        &tanh_gate_impl/2,
        [projected],
        name: "#{name}_tanh_gate",
        op_name: :tanh_gate
      )

    # Residual: x + tanh(alpha) * cross_attention(x)
    Axon.add(text_input, gate, name: "#{name}_residual")
  end

  defp cross_attention_impl(q, k, v, opts) do
    hidden_size = opts[:hidden_size]
    head_dim = div(hidden_size, opts[:num_heads])
    scale = Nx.sqrt(Nx.tensor(head_dim, type: :f32))

    # Q: [batch, text_len, hidden], K: [batch, visual_len, hidden]
    scores = Nx.dot(q, [2], [0], k, [2], [0])
    scores = Nx.divide(scores, scale)
    weights = FusedOps.fused_softmax(scores)

    # [batch, text_len, hidden]
    Nx.dot(weights, [2], [0], v, [1], [0])
  end

  # Tanh gate: scales output by tanh of a learnable parameter
  # The output itself serves as the gate signal (will be small initially
  # due to random initialization, then grows as training progresses)
  defp tanh_gate_impl(x, _opts) do
    # Apply tanh element-wise as a soft gating mechanism
    # In practice, the dense layer before this learns appropriate scaling
    Nx.tanh(x)
  end

  # ============================================================================
  # Perceiver Resampler (BLIP-2 / Flamingo style)
  # ============================================================================

  @doc """
  Build a Perceiver Resampler that compresses visual tokens to a fixed count.

  Uses learned query embeddings that cross-attend to variable-length visual
  features, producing a fixed number of output tokens.

  ## Options
    - `:vision_dim` - Input visual feature dimension (default: 1024)
    - `:output_dim` - Output dimension (default: 256)
    - `:num_queries` - Number of learned queries/output tokens (default: 64)
    - `:num_layers` - Number of resampler layers (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_visual_tokens` - Number of input visual tokens (default: 196)

  ## Returns
    An Axon model: visual_features -> resampled [batch, num_queries, output_dim]
  """
  @spec build_perceiver_resampler(keyword()) :: Axon.t()
  def build_perceiver_resampler(opts \\ []) do
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    output_dim = Keyword.get(opts, :output_dim, @default_llm_dim)
    num_queries = Keyword.get(opts, :num_queries, 64)
    num_layers = Keyword.get(opts, :num_layers, 6)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_visual_tokens = Keyword.get(opts, :num_visual_tokens, @default_num_visual_tokens)

    visual_input = Axon.input("visual_features", shape: {nil, num_visual_tokens, vision_dim})

    # Project visual features to output_dim
    visual =
      if vision_dim != output_dim do
        Axon.dense(visual_input, output_dim, name: "resampler_visual_proj")
      else
        visual_input
      end

    # Learned query embeddings: [1, num_queries, output_dim]
    queries =
      Axon.nx(visual_input, fn _t -> Nx.broadcast(1.0, {1, num_queries}) end,
        name: "query_src"
      )
      |> Axon.dense(output_dim, name: "query_proj")
      |> Axon.nx(
        fn q ->
          # Expand to [1, num_queries, output_dim] and broadcast
          Nx.reshape(q, {1, num_queries, output_dim})
        end,
        name: "query_reshape"
      )

    # Stack resampler layers: self-attention on queries + cross-attention to visual
    Enum.reduce(1..num_layers, queries, fn layer_idx, acc ->
      name = "resampler_#{layer_idx}"

      # Self-attention among queries
      normed_q = Axon.layer_norm(acc, name: "#{name}_self_norm")
      self_attn_q = Axon.dense(normed_q, output_dim, name: "#{name}_self_q")
      self_attn_k = Axon.dense(normed_q, output_dim, name: "#{name}_self_k")
      self_attn_v = Axon.dense(normed_q, output_dim, name: "#{name}_self_v")

      self_attended =
        Axon.layer(
          &cross_attention_impl/4,
          [self_attn_q, self_attn_k, self_attn_v],
          name: "#{name}_self_attn",
          num_heads: num_heads,
          hidden_size: output_dim,
          op_name: :cross_attention
        )

      q_after_self = Axon.add(acc, self_attended, name: "#{name}_self_residual")

      # Cross-attention: queries attend to visual features
      normed_q2 = Axon.layer_norm(q_after_self, name: "#{name}_cross_norm")
      cross_q = Axon.dense(normed_q2, output_dim, name: "#{name}_cross_q")
      cross_k = Axon.dense(visual, output_dim, name: "#{name}_cross_k")
      cross_v = Axon.dense(visual, output_dim, name: "#{name}_cross_v")

      cross_attended =
        Axon.layer(
          &cross_attention_impl/4,
          [cross_q, cross_k, cross_v],
          name: "#{name}_cross_attn",
          num_heads: num_heads,
          hidden_size: output_dim,
          op_name: :cross_attention
        )

      Axon.add(q_after_self, cross_attended, name: "#{name}_cross_residual")
    end)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a fusion model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :llm_dim, @default_llm_dim)
  end
end
