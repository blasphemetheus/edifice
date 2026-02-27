defmodule Edifice.Robotics.OpenVLA do
  @moduledoc """
  OpenVLA: Open-Source Vision-Language-Action Model.

  OpenVLA is a vision-language-action model for robot manipulation that combines
  a DINOv2 vision encoder with a Llama-style language model backbone to predict
  discretized robot actions from images and text instructions.

  ## Key Innovation: Action Tokenization

  Instead of predicting continuous actions directly, OpenVLA discretizes each
  action dimension into bins (default 256), treating robot control as a
  sequence-to-sequence problem. The model autoregressively generates action
  tokens conditioned on visual and language tokens.

  ## Architecture

  ```
  Image [batch, C, H, W]           Text "pick up the red block"
        |                                    |
        v                                    v
  +================+                  +================+
  |   DINOv2 ViT   |                  |  LLM Tokenizer |
  +================+                  +================+
        |                                    |
        v                                    v
  Visual Tokens                       Text Token IDs
  [batch, num_patches, vision_dim]   [batch, text_len]
        |                                    |
        v                                    v
  [Vision-LM Projection]             [LLM Embedding]
  [batch, num_patches, hidden_dim]   [batch, text_len, hidden_dim]
        |                                    |
        +---------------+--------------------+
                        |
                        v
              [Concatenate: visual | text]
              [batch, num_patches + text_len, hidden_dim]
                        |
                        v
              +====================+
              |  Llama-style LLM   |
              |  (Decoder-Only)    |
              |  with Causal Mask  |
              +====================+
                        |
                        v
              Action Token Logits
              [batch, action_dim, num_bins]
                        |
                        v
              Argmax -> Detokenize
                        |
                        v
              Continuous Actions
              [batch, action_dim]
  ```

  ## Action Tokenization

  For action dimension `d` with range `[a_min, a_max]`:
  - Bin index = floor((action - a_min) / (a_max - a_min) * (num_bins - 1))
  - Action = a_min + (bin_index / (num_bins - 1)) * (a_max - a_min)

  Default: 256 bins per dimension, covering normalized range [-1, 1].

  ## Usage

      model = OpenVLA.build(
        image_size: 224,
        vision_encoder: :dino_v2,
        action_dim: 7,
        num_action_bins: 256,
        hidden_dim: 2048,
        num_layers: 24,
        num_heads: 16
      )

      # Forward pass: image + text -> action logits
      action_logits = model.predict(params, %{"image" => img, "text_tokens" => tokens})

      # Tokenize ground truth actions for training
      action_tokens = OpenVLA.tokenize_actions(actions, 256)

      # Compute loss (cross-entropy on action tokens only)
      loss = OpenVLA.vla_loss(action_logits, action_tokens)

      # Detokenize predictions for inference
      actions = OpenVLA.detokenize_actions(predicted_tokens, 256)

  ## References

  - Paper: "OpenVLA: An Open-Source Vision-Language-Action Model"
    (Kim et al., 2024) - https://arxiv.org/abs/2406.09246
  - Project: https://openvla.github.io/
  """

  import Nx.Defn

  alias Edifice.Blocks.{FFN, RMSNorm}

  # Default hyperparameters
  @default_image_size 224
  @default_patch_size 14
  @default_in_channels 3
  @default_vision_dim 384
  @default_hidden_dim 2048
  @default_num_layers 24
  @default_num_heads 16
  @default_num_kv_heads 4
  @default_action_dim 7
  @default_num_action_bins 256
  @default_max_text_len 64
  @default_dropout 0.1
  @default_action_min -1.0
  @default_action_max 1.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:image_size, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:vision_dim, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:action_dim, pos_integer()}
          | {:num_action_bins, pos_integer()}
          | {:max_text_len, pos_integer()}
          | {:dropout, float()}
          | {:rope, boolean()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an OpenVLA model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - ViT patch size (default: 14)
    - `:in_channels` - Number of input channels (default: 3)
    - `:vision_dim` - Vision encoder output dimension (default: 384)
    - `:hidden_dim` - LLM hidden dimension (default: 2048)
    - `:num_layers` - Number of LLM layers (default: 24)
    - `:num_heads` - Number of attention heads (default: 16)
    - `:num_kv_heads` - Number of KV heads for GQA (default: 4)
    - `:action_dim` - Robot action dimension (default: 7)
    - `:num_action_bins` - Number of bins for action discretization (default: 256)
    - `:max_text_len` - Maximum text sequence length (default: 64)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:rope` - Apply RoPE to attention (default: false)

  ## Returns

    An Axon model with inputs `"image"` and `"text_tokens"`, outputting
    `[batch, action_dim, num_action_bins]` logits.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    action_dim = Keyword.get(opts, :action_dim, @default_action_dim)
    num_action_bins = Keyword.get(opts, :num_action_bins, @default_num_action_bins)
    max_text_len = Keyword.get(opts, :max_text_len, @default_max_text_len)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_rope = Keyword.get(opts, :rope, false)

    num_patches = div(image_size, patch_size) * div(image_size, patch_size)

    # Inputs
    image = Axon.input("image", shape: {nil, in_channels, image_size, image_size})
    text_tokens = Axon.input("text_tokens", shape: {nil, max_text_len, hidden_dim})

    # Vision encoder: DINOv2-style ViT (without DINO head, just features)
    visual_tokens = encode_image(image, opts)

    # Project visual tokens to LLM dimension
    visual_proj =
      if vision_dim != hidden_dim do
        visual_tokens
        |> Axon.dense(hidden_dim, name: "vision_projection_1")
        |> Axon.activation(:gelu, name: "vision_projection_gelu")
        |> Axon.dense(hidden_dim, name: "vision_projection_2")
      else
        visual_tokens
      end

    # Concatenate visual and text tokens: [visual | text]
    # This creates a unified sequence for the LLM
    fused =
      Axon.layer(
        &concat_visual_text/3,
        [visual_proj, text_tokens],
        name: "fuse_visual_text",
        num_patches: num_patches,
        op_name: :vla_fuse
      )

    # Llama-style decoder-only LLM (total sequence = visual + text tokens)
    _total_seq_len = num_patches + max_text_len

    x =
      Enum.reduce(0..(num_layers - 1), fused, fn layer_idx, acc ->
        build_llm_block(acc, hidden_dim, num_heads, num_kv_heads, dropout, use_rope, layer_idx)
      end)

    # Final normalization
    x = RMSNorm.layer(x, hidden_size: hidden_dim, name: "final_norm")

    # Extract the last `action_dim` positions for action prediction
    # In OpenVLA, action tokens are predicted after the combined visual+text tokens
    # We use learned query positions appended to the sequence
    action_queries = build_action_queries(text_tokens, action_dim, hidden_dim)

    # Cross-attend action queries to the encoded sequence
    action_features = build_action_decoder(action_queries, x, hidden_dim, num_heads, dropout)

    # Project to action bin logits: [batch, action_dim, num_action_bins]
    Axon.dense(action_features, num_action_bins, name: "action_head")
  end

  @doc """
  Encode an image through the vision encoder (DINOv2-style ViT).

  Returns visual tokens without the DINO projection head.

  ## Parameters

    - `image` - Axon node for image input
    - `opts` - Build options

  ## Returns

    Axon node with shape `[batch, num_patches, vision_dim]`.
  """
  @spec encode_image(Axon.t(), keyword()) :: Axon.t()
  def encode_image(image, opts) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    num_vit_heads = max(div(vision_dim, 64), 4)
    num_vit_layers = 12

    num_patches = div(image_size, patch_size) * div(image_size, patch_size)
    mlp_hidden = vision_dim * 4

    # Patch embedding
    x =
      Edifice.Blocks.PatchEmbed.layer(image,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: vision_dim,
        name: "vit_patch_embed"
      )

    # Add learnable position embeddings
    x = add_position_embedding(x, num_patches, vision_dim, name: "vit_pos_embed")

    # ViT transformer blocks (no CLS token needed for VLA)
    x =
      Enum.reduce(0..(num_vit_layers - 1), x, fn idx, acc ->
        vit_block(acc, vision_dim, num_vit_heads, mlp_hidden, name: "vit_block_#{idx}")
      end)

    # Final layer norm
    Axon.layer_norm(x, name: "vit_final_norm")
  end

  # ViT transformer block
  defp vit_block(input, embed_dim, num_heads, mlp_hidden, opts) do
    name = Keyword.get(opts, :name, "vit_block")

    # Pre-norm self-attention
    normed = Axon.layer_norm(input, name: "#{name}_norm1")
    attended = vit_self_attention(normed, embed_dim, num_heads, name: "#{name}_attn")
    x = Axon.add(input, attended, name: "#{name}_residual1")

    # Pre-norm MLP
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      normed2
      |> Axon.dense(mlp_hidden, name: "#{name}_mlp_fc1")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> Axon.dense(embed_dim, name: "#{name}_mlp_fc2")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  # ViT self-attention (bidirectional, no causal mask)
  defp vit_self_attention(input, embed_dim, num_heads, opts) do
    name = Keyword.get(opts, :name, "vit_attn")
    head_dim = div(embed_dim, num_heads)

    qkv = Axon.dense(input, embed_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.layer(
        &vit_mha_impl/2,
        [qkv],
        name: "#{name}_compute",
        embed_dim: embed_dim,
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :vit_mha
      )

    Axon.dense(attended, embed_dim, name: "#{name}_proj")
  end

  defp vit_mha_impl(qkv, opts) do
    embed_dim = opts[:embed_dim]
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(qkv, 0)
    seq_len = Nx.axis_size(qkv, 1)

    q = Nx.slice_along_axis(qkv, 0, embed_dim, axis: 2)
    k = Nx.slice_along_axis(qkv, embed_dim, embed_dim, axis: 2)
    v = Nx.slice_along_axis(qkv, embed_dim * 2, embed_dim, axis: 2)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # No causal mask for ViT (bidirectional attention)
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Add learnable position embeddings
  defp add_position_embedding(input, seq_len, embed_dim, opts) do
    name = Keyword.get(opts, :name, "pos_embed")

    pos_source =
      Axon.nx(input, fn _t -> Nx.iota({1, seq_len}, axis: 1) |> Nx.divide(seq_len) end,
        name: "#{name}_src"
      )

    pos_proj = Axon.dense(pos_source, embed_dim, name: "#{name}_proj")

    Axon.layer(
      fn input_tensor, pos_embed, _opts -> Nx.add(input_tensor, pos_embed) end,
      [input, pos_proj],
      name: "#{name}_add",
      op_name: :add_pos_embed
    )
  end

  # Concatenate visual and text tokens
  defp concat_visual_text(visual, text, _opts) do
    Nx.concatenate([visual, text], axis: 1)
  end

  # Build a single LLM block (Llama-style)
  defp build_llm_block(input, hidden_dim, num_heads, num_kv_heads, dropout, use_rope, layer_idx) do
    name = "llm_block_#{layer_idx}"
    head_dim = div(hidden_dim, num_heads)

    # Pre-norm GQA attention
    normed = RMSNorm.layer(input, hidden_size: hidden_dim, name: "#{name}_attn_norm")

    attn_out =
      build_gqa_attention(normed, hidden_dim, num_heads, num_kv_heads, head_dim, use_rope, name)

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # Pre-norm SwiGLU FFN
    normed2 = RMSNorm.layer(x, hidden_size: hidden_dim, name: "#{name}_ffn_norm")
    ffn_out = FFN.gated_layer(normed2, hidden_size: hidden_dim, name: "#{name}_ffn")
    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_dropout")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # Build GQA attention for LLM
  defp build_gqa_attention(input, hidden_dim, num_heads, num_kv_heads, head_dim, use_rope, name) do
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    q_proj = Axon.dense(input, q_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, kv_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, kv_dim, name: "#{name}_v_proj")

    output =
      Axon.layer(
        &gqa_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        head_dim: head_dim,
        rope: use_rope,
        op_name: :vla_gqa
      )

    Axon.dense(output, hidden_dim, name: "#{name}_out_proj")
  end

  # GQA attention implementation with causal masking
  defp gqa_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    head_dim = opts[:head_dim]
    use_rope = opts[:rope] || false

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)
    heads_per_group = div(num_heads, num_kv_heads)

    q =
      q
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k =
      k
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE if enabled
    {q, k} =
      if use_rope do
        q_flat = Nx.reshape(q, {batch * num_heads, seq_len, head_dim})
        k_flat = Nx.reshape(k, {batch * num_kv_heads, seq_len, head_dim})

        {q_rot, k_rot} = Edifice.Blocks.RoPE.apply_rotary(q_flat, k_flat)

        q_out = Nx.reshape(q_rot, {batch, num_heads, seq_len, head_dim})
        k_out = Nx.reshape(k_rot, {batch, num_kv_heads, seq_len, head_dim})
        {q_out, k_out}
      else
        {q, k}
      end

    # Repeat K, V for GQA
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

    # Scaled dot-product attention with causal mask
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
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

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Build learnable action queries
  defp build_action_queries(text_input, action_dim, hidden_dim) do
    # Use a learnable parameter for action queries
    query_embed =
      Axon.param("action_query_embed", {action_dim, hidden_dim}, initializer: :glorot_uniform)

    Axon.layer(
      fn text, q_embed, _opts ->
        # Get batch size from text input and broadcast queries
        batch = Nx.axis_size(text, 0)
        Nx.broadcast(Nx.new_axis(q_embed, 0), {batch, action_dim, hidden_dim})
      end,
      [text_input, query_embed],
      name: "broadcast_action_queries",
      op_name: :broadcast_queries
    )
  end

  # Build action decoder: cross-attention from action queries to encoded sequence
  defp build_action_decoder(action_queries, encoded_seq, hidden_dim, num_heads, dropout) do
    head_dim = div(hidden_dim, num_heads)

    # Self-attention among action queries
    q_normed = Axon.layer_norm(action_queries, name: "action_self_attn_norm")

    self_attn =
      build_self_attention(q_normed, hidden_dim, num_heads, head_dim, "action_self_attn")

    self_attn = maybe_dropout(self_attn, dropout, "action_self_attn_dropout")
    x = Axon.add(action_queries, self_attn, name: "action_self_attn_residual")

    # Cross-attention: action queries attend to encoded visual+text sequence
    x_normed = Axon.layer_norm(x, name: "action_cross_attn_norm")

    cross_attn =
      Edifice.Blocks.CrossAttention.layer(x_normed, encoded_seq,
        hidden_size: hidden_dim,
        num_heads: num_heads,
        name: "action_cross_attn"
      )

    cross_attn = maybe_dropout(cross_attn, dropout, "action_cross_attn_dropout")
    x = Axon.add(x, cross_attn, name: "action_cross_attn_residual")

    # FFN
    x_normed2 = Axon.layer_norm(x, name: "action_ffn_norm")
    ffn_out = FFN.gated_layer(x_normed2, hidden_size: hidden_dim, name: "action_ffn")
    ffn_out = maybe_dropout(ffn_out, dropout, "action_ffn_dropout")
    Axon.add(x, ffn_out, name: "action_ffn_residual")
  end

  # Self-attention for action queries (bidirectional)
  defp build_self_attention(input, hidden_dim, num_heads, head_dim, name) do
    q = Axon.dense(input, hidden_dim, name: "#{name}_q")
    k = Axon.dense(input, hidden_dim, name: "#{name}_k")
    v = Axon.dense(input, hidden_dim, name: "#{name}_v")

    attended =
      Axon.layer(
        &self_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :action_self_attn
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  defp self_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Action Tokenization
  # ============================================================================

  @doc """
  Tokenize continuous actions into discrete bin indices.

  Discretizes each action dimension into `num_bins` uniform bins.

  ## Parameters

    - `actions` - Continuous actions tensor `[batch, action_dim]` or `[action_dim]`
    - `num_bins` - Number of bins per dimension (default: 256)

  ## Options

    - `:action_min` - Minimum action value (default: -1.0)
    - `:action_max` - Maximum action value (default: 1.0)

  ## Returns

    Integer tensor of bin indices with same shape as input.
  """
  @spec tokenize_actions(Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def tokenize_actions(actions, num_bins \\ @default_num_action_bins, opts \\ []) do
    action_min = Keyword.get(opts, :action_min, @default_action_min)
    action_max = Keyword.get(opts, :action_max, @default_action_max)

    tokenize_actions_impl(actions, num_bins, action_min, action_max)
  end

  defnp tokenize_actions_impl(actions, num_bins, action_min, action_max) do
    # Normalize to [0, 1]
    normalized = (actions - action_min) / (action_max - action_min)

    # Clamp to valid range
    normalized = Nx.clip(normalized, 0.0, 1.0)

    # Convert to bin indices
    bin_indices = Nx.floor(normalized * (num_bins - 1))
    Nx.as_type(bin_indices, :s32)
  end

  @doc """
  Detokenize discrete bin indices back to continuous actions.

  Converts bin indices to the center of each bin.

  ## Parameters

    - `action_tokens` - Integer tensor of bin indices `[batch, action_dim]`
    - `num_bins` - Number of bins per dimension (default: 256)

  ## Options

    - `:action_min` - Minimum action value (default: -1.0)
    - `:action_max` - Maximum action value (default: 1.0)

  ## Returns

    Float tensor of continuous actions with same shape as input.
  """
  @spec detokenize_actions(Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def detokenize_actions(action_tokens, num_bins \\ @default_num_action_bins, opts \\ []) do
    action_min = Keyword.get(opts, :action_min, @default_action_min)
    action_max = Keyword.get(opts, :action_max, @default_action_max)

    detokenize_actions_impl(action_tokens, num_bins, action_min, action_max)
  end

  defnp detokenize_actions_impl(action_tokens, num_bins, action_min, action_max) do
    # Convert to float
    tokens_float = Nx.as_type(action_tokens, :f32)

    # Map to [0, 1] using bin centers
    normalized = (tokens_float + 0.5) / num_bins

    # Scale to action range
    action_min + normalized * (action_max - action_min)
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute VLA loss (cross-entropy on action tokens).

  ## Parameters

    - `logits` - Predicted action logits `[batch, action_dim, num_bins]`
    - `target_tokens` - Ground truth action tokens `[batch, action_dim]`

  ## Returns

    Scalar loss tensor.
  """
  @spec vla_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def vla_loss(logits, target_tokens) do
    vla_loss_impl(logits, target_tokens)
  end

  defnp vla_loss_impl(logits, target_tokens) do
    # logits: [batch, action_dim, num_bins]
    # target_tokens: [batch, action_dim] (integer indices)
    batch = Nx.axis_size(logits, 0)
    action_dim = Nx.axis_size(logits, 1)
    num_bins = Nx.axis_size(logits, 2)

    # Reshape for cross-entropy: [batch * action_dim, num_bins]
    logits_flat = Nx.reshape(logits, {batch * action_dim, num_bins})
    targets_flat = Nx.reshape(target_tokens, {batch * action_dim})

    # Log-softmax
    max_logits = Nx.reduce_max(logits_flat, axes: [1], keep_axes: true)

    log_probs =
      logits_flat - max_logits -
        Nx.log(Nx.sum(Nx.exp(logits_flat - max_logits), axes: [1], keep_axes: true))

    # Gather log probabilities at target indices
    # One-hot encode targets
    targets_one_hot =
      Nx.equal(
        Nx.iota({batch * action_dim, num_bins}, axis: 1),
        Nx.reshape(targets_flat, {batch * action_dim, 1})
      )

    # Select log probs at target positions
    target_log_probs = Nx.sum(log_probs * targets_one_hot, axes: [1])

    # Mean negative log likelihood
    -Nx.mean(target_log_probs)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an OpenVLA model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, @default_action_dim)
    num_bins = Keyword.get(opts, :num_action_bins, @default_num_action_bins)
    action_dim * num_bins
  end

  @doc """
  Calculate approximate parameter count for an OpenVLA model.
  """
  @spec param_count(keyword()) :: pos_integer()
  def param_count(opts) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    vision_dim = Keyword.get(opts, :vision_dim, @default_vision_dim)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    action_dim = Keyword.get(opts, :action_dim, @default_action_dim)
    num_bins = Keyword.get(opts, :num_action_bins, @default_num_action_bins)

    _num_patches = div(image_size, patch_size) * div(image_size, patch_size)
    vit_layers = 12

    head_dim = div(hidden_dim, num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    ffn_inner = hidden_dim * 4

    # ViT encoder
    vit_patch_embed = 3 * patch_size * patch_size * vision_dim

    vit_per_layer =
      vision_dim * 3 * vision_dim + vision_dim * vision_dim + vision_dim * 4 * vision_dim * 2

    vit_total = vit_patch_embed + vit_layers * vit_per_layer

    # Vision projection
    vision_proj = vision_dim * hidden_dim * 2

    # LLM layers
    llm_per_layer =
      hidden_dim * q_dim + hidden_dim * kv_dim * 2 + q_dim * hidden_dim +
        3 * hidden_dim * ffn_inner

    llm_total = num_layers * llm_per_layer

    # Action head
    action_params = action_dim * hidden_dim + hidden_dim * num_bins

    vit_total + vision_proj + llm_total + action_params
  end

  @doc """
  Recommended default configuration for OpenVLA.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      image_size: 224,
      patch_size: 14,
      vision_dim: 384,
      hidden_dim: 2048,
      num_layers: 24,
      num_heads: 16,
      num_kv_heads: 4,
      action_dim: 7,
      num_action_bins: 256,
      max_text_len: 64,
      dropout: 0.1
    ]
  end

  @doc """
  Get small model configuration (for testing/prototyping).
  """
  @spec small_config() :: keyword()
  def small_config do
    [
      image_size: 112,
      patch_size: 14,
      vision_dim: 192,
      hidden_dim: 256,
      num_layers: 4,
      num_heads: 4,
      num_kv_heads: 2,
      action_dim: 7,
      num_action_bins: 256,
      max_text_len: 32,
      dropout: 0.1
    ]
  end
end
