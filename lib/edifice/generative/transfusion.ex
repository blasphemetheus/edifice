defmodule Edifice.Generative.Transfusion do
  @moduledoc """
  Transfusion: Unified Autoregressive Text + Diffusion Image Generation.

  <!-- verified: true, date: 2026-02-23 -->

  A single transformer model that jointly handles discrete text tokens
  (autoregressive next-token prediction) and continuous image patches
  (denoising diffusion) in one shared backbone.

  ## Key Innovation: Mixed Attention Mask

  Text tokens and image patches share the same transformer layers, but
  attend with different masks:

  - **Text positions** (causal): each token sees only preceding tokens
  - **Image positions** (bidirectional within image): each patch sees all
    other patches in the same image, plus all preceding text context

  Combined rule:
  ```
  mask[i, j] = 1  if  j ≤ i                        # causal for text
               OR  (image[i] AND image[j])            # bidir within image
  ```

  ## Architecture

  ```
  Inputs: sequence [batch, seq_len, embed_dim]  (text embeddings + image patches)
          modality_mask [batch, seq_len]          (0=text, 1=image)
          timestep [batch]                        (diffusion step for image)
          |
          v
  Modality type embedding  (learnable TEXT / IMAGE vectors added to tokens)
          |
          v
  Input projection  →  hidden_size
          |
          v
  +----------------------------------------------+
  |  Transfusion Block  ×  num_layers             |
  |                                               |
  |  Add time_embed at image positions            |
  |  LayerNorm  →  Mixed Attention  →  Residual   |
  |  LayerNorm  →  FFN (GELU)       →  Residual   |
  +----------------------------------------------+
          |
  Final LayerNorm
          |
       ┌──┴──┐
  text_head    image_head
  [b,s,V]     [b,s,P]
  ```

  ## Dual Loss

  - **Text tokens**: cross-entropy against next-token targets
  - **Image patches**: MSE between predicted and target noise/velocity
  - Total: `text_weight * L_CE + image_weight * L_MSE`

  ## Usage

      model = Transfusion.build(
        embed_dim: 64,
        hidden_size: 256,
        num_heads: 8,
        num_layers: 6,
        vocab_size: 32_000,
        patch_dim: 64
      )

      # Build the mixed attention mask for a 20-token text + 16-patch image
      mask = Transfusion.build_mixed_mask(20, 16)

      # Compute training loss
      loss = Transfusion.transfusion_loss(text_logits, image_pred, %{
        text_targets:  token_ids,      # [batch, seq_len] integer indices
        image_targets: noise_targets,  # [batch, seq_len, patch_dim]
        text_mask:     text_positions, # [batch, seq_len] float, 1 at text positions
        image_mask:    image_positions # [batch, seq_len] float, 1 at image positions
      })

  ## References

  - Paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
  - Authors: Chunting Zhou et al., Meta (2024)
  - arXiv: https://arxiv.org/abs/2408.11039
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_layers 6
  @default_vocab_size 32_000
  @default_patch_dim 64
  @default_dropout 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Transfusion model for joint text + image generation.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:vocab_size` - Text vocabulary size for CE head (default: 32_000)
    - `:patch_dim` - Image patch feature dimension for diffusion head (default: 64)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    `Axon.container(%{text_logits: [batch, seq, vocab_size], image_pred: [batch, seq, patch_dim]})`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:patch_dim, pos_integer()}
          | {:vocab_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    patch_dim = Keyword.get(opts, :patch_dim, @default_patch_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    # Named inputs
    sequence = Axon.input("sequence", shape: {nil, nil, embed_dim})
    modality_mask = Axon.input("modality_mask", shape: {nil, nil})
    timestep = Axon.input("timestep", shape: {nil})

    # Step 1: add learnable TEXT / IMAGE type embeddings per position
    text_type_embed = Axon.param("text_type_embed", {embed_dim}, initializer: :glorot_uniform)
    img_type_embed = Axon.param("img_type_embed", {embed_dim}, initializer: :glorot_uniform)

    x =
      Axon.layer(
        &modality_embed_impl/5,
        [sequence, modality_mask, text_type_embed, img_type_embed],
        name: "modality_embed",
        op_name: :modality_embed
      )

    # Step 2: project to hidden_size if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(x, hidden_size, name: "input_proj")
      else
        x
      end

    # Step 3: sinusoidal timestep embedding for diffusion conditioning
    time_embed = build_timestep_embed(timestep, hidden_size)

    # Step 4: N Transfusion blocks (shared backbone)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_transfusion_block(acc, modality_mask, time_embed,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          dropout: dropout,
          name: "layer_#{layer_idx}"
        )
      end)

    # Step 5: final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Step 6: dual heads
    text_logits = Axon.dense(x, vocab_size, name: "text_head")
    image_pred = Axon.dense(x, patch_dim, name: "image_head")

    Axon.container(%{text_logits: text_logits, image_pred: image_pred})
  end

  # Add timestep conditioning to image positions then run a full transformer block.
  defp build_transfusion_block(input, modality_mask, time_embed, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "tf_block")

    # Inject diffusion timestep into image positions only
    x =
      Axon.layer(
        &add_time_to_image_impl/4,
        [input, modality_mask, time_embed],
        name: "#{name}_time_cond",
        op_name: :time_image_cond
      )

    # Attention sublayer
    attn_normed = Axon.layer_norm(x, name: "#{name}_attn_norm")

    q = Axon.dense(attn_normed, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(attn_normed, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(attn_normed, hidden_size, name: "#{name}_v_proj")

    attn_out =
      Axon.layer(
        &mixed_attention_impl/5,
        [q, k, v, modality_mask],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :mixed_attention
      )

    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(x, attn_out, name: "#{name}_attn_residual")

    # FFN sublayer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Attention mask utilities
  # ============================================================================

  @doc """
  Build the Transfusion mixed attention mask for a text+image sequence.

  Produces a boolean matrix of shape `[text_len + image_len, text_len + image_len]`
  where `true` means "allow attention":

  - Text queries see all preceding positions (causal).
  - Image queries see all other image patches (bidirectional) and all preceding text.

  Combined rule: `mask[i, j] = (j ≤ i) OR (image[i] AND image[j])`

  ## Parameters

    - `text_len` - Number of text token positions
    - `image_len` - Number of image patch positions (appended after text)

  ## Options

  Currently unused; reserved for future per-image-region masks.

  ## Returns

    Boolean `Nx.Tensor.t()` of shape `[text_len + image_len, text_len + image_len]`.
    `true` = allowed, `false` = masked.
  """
  @spec build_mixed_mask(pos_integer(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def build_mixed_mask(text_len, image_len, _opts \\ []) do
    total = text_len + image_len

    # Modality indicator: 0 for text, 1 for image
    # Handle degenerate cases where one of the lengths is 0
    modality =
      cond do
        text_len == 0 ->
          Nx.broadcast(Nx.tensor(1, type: :s32), {image_len})

        image_len == 0 ->
          Nx.broadcast(Nx.tensor(0, type: :s32), {text_len})

        true ->
          Nx.concatenate([
            Nx.broadcast(Nx.tensor(0, type: :s32), {text_len}),
            Nx.broadcast(Nx.tensor(1, type: :s32), {image_len})
          ])
      end

    rows = Nx.iota({total, total}, axis: 0)
    cols = Nx.iota({total, total}, axis: 1)

    # Causal component: j <= i
    causal = Nx.greater_equal(rows, cols)

    # Bidirectional image-to-image component
    is_image_i = Nx.reshape(modality, {total, 1})
    is_image_j = Nx.reshape(modality, {1, total})
    img_to_img = Nx.logical_and(is_image_i, is_image_j)

    # Union: text sees past causally; image sees all image + past text
    Nx.logical_or(causal, img_to_img)
  end

  # ============================================================================
  # Loss
  # ============================================================================

  @doc """
  Compute the combined Transfusion training loss.

  Combines cross-entropy on text positions with MSE on image positions,
  each masked and averaged over only the relevant positions.

  ## Parameters

    - `text_logits` - `[batch, seq_len, vocab_size]` raw logits from text head
    - `image_pred` - `[batch, seq_len, patch_dim]` predicted noise/velocity
    - `targets` - Map with:
        - `:text_targets` — `[batch, seq_len]` integer token IDs (next-token labels)
        - `:image_targets` — `[batch, seq_len, patch_dim]` target denoised patches
        - `:text_mask` — `[batch, seq_len]` float 1.0 at text positions, 0.0 elsewhere
        - `:image_mask` — `[batch, seq_len]` float 1.0 at image positions, 0.0 elsewhere

  ## Options

    - `:text_weight` - Weight for CE text loss (default: 1.0)
    - `:image_weight` - Weight for MSE image loss (default: 1.0)

  ## Returns

    Scalar loss tensor: `text_weight * L_CE + image_weight * L_MSE`.
  """
  @spec transfusion_loss(Nx.Tensor.t(), Nx.Tensor.t(), map(), keyword()) :: Nx.Tensor.t()
  def transfusion_loss(text_logits, image_pred, targets, opts \\ []) do
    text_weight = Keyword.get(opts, :text_weight, 1.0)
    image_weight = Keyword.get(opts, :image_weight, 1.0)

    text_loss = compute_text_ce_loss(text_logits, targets[:text_targets], targets[:text_mask])
    image_loss = compute_image_mse_loss(image_pred, targets[:image_targets], targets[:image_mask])

    Nx.add(
      Nx.multiply(text_weight, text_loss),
      Nx.multiply(image_weight, image_loss)
    )
  end

  # Masked cross-entropy loss for text next-token prediction.
  defp compute_text_ce_loss(logits, targets, mask) do
    # logits: [batch, seq_len, vocab_size]
    # targets: [batch, seq_len] integer token IDs
    # mask: [batch, seq_len] float, 1.0 at text positions
    vocab_size = Nx.axis_size(logits, 2)

    # Numerically stable log-softmax over vocab axis
    max_logits = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logits)
    log_sum = Nx.log(Nx.add(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true), 1.0e-8))
    log_probs = Nx.subtract(shifted, log_sum)
    # [batch, seq_len, vocab_size]

    # One-hot encode targets: [batch, seq_len, vocab_size]
    target_idx = Nx.as_type(targets, :s64)
    vocab_range = Nx.iota({vocab_size}, type: :s64) |> Nx.reshape({1, 1, vocab_size})
    one_hot = Nx.equal(Nx.new_axis(target_idx, -1), vocab_range) |> Nx.as_type(:f32)

    # CE per position: -sum(one_hot * log_probs, axis=-1)  =>  [batch, seq_len]
    ce_per_pos = Nx.negate(Nx.sum(Nx.multiply(one_hot, log_probs), axes: [-1]))

    # Masked mean over text positions
    Nx.divide(
      Nx.sum(Nx.multiply(mask, ce_per_pos)),
      Nx.add(Nx.sum(mask), 1.0e-8)
    )
  end

  # Masked MSE loss for image patch denoising.
  defp compute_image_mse_loss(image_pred, image_targets, mask) do
    # image_pred, image_targets: [batch, seq_len, patch_dim]
    # mask: [batch, seq_len] float, 1.0 at image positions
    diff = Nx.subtract(image_pred, image_targets)
    mse_per_pos = Nx.mean(Nx.multiply(diff, diff), axes: [-1])
    # [batch, seq_len]

    Nx.divide(
      Nx.sum(Nx.multiply(mask, mse_per_pos)),
      Nx.add(Nx.sum(mask), 1.0e-8)
    )
  end

  # ============================================================================
  # Private: layer implementations
  # ============================================================================

  # Add learnable TEXT or IMAGE type embedding to each sequence position.
  # seq: [batch, seq_len, embed_dim]
  # modality_mask: [batch, seq_len], 0=text, 1=image
  # text_e, img_e: [embed_dim] learnable vectors
  defp modality_embed_impl(seq, modality_mask, text_e, img_e, _opts) do
    batch = Nx.axis_size(seq, 0)
    seq_len = Nx.axis_size(seq, 1)
    embed_dim = Nx.axis_size(seq, 2)

    mask_f =
      modality_mask
      |> Nx.reshape({batch, seq_len, 1})
      |> Nx.as_type(Nx.type(seq))
      |> Nx.broadcast({batch, seq_len, embed_dim})

    text_expanded =
      Nx.broadcast(Nx.reshape(text_e, {1, 1, embed_dim}), {batch, seq_len, embed_dim})

    img_expanded = Nx.broadcast(Nx.reshape(img_e, {1, 1, embed_dim}), {batch, seq_len, embed_dim})

    type_emb =
      Nx.add(
        Nx.multiply(Nx.subtract(1.0, mask_f), text_expanded),
        Nx.multiply(mask_f, img_expanded)
      )

    Nx.add(seq, type_emb)
  end

  # Add time embedding to image positions only.
  # x: [batch, seq_len, hidden_size]
  # modality_mask: [batch, seq_len]
  # time_embed: [batch, hidden_size]
  defp add_time_to_image_impl(x, modality_mask, time_embed, _opts) do
    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)
    hidden_size = Nx.axis_size(x, 2)

    # Image gate: [batch, seq_len, hidden_size]
    img_gate =
      modality_mask
      |> Nx.reshape({batch, seq_len, 1})
      |> Nx.as_type(Nx.type(x))
      |> Nx.broadcast({batch, seq_len, hidden_size})

    # time_embed broadcast over sequence positions
    time_expanded =
      time_embed
      |> Nx.reshape({batch, 1, hidden_size})
      |> Nx.broadcast({batch, seq_len, hidden_size})

    Nx.add(x, Nx.multiply(img_gate, time_expanded))
  end

  # Multi-head self-attention with mixed causal/bidirectional mask.
  # q, k, v: [batch, seq_len, hidden_size]
  # modality_mask: [batch, seq_len], 0=text, 1=image
  defp mixed_attention_impl(q, k, v, modality_mask, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, heads, seq_len, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)
    # scores: [batch, num_heads, seq_len, seq_len]

    # Causal component: mask[i, j] = (j <= i)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal = Nx.greater_equal(rows, cols)
    # [seq_len, seq_len]

    # Image-to-image component: both query and key are image tokens
    is_image_i = Nx.reshape(modality_mask, {batch, seq_len, 1})
    is_image_j = Nx.reshape(modality_mask, {batch, 1, seq_len})
    img_to_img = Nx.logical_and(is_image_i, is_image_j)
    # [batch, seq_len, seq_len]

    # Expand causal to [batch, num_heads, seq_len, seq_len]
    causal_exp =
      causal
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    # Expand img_to_img to [batch, num_heads, seq_len, seq_len]
    img_exp =
      img_to_img
      |> Nx.reshape({batch, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    # Combined mask: allow if causal OR both image
    combined = Nx.logical_or(causal_exp, img_exp)

    scores =
      Nx.select(
        combined,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Stable softmax over keys
    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Private: timestep embedding (sinusoidal → MLP)
  # ============================================================================

  defp build_timestep_embed(timestep, hidden_size) do
    embed =
      Axon.layer(
        &sinusoidal_embed_impl/2,
        [timestep],
        name: "time_sinusoidal",
        hidden_size: hidden_size,
        op_name: :sinusoidal_embed
      )

    embed
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp sinusoidal_embed_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    t_f = Nx.as_type(t, :f32)
    angles = Nx.multiply(Nx.new_axis(t_f, 1), Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output hidden size of a Transfusion model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Approximate parameter count for a Transfusion model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    patch_dim = Keyword.get(opts, :patch_dim, @default_patch_dim)

    head_dim = div(hidden_size, num_heads)
    inner_size = hidden_size * 4

    # Per-layer: Q+K+V+out projections + FFN up+down + two layer norms
    attn_params = hidden_size * (num_heads * head_dim) * 4
    ffn_params = hidden_size * inner_size + inner_size * hidden_size
    per_layer = attn_params + ffn_params

    # Modality type embeddings
    type_embeds = embed_dim * 2

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    # Time embedding MLP
    time_mlp = hidden_size * hidden_size * 2

    # Output heads
    heads = hidden_size * vocab_size + hidden_size * patch_dim

    type_embeds + input_proj + time_mlp + per_layer * num_layers + heads
  end

  @doc """
  Recommended defaults for a small Transfusion model.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 8,
      num_layers: 6,
      vocab_size: 32_000,
      patch_dim: 64,
      dropout: 0.0
    ]
  end
end
