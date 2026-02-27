defmodule Edifice.Detection.SAM2 do
  @moduledoc """
  SAM 2: Segment Anything Model 2.

  A promptable segmentation model that takes an image and point prompts to
  produce segmentation masks with quality scores. This implements the
  image-mode architecture (no video memory attention). Key components:

  1. **CNN backbone** — extracts spatial features with stride-2 downsampling
  2. **Prompt encoder** — encodes point prompts via Fourier positional encoding
     plus learnable foreground/background type embeddings
  3. **Two-way transformer mask decoder** — bidirectional cross-attention
     between prompt tokens and image features, with upsampling and per-mask
     prediction heads
  4. **IoU prediction head** — MLP estimating mask quality scores

  The two-way transformer is the distinctive component: each layer performs
  (1) self-attention on tokens, (2) cross-attention from tokens to image,
  (3) token MLP, (4) cross-attention from image to tokens. This bidirectional
  flow lets both representations condition on each other.

  ## Architecture

  ```
  Image [B, H, W, C]   Points [B, N, 2]   Labels [B, N]
        |                      |                  |
  +-----v-----------+  +------v------------------v------+
  | CNN Backbone     |  | Prompt Encoder                 |
  | → [B, h, w, D]  |  | Fourier PE + type embed → [B,N,D]
  +-----+-----------+  +------+-------------------------+
        |                      |
  +-----v----------------------v----------------------+
  | Two-Way Transformer Mask Decoder                   |
  |   tokens = [iou_tok, mask_tok_1..M, prompt_toks]  |
  |   Per layer:                                       |
  |     1. Self-attention on tokens                    |
  |     2. Cross-attn: tokens → image (with 2D PE)    |
  |     3. MLP on tokens                              |
  |     4. Cross-attn: image → tokens                 |
  |   Final: token→image cross-attn + 4x upsample    |
  +-----+---------------------+-----------------------+
        |                     |
  +-----v------+        +----v--------+
  | Mask Pred   |        | IoU Head    |
  | dot product |        | MLP → [B,M] |
  +-----+------+        +----+--------+
        |                     |
  [B, M, H', W']        [B, M]
  mask logits          quality scores
  ```

  ## Usage

      model = SAM2.build(
        image_size: 1024,
        hidden_dim: 256,
        num_heads: 8,
        max_points: 16
      )

      # Inputs: %{"image" => ..., "points" => ..., "labels" => ...}
      # points: [batch, max_points, 2] — (x, y) normalized to [0, 1]
      # labels: [batch, max_points] — 1.0 foreground, 0.0 background
      # Outputs: %{masks: [B, M, H', W'], iou_scores: [B, M]}

  ## References

  - Ravi et al., "SAM 2: Segment Anything in Images and Videos" (2024)
    https://arxiv.org/abs/2408.00714
  """

  alias Edifice.Blocks.{CrossAttention, SDPA, SinusoidalPE2D, Upsample2x}

  @default_image_size 1024
  @default_in_channels 3
  @default_hidden_dim 256
  @default_num_heads 8
  @default_num_decoder_layers 2
  @default_ffn_dim 2048
  @default_num_multimask_outputs 3
  @default_max_points 16
  @default_dropout 0.0
  @default_backbone_stages 4
  @default_pe_dim 128

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_decoder_layers, pos_integer()}
          | {:ffn_dim, pos_integer()}
          | {:num_multimask_outputs, pos_integer()}
          | {:max_points, pos_integer()}
          | {:dropout, float()}
          | {:backbone_stages, pos_integer()}
          | {:pe_dim, pos_integer()}

  @doc """
  Build a SAM 2 model (image mode).

  ## Options

    - `:image_size` - Input image size, square (default: 1024)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_dim` - Hidden dimension / embedding dim (default: 256)
    - `:num_heads` - Attention heads in mask decoder (default: 8)
    - `:num_decoder_layers` - Two-way transformer decoder depth (default: 2)
    - `:ffn_dim` - Mask decoder FFN hidden dim (default: 2048)
    - `:num_multimask_outputs` - Number of multi-mask outputs (default: 3)
    - `:max_points` - Maximum number of point prompts (default: 16)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:backbone_stages` - Stride-2 conv stages in backbone (default: 4, stride 16)
    - `:pe_dim` - Fourier PE dimension for prompts (default: 128)

  ## Returns

    An `Axon.container` outputting `%{masks: ..., iou_scores: ...}`.
    - `masks`: `[batch, num_multimask_outputs + 1, mask_h, mask_w]` (logits)
    - `iou_scores`: `[batch, num_multimask_outputs + 1]` (quality scores)
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_decoder_layers = Keyword.get(opts, :num_decoder_layers, @default_num_decoder_layers)
    ffn_dim = Keyword.get(opts, :ffn_dim, @default_ffn_dim)
    num_multimask = Keyword.get(opts, :num_multimask_outputs, @default_num_multimask_outputs)
    max_points = Keyword.get(opts, :max_points, @default_max_points)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    backbone_stages = Keyword.get(opts, :backbone_stages, @default_backbone_stages)
    pe_dim = Keyword.get(opts, :pe_dim, @default_pe_dim)

    # +1 for single-mask mode token
    num_mask_tokens = num_multimask + 1

    # Inputs
    image = Axon.input("image", shape: {nil, image_size, image_size, in_channels})
    points = Axon.input("points", shape: {nil, max_points, 2})
    labels = Axon.input("labels", shape: {nil, max_points})

    # Image encoder: CNN backbone + 1x1 neck projection
    features = cnn_backbone(image, hidden_dim, backbone_stages, "backbone")
    image_embeddings = Axon.conv(features, hidden_dim, kernel_size: {1, 1}, name: "neck_proj")

    # Prompt encoder: Fourier PE + type embeddings
    prompt_embeddings =
      prompt_encoder(points, labels, hidden_dim, pe_dim, "prompt_enc")

    # Two-way transformer mask decoder
    {output_tokens, upscaled_image} =
      mask_decoder(
        image_embeddings,
        prompt_embeddings,
        hidden_dim,
        num_heads,
        ffn_dim,
        num_decoder_layers,
        num_mask_tokens,
        dropout,
        "mask_dec"
      )

    # Mask prediction: dot product of mask tokens with upscaled features
    masks =
      mask_prediction(output_tokens, upscaled_image, num_mask_tokens, hidden_dim, "mask_pred")

    # IoU prediction: MLP on the IoU token
    iou_scores = iou_prediction(output_tokens, num_mask_tokens, hidden_dim, "iou_head")

    Axon.container(%{masks: masks, iou_scores: iou_scores})
  end

  @doc """
  Get the output size of a SAM 2 model.

  Returns `num_multimask_outputs + 1` (number of mask predictions).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_multimask = Keyword.get(opts, :num_multimask_outputs, @default_num_multimask_outputs)
    num_multimask + 1
  end

  # ============================================================================
  # CNN Backbone
  # ============================================================================

  # Simplified CNN backbone producing features at stride 2^num_stages.
  # Uses strided convolutions with batch norm and GELU (matching Hiera).
  @spec cnn_backbone(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp cnn_backbone(input, out_channels, num_stages, name) do
    stage_channels =
      Enum.map(1..num_stages, fn i ->
        if i == num_stages, do: out_channels, else: min(64 * i, out_channels)
      end)

    Enum.reduce(Enum.with_index(stage_channels, 1), input, fn {channels, i}, acc ->
      kernel = if i == 1, do: {7, 7}, else: {3, 3}

      acc
      |> Axon.conv(channels,
        kernel_size: kernel,
        strides: 2,
        padding: :same,
        name: "#{name}_conv#{i}"
      )
      |> Axon.batch_norm(name: "#{name}_bn#{i}")
      |> Axon.activation(:gelu, name: "#{name}_act#{i}")
    end)
  end

  # ============================================================================
  # Prompt Encoder
  # ============================================================================

  # Encodes point prompts using Fourier positional encoding + type embeddings.
  # Points are (x, y) normalized to [0, 1]. Labels are 1.0 (fg) or 0.0 (bg).
  @spec prompt_encoder(Axon.t(), Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp prompt_encoder(points, labels, hidden_dim, pe_dim, name) do
    # Learnable Gaussian matrix for random Fourier features
    pe_matrix =
      Axon.param("#{name}_pe_matrix", {2, pe_dim}, initializer: :glorot_uniform)

    # Fourier PE: points [B, N, 2] × pe_matrix [2, pe_dim] → sin/cos → [B, N, 2*pe_dim]
    point_pe =
      Axon.layer(
        fn pts, pe_mat, _opts ->
          projected = Nx.dot(pts, [2], pe_mat, [0])
          scaled = Nx.multiply(projected, 2.0 * :math.pi())
          Nx.concatenate([Nx.sin(scaled), Nx.cos(scaled)], axis: 2)
        end,
        [points, pe_matrix],
        name: "#{name}_fourier_pe",
        op_name: :fourier_pe
      )

    # Project to hidden_dim
    point_embed = Axon.dense(point_pe, hidden_dim, name: "#{name}_pe_proj")

    # Learned foreground/background type embeddings
    fg_embed =
      Axon.param("#{name}_fg_embed", {1, 1, hidden_dim}, initializer: :glorot_uniform)

    bg_embed =
      Axon.param("#{name}_bg_embed", {1, 1, hidden_dim}, initializer: :glorot_uniform)

    # Add type embedding: fg where label=1, bg where label=0
    Axon.layer(
      fn pt_emb, lbl, fg, bg, _opts ->
        mask = Nx.new_axis(lbl, -1)
        type_emb = Nx.add(Nx.multiply(mask, fg), Nx.multiply(Nx.subtract(1.0, mask), bg))
        Nx.add(pt_emb, type_emb)
      end,
      [point_embed, labels, fg_embed, bg_embed],
      name: "#{name}_add_type",
      op_name: :add_type_embed
    )
  end

  # ============================================================================
  # Two-Way Transformer Mask Decoder
  # ============================================================================

  # Bidirectional cross-attention between prompt tokens and image features.
  # Returns processed output tokens and 4x-upscaled image features.
  @spec mask_decoder(
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          float(),
          String.t()
        ) :: {Axon.t(), Axon.t()}
  defp mask_decoder(
         image_embeddings,
         prompt_embeddings,
         hidden_dim,
         num_heads,
         ffn_dim,
         num_layers,
         num_mask_tokens,
         dropout,
         name
       ) do
    # Total output tokens: 1 IoU + num_mask_tokens mask tokens
    total_output_tokens = num_mask_tokens + 1

    output_token_param =
      Axon.param("#{name}_output_tokens", {total_output_tokens, hidden_dim},
        initializer: :glorot_uniform
      )

    # Flatten image: [B, h, w, D] → [B, h*w, D]
    image_flat =
      Axon.nx(
        image_embeddings,
        fn t ->
          {b, h, w, d} = Nx.shape(t)
          Nx.reshape(t, {b, h * w, d})
        end,
        name: "#{name}_flatten_img"
      )

    # 2D sinusoidal PE for image positions
    image_pe =
      Axon.nx(
        image_flat,
        fn t ->
          {_b, seq_len, dim} = Nx.shape(t)
          SinusoidalPE2D.build_table(seq_len, dim)
        end,
        name: "#{name}_image_pe"
      )

    # Concatenate output tokens with prompt tokens: [B, total_out + N_prompt, D]
    tokens =
      Axon.layer(
        fn prompt_emb, out_tok, _opts ->
          batch_size = Nx.axis_size(prompt_emb, 0)
          {n_out, d} = Nx.shape(out_tok)
          out_batch = Nx.broadcast(Nx.new_axis(out_tok, 0), {batch_size, n_out, d})
          Nx.concatenate([out_batch, prompt_emb], axis: 1)
        end,
        [prompt_embeddings, output_token_param],
        name: "#{name}_concat_tokens",
        op_name: :concat_tokens
      )

    # Two-way transformer blocks
    {tokens, image_flat} =
      Enum.reduce(1..num_layers, {tokens, image_flat}, fn i, {tok, img} ->
        two_way_block(
          tok,
          img,
          image_pe,
          hidden_dim,
          num_heads,
          ffn_dim,
          dropout,
          "#{name}_tw#{i}"
        )
      end)

    # Final token-to-image cross-attention
    tok_norm = Axon.layer_norm(tokens, name: "#{name}_final_tok_norm")
    img_with_pe = Axon.add(image_flat, image_pe, name: "#{name}_final_img_pe")

    final_ca =
      CrossAttention.layer(tok_norm, img_with_pe, image_flat,
        hidden_size: hidden_dim,
        num_heads: num_heads,
        name: "#{name}_final_ca"
      )

    tokens = Axon.add(tokens, final_ca, name: "#{name}_final_ca_res")
    tokens = Axon.layer_norm(tokens, name: "#{name}_final_norm")

    # Extract output tokens (first total_output_tokens)
    output_tokens =
      Axon.layer(
        fn tok, _opts ->
          Nx.slice_along_axis(tok, 0, total_output_tokens, axis: 1)
        end,
        [tokens],
        name: "#{name}_extract_out",
        op_name: :slice_tokens
      )

    # Upscale image features 4x
    upscaled = upscale_image(image_flat, image_embeddings, hidden_dim, "#{name}_up")

    {output_tokens, upscaled}
  end

  # Two-way transformer block with bidirectional cross-attention.
  @spec two_way_block(
          Axon.t(),
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          float(),
          String.t()
        ) :: {Axon.t(), Axon.t()}
  defp two_way_block(tokens, image, image_pe, hidden_dim, num_heads, ffn_dim, dropout, name) do
    # 1. Self-attention on tokens
    tok_norm = Axon.layer_norm(tokens, name: "#{name}_sa_norm")
    sa = self_attention(tok_norm, hidden_dim, num_heads, "#{name}_sa")
    sa = maybe_dropout(sa, dropout, "#{name}_sa_drop")
    tokens = Axon.add(tokens, sa, name: "#{name}_sa_res")

    # 2. Cross-attention: tokens → image (PE added to image keys)
    tok_norm = Axon.layer_norm(tokens, name: "#{name}_t2i_norm")
    img_with_pe = Axon.add(image, image_pe, name: "#{name}_t2i_pe")

    t2i =
      CrossAttention.layer(tok_norm, img_with_pe, image,
        hidden_size: hidden_dim,
        num_heads: num_heads,
        name: "#{name}_t2i"
      )

    t2i = maybe_dropout(t2i, dropout, "#{name}_t2i_drop")
    tokens = Axon.add(tokens, t2i, name: "#{name}_t2i_res")

    # 3. MLP on tokens
    tok_norm = Axon.layer_norm(tokens, name: "#{name}_ffn_norm")

    ffn =
      tok_norm
      |> Axon.dense(ffn_dim, name: "#{name}_ffn_up")
      |> Axon.activation(:relu, name: "#{name}_ffn_act")
      |> maybe_dropout(dropout, "#{name}_ffn_drop1")
      |> Axon.dense(hidden_dim, name: "#{name}_ffn_down")
      |> maybe_dropout(dropout, "#{name}_ffn_drop2")

    tokens = Axon.add(tokens, ffn, name: "#{name}_ffn_res")

    # 4. Cross-attention: image → tokens
    img_norm = Axon.layer_norm(image, name: "#{name}_i2t_inorm")
    tok_keys = Axon.layer_norm(tokens, name: "#{name}_i2t_tnorm")

    i2t =
      CrossAttention.layer(img_norm, tok_keys, tokens,
        hidden_size: hidden_dim,
        num_heads: num_heads,
        name: "#{name}_i2t"
      )

    i2t = maybe_dropout(i2t, dropout, "#{name}_i2t_drop")
    image = Axon.add(image, i2t, name: "#{name}_i2t_res")

    {tokens, image}
  end

  # ============================================================================
  # Mask and IoU Prediction Heads
  # ============================================================================

  # Dot-product mask prediction: each mask token is projected through an MLP,
  # then dot-producted with upscaled image features to produce spatial masks.
  @spec mask_prediction(Axon.t(), Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp mask_prediction(output_tokens, upscaled_image, num_mask_tokens, hidden_dim, name) do
    up_channels = max(div(hidden_dim, 4), 1)

    # Extract mask tokens (skip IoU token at index 0)
    mask_tokens =
      Axon.layer(
        fn tok, _opts ->
          Nx.slice_along_axis(tok, 1, num_mask_tokens, axis: 1)
        end,
        [output_tokens],
        name: "#{name}_extract_masks",
        op_name: :slice_tokens
      )

    # Hypernetwork: project mask tokens to match upscaled channel dim
    mask_hyper =
      mask_tokens
      |> Axon.dense(hidden_dim, name: "#{name}_hyper1")
      |> Axon.activation(:relu, name: "#{name}_hyper_act1")
      |> Axon.dense(hidden_dim, name: "#{name}_hyper2")
      |> Axon.activation(:relu, name: "#{name}_hyper_act2")
      |> Axon.dense(up_channels, name: "#{name}_hyper3")

    # Dot product: [B, M, up_ch] · [B, H'*W', up_ch]^T → [B, M, H', W']
    Axon.layer(
      fn hyper, img, _opts ->
        {batch, n_masks, d} = Nx.shape(hyper)
        {_, h, w, _} = Nx.shape(img)
        img_flat = Nx.reshape(img, {batch, h * w, d})
        masks = Nx.dot(hyper, [2], [0], img_flat, [2], [0])
        Nx.reshape(masks, {batch, n_masks, h, w})
      end,
      [mask_hyper, upscaled_image],
      name: "#{name}_dot",
      op_name: :mask_dot_product
    )
  end

  # IoU prediction: 3-layer MLP on the IoU token (first output token).
  @spec iou_prediction(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp iou_prediction(output_tokens, num_mask_tokens, hidden_dim, name) do
    iou_token =
      Axon.layer(
        fn tok, _opts ->
          tok
          |> Nx.slice_along_axis(0, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        [output_tokens],
        name: "#{name}_extract_iou",
        op_name: :slice_tokens
      )

    iou_token
    |> Axon.dense(hidden_dim, name: "#{name}_mlp1")
    |> Axon.activation(:relu, name: "#{name}_act1")
    |> Axon.dense(hidden_dim, name: "#{name}_mlp2")
    |> Axon.activation(:relu, name: "#{name}_act2")
    |> Axon.dense(num_mask_tokens, name: "#{name}_out")
  end

  # ============================================================================
  # Image Upscaling
  # ============================================================================

  # Upscale image features 4x: reshape to spatial → 2x upsample + conv → repeat.
  @spec upscale_image(Axon.t(), Axon.t(), pos_integer(), String.t()) :: Axon.t()
  defp upscale_image(image_flat, image_spatial_ref, hidden_dim, name) do
    half_dim = max(div(hidden_dim, 2), 1)
    quarter_dim = max(div(hidden_dim, 4), 1)

    # Reshape flat back to spatial using reference for dimensions
    spatial =
      Axon.layer(
        fn flat, ref, _opts ->
          {b, _hw, d} = Nx.shape(flat)
          {_, h, w, _} = Nx.shape(ref)
          Nx.reshape(flat, {b, h, w, d})
        end,
        [image_flat, image_spatial_ref],
        name: "#{name}_to_spatial",
        op_name: :reshape_spatial
      )

    # 4x upscale: two stages of 2x upsample + conv
    spatial
    |> Upsample2x.layer("#{name}_up1")
    |> Axon.conv(half_dim, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv1")
    |> Axon.layer_norm(name: "#{name}_ln1")
    |> Axon.activation(:gelu, name: "#{name}_act1")
    |> Upsample2x.layer("#{name}_up2")
    |> Axon.conv(quarter_dim, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv2")
    |> Axon.activation(:gelu, name: "#{name}_act2")
  end

  # ============================================================================
  # Attention Helpers
  # ============================================================================

  @spec self_attention(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp self_attention(x, hidden_dim, num_heads, name) do
    head_dim = div(hidden_dim, num_heads)
    q = Axon.dense(x, hidden_dim, name: "#{name}_q")
    k = Axon.dense(x, hidden_dim, name: "#{name}_k")
    v = Axon.dense(x, hidden_dim, name: "#{name}_v")

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          SDPA.compute(q_t, k_t, v_t, num_heads, head_dim)
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :mha_compute
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  @spec maybe_dropout(Axon.t(), float(), String.t()) :: Axon.t()
  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input
end
