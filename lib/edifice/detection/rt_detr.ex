defmodule Edifice.Detection.RTDETR do
  @moduledoc """
  RT-DETR: Real-Time Detection Transformer.

  A real-time end-to-end object detector that outperforms YOLO models while
  maintaining the NMS-free design of DETR. Key innovations over standard DETR:

  1. **Hybrid encoder**: AIFI (self-attention on S5 only) + CCFM (CNN-based
     bidirectional feature pyramid fusion). This replaces DETR's expensive
     6-layer encoder over all spatial tokens with a cheap 1-layer encoder on
     the smallest feature map plus CNN cross-scale fusion.

  2. **Content-aware query selection**: Instead of learned query embeddings,
     a lightweight encoder head predicts class scores and bboxes at every
     spatial position. The top-K scoring positions become decoder queries
     with their predicted bboxes as initial reference points.

  3. **Iterative box refinement**: Each decoder layer refines the bounding
     box prediction from the previous layer via residual updates.

  ## Architecture

  ```
  Image [batch, H, W, C]
        |
  +-----v--------------------+
  | Multi-scale CNN Backbone  |  3 feature levels
  +-----+--------------------+
        |  S3 [stride 8], S4 [stride 16], S5 [stride 32]
        v
  +-----v--------------------+
  | Input Projections (1x1)   |  All → hidden_dim channels
  +-----+--------------------+
        |
  +-----v--------------------+
  | AIFI: Transformer on S5   |  1-layer self-attention on smallest map
  +-----+--------------------+
        |  F5 (refined S5)
        v
  +-----v--------------------+
  | CCFM: Bidirectional FPN   |  Top-down + bottom-up CNN fusion
  +-----+--------------------+
        |  {P3, N4, N5} multi-scale features
        v
  +-----v--------------------+
  | Flatten + Encoder Heads   |  Class scores + bbox at every position
  | Top-K Query Selection     |  Select top-K as decoder queries
  +-----+--------------------+
        |  [batch, K, hidden_dim] queries + reference points
        v
  +-----v--------------------+
  | Transformer Decoder x N   |  Self-attn + cross-attn + FFN
  | + Iterative Box Refine    |  Each layer refines bbox predictions
  +-----+--------------------+
        |
  +-----v-----------+  +--------v---------+
  | Class Head       |  | BBox Head (MLP)   |
  | Linear → C       |  | 3-layer → sigmoid |
  +------------------+  +-------------------+
        |                       |
        v                       v
  [batch, K, num_classes]  [batch, K, 4]
  ```

  ## Usage

      model = RTDETR.build(
        image_size: 640,
        num_queries: 300,
        num_classes: 80,
        hidden_dim: 256
      )

      # model outputs %{class_logits: ..., bbox_pred: ...}
      # class_logits: [batch, num_queries, num_classes] (no background class)
      # bbox_pred: [batch, num_queries, 4] (cx, cy, w, h in [0,1])

  ## References

  - Lv et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024)
    https://arxiv.org/abs/2304.08069
  """

  alias Edifice.Blocks.{BBoxHead, SDPA, SinusoidalPE2D, TransformerBlock, Upsample2x}

  @default_image_size 640
  @default_in_channels 3
  @default_hidden_dim 256
  @default_num_heads 8
  @default_num_decoder_layers 6
  @default_ffn_dim 1024
  @default_num_queries 300
  @default_num_classes 80
  @default_dropout 0.0
  @default_backbone_stages 3

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_decoder_layers, pos_integer()}
          | {:ffn_dim, pos_integer()}
          | {:num_queries, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:dropout, float()}
          | {:backbone_stages, pos_integer()}

  @doc """
  Build an RT-DETR model.

  ## Options

    - `:image_size` - Input image size, square (default: 640)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_dim` - Hidden dimension throughout (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_decoder_layers` - Number of transformer decoder layers (default: 6)
    - `:ffn_dim` - Feed-forward network hidden dimension (default: 1024)
    - `:num_queries` - Number of top-K selected queries (default: 300)
    - `:num_classes` - Number of object classes, no background (default: 80)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:backbone_stages` - Conv stages per backbone level (default: 3)

  ## Returns

    An `Axon.container` outputting `%{class_logits: ..., bbox_pred: ...}`.
    - `class_logits`: `[batch, num_queries, num_classes]` (no background class)
    - `bbox_pred`: `[batch, num_queries, 4]` (cx, cy, w, h in [0, 1])
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_decoder_layers = Keyword.get(opts, :num_decoder_layers, @default_num_decoder_layers)
    ffn_dim = Keyword.get(opts, :ffn_dim, @default_ffn_dim)
    num_queries = Keyword.get(opts, :num_queries, @default_num_queries)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    backbone_stages = Keyword.get(opts, :backbone_stages, @default_backbone_stages)

    input = Axon.input("image", shape: {nil, image_size, image_size, in_channels})

    # Multi-scale backbone: 3 feature levels at strides 8, 16, 32
    {s3, s4, s5} = multi_scale_backbone(input, hidden_dim, backbone_stages, "backbone")

    # Input projections: each level → hidden_dim
    s3_proj = Axon.conv(s3, hidden_dim, kernel_size: {1, 1}, name: "proj_s3")
    s4_proj = Axon.conv(s4, hidden_dim, kernel_size: {1, 1}, name: "proj_s4")
    s5_proj = Axon.conv(s5, hidden_dim, kernel_size: {1, 1}, name: "proj_s5")

    # AIFI: 1-layer transformer encoder on S5 only
    f5 = aifi_encoder(s5_proj, hidden_dim, num_heads, ffn_dim, dropout, "aifi")

    # CCFM: bidirectional feature pyramid fusion
    {p3, n4, n5} = ccfm_fusion(s3_proj, s4_proj, f5, hidden_dim, "ccfm")

    # Flatten multi-scale features and concatenate
    memory = flatten_and_concat([p3, n4, n5], "encoder_memory")

    # Encoder prediction heads for query selection
    enc_class = Axon.dense(memory, num_classes, name: "enc_score_head")
    enc_bbox = BBoxHead.layer(memory, hidden_dim, "enc_bbox")

    # Top-K query selection: select top-K by max class score
    {selected_features, selected_bboxes} =
      topk_query_selection(memory, enc_class, enc_bbox, num_queries, "query_select")

    # Transformer decoder with iterative box refinement
    {decoded, refined_bbox} =
      decoder_with_refinement(
        selected_features,
        selected_bboxes,
        memory,
        hidden_dim,
        num_heads,
        ffn_dim,
        num_decoder_layers,
        dropout,
        "decoder"
      )

    # Final prediction heads
    class_logits = Axon.dense(decoded, num_classes, name: "class_head")

    Axon.container(%{class_logits: class_logits, bbox_pred: refined_bbox})
  end

  @doc """
  Get the output size of an RT-DETR model.

  Returns the total output dimension per query: `num_classes + 4`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    num_classes + 4
  end

  # ============================================================================
  # Multi-Scale CNN Backbone
  # ============================================================================

  # Produces 3 feature maps at strides 8, 16, 32 (channels-last).
  # Each level uses `backbone_stages` conv layers at stride 1, then a
  # stride-2 transition to the next level. Initial downsampling uses
  # 2 stride-2 convs to reach stride 4, then each level adds one more 2x.
  @spec multi_scale_backbone(Axon.t(), pos_integer(), pos_integer(), String.t()) ::
          {Axon.t(), Axon.t(), Axon.t()}
  defp multi_scale_backbone(input, hidden_dim, num_stages, name) do
    ch = max(div(hidden_dim, 4), 16)

    # Stem: stride 4 via two stride-2 convs
    stem =
      input
      |> Axon.conv(ch, kernel_size: {3, 3}, strides: 2, padding: :same, name: "#{name}_stem1")
      |> Axon.batch_norm(name: "#{name}_stem1_bn")
      |> Axon.activation(:relu, name: "#{name}_stem1_act")
      |> Axon.conv(ch, kernel_size: {3, 3}, strides: 2, padding: :same, name: "#{name}_stem2")
      |> Axon.batch_norm(name: "#{name}_stem2_bn")
      |> Axon.activation(:relu, name: "#{name}_stem2_act")

    # Level S3: stride 8 (one more stride-2 + stages at stride 1)
    s3 = build_level(stem, ch, num_stages, "#{name}_s3")

    # Level S4: stride 16
    s4 = build_level(s3, ch * 2, num_stages, "#{name}_s4")

    # Level S5: stride 32
    s5 = build_level(s4, ch * 4, num_stages, "#{name}_s5")

    {s3, s4, s5}
  end

  # One backbone level: stride-2 downsample + N conv blocks at stride 1.
  @spec build_level(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp build_level(input, channels, num_stages, name) do
    # Downsample by 2x
    x =
      input
      |> Axon.conv(channels,
        kernel_size: {3, 3},
        strides: 2,
        padding: :same,
        name: "#{name}_down"
      )
      |> Axon.batch_norm(name: "#{name}_down_bn")
      |> Axon.activation(:relu, name: "#{name}_down_act")

    # Additional conv stages at stride 1
    Enum.reduce(1..max(num_stages - 1, 1), x, fn i, acc ->
      acc
      |> Axon.conv(channels, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv#{i}")
      |> Axon.batch_norm(name: "#{name}_bn#{i}")
      |> Axon.activation(:relu, name: "#{name}_act#{i}")
    end)
  end

  # ============================================================================
  # AIFI: Attention-based Intra-scale Feature Interaction
  # ============================================================================

  # 1-layer transformer encoder applied to S5 only. Flattens spatial dims,
  # adds 2D sinusoidal PE, runs self-attention, reshapes back.
  @spec aifi_encoder(Axon.t(), pos_integer(), pos_integer(), pos_integer(), float(), String.t()) ::
          Axon.t()
  defp aifi_encoder(s5, hidden_dim, num_heads, ffn_dim, dropout, name) do
    # Flatten: [batch, H, W, D] → [batch, H*W, D]
    flat =
      Axon.nx(
        s5,
        fn t ->
          {b, h, w, d} = Nx.shape(t)
          Nx.reshape(t, {b, h * w, d})
        end,
        name: "#{name}_flatten"
      )

    # 2D positional encoding
    pe =
      Axon.nx(
        flat,
        fn t ->
          {_b, seq_len, dim} = Nx.shape(t)
          SinusoidalPE2D.build_table(seq_len, dim)
        end,
        name: "#{name}_pe"
      )

    # Single encoder layer
    encoded = encoder_layer(flat, pe, hidden_dim, num_heads, ffn_dim, dropout, name)

    # Reshape back to spatial: [batch, H*W, D] → [batch, H, W, D]
    Axon.layer(
      fn encoded_t, s5_t, _opts ->
        {b, _hw, d} = Nx.shape(encoded_t)
        {_, h, w, _} = Nx.shape(s5_t)
        Nx.reshape(encoded_t, {b, h, w, d})
      end,
      [encoded, s5],
      name: "#{name}_reshape",
      op_name: :reshape_spatial
    )
  end

  # ============================================================================
  # CCFM: CNN-based Cross-scale Feature Fusion Module
  # ============================================================================

  # PANet-style bidirectional feature pyramid: top-down then bottom-up.
  # Uses conv blocks for fusion (simplified CSPRepLayers).
  @spec ccfm_fusion(Axon.t(), Axon.t(), Axon.t(), pos_integer(), String.t()) ::
          {Axon.t(), Axon.t(), Axon.t()}
  defp ccfm_fusion(s3, s4, f5, hidden_dim, name) do
    # Top-down pathway
    # Upsample F5 and fuse with S4
    f5_up = Upsample2x.layer(f5, "#{name}_f5_up")
    p4_cat = concat_features(f5_up, s4, "#{name}_p4_cat")
    p4 = fusion_block(p4_cat, hidden_dim, "#{name}_p4_fuse")

    # Upsample P4 and fuse with S3
    p4_up = Upsample2x.layer(p4, "#{name}_p4_up")
    p3_cat = concat_features(p4_up, s3, "#{name}_p3_cat")
    p3 = fusion_block(p3_cat, hidden_dim, "#{name}_p3_fuse")

    # Bottom-up pathway
    # Downsample P3 and fuse with P4
    p3_down = downsample_2x(p3, hidden_dim, "#{name}_p3_down")
    n4_cat = concat_features(p3_down, p4, "#{name}_n4_cat")
    n4 = fusion_block(n4_cat, hidden_dim, "#{name}_n4_fuse")

    # Downsample N4 and fuse with F5
    n4_down = downsample_2x(n4, hidden_dim, "#{name}_n4_down")
    n5_cat = concat_features(n4_down, f5, "#{name}_n5_cat")
    n5 = fusion_block(n5_cat, hidden_dim, "#{name}_n5_fuse")

    {p3, n4, n5}
  end

  # Stride-2 downsample via conv (channels-last)
  @spec downsample_2x(Axon.t(), pos_integer(), String.t()) :: Axon.t()
  defp downsample_2x(input, channels, name) do
    input
    |> Axon.conv(channels, kernel_size: {3, 3}, strides: 2, padding: :same, name: "#{name}_conv")
    |> Axon.batch_norm(name: "#{name}_bn")
    |> Axon.activation(:relu, name: "#{name}_act")
  end

  # Concatenate two feature maps along channels (channels-last)
  @spec concat_features(Axon.t(), Axon.t(), String.t()) :: Axon.t()
  defp concat_features(a, b, name) do
    Axon.concatenate([a, b], axis: -1, name: name)
  end

  # Fusion block: 1x1 channel reduce + 3x3 conv + BN + ReLU (simplified CSPRepLayer)
  @spec fusion_block(Axon.t(), pos_integer(), String.t()) :: Axon.t()
  defp fusion_block(input, out_channels, name) do
    input
    |> Axon.conv(out_channels, kernel_size: {1, 1}, name: "#{name}_reduce")
    |> Axon.batch_norm(name: "#{name}_reduce_bn")
    |> Axon.activation(:relu, name: "#{name}_reduce_act")
    |> Axon.conv(out_channels, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv")
    |> Axon.batch_norm(name: "#{name}_conv_bn")
    |> Axon.activation(:relu, name: "#{name}_conv_act")
  end

  # ============================================================================
  # Flatten and Concatenate Multi-Scale Features
  # ============================================================================

  # Flatten each spatial feature map and concatenate along sequence dim.
  @spec flatten_and_concat([Axon.t()], String.t()) :: Axon.t()
  defp flatten_and_concat(feature_maps, name) do
    flattened =
      feature_maps
      |> Enum.with_index()
      |> Enum.map(fn {fm, i} ->
        Axon.nx(
          fm,
          fn t ->
            {b, h, w, d} = Nx.shape(t)
            Nx.reshape(t, {b, h * w, d})
          end,
          name: "#{name}_flat_#{i}"
        )
      end)

    Axon.concatenate(flattened, axis: 1, name: "#{name}_concat")
  end

  # ============================================================================
  # Top-K Query Selection
  # ============================================================================

  # Select top-K spatial positions by max classification score.
  # Returns selected features (decoder content queries) and selected bboxes
  # (decoder initial reference points).
  @spec topk_query_selection(Axon.t(), Axon.t(), Axon.t(), pos_integer(), String.t()) ::
          {Axon.t(), Axon.t()}
  defp topk_query_selection(memory, enc_class, enc_bbox, num_queries, name) do
    selected_features =
      Axon.layer(
        fn mem, cls, _opts ->
          topk_gather(mem, cls, num_queries)
        end,
        [memory, enc_class],
        name: "#{name}_features",
        op_name: :topk_select
      )

    selected_bboxes =
      Axon.layer(
        fn bbox, cls, _opts ->
          topk_gather(bbox, cls, num_queries)
        end,
        [enc_bbox, enc_class],
        name: "#{name}_bboxes",
        op_name: :topk_select
      )

    {selected_features, selected_bboxes}
  end

  # Gather rows from `values` at the top-K indices by max score from `scores`.
  @spec topk_gather(Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  defp topk_gather(values, scores, k) do
    {batch, _seq, dim} = Nx.shape(values)

    # Max class score per position: [batch, seq]
    max_scores = Nx.reduce_max(scores, axes: [-1])

    # Get top-K indices: [batch, K]
    {_top_vals, top_indices} = Nx.top_k(max_scores, k: k)

    # Gather selected rows: [batch, K, dim]
    # Expand indices for gather: [batch, K, 1] → broadcast to [batch, K, dim]
    idx = Nx.new_axis(top_indices, -1)
    idx_expanded = Nx.broadcast(idx, {batch, k, dim})
    Nx.take_along_axis(values, idx_expanded, axis: 1)
  end

  # ============================================================================
  # Transformer Decoder with Iterative Box Refinement
  # ============================================================================

  @spec decoder_with_refinement(
          Axon.t(),
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          float(),
          String.t()
        ) :: {Axon.t(), Axon.t()}
  defp decoder_with_refinement(
         queries,
         ref_bboxes,
         memory,
         hidden_dim,
         num_heads,
         ffn_dim,
         num_layers,
         dropout,
         name
       ) do
    Enum.reduce(1..num_layers, {queries, ref_bboxes}, fn i, {hidden, ref_bbox} ->
      new_hidden =
        TransformerBlock.layer(hidden, memory,
          attention_fn: fn x_norm, attn_name ->
            multi_head_attention(x_norm, x_norm, x_norm, hidden_dim, num_heads, attn_name)
          end,
          cross_attention_fn: fn q_norm, mem, ca_name ->
            Edifice.Blocks.CrossAttention.layer(q_norm, mem,
              hidden_size: hidden_dim,
              num_heads: num_heads,
              name: ca_name
            )
          end,
          hidden_size: hidden_dim,
          custom_ffn: fn x_norm, ffn_name ->
            x_norm
            |> Axon.dense(ffn_dim, name: "#{ffn_name}_up")
            |> Axon.activation(:relu, name: "#{ffn_name}_act")
            |> maybe_dropout(dropout, "#{ffn_name}_drop1")
            |> Axon.dense(hidden_dim, name: "#{ffn_name}_down")
          end,
          dropout: dropout,
          name: "#{name}_#{i}"
        )

      # Iterative box refinement: predict delta, add to previous, apply sigmoid
      bbox_delta = BBoxHead.layer(new_hidden, hidden_dim, "#{name}_#{i}_bbox")

      # inv_sigmoid(prev) + delta → sigmoid → new reference
      refined =
        Axon.layer(
          fn delta, prev, _opts ->
            # inv_sigmoid: log(x / (1 - x)), clamped for stability
            prev_clamped = Nx.clip(prev, 1.0e-6, 1.0 - 1.0e-6)
            inv_sig = Nx.log(Nx.divide(prev_clamped, Nx.subtract(1.0, prev_clamped)))
            Nx.sigmoid(Nx.add(inv_sig, delta))
          end,
          [bbox_delta, ref_bbox],
          name: "#{name}_#{i}_refine",
          op_name: :box_refine
        )

      {new_hidden, refined}
    end)
  end

  # ============================================================================
  # Multi-Head Attention
  # ============================================================================

  @spec multi_head_attention(
          Axon.t(),
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          String.t()
        ) ::
          Axon.t()
  defp multi_head_attention(query, key, value, hidden_dim, num_heads, name) do
    head_dim = div(hidden_dim, num_heads)
    q = Axon.dense(query, hidden_dim, name: "#{name}_q")
    k = Axon.dense(key, hidden_dim, name: "#{name}_k")
    v = Axon.dense(value, hidden_dim, name: "#{name}_v")

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

  # Encoder layer: pre-norm self-attention + FFN with PE on Q/K.
  @spec encoder_layer(
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          float(),
          String.t()
        ) ::
          Axon.t()
  defp encoder_layer(x, pe, hidden_dim, num_heads, ffn_dim, dropout, name) do
    head_dim = div(hidden_dim, num_heads)
    x_norm = Axon.layer_norm(x, name: "#{name}_sa_norm")

    q = Axon.dense(x_norm, hidden_dim, name: "#{name}_sa_q")
    k = Axon.dense(x_norm, hidden_dim, name: "#{name}_sa_k")
    v = Axon.dense(x_norm, hidden_dim, name: "#{name}_sa_v")

    q = Axon.add(q, pe, name: "#{name}_sa_q_pe")
    k = Axon.add(k, pe, name: "#{name}_sa_k_pe")

    sa =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          SDPA.compute(q_t, k_t, v_t, num_heads, head_dim)
        end,
        [q, k, v],
        name: "#{name}_sa_compute",
        op_name: :mha_compute
      )

    sa = Axon.dense(sa, hidden_dim, name: "#{name}_sa_out")
    sa = maybe_dropout(sa, dropout, "#{name}_sa_drop")
    x = Axon.add(x, sa, name: "#{name}_sa_res")

    x_norm = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn =
      x_norm
      |> Axon.dense(ffn_dim, name: "#{name}_ffn_up")
      |> Axon.activation(:gelu, name: "#{name}_ffn_act")
      |> maybe_dropout(dropout, "#{name}_ffn_drop1")
      |> Axon.dense(hidden_dim, name: "#{name}_ffn_down")
      |> maybe_dropout(dropout, "#{name}_ffn_drop2")

    Axon.add(x, ffn, name: "#{name}_ffn_res")
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input
end
