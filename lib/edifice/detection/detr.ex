defmodule Edifice.Detection.DETR do
  @moduledoc """
  DETR: End-to-End Object Detection with Transformers.

  Reformulates object detection as a direct set prediction problem, eliminating
  hand-designed components like non-maximum suppression (NMS) and anchor
  generation. A CNN backbone extracts image features, which are flattened with
  2D positional encodings and processed by a transformer encoder. Learned object
  queries attend to the encoded features via a transformer decoder, and
  per-query prediction heads output class logits and bounding box coordinates.

  The set-based design means each object query predicts at most one object.
  During training, predictions are matched to ground-truth objects via the
  Hungarian algorithm (bipartite matching), and the loss combines
  cross-entropy, L1, and generalized IoU. This module builds the architecture;
  matching and loss computation are handled externally.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v-----------------------+
  | CNN Backbone (ResNet-style)  |  Conv layers → feature map
  +-----+-----------------------+
        |  [batch, backbone_channels, H', W']
        v
  +-----v-----------------------+
  | 1x1 Conv Projection         |  Reduce to hidden_dim
  +-----+-----------------------+
        |  [batch, hidden_dim, H', W']
        v
  Flatten + Transpose → [batch, H'*W', hidden_dim]
  Add 2D sinusoidal positional encoding
        |
  +-----v-----------------------+
  | Transformer Encoder x N      |  Self-attention over spatial features
  +-----+-----------------------+
        |  Memory: [batch, H'*W', hidden_dim]
        v
  +-----v-----------------------+
  | Transformer Decoder x N      |  Object queries cross-attend to memory
  | (object queries as input)    |  + self-attend to each other
  +-----+-----------------------+
        |  [batch, num_queries, hidden_dim]
        v
  +-----v-----------+  +--------v---------+
  | Class Head       |  | BBox Head (MLP)   |
  | Linear → C+1     |  | 3-layer → sigmoid |
  +------------------+  +-------------------+
        |                       |
        v                       v
  [batch, num_queries,    [batch, num_queries, 4]
   num_classes + 1]       (cx, cy, w, h) in [0,1]
  ```

  ## Usage

      model = DETR.build(
        image_size: 512,
        num_queries: 100,
        num_classes: 80,
        hidden_dim: 256
      )

      # model outputs %{class_logits: ..., bbox_pred: ...}
      # class_logits: [batch, num_queries, num_classes + 1]
      # bbox_pred: [batch, num_queries, 4]

  ## References

  - Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020)
    https://arxiv.org/abs/2005.12872
  """

  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Utils.FusedOps

  @default_image_size 512
  @default_in_channels 3
  @default_hidden_dim 256
  @default_num_heads 8
  @default_num_encoder_layers 6
  @default_num_decoder_layers 6
  @default_ffn_dim 2048
  @default_num_queries 100
  @default_num_classes 80
  @default_dropout 0.1
  @default_backbone_channels 256
  @default_backbone_stages 5

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_encoder_layers, pos_integer()}
          | {:num_decoder_layers, pos_integer()}
          | {:ffn_dim, pos_integer()}
          | {:num_queries, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:dropout, float()}
          | {:backbone_channels, pos_integer()}
          | {:backbone_stages, pos_integer()}

  @doc """
  Build a DETR model.

  ## Options

    - `:image_size` - Input image size, square (default: 512)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_dim` - Hidden dimension throughout the transformer (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_encoder_layers` - Number of transformer encoder layers (default: 6)
    - `:num_decoder_layers` - Number of transformer decoder layers (default: 6)
    - `:ffn_dim` - Feed-forward network hidden dimension (default: 2048)
    - `:num_queries` - Number of learned object queries (default: 100)
    - `:num_classes` - Number of object classes (default: 80)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:backbone_channels` - Backbone output channels before projection (default: 256)
    - `:backbone_stages` - Number of stride-2 conv stages in backbone (default: 5, giving 32x downscale)

  ## Returns

    An `Axon.container` outputting `%{class_logits: ..., bbox_pred: ...}`.
    - `class_logits`: `[batch, num_queries, num_classes + 1]` (includes no-object class)
    - `bbox_pred`: `[batch, num_queries, 4]` (cx, cy, w, h in [0, 1])
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_encoder_layers = Keyword.get(opts, :num_encoder_layers, @default_num_encoder_layers)
    num_decoder_layers = Keyword.get(opts, :num_decoder_layers, @default_num_decoder_layers)
    ffn_dim = Keyword.get(opts, :ffn_dim, @default_ffn_dim)
    num_queries = Keyword.get(opts, :num_queries, @default_num_queries)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    backbone_channels = Keyword.get(opts, :backbone_channels, @default_backbone_channels)
    backbone_stages = Keyword.get(opts, :backbone_stages, @default_backbone_stages)

    # Input image (channels-last for Axon conv layers)
    input = Axon.input("image", shape: {nil, image_size, image_size, in_channels})

    # CNN backbone: produces [batch, H', W', backbone_channels]
    features = cnn_backbone(input, backbone_channels, backbone_stages, "backbone")

    # 1x1 conv projection to hidden_dim: [batch, H', W', hidden_dim]
    features = Axon.conv(features, hidden_dim, kernel_size: {1, 1}, name: "input_proj")

    # Flatten spatial dims: [batch, H'*W', hidden_dim]
    features =
      Axon.nx(
        features,
        fn t ->
          {batch, h, w, d} = Nx.shape(t)
          Nx.reshape(t, {batch, h * w, d})
        end,
        name: "flatten_spatial"
      )

    # 2D sinusoidal positional encoding (added at each encoder layer)
    spatial_pe =
      Axon.nx(
        features,
        fn t ->
          {_batch, seq_len, dim} = Nx.shape(t)
          build_2d_sinusoidal_pe(seq_len, dim)
        end,
        name: "spatial_pe"
      )

    # Transformer encoder
    memory =
      Enum.reduce(1..num_encoder_layers, features, fn i, acc ->
        encoder_layer(acc, spatial_pe, hidden_dim, num_heads, ffn_dim, dropout, "enc_#{i}")
      end)

    # Learned object queries: [num_queries, hidden_dim]
    query_embed =
      Axon.param("object_queries", {num_queries, hidden_dim}, initializer: :glorot_uniform)

    # Broadcast queries to batch: [batch, num_queries, hidden_dim]
    queries =
      Axon.layer(
        fn mem, qe, _opts ->
          batch_size = Nx.axis_size(mem, 0)
          Nx.broadcast(Nx.new_axis(qe, 0), {batch_size, Nx.axis_size(qe, 0), Nx.axis_size(qe, 1)})
        end,
        [memory, query_embed],
        name: "broadcast_queries",
        op_name: :broadcast_queries
      )

    # Decoder target starts as zeros, query_embed is positional
    target =
      Axon.nx(
        queries,
        fn q ->
          Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), Nx.shape(q))
        end,
        name: "decoder_target_init"
      )

    # Transformer decoder
    decoded =
      TransformerBlock.stack(target, memory, num_decoder_layers,
        attention_fn: fn x_norm, name ->
          attention_with_pe(x_norm, x_norm, x_norm, queries, queries, hidden_dim, num_heads, name)
        end,
        cross_attention_fn: fn q_norm, mem, name ->
          attention_with_pe(q_norm, mem, mem, queries, spatial_pe, hidden_dim, num_heads, name)
        end,
        hidden_size: hidden_dim,
        custom_ffn: fn x_norm, name ->
          x_norm
          |> Axon.dense(ffn_dim, name: "#{name}_up")
          |> Axon.activation(:relu, name: "#{name}_act")
          |> maybe_dropout(dropout, "#{name}_drop1")
          |> Axon.dense(hidden_dim, name: "#{name}_down")
        end,
        dropout: dropout,
        name: "dec"
      )

    # Prediction heads
    # Class head: [batch, num_queries, num_classes + 1]
    class_logits = Axon.dense(decoded, num_classes + 1, name: "class_head")

    # BBox head: 3-layer MLP with ReLU + sigmoid output → [batch, num_queries, 4]
    bbox_pred =
      decoded
      |> Axon.dense(hidden_dim, name: "bbox_mlp_1")
      |> Axon.activation(:relu, name: "bbox_act_1")
      |> Axon.dense(hidden_dim, name: "bbox_mlp_2")
      |> Axon.activation(:relu, name: "bbox_act_2")
      |> Axon.dense(4, name: "bbox_mlp_3")
      |> Axon.activation(:sigmoid, name: "bbox_sigmoid")

    Axon.container(%{class_logits: class_logits, bbox_pred: bbox_pred})
  end

  @doc """
  Get the output size of a DETR model.

  Returns the total output dimension per query: `num_classes + 1 + 4`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    num_classes + 1 + 4
  end

  # ============================================================================
  # CNN Backbone
  # ============================================================================

  # Simplified CNN backbone (ResNet-style). In production, this would be a
  # pretrained ResNet-50 with frozen batch norm. Here we use a series of
  # strided convolutions to downsample the image. Each stage halves spatial
  # dimensions. 5 stages = 32x downscale, 3 stages = 8x, etc.
  # Channels-last format: [batch, H, W, C].
  @spec cnn_backbone(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp cnn_backbone(input, out_channels, num_stages, name) do
    # Channel progression: ramp up to out_channels across stages
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
      |> Axon.activation(:relu, name: "#{name}_act#{i}")
    end)
  end

  # ============================================================================
  # Transformer Encoder Layer
  # ============================================================================

  # Standard transformer encoder layer with positional encoding added to Q/K
  # at each layer (not just the input), as per the DETR paper.
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
  defp encoder_layer(x, spatial_pe, hidden_dim, num_heads, ffn_dim, dropout, name) do
    # Pre-norm self-attention with positional encoding on Q and K
    x_norm = Axon.layer_norm(x, name: "#{name}_self_attn_norm")

    self_attn =
      attention_with_pe(
        x_norm,
        x_norm,
        x_norm,
        spatial_pe,
        spatial_pe,
        hidden_dim,
        num_heads,
        "#{name}_self_attn"
      )

    self_attn = maybe_dropout(self_attn, dropout, "#{name}_self_attn_drop")
    x = Axon.add(x, self_attn, name: "#{name}_self_attn_residual")

    # Pre-norm FFN
    x_norm = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      x_norm
      |> Axon.dense(ffn_dim, name: "#{name}_ffn_up")
      |> Axon.activation(:relu, name: "#{name}_ffn_act")
      |> maybe_dropout(dropout, "#{name}_ffn_drop1")
      |> Axon.dense(hidden_dim, name: "#{name}_ffn_down")
      |> maybe_dropout(dropout, "#{name}_ffn_drop2")

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Multi-Head Attention with Positional Encoding
  # ============================================================================

  # Attention where positional encodings are added to Q and K before computing
  # scores, as specified in the DETR paper. Values are unmodified.
  @spec attention_with_pe(
          Axon.t(),
          Axon.t(),
          Axon.t(),
          Axon.t(),
          Axon.t(),
          pos_integer(),
          pos_integer(),
          String.t()
        ) :: Axon.t()
  defp attention_with_pe(query, key, value, q_pe, k_pe, hidden_dim, num_heads, name) do
    q = Axon.dense(query, hidden_dim, name: "#{name}_q")
    k = Axon.dense(key, hidden_dim, name: "#{name}_k")
    v = Axon.dense(value, hidden_dim, name: "#{name}_v")

    # Add positional encodings to Q and K
    q = Axon.add(q, q_pe, name: "#{name}_q_pe")
    k = Axon.add(k, k_pe, name: "#{name}_k_pe")

    head_dim = div(hidden_dim, num_heads)

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          compute_attention(q_t, k_t, v_t, num_heads, head_dim)
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :mha_compute
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  @spec compute_attention(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer()
        ) ::
          Nx.Tensor.t()
  defp compute_attention(q, k, v, num_heads, head_dim) do
    {batch, q_len, _} = Nx.shape(q)
    {_, kv_len, _} = Nx.shape(k)

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, q_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)
    weights = FusedOps.fused_softmax(scores)

    # Apply to values: [batch, heads, q_len, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, q_len, hidden_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, q_len, num_heads * head_dim})
  end

  # ============================================================================
  # 2D Sinusoidal Positional Encoding
  # ============================================================================

  # Builds 2D sinusoidal positional encoding for a flattened spatial grid.
  # Assumes the spatial tokens come from a roughly square feature map.
  # Encodes x and y positions independently, then concatenates.
  @spec build_2d_sinusoidal_pe(pos_integer(), pos_integer()) :: Nx.Tensor.t()
  defp build_2d_sinusoidal_pe(seq_len, dim) do
    # Approximate grid dimensions (assume roughly square)
    h = seq_len |> :math.sqrt() |> ceil() |> trunc()
    w = ceil(seq_len / h) |> trunc()

    half_dim = div(dim, 2)
    quarter_dim = div(half_dim, 2)

    # Frequency bands
    freq_indices = Nx.iota({quarter_dim})

    inv_freq =
      Nx.exp(
        Nx.negate(Nx.multiply(freq_indices, Nx.divide(Nx.log(Nx.tensor(10_000.0)), quarter_dim)))
      )

    # Y positions (rows)
    y_pos = Nx.iota({h, 1}) |> Nx.broadcast({h, w}) |> Nx.reshape({h * w, 1})
    y_pos = Nx.divide(y_pos, Nx.tensor(max(h - 1, 1), type: :f32))

    # X positions (cols)
    x_pos = Nx.iota({1, w}) |> Nx.broadcast({h, w}) |> Nx.reshape({h * w, 1})
    x_pos = Nx.divide(x_pos, Nx.tensor(max(w - 1, 1), type: :f32))

    # Sinusoidal encoding for y: [h*w, quarter_dim] each
    y_angles = Nx.dot(y_pos, Nx.reshape(inv_freq, {1, quarter_dim}))
    y_pe = Nx.concatenate([Nx.sin(y_angles), Nx.cos(y_angles)], axis: 1)

    # Sinusoidal encoding for x: [h*w, quarter_dim] each
    x_angles = Nx.dot(x_pos, Nx.reshape(inv_freq, {1, quarter_dim}))
    x_pe = Nx.concatenate([Nx.sin(x_angles), Nx.cos(x_angles)], axis: 1)

    # Concatenate: [h*w, dim]
    pe = Nx.concatenate([y_pe, x_pe], axis: 1)

    # Truncate or pad to exact seq_len (in case h*w > seq_len)
    pe = Nx.slice_along_axis(pe, 0, seq_len, axis: 0)

    # Add batch dimension: [1, seq_len, dim]
    Nx.reshape(pe, {1, seq_len, dim})
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input
end
