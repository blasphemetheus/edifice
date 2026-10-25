defmodule Edifice.Pretrained.KeyMaps.DETR do
  @moduledoc """
  Key map for loading HuggingFace DETR checkpoints into Edifice DETR models.

  Maps parameter names from the `facebook/detr-resnet-50` SafeTensors format
  to the Axon layer names used by `Edifice.Detection.DETR` with
  `backbone: :resnet50, norm_position: :post`.

  ## Key Differences

  - **Backbone naming**: HuggingFace uses timm-style ResNet naming
    (`model.backbone.conv_encoder.model.layer1.0.conv1.weight`) while Edifice
    uses `backbone_stage0_block0_conv1.kernel`. The timm `layerN` maps to
    Edifice `stage{N-1}`.
  - **Conv2d layout**: PyTorch `{O, I, H, W}` → Axon `{H, W, I, O}`.
  - **No conv bias**: ResNet backbone convolutions have `bias=False`.
  - **BatchNorm**: HuggingFace uses FrozenBatchNorm2d which registers
    `weight`/`bias`/`running_mean`/`running_var` as buffers (no `num_batches_tracked`).
  - **Layer index shift**: HuggingFace encoder/decoder layers are 0-indexed,
    Edifice uses 1-indexed (`layers.0` → `enc_1`).
  - **Classifier prefix**: HuggingFace uses `class_labels_classifier` (on the
    top-level model, no `model.` prefix) and `bbox_predictor.layers.{0,1,2}`.
  - **Object queries**: HuggingFace stores as `model.query_position_embeddings.weight`
    (nn.Embedding), Edifice as `object_queries` (Axon.param).

  ## Usage

      Edifice.Pretrained.from_hub("facebook/detr-resnet-50",
        build_opts: [backbone: :resnet50, norm_position: :post]
      )

  """

  @behaviour Edifice.Pretrained.KeyMap

  # --- Backbone: timm ResNet-50 ---
  # Stem: conv1, bn1
  @backbone_stem_re ~r/^model\.backbone\.conv_encoder\.model\.(conv1|bn1)\.(.+)$/

  # Bottleneck: layer{1-4}.{block}.{conv1|conv2|conv3|bn1|bn2|bn3}.{param}
  @backbone_block_re ~r/^model\.backbone\.conv_encoder\.model\.layer(\d+)\.(\d+)\.(conv\d|bn\d)\.(.+)$/

  # Downsample: layer{1-4}.{block}.downsample.{0=conv|1=bn}.{param}
  @backbone_downsample_re ~r/^model\.backbone\.conv_encoder\.model\.layer(\d+)\.(\d+)\.downsample\.(\d+)\.(.+)$/

  # --- Input projection ---
  @input_proj_re ~r/^model\.input_projection\.(weight|bias)$/

  # --- Encoder ---
  @encoder_attn_re ~r/^model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$/
  @encoder_fc_re ~r/^model\.encoder\.layers\.(\d+)\.(fc1|fc2)\.(weight|bias)$/
  @encoder_norm_re ~r/^model\.encoder\.layers\.(\d+)\.(self_attn_layer_norm|final_layer_norm)\.(weight|bias)$/

  # --- Decoder ---
  @decoder_self_attn_re ~r/^model\.decoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$/
  @decoder_cross_attn_re ~r/^model\.decoder\.layers\.(\d+)\.encoder_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$/
  @decoder_fc_re ~r/^model\.decoder\.layers\.(\d+)\.(fc1|fc2)\.(weight|bias)$/
  @decoder_norm_re ~r/^model\.decoder\.layers\.(\d+)\.(self_attn_layer_norm|encoder_attn_layer_norm|final_layer_norm)\.(weight|bias)$/
  @decoder_output_norm_re ~r/^model\.decoder\.layernorm\.(weight|bias)$/

  # --- Object queries ---
  @query_embed_re ~r/^model\.query_position_embeddings\.weight$/

  # --- Prediction heads ---
  @class_head_re ~r/^class_labels_classifier\.(weight|bias)$/
  @bbox_head_re ~r/^bbox_predictor\.layers\.(\d+)\.(weight|bias)$/

  @impl true
  def map_key(key) do
    with :no_match <- map_backbone_key(key),
         :no_match <- map_encoder_key(key),
         :no_match <- map_decoder_key(key) do
      map_head_key(key)
    end
  end

  defp map_backbone_key(key) do
    cond do
      # Backbone stem
      match = Regex.run(@backbone_stem_re, key) ->
        [_, module, param] = match
        map_backbone_stem(module, param)

      # Backbone bottleneck blocks
      match = Regex.run(@backbone_block_re, key) ->
        [_, layer, block, module, param] = match
        map_backbone_block(layer, block, module, param)

      # Backbone downsample
      match = Regex.run(@backbone_downsample_re, key) ->
        [_, layer, block, idx, param] = match
        map_backbone_downsample(layer, block, idx, param)

      # Input projection
      match = Regex.run(@input_proj_re, key) ->
        [_, param] = match
        map_input_proj(param)

      true ->
        :no_match
    end
  end

  defp map_encoder_key(key) do
    cond do
      # Encoder self-attention
      match = Regex.run(@encoder_attn_re, key) ->
        [_, layer, proj, param] = match
        map_encoder_attn(layer, proj, param)

      # Encoder FFN
      match = Regex.run(@encoder_fc_re, key) ->
        [_, layer, fc, param] = match
        map_encoder_fc(layer, fc, param)

      # Encoder norms
      match = Regex.run(@encoder_norm_re, key) ->
        [_, layer, norm, param] = match
        map_encoder_norm(layer, norm, param)

      true ->
        :no_match
    end
  end

  defp map_decoder_key(key) do
    cond do
      # Decoder self-attention
      match = Regex.run(@decoder_self_attn_re, key) ->
        [_, layer, proj, param] = match
        map_decoder_self_attn(layer, proj, param)

      # Decoder cross-attention
      match = Regex.run(@decoder_cross_attn_re, key) ->
        [_, layer, proj, param] = match
        map_decoder_cross_attn(layer, proj, param)

      # Decoder FFN
      match = Regex.run(@decoder_fc_re, key) ->
        [_, layer, fc, param] = match
        map_decoder_fc(layer, fc, param)

      # Decoder norms
      match = Regex.run(@decoder_norm_re, key) ->
        [_, layer, norm, param] = match
        map_decoder_norm(layer, norm, param)

      # Decoder output layernorm
      match = Regex.run(@decoder_output_norm_re, key) ->
        [_, param] = match
        map_norm_param("decoder_norm", param)

      true ->
        :no_match
    end
  end

  defp map_head_key(key) do
    cond do
      # Object queries
      Regex.match?(@query_embed_re, key) ->
        "object_queries.kernel"

      # Class head
      match = Regex.run(@class_head_re, key) ->
        [_, param] = match
        map_dense_param("class_head", param)

      # BBox head
      match = Regex.run(@bbox_head_re, key) ->
        [_, idx, param] = match
        map_bbox_head(idx, param)

      # Skip backbone position embedding (sine — no parameters) and pooler
      String.starts_with?(key, "model.backbone.position_embedding") ->
        :skip

      true ->
        :unmapped
    end
  end

  # --- Backbone stem ---

  defp map_backbone_stem("conv1", "weight"), do: "backbone_stem_conv.kernel"
  defp map_backbone_stem("bn1", param), do: map_bn_param("backbone_stem_bn", param)
  defp map_backbone_stem(_module, _param), do: :unmapped

  # --- Backbone bottleneck blocks ---
  # timm layer{N} → Edifice stage{N-1}

  defp map_backbone_block(layer, block, module, param) do
    stage = String.to_integer(layer) - 1
    prefix = "backbone_stage#{stage}_block#{block}"

    case module do
      "conv" <> n -> map_dense_param("#{prefix}_conv#{n}", param)
      "bn" <> n -> map_bn_param("#{prefix}_bn#{n}", param)
      _ -> :unmapped
    end
  end

  # --- Backbone downsample (projection shortcut) ---
  # downsample.0 = Conv2d, downsample.1 = BatchNorm2d

  defp map_backbone_downsample(layer, block, "0", param) do
    stage = String.to_integer(layer) - 1
    map_dense_param("backbone_stage#{stage}_block#{block}_skip_proj", param)
  end

  defp map_backbone_downsample(layer, block, "1", param) do
    stage = String.to_integer(layer) - 1
    map_bn_param("backbone_stage#{stage}_block#{block}_skip_bn", param)
  end

  defp map_backbone_downsample(_layer, _block, _idx, _param), do: :unmapped

  # --- Input projection ---

  defp map_input_proj("weight"), do: "input_proj.kernel"
  defp map_input_proj("bias"), do: "input_proj.bias"
  defp map_input_proj(_), do: :unmapped

  # --- Encoder attention ---
  # HF 0-indexed → Edifice 1-indexed

  defp map_encoder_attn(layer, proj, param) do
    i = String.to_integer(layer) + 1
    edifice_proj = attn_proj_name(proj)
    map_dense_param("enc_#{i}_self_attn_#{edifice_proj}", param)
  end

  defp map_encoder_fc(layer, fc, param) do
    i = String.to_integer(layer) + 1

    edifice_name =
      case fc do
        "fc1" -> "enc_#{i}_ffn_up"
        "fc2" -> "enc_#{i}_ffn_down"
      end

    map_dense_param(edifice_name, param)
  end

  defp map_encoder_norm(layer, norm, param) do
    i = String.to_integer(layer) + 1

    edifice_name =
      case norm do
        "self_attn_layer_norm" -> "enc_#{i}_self_attn_norm"
        "final_layer_norm" -> "enc_#{i}_ffn_norm"
      end

    map_norm_param(edifice_name, param)
  end

  # --- Decoder self-attention ---

  defp map_decoder_self_attn(layer, proj, param) do
    i = String.to_integer(layer) + 1
    edifice_proj = attn_proj_name(proj)
    map_dense_param("dec_block_#{i}_attn_#{edifice_proj}", param)
  end

  # --- Decoder cross-attention ---

  defp map_decoder_cross_attn(layer, proj, param) do
    i = String.to_integer(layer) + 1
    edifice_proj = attn_proj_name(proj)
    map_dense_param("dec_block_#{i}_cross_attn_#{edifice_proj}", param)
  end

  defp map_decoder_fc(layer, fc, param) do
    i = String.to_integer(layer) + 1

    edifice_name =
      case fc do
        "fc1" -> "dec_block_#{i}_ffn_up"
        "fc2" -> "dec_block_#{i}_ffn_down"
      end

    map_dense_param(edifice_name, param)
  end

  defp map_decoder_norm(layer, norm, param) do
    i = String.to_integer(layer) + 1

    edifice_name =
      case norm do
        "self_attn_layer_norm" -> "dec_block_#{i}_attn_norm"
        "encoder_attn_layer_norm" -> "dec_block_#{i}_cross_attn_norm"
        "final_layer_norm" -> "dec_block_#{i}_ffn_norm"
      end

    map_norm_param(edifice_name, param)
  end

  # --- BBox head ---
  # HF bbox_predictor.layers.{0,1,2} → Edifice bbox_mlp{1,2,3}

  defp map_bbox_head(idx, param) do
    mlp_idx = String.to_integer(idx) + 1
    map_dense_param("bbox_mlp#{mlp_idx}", param)
  end

  # --- Helpers ---

  defp attn_proj_name("q_proj"), do: "q"
  defp attn_proj_name("k_proj"), do: "k"
  defp attn_proj_name("v_proj"), do: "v"
  defp attn_proj_name("out_proj"), do: "out"

  defp map_dense_param(prefix, "weight"), do: "#{prefix}.kernel"
  defp map_dense_param(prefix, "bias"), do: "#{prefix}.bias"
  defp map_dense_param(_prefix, _param), do: :unmapped

  defp map_norm_param(prefix, "weight"), do: "#{prefix}.gamma"
  defp map_norm_param(prefix, "bias"), do: "#{prefix}.beta"
  defp map_norm_param(_prefix, _param), do: :unmapped

  defp map_bn_param(prefix, "weight"), do: "#{prefix}.gamma"
  defp map_bn_param(prefix, "bias"), do: "#{prefix}.beta"
  defp map_bn_param(prefix, "running_mean"), do: "#{prefix}.mean"
  defp map_bn_param(prefix, "running_var"), do: "#{prefix}.var"
  defp map_bn_param(_prefix, "num_batches_tracked"), do: :skip
  defp map_bn_param(_prefix, _param), do: :unmapped

  @impl true
  def tensor_transforms do
    [
      # Conv kernels: PyTorch {O, I, H, W} → Axon {H, W, I, O}
      # Dense kernels: PyTorch {out, in} → Axon {in, out}
      {~r/\.kernel$/, &transform_kernel/1}
    ]
  end

  defp transform_kernel(tensor) do
    case Nx.rank(tensor) do
      4 -> Nx.transpose(tensor, axes: [2, 3, 1, 0])
      2 -> Nx.transpose(tensor)
      _ -> tensor
    end
  end
end
