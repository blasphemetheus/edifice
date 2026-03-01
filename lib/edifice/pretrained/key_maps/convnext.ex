defmodule Edifice.Pretrained.KeyMaps.ConvNeXt do
  @moduledoc """
  Key map for loading HuggingFace ConvNeXt checkpoints into Edifice ConvNeXt models.

  Maps parameter names from the `facebook/convnext-tiny-224` SafeTensors format
  to the Axon layer names used by `Edifice.Vision.ConvNeXt`.

  ## Key Differences

  - **Pointwise convolutions**: HuggingFace uses `nn.Linear` (rank-2 weights) for
    `pwconv1`/`pwconv2`, while Edifice uses `Axon.conv` with 1x1 kernels (rank-4).
    Linear weights are transposed and reshaped to `{1, 1, in, out}`.
  - **Conv2d layout**: PyTorch stores conv weights as `{out, in, kH, kW}` (OIHW),
    while Axon uses `{kH, kW, in, out}`. All conv weights are permuted accordingly.
  - **Layer scale**: HuggingFace stores `layer_scale_parameter` as a bare 1D tensor.
    Edifice stores it as `stage{i}_block{j}_gamma` inside the `layer_scale` layer,
    shaped `{1, 1, 1, dim}`.
  - **Downsampling index shift**: HuggingFace places downsampling at the *start* of
    stages 1-3, while Edifice places it *between* stages. HF `stages.{i+1}.downsampling_layer`
    maps to Edifice `downsample_{i}`.

  ## Usage

      model_state = Edifice.Pretrained.load(
        Edifice.Pretrained.KeyMaps.ConvNeXt,
        "model.safetensors"
      )

  """

  @behaviour Edifice.Pretrained.KeyMap

  # HF: convnext.encoder.stages.{stage}.layers.{block}.{rest}
  @block_re ~r/^convnext\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(.+)$/

  # HF: convnext.encoder.stages.{stage}.downsampling_layer.{0|1}.{rest}
  @downsample_re ~r/^convnext\.encoder\.stages\.(\d+)\.downsampling_layer\.(\d+)\.(.+)$/

  # --- Stem ---
  @impl true
  def map_key("convnext.embeddings.patch_embeddings.weight"), do: "stem_conv.kernel"
  def map_key("convnext.embeddings.patch_embeddings.bias"), do: "stem_conv.bias"
  def map_key("convnext.embeddings.layernorm.weight"), do: "stem_norm.gamma"
  def map_key("convnext.embeddings.layernorm.bias"), do: "stem_norm.beta"

  # --- Final norm + classifier ---
  def map_key("convnext.layernorm.weight"), do: "final_norm.gamma"
  def map_key("convnext.layernorm.bias"), do: "final_norm.beta"
  def map_key("classifier.weight"), do: "classifier.kernel"
  def map_key("classifier.bias"), do: "classifier.bias"

  def map_key(key) do
    cond do
      match = Regex.run(@block_re, key) ->
        [_, stage, block, rest] = match
        map_block(stage, block, rest)

      match = Regex.run(@downsample_re, key) ->
        [_, stage, idx, rest] = match
        map_downsample(stage, idx, rest)

      true ->
        :unmapped
    end
  end

  # --- Block parameters ---
  defp map_block(s, b, "dwconv.weight"),
    do: "stage#{s}_block#{b}_dw_conv.kernel"

  defp map_block(s, b, "dwconv.bias"),
    do: "stage#{s}_block#{b}_dw_conv.bias"

  defp map_block(s, b, "layernorm.weight"),
    do: "stage#{s}_block#{b}_norm.gamma"

  defp map_block(s, b, "layernorm.bias"),
    do: "stage#{s}_block#{b}_norm.beta"

  # HF pwconv1/pwconv2 are nn.Linear; Edifice pw_expand/pw_project are Axon.conv 1x1
  defp map_block(s, b, "pwconv1.weight"),
    do: "stage#{s}_block#{b}_pw_expand.kernel"

  defp map_block(s, b, "pwconv1.bias"),
    do: "stage#{s}_block#{b}_pw_expand.bias"

  defp map_block(s, b, "pwconv2.weight"),
    do: "stage#{s}_block#{b}_pw_project.kernel"

  defp map_block(s, b, "pwconv2.bias"),
    do: "stage#{s}_block#{b}_pw_project.bias"

  # Layer scale parameter — bare 1D tensor in HF, nested under layer_scale layer in Edifice
  defp map_block(s, b, "layer_scale_parameter"),
    do: "stage#{s}_block#{b}_layer_scale.stage#{s}_block#{b}_gamma"

  defp map_block(_s, _b, _rest), do: :unmapped

  # --- Downsampling ---
  # HF puts downsampling at the start of stages 1,2,3.
  # Edifice puts it between stages as downsample_0, downsample_1, downsample_2.
  # So HF stages.{i}.downsampling_layer → Edifice downsample_{i-1}
  # downsampling_layer.0 = LayerNorm, downsampling_layer.1 = Conv2d
  defp map_downsample(stage, "0", "weight") do
    ds_idx = String.to_integer(stage) - 1
    "downsample_#{ds_idx}_norm.gamma"
  end

  defp map_downsample(stage, "0", "bias") do
    ds_idx = String.to_integer(stage) - 1
    "downsample_#{ds_idx}_norm.beta"
  end

  defp map_downsample(stage, "1", "weight") do
    ds_idx = String.to_integer(stage) - 1
    "downsample_#{ds_idx}_conv.kernel"
  end

  defp map_downsample(stage, "1", "bias") do
    ds_idx = String.to_integer(stage) - 1
    "downsample_#{ds_idx}_conv.bias"
  end

  defp map_downsample(_stage, _idx, _rest), do: :unmapped

  @impl true
  def tensor_transforms do
    [
      # Pointwise conv kernels: HF Linear {out, in} → Axon Conv {1, 1, in, out}
      {~r/pw_(expand|project)\.kernel$/, &linear_to_conv1x1/1},
      # Conv/Dense kernels: rank 4 → OIHW→HWIO permute, rank 2 → transpose
      {~r/\.kernel$/, &transform_kernel/1},
      # Layer scale: HF {dim} → Edifice {1, 1, 1, dim}
      {~r/_gamma$/, &reshape_layer_scale/1}
    ]
  end

  # nn.Linear weight {out, in} → Axon.conv 1x1 kernel {1, 1, in, out}
  defp linear_to_conv1x1(tensor) do
    {out, inp} = Nx.shape(tensor)
    tensor |> Nx.transpose() |> Nx.reshape({1, 1, inp, out})
  end

  # PyTorch Conv2d {O, I, H, W} → Axon Conv {H, W, I, O}
  # PyTorch Linear {out, in} → Axon Dense {in, out}
  defp transform_kernel(tensor) do
    case Nx.rank(tensor) do
      4 -> Nx.transpose(tensor, axes: [2, 3, 1, 0])
      2 -> Nx.transpose(tensor)
      _ -> tensor
    end
  end

  # HF layer_scale_parameter is {dim}, Edifice gamma is {1, 1, 1, dim}
  defp reshape_layer_scale(tensor) do
    {dim} = Nx.shape(tensor)
    Nx.reshape(tensor, {1, 1, 1, dim})
  end
end
