defmodule Edifice.Pretrained.KeyMaps.ViT do
  @moduledoc """
  Key map for loading HuggingFace ViT checkpoints into Edifice ViT models.

  Maps parameter names from the `google/vit-base-patch16-224` SafeTensors format
  to the Axon layer names used by `Edifice.Vision.ViT`.

  ## QKV Concatenation

  HuggingFace ViT stores separate Q, K, V weight matrices, while Edifice uses a
  combined QKV projection. This key map uses `concat_keys/0` to concatenate the
  three source matrices into one target along axis 0 (output dimension).

  ## Usage

      model_state = Edifice.Pretrained.load(
        Edifice.Pretrained.KeyMaps.ViT,
        "model.safetensors"
      )

  """

  @behaviour Edifice.Pretrained.KeyMap

  alias Edifice.Pretrained.Transform

  # Regex for matching encoder layer keys: captures (layer_index, rest_of_path)
  @encoder_layer_re ~r/^vit\.encoder\.layer\.(\d+)\.(.+)$/

  @impl true
  def map_key("vit.embeddings.patch_embeddings.projection.weight"),
    do: "patch_embed_proj.kernel"

  def map_key("vit.embeddings.patch_embeddings.projection.bias"),
    do: "patch_embed_proj.bias"

  def map_key("vit.embeddings.cls_token"), do: "cls_token_proj.kernel"
  def map_key("vit.embeddings.position_embeddings"), do: "pos_embed_proj.kernel"

  def map_key("vit.layernorm.weight"), do: "final_norm.gamma"
  def map_key("vit.layernorm.bias"), do: "final_norm.beta"

  def map_key("classifier.weight"), do: "classifier.kernel"
  def map_key("classifier.bias"), do: "classifier.bias"

  def map_key(key) do
    case Regex.run(@encoder_layer_re, key) do
      [_, idx, rest] -> map_encoder_layer(idx, rest)
      nil -> map_fallback(key)
    end
  end

  defp map_encoder_layer(idx, "layernorm_before.weight"), do: "block_#{idx}_norm1.gamma"
  defp map_encoder_layer(idx, "layernorm_before.bias"), do: "block_#{idx}_norm1.beta"

  # Q/K/V mapped to intermediate keys — concat_keys/0 combines them
  defp map_encoder_layer(idx, "attention.attention.query.weight"),
    do: "block_#{idx}_attn_q.kernel"

  defp map_encoder_layer(idx, "attention.attention.query.bias"),
    do: "block_#{idx}_attn_q.bias"

  defp map_encoder_layer(idx, "attention.attention.key.weight"),
    do: "block_#{idx}_attn_k.kernel"

  defp map_encoder_layer(idx, "attention.attention.key.bias"),
    do: "block_#{idx}_attn_k.bias"

  defp map_encoder_layer(idx, "attention.attention.value.weight"),
    do: "block_#{idx}_attn_v.kernel"

  defp map_encoder_layer(idx, "attention.attention.value.bias"),
    do: "block_#{idx}_attn_v.bias"

  defp map_encoder_layer(idx, "attention.output.dense.weight"),
    do: "block_#{idx}_attn_proj.kernel"

  defp map_encoder_layer(idx, "attention.output.dense.bias"),
    do: "block_#{idx}_attn_proj.bias"

  defp map_encoder_layer(idx, "layernorm_after.weight"), do: "block_#{idx}_norm2.gamma"
  defp map_encoder_layer(idx, "layernorm_after.bias"), do: "block_#{idx}_norm2.beta"

  defp map_encoder_layer(idx, "intermediate.dense.weight"),
    do: "block_#{idx}_mlp_fc1.kernel"

  defp map_encoder_layer(idx, "intermediate.dense.bias"),
    do: "block_#{idx}_mlp_fc1.bias"

  defp map_encoder_layer(idx, "output.dense.weight"), do: "block_#{idx}_mlp_fc2.kernel"
  defp map_encoder_layer(idx, "output.dense.bias"), do: "block_#{idx}_mlp_fc2.bias"

  defp map_encoder_layer(_idx, _rest), do: :unmapped

  # HF ViT pooler is not used in Edifice — skip it
  defp map_fallback("vit.pooler." <> _), do: :skip
  defp map_fallback(_), do: :unmapped

  @impl true
  def tensor_transforms do
    [
      {~r/\.kernel$/, fn tensor ->
        case Nx.rank(tensor) do
          # CLS token: [1, 1, D] -> [1, D]
          3 -> Nx.squeeze(tensor, axes: [0])
          # Linear weights: transpose [out, in] -> [in, out]
          2 -> Transform.transpose_linear(tensor)
          _ -> tensor
        end
      end},
      {~r/\.bias$/, &Function.identity/1}
    ]
  end

  @impl true
  def concat_keys do
    build_concat_keys(12)
  end

  @doc """
  Builds concat key rules for the given number of encoder layers.

  Useful when loading non-base ViT variants with different layer counts.
  """
  def build_concat_keys(num_layers) do
    for i <- 0..(num_layers - 1), {suffix, axis} <- [{"kernel", 1}, {"bias", 0}], into: %{} do
      target = "block_#{i}_attn_qkv.#{suffix}"

      sources = [
        "block_#{i}_attn_q.#{suffix}",
        "block_#{i}_attn_k.#{suffix}",
        "block_#{i}_attn_v.#{suffix}"
      ]

      {target, {sources, axis}}
    end
  end
end
