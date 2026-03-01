defmodule Edifice.Pretrained.KeyMaps.Whisper do
  @moduledoc """
  Key map for loading HuggingFace Whisper checkpoints into Edifice Whisper models.

  Maps parameter names from the `openai/whisper-base` SafeTensors format
  to the Axon layer names used by `Edifice.Audio.Whisper`.

  ## Index Shift

  HuggingFace uses 0-based layer indices (`layers.0`, `layers.1`, ...),
  while Edifice uses 1-based block names (`enc_block_1`, `enc_block_2`, ...).
  All layer indices are shifted by +1 during mapping.

  ## Skipped Keys

  - `model.encoder.embed_positions.weight` — Edifice uses sinusoidal PE (not trainable)
  - `proj_out.weight` — Edifice uses its own `dec_output_proj` dense layer

  ## Usage

      model_state = Edifice.Pretrained.load(
        Edifice.Pretrained.KeyMaps.Whisper,
        "model.safetensors"
      )

  """

  @behaviour Edifice.Pretrained.KeyMap

  alias Edifice.Pretrained.Transform

  @encoder_layer_re ~r/^model\.encoder\.layers\.(\d+)\.(.+)$/
  @decoder_layer_re ~r/^model\.decoder\.layers\.(\d+)\.(.+)$/

  @impl true
  def map_key("model.encoder.conv1.weight"), do: "enc_conv1.kernel"
  def map_key("model.encoder.conv1.bias"), do: "enc_conv1.bias"
  def map_key("model.encoder.conv2.weight"), do: "enc_conv2.kernel"
  def map_key("model.encoder.conv2.bias"), do: "enc_conv2.bias"

  # Sinusoidal PE — not trainable in Edifice
  def map_key("model.encoder.embed_positions.weight"), do: :skip

  def map_key("model.encoder.layer_norm.weight"), do: "enc_final_norm.gamma"
  def map_key("model.encoder.layer_norm.bias"), do: "enc_final_norm.beta"

  def map_key("model.decoder.embed_tokens.weight"), do: "dec_token_embed.kernel"
  def map_key("model.decoder.embed_positions.weight"), do: "dec_pos_embed.kernel"

  def map_key("model.decoder.layer_norm.weight"), do: "dec_final_norm.gamma"
  def map_key("model.decoder.layer_norm.bias"), do: "dec_final_norm.beta"

  # Edifice uses its own dec_output_proj
  def map_key("proj_out.weight"), do: :skip

  def map_key(key) do
    cond do
      match = Regex.run(@encoder_layer_re, key) ->
        [_, idx_str, rest] = match
        idx = String.to_integer(idx_str) + 1
        map_encoder_layer(idx, rest)

      match = Regex.run(@decoder_layer_re, key) ->
        [_, idx_str, rest] = match
        idx = String.to_integer(idx_str) + 1
        map_decoder_layer(idx, rest)

      true ->
        :unmapped
    end
  end

  # Encoder self-attention
  defp map_encoder_layer(i, "self_attn.q_proj.weight"), do: "enc_block_#{i}_attn_q.kernel"
  defp map_encoder_layer(i, "self_attn.q_proj.bias"), do: "enc_block_#{i}_attn_q.bias"
  defp map_encoder_layer(i, "self_attn.k_proj.weight"), do: "enc_block_#{i}_attn_k.kernel"
  defp map_encoder_layer(i, "self_attn.k_proj.bias"), do: "enc_block_#{i}_attn_k.bias"
  defp map_encoder_layer(i, "self_attn.v_proj.weight"), do: "enc_block_#{i}_attn_v.kernel"
  defp map_encoder_layer(i, "self_attn.v_proj.bias"), do: "enc_block_#{i}_attn_v.bias"
  defp map_encoder_layer(i, "self_attn.out_proj.weight"), do: "enc_block_#{i}_attn_out.kernel"
  defp map_encoder_layer(i, "self_attn.out_proj.bias"), do: "enc_block_#{i}_attn_out.bias"
  defp map_encoder_layer(i, "self_attn_layer_norm.weight"), do: "enc_block_#{i}_attn_norm.gamma"
  defp map_encoder_layer(i, "self_attn_layer_norm.bias"), do: "enc_block_#{i}_attn_norm.beta"

  # Encoder FFN
  defp map_encoder_layer(i, "fc1.weight"), do: "enc_block_#{i}_ffn_up.kernel"
  defp map_encoder_layer(i, "fc1.bias"), do: "enc_block_#{i}_ffn_up.bias"
  defp map_encoder_layer(i, "fc2.weight"), do: "enc_block_#{i}_ffn_down.kernel"
  defp map_encoder_layer(i, "fc2.bias"), do: "enc_block_#{i}_ffn_down.bias"
  defp map_encoder_layer(i, "final_layer_norm.weight"), do: "enc_block_#{i}_ffn_norm.gamma"
  defp map_encoder_layer(i, "final_layer_norm.bias"), do: "enc_block_#{i}_ffn_norm.beta"
  defp map_encoder_layer(_i, _rest), do: :unmapped

  # Decoder self-attention
  defp map_decoder_layer(i, "self_attn.q_proj.weight"), do: "dec_block_#{i}_attn_q.kernel"
  defp map_decoder_layer(i, "self_attn.q_proj.bias"), do: "dec_block_#{i}_attn_q.bias"
  defp map_decoder_layer(i, "self_attn.k_proj.weight"), do: "dec_block_#{i}_attn_k.kernel"
  defp map_decoder_layer(i, "self_attn.k_proj.bias"), do: "dec_block_#{i}_attn_k.bias"
  defp map_decoder_layer(i, "self_attn.v_proj.weight"), do: "dec_block_#{i}_attn_v.kernel"
  defp map_decoder_layer(i, "self_attn.v_proj.bias"), do: "dec_block_#{i}_attn_v.bias"
  defp map_decoder_layer(i, "self_attn.out_proj.weight"), do: "dec_block_#{i}_attn_out.kernel"
  defp map_decoder_layer(i, "self_attn.out_proj.bias"), do: "dec_block_#{i}_attn_out.bias"
  defp map_decoder_layer(i, "self_attn_layer_norm.weight"), do: "dec_block_#{i}_attn_norm.gamma"
  defp map_decoder_layer(i, "self_attn_layer_norm.bias"), do: "dec_block_#{i}_attn_norm.beta"

  # Decoder cross-attention
  defp map_decoder_layer(i, "encoder_attn.q_proj.weight"),
    do: "dec_block_#{i}_cross_attn_q_proj.kernel"

  defp map_decoder_layer(i, "encoder_attn.q_proj.bias"),
    do: "dec_block_#{i}_cross_attn_q_proj.bias"

  defp map_decoder_layer(i, "encoder_attn.k_proj.weight"),
    do: "dec_block_#{i}_cross_attn_k_proj.kernel"

  defp map_decoder_layer(i, "encoder_attn.k_proj.bias"),
    do: "dec_block_#{i}_cross_attn_k_proj.bias"

  defp map_decoder_layer(i, "encoder_attn.v_proj.weight"),
    do: "dec_block_#{i}_cross_attn_v_proj.kernel"

  defp map_decoder_layer(i, "encoder_attn.v_proj.bias"),
    do: "dec_block_#{i}_cross_attn_v_proj.bias"

  defp map_decoder_layer(i, "encoder_attn.out_proj.weight"),
    do: "dec_block_#{i}_cross_attn_out_proj.kernel"

  defp map_decoder_layer(i, "encoder_attn.out_proj.bias"),
    do: "dec_block_#{i}_cross_attn_out_proj.bias"

  defp map_decoder_layer(i, "encoder_attn_layer_norm.weight"),
    do: "dec_block_#{i}_cross_attn_norm.gamma"

  defp map_decoder_layer(i, "encoder_attn_layer_norm.bias"),
    do: "dec_block_#{i}_cross_attn_norm.beta"

  # Decoder FFN
  defp map_decoder_layer(i, "fc1.weight"), do: "dec_block_#{i}_ffn_up.kernel"
  defp map_decoder_layer(i, "fc1.bias"), do: "dec_block_#{i}_ffn_up.bias"
  defp map_decoder_layer(i, "fc2.weight"), do: "dec_block_#{i}_ffn_down.kernel"
  defp map_decoder_layer(i, "fc2.bias"), do: "dec_block_#{i}_ffn_down.bias"
  defp map_decoder_layer(i, "final_layer_norm.weight"), do: "dec_block_#{i}_ffn_norm.gamma"
  defp map_decoder_layer(i, "final_layer_norm.bias"), do: "dec_block_#{i}_ffn_norm.beta"
  defp map_decoder_layer(_i, _rest), do: :unmapped

  @impl true
  def tensor_transforms do
    [
      # Transpose linear weight matrices but not embedding kernels
      # Embeddings (token_embed, pos_embed) are already {vocab/seq, dim} in both HF and Axon
      {~r/(?<!embed)\.kernel$/, &Transform.transpose_linear/1}
    ]
  end
end
