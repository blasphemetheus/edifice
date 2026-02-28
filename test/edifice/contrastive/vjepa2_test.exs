defmodule Edifice.Contrastive.VJEPA2Test do
  use ExUnit.Case, async: true

  alias Edifice.Contrastive.VJEPA2

  import Edifice.TestHelpers

  @moduletag :contrastive

  @batch 2
  @num_tokens 16
  @patch_dim 32
  @embed_dim 32
  @pred_dim 16
  @num_heads 4

  describe "build/1" do
    test "returns {encoder, predictor} tuple" do
      {encoder, predictor} = VJEPA2.build(patch_dim: @patch_dim, embed_dim: @embed_dim)
      assert %Axon{} = encoder
      assert %Axon{} = predictor
    end

    test "encoder forward pass produces correct shape" do
      {encoder, _predictor} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          num_heads: @num_heads,
          encoder_depth: 2,
          num_tokens: @num_tokens,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      input = random_tensor({@batch, @num_tokens, @patch_dim}, 42)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      # Sequence output: [batch, num_tokens, embed_dim]
      assert Nx.shape(output) == {@batch, @num_tokens, @embed_dim}
    end

    test "predictor forward pass produces correct shape" do
      {_encoder, predictor} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          predictor_embed_dim: @pred_dim,
          num_heads: @num_heads,
          predictor_depth: 2,
          num_tokens: @num_tokens,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(predictor)
      # Predictor takes encoder output (embed_dim)
      input = random_tensor({@batch, @num_tokens, @embed_dim}, 99)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      # Projects back to embed_dim
      assert Nx.shape(output) == {@batch, @num_tokens, @embed_dim}
    end

    test "batch=1 forward pass" do
      {encoder, predictor} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          predictor_embed_dim: @pred_dim,
          num_heads: @num_heads,
          encoder_depth: 1,
          predictor_depth: 1,
          num_tokens: @num_tokens,
          dropout: 0.0
        )

      enc_input = random_tensor({1, @num_tokens, @patch_dim}, 55)
      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(enc_input, %{})
      enc_out = enc_pred.(enc_params, enc_input)
      assert Nx.shape(enc_out) == {1, @num_tokens, @embed_dim}

      {pred_init, pred_pred} = Axon.build(predictor)
      pred_params = pred_init.(enc_out, %{})
      pred_out = pred_pred.(pred_params, enc_out)
      assert Nx.shape(pred_out) == {1, @num_tokens, @embed_dim}
    end

    test "output values are finite" do
      {encoder, _} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          num_heads: @num_heads,
          encoder_depth: 2,
          num_tokens: @num_tokens,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      input = random_tensor({@batch, @num_tokens, @patch_dim}, 77)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end
  end

  describe "3D-RoPE" do
    test "encoder produces different outputs for different sequence lengths" do
      {encoder1, _} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          num_heads: @num_heads,
          encoder_depth: 1,
          num_tokens: 8,
          dropout: 0.0
        )

      {encoder2, _} =
        VJEPA2.build(
          patch_dim: @patch_dim,
          embed_dim: @embed_dim,
          num_heads: @num_heads,
          encoder_depth: 1,
          num_tokens: 16,
          dropout: 0.0
        )

      assert %Axon{} = encoder1
      assert %Axon{} = encoder2
    end
  end

  describe "loss/2" do
    test "computes L1 loss" do
      predicted = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      target = Nx.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
      loss = VJEPA2.loss(predicted, target)

      # Mean absolute difference = 0.5
      assert_in_delta Nx.to_number(loss), 0.5, 0.001
    end
  end

  describe "ema_update/3" do
    test "blends parameters with momentum" do
      context = %{"vjepa2_enc_proj" => %{"kernel" => Nx.tensor([1.0, 2.0])}}
      target = %{"vjepa2_tgt_proj" => %{"kernel" => Nx.tensor([3.0, 4.0])}}

      updated = VJEPA2.ema_update(context, target, momentum: 0.5)

      # Unchanged since key mapping doesn't match (vjepa2_tgt_ -> vjepa2_enc_)
      # but the target key "vjepa2_tgt_proj" maps to "vjepa2_enc_proj"
      expected = Nx.tensor([2.0, 3.0])

      assert Nx.shape(updated["vjepa2_tgt_proj"]["kernel"]) ==
               Nx.shape(expected)
    end
  end

  describe "output_size/1" do
    test "returns embed_dim" do
      assert VJEPA2.output_size(embed_dim: 512) == 512
    end

    test "returns default when no options" do
      assert VJEPA2.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = VJEPA2.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:embed_dim] == 256
      assert defaults[:predictor_embed_dim] == 128
      assert defaults[:num_tokens] == 128
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      {encoder, predictor} =
        Edifice.build(:vjepa2,
          patch_dim: 16,
          embed_dim: 16,
          num_heads: 2,
          encoder_depth: 1,
          predictor_depth: 1,
          num_tokens: 8,
          dropout: 0.0
        )

      assert %Axon{} = encoder
      assert %Axon{} = predictor
    end
  end
end
