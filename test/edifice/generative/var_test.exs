defmodule Edifice.Generative.VARTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.VAR

  describe "build/1" do
    test "builds VAR model with default options" do
      model =
        VAR.build(
          hidden_size: 64,
          num_layers: 2,
          num_heads: 2,
          scales: [1, 2, 4],
          codebook_size: 32
        )

      assert %Axon{} = model
    end

    test "builds model with custom scales" do
      model =
        VAR.build(
          hidden_size: 32,
          num_layers: 1,
          num_heads: 2,
          scales: [1, 2],
          codebook_size: 16
        )

      assert %Axon{} = model
    end

    test "model produces output with correct shape" do
      scales = [1, 2]
      # 1x1 + 2x2
      total_tokens = 1 + 4
      hidden_size = 32
      codebook_size = 16

      model =
        VAR.build(
          hidden_size: hidden_size,
          num_layers: 1,
          num_heads: 2,
          scales: scales,
          codebook_size: codebook_size,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, total_tokens, hidden_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.1, {2, total_tokens, hidden_size})
      output = predict_fn.(params, %{"scale_embeddings" => input})

      # Should have prediction head for scale 1 (predicting scale 2 = 2x2 = 4 tokens)
      assert Map.has_key?(output, "scale_1_logits")
      scale_1_logits = output["scale_1_logits"]
      assert {2, 4, 16} == Nx.shape(scale_1_logits)
    end
  end

  describe "build_tokenizer/1" do
    test "builds encoder and decoder" do
      {encoder, decoder} =
        VAR.build_tokenizer(
          image_size: 16,
          scales: [1, 2, 4],
          codebook_size: 32,
          embed_dim: 16,
          in_channels: 3
        )

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end

    test "encoder produces scale tokens" do
      scales = [1, 2]

      {encoder, _decoder} =
        VAR.build_tokenizer(
          image_size: 8,
          scales: scales,
          codebook_size: 16,
          embed_dim: 8,
          in_channels: 3
        )

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 8, 8, 3}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {2, 8, 8, 3})
      output = predict_fn.(params, %{"image" => input})

      assert Map.has_key?(output, "scale_0")
      assert Map.has_key?(output, "scale_1")
    end
  end

  describe "total_tokens/1" do
    test "computes correct total for default scales" do
      # [1, 2, 4, 8, 16] -> 1 + 4 + 16 + 64 + 256 = 341
      assert VAR.total_tokens([1, 2, 4, 8, 16]) == 341
    end

    test "computes correct total for custom scales" do
      # 1 + 4
      assert VAR.total_tokens([1, 2]) == 5
      # 1 + 4 + 16
      assert VAR.total_tokens([1, 2, 4]) == 21
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = VAR.recommended_defaults()

      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :scales)
      assert Keyword.has_key?(defaults, :codebook_size)
    end
  end
end
