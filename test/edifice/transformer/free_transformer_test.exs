defmodule Edifice.Transformer.FreeTransformerTest do
  use ExUnit.Case, async: true

  alias Edifice.Transformer.FreeTransformer

  import Edifice.TestHelpers

  @moduletag :transformer

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  describe "build/1" do
    test "builds model with default options" do
      model = FreeTransformer.build(embed_dim: @embed_dim)
      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        FreeTransformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 2,
          num_layers: 4,
          num_latent_bits: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, @embed_dim}, 42)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "batch=1 forward pass" do
      model =
        FreeTransformer.build(
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 4,
          num_latent_bits: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({1, 4, 16}, 99)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 16}
    end

    test "output values are finite" do
      model =
        FreeTransformer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 2,
          num_layers: 4,
          num_latent_bits: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, @embed_dim}, 77)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end
  end

  describe "kl_loss/2" do
    test "computes KL divergence" do
      logits = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      loss = FreeTransformer.kl_loss(logits)

      # At logits=0, p=0.5, KL(Bernoulli(0.5)||Uniform) = 0
      assert Nx.to_number(loss) >= 0.0
    end

    test "KL increases with confident predictions" do
      uniform_logits = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      confident_logits = Nx.tensor([[5.0, 5.0, 5.0, 5.0]])

      kl_uniform = FreeTransformer.kl_loss(uniform_logits, 0.0)
      kl_confident = FreeTransformer.kl_loss(confident_logits, 0.0)

      assert Nx.to_number(kl_confident) > Nx.to_number(kl_uniform)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert FreeTransformer.output_size(hidden_size: 512) == 512
    end

    test "returns default" do
      assert FreeTransformer.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = FreeTransformer.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:num_latent_bits] == 8
      assert defaults[:num_layers] == 6
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model =
        Edifice.build(:free_transformer,
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 4,
          num_latent_bits: 4,
          dropout: 0.0
        )

      assert %Axon{} = model
    end
  end
end
