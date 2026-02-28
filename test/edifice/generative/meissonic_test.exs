defmodule Edifice.Generative.MeissonicTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.Meissonic

  @batch 2
  @num_image_tokens 16
  @codebook_size 64
  @hidden_size 32
  @text_dim 48
  @text_len 4
  @num_heads 4
  @cond_dim 16

  @small_opts [
    codebook_size: @codebook_size,
    num_image_tokens: @num_image_tokens,
    hidden_size: @hidden_size,
    text_dim: @text_dim,
    num_mm_layers: 2,
    num_sm_layers: 4,
    num_heads: @num_heads,
    mlp_ratio: 2.0,
    cond_dim: @cond_dim
  ]

  defp random_input do
    key = Nx.Random.key(42)

    {tokens, key} =
      Nx.Random.randint(key, 0, @codebook_size, shape: {@batch, @num_image_tokens}, type: :s64)

    {text, key} = Nx.Random.normal(key, shape: {@batch, @text_len, @text_dim})
    {pooled, key} = Nx.Random.normal(key, shape: {@batch, @text_dim})
    {conds, _key} = Nx.Random.normal(key, shape: {@batch, @cond_dim})

    %{
      "image_tokens" => tokens,
      "text_hidden" => text,
      "pooled_text" => pooled,
      "micro_conds" => conds
    }
  end

  defp build_and_run(opts) do
    model = Meissonic.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "image_tokens" => Nx.template({@batch, @num_image_tokens}, :s64),
      "text_hidden" => Nx.template({@batch, @text_len, @text_dim}, :f32),
      "pooled_text" => Nx.template({@batch, @text_dim}, :f32),
      "micro_conds" => Nx.template({@batch, @cond_dim}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_input())
    {model, output}
  end

  describe "Meissonic.build/1" do
    test "returns an Axon model" do
      model = Meissonic.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@small_opts)
      assert Nx.shape(output) == {@batch, @num_image_tokens, @codebook_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@small_opts)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "configuration variants" do
    test "different mm/sm layer ratio" do
      opts = Keyword.merge(@small_opts, num_mm_layers: 1, num_sm_layers: 2)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @num_image_tokens, @codebook_size}
    end

    test "larger codebook" do
      opts = Keyword.put(@small_opts, :codebook_size, 128)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @num_image_tokens, 128}
    end
  end

  describe "output_size/1" do
    test "returns codebook_size" do
      assert Meissonic.output_size(codebook_size: 8192) == 8192
      assert Meissonic.output_size(codebook_size: 16384) == 16384
    end
  end
end
