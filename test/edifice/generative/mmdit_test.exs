defmodule Edifice.Generative.MMDiTTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.MMDiT

  @batch 2
  @img_dim 8
  @txt_dim 16
  @hidden_size 32
  @depth 2
  @num_heads 4
  @img_tokens 16
  @txt_tokens 8

  @opts [
    img_dim: @img_dim,
    txt_dim: @txt_dim,
    hidden_size: @hidden_size,
    depth: @depth,
    num_heads: @num_heads,
    img_tokens: @img_tokens,
    txt_tokens: @txt_tokens
  ]

  defp random_inputs do
    key = Nx.Random.key(42)
    {img, key} = Nx.Random.uniform(key, shape: {@batch, @img_tokens, @img_dim})
    {txt, key} = Nx.Random.uniform(key, shape: {@batch, @txt_tokens, @txt_dim})
    {timestep, key} = Nx.Random.uniform(key, shape: {@batch})
    {pooled, _key} = Nx.Random.uniform(key, shape: {@batch, @hidden_size})

    %{
      "img_latent" => img,
      "txt_embed" => txt,
      "timestep" => timestep,
      "pooled_text" => pooled
    }
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = MMDiT.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MMDiT.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "img_latent" => Nx.template({@batch, @img_tokens, @img_dim}, :f32),
            "txt_embed" => Nx.template({@batch, @txt_tokens, @txt_dim}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "pooled_text" => Nx.template({@batch, @hidden_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_inputs())

      assert Nx.shape(output) == {@batch, @img_tokens, @img_dim}
    end

    test "output contains finite values" do
      model = MMDiT.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "img_latent" => Nx.template({@batch, @img_tokens, @img_dim}, :f32),
            "txt_embed" => Nx.template({@batch, @txt_tokens, @txt_dim}, :f32),
            "timestep" => Nx.template({@batch}, :f32),
            "pooled_text" => Nx.template({@batch, @hidden_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_inputs())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns img_tokens * img_dim" do
      assert MMDiT.output_size(@opts) == @img_tokens * @img_dim
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MMDiT.recommended_defaults()
      assert Keyword.has_key?(defaults, :img_dim)
      assert Keyword.has_key?(defaults, :txt_dim)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :depth)
    end
  end
end
