defmodule Edifice.Vision.JanusTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.Janus

  @tiny_opts [
    hidden_size: 32,
    patch_dim: 16,
    num_patches: 4,
    vit_layers: 1,
    vit_heads: 2,
    vit_ffn_mult: 2,
    aligner_depth: 2,
    codebook_size: 64,
    gen_embed_dim: 4,
    gen_head_intermediate: 16
  ]

  describe "build/1" do
    test "understanding encoder produces correct shape" do
      {encoder, _gen_head} = Janus.build(@tiny_opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), %{})
      input = Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(100)
      out = predict_fn.(params, input)

      # [batch=2, num_patches=4, hidden_size=32]
      assert Nx.shape(out) == {2, 4, 32}
    end

    test "generation head produces correct shape" do
      {_encoder, gen_head} = Janus.build(@tiny_opts)

      {init_fn, predict_fn} = Axon.build(gen_head)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.iota({2, 8, 32}, type: :f32) |> Nx.divide(100)
      out = predict_fn.(params, input)

      # [batch=2, seq=8, codebook_size=64]
      assert Nx.shape(out) == {2, 8, 64}
    end

    test "outputs are finite" do
      {encoder, gen_head} = Janus.build(@tiny_opts)

      # Understanding encoder
      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), %{})
      input = Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(100)
      enc_out = predict_fn.(params, input)

      assert enc_out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert enc_out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0

      # Generation head
      {init_fn2, predict_fn2} = Axon.build(gen_head)
      params2 = init_fn2.(Nx.template({2, 8, 32}, :f32), %{})
      gen_input = Nx.iota({2, 8, 32}, type: :f32) |> Nx.divide(100)
      gen_out = predict_fn2.(params2, gen_input)

      assert gen_out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert gen_out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      {encoder, gen_head} = Janus.build(@tiny_opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({1, 4, 16}, :f32), %{})
      out = predict_fn.(params, Nx.iota({1, 4, 16}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {1, 4, 32}

      {init_fn2, predict_fn2} = Axon.build(gen_head)
      params2 = init_fn2.(Nx.template({1, 6, 32}, :f32), %{})
      out2 = predict_fn2.(params2, Nx.iota({1, 6, 32}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out2) == {1, 6, 64}
    end

    test "different hidden size" do
      opts = Keyword.put(@tiny_opts, :hidden_size, 48)
      {encoder, gen_head} = Janus.build(opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 4, 48}

      {init_fn2, predict_fn2} = Axon.build(gen_head)
      params2 = init_fn2.(Nx.template({2, 8, 48}, :f32), %{})
      out2 = predict_fn2.(params2, Nx.iota({2, 8, 48}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out2) == {2, 8, 64}
    end

    test "different codebook size" do
      opts = Keyword.put(@tiny_opts, :codebook_size, 128)
      {_encoder, gen_head} = Janus.build(opts)

      {init_fn, predict_fn} = Axon.build(gen_head)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 8, 32}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 8, 128}
    end

    test "deeper ViT encoder" do
      opts = Keyword.put(@tiny_opts, :vit_layers, 3)
      {encoder, _gen_head} = Janus.build(opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 4, 32}
    end

    test "single aligner layer" do
      opts = Keyword.put(@tiny_opts, :aligner_depth, 1)
      {encoder, _gen_head} = Janus.build(opts)

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 4, 16}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 4, 16}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 4, 32}
    end
  end

  describe "output_size/1" do
    test "returns codebook_size" do
      assert Janus.output_size(codebook_size: 256) == 256
    end

    test "returns default codebook_size" do
      assert Janus.output_size() == 16_384
    end
  end
end
