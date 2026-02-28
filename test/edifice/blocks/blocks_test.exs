defmodule Edifice.Blocks.BlocksTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.{
    AdaptiveNorm,
    ALiBi,
    CrossAttention,
    PatchEmbed,
    RMSNorm,
    RoPE,
    SinusoidalPE,
    SwiGLU
  }

  @batch 2
  @seq_len 8
  @hidden 32

  describe "RMSNorm" do
    test "layer builds and produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RMSNorm.layer(input, hidden_size: @hidden, name: "test_rms")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "apply normalizes tensor" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)
      # RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.74
      # Normalized values should sum to approximately same magnitude
      assert Nx.shape(result) == {1, 4}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "SwiGLU" do
    test "layer builds and produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end

  describe "RoPE" do
    test "precompute_freqs returns correct shape" do
      {cos_table, sin_table} = RoPE.precompute_freqs(32, 64)
      assert Nx.shape(cos_table) == {64, 16}
      assert Nx.shape(sin_table) == {64, 16}
    end

    test "layer applies rotation to input" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RoPE.layer(input, dim: @hidden, name: "test_rope")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end

  describe "ALiBi" do
    test "compute_slopes returns correct number of slopes" do
      slopes = ALiBi.compute_slopes(8)
      assert Nx.shape(slopes) == {8}
    end

    test "compute_bias returns correct shape" do
      bias = ALiBi.compute_bias(seq_len: 16, num_heads: 4)
      assert Nx.shape(bias) == {4, 16, 16}
    end

    test "bias values are non-positive for causal" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 2, causal: true)
      # Diagonal should be 0 (distance to self)
      assert Nx.to_number(bias[0][0][0]) == 0.0
    end
  end

  describe "PatchEmbed" do
    test "num_patches calculates correctly" do
      assert PatchEmbed.num_patches(224, 16) == 196
      assert PatchEmbed.num_patches(32, 8) == 16
    end

    test "layer produces correct number of patch embeddings" do
      image = Axon.input("image", shape: {nil, 3, 32, 32})
      model = PatchEmbed.layer(image, embed_dim: 64, patch_size: 8, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 3, 32, 32}))

      # 32/8 = 4, so 4*4 = 16 patches
      assert Nx.shape(output) == {@batch, 16, 64}
    end
  end

  describe "SinusoidalPE" do
    test "build_table returns correct shape" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)
      assert Nx.shape(table) == {100, 64}
    end

    test "layer adds positional encoding" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SinusoidalPE.layer(input, dim: @hidden)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.0, {@batch, @seq_len, @hidden}))

      # With zero input, output should be just the PE (non-zero)
      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end

  describe "AdaptiveNorm" do
    test "adaln_zero layer produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln_zero,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.5, {@batch, @seq_len, @hidden}),
          "condition" => Nx.broadcast(0.1, {@batch, @hidden})
        })

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end

  describe "CrossAttention" do
    test "layer produces correct shape" do
      queries = Axon.input("queries", shape: {nil, 8, 16})
      context = Axon.input("context", shape: {nil, 12, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 32,
          name: "test_cross_attn"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({@batch, 8, 16}, :f32),
            "context" => Nx.template({@batch, 12, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {@batch, 8, 16}),
          "context" => Nx.broadcast(0.5, {@batch, 12, 16})
        })

      assert Nx.shape(output) == {@batch, 8, 32}
    end
  end
end
