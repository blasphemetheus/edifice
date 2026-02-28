defmodule Edifice.Attention.MTATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Attention.MTA

  @moduletag timeout: 120_000

  @batch 2
  @seq_len 4
  @embed_dim 16
  @hidden_size 16
  @num_heads 2

  @small_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: 2,
    c_q: 2,
    c_k: 3,
    c_h: 2,
    kq_conv_every: 1,
    head_conv_every: 1,
    dropout: 0.0,
    window_size: @seq_len
  ]

  describe "build/1" do
    test "returns an Axon model" do
      model = MTA.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MTA.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output values are finite" do
      model = MTA.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert_finite!(output)
    end

    test "works with multiple layers" do
      opts = Keyword.put(@small_opts, :num_layers, 3)
      model = MTA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "without KQ conv" do
    test "works when kq_conv_every exceeds num_layers" do
      opts = Keyword.merge(@small_opts, kq_conv_every: 100, num_layers: 2)
      model = MTA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "without head mixing" do
    test "works when head_conv_every exceeds num_layers" do
      opts = Keyword.merge(@small_opts, head_conv_every: 100, num_layers: 2)
      model = MTA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "conv every layer" do
    test "works with kq_conv_every: 1" do
      opts = Keyword.merge(@small_opts, kq_conv_every: 1, num_layers: 2)
      model = MTA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({@batch, @seq_len, @embed_dim})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MTA.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert MTA.output_size([]) == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MTA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :c_q)
      assert Keyword.has_key?(defaults, :c_k)
      assert Keyword.has_key?(defaults, :c_h)
    end
  end
end
