defmodule Edifice.Blocks.FFNTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.FFN

  @batch 2
  @seq_len 8
  @hidden 32

  describe "layer/2 (standard FFN)" do
    @tag :smoke
    test "output dimension equals input dimension" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "default expansion_factor=4 creates correct inner size" do
      hidden = 64
      input = Axon.input("input", shape: {nil, @seq_len, hidden})
      model = FFN.layer(input, hidden_size: hidden, name: "test_ffn")

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, hidden}, :f32), Axon.ModelState.empty())

      up_kernel = params.data["test_ffn_up"]["kernel"]
      {in_dim, inner_dim} = Nx.shape(up_kernel)
      assert in_dim == hidden
      assert inner_dim == hidden * 4
    end

    test "custom expansion_factor" do
      hidden = 32
      input = Axon.input("input", shape: {nil, @seq_len, hidden})
      model = FFN.layer(input, hidden_size: hidden, expansion_factor: 2, name: "test_ffn")

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, hidden}, :f32), Axon.ModelState.empty())

      up_kernel = params.data["test_ffn_up"]["kernel"]
      {_, inner_dim} = Nx.shape(up_kernel)
      assert inner_dim == hidden * 2
    end

    test "explicit inner_size overrides expansion_factor" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      model =
        FFN.layer(input,
          hidden_size: @hidden,
          inner_size: 100,
          expansion_factor: 8,
          name: "test_ffn"
        )

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      up_kernel = params.data["test_ffn_up"]["kernel"]
      {_, inner_dim} = Nx.shape(up_kernel)
      assert inner_dim == 100
    end

    test "deterministic at dropout=0" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, dropout: 0.0, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      out1 = predict_fn.(params, test_input)
      out2 = predict_fn.(params, test_input)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(out1, out2))))
      assert diff < 1.0e-6
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, @hidden}))

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end

    test "output is finite" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.layer(input, hidden_size: @hidden, name: "test_ffn")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      output = predict_fn.(params, test_input)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "gated_layer/2" do
    test "delegates to SwiGLU with correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = FFN.gated_layer(input, hidden_size: @hidden, name: "gated")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end
  end
end
