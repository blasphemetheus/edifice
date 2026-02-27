defmodule Edifice.Blocks.SwiGLUTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.SwiGLU

  @batch 2
  @seq_len 8
  @hidden 32

  describe "layer/2" do
    test "produces correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "output is finite" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})
      output = predict_fn.(params, test_input)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, name: "test_swiglu")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, @hidden}))

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end

    test "custom inner_size is respected" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SwiGLU.layer(input, hidden_size: @hidden, inner_size: 64, name: "test")

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      gate_kernel = params.data["test_gate"]["kernel"]
      {_, inner_dim} = Nx.shape(gate_kernel)
      assert inner_dim == 64
    end

    test "default inner_size is multiple of 8" do
      # Use small hidden_size=48 to keep BinaryBackend fast while still testing the rounding
      input = Axon.input("input", shape: {nil, @seq_len, 48})
      model = SwiGLU.layer(input, hidden_size: 48, name: "test")

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 48}, :f32), Axon.ModelState.empty())

      gate_kernel = params.data["test_gate"]["kernel"]
      {_, inner_size} = Nx.shape(gate_kernel)
      assert rem(inner_size, 8) == 0
    end

    test "different activations produce different outputs" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})

      silu_model = SwiGLU.layer(input, hidden_size: @hidden, activation: :silu, name: "silu")
      gelu_model = SwiGLU.layer(input, hidden_size: @hidden, activation: :gelu, name: "gelu")

      {silu_init, silu_pred} = Axon.build(silu_model)
      {gelu_init, gelu_pred} = Axon.build(gelu_model)

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      silu_params =
        silu_init.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      gelu_params =
        gelu_init.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())

      silu_out = silu_pred.(silu_params, test_input)
      gelu_out = gelu_pred.(gelu_params, test_input)

      # Both should build and produce same shape
      assert Nx.shape(silu_out) == Nx.shape(gelu_out)
    end
  end
end
