defmodule Edifice.Meta.QATTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.QAT

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_layers 2

  defp base_opts(quantize) do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_heads: 4,
      num_layers: @num_layers,
      quantize: quantize,
      seq_len: @seq_len,
      dropout: 0.0
    ]
  end

  describe "build/1 with :binary" do
    test "produces correct output shape" do
      model = QAT.build(base_opts(:binary))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @seq_len, @embed_dim}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim}))
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with :ternary" do
    test "produces correct output shape and finite values" do
      model = QAT.build(base_opts(:ternary))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @seq_len, @embed_dim}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim}))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with :int4" do
    test "produces correct output shape and finite values" do
      model = QAT.build(base_opts(:int4))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @seq_len, @embed_dim}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim}))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with :int8" do
    test "produces correct output shape and finite values" do
      model = QAT.build(base_opts(:int8))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          Nx.template({@batch, @seq_len, @embed_dim}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim}))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert QAT.output_size(hidden_size: 128) == 128
    end
  end
end
