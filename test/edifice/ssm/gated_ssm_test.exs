defmodule Edifice.SSM.GatedSSMTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.GatedSSM

  @batch 4
  @seq_len 12
  @embed_size 64
  @hidden_size 32
  @state_size 8
  @num_layers 2

  @opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    state_size: @state_size,
    num_layers: @num_layers,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = GatedSSM.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = GatedSSM.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = GatedSSM.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GatedSSM.output_size(@opts) == @hidden_size
    end
  end
end
