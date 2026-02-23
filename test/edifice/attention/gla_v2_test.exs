defmodule Edifice.Attention.GLAv2Test do
  use ExUnit.Case, async: true

  alias Edifice.Attention.GLAv2

  @batch 4
  @seq_len 12
  @embed_dim 64
  @hidden_size 32
  @num_heads 4
  @head_dim 8
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    head_dim: @head_dim,
    num_layers: @num_layers,
    conv_kernel_size: 4,
    forget_gate_init: 3.0,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = GLAv2.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = GLAv2.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = GLAv2.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GLAv2.output_size(@opts) == @hidden_size
    end
  end
end
