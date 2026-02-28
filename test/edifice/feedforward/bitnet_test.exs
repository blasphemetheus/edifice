defmodule Edifice.Feedforward.BitNetTest do
  use ExUnit.Case, async: true
  @moduletag :feedforward

  alias Edifice.Feedforward.BitNet

  @batch_size 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 64
  @num_heads 4

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: 2,
    window_size: @seq_len,
    dropout: 0.0,
    quantize: :ternary
  ]

  defp build_and_run(opts) do
    model = BitNet.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())

    input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = BitNet.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape with ternary quantization" do
      {_model, output} = build_and_run(@opts)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output contains finite values with ternary" do
      {_model, output} = build_and_run(@opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "produces correct output shape with binary quantization" do
      opts = Keyword.put(@opts, :quantize, :binary)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output contains finite values with binary" do
      opts = Keyword.put(@opts, :quantize, :binary)
      {_model, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works when embed_dim equals hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: @hidden_size, hidden_size: @hidden_size)

      model = BitNet.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert BitNet.output_size(@opts) == @hidden_size
    end
  end
end
