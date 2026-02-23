defmodule Edifice.Interpretability.TranscoderTest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.Transcoder

  @batch 4
  @input_size 32
  @output_size 48
  @dict_size 64
  @top_k 8

  @opts [
    input_size: @input_size,
    output_size: @output_size,
    dict_size: @dict_size,
    top_k: @top_k
  ]

  defp template, do: %{"transcoder_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = Transcoder.build(@opts)
      assert %Axon{} = model
    end

    test "maps input_size to output_size" do
      model = Transcoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"transcoder_input" => random_input()})
      assert Nx.shape(output) == {@batch, @output_size}
    end

    test "output contains finite values" do
      model = Transcoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"transcoder_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "loss/4" do
    test "returns scalar loss" do
      target = Nx.broadcast(1.0, {@batch, @output_size})
      reconstruction = Nx.broadcast(0.9, {@batch, @output_size})
      hidden_acts = Nx.broadcast(0.5, {@batch, @dict_size})

      loss = Transcoder.loss(target, reconstruction, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "output_size/1" do
    test "returns output_size" do
      assert Transcoder.output_size(@opts) == @output_size
    end
  end
end
