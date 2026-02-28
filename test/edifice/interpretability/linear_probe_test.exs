defmodule Edifice.Interpretability.LinearProbeTest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.LinearProbe

  @batch 4
  @input_size 32
  @num_classes 5

  defp template, do: %{"probe_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1 with classification" do
    @opts [input_size: @input_size, num_classes: @num_classes, task: :classification]

    test "builds an Axon model" do
      model = LinearProbe.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = LinearProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"probe_input" => random_input()})
      assert Nx.shape(output) == {@batch, @num_classes}
    end

    test "output sums to 1 per sample (softmax)" do
      model = LinearProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"probe_input" => random_input()})

      sums = Nx.sum(output, axes: [1])

      for i <- 0..(@batch - 1) do
        assert_in_delta Nx.to_number(sums[i]), 1.0, 1.0e-5
      end
    end
  end

  describe "build/1 with binary" do
    @opts [input_size: @input_size, num_classes: 1, task: :binary]

    test "produces sigmoid output in [0, 1]" do
      model = LinearProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"probe_input" => random_input()})

      assert Nx.shape(output) == {@batch, 1}
      assert Nx.all(Nx.greater_equal(output, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(output, 1.0)) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with regression" do
    @opts [input_size: @input_size, num_classes: 1, task: :regression]

    test "produces unbounded linear output" do
      model = LinearProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"probe_input" => random_input()})

      assert Nx.shape(output) == {@batch, 1}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert LinearProbe.output_size(num_classes: 10) == 10
    end

    test "defaults to 2" do
      assert LinearProbe.output_size([]) == 2
    end
  end

  describe "Edifice.build/2" do
    test "builds linear_probe via registry" do
      model = Edifice.build(:linear_probe, input_size: @input_size, num_classes: @num_classes)
      assert %Axon{} = model
    end
  end
end
