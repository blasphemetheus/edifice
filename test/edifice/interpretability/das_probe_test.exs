defmodule Edifice.Interpretability.DASProbeTest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.DASProbe

  @batch 4
  @input_size 32
  @subspace_dim 8
  @num_classes 5

  @opts [
    input_size: @input_size,
    subspace_dim: @subspace_dim,
    num_classes: @num_classes
  ]

  defp template, do: %{"das_probe_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1 with classification" do
    test "builds an Axon model" do
      model = DASProbe.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = DASProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"das_probe_input" => random_input()})
      assert Nx.shape(output) == {@batch, @num_classes}
    end

    test "output sums to 1 per sample (softmax)" do
      model = DASProbe.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"das_probe_input" => random_input()})

      sums = Nx.sum(output, axes: [1])

      for i <- 0..(@batch - 1) do
        assert_in_delta Nx.to_number(sums[i]), 1.0, 1.0e-5
      end
    end
  end

  describe "build/1 with binary" do
    test "produces sigmoid output" do
      model = DASProbe.build(Keyword.merge(@opts, task: :binary, num_classes: 1))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"das_probe_input" => random_input()})

      assert Nx.shape(output) == {@batch, 1}
      assert Nx.all(Nx.greater_equal(output, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(output, 1.0)) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with regression" do
    test "produces unbounded linear output" do
      model = DASProbe.build(Keyword.merge(@opts, task: :regression, num_classes: 1))
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"das_probe_input" => random_input()})

      assert Nx.shape(output) == {@batch, 1}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_projection/1" do
    test "produces subspace-projected activations" do
      proj = DASProbe.build_projection(@opts)
      {init_fn, predict_fn} = Axon.build(proj, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      projected = predict_fn.(params, %{"das_probe_input" => random_input()})

      assert Nx.shape(projected) == {@batch, @subspace_dim}
    end

    test "projection has no bias" do
      proj = DASProbe.build_projection(@opts)
      {init_fn, _predict_fn} = Axon.build(proj, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      data = params.data

      assert Map.has_key?(data, "das_projection")
      refute Map.has_key?(data["das_projection"], "bias")
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert DASProbe.output_size(@opts) == @num_classes
    end
  end

  describe "Edifice.build/2" do
    test "builds das_probe via registry" do
      model = Edifice.build(:das_probe, @opts)
      assert %Axon{} = model
    end
  end
end
