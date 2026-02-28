defmodule Edifice.Interpretability.ConceptBottleneckTest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.ConceptBottleneck

  @batch 4
  @input_size 32
  @num_concepts 8
  @num_classes 5

  @opts [
    input_size: @input_size,
    num_concepts: @num_concepts,
    num_classes: @num_classes
  ]

  defp template, do: %{"cbm_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = ConceptBottleneck.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = ConceptBottleneck.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"cbm_input" => random_input()})
      assert Nx.shape(output) == {@batch, @num_classes}
    end

    test "output is valid probability distribution (softmax)" do
      model = ConceptBottleneck.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"cbm_input" => random_input()})

      sums = Nx.sum(output, axes: [1])

      for i <- 0..(@batch - 1) do
        assert_in_delta Nx.to_number(sums[i]), 1.0, 1.0e-5
      end
    end
  end

  describe "build_concept_predictor/1" do
    test "produces concept activations in [0, 1]" do
      predictor = ConceptBottleneck.build_concept_predictor(@opts)
      {init_fn, predict_fn} = Axon.build(predictor, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      concepts = predict_fn.(params, %{"cbm_input" => random_input()})

      assert Nx.shape(concepts) == {@batch, @num_concepts}
      assert Nx.all(Nx.greater_equal(concepts, 0.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(concepts, 1.0)) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert ConceptBottleneck.output_size(@opts) == @num_classes
    end
  end

  describe "Edifice.build/2" do
    test "builds concept_bottleneck via registry" do
      model = Edifice.build(:concept_bottleneck, @opts)
      assert %Axon{} = model
    end
  end
end
