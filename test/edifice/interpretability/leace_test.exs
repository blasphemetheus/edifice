defmodule Edifice.Interpretability.LEACETest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.LEACE

  @batch 4
  @input_size 32
  @concept_dim 2

  @opts [input_size: @input_size, concept_dim: @concept_dim]

  defp template, do: %{"leace_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = LEACE.build(@opts)
      assert %Axon{} = model
    end

    test "produces same shape as input (erasure preserves dimension)" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "erased output differs from input (projection is non-trivial)" do
      model = LEACE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      input = random_input()
      output = predict_fn.(params, %{"leace_input" => input})

      # Output should differ from input (concept component was subtracted)
      diff = Nx.mean(Nx.abs(Nx.subtract(output, input))) |> Nx.to_number()
      # With random initialization, the projection is non-zero
      assert diff > 0.0
    end

    test "uses bias-free projection layers" do
      model = LEACE.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      data = params.data

      assert Map.has_key?(data, "leace_concept_proj")
      refute Map.has_key?(data["leace_concept_proj"], "bias")
      assert Map.has_key?(data, "leace_concept_recon")
      refute Map.has_key?(data["leace_concept_recon"], "bias")
    end
  end

  describe "build/1 with concept_dim=1" do
    test "erases a single direction" do
      model = LEACE.build(input_size: @input_size, concept_dim: 1)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"leace_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert LEACE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds leace via registry" do
      model = Edifice.build(:leace, @opts)
      assert %Axon{} = model
    end
  end
end
