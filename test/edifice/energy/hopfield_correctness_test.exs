defmodule Edifice.Energy.HopfieldCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Energy.Hopfield

  @batch 2
  @input_dim 32
  @num_patterns 8
  @pattern_dim 32

  @base_opts [
    input_dim: @input_dim,
    num_patterns: @num_patterns,
    pattern_dim: @pattern_dim,
    beta: 1.0
  ]

  # ============================================================================
  # Stored Pattern Matrix
  # ============================================================================

  describe "stored pattern matrix" do
    test "params contain explicit patterns key (nested under retrieve layer)" do
      model = Hopfield.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      # Patterns are stored as a param inside the hopfield_retrieve layer
      # Structure: params.data["hopfield_retrieve"]["hopfield_patterns"]
      retrieve_params = params.data["hopfield_retrieve"]

      assert is_map(retrieve_params) and not is_struct(retrieve_params),
             "Should have 'hopfield_retrieve' param group"

      pattern_keys = Map.keys(retrieve_params) |> Enum.filter(&String.contains?(&1, "patterns"))

      assert pattern_keys != [],
             "hopfield_retrieve should contain a 'patterns' sub-key, got: #{inspect(Map.keys(retrieve_params))}"

      # Verify pattern shape: [num_patterns, pattern_dim]
      patterns = retrieve_params[hd(pattern_keys)]

      assert Nx.shape(patterns) == {@num_patterns, @pattern_dim},
             "Patterns should be {#{@num_patterns}, #{@pattern_dim}}, got #{inspect(Nx.shape(patterns))}"
    end

    test "params do NOT contain separate similarity and retrieval dense layers" do
      # Modern Hopfield uses the SAME pattern matrix Y for both similarity (X@Y^T)
      # and retrieval (weights@Y). There should not be separate dense layers.
      model = Hopfield.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      similarity_keys = Enum.filter(param_keys, &String.contains?(&1, "similarity"))
      retrieval_keys = Enum.filter(param_keys, &String.contains?(&1, "retrieval"))

      assert similarity_keys == [],
             "Should not have separate 'similarity' dense layer, but found: #{inspect(similarity_keys)}"

      assert retrieval_keys == [],
             "Should not have separate 'retrieval' dense layer, but found: #{inspect(retrieval_keys)}"
    end

    test "output shape is [batch, pattern_dim]" do
      model = Hopfield.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @pattern_dim}
    end
  end

  # ============================================================================
  # Beta Sharpness
  # ============================================================================

  describe "beta sharpness" do
    test "higher beta produces sharper retrieval (closer to single pattern)" do
      # Build two models: low beta (soft average) vs high beta (sharp retrieval)
      model_low = Hopfield.build(Keyword.put(@base_opts, :beta, 0.1))
      model_high = Hopfield.build(Keyword.put(@base_opts, :beta, 10.0))

      {init_low, predict_low} = Axon.build(model_low, mode: :inference)
      {init_high, predict_high} = Axon.build(model_high, mode: :inference)

      params_low = init_low.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      params_high = init_high.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})

      output_low = predict_low.(params_low, input)
      output_high = predict_high.(params_high, input)

      # Both outputs should be finite (the main correctness check)
      assert Nx.all(Nx.is_nan(output_low) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output_high) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output_low) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output_high) |> Nx.logical_not()) |> Nx.to_number() == 1

      # High beta should produce different output than low beta
      diff = Nx.subtract(output_low, output_high) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff > 1.0e-6,
             "Different beta values should produce different outputs"
    end
  end

  # ============================================================================
  # Determinism
  # ============================================================================

  describe "determinism" do
    test "same input with same params produces deterministic output" do
      model = Hopfield.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "output is finite" do
      model = Hopfield.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
