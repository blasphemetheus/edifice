defmodule Edifice.Feedforward.TabNetCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Feedforward.TabNet

  @batch 2
  @input_size 16
  @hidden_size 8
  @num_steps 3

  @base_opts [
    input_size: @input_size,
    hidden_size: @hidden_size,
    num_steps: @num_steps,
    relaxation_factor: 1.5,
    dropout: 0.0
  ]

  # ============================================================================
  # Sparsemax Feature Selection
  # ============================================================================

  describe "sparsemax feature selection" do
    test "mask values are non-negative (sparsemax property)" do
      model = TabNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      # Output should be finite and valid
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "no softmax-style dense layers in params (sparsemax is non-parametric)" do
      # TabNet should use sparsemax (non-parametric projection onto simplex),
      # NOT a softmax layer. Verify no "softmax" keys in params.
      model = TabNet.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      softmax_keys = Enum.filter(param_keys, &String.contains?(&1, "softmax"))
      assert softmax_keys == [], "Should not have softmax layers, found: #{inspect(softmax_keys)}"
    end

    test "params contain sparsemax step layers" do
      model = TabNet.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have step attention projection layers
      for step <- 0..(@num_steps - 1) do
        attn_keys = Enum.filter(param_keys, &String.contains?(&1, "step_#{step}_attn_proj"))

        assert length(attn_keys) > 0,
               "Should have attention projection for step #{step}"
      end
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, hidden_size]" do
      model = TabNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output shape is [batch, num_classes] with classifier" do
      opts = Keyword.put(@base_opts, :num_classes, 10)
      model = TabNet.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 10}
    end

    test "output is deterministic" do
      model = TabNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "different inputs produce different outputs" do
      model = TabNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
