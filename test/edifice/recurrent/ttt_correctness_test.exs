defmodule Edifice.Recurrent.TTTCorrectnessTest do
  @moduledoc """
  Correctness tests for TTT paper-faithful implementation.
  Verifies W_0 init, eta/d scaling, inner LayerNorm, output gating,
  and numerical stability.
  """
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.TTT

  @batch 2
  @embed_dim 32
  @hidden_size 32
  @inner_size 16
  @seq_len 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    inner_size: @inner_size,
    num_layers: 1,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp build_and_init(opts \\ @base_opts) do
    model = TTT.build(opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    params =
      init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

    {model, params, predict_fn}
  end

  defp get_recurrence_params(params) do
    recurrence_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "recurrence"))
    {recurrence_key, params.data[recurrence_key]}
  end

  # ============================================================================
  # Fix 1: W_0 ~ N(0, 0.02)
  # ============================================================================

  describe "W_0 initialization" do
    test "params contain w0 key (nested under recurrence layer)" do
      {_model, params, _predict_fn} = build_and_init()
      {recurrence_key, recurrence_params} = get_recurrence_params(params)

      assert recurrence_key != nil,
             "Should have a 'recurrence' param group, got: #{inspect(Map.keys(params.data))}"

      assert is_map(recurrence_params) and not is_struct(recurrence_params),
             "Recurrence param group should be a map with sub-params"

      w0_keys = Map.keys(recurrence_params) |> Enum.filter(&String.contains?(&1, "w0"))

      assert w0_keys != [],
             "Recurrence params should contain 'w0', got: #{inspect(Map.keys(recurrence_params))}"
    end

    test "w0 has shape [inner_size, inner_size]" do
      {_model, params, _predict_fn} = build_and_init()
      {_key, recurrence_params} = get_recurrence_params(params)
      w0_key = Enum.find(Map.keys(recurrence_params), &String.contains?(&1, "w0"))
      w0_value = recurrence_params[w0_key]

      assert Nx.shape(w0_value) == {@inner_size, @inner_size}
    end

    test "w0 values are small (N(0, 0.02) init, not glorot)" do
      {_model, params, _predict_fn} = build_and_init()
      {_key, recurrence_params} = get_recurrence_params(params)
      w0_key = Enum.find(Map.keys(recurrence_params), &String.contains?(&1, "w0"))
      w0_value = recurrence_params[w0_key]

      # N(0, 0.02) should have max |value| << 0.5 almost always
      # glorot_uniform for 16x16 would have values up to ~0.5
      max_abs = Nx.abs(w0_value) |> Nx.reduce_max() |> Nx.to_number()
      stddev = Nx.standard_deviation(w0_value) |> Nx.to_number()

      assert max_abs < 0.2,
             "W_0 max |value| should be small (N(0, 0.02)), got #{max_abs}"

      assert stddev < 0.05,
             "W_0 stddev should be near 0.02, got #{stddev}"
    end
  end

  # ============================================================================
  # Fix 3: Inner LayerNorm params
  # ============================================================================

  describe "inner LayerNorm parameters" do
    test "recurrence contains ln_gamma and ln_beta" do
      {_model, params, _predict_fn} = build_and_init()
      {_key, recurrence_params} = get_recurrence_params(params)
      param_keys = Map.keys(recurrence_params)

      gamma_keys = Enum.filter(param_keys, &String.contains?(&1, "inner_ln_gamma"))
      beta_keys = Enum.filter(param_keys, &String.contains?(&1, "inner_ln_beta"))

      assert gamma_keys != [],
             "Should have inner_ln_gamma param, got: #{inspect(param_keys)}"

      assert beta_keys != [],
             "Should have inner_ln_beta param, got: #{inspect(param_keys)}"
    end

    test "ln_gamma initialized to ones, ln_beta to zeros" do
      {_model, params, _predict_fn} = build_and_init()
      {_key, recurrence_params} = get_recurrence_params(params)

      gamma_key = Enum.find(Map.keys(recurrence_params), &String.contains?(&1, "inner_ln_gamma"))
      beta_key = Enum.find(Map.keys(recurrence_params), &String.contains?(&1, "inner_ln_beta"))

      gamma = recurrence_params[gamma_key]
      beta = recurrence_params[beta_key]

      assert Nx.shape(gamma) == {@inner_size}
      assert Nx.shape(beta) == {@inner_size}

      # gamma should be all ones
      gamma_diff =
        Nx.subtract(gamma, Nx.broadcast(1.0, {@inner_size}))
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert gamma_diff < 1.0e-6, "ln_gamma should be initialized to ones"

      # beta should be all zeros
      beta_max = Nx.abs(beta) |> Nx.reduce_max() |> Nx.to_number()
      assert beta_max < 1.0e-6, "ln_beta should be initialized to zeros"
    end
  end

  # ============================================================================
  # Fix 4: Output gating
  # ============================================================================

  describe "output gating" do
    test "gate projection exists when output_gate: true (default)" do
      {_model, params, _predict_fn} = build_and_init()
      param_keys = Map.keys(params.data)

      gate_keys = Enum.filter(param_keys, &String.contains?(&1, "gate_proj"))

      assert gate_keys != [],
             "Should have gate_proj params when output_gate: true, got: #{inspect(param_keys)}"
    end

    test "no gate projection when output_gate: false" do
      opts = Keyword.put(@base_opts, :output_gate, false)
      {_model, params, _predict_fn} = build_and_init(opts)
      param_keys = Map.keys(params.data)

      gate_keys = Enum.filter(param_keys, &String.contains?(&1, "gate_proj"))

      assert gate_keys == [],
             "Should NOT have gate_proj params when output_gate: false, got: #{inspect(gate_keys)}"
    end

    test "gated and ungated produce different outputs" do
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      {_model1, params1, predict_fn1} = build_and_init(@base_opts)
      output_gated = predict_fn1.(params1, input)

      opts_no_gate = Keyword.put(@base_opts, :output_gate, false)
      {_model2, params2, predict_fn2} = build_and_init(opts_no_gate)
      output_ungated = predict_fn2.(params2, input)

      # Different models, different params, should produce different outputs
      diff =
        Nx.subtract(output_gated, output_ungated) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff > 1.0e-6, "Gated and ungated should differ"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, hidden_size]" do
      {_model, params, predict_fn} = build_and_init()
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is deterministic in inference mode" do
      {_model, params, predict_fn} = build_and_init()
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "output is finite (no NaN/Inf)" do
      {_model, params, predict_fn} = build_and_init()
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is finite with larger input magnitudes" do
      # Stress test: larger inputs that would cause NaN with old init/scaling
      {_model, params, predict_fn} = build_and_init()
      key = Nx.Random.key(99)
      {input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @embed_dim})
      # Scale up to simulate real embedding magnitudes
      input = Nx.multiply(input, 5.0)
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1,
             "Output should be finite even with large inputs (eta/d scaling prevents explosion)"

      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different inputs produce different outputs" do
      {_model, params, predict_fn} = build_and_init()
      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end

  # ============================================================================
  # Variant support
  # ============================================================================

  describe "MLP variant" do
    test "MLP variant builds and produces finite output" do
      opts = Keyword.put(@base_opts, :variant, :mlp)
      {_model, params, predict_fn} = build_and_init(opts)

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
