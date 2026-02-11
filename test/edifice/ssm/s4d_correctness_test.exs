defmodule Edifice.SSM.S4DCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.S4D

  @batch 2
  @embed_size 32
  @hidden_size 32
  @state_size 8
  @seq_len 4

  @base_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    state_size: @state_size,
    num_layers: 1,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Learnable SSM Parameters
  # ============================================================================

  describe "learnable SSM parameters" do
    test "params contain a_log and dt_log keys (nested under SSM layer)" do
      model = S4D.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # S4D params are nested: params.data["s4d_block_1_ssm"] contains
      # "s4d_block_1_a_log" and "s4d_block_1_dt_log"
      ssm_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "ssm"))
      assert ssm_key != nil, "Should have an SSM param group"

      ssm_params = params.data[ssm_key]
      assert is_map(ssm_params) and not is_struct(ssm_params),
        "SSM param group should be a map with sub-params"

      ssm_sub_keys = Map.keys(ssm_params)

      a_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "a_log"))
      assert length(a_log_keys) > 0,
        "SSM params should contain 'a_log', got sub-keys: #{inspect(ssm_sub_keys)}"

      dt_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "dt_log"))
      assert length(dt_log_keys) > 0,
        "SSM params should contain 'dt_log', got sub-keys: #{inspect(ssm_sub_keys)}"
    end

    test "a_log is initialized to log(1..N) (S4D-Lin initialization)" do
      model = S4D.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Find the SSM layer and extract a_log
      ssm_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "ssm"))
      ssm_params = params.data[ssm_key]
      a_log_key = Enum.find(Map.keys(ssm_params), &String.contains?(&1, "a_log"))

      assert a_log_key != nil, "Should have an a_log sub-key"

      a_log_value = ssm_params[a_log_key]

      # Expected: log(1), log(2), ..., log(N) for state_size N
      expected = Nx.log(Nx.add(Nx.iota({@state_size}, type: :f32), 1.0))

      diff = Nx.subtract(a_log_value, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5,
        "a_log should be initialized to log(1..N), max diff = #{diff}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, hidden_size]" do
      model = S4D.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "different inputs produce different outputs" do
      model = S4D.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end

    test "output is finite (no NaN/Inf)" do
      model = S4D.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "model is deterministic without dropout" do
      model = S4D.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
