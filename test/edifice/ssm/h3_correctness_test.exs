defmodule Edifice.SSM.H3CorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.H3

  @batch 2
  @embed_dim 32
  @hidden_size 32
  @state_size 8
  @seq_len 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    state_size: @state_size,
    conv_size: 4,
    num_layers: 1,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Learnable SSM Parameters
  # ============================================================================

  describe "learnable SSM parameters" do
    test "params contain a_log and dt_log keys for shift SSM" do
      model = H3.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      ssm_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "shift_ssm"))

      assert ssm_key != nil,
             "Should have shift SSM param group, got: #{inspect(Map.keys(params.data))}"

      ssm_params = params.data[ssm_key]
      assert is_map(ssm_params) and not is_struct(ssm_params)

      ssm_sub_keys = Map.keys(ssm_params)

      a_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "a_log"))

      assert a_log_keys != [],
             "Shift SSM should contain 'a_log', got: #{inspect(ssm_sub_keys)}"

      dt_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "dt_log"))

      assert dt_log_keys != [],
             "Shift SSM should contain 'dt_log', got: #{inspect(ssm_sub_keys)}"
    end

    test "params contain a_log and dt_log keys for diag SSM" do
      model = H3.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      ssm_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "diag_ssm"))
      assert ssm_key != nil, "Should have diag SSM param group"

      ssm_params = params.data[ssm_key]
      ssm_sub_keys = Map.keys(ssm_params)

      a_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "a_log"))
      assert a_log_keys != [], "Diag SSM should contain 'a_log'"

      dt_log_keys = Enum.filter(ssm_sub_keys, &String.contains?(&1, "dt_log"))
      assert dt_log_keys != [], "Diag SSM should contain 'dt_log'"
    end

    test "params contain depthwise conv weights (not window_mean)" do
      model = H3.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      conv_keys = Enum.filter(param_keys, &String.contains?(&1, "dw_conv"))

      assert conv_keys != [],
             "Should have depthwise conv params, got: #{inspect(param_keys)}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, hidden_size]" do
      model = H3.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = H3.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is deterministic" do
      model = H3.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
