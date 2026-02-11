defmodule Edifice.SSM.S4CorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.S4

  @batch 2
  @embed_size 16
  @hidden_size 16
  @state_size 8
  @seq_len 10

  @s4_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    state_size: @state_size,
    num_layers: 1,
    dropout: 0.0,
    window_size: @seq_len,
    seq_len: @seq_len
  ]

  # ============================================================================
  # SSM Structure
  # ============================================================================

  describe "S4 SSM structure" do
    test "model has dt projection parameters" do
      model = S4.build(@s4_opts)
      {init_fn, _predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have dt projection (input-dependent discretization)
      assert Enum.any?(param_keys, &String.contains?(&1, "dt_proj"))

      # Should have B and C projections
      assert Enum.any?(param_keys, &String.contains?(&1, "b_proj"))
      assert Enum.any?(param_keys, &String.contains?(&1, "c_proj"))
    end

    test "output is per-channel (not broadcast scalar)" do
      model = S4.build(@s4_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size]
      assert Nx.shape(output) == {@batch, @hidden_size}

      # Different channels should have different values (not broadcast from scalar)
      first_sample = Nx.slice(output, [0, 0], [1, @hidden_size]) |> Nx.reshape({@hidden_size})
      values = Nx.to_flat_list(first_sample)
      unique_count = values |> Enum.uniq() |> length()
      assert unique_count > 1
    end
  end

  # ============================================================================
  # Temporal Dynamics
  # ============================================================================

  describe "temporal dynamics" do
    test "output depends on sequence content (not just last frame)" do
      model = S4.build(@s4_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      # Create input2: same last frame but different history
      {different_history, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len - 1, @embed_size})
      last_frame = Nx.slice(input1, [0, @seq_len - 1, 0], [@batch, 1, @embed_size])
      input2 = Nx.concatenate([different_history, last_frame], axis: 1)

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      # Same last frame, different history -> different outputs (proves SSM tracks state)
      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "constant input produces smooth output" do
      model = S4.build(@s4_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Constant input: all ones
      input = Nx.broadcast(1.0, {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be finite
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # HiPPO Stability
  # ============================================================================

  describe "HiPPO stability" do
    test "deeper model still produces finite outputs" do
      # 4 layers should still be numerically stable thanks to HiPPO init
      deep_opts = Keyword.put(@s4_opts, :num_layers, 4)
      model = S4.build(deep_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "model is deterministic without dropout" do
      model = S4.build(@s4_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end
end
