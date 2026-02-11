defmodule Edifice.Recurrent.TTTCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.TTT

  @batch 2
  @embed_size 32
  @hidden_size 32
  @inner_size 16
  @seq_len 4

  @base_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    inner_size: @inner_size,
    num_layers: 1,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Learnable W_0
  # ============================================================================

  describe "learnable W_0" do
    test "params contain w0 key (nested under recurrence layer)" do
      model = TTT.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # W_0 is nested: params.data["ttt_1_recurrence"]["ttt_1_w0"]
      recurrence_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "recurrence"))

      assert recurrence_key != nil,
             "Should have a 'recurrence' param group, got: #{inspect(Map.keys(params.data))}"

      recurrence_params = params.data[recurrence_key]

      assert is_map(recurrence_params) and not is_struct(recurrence_params),
             "Recurrence param group should be a map with sub-params"

      w0_keys = Map.keys(recurrence_params) |> Enum.filter(&String.contains?(&1, "w0"))

      assert length(w0_keys) > 0,
             "Recurrence params should contain 'w0' for learnable initial weights, got: #{inspect(Map.keys(recurrence_params))}"
    end

    test "w0 has shape [inner_size, inner_size]" do
      model = TTT.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      recurrence_key = Enum.find(Map.keys(params.data), &String.contains?(&1, "recurrence"))
      recurrence_params = params.data[recurrence_key]
      w0_key = Enum.find(Map.keys(recurrence_params), &String.contains?(&1, "w0"))

      assert w0_key != nil

      w0_value = recurrence_params[w0_key]

      assert Nx.shape(w0_value) == {@inner_size, @inner_size},
             "W_0 should be {#{@inner_size}, #{@inner_size}}, got #{inspect(Nx.shape(w0_value))}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, hidden_size]" do
      model = TTT.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is deterministic in inference mode" do
      model = TTT.build(@base_opts)
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

    test "output is finite (no NaN/Inf)" do
      model = TTT.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different inputs produce different outputs" do
      model = TTT.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
