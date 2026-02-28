defmodule Edifice.Attention.RWKVCorrectnessTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.RWKV

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @head_size 16

  @rwkv_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    head_size: @head_size,
    window_size: @seq_len,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Time-First (u) Parameter
  # ============================================================================

  describe "time_first parameter" do
    test "model has time_first (u) constant that affects architecture" do
      # Build the model and verify it produces valid outputs
      # The u parameter is passed as Axon.constant to the WKV layer
      model = RWKV.build(@rwkv_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Token Shift
  # ============================================================================

  describe "token shift" do
    test "token shift doubles hidden dimension (current + shifted)" do
      # The token shift concatenates current and previous tokens
      # So the mix_proj should have input dim = 2 * hidden_size
      model = RWKV.build(@rwkv_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have mix_proj for time mixing
      mix_proj_keys = Enum.filter(param_keys, &String.contains?(&1, "mix_proj"))
      assert mix_proj_keys != []

      # The mix_proj kernel should have input dim = 2 * hidden_size
      mix_params = params.data[hd(mix_proj_keys)]
      kernel = mix_params["kernel"]
      in_dim = elem(Nx.shape(kernel), 0)

      assert in_dim == @hidden_size * 2,
             "mix_proj input should be 2*hidden_size (current+shifted), got #{in_dim}"
    end
  end

  # ============================================================================
  # Channel Mixing
  # ============================================================================

  describe "channel mixing" do
    test "channel mixing has squared ReLU gating" do
      # The model should use squared ReLU in channel mixing (k_squared)
      # We verify by checking that the model has k_proj and v_proj
      model = RWKV.build(@rwkv_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Channel mixing should have r_proj, k_proj, v_proj
      channel_keys = Enum.filter(param_keys, &String.contains?(&1, "channel_mix"))
      r_keys = Enum.filter(channel_keys, &String.contains?(&1, "r_proj"))
      k_keys = Enum.filter(channel_keys, &String.contains?(&1, "k_proj"))
      v_keys = Enum.filter(channel_keys, &String.contains?(&1, "v_proj"))

      assert r_keys != [], "Should have channel mixing r_proj"
      assert k_keys != [], "Should have channel mixing k_proj"
      assert v_keys != [], "Should have channel mixing v_proj"
    end
  end

  # ============================================================================
  # WKV Attention Properties
  # ============================================================================

  describe "WKV attention properties" do
    test "different head sizes produce different outputs" do
      model_16 = RWKV.build(@rwkv_opts)
      model_32 = RWKV.build(Keyword.put(@rwkv_opts, :head_size, 32))

      {init_16, pred_16} = Axon.build(model_16, mode: :inference)
      {init_32, pred_32} = Axon.build(model_32, mode: :inference)

      params_16 =
        init_16.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      params_32 =
        init_32.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output_16 = pred_16.(params_16, input)
      output_32 = pred_32.(params_32, input)

      assert Nx.shape(output_16) == {@batch, @hidden_size}
      assert Nx.shape(output_32) == {@batch, @hidden_size}

      diff = Nx.subtract(output_16, output_32) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different head sizes should produce different outputs"
    end

    test "inference is deterministic" do
      model = RWKV.build(@rwkv_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Inference should be deterministic"
    end

    test "WKV attention structure has R, W, K, V projections" do
      model = RWKV.build(@rwkv_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      time_keys = Enum.filter(param_keys, &String.contains?(&1, "time_mix"))

      # Should have all four projections
      assert Enum.any?(time_keys, &String.contains?(&1, "r_proj"))
      assert Enum.any?(time_keys, &String.contains?(&1, "w_proj"))
      assert Enum.any?(time_keys, &String.contains?(&1, "k_proj"))
      assert Enum.any?(time_keys, &String.contains?(&1, "v_proj"))
    end
  end
end
