defmodule Edifice.Recurrent.XLSTMCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.XLSTM

  @batch 2
  @seq_len 8
  @embed_size 32
  @hidden_size 32
  @num_heads 4
  @head_dim 8

  @mlstm_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    num_layers: 1,
    num_heads: @num_heads,
    head_dim: @head_dim,
    variant: :mlstm,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  @slstm_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    num_layers: 1,
    variant: :slstm,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # mLSTM: Matrix Memory via D-Matrix
  # ============================================================================

  describe "mLSTM matrix memory" do
    test "mLSTM uses scalar per-head gates (not hidden-sized)" do
      # In the D-matrix formulation, i and f gates are scalar per head
      # So projection should be: 2*num_heads + hidden + 3*kv_dim
      model = XLSTM.build(@mlstm_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Find the mLSTM projection layer
      proj_keys = Enum.filter(param_keys, &String.contains?(&1, "mlstm_proj"))
      assert proj_keys != []

      proj_params = params.data[hd(proj_keys)]
      kernel = proj_params["kernel"]
      out_dim = elem(Nx.shape(kernel), 1)

      # Expected: num_heads*2 (i,f scalar per head) + hidden (o) + kv_dim*3 (k,v,q)
      kv_dim = @num_heads * @head_dim
      expected = @num_heads * 2 + @hidden_size + kv_dim * 3

      assert out_dim == expected,
             "mLSTM projection should be #{expected} (2*heads + hidden + 3*kv), got #{out_dim}"
    end

    test "mLSTM produces correct output shape" do
      model = XLSTM.build(@mlstm_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "mLSTM is deterministic in inference" do
      model = XLSTM.build(@mlstm_opts)
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

  # ============================================================================
  # sLSTM: Exponential Gating with Normalizer
  # ============================================================================

  describe "sLSTM exponential gating" do
    test "sLSTM produces correct output shape" do
      model = XLSTM.build(@slstm_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "sLSTM has 4x hidden gate projection (i, f, z, o)" do
      model = XLSTM.build(@slstm_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      gate_keys = Enum.filter(param_keys, &String.contains?(&1, "gates_proj"))
      assert gate_keys != []

      gate_params = params.data[hd(gate_keys)]
      kernel = gate_params["kernel"]
      out_dim = elem(Nx.shape(kernel), 1)

      assert out_dim == @hidden_size * 4,
             "sLSTM gates should be 4*hidden (i, f, z, o), got #{out_dim}"
    end
  end

  # ============================================================================
  # Mixed Variant
  # ============================================================================

  describe "mixed variant" do
    test "mixed model alternates sLSTM and mLSTM" do
      model =
        XLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: @num_heads,
          head_dim: @head_dim,
          variant: :mixed,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, _} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Layer 1 (odd) should be sLSTM
      slstm_keys = Enum.filter(param_keys, &String.contains?(&1, "block_1_slstm"))
      assert slstm_keys != [], "Layer 1 should be sLSTM"

      # Layer 2 (even) should be mLSTM
      mlstm_keys = Enum.filter(param_keys, &String.contains?(&1, "block_2_mlstm"))
      assert mlstm_keys != [], "Layer 2 should be mLSTM"
    end

    test "different variants produce different outputs" do
      model_s = XLSTM.build(Keyword.put(@mlstm_opts, :variant, :slstm))
      model_m = XLSTM.build(@mlstm_opts)

      {init_s, pred_s} = Axon.build(model_s, mode: :inference)
      {init_m, pred_m} = Axon.build(model_m, mode: :inference)

      params_s =
        init_s.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      params_m =
        init_m.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output_s = pred_s.(params_s, input)
      output_m = pred_m.(params_m, input)

      diff = Nx.subtract(output_s, output_m) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "sLSTM and mLSTM should produce different outputs"
    end
  end
end
