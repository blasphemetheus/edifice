defmodule Edifice.SSM.HyenaCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.SSM.Hyena

  @batch 2
  @seq_len 12
  @embed_size 16
  @hidden_size 16
  @filter_size 8

  @hyena_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    order: 2,
    filter_size: @filter_size,
    num_layers: 1,
    window_size: @seq_len,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Filter MLP Structure
  # ============================================================================

  describe "filter MLP structure" do
    test "model has filter MLP dense parameters" do
      model = Hyena.build(@hyena_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have filter dense layers (3 per order, 2 orders)
      filter_keys = Enum.filter(param_keys, &String.contains?(&1, "filter"))

      assert length(filter_keys) >= 6,
             "Expected >=6 filter params (3 dense x 2 orders), got #{length(filter_keys)}"

      # Each filter has dense1 -> sin -> dense2 -> sin -> dense3
      for idx <- 0..1 do
        assert Enum.any?(filter_keys, &String.contains?(&1, "filter#{idx}_dense1"))
        assert Enum.any?(filter_keys, &String.contains?(&1, "filter#{idx}_dense3"))
      end
    end

    test "filter MLP final layer outputs hidden_size dimensions" do
      model = Hyena.build(@hyena_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Final filter dense should project to hidden_size
      dense3_params = params.data["hyena_block_1_filter0_dense3"]
      kernel = dense3_params["kernel"]

      # Output dim should be hidden_size
      out_dim = elem(Nx.shape(kernel), 1)

      assert out_dim == @hidden_size,
             "filter output should be hidden_size=#{@hidden_size}, got #{out_dim}"
    end
  end

  # ============================================================================
  # Short Depthwise Convolution
  # ============================================================================

  describe "short depthwise convolution" do
    test "model has short depthwise conv parameters with kernel_size=3" do
      model = Hyena.build(@hyena_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have short depthwise conv
      dw_keys = Enum.filter(param_keys, &String.contains?(&1, "short_dw_conv"))
      assert dw_keys != [], "Should have short depthwise conv parameters"

      # Check kernel shape: for depthwise 1D conv, kernel is {kernel_size, 1, channels}
      for key <- dw_keys do
        layer_params = params.data[key]

        if Map.has_key?(layer_params, "kernel") do
          kernel = layer_params["kernel"]
          shape = Nx.shape(kernel)
          # Axon 1D conv kernel: {kernel_size, in_channels/groups, out_channels}
          assert elem(shape, 0) == 3, "short conv kernel size should be 3, got #{elem(shape, 0)}"
        end
      end
    end
  end

  # ============================================================================
  # Gating Structure
  # ============================================================================

  describe "gating structure" do
    test "order determines number of conv+gate rounds" do
      # order=1: 1 round (v, x1)
      model_o1 = Hyena.build(Keyword.put(@hyena_opts, :order, 1))
      {init_fn_1, _} = Axon.build(model_o1, mode: :inference)

      params_1 =
        init_fn_1.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # order=2: 2 rounds (v, x1, x2)
      model_o2 = Hyena.build(@hyena_opts)
      {init_fn_2, _} = Axon.build(model_o2, mode: :inference)

      params_2 =
        init_fn_2.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      keys_1 = Map.keys(params_1.data)
      keys_2 = Map.keys(params_2.data)

      # Order 1 should have 1 filter set, order 2 should have 2
      filter0_1 = Enum.filter(keys_1, &String.contains?(&1, "filter0"))
      filter1_1 = Enum.filter(keys_1, &String.contains?(&1, "filter1"))
      filter0_2 = Enum.filter(keys_2, &String.contains?(&1, "filter0"))
      filter1_2 = Enum.filter(keys_2, &String.contains?(&1, "filter1"))

      assert filter0_1 != []
      assert filter1_1 == [], "order=1 should NOT have filter1"
      assert filter0_2 != []
      assert filter1_2 != [], "order=2 should have filter1"
    end

    test "different orders produce different outputs" do
      model_o1 = Hyena.build(Keyword.put(@hyena_opts, :order, 1))
      model_o2 = Hyena.build(@hyena_opts)

      {init_1, pred_1} = Axon.build(model_o1, mode: :inference)
      {init_2, pred_2} = Axon.build(model_o2, mode: :inference)

      params_1 =
        init_1.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      params_2 =
        init_2.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output_1 = pred_1.(params_1, input)
      output_2 = pred_2.(params_2, input)

      assert Nx.shape(output_1) == {@batch, @hidden_size}
      assert Nx.shape(output_2) == {@batch, @hidden_size}

      # Different orders should produce different outputs
      diff = Nx.subtract(output_1, output_2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different orders should produce different outputs"
    end
  end

  # ============================================================================
  # Causal Behavior
  # ============================================================================

  describe "causal behavior" do
    test "model is deterministic in inference mode" do
      model = Hyena.build(@hyena_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Inference should be deterministic"
    end

    test "different inputs produce different outputs" do
      model = Hyena.build(@hyena_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      {input2, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
