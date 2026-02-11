defmodule Edifice.Meta.MetaNewTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.SwitchMoE
  alias Edifice.Meta.SoftMoE
  alias Edifice.Meta.LoRA
  alias Edifice.Meta.Adapter

  @batch_size 2
  @embed_size 32
  @hidden_size 32
  @seq_len 8

  # ============================================================================
  # SwitchMoE Tests
  # ============================================================================

  describe "SwitchMoE.build/1" do
    test "produces correct output shape" do
      model =
        SwitchMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 4,
          expert_size: 64,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      # Extracts last timestep: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with 2 experts" do
      model =
        SwitchMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 2,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    @tag :slow
    test "with 8 experts" do
      model =
        SwitchMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 8,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output_size/1 returns correct value" do
      assert SwitchMoE.output_size(hidden_size: 64) == 64
      assert SwitchMoE.output_size() == 256
    end
  end

  # ============================================================================
  # SoftMoE Tests
  # ============================================================================

  describe "SoftMoE.build/1" do
    test "produces correct output shape" do
      model =
        SoftMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 4,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      # Extracts last timestep: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with 2 experts" do
      model =
        SoftMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 2,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    @tag :slow
    test "with 8 experts" do
      model =
        SoftMoE.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_experts: 8,
          num_layers: 1,
          dropout: 0.0,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output_size/1 returns correct value" do
      assert SoftMoE.output_size(hidden_size: 128) == 128
      assert SoftMoE.output_size() == 256
    end
  end

  # ============================================================================
  # LoRA Tests
  # ============================================================================

  describe "LoRA.build/1" do
    test "produces correct output shape" do
      model =
        LoRA.build(
          input_size: @hidden_size,
          output_size: @hidden_size,
          rank: 4,
          alpha: 8.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with different input and output sizes" do
      model =
        LoRA.build(
          input_size: @hidden_size,
          output_size: 64,
          rank: 4,
          alpha: 8.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "initial output is near zero due to zero-initialized B" do
      model =
        LoRA.build(
          input_size: @hidden_size,
          output_size: @hidden_size,
          rank: 4,
          alpha: 8.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input})

      # B is initialized to zeros, so output should be near zero
      max_val = output |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_val < 0.01
    end
  end

  describe "LoRA.wrap/3" do
    test "wraps an existing dense layer with correct output shape" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      original = Axon.dense(input, @hidden_size, name: "base_dense")

      adapted =
        LoRA.wrap(input, original,
          rank: 4,
          alpha: 8.0,
          output_size: @hidden_size,
          name: "lora_wrap"
        )

      {init_fn, predict_fn} = Axon.build(adapted)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input_data})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "LoRA.lora_delta/3" do
    test "produces delta with correct shape" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      delta = LoRA.lora_delta(input, 64, rank: 4, alpha: 8.0)

      {init_fn, predict_fn} = Axon.build(delta)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input_data})

      assert Nx.shape(output) == {@batch_size, 64}
    end
  end

  # ============================================================================
  # Adapter Tests
  # ============================================================================

  describe "Adapter.build/1" do
    test "produces correct output shape" do
      model =
        Adapter.build(
          hidden_size: @hidden_size,
          bottleneck_size: 8
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input})

      # Adapter preserves input dimension
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "initial output close to input due to zero-initialized up-projection" do
      model =
        Adapter.build(
          hidden_size: @hidden_size,
          bottleneck_size: 8
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input})

      # Up-projection is zero-initialized, so adapter(x) ~= 0
      # Output should be close to input (x + 0 = x)
      diff = Nx.subtract(output, input) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 0.01
    end
  end

  describe "Adapter.wrap/2" do
    test "wraps an existing layer with correct output shape" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      layer_output = Axon.dense(input, @hidden_size, name: "base_layer")

      adapted =
        Adapter.wrap(layer_output,
          hidden_size: @hidden_size,
          bottleneck_size: 8,
          name: "adapter_wrap"
        )

      {init_fn, predict_fn} = Axon.build(adapted)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input_data})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "Adapter.adapter_block/3" do
    test "produces residual output with correct shape" do
      input = Axon.input("input", shape: {nil, @hidden_size})

      block =
        Adapter.adapter_block(input, @hidden_size,
          bottleneck_size: 8,
          activation: :relu,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(block)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch_size, @hidden_size})
      output = predict_fn.(params, %{"input" => input_data})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "Adapter.output_size/1" do
    test "returns hidden_size" do
      assert Adapter.output_size(hidden_size: 64) == 64
      assert Adapter.output_size(hidden_size: 256) == 256
    end
  end
end
