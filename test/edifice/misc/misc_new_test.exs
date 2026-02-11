defmodule Edifice.Misc.MiscNewTest do
  use ExUnit.Case, async: true

  alias Edifice.Feedforward.TabNet
  alias Edifice.Convolutional.MobileNet
  alias Edifice.Convolutional.EfficientNet
  alias Edifice.Energy.NeuralODE
  alias Edifice.Probabilistic.EvidentialNN
  alias Edifice.Neuromorphic.ANN2SNN
  alias Edifice.Memory.MemoryNetwork

  @batch_size 2

  # ============================================================================
  # TabNet Tests
  # ============================================================================

  describe "TabNet.build/1" do
    @input_size 16
    @hidden_size 8

    test "produces correct output shape" do
      model =
        TabNet.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_steps: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with num_classes adds classification head" do
      model =
        TabNet.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_steps: 2,
          num_classes: 5
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 5}
    end

    test "output_size/1 returns correct value" do
      assert TabNet.output_size(hidden_size: 32) == 32
      assert TabNet.output_size(hidden_size: 32, num_classes: 10) == 10
    end
  end

  # ============================================================================
  # MobileNet Tests
  # ============================================================================

  describe "MobileNet.build/1" do
    @input_dim 16

    test "produces correct output shape" do
      model =
        MobileNet.build(
          input_dim: @input_dim,
          hidden_dims: [8, 16]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_dim})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 16}
    end

    test "with num_classes adds classification head" do
      model =
        MobileNet.build(
          input_dim: @input_dim,
          hidden_dims: [8, 16],
          num_classes: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_dim})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 4}
    end

    test "with width_multiplier scales channels" do
      model =
        MobileNet.build(
          input_dim: @input_dim,
          hidden_dims: [16, 32],
          width_multiplier: 0.5
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_dim})
      output = predict_fn.(params, %{"input" => input})

      # 32 * 0.5 = 16
      assert Nx.shape(output) == {@batch_size, 16}
    end

    test "output_size/1 returns correct value" do
      assert MobileNet.output_size(hidden_dims: [64, 128]) == 128
      assert MobileNet.output_size(hidden_dims: [64], num_classes: 10) == 10
    end
  end

  # ============================================================================
  # EfficientNet Tests
  # ============================================================================

  describe "EfficientNet.build/1" do
    @input_dim 32

    test "produces correct output shape" do
      model =
        EfficientNet.build(
          input_dim: @input_dim,
          base_dim: 8,
          depth_multiplier: 1.0,
          width_multiplier: 0.25
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_dim})
      output = predict_fn.(params, %{"input" => input})

      # Output is head_dim = scale_width(1280, 0.25)
      {_batch, out_dim} = Nx.shape(output)
      assert out_dim > 0
    end

    test "with num_classes adds classification head" do
      model =
        EfficientNet.build(
          input_dim: @input_dim,
          base_dim: 8,
          width_multiplier: 0.25,
          num_classes: 6
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_dim})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 6}
    end

    test "output_size/1 returns correct value" do
      assert EfficientNet.output_size(num_classes: 10) == 10
    end
  end

  # ============================================================================
  # NeuralODE Tests
  # ============================================================================

  describe "NeuralODE.build/1" do
    @input_size 16
    @hidden_size 16

    test "produces correct output shape" do
      model =
        NeuralODE.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_steps: 3,
          step_size: 0.1
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "with different input and hidden sizes" do
      model =
        NeuralODE.build(
          input_size: @input_size,
          hidden_size: 32,
          num_steps: 2,
          step_size: 0.1
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 32}
    end

    test "with output_size projection" do
      model =
        NeuralODE.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_steps: 2,
          step_size: 0.1,
          output_size: 8
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 8}
    end

    test "output_size/1 returns correct value" do
      assert NeuralODE.output_size(hidden_size: 64) == 64
      assert NeuralODE.output_size(hidden_size: 64, output_size: 32) == 32
    end
  end

  describe "NeuralODE.build_shared/1" do
    @input_size 16
    @hidden_size 16

    test "produces correct output shape with shared dynamics" do
      model =
        NeuralODE.build_shared(
          input_size: @input_size,
          hidden_size: @hidden_size,
          num_steps: 3,
          step_size: 0.1
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  # ============================================================================
  # EvidentialNN Tests
  # ============================================================================

  describe "EvidentialNN.build/1" do
    @input_size 16

    test "produces Dirichlet alpha parameters with correct shape" do
      model =
        EvidentialNN.build(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          num_classes: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      alpha = predict_fn.(params, %{"input" => input})

      assert Nx.shape(alpha) == {@batch_size, 4}

      # Alpha values should all be > 1 (evidence + 1)
      min_alpha = alpha |> Nx.reduce_min() |> Nx.to_number()
      assert min_alpha > 1.0
    end

    test "output_size/1 returns num_classes" do
      assert EvidentialNN.output_size(num_classes: 10) == 10
      assert EvidentialNN.output_size(num_classes: 3) == 3
    end
  end

  describe "EvidentialNN.uncertainty/1" do
    test "computes epistemic and aleatoric uncertainty" do
      # Create synthetic alpha values
      alpha = Nx.tensor([[5.0, 3.0, 2.0], [10.0, 10.0, 10.0]])
      {epistemic, aleatoric} = EvidentialNN.uncertainty(alpha)

      assert Nx.shape(epistemic) == {2}
      assert Nx.shape(aleatoric) == {2}

      # Epistemic: K/S - higher when less evidence
      epistemic_vals = Nx.to_flat_list(epistemic)
      # First sample: S=10, K=3 -> 3/10 = 0.3
      assert_in_delta Enum.at(epistemic_vals, 0), 0.3, 0.01
      # Second sample: S=30, K=3 -> 3/30 = 0.1
      assert_in_delta Enum.at(epistemic_vals, 1), 0.1, 0.01
    end
  end

  describe "EvidentialNN.expected_probability/1" do
    test "computes expected probabilities that sum to 1" do
      alpha = Nx.tensor([[5.0, 3.0, 2.0], [10.0, 10.0, 10.0]])
      probs = EvidentialNN.expected_probability(alpha)

      assert Nx.shape(probs) == {2, 3}

      # Each row should sum to 1
      sums = Nx.sum(probs, axes: [1])
      for s <- Nx.to_flat_list(sums) do
        assert_in_delta s, 1.0, 1.0e-5
      end
    end
  end

  describe "EvidentialNN.evidential_loss/3" do
    test "returns scalar loss" do
      alpha = Nx.tensor([[5.0, 3.0, 2.0], [2.0, 8.0, 1.0]])
      targets = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

      loss = EvidentialNN.evidential_loss(alpha, targets)
      assert Nx.shape(loss) == {}

      loss_val = Nx.to_number(loss)
      assert loss_val > 0.0
    end
  end

  # ============================================================================
  # ANN2SNN Tests
  # ============================================================================

  describe "ANN2SNN.build/1 (ANN)" do
    @input_size 16

    test "produces correct output shape" do
      model =
        ANN2SNN.build(
          input_size: @input_size,
          hidden_sizes: [16, 8]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      # Output is last hidden size when no explicit output_size
      assert Nx.shape(output) == {@batch_size, 8}
    end

    test "with explicit output_size" do
      model =
        ANN2SNN.build(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          output_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 4}
    end
  end

  describe "ANN2SNN.build_snn/1" do
    @input_size 16

    test "produces correct output shape" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch_size, 8}
    end

    test "output values are non-negative (spike rates)" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch_size, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(2.0, {@batch_size, @input_size})
      output = predict_fn.(params, %{"input" => input})

      min_val = output |> Nx.reduce_min() |> Nx.to_number()
      assert min_val >= 0.0
    end

    test "output_size/1 returns correct value" do
      assert ANN2SNN.output_size(hidden_sizes: [64, 32]) == 32
      assert ANN2SNN.output_size(hidden_sizes: [64, 32], output_size: 10) == 10
    end
  end

  # ============================================================================
  # MemoryNetwork Tests
  # ============================================================================

  describe "MemoryNetwork.build/1" do
    @input_dim 16
    @num_memories 6
    @memory_dim 8

    test "produces correct output shape" do
      model =
        MemoryNetwork.build(
          input_dim: @input_dim,
          memory_dim: @memory_dim,
          num_hops: 2,
          output_dim: 8
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "query" => Nx.template({@batch_size, @input_dim}, :f32),
            "memories" => Nx.template({@batch_size, @num_memories, @input_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      query = Nx.broadcast(0.5, {@batch_size, @input_dim})
      memories = Nx.broadcast(0.3, {@batch_size, @num_memories, @input_dim})

      output =
        predict_fn.(params, %{
          "query" => query,
          "memories" => memories
        })

      assert Nx.shape(output) == {@batch_size, 8}
    end

    test "default output_dim equals input_dim" do
      model =
        MemoryNetwork.build(
          input_dim: @input_dim,
          memory_dim: @memory_dim,
          num_hops: 2
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "query" => Nx.template({@batch_size, @input_dim}, :f32),
            "memories" => Nx.template({@batch_size, @num_memories, @input_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      query = Nx.broadcast(0.5, {@batch_size, @input_dim})
      memories = Nx.broadcast(0.3, {@batch_size, @num_memories, @input_dim})

      output =
        predict_fn.(params, %{
          "query" => query,
          "memories" => memories
        })

      assert Nx.shape(output) == {@batch_size, @input_dim}
    end

    test "with single hop" do
      model =
        MemoryNetwork.build(
          input_dim: @input_dim,
          memory_dim: @memory_dim,
          num_hops: 1
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "query" => Nx.template({@batch_size, @input_dim}, :f32),
            "memories" => Nx.template({@batch_size, @num_memories, @input_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      query = Nx.broadcast(0.5, {@batch_size, @input_dim})
      memories = Nx.broadcast(0.3, {@batch_size, @num_memories, @input_dim})

      output =
        predict_fn.(params, %{
          "query" => query,
          "memories" => memories
        })

      assert Nx.shape(output) == {@batch_size, @input_dim}
    end
  end
end
