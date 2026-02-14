defmodule Edifice.Energy.EBMTest do
  use ExUnit.Case, async: true

  alias Edifice.Energy.EBM

  @batch 2
  @input_size 16

  describe "build/1" do
    test "builds an Axon model" do
      model = EBM.build(input_size: @input_size)
      assert %Axon{} = model
    end

    test "forward pass produces scalar energy per sample" do
      model = EBM.build(input_size: @input_size, hidden_sizes: [8, 4])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_size}))
      assert Nx.shape(output) == {@batch, 1}
    end

    test "supports dropout" do
      model = EBM.build(input_size: @input_size, hidden_sizes: [8], dropout: 0.2)
      assert %Axon{} = model
    end

    test "supports custom activation" do
      model = EBM.build(input_size: @input_size, hidden_sizes: [8], activation: :relu)
      assert %Axon{} = model
    end
  end

  describe "build_energy_fn/2" do
    test "builds energy function from existing input" do
      input = Axon.input("input", shape: {nil, @input_size})
      model = EBM.build_energy_fn(input, hidden_sizes: [8, 4])
      assert %Axon{} = model
    end
  end

  describe "contrastive_divergence_loss/3" do
    test "computes loss from real and negative energies" do
      real_energy = Nx.tensor([[1.0], [2.0]], type: :f32)
      neg_energy = Nx.tensor([[3.0], [4.0]], type: :f32)
      loss = EBM.contrastive_divergence_loss(real_energy, neg_energy)
      # CD loss = mean(real) - mean(neg) + reg
      # mean(real) = 1.5, mean(neg) = 3.5
      # reg = 0.01 * (mean(real^2) + mean(neg^2)) = 0.01 * (2.5 + 12.5) = 0.15
      # total = 1.5 - 3.5 + 0.15 = -1.85
      assert Nx.shape(loss) == {}
      val = Nx.to_number(loss)
      assert abs(val - -1.85) < 0.01
    end

    test "supports custom regularization" do
      real_energy = Nx.tensor([[1.0]], type: :f32)
      neg_energy = Nx.tensor([[2.0]], type: :f32)
      loss = EBM.contrastive_divergence_loss(real_energy, neg_energy, reg_alpha: 0.1)
      assert Nx.shape(loss) == {}
    end
  end

  describe "energy/3" do
    test "computes energy of inputs" do
      model = EBM.build(input_size: @input_size, hidden_sizes: [8])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {@batch, @input_size})
      energy = EBM.energy(predict_fn, params, input)
      assert Nx.shape(energy) == {@batch, 1}
    end
  end

  describe "langevin_sample/4" do
    # Note: Langevin sampling uses Nx.Defn.grad to compute energy gradients
    # w.r.t. inputs. Axon's compiled predict_fn uses internal JIT that can't
    # be traced through grad, so we use a simple numerical energy function.
    defp simple_energy_fn(_params, %{"input" => x}) do
      # Quadratic energy: E(x) = sum(x^2) per sample
      Nx.sum(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    end

    test "generates samples of correct shape" do
      init = Nx.broadcast(0.5, {2, 4})

      samples =
        EBM.langevin_sample(&simple_energy_fn/2, %{}, 4,
          init: init,
          steps: 3,
          step_size: 0.1,
          noise_scale: 0.001
        )

      assert Nx.shape(samples) == {2, 4}
      assert Nx.all(Nx.is_nan(samples) |> Nx.bitwise_not()) |> Nx.to_number() == 1
    end

    test "accepts initial samples" do
      init = Nx.broadcast(0.5, {2, 4})

      samples =
        EBM.langevin_sample(&simple_energy_fn/2, %{}, 4,
          init: init,
          steps: 2,
          step_size: 0.01
        )

      assert Nx.shape(samples) == {2, 4}
    end

    test "supports clamping" do
      init = Nx.broadcast(0.5, {2, 4})

      samples =
        EBM.langevin_sample(&simple_energy_fn/2, %{}, 4,
          init: init,
          steps: 2,
          step_size: 0.01,
          clamp: {-1.0, 1.0}
        )

      assert Nx.shape(samples) == {2, 4}
      assert Nx.all(Nx.greater_equal(samples, -1.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(samples, 1.0)) |> Nx.to_number() == 1
    end
  end
end
