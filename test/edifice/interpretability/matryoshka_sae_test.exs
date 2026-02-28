defmodule Edifice.Interpretability.MatryoshkaSAETest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.MatryoshkaSAE

  @batch 4
  @input_size 32
  @dict_size 64
  @top_k 8

  @opts [
    input_size: @input_size,
    dict_size: @dict_size,
    top_k: @top_k
  ]

  defp template, do: %{"matryoshka_sae_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = MatryoshkaSAE.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = MatryoshkaSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"matryoshka_sae_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = MatryoshkaSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"matryoshka_sae_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_encoder/1" do
    test "produces sparse hidden activations" do
      encoder = MatryoshkaSAE.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"matryoshka_sae_input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      hidden = predict_fn.(params, %{"matryoshka_sae_input" => random_input()})
      assert Nx.shape(hidden) == {@batch, @dict_size}

      num_nonzero = Nx.sum(Nx.greater(Nx.abs(hidden), 1.0e-6)) |> Nx.to_number()
      assert num_nonzero <= @batch * @top_k + @batch
    end
  end

  describe "loss/4" do
    test "returns scalar loss with weighted L1" do
      input = Nx.broadcast(1.0, {@batch, @input_size})
      reconstruction = Nx.broadcast(0.9, {@batch, @input_size})
      hidden_acts = Nx.broadcast(0.5, {@batch, @dict_size})

      loss = MatryoshkaSAE.loss(input, reconstruction, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "weighted L1 penalizes later features more" do
      input = Nx.broadcast(0.0, {@batch, @input_size})

      # Only early features active
      early =
        Nx.concatenate(
          [
            Nx.broadcast(1.0, {@batch, div(@dict_size, 2)}),
            Nx.broadcast(0.0, {@batch, div(@dict_size, 2)})
          ],
          axis: 1
        )

      # Only late features active
      late =
        Nx.concatenate(
          [
            Nx.broadcast(0.0, {@batch, div(@dict_size, 2)}),
            Nx.broadcast(1.0, {@batch, div(@dict_size, 2)})
          ],
          axis: 1
        )

      loss_early = MatryoshkaSAE.loss(input, input, early, l1_coeff: 1.0)
      loss_late = MatryoshkaSAE.loss(input, input, late, l1_coeff: 1.0)

      # Late features should have higher penalty
      assert Nx.to_number(loss_late) > Nx.to_number(loss_early)
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert MatryoshkaSAE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds matryoshka_sae via registry" do
      model = Edifice.build(:matryoshka_sae, @opts)
      assert %Axon{} = model
    end
  end
end
