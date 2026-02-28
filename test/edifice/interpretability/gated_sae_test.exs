defmodule Edifice.Interpretability.GatedSAETest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.GatedSAE

  @batch 4
  @input_size 32
  @dict_size 64
  @top_k 8

  @opts [
    input_size: @input_size,
    dict_size: @dict_size,
    top_k: @top_k
  ]

  defp template, do: %{"gated_sae_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = GatedSAE.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = GatedSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"gated_sae_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = GatedSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"gated_sae_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "uses separate encoder and gate parameters" do
      model = GatedSAE.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())

      data = params.data
      assert Map.has_key?(data, "gated_sae_encoder")
      assert Map.has_key?(data, "gated_sae_gate")
      assert Map.has_key?(data, "gated_sae_decoder")
    end
  end

  describe "build_encoder/1" do
    test "produces sparse hidden activations" do
      encoder = GatedSAE.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"gated_sae_input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      hidden = predict_fn.(params, %{"gated_sae_input" => random_input()})
      assert Nx.shape(hidden) == {@batch, @dict_size}

      # Most values should be zero (top-k gated sparsity)
      num_nonzero = Nx.sum(Nx.greater(Nx.abs(hidden), 1.0e-6)) |> Nx.to_number()
      assert num_nonzero <= @batch * @top_k + @batch
    end
  end

  describe "loss/4" do
    test "returns scalar loss" do
      input = Nx.broadcast(1.0, {@batch, @input_size})
      reconstruction = Nx.broadcast(0.9, {@batch, @input_size})
      hidden_acts = Nx.broadcast(0.5, {@batch, @dict_size})

      loss = GatedSAE.loss(input, reconstruction, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end

    test "zero reconstruction error and zero activations gives zero loss" do
      input = Nx.broadcast(1.0, {@batch, @input_size})
      hidden_acts = Nx.broadcast(0.0, {@batch, @dict_size})

      loss = GatedSAE.loss(input, input, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.to_number(loss) < 0.001
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert GatedSAE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds gated_sae via registry" do
      model = Edifice.build(:gated_sae, @opts)
      assert %Axon{} = model
    end
  end
end
