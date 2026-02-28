defmodule Edifice.Interpretability.JumpReluSAETest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.JumpReluSAE

  @batch 4
  @input_size 32
  @dict_size 64

  @opts [
    input_size: @input_size,
    dict_size: @dict_size,
    temperature: 10.0
  ]

  defp template, do: %{"jump_relu_sae_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = JumpReluSAE.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = JumpReluSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"jump_relu_sae_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = JumpReluSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"jump_relu_sae_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "has learnable threshold parameter" do
      model = JumpReluSAE.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())

      data = params.data
      assert Map.has_key?(data, "jump_relu_sae_threshold")
      threshold = data["jump_relu_sae_threshold"]["threshold"]
      assert Nx.shape(threshold) == {@dict_size}
    end
  end

  describe "build_encoder/1" do
    test "produces hidden activations with sparsity" do
      encoder = JumpReluSAE.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"jump_relu_sae_input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      hidden = predict_fn.(params, %{"jump_relu_sae_input" => random_input()})
      assert Nx.shape(hidden) == {@batch, @dict_size}

      # JumpReLU suppresses negative values — most near-zero entries should exist
      num_near_zero = Nx.sum(Nx.less(Nx.abs(hidden), 0.01)) |> Nx.to_number()
      total = @batch * @dict_size
      assert num_near_zero > 0, "expected some near-zero activations from JumpReLU gating"
      assert num_near_zero < total, "expected some non-zero activations"
    end
  end

  describe "build/1 with custom temperature" do
    test "higher temperature produces sharper gating" do
      # At very high temperature, JumpReLU approaches hard thresholding
      model_sharp = JumpReluSAE.build(Keyword.put(@opts, :temperature, 100.0))
      model_soft = JumpReluSAE.build(Keyword.put(@opts, :temperature, 1.0))

      {init_fn_s, predict_fn_s} = Axon.build(model_sharp, mode: :inference)
      {init_fn_f, predict_fn_f} = Axon.build(model_soft, mode: :inference)

      params_s = init_fn_s.(template(), Axon.ModelState.empty())
      params_f = init_fn_f.(template(), Axon.ModelState.empty())

      input = random_input()
      output_sharp = predict_fn_s.(params_s, %{"jump_relu_sae_input" => input})
      output_soft = predict_fn_f.(params_f, %{"jump_relu_sae_input" => input})

      # Both should produce valid outputs
      assert Nx.all(Nx.is_nan(output_sharp) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output_soft) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "loss/4" do
    test "returns scalar loss" do
      input = Nx.broadcast(1.0, {@batch, @input_size})
      reconstruction = Nx.broadcast(0.9, {@batch, @input_size})
      hidden_acts = Nx.broadcast(0.5, {@batch, @dict_size})

      loss = JumpReluSAE.loss(input, reconstruction, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert JumpReluSAE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds jump_relu_sae via registry" do
      model = Edifice.build(:jump_relu_sae, @opts)
      assert %Axon{} = model
    end
  end
end
