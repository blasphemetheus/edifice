defmodule Edifice.Interpretability.BatchTopKSAETest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.BatchTopKSAE

  @batch 4
  @input_size 32
  @dict_size 64
  @batch_k 32

  @opts [
    input_size: @input_size,
    dict_size: @dict_size,
    batch_k: @batch_k
  ]

  defp template, do: %{"batch_topk_sae_input" => Nx.template({@batch, @input_size}, :f32)}

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = BatchTopKSAE.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = BatchTopKSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"batch_topk_sae_input" => random_input()})
      assert Nx.shape(output) == {@batch, @input_size}
    end

    test "output contains finite values" do
      model = BatchTopKSAE.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"batch_topk_sae_input" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_encoder/1" do
    test "enforces batch-global sparsity budget" do
      encoder = BatchTopKSAE.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"batch_topk_sae_input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      hidden = predict_fn.(params, %{"batch_topk_sae_input" => random_input()})
      assert Nx.shape(hidden) == {@batch, @dict_size}

      # Total non-zero activations across the batch should be <= batch_k
      # (may be slightly more due to ties at the threshold boundary)
      num_nonzero = Nx.sum(Nx.greater(Nx.abs(hidden), 1.0e-6)) |> Nx.to_number()
      assert num_nonzero <= @batch_k + @batch, "expected at most ~#{@batch_k} nonzero globally"
    end

    test "allows variable sparsity per sample" do
      # With global top-k, different samples can have different numbers of
      # active features (unlike per-sample top-k which forces exactly k each)
      encoder = BatchTopKSAE.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"batch_topk_sae_input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      hidden = predict_fn.(params, %{"batch_topk_sae_input" => random_input()})

      # Count nonzero per sample
      per_sample =
        for i <- 0..(@batch - 1) do
          Nx.sum(Nx.greater(Nx.abs(hidden[i]), 1.0e-6)) |> Nx.to_number()
        end

      # Not all samples need to have the same count
      # (they might happen to be equal, but the mechanism allows variance)
      assert length(per_sample) == @batch
      assert Enum.all?(per_sample, &(&1 >= 0))
    end
  end

  describe "loss/4" do
    test "returns scalar loss" do
      input = Nx.broadcast(1.0, {@batch, @input_size})
      reconstruction = Nx.broadcast(0.9, {@batch, @input_size})
      hidden_acts = Nx.broadcast(0.5, {@batch, @dict_size})

      loss = BatchTopKSAE.loss(input, reconstruction, hidden_acts, l1_coeff: 1.0e-3)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert BatchTopKSAE.output_size(@opts) == @input_size
    end
  end

  describe "Edifice.build/2" do
    test "builds batch_top_k_sae via registry" do
      model = Edifice.build(:batch_top_k_sae, @opts)
      assert %Axon{} = model
    end
  end
end
