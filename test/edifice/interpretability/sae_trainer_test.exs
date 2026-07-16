defmodule Edifice.Interpretability.SAETrainerTest do
  @moduledoc """
  LEACE-grade functional guarantees for the SAE fit path: plant a sparse
  dictionary, fit, verify the method's contract — recovery above chance,
  reconstruction beating the mean baseline, the decoder-norm constraint
  actually enforced, and batch-independent inference firing.
  """
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.{BatchTopKSAE, SAETrainer, SparseAutoencoder}

  @moduletag :interpretability
  @moduletag timeout: 300_000

  # Planted problem: n samples, each a positive combination of exactly
  # `active` atoms from a random unit-row dictionary {atoms, d} + noise
  defp planted_data(n, d, atoms, active, seed) do
    key = Nx.Random.key(seed)
    {dict, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {atoms, d})

    norms = dict |> Nx.pow(2) |> Nx.sum(axes: [1], keep_axes: true) |> Nx.sqrt()
    dict = Nx.divide(dict, norms)

    {raw, key} = Nx.Random.uniform(key, 0.5, 1.5, shape: {n, atoms})
    {scores, key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {n, atoms})

    # Keep each row's `active` highest-scored atoms
    {top, _} = Nx.top_k(scores, k: active)
    kth = Nx.slice_along_axis(top, active - 1, 1, axis: 1)
    codes = Nx.select(Nx.greater_equal(scores, kth), raw, 0.0)

    {noise, _key} = Nx.Random.normal(key, 0.0, 0.01, shape: {n, d})
    x = codes |> Nx.dot(dict) |> Nx.add(noise)

    {x, dict, codes}
  end

  defp mse(a, b), do: a |> Nx.subtract(b) |> Nx.pow(2) |> Nx.mean() |> Nx.to_number()

  describe "fit/3 — planted dictionary recovery (BatchTopK)" do
    test "recovers planted atoms above chance and beats the mean baseline" do
      n = 400
      {x, dict, _codes} = planted_data(n, 16, 8, 2, 0)

      result =
        SAETrainer.fit(:batch_top_k_sae, x,
          dict_size: 16,
          batch_k: n * 2,
          steps: 250,
          lr: 5.0e-2,
          l1_coeff: 1.0e-4
        )

      # Learning happened
      assert List.last(result.history) < hd(result.history) * 0.5

      # Reconstruction beats predicting the per-dim mean
      out = predict_container(result, x)
      mean_baseline = Nx.broadcast(Nx.mean(x, axes: [0]), Nx.shape(x))
      assert mse(x, out.reconstruction) < mse(x, mean_baseline) * 0.5

      # Dictionary recovery: for each planted atom, the best-matching
      # learned decoder row (both unit-norm) must align well above chance
      # (E|cos| for random unit vectors in R^16 is ~0.2)
      decoder = result.params.data["batch_topk_sae_decoder"]["kernel"]
      cos = Nx.dot(dict, Nx.transpose(decoder)) |> Nx.abs()
      best = Nx.reduce_max(cos, axes: [1])
      mean_best = best |> Nx.mean() |> Nx.to_number()

      assert mean_best > 0.5,
             "mean best-match |cos| #{mean_best} — dictionary not recovered above chance"

      # Fit metadata
      assert is_number(result.threshold) and result.threshold > 0.0
      assert is_integer(result.dead_count)
    end

    test "decoder rows are unit-norm after fit (the L1-degeneracy guard)" do
      {x, _dict, _codes} = planted_data(120, 12, 6, 2, 3)

      result =
        SAETrainer.fit(:batch_top_k_sae, x,
          dict_size: 12,
          batch_k: 240,
          steps: 40,
          lr: 5.0e-2
        )

      kernel = result.params.data["batch_topk_sae_decoder"]["kernel"]
      norms = kernel |> Nx.pow(2) |> Nx.sum(axes: [1]) |> Nx.sqrt()

      assert Nx.all_close(norms, Nx.broadcast(1.0, Nx.shape(norms)), atol: 1.0e-4)
             |> Nx.to_number() == 1
    end

    test "AuxK gives dead features a gradient path (smoke + finiteness)" do
      {x, _dict, _codes} = planted_data(120, 12, 6, 2, 5)

      result =
        SAETrainer.fit(:batch_top_k_sae, x,
          dict_size: 24,
          batch_k: 240,
          steps: 60,
          lr: 5.0e-2,
          aux_k: 4,
          aux_coeff: 1.0 / 32.0
        )

      assert Enum.all?(result.history, &(is_number(&1) and &1 == &1))
      assert List.last(result.history) < hd(result.history)
    end
  end

  describe "fit/3 — other modules" do
    test "plain SparseAutoencoder fits through the same trainer" do
      {x, _dict, _codes} = planted_data(150, 12, 6, 2, 7)

      result = SAETrainer.fit(:sparse_autoencoder, x, dict_size: 12, top_k: 2, steps: 80, lr: 5.0e-2)

      assert result.module == SparseAutoencoder
      assert List.last(result.history) < hd(result.history) * 0.7
    end
  end

  describe "inference_threshold — batch-independent firing" do
    test "a sample's firing set is identical alone vs inside a batch" do
      n = 200
      {x, _dict, _codes} = planted_data(n, 16, 8, 2, 11)

      result =
        SAETrainer.fit(:batch_top_k_sae, x,
          dict_size: 16,
          batch_k: n * 2,
          steps: 150,
          lr: 5.0e-2
        )

      infer_opts = [
        input_size: 16,
        dict_size: 16,
        inference_threshold: result.threshold,
        output: :container
      ]

      model = BatchTopKSAE.build(infer_opts)
      {_, predict_fn} = Axon.build(model, mode: :inference)

      sample = Nx.slice_along_axis(x, 0, 1, axis: 0)
      batch = Nx.slice_along_axis(x, 0, 10, axis: 0)

      solo = predict_fn.(result.params, %{"batch_topk_sae_input" => sample})
      in_batch = predict_fn.(result.params, %{"batch_topk_sae_input" => batch})

      solo_hidden = solo.hidden
      batch_row = Nx.slice_along_axis(in_batch.hidden, 0, 1, axis: 0)

      # Bitwise: the fixed threshold makes firing a pure per-sample function
      assert Nx.to_binary(solo_hidden) == Nx.to_binary(batch_row)

      # And WITHOUT the fixed threshold, batch top-k firing genuinely
      # depends on batch composition (the failure mode being fixed) —
      # verified by the k-th value differing between the two runs
      topk_model = BatchTopKSAE.build(input_size: 16, dict_size: 16, batch_k: 4, output: :container)
      {_, topk_predict} = Axon.build(topk_model, mode: :inference)

      solo_tk = topk_predict.(result.params, %{"batch_topk_sae_input" => sample})
      batch_tk = topk_predict.(result.params, %{"batch_topk_sae_input" => batch})
      batch_tk_row = Nx.slice_along_axis(batch_tk.hidden, 0, 1, axis: 0)

      refute Nx.to_binary(solo_tk.hidden) == Nx.to_binary(batch_tk_row)
    end
  end

  defp predict_container(result, x) do
    {_, predict_fn} = Axon.build(result.model, mode: :inference)
    input_key = result.model |> Axon.get_inputs() |> Map.keys() |> hd()
    predict_fn.(result.params, %{input_key => x})
  end
end
