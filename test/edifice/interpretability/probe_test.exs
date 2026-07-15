defmodule Edifice.Interpretability.ProbeTest do
  @moduledoc """
  Functional guarantees for the linear probing harness (LEACE-standard
  tests: plant a signal, verify the method's contract — not shapes).
  """
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.Probe

  @moduletag :interpretability

  # Planted linearly-separable data: y = (x · v > 0), fully recoverable
  # by any sound linear probe
  defp separable_data(n, d, seed) do
    key = Nx.Random.key(seed)
    {x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n, d})
    {v, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {d})

    y = x |> Nx.dot(v) |> Nx.greater(0.0) |> Nx.as_type(:s64)
    {x, y}
  end

  describe "fit_eval/6 — recovery guarantees" do
    test "recovers a planted linearly separable signal" do
      {x, y} = separable_data(400, 8, 0)
      {x_eval, y_eval} = separable_data(200, 8, 0)

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)

      # Same generating hyperplane in train and eval → near-perfect probe.
      # 0.95 leaves slack for the few near-boundary points at n=200.
      assert result.balanced_accuracy > 0.95
      assert result.accuracy > 0.95
      assert result.n_train == 400
      assert result.n_eval == 200
      assert is_number(result.train_ms)
    end

    test "class weighting recovers the rare class on imbalanced labels" do
      # ~5% positive rate: y = (x0 > 1.65). Unweighted CE would happily
      # predict all-negative (accuracy 95%, rare-class recall 0)
      key = Nx.Random.key(3)
      {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {2_000, 6})
      y = x |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])
      y = Nx.greater(y, 1.65) |> Nx.as_type(:s64)

      {x_eval, x} = {Nx.slice_along_axis(x, 0, 600, axis: 0), Nx.slice_along_axis(x, 600, 1_400, axis: 0)}
      {y_eval, y} = {Nx.slice_along_axis(y, 0, 600, axis: 0), Nx.slice_along_axis(y, 600, 1_400, axis: 0)}

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)

      [neg_recall, pos_recall] = result.per_class_recall
      assert pos_recall > 0.8, "rare-class recall #{pos_recall} — class weighting failed"
      assert neg_recall > 0.8
      assert result.balanced_accuracy > 0.8

      # Majority baseline for 2 present classes is 0.5 — the probe must
      # clearly beat the number it is reported next to
      assert result.majority_baseline == 0.5
      assert result.balanced_accuracy > result.majority_baseline + 0.25
    end

    test "shuffled-label control scores ~chance (Hewitt-Liang)" do
      # d=32: a random direction correlates with the true hyperplane by
      # ~1/sqrt(d) — at low d a single seed can score 0.7 by pure chance.
      # Proper train/eval split (disjoint rows, same generating hyperplane).
      {x_all, y_all} = separable_data(650, 32, 5)
      x = Nx.slice_along_axis(x_all, 0, 500, axis: 0)
      y = Nx.slice_along_axis(y_all, 0, 500, axis: 0)
      x_eval = Nx.slice_along_axis(x_all, 500, 150, axis: 0)
      y_eval = Nx.slice_along_axis(y_all, 500, 150, axis: 0)

      result = Probe.shuffled_control(x, y, x_eval, y_eval, 2)

      # Chance for 2 balanced classes is 0.5; anything well above means the
      # probe itself (not the representation) is doing the work
      assert result.balanced_accuracy < 0.65,
             "shuffled control scored #{result.balanced_accuracy} — probe setup leaks"

      # Sanity: the same split with TRUE labels is near-perfect, so the
      # control's ~chance score reflects the shuffling, not a broken probe
      true_result = Probe.fit_eval(x, y, x_eval, y_eval, 2)
      assert true_result.balanced_accuracy > 0.9
    end
  end

  describe "fit_eval/6 — hygiene guards" do
    test "rows labeled -1 are masked out of both splits" do
      {x, y} = separable_data(100, 4, 8)
      # Poison 30 train labels with -1 (not applicable)
      y = Nx.put_slice(y, [0], Nx.broadcast(Nx.tensor(-1, type: :s64), {30}))

      {x_eval, y_eval} = separable_data(50, 4, 8)

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)

      assert result.n_train == 70
      assert result.n_eval == 50
      # Masking must not break recovery
      assert result.balanced_accuracy > 0.9
    end

    test "single-class eval yields nil balanced accuracy (degeneracy guard)" do
      {x, y} = separable_data(100, 4, 9)
      {x_eval, _} = separable_data(50, 4, 9)
      y_eval = Nx.broadcast(Nx.tensor(1, type: :s64), {50})

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)

      # Predicting the only class would score 1.0 — must be nil, not 1.0
      assert result.balanced_accuracy == nil
      assert result.majority_baseline == 1.0
    end

    test "empty split after masking returns the nil result map" do
      {x, _y} = separable_data(20, 4, 10)
      y = Nx.broadcast(Nx.tensor(-1, type: :s64), {20})
      {x_eval, y_eval} = separable_data(10, 4, 10)

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)

      assert result.n_train == 0
      assert result.balanced_accuracy == nil
      assert result.params == nil
    end
  end

  describe "predict/3 and aggregation" do
    test "predict with fitted params reproduces eval predictions" do
      {x, y} = separable_data(300, 6, 11)
      {x_eval, y_eval} = separable_data(100, 6, 11)

      result = Probe.fit_eval(x, y, x_eval, y_eval, 2)
      pred = Probe.predict(result.params, x_eval)

      acc = Nx.mean(Nx.equal(pred, y_eval)) |> Nx.to_number()
      assert_in_delta acc, result.accuracy, 1.0e-6
    end

    test "mean_balanced_accuracy is nil-safe over lists and maps" do
      results = [
        %{balanced_accuracy: 0.9},
        %{balanced_accuracy: nil},
        %{balanced_accuracy: 0.7}
      ]

      assert_in_delta Probe.mean_balanced_accuracy(results), 0.8, 1.0e-9

      as_map = %{a: %{balanced_accuracy: 1.0}, b: %{balanced_accuracy: 0.5}}
      assert_in_delta Probe.mean_balanced_accuracy(as_map), 0.75, 1.0e-9
    end
  end
end
