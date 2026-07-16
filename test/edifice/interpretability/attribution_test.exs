defmodule Edifice.Interpretability.AttributionTest do
  @moduledoc """
  Functional guarantees for gradient×input attribution: a model that only
  reads certain input dims must attribute (almost) all saliency to them.
  """
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.Attribution

  @moduletag :interpretability

  # Linear "model" reading ONLY dims 0 and 1 of a 4-dim input
  defp two_dim_model do
    w =
      Nx.tensor([
        [1.0, -1.0],
        [2.0, 0.5],
        [0.0, 0.0],
        [0.0, 0.0]
      ])

    predict_fn = fn params, x -> Nx.dot(x, params.w) end
    {predict_fn, %{w: w}}
  end

  describe "saliency/4" do
    test "attributes only to the dims the model actually reads" do
      {predict_fn, params} = two_dim_model()
      key = Nx.Random.key(0)
      {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {16, 4})

      sal = Attribution.saliency(predict_fn, params, x)

      assert Nx.shape(sal) == {16, 4}

      dead = sal |> Nx.slice_along_axis(2, 2, axis: 1) |> Nx.reduce_max() |> Nx.to_number()
      live = sal |> Nx.slice_along_axis(0, 2, axis: 1) |> Nx.reduce_max() |> Nx.to_number()

      # Dims 2/3 have zero weight → exactly zero gradient
      assert dead < 1.0e-7
      assert live > 0.0
    end

    test "sums saliency over the window axis for rank-3 inputs" do
      w = Nx.tensor([[1.0, -1.0], [0.0, 2.0], [0.0, 0.0]])
      predict_fn = fn params, s -> s |> Nx.sum(axes: [1]) |> Nx.dot(params.w) end

      key = Nx.Random.key(1)
      {s, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {5, 7, 3})

      sal = Attribution.saliency(predict_fn, %{w: w}, s)

      assert Nx.shape(sal) == {5, 3}
      dead = sal |> Nx.slice_along_axis(2, 1, axis: 1) |> Nx.reduce_max() |> Nx.to_number()
      assert dead < 1.0e-7
    end

    test ":select picks the logits out of a multi-head output" do
      w_a = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      w_b = Nx.tensor([[0.0, 5.0], [5.0, 0.0]])
      predict_fn = fn params, x -> {Nx.dot(x, params.a), Nx.dot(x, params.b)} end
      params = %{a: w_a, b: w_b}

      key = Nx.Random.key(2)
      {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {8, 2})

      sal_a = Attribution.saliency(predict_fn, params, x, select: &elem(&1, 0))
      sal_b = Attribution.saliency(predict_fn, params, x, select: &elem(&1, 1))

      # Head b's weights are 5x head a's → clearly different attributions
      ratio =
        Nx.divide(Nx.sum(sal_b), Nx.max(Nx.sum(sal_a), 1.0e-9)) |> Nx.to_number()

      assert ratio > 2.0
    end
  end

  describe "group_shares/2" do
    test "shares over a partition sum to 1, dead groups get 0" do
      {predict_fn, params} = two_dim_model()
      key = Nx.Random.key(3)
      {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {10, 4})

      sal = Attribution.saliency(predict_fn, params, x)

      shares =
        Attribution.group_shares(sal, %{live: [0, 1], dead: [2, 3], empty: []})

      total = Nx.add(shares.live, shares.dead)
      assert Nx.all_close(total, Nx.broadcast(1.0, {10})) |> Nx.to_number() == 1

      assert Nx.reduce_max(shares.dead) |> Nx.to_number() < 1.0e-7
      assert Nx.reduce_max(shares.empty) |> Nx.to_number() == 0.0

      live_min = Nx.reduce_min(shares.live) |> Nx.to_number()
      assert live_min > 0.99
    end
  end
end
