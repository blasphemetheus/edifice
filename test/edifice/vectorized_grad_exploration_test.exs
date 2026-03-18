defmodule Edifice.VectorizedGradExplorationTest do
  @moduledoc """
  Exploration tests for Nx vectorized gradients (PR #1697).

  Documents which patterns work and which hit known limitations.
  These tests validate our fork's vectorized-inside-grad support
  and track when limitations are resolved.

  Run with: EDIFICE_LOCAL_NX=1 mix test test/edifice/vectorized_grad_exploration_test.exs
  """
  use ExUnit.Case, async: true

  import Nx.Defn

  # Helper: compare vectorized tensors by devectorizing first
  defp assert_vectorized_close(result, expected, opts \\ []) do
    r = Nx.devectorize(result, keep_names: false)
    e = Nx.devectorize(expected, keep_names: false)
    assert Nx.all_close(r, e, opts) |> Nx.to_number() == 1
  end

  # ============================================================================
  # WORKING: Elementwise ops with vectorized inputs
  # ============================================================================

  describe "working: elementwise vectorized grad" do
    defn grad_sum_square(x) do
      grad(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
    end

    test "grad of sum(x*x) with vectorized input" do
      x = Nx.iota({3}, vectorized_axes: [batch: 2])
      result = grad_sum_square(x)

      assert result.vectorized_axes == [batch: 2]
      assert Nx.shape(result) == {3}
      # d/dx sum(x^2) = 2x
      expected = Nx.multiply(x, 2)
      assert_vectorized_close(result, expected)
    end

    defn grad_sum_exp(x) do
      grad(x, fn x -> Nx.sum(Nx.exp(x)) end)
    end

    test "grad of sum(exp(x)) with vectorized input" do
      x = Nx.iota({3}, vectorized_axes: [batch: 2])
      result = grad_sum_exp(x)
      expected = Nx.exp(x)
      assert_vectorized_close(result, expected, atol: 1.0e-5)
    end
  end

  # ============================================================================
  # WORKING: Reduce ops with vectorized inputs
  # ============================================================================

  describe "working: reduce vectorized grad" do
    defn grad_sum(x) do
      grad(x, fn x -> Nx.sum(x) end)
    end

    test "grad of sum(x) with vectorized input is all ones" do
      x = Nx.iota({4}, vectorized_axes: [batch: 3])
      result = grad_sum(x)
      assert result.vectorized_axes == [batch: 3]
      expected = Nx.broadcast(1.0, x)
      assert_vectorized_close(result, expected)
    end
  end

  # ============================================================================
  # WORKING: Multi-axis vectorization
  # ============================================================================

  describe "working: multi-axis vectorized grad" do
    defn grad_multi_axis(x) do
      grad(x, fn x -> Nx.sum(Nx.pow(x, 2)) end)
    end

    test "grad with multiple vectorized axes" do
      x = Nx.iota({2}, vectorized_axes: [a: 2, b: 3])
      result = grad_multi_axis(x)
      assert result.vectorized_axes == [a: 2, b: 3]
      expected = Nx.multiply(x, 2)
      assert_vectorized_close(result, expected)
    end
  end

  # ============================================================================
  # WORKING: vectorize/devectorize inside grad
  # ============================================================================

  describe "working: vectorize/devectorize inside grad" do
    defn grad_vectorize_devectorize(x) do
      grad(x, fn x ->
        v = Nx.vectorize(Nx.reshape(x, {2, 3}), :batch)
        d = Nx.devectorize(v, keep_names: false)
        Nx.sum(Nx.multiply(d, d))
      end)
    end

    test "vectorize and devectorize inside grad" do
      x = Nx.iota({6})
      result = grad_vectorize_devectorize(x)
      # d/dx sum(x^2) = 2x
      expected = Nx.multiply(Nx.iota({6}), 2)
      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # KNOWN LIMITATION: Cross-vectorized dot product grad
  # ============================================================================

  describe "known limitation: cross-vectorized dot" do
    @tag :skip
    @tag :known_limitation
    test "per-example grad through dense layer (vectorized input × plain weight)" do
      # This is the pattern Axon.dense uses: x_vectorized @ w_plain
      # Currently fails with: "expected length of axes (5) to match rank of shape (4)"
      # Tracked in PR #1697 and upstream issue #1533
      w = Nx.tensor([[1.0, -1.0], [0.5, 0.5]])
      x_batch = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

      _result =
        Nx.Defn.grad(w, fn w ->
          x_v = Nx.vectorize(x_batch, :example)
          Nx.sum(Nx.dot(x_v, w))
        end)

      # When this works, result should have vectorized_axes: [example: 3]
      # and represent per-example gradients of w
    end
  end

  # ============================================================================
  # POTENTIAL USE CASES FOR EDIFICE (once cross-vectorized dot works)
  # ============================================================================

  # 1. Per-example gradients for differential privacy (DP-SGD):
  #    Clip per-example gradient norms, then aggregate
  #
  # 2. Influence functions:
  #    Compute per-example gradient to identify influential training examples
  #
  # 3. Curriculum learning:
  #    Weight examples by their gradient magnitude
  #
  # 4. Per-example loss weighting in contrastive learning:
  #    Different temperature per positive pair
  #
  # These all require the cross-vectorized dot product to work,
  # since they need per-example gradients through dense layers.
end
