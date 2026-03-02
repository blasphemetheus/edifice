defmodule Edifice.CUDA.AttentionBackwardTest do
  @moduledoc """
  Gradient correctness tests for attention backward kernels.

  Tests run on BinaryBackend (no GPU required) — they validate the Elixir
  backward fallback functions against Nx autodiff of the forward function.

  Tests for all 3 variants:
  - Flash attention (standard)
  - LASER attention
  - FoX attention
  """
  use ExUnit.Case, async: true

  # ============================================================================
  # Flash Attention backward tests
  # ============================================================================

  describe "flash_attention_backward_fallback" do
    test "gradient shapes match inputs for small seq" do
      batch = 1
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.flash_attention_backward_fallback(q, k, v, 0, grad_output)

      assert Nx.shape(grad_q) == {batch, heads, seq, dim}
      assert Nx.shape(grad_k) == {batch, heads, seq, dim}
      assert Nx.shape(grad_v) == {batch, heads, seq, dim}
    end

    test "non-causal gradients match Nx autodiff" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(123)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})

      # Reference: autodiff of fallback forward
      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          Edifice.CUDA.FusedScan.flash_attention_fallback(q, k, v, 0) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.flash_attention_backward_fallback(q, k, v, 0, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-4)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-4)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-4)
    end

    test "causal gradients match Nx autodiff" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(456)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          Edifice.CUDA.FusedScan.flash_attention_fallback(q, k, v, 1) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.flash_attention_backward_fallback(q, k, v, 1, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-4)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-4)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-4)
    end

    test "batch=2 non-causal" do
      batch = 2
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(789)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          Edifice.CUDA.FusedScan.flash_attention_fallback(q, k, v, 0) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.flash_attention_backward_fallback(q, k, v, 0, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-4)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-4)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # LASER Attention backward tests
  # ============================================================================

  describe "laser_attention_backward_fallback" do
    test "gradient shapes match inputs" do
      batch = 1
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})

      v_max = Nx.reduce_max(v, axes: [2], keep_axes: true)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.laser_attention_backward_fallback(q, k, v, v_max, 1, grad_output)

      assert Nx.shape(grad_q) == {batch, heads, seq, dim}
      assert Nx.shape(grad_k) == {batch, heads, seq, dim}
      assert Nx.shape(grad_v) == {batch, heads, seq, dim}
    end

    test "causal gradients match Nx autodiff" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(321)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})

      v_max = Nx.reduce_max(v, axes: [2], keep_axes: true)

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          vm = Nx.reduce_max(v, axes: [2], keep_axes: true)
          Edifice.CUDA.FusedScan.laser_attention_fallback(q, k, v, vm, 1) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.laser_attention_backward_fallback(q, k, v, v_max, 1, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-3)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-3)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-3)
    end

    test "batch=2 causal" do
      batch = 2
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(654)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})

      v_max = Nx.reduce_max(v, axes: [2], keep_axes: true)

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          vm = Nx.reduce_max(v, axes: [2], keep_axes: true)
          Edifice.CUDA.FusedScan.laser_attention_fallback(q, k, v, vm, 1) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v} =
        Edifice.CUDA.FusedScan.laser_attention_backward_fallback(q, k, v, v_max, 1, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-3)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-3)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-3)
    end
  end

  # ============================================================================
  # FoX Attention backward tests
  # ============================================================================

  describe "fox_attention_backward_fallback" do
    test "gradient shapes match inputs" do
      batch = 1
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {cs, _key} = Nx.Random.uniform(key, -2.0, 0.0, shape: {batch, heads, seq}, type: {:f, 32})

      # Make cs monotonically decreasing (cumulative log-forget)
      cs = Nx.cumulative_sum(cs, axis: 2)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v, grad_cs} =
        Edifice.CUDA.FusedScan.fox_attention_backward_fallback(q, k, v, cs, grad_output)

      assert Nx.shape(grad_q) == {batch, heads, seq, dim}
      assert Nx.shape(grad_k) == {batch, heads, seq, dim}
      assert Nx.shape(grad_v) == {batch, heads, seq, dim}
      assert Nx.shape(grad_cs) == {batch, heads, seq}
    end

    test "dQ/dK/dV gradients match Nx autodiff" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(111)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {cs_raw, _key} = Nx.Random.uniform(key, -1.0, 0.0, shape: {batch, heads, seq}, type: {:f, 32})
      cs = Nx.cumulative_sum(cs_raw, axis: 2)

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          Edifice.CUDA.FusedScan.fox_attention_fallback(q, k, v, cs) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v, _grad_cs} =
        Edifice.CUDA.FusedScan.fox_attention_backward_fallback(q, k, v, cs, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-4)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-4)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-4)
    end

    test "grad_cs matches Nx autodiff" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(222)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {cs_raw, _key} = Nx.Random.uniform(key, -1.0, 0.0, shape: {batch, heads, seq}, type: {:f, 32})
      cs = Nx.cumulative_sum(cs_raw, axis: 2)

      ref_gcs =
        Nx.Defn.grad(cs, fn cs ->
          Edifice.CUDA.FusedScan.fox_attention_fallback(q, k, v, cs) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {_grad_q, _grad_k, _grad_v, grad_cs} =
        Edifice.CUDA.FusedScan.fox_attention_backward_fallback(q, k, v, cs, grad_output)

      assert_all_close(grad_cs, ref_gcs, atol: 1.0e-4)
    end

    test "batch=2" do
      batch = 2
      heads = 2
      seq = 8
      dim = 4

      key = Nx.Random.key(333)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {cs_raw, _key} = Nx.Random.uniform(key, -1.0, 0.0, shape: {batch, heads, seq}, type: {:f, 32})
      cs = Nx.cumulative_sum(cs_raw, axis: 2)

      {ref_gq, ref_gk, ref_gv} =
        Nx.Defn.grad({q, k, v}, fn {q, k, v} ->
          Edifice.CUDA.FusedScan.fox_attention_fallback(q, k, v, cs) |> Nx.sum()
        end)

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, heads, seq, dim})

      {grad_q, grad_k, grad_v, _grad_cs} =
        Edifice.CUDA.FusedScan.fox_attention_backward_fallback(q, k, v, cs, grad_output)

      assert_all_close(grad_q, ref_gq, atol: 1.0e-4)
      assert_all_close(grad_k, ref_gk, atol: 1.0e-4)
      assert_all_close(grad_v, ref_gv, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp assert_all_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    diff = Nx.abs(Nx.subtract(actual, expected))
    max_diff = Nx.to_number(Nx.reduce_max(diff))
    assert max_diff < atol, "max diff #{max_diff} exceeds tolerance #{atol}"
  end
end
