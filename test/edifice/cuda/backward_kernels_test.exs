defmodule Edifice.CUDA.BackwardKernelsTest do
  @moduledoc """
  Gradient correctness tests for backward CUDA kernels.

  Tests run on BinaryBackend (no GPU required) — they validate the Elixir
  backward fallback functions against Nx autodiff of the forward scan.

  Test strategy:
  1. Backward fallback matches Nx autodiff (forward scan differentiated by Nx)
  2. Numerical gradient check via finite differences
  3. Various shapes (batch=1, seq=1, large hidden, large seq)
  """
  use ExUnit.Case, async: true

  # ============================================================================
  # Linear scan backward tests
  # ============================================================================

  describe "linear_scan_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {a_vals, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      # Forward scan to get output
      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)

      # Use Nx autodiff to get reference gradients
      # We compute grad of sum(forward(a,b)) w.r.t. a and b
      grad_fn = fn a, b ->
        Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
      end

      {ref_grad_a, ref_grad_b} = Nx.Defn.grad({a_vals, b_vals}, fn {a, b} -> grad_fn.(a, b) end)

      # Our backward fallback with ones gradient (since we took sum)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      {grad_a, grad_b, _grad_h0} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 1.0e-5)
      assert_all_close(grad_b, ref_grad_b, atol: 1.0e-5)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(99)
      {a_vals, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      # Finite difference for a_vals
      eps = 1.0e-3
      numerical_grad_a = finite_diff_grad(a_vals, fn a -> Edifice.CUDA.FusedScan.linear_scan_fallback(a, b_vals) |> Nx.sum() end, eps)
      numerical_grad_b = finite_diff_grad(b_vals, fn b -> Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b) |> Nx.sum() end, eps)

      assert_all_close(grad_a, numerical_grad_a, atol: 1.0e-2)
      assert_all_close(grad_b, numerical_grad_b, atol: 1.0e-2)
    end

    test "handles batch=1 and seq=1" do
      a = Nx.tensor([[[0.5]]], type: {:f, 32})
      b = Nx.tensor([[[1.0]]], type: {:f, 32})
      h0 = Nx.tensor([[0.0]], type: {:f, 32})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a, b)
      grad_output = Nx.tensor([[[1.0]]], type: {:f, 32})

      {grad_a, grad_b, grad_h0} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a, h0, forward_out, grad_output)

      # h_0 = 0, h_1 = 0.5 * 0 + 1.0 = 1.0
      # da = dh * h_{-1} = 1.0 * 0.0 = 0.0
      # db = dh = 1.0
      # dh0 = dh * a = 1.0 * 0.5 = 0.5
      assert_all_close(grad_a, Nx.tensor([[[0.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_b, Nx.tensor([[[1.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_h0, Nx.tensor([[0.5]], type: {:f, 32}), atol: 1.0e-6)
    end

    test "larger hidden dimension" do
      batch = 2
      seq_len = 8
      hidden = 32

      key = Nx.Random.key(123)
      {a_vals, key} = Nx.Random.uniform(key, 0.3, 0.7, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      {ref_grad_a, ref_grad_b} =
        Nx.Defn.grad({a_vals, b_vals}, fn {a, b} ->
          Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
        end)

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 1.0e-4)
      assert_all_close(grad_b, ref_grad_b, atol: 1.0e-4)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {a_vals, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      {ref_grad_a, ref_grad_b} =
        Nx.Defn.grad({a_vals, b_vals}, fn {a, b} ->
          Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
        end)

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 0.05)
      assert_all_close(grad_b, ref_grad_b, atol: 0.05)
    end
  end

  # ============================================================================
  # MinGRU backward tests
  # ============================================================================

  describe "mingru_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      # Forward
      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)

      # Reference grads via Nx autodiff
      {ref_grad_z, ref_grad_cand} =
        Nx.Defn.grad({z, candidates}, fn {z, cand} ->
          forward_fn.(z, cand) |> Nx.sum()
        end)

      # Our backward
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_z, grad_cand, _grad_h0} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 1.0e-5)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-5)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)
        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)
        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_z, ref_grad_cand} =
        Nx.Defn.grad({z, candidates}, fn {z, cand} -> forward_fn.(z, cand) |> Nx.sum() end)

      {grad_z, grad_cand, _} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 0.05)
      assert_all_close(grad_cand, ref_grad_cand, atol: 0.05)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(77)
      {z, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_z, grad_cand, _} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      eps = 1.0e-3
      numerical_z = finite_diff_grad(z, fn z -> forward_fn.(z, candidates) |> Nx.sum() end, eps)
      numerical_cand = finite_diff_grad(candidates, fn c -> forward_fn.(z, c) |> Nx.sum() end, eps)

      assert_all_close(grad_z, numerical_z, atol: 1.0e-2)
      assert_all_close(grad_cand, numerical_cand, atol: 1.0e-2)
    end

    test "handles seq_len=1" do
      z = Nx.tensor([[[0.3, 0.7]]], type: {:f, 32})
      cand = Nx.tensor([[[2.0, -1.0]]], type: {:f, 32})
      h0 = Nx.tensor([[0.0, 0.0]], type: {:f, 32})
      forward_out = Nx.tensor([[[0.6, -0.7]]], type: {:f, 32})
      grad_output = Nx.tensor([[[1.0, 1.0]]], type: {:f, 32})

      {grad_z, grad_cand, grad_h0} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, cand, h0, forward_out, grad_output)

      # dz = dh * (c - h_prev) = 1 * (2-0, -1-0) = (2, -1)
      # dc = dh * z = 1 * (0.3, 0.7) = (0.3, 0.7)
      # dh0 = dh * (1-z) = 1 * (0.7, 0.3) = (0.7, 0.3)
      assert_all_close(grad_z, Nx.tensor([[[2.0, -1.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_cand, Nx.tensor([[[0.3, 0.7]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_h0, Nx.tensor([[0.7, 0.3]], type: {:f, 32}), atol: 1.0e-6)
    end
  end

  # ============================================================================
  # MinLSTM backward tests
  # ============================================================================

  describe "minlstm_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {f_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      # Forward function matching the kernel's behavior
      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)

      # Reference grads
      {ref_grad_f, ref_grad_i, ref_grad_cand} =
        Nx.Defn.grad({f_gate, i_gate, candidates}, fn {f, i, cand} ->
          forward_fn.(f, i, cand) |> Nx.sum()
        end)

      # Our backward
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_f, grad_i, grad_cand, _grad_h0} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_f, ref_grad_f, atol: 1.0e-4)
      assert_all_close(grad_i, ref_grad_i, atol: 1.0e-4)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-4)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(55)
      {f_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_f, grad_i, grad_cand, _} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      eps = 1.0e-3
      numerical_f = finite_diff_grad(f_gate, fn f -> forward_fn.(f, i_gate, candidates) |> Nx.sum() end, eps)
      numerical_i = finite_diff_grad(i_gate, fn i -> forward_fn.(f_gate, i, candidates) |> Nx.sum() end, eps)
      numerical_cand = finite_diff_grad(candidates, fn c -> forward_fn.(f_gate, i_gate, c) |> Nx.sum() end, eps)

      assert_all_close(grad_f, numerical_f, atol: 1.0e-2)
      assert_all_close(grad_i, numerical_i, atol: 1.0e-2)
      assert_all_close(grad_cand, numerical_cand, atol: 1.0e-2)
    end

    test "larger shapes" do
      batch = 2
      seq_len = 8
      hidden = 16

      key = Nx.Random.key(200)
      {f_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_f, ref_grad_i, ref_grad_cand} =
        Nx.Defn.grad({f_gate, i_gate, candidates}, fn {f, i, cand} ->
          forward_fn.(f, i, cand) |> Nx.sum()
        end)

      {grad_f, grad_i, grad_cand, _} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_f, ref_grad_f, atol: 1.0e-3)
      assert_all_close(grad_i, ref_grad_i, atol: 1.0e-3)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-3)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {f_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {i_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      norm_eps = 1.0e-6

      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)
        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)
        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_f, ref_grad_i, ref_grad_cand} =
        Nx.Defn.grad({f_gate, i_gate, candidates}, fn {f, i, cand} ->
          forward_fn.(f, i, cand) |> Nx.sum()
        end)

      {grad_f, grad_i, grad_cand, _} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_f, ref_grad_f, atol: 0.05)
      assert_all_close(grad_i, ref_grad_i, atol: 0.05)
      assert_all_close(grad_cand, ref_grad_cand, atol: 0.05)
    end
  end

  # ============================================================================
  # ELU-GRU backward tests
  # ============================================================================

  describe "elu_gru_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {c, _key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn z, c ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, c)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_z, ref_grad_c} =
        Nx.Defn.grad({z, c}, fn {z, c} -> forward_fn.(z, c) |> Nx.sum() end)

      {grad_z, grad_c, _} =
        Edifice.CUDA.FusedScan.elu_gru_backward_fallback(z, c, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 1.0e-5)
      assert_all_close(grad_c, ref_grad_c, atol: 1.0e-5)
    end

    test "numerical gradient check" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(99)
      {z, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {c, _key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      forward_out = elu_gru_forward(z, c, h0)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_z, grad_c, _} =
        Edifice.CUDA.FusedScan.elu_gru_backward_fallback(z, c, h0, forward_out, grad_output)

      eps = 1.0e-3
      numer_z = finite_diff_grad(z, fn z -> elu_gru_forward(z, c, h0) |> Nx.sum() end, eps)
      numer_c = finite_diff_grad(c, fn c -> elu_gru_forward(z, c, h0) |> Nx.sum() end, eps)

      assert_all_close(grad_z, numer_z, atol: 1.0e-2)
      assert_all_close(grad_c, numer_c, atol: 1.0e-2)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {c, _key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      forward_out = elu_gru_forward(z, c, h0)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_z, ref_grad_c} =
        Nx.Defn.grad({z, c}, fn {z, c} -> elu_gru_forward(z, c, h0) |> Nx.sum() end)

      {grad_z, grad_c, _} =
        Edifice.CUDA.FusedScan.elu_gru_backward_fallback(z, c, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 0.05)
      assert_all_close(grad_c, ref_grad_c, atol: 0.05)
    end
  end

  # ============================================================================
  # Real-GRU backward tests
  # ============================================================================

  describe "real_gru_backward_fallback" do
    test "matches Nx autodiff — identical math to MinGRU" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_z, ref_grad_cand} =
        Nx.Defn.grad({z, candidates}, fn {z, c} -> forward_fn.(z, c) |> Nx.sum() end)

      {grad_z, grad_cand, _} =
        Edifice.CUDA.FusedScan.real_gru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 1.0e-5)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-5)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)
        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)
        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_z, ref_grad_cand} =
        Nx.Defn.grad({z, candidates}, fn {z, c} -> forward_fn.(z, c) |> Nx.sum() end)

      {grad_z, grad_cand, _} =
        Edifice.CUDA.FusedScan.real_gru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 0.05)
      assert_all_close(grad_cand, ref_grad_cand, atol: 0.05)
    end
  end

  # ============================================================================
  # DiagLinear backward tests
  # ============================================================================

  describe "diag_linear_backward_fallback" do
    test "matches Nx autodiff — same math as linear_scan" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {a_sig, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_sig, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_a, ref_grad_b} =
        Nx.Defn.grad({a_sig, b_vals}, fn {a, b} ->
          Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
        end)

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.diag_linear_backward_fallback(a_sig, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 1.0e-5)
      assert_all_close(grad_b, ref_grad_b, atol: 1.0e-5)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {a_sig, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_sig, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_a, ref_grad_b} =
        Nx.Defn.grad({a_sig, b_vals}, fn {a, b} ->
          Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
        end)

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.diag_linear_backward_fallback(a_sig, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 0.05)
      assert_all_close(grad_b, ref_grad_b, atol: 0.05)
    end
  end

  # ============================================================================
  # LSTM backward tests
  # ============================================================================

  describe "lstm_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 4 * hidden}, type: {:f, 32})
      {r_weight, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 4 * hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn wx, r ->
        lstm_forward(wx, r, h0, c0, hidden)
      end

      forward_out = forward_fn.(wx, r_weight)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_wx, ref_grad_r} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> forward_fn.(w, r) |> Nx.sum() end)

      {grad_wx, _grad_h0, _grad_c0} =
        Edifice.CUDA.FusedScan.lstm_backward_fallback(wx, r_weight, h0, c0, forward_out, grad_output)

      # Compute grad_R in the same way fused_scan.ex does
      h_prev = Nx.concatenate([Nx.reshape(h0, {batch, 1, hidden}), Nx.slice_along_axis(forward_out, 0, seq_len - 1, axis: 1)], axis: 1)
      grad_r = Nx.dot(Nx.reshape(h_prev, {batch * seq_len, hidden}) |> Nx.transpose(),
                       Nx.reshape(grad_wx, {batch * seq_len, 4 * hidden}))

      assert_all_close(grad_wx, ref_grad_wx, atol: 1.0e-4)
      assert_all_close(grad_r, ref_grad_r, atol: 1.0e-4)
    end

    test "numerical gradient check for wx" do
      batch = 1
      seq_len = 2
      hidden = 2

      key = Nx.Random.key(99)
      {wx, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, 4 * hidden}, type: {:f, 32})
      {r_weight, _key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {hidden, 4 * hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_out = lstm_forward(wx, r_weight, h0, c0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.lstm_backward_fallback(wx, r_weight, h0, c0, forward_out, grad_output)

      eps = 1.0e-3
      numer_wx = finite_diff_grad(wx, fn w -> lstm_forward(w, r_weight, h0, c0, hidden) |> Nx.sum() end, eps)

      assert_all_close(grad_wx, numer_wx, atol: 1.0e-2)
    end

    test "matches Nx autodiff for bf16 wx inputs" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 4 * hidden}, type: {:bf, 16})
      {r_weight, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 4 * hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      forward_out = lstm_forward(wx, r_weight, h0, c0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_wx, _ref_grad_r} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> lstm_forward(w, r, h0, c0, hidden) |> Nx.sum() end)

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.lstm_backward_fallback(wx, r_weight, h0, c0, forward_out, grad_output)

      assert_all_close(grad_wx, ref_grad_wx, atol: 0.1)
    end
  end

  # ============================================================================
  # GRU backward tests
  # ============================================================================

  describe "gru_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 3 * hidden}, type: {:f, 32})
      {r_weight, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 3 * hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn wx, r ->
        gru_forward(wx, r, h0, hidden)
      end

      forward_out = forward_fn.(wx, r_weight)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_wx, ref_grad_r} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> forward_fn.(w, r) |> Nx.sum() end)

      {grad_wx, grad_rh, _grad_h0} =
        Edifice.CUDA.FusedScan.gru_backward_fallback(wx, r_weight, h0, nil, forward_out, grad_output)

      # Compute grad_R from grad_rh
      h_prev = Nx.concatenate([Nx.reshape(h0, {batch, 1, hidden}), Nx.slice_along_axis(forward_out, 0, seq_len - 1, axis: 1)], axis: 1)
      grad_r = Nx.dot(Nx.reshape(h_prev, {batch * seq_len, hidden}) |> Nx.transpose(),
                       Nx.reshape(grad_rh, {batch * seq_len, 3 * hidden}))

      assert_all_close(grad_wx, ref_grad_wx, atol: 1.0e-4)
      assert_all_close(grad_r, ref_grad_r, atol: 1.0e-4)
    end

    test "gru_scan/4 forward (fallback tier) matches reference with h0 + bhn" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(11)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 3 * hidden}, type: {:f, 32})
      {r_weight, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 3 * hidden}, type: {:f, 32})
      {bhn, key} = Nx.Random.uniform(key, -0.4, 0.4, shape: {hidden}, type: {:f, 32})
      {h0, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, hidden}, type: {:f, 32})

      out = Edifice.CUDA.FusedScan.gru_scan(wx, r_weight, h0, bhn)
      ref = gru_forward(wx, r_weight, h0, hidden, bhn)

      assert_all_close(out, ref, atol: 1.0e-5)
    end

    test "matches Nx autodiff with nonzero bhn and h0 (BPTT carry semantics)" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(7)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 3 * hidden}, type: {:f, 32})
      {r_weight, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 3 * hidden}, type: {:f, 32})
      {bhn, key} = Nx.Random.uniform(key, -0.4, 0.4, shape: {hidden}, type: {:f, 32})
      {h0, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, hidden}, type: {:f, 32})

      forward_out = gru_forward(wx, r_weight, h0, hidden, bhn)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_wx, ref_grad_r, ref_grad_bhn} =
        Nx.Defn.grad({wx, r_weight, bhn}, fn {w, r, b} ->
          gru_forward(w, r, h0, hidden, b) |> Nx.sum()
        end)

      {grad_wx, grad_rh, _grad_h0} =
        Edifice.CUDA.FusedScan.gru_backward_fallback(wx, r_weight, h0, bhn, forward_out, grad_output)

      h_prev = Nx.concatenate([Nx.reshape(h0, {batch, 1, hidden}), Nx.slice_along_axis(forward_out, 0, seq_len - 1, axis: 1)], axis: 1)
      grad_r = Nx.dot(Nx.reshape(h_prev, {batch * seq_len, hidden}) |> Nx.transpose(),
                       Nx.reshape(grad_rh, {batch * seq_len, 3 * hidden}))

      # grad_bhn = sum over (b, t) of grad_rh's n-column
      grad_bhn =
        grad_rh
        |> Nx.slice_along_axis(2 * hidden, hidden, axis: 2)
        |> Nx.sum(axes: [0, 1])

      assert_all_close(grad_wx, ref_grad_wx, atol: 1.0e-4)
      assert_all_close(grad_r, ref_grad_r, atol: 1.0e-4)
      assert_all_close(grad_bhn, ref_grad_bhn, atol: 1.0e-4)
    end

    test "numerical gradient check for wx" do
      batch = 1
      seq_len = 2
      hidden = 2

      key = Nx.Random.key(99)
      {wx, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, 3 * hidden}, type: {:f, 32})
      {r_weight, _key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {hidden, 3 * hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_out = gru_forward(wx, r_weight, h0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.gru_backward_fallback(wx, r_weight, h0, nil, forward_out, grad_output)

      eps = 1.0e-3
      numer_wx = finite_diff_grad(wx, fn w -> gru_forward(w, r_weight, h0, hidden) |> Nx.sum() end, eps)

      assert_all_close(grad_wx, numer_wx, atol: 1.0e-2)
    end

    test "matches Nx autodiff for bf16 wx inputs" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, 3 * hidden}, type: {:bf, 16})
      {r_weight, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, 3 * hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      forward_out = gru_forward(wx, r_weight, h0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_wx, _ref_grad_r} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> gru_forward(w, r, h0, hidden) |> Nx.sum() end)

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.gru_backward_fallback(wx, r_weight, h0, nil, forward_out, grad_output)

      assert_all_close(grad_wx, ref_grad_wx, atol: 0.1)
    end
  end

  # ============================================================================
  # Liquid backward tests
  # ============================================================================

  describe "liquid_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {tau, key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {activation, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      forward_out = liquid_forward(tau, activation, h0)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_tau, ref_grad_act} =
        Nx.Defn.grad({tau, activation}, fn {t, a} -> liquid_forward(t, a, h0) |> Nx.sum() end)

      {grad_tau, grad_act, _} =
        Edifice.CUDA.FusedScan.liquid_backward_fallback(tau, activation, h0, forward_out, grad_output)

      assert_all_close(grad_tau, ref_grad_tau, atol: 1.0e-4)
      assert_all_close(grad_act, ref_grad_act, atol: 1.0e-4)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {tau, key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {activation, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      forward_out = liquid_forward(tau, activation, h0)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_tau, ref_grad_act} =
        Nx.Defn.grad({tau, activation}, fn {t, a} -> liquid_forward(t, a, h0) |> Nx.sum() end)

      {grad_tau, grad_act, _} =
        Edifice.CUDA.FusedScan.liquid_backward_fallback(tau, activation, h0, forward_out, grad_output)

      assert_all_close(grad_tau, ref_grad_tau, atol: 0.1)
      assert_all_close(grad_act, ref_grad_act, atol: 0.05)
    end
  end

  # ============================================================================
  # DeltaNet backward tests
  # ============================================================================

  describe "delta_rule_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {beta, _key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})

      forward_fn = fn q, k, v, beta -> delta_net_forward(q, k, v, beta) end

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, ref_gk, ref_gv, ref_gbeta} =
        Nx.Defn.grad({q, k, v, beta}, fn {q, k, v, b} -> forward_fn.(q, k, v, b) |> Nx.sum() end)

      {gq, gk, gv, gbeta} =
        Edifice.CUDA.FusedScan.delta_rule_backward_fallback(q, k, v, beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 1.0e-3)
      assert_all_close(gk, ref_gk, atol: 1.0e-3)
      assert_all_close(gv, ref_gv, atol: 1.0e-3)
      assert_all_close(gbeta, ref_gbeta, atol: 1.0e-3)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {beta, _key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, ref_gk, ref_gv, ref_gbeta} =
        Nx.Defn.grad({q, k, v, beta}, fn {q, k, v, b} -> delta_net_forward(q, k, v, b) |> Nx.sum() end)

      {gq, gk, gv, gbeta} =
        Edifice.CUDA.FusedScan.delta_rule_backward_fallback(q, k, v, beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 0.1)
      assert_all_close(gk, ref_gk, atol: 0.1)
      assert_all_close(gv, ref_gv, atol: 0.1)
      assert_all_close(gbeta, ref_gbeta, atol: 0.1)
    end
  end

  # ============================================================================
  # GatedDeltaNet backward tests
  # ============================================================================

  describe "gated_delta_net_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {beta, key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {alpha, _key} = Nx.Random.uniform(key, 0.8, 0.99, shape: {batch, seq_len, num_heads}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, ref_gk, ref_gv, ref_gbeta, ref_galpha} =
        Nx.Defn.grad({q, k, v, beta, alpha}, fn {q, k, v, b, a} ->
          gated_delta_net_forward(q, k, v, b, a) |> Nx.sum()
        end)

      {gq, gk, gv, gbeta, galpha} =
        Edifice.CUDA.FusedScan.gated_delta_net_backward_fallback(q, k, v, beta, alpha, grad_output)

      assert_all_close(gq, ref_gq, atol: 1.0e-3)
      assert_all_close(gk, ref_gk, atol: 1.0e-3)
      assert_all_close(gv, ref_gv, atol: 1.0e-3)
      assert_all_close(gbeta, ref_gbeta, atol: 1.0e-3)
      assert_all_close(galpha, ref_galpha, atol: 1.0e-3)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {beta, key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {alpha, _key} = Nx.Random.uniform(key, 0.8, 0.99, shape: {batch, seq_len, num_heads}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, ref_gk, ref_gv, ref_gbeta, ref_galpha} =
        Nx.Defn.grad({q, k, v, beta, alpha}, fn {q, k, v, b, a} ->
          gated_delta_net_forward(q, k, v, b, a) |> Nx.sum()
        end)

      {gq, gk, gv, gbeta, galpha} =
        Edifice.CUDA.FusedScan.gated_delta_net_backward_fallback(q, k, v, beta, alpha, grad_output)

      assert_all_close(gq, ref_gq, atol: 0.1)
      assert_all_close(gk, ref_gk, atol: 0.1)
      assert_all_close(gv, ref_gv, atol: 0.1)
      assert_all_close(gbeta, ref_gbeta, atol: 0.1)
      assert_all_close(galpha, ref_galpha, atol: 0.1)
    end
  end

  # ============================================================================
  # Selective scan backward tests
  # ============================================================================

  describe "selective_scan_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 1
      seq_len = 3
      hidden = 4
      state_size = 2

      key = Nx.Random.key(42)
      {x, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {dt, key} = Nx.Random.uniform(key, 0.01, 0.05, shape: {batch, seq_len, hidden}, type: {:f, 32})
      a = Nx.broadcast(Nx.tensor(-0.5, type: {:f, 32}), {hidden, state_size})
      {b, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, state_size}, type: {:f, 32})
      {c, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, state_size}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_gx, ref_gdt, ref_gb, ref_gc} =
        Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
          selective_scan_forward(x, dt, a, b, c) |> Nx.sum()
        end)

      {gx, gdt, gb, gc} =
        Edifice.CUDA.FusedScan.selective_scan_backward_fallback(x, dt, a, b, c, grad_output)

      assert_all_close(gx, ref_gx, atol: 1.0e-3)
      assert_all_close(gdt, ref_gdt, atol: 0.15)
      assert_all_close(gb, ref_gb, atol: 1.0e-3)
      assert_all_close(gc, ref_gc, atol: 1.0e-3)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 1
      seq_len = 3
      hidden = 4
      state_size = 2

      key = Nx.Random.key(42)
      {x, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      {dt, key} = Nx.Random.uniform(key, 0.01, 0.05, shape: {batch, seq_len, hidden}, type: {:bf, 16})
      a = Nx.broadcast(Nx.tensor(-0.5, type: {:bf, 16}), {hidden, state_size})
      {b, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, state_size}, type: {:bf, 16})
      {c, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, state_size}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_gx, ref_gdt, ref_gb, ref_gc} =
        Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
          selective_scan_forward(x, dt, a, b, c) |> Nx.sum()
        end)

      {gx, gdt, gb, gc} =
        Edifice.CUDA.FusedScan.selective_scan_backward_fallback(x, dt, a, b, c, grad_output)

      assert_all_close(gx, ref_gx, atol: 0.1)
      assert_all_close(gdt, ref_gdt, atol: 0.25)
      assert_all_close(gb, ref_gb, atol: 0.1)
      assert_all_close(gc, ref_gc, atol: 0.1)
    end
  end

  # ============================================================================
  # KDA backward tests
  # ============================================================================

  describe "kda_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {alpha, key} = Nx.Random.uniform(key, -0.5, -0.1, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {beta, _key} = Nx.Random.uniform(key, 0.3, 0.7, shape: {batch, seq_len, num_heads}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, ref_gk, ref_gv, ref_galpha, ref_gbeta} =
        Nx.Defn.grad({q, k, v, alpha, beta}, fn {q, k, v, a, b} ->
          kda_forward(q, k, v, a, b) |> Nx.sum()
        end)

      {gq, gk, gv, galpha, gbeta} =
        Edifice.CUDA.FusedScan.kda_backward_fallback(q, k, v, alpha, beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 1.0e-3)
      assert_all_close(gv, ref_gv, atol: 1.0e-3)
      assert_all_close(gbeta, ref_gbeta, atol: 1.0e-2)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {alpha, key} = Nx.Random.uniform(key, -0.5, -0.1, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {beta, _key} = Nx.Random.uniform(key, 0.3, 0.7, shape: {batch, seq_len, num_heads}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, num_heads, head_dim})

      {ref_gq, _ref_gk, ref_gv, _ref_galpha, ref_gbeta} =
        Nx.Defn.grad({q, k, v, alpha, beta}, fn {q, k, v, a, b} ->
          kda_forward(q, k, v, a, b) |> Nx.sum()
        end)

      {gq, _gk, gv, _galpha, gbeta} =
        Edifice.CUDA.FusedScan.kda_backward_fallback(q, k, v, alpha, beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 0.15)
      assert_all_close(gv, ref_gv, atol: 0.15)
      assert_all_close(gbeta, ref_gbeta, atol: 0.15)
    end
  end

  # ============================================================================
  # RLA backward tests
  # ============================================================================

  describe "rla_backward_fallback" do
    test "produces finite gradients for basic case" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {alpha, key} = Nx.Random.uniform(key, 0.8, 0.95, shape: {batch, seq_len, num_heads}, type: {:f, 32})
      {beta, key} = Nx.Random.uniform(key, 0.1, 0.3, shape: {batch, seq_len, num_heads}, type: {:f, 32})
      {gamma, _key} = Nx.Random.uniform(key, 0.05, 0.15, shape: {batch, seq_len, num_heads}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

      {gq, gk, gv, ga, gb, gg} =
        Edifice.CUDA.FusedScan.rla_backward_fallback(q, k, v, alpha, beta, gamma, grad_output, :rla, 1.0)

      assert Nx.all(Nx.is_nan(gq) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(gv) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(ga) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.shape(gq) == {batch, seq_len, num_heads, head_dim}
      assert Nx.shape(ga) == {batch, seq_len, num_heads}
    end

    test "produces finite gradients for bf16 inputs" do
      batch = 1
      seq_len = 3
      num_heads = 2
      head_dim = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {alpha, key} = Nx.Random.uniform(key, 0.8, 0.95, shape: {batch, seq_len, num_heads}, type: {:bf, 16})
      {beta, key} = Nx.Random.uniform(key, 0.1, 0.3, shape: {batch, seq_len, num_heads}, type: {:bf, 16})
      {gamma, _key} = Nx.Random.uniform(key, 0.05, 0.15, shape: {batch, seq_len, num_heads}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, num_heads, head_dim})

      {gq, gk, gv, ga, gb, gg} =
        Edifice.CUDA.FusedScan.rla_backward_fallback(q, k, v, alpha, beta, gamma, grad_output, :rla, 1.0)

      assert Nx.all(Nx.is_nan(gq) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(gv) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.shape(gq) == {batch, seq_len, num_heads, head_dim}
      assert Nx.shape(ga) == {batch, seq_len, num_heads}
    end
  end

  # ============================================================================
  # TTT backward tests
  # ============================================================================

  describe "ttt_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 1
      seq_len = 3
      inner_size = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:f, 32})
      {eta, key} = Nx.Random.uniform(key, 0.01, 0.1, shape: {batch, seq_len, inner_size}, type: {:f, 32})
      {w0, _key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {inner_size, inner_size}, type: {:f, 32})
      ln_gamma = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {inner_size})
      ln_beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {inner_size})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, inner_size})

      {ref_gq, ref_gk, ref_gv, ref_geta, _, _, _} =
        Nx.Defn.grad({q, k, v, eta, w0, ln_gamma, ln_beta}, fn {q, k, v, eta, w0, lng, lnb} ->
          ttt_forward(q, k, v, eta, w0, lng, lnb) |> Nx.sum()
        end)

      {gq, gk, gv, geta, _gw0, _glng, _glnb} =
        Edifice.CUDA.FusedScan.ttt_backward_fallback(q, k, v, eta, w0, ln_gamma, ln_beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 1.0e-3)
      assert_all_close(gk, ref_gk, atol: 1.0e-3)
      assert_all_close(gv, ref_gv, atol: 1.0e-3)
      assert_all_close(geta, ref_geta, atol: 1.0e-3)
    end

    test "matches Nx autodiff for bf16 inputs" do
      batch = 1
      seq_len = 3
      inner_size = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, inner_size}, type: {:bf, 16})
      {eta, key} = Nx.Random.uniform(key, 0.01, 0.1, shape: {batch, seq_len, inner_size}, type: {:bf, 16})
      {w0, _key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {inner_size, inner_size}, type: {:bf, 16})
      ln_gamma = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {inner_size})
      ln_beta = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {inner_size})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, inner_size})

      {ref_gq, ref_gk, ref_gv, ref_geta, _, _, _} =
        Nx.Defn.grad({q, k, v, eta, w0, ln_gamma, ln_beta}, fn {q, k, v, eta, w0, lng, lnb} ->
          ttt_forward(q, k, v, eta, w0, lng, lnb) |> Nx.sum()
        end)

      {gq, gk, gv, geta, _, _, _} =
        Edifice.CUDA.FusedScan.ttt_backward_fallback(q, k, v, eta, w0, ln_gamma, ln_beta, grad_output)

      assert_all_close(gq, ref_gq, atol: 0.15)
      assert_all_close(gk, ref_gk, atol: 0.15)
      assert_all_close(gv, ref_gv, atol: 0.15)
      assert_all_close(geta, ref_geta, atol: 0.15)
    end
  end

  # ============================================================================
  # DeltaProduct backward tests
  # ============================================================================

  describe "delta_product_backward_fallback" do
    test "produces finite gradients for basic case" do
      batch = 1
      seq_len = 2
      num_heads = 2
      head_dim = 2
      num_householder = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_householder, num_heads, head_dim}, type: {:f, 32})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_householder, num_heads, head_dim}, type: {:f, 32})
      {beta, _key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_householder, num_heads}, type: {:f, 32})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, num_heads, head_dim})

      {gq, gk, gv, gbeta} =
        Edifice.CUDA.FusedScan.delta_product_backward_fallback(q, k, v, beta, grad_output)

      assert Nx.all(Nx.is_nan(gq) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(gv) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.shape(gq) == {batch, seq_len, num_heads, head_dim}
      assert Nx.shape(gk) == {batch, seq_len, num_householder, num_heads, head_dim}
      assert Nx.shape(gbeta) == {batch, seq_len, num_householder, num_heads}
    end

    test "produces finite gradients for bf16 inputs" do
      batch = 1
      seq_len = 2
      num_heads = 2
      head_dim = 2
      num_householder = 2

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_heads, head_dim}, type: {:bf, 16})
      {k, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_householder, num_heads, head_dim}, type: {:bf, 16})
      {v, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, num_householder, num_heads, head_dim}, type: {:bf, 16})
      {beta, _key} = Nx.Random.uniform(key, 0.1, 0.5, shape: {batch, seq_len, num_householder, num_heads}, type: {:bf, 16})

      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, num_heads, head_dim})

      {gq, gk, gv, gbeta} =
        Edifice.CUDA.FusedScan.delta_product_backward_fallback(q, k, v, beta, grad_output)

      assert Nx.all(Nx.is_nan(gq) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(gv) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.shape(gq) == {batch, seq_len, num_heads, head_dim}
      assert Nx.shape(gk) == {batch, seq_len, num_householder, num_heads, head_dim}
    end
  end

  # ============================================================================
  # sLSTM backward tests
  # ============================================================================

  describe "slstm_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, 4 * hidden}, type: {:f, 32})
      {r_weight, _key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {hidden, 4 * hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_out = slstm_forward(wx, r_weight, h0, c0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_wx, _ref_grad_r} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> slstm_forward(w, r, h0, c0, hidden) |> Nx.sum() end)

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.slstm_backward_fallback(wx, r_weight, h0, c0, forward_out, grad_output)

      assert_all_close(grad_wx, ref_grad_wx, atol: 0.02)
    end

    test "matches Nx autodiff for bf16 wx inputs" do
      batch = 2
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(42)
      {wx, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq_len, 4 * hidden}, type: {:bf, 16})
      {r_weight, _key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {hidden, 4 * hidden}, type: {:bf, 16})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:bf, 16}), {batch, hidden})

      forward_out = slstm_forward(wx, r_weight, h0, c0, hidden)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:bf, 16}), {batch, seq_len, hidden})

      {ref_grad_wx, _} =
        Nx.Defn.grad({wx, r_weight}, fn {w, r} -> slstm_forward(w, r, h0, c0, hidden) |> Nx.sum() end)

      {grad_wx, _, _} =
        Edifice.CUDA.FusedScan.slstm_backward_fallback(wx, r_weight, h0, c0, forward_out, grad_output)

      assert_all_close(grad_wx, ref_grad_wx, atol: 0.15)
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

  # ELU-GRU forward scan for testing
  defp elu_gru_forward(z, c, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(z)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Standard LSTM forward scan for testing
  defp lstm_forward(wx, recurrent_weight, h0, c0, hidden) do
    {_batch, seq_len, _hidden4} = Nx.shape(wx)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0}, []}, fn t, {{h_p, c_p}, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1) |> Nx.sigmoid()
        f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1) |> Nx.sigmoid()
        g_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, g_t))
        h_t = Nx.multiply(o_t, Nx.tanh(c_t))

        {{h_t, c_t}, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Standard GRU forward scan for testing
  defp gru_forward(wx, recurrent_weight, h0, hidden, bhn \\ nil) do
    {_batch, seq_len, _hidden3} = Nx.shape(wx)
    bhn = if bhn, do: bhn, else: Nx.broadcast(Nx.tensor(0.0, type: Nx.type(wx)), {hidden})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_p, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])

        r_t = Nx.add(Nx.slice_along_axis(wx_t, 0, hidden, axis: 1),
                     Nx.slice_along_axis(rh_t, 0, hidden, axis: 1)) |> Nx.sigmoid()
        z_t = Nx.add(Nx.slice_along_axis(wx_t, hidden, hidden, axis: 1),
                     Nx.slice_along_axis(rh_t, hidden, hidden, axis: 1)) |> Nx.sigmoid()
        rh_n = Nx.add(Nx.slice_along_axis(rh_t, 2 * hidden, hidden, axis: 1), bhn)
        n_t = Nx.add(Nx.slice_along_axis(wx_t, 2 * hidden, hidden, axis: 1),
                     Nx.multiply(r_t, rh_n)) |> Nx.tanh()
        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), n_t), Nx.multiply(z_t, h_p))

        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Liquid forward scan for testing
  defp liquid_forward(tau, activation, h0) do
    {_batch, seq_len, _hidden} = Nx.shape(tau)

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        tau_t = Nx.slice_along_axis(tau, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        act_t = Nx.slice_along_axis(activation, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        decay = Nx.exp(Nx.negate(Nx.divide(1.0, tau_t)))
        h_t = Nx.add(act_t, Nx.multiply(Nx.subtract(h_prev, act_t), decay))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # DeltaNet forward scan for testing
  defp delta_net_forward(q, k, v, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        sk = Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        diff = Nx.subtract(v_t, sk)
        delta = Nx.multiply(Nx.new_axis(beta_t, -1), Nx.multiply(Nx.new_axis(diff, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_prev, delta)
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # GatedDeltaNet forward scan for testing
  defp gated_delta_net_forward(q, k, v, beta, alpha) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        alpha_broad = alpha_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        s_decayed = Nx.multiply(alpha_broad, s_prev)
        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        diff = Nx.subtract(v_t, sk)
        delta = Nx.multiply(Nx.new_axis(beta_t, -1), Nx.multiply(Nx.new_axis(diff, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Selective scan forward for testing
  defp selective_scan_forward(x, dt, a, b, c) do
    {batch, seq_len, hidden} = Nx.shape(x)
    state_size = Nx.axis_size(a, 1)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(x)), {batch, hidden, state_size})

    {_, y_list} =
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.slice_along_axis(dt, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.clip(dt_t, 0.001, 0.1)
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        dt_exp = Nx.new_axis(dt_t, 2)
        a_bar = Nx.exp(Nx.multiply(dt_exp, Nx.reshape(a, {1, hidden, state_size})))
        b_bar = Nx.multiply(dt_exp, Nx.reshape(b_t, {batch, 1, state_size}))
        h_new = Nx.add(Nx.multiply(a_bar, h_prev), Nx.multiply(b_bar, Nx.new_axis(x_t, 2)))

        y_t = Nx.sum(Nx.multiply(h_new, Nx.reshape(c_t, {batch, 1, state_size})), axes: [2])
        {h_new, [y_t | acc]}
      end)

    y_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # KDA forward scan for testing
  defp kda_forward(q, k, v, alpha, beta) do
    {batch, seq_len, num_heads, head_dim} = Nx.shape(q)
    s0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(q)), {batch, num_heads, head_dim, head_dim})

    {_, o_list} =
      Enum.reduce(0..(seq_len - 1), {s0, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        decay = Nx.exp(alpha_t) |> Nx.new_axis(3)
        s_decayed = Nx.multiply(decay, s_prev)
        sk = Nx.dot(s_decayed, [3], [0, 1], Nx.new_axis(k_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        error = Nx.subtract(v_t, sk)
        beta_bc = beta_t |> Nx.new_axis(-1) |> Nx.new_axis(-1)
        delta = Nx.multiply(beta_bc, Nx.multiply(Nx.new_axis(error, -1), Nx.new_axis(k_t, -2)))
        s_t = Nx.add(s_decayed, delta)
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, -1), [2], [0, 1]) |> Nx.squeeze(axes: [-1])
        {s_t, [o_t | acc]}
      end)

    o_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # TTT forward scan for testing
  defp ttt_forward(q, k, v, eta, w0, ln_gamma, ln_beta) do
    {batch, seq_len, inner_size} = Nx.shape(q)

    w_init =
      if Nx.rank(w0) == 2 do
        Nx.broadcast(w0, {batch, inner_size, inner_size})
      else
        w0
      end

    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {w_init, []}, fn t, {w_prev, acc} ->
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_t = Nx.slice_along_axis(eta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        mean = Nx.mean(pred, axes: [-1], keep_axes: true)
        var = Nx.variance(pred, axes: [-1], keep_axes: true)
        pred_normed = Nx.divide(Nx.subtract(pred, mean), Nx.sqrt(Nx.add(var, 1.0e-6)))
        pred_normed = Nx.add(Nx.multiply(pred_normed, ln_gamma), ln_beta)

        error = Nx.subtract(pred_normed, v_t)
        grad = Nx.dot(Nx.new_axis(Nx.multiply(eta_t, error), 2), [2], [0], Nx.new_axis(k_t, 1), [1], [0])
        w_new = Nx.subtract(w_prev, grad)

        out = Nx.dot(w_new, [2], [0], Nx.new_axis(q_t, 2), [1], [0]) |> Nx.squeeze(axes: [2])
        {w_new, [out | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # sLSTM forward scan for testing
  defp slstm_forward(wx, recurrent_weight, h0, c0, hidden) do
    {batch, seq_len, _hidden4} = Nx.shape(wx)

    n0 = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(wx)), Nx.shape(h0))
    m0 = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(wx)), Nx.shape(h0))

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {{h0, c0, n0, m0}, []}, fn t, {{h_p, c_p, n_p, m_p}, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh_t = Nx.dot(h_p, [1], recurrent_weight, [0])
        gates_t = Nx.add(wx_t, rh_t)

        log_i_t = Nx.slice_along_axis(gates_t, 0, hidden, axis: 1)
        log_f_t = Nx.slice_along_axis(gates_t, hidden, hidden, axis: 1)
        z_t = Nx.slice_along_axis(gates_t, hidden * 2, hidden, axis: 1) |> Nx.tanh()
        o_t = Nx.slice_along_axis(gates_t, hidden * 3, hidden, axis: 1) |> Nx.sigmoid()

        m_t = Nx.max(Nx.add(log_f_t, m_p), log_i_t)
        i_t = Nx.exp(Nx.subtract(log_i_t, m_t))
        f_t = Nx.exp(Nx.subtract(Nx.add(log_f_t, m_p), m_t))

        c_t = Nx.add(Nx.multiply(f_t, c_p), Nx.multiply(i_t, z_t))
        n_t = Nx.add(Nx.multiply(f_t, n_p), i_t)
        safe_denom = Nx.max(Nx.abs(n_t), 1.0)
        h_t = Nx.multiply(o_t, Nx.divide(c_t, safe_denom))

        {{h_t, c_t, n_t, m_t}, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Compute numerical gradient via central finite differences
  defp finite_diff_grad(tensor, f, eps) do
    flat = Nx.reshape(tensor, {Nx.size(tensor)})
    n = Nx.size(tensor)

    grads =
      for idx <- 0..(n - 1) do
        # +eps
        delta = Nx.indexed_put(Nx.broadcast(0.0, {n}), Nx.tensor([[idx]]), Nx.tensor([eps]))
        plus = Nx.reshape(Nx.add(flat, delta), Nx.shape(tensor))
        f_plus = Nx.to_number(f.(plus))

        # -eps
        minus = Nx.reshape(Nx.subtract(flat, delta), Nx.shape(tensor))
        f_minus = Nx.to_number(f.(minus))

        (f_plus - f_minus) / (2 * eps)
      end

    Nx.tensor(grads, type: {:f, 32}) |> Nx.reshape(Nx.shape(tensor))
  end
end
