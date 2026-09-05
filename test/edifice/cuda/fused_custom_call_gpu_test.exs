defmodule Edifice.CUDA.FusedCustomCallGpuTest do
  @moduledoc """
  End-to-end equivalence tests for the REVIVED custom-call tier
  (Edifice.Block.FusedOp defimpl + FFI kernels linked into the EXLA
  fork). Excluded by default (:exla_only): they need CUDA, the fork
  with kernels linked, and the explicit activation flag —

      EDIFICE_FUSED_CUSTOM_CALL=1 mix test \\
        test/edifice/cuda/fused_custom_call_gpu_test.exs \\
        --include exla_only

  Each test computes the fused (custom-call) result inside an
  EXLA-compiled defn and compares against the pure-Nx fallback with the
  tier forced off. GRU cases run at hidden=512 deliberately — that
  exercises the 2026-09-04 H>256 one-block-per-batch fix on real
  hardware (the old kernel silently read uninitialized shared memory
  there).
  """
  use ExUnit.Case, async: false
  @moduletag :exla_only

  alias Edifice.CUDA.FusedScan

  @batch 4
  @seq 32
  @hidden 512

  setup_all do
    unless System.get_env("EDIFICE_FUSED_CUSTOM_CALL") == "1" and
             Edifice.Block.native_impl?() do
      raise "these tests need EDIFICE_FUSED_CUSTOM_CALL=1 + the kernel-linked EXLA fork"
    end

    Nx.global_default_backend({EXLA.Backend, client: :cuda})
    :ok
  end

  defp forced_fallback(fun) do
    Process.put(:__edifice_force_fallback__, true)

    try do
      fun.()
    after
      Process.delete(:__edifice_force_fallback__)
    end
  end

  defp gru_inputs do
    key = Nx.Random.key(1234)
    {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {@batch, @seq, 3 * @hidden}, type: {:f, 32})
    {r, key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {@hidden, 3 * @hidden}, type: {:f, 32})
    {h0, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {@batch, @hidden}, type: {:f, 32})
    {bhn, _} = Nx.Random.uniform(key, -0.3, 0.3, shape: {@hidden}, type: {:f, 32})
    {wx, r, h0, bhn}
  end

  test "gru_scan fused forward == fallback at H=512 (h0 + bhn)" do
    {wx, r, h0, bhn} = gru_inputs()

    # Both sides jitted (distinct definition sites — jit caches per
    # site): an EAGER reference call would dispatch the NIF arm
    # (cuda_available? matches eager EXLA tensors; forced_fallback only
    # suppresses the custom-call tier), and the NIF is not loaded here.
    fused_fn =
      Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

    ref_fn =
      Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

    fused = fused_fn.(wx, r, h0, bhn)
    ref = forced_fallback(fn -> ref_fn.(wx, r, h0, bhn) end)

    assert Nx.shape(fused) == {@batch, @seq, @hidden}

    # Arbitrate BOTH sides against an f64 BinaryBackend truth rather
    # than each other: measured 2026-09-05, the KERNEL is ~2.8e-6 from
    # truth while the XLA-compiled pure-Nx fallback drifts ~2.5e-3
    # (approximate GPU transcendentals compounding through the
    # recurrence). Fused-vs-fallback tolerance must therefore be the
    # fallback's error budget, not f32 epsilon.
    truth = gru_f64_truth(wx, r, h0, bhn)
    assert_all_close(fused, truth, atol: 1.0e-4)
    assert_all_close(ref, truth, atol: 1.0e-2)
  end

  defp gru_f64_truth(wx, r, h0, bhn) do
    to64 = fn t -> t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type({:f, 64}) end
    [wx, r, h0, bhn] = Enum.map([wx, r, h0, bhn], to64)
    {_batch, seq, hidden3} = Nx.shape(wx)
    hidden = div(hidden3, 3)

    {_, hs} =
      Enum.reduce(0..(seq - 1), {h0, []}, fn t, {h_p, acc} ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        rh = Nx.dot(h_p, [1], r, [0])

        r_g =
          Nx.sigmoid(Nx.add(Nx.slice_along_axis(wx_t, 0, hidden, axis: 1),
            Nx.slice_along_axis(rh, 0, hidden, axis: 1)))

        z_g =
          Nx.sigmoid(Nx.add(Nx.slice_along_axis(wx_t, hidden, hidden, axis: 1),
            Nx.slice_along_axis(rh, hidden, hidden, axis: 1)))

        n_g =
          Nx.tanh(Nx.add(Nx.slice_along_axis(wx_t, 2 * hidden, hidden, axis: 1),
            Nx.multiply(r_g, Nx.add(Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1), bhn))))

        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_g), n_g), Nx.multiply(z_g, h_p))
        {h_t, [h_t | acc]}
      end)

    hs |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  test "gru_scan fused gradients == fallback gradients at H=512" do
    {wx, r, h0, bhn} = gru_inputs()

    # NOTE: the fused and fallback sides must be DISTINCT anonymous-fn
    # definition sites — Nx.Defn.jit caches per definition site, so
    # re-invoking one fn under forced_fallback would reuse the already
    # compiled fused executable instead of re-tracing.
    # h0 must be a jit ARGUMENT — closing over an eager EXLA tensor
    # inside the traced fn mixes backends at the Nx.block boundary.
    grad_fused = fn wx, r, bhn, h0 ->
      Nx.Defn.grad({wx, r, bhn}, fn {wx, r, bhn} ->
        FusedScan.gru_scan(wx, r, h0, bhn) |> Nx.pow(2) |> Nx.mean()
      end)
    end

    grad_ref = fn wx, r, bhn, h0 ->
      Nx.Defn.grad({wx, r, bhn}, fn {wx, r, bhn} ->
        FusedScan.gru_scan(wx, r, h0, bhn) |> Nx.pow(2) |> Nx.mean()
      end)
    end

    {gwx_f, gr_f, gbhn_f} = Nx.Defn.jit(grad_fused, compiler: EXLA).(wx, r, bhn, h0)

    {gwx_ref, gr_ref, gbhn_ref} =
      forced_fallback(fn -> Nx.Defn.jit(grad_ref, compiler: EXLA).(wx, r, bhn, h0) end)

    assert_all_close(gwx_f, gwx_ref, atol: 5.0e-4)
    assert_all_close(gr_f, gr_ref, atol: 5.0e-4)
    assert_all_close(gbhn_f, gbhn_ref, atol: 5.0e-4)
  end

  test "selective_scan fused forward + grads == fallback (state=16)" do
    state = 16
    key = Nx.Random.key(77)
    {x, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {@batch, @seq, @hidden}, type: {:f, 32})
    {dt, key} = Nx.Random.uniform(key, 0.001, 0.1, shape: {@batch, @seq, @hidden}, type: {:f, 32})
    {b, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {@batch, @seq, state}, type: {:f, 32})
    {c, _} = Nx.Random.uniform(key, -0.5, 0.5, shape: {@batch, @seq, state}, type: {:f, 32})

    # A must be the CANONICAL -(1..state) broadcast that Mamba passes
    # (mamba.ex parallel_scan_ssm_impl): the Elixir fallback IGNORES the
    # A argument and hardcodes exactly this diagonal, so kernel and
    # fallback only agree on-contract. (A random A here diverges by
    # design — that sharp edge is why this comment exists.)
    a_diag = Nx.negate(Nx.add(Nx.iota({state}), 1.0))
    a = Nx.broadcast(Nx.reshape(a_diag, {1, state}), {@hidden, state})

    # Distinct definition sites per tier — see the jit-cache note above.
    fwd_fused = Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end, compiler: EXLA)
    fwd_ref = Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end, compiler: EXLA)

    fused = fwd_fused.(x, dt, a, b, c)
    ref = forced_fallback(fn -> fwd_ref.(x, dt, a, b, c) end)

    # Same f64-truth arbitration as the GRU test (fallback drifts via
    # XLA's approximate exp over the recurrence; kernel is near-exact).
    truth = selective_f64_truth(x, dt, a, b, c)
    assert_all_close(fused, truth, atol: 1.0e-4)
    assert_all_close(ref, truth, atol: 1.0e-1)

    # a passed as jit argument (same closure/backend rule as h0 above)
    grad_fused = fn x, dt, b, c, a ->
      Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
        FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.mean()
      end)
    end

    grad_ref = fn x, dt, b, c, a ->
      Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
        FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.mean()
      end)
    end

    {gx_f, gdt_f, gb_f, gc_f} = Nx.Defn.jit(grad_fused, compiler: EXLA).(x, dt, b, c, a)

    # Materialize the fused grads off-device before the second jit runs:
    # their device buffers were observed invalidated ("deleted or
    # donated") after the ref executable executed.
    [gx_f, gdt_f, gb_f, gc_f] =
      Enum.map([gx_f, gdt_f, gb_f, gc_f], &Nx.backend_copy(&1, Nx.BinaryBackend))

    {gx_r, gdt_r, gb_r, gc_r} =
      forced_fallback(fn -> Nx.Defn.jit(grad_ref, compiler: EXLA).(x, dt, b, c, a) end)

    assert_all_close(gx_f, gx_r, atol: 5.0e-4)
    assert_all_close(gdt_f, gdt_r, atol: 5.0e-4)
    assert_all_close(gb_f, gb_r, atol: 5.0e-4)
    assert_all_close(gc_f, gc_r, atol: 5.0e-4)
  end

  defp selective_f64_truth(x, dt, a, b, c) do
    to64 = fn t -> t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type({:f, 64}) end
    [x, dt, a, b, c] = Enum.map([x, dt, a, b, c], to64)
    {batch, seq, hidden} = Nx.shape(x)
    {_h, state} = Nx.shape(a)

    dt = Nx.clip(dt, 0.001, 0.1)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 64}), {batch, hidden, state})

    {_, ys} =
      Enum.reduce(0..(seq - 1), {h0, []}, fn t, {h, acc} ->
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        dt_t = Nx.slice_along_axis(dt, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        dt_e = Nx.new_axis(dt_t, 2)
        a_bar = Nx.exp(Nx.multiply(dt_e, Nx.new_axis(a, 0)))
        b_bar = Nx.multiply(dt_e, Nx.new_axis(b_t, 1))
        h = Nx.add(Nx.multiply(a_bar, h), Nx.multiply(b_bar, Nx.new_axis(x_t, 2)))
        y = Nx.sum(Nx.multiply(h, Nx.new_axis(c_t, 1)), axes: [2])
        {h, [y | acc]}
      end)

    ys |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  defp assert_all_close(a, b, opts) do
    atol = Keyword.fetch!(opts, :atol)

    max_diff =
      Nx.abs(Nx.subtract(a, b)) |> Nx.reduce_max() |> Nx.to_number()

    assert max_diff <= atol,
           "max |fused - fallback| = #{max_diff} > #{atol}"
  end
end
