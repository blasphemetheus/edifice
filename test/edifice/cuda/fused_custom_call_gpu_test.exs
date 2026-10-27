defmodule Edifice.CUDA.FusedCustomCallGpuTest do
  @moduledoc """
  End-to-end equivalence tests for the REVIVED custom-call tier
  (Edifice.Block.FusedOp defimpl + dynamically loaded FFI kernels).
  Excluded by default (:exla_only): they need CUDA, an EXLA build with
  load_dylib/1, `mix edifice.cuda.build`, and the explicit activation flag —

      EDIFICE_FUSED_CUSTOM_CALL=1 mix test \\
        test/edifice/cuda/fused_custom_call_gpu_test.exs \\
        --include exla_only

  Each test computes the fused (custom-call) result inside an
  EXLA-compiled defn and compares against the pure-Nx fallback with the
  tier forced off, and both against an f64 truth. Only the selective scan
  has a handler; the GRU handler was retired on 2026-10-26
  (docs/kernel_tier_benchmarks.md).
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
      raise "these tests need EDIFICE_FUSED_CUSTOM_CALL=1 and EXLA.CustomCall support"
    end

    # Native scratch/local memory is allocated outside XLA's pool. The default
    # 90% reservation can starve a later launch after other kernels run.
    clients = Application.fetch_env!(:exla, :clients)
    cuda = Keyword.merge(Keyword.fetch!(clients, :cuda), memory_fraction: 0.7)
    Application.put_env(:exla, :clients, Keyword.put(clients, :cuda, cuda))
    on_exit(fn -> Application.put_env(:exla, :clients, clients) end)
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

  @tag :selective_scan
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
    fwd_fused =
      Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
        compiler: EXLA
      )

    fwd_ref =
      Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
        compiler: EXLA
      )

    fused = fwd_fused.(x, dt, a, b, c)
    ref = forced_fallback(fn -> fwd_ref.(x, dt, a, b, c) end)

    truth = selective_f64_truth(x, dt, a, b, c)
    Nx.Testing.assert_all_close(fused, truth, atol: 1.0e-5, rtol: 1.0e-4)
    Nx.Testing.assert_all_close(ref, truth, atol: 1.0e-5, rtol: 1.0e-4)

    # a passed as jit argument (same closure/backend rule as h0 above)
    grad_fused = fn x, dt, b, c, a ->
      Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
        FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.sum()
      end)
    end

    grad_ref = fn x, dt, b, c, a ->
      Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
        FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.sum()
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

    for {fused, reference} <- [{gx_f, gx_r}, {gdt_f, gdt_r}, {gb_f, gb_r}, {gc_f, gc_r}] do
      Nx.Testing.assert_all_close(fused, reference, atol: 1.0e-5, rtol: 1.0e-4)
    end
  end

  defp selective_f64_truth(x, dt, a, b, c) do
    previous_backend = Nx.default_backend(Nx.BinaryBackend)

    try do
      selective_f64_truth_impl(x, dt, a, b, c)
    after
      Nx.default_backend(previous_backend)
    end
  end

  defp selective_f64_truth_impl(x, dt, a, b, c) do
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
end
