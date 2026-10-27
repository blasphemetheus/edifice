defmodule Edifice.CUDA.KernelMatrixGpuTest do
  @moduledoc """
  GPU parity matrix: the custom-call tier against the pure-Nx reference, on
  real hardware.

  Excluded by default. Run with the CUDA client and the FFI library built
  (`scripts/check_cuda_kernels.sh` does all of it):

      mix edifice.cuda.build --arch sm_120
      EDIFICE_TEST_CLIENT=cuda mix test --only cuda_kernels

  The custom-call tier (`libedifice_xla.so` through `EXLA.load_dylib/1`)
  covers the ops in `Edifice.Block.native_ops/0`, selective scan forward and
  backward. Each is compiled by EXLA with the tier on and compared with the
  same code compiled with the tier forced off, at the op level and inside
  the 2-layer Mamba and GRU models at batch 4. A direct check pins that the
  GRU scan, whose handler was retired on 2026-10-26, ignores the tier.

  The NIF tier that this file also covered (every forward kernel on eager
  tensors, plus bf16) was removed the same day; see
  `docs/kernel_tier_benchmarks.md`. Fallback backward math is pinned against
  autodiff in `backward_kernels_test.exs`.
  """
  use ExUnit.Case, async: false
  @moduletag :exla_only
  @moduletag :cuda_kernels
  @moduletag timeout: 300_000

  alias Edifice.CUDA.FusedScan

  @cuda {EXLA.Backend, client: :cuda}

  @b 2
  @t 16
  @h 64

  setup_all do
    unless System.get_env("EDIFICE_TEST_CLIENT") == "cuda" do
      raise "run with EDIFICE_TEST_CLIENT=cuda (the suite pins EXLA to the host client otherwise)"
    end

    # The selective-scan backward handler allocates its workspace outside
    # XLA's pool; leave it room.
    clients = Application.fetch_env!(:exla, :clients)
    cuda = Keyword.merge(Keyword.fetch!(clients, :cuda), memory_fraction: 0.6)
    Application.put_env(:exla, :clients, Keyword.put(clients, :cuda, cuda))
    on_exit(fn -> Application.put_env(:exla, :clients, clients) end)
    :ok
  end

  # ── helpers ─────────────────────────────────────────────────────────

  defp rand(key, shape, lo, hi, type \\ {:f, 32}) do
    Nx.Random.uniform(key, lo, hi, shape: shape, type: type)
  end

  defp assert_close(got, want, opts) do
    atol = Keyword.fetch!(opts, :atol)
    rtol = Keyword.get(opts, :rtol, 0.0)
    assert Nx.shape(got) == Nx.shape(want)

    got = Nx.as_type(got, {:f, 32})
    want = Nx.as_type(want, {:f, 32})
    diff = Nx.abs(Nx.subtract(got, want))
    budget = Nx.add(atol, Nx.multiply(rtol, Nx.abs(want)))
    worst = Nx.reduce_max(Nx.subtract(diff, budget)) |> Nx.to_number()

    refute Nx.to_number(Nx.any(Nx.is_nan(got))) == 1, "kernel output has NaN"

    assert worst <= 0,
           "max |kernel - reference| exceeds budget by #{worst} (atol=#{atol}, rtol=#{rtol}); " <>
             "max abs diff #{Nx.to_number(Nx.reduce_max(diff))}"
  end

  # ── custom-call tier (EXLA FFI handlers) forward + gradients ──────

  describe "custom-call tier == fallback, forward and gradients" do
    setup do
      unless Edifice.Block.native_impl?(), do: raise("EXLA.CustomCall protocol unavailable")
      prev = System.get_env("EDIFICE_FUSED_CUSTOM_CALL")
      System.put_env("EDIFICE_FUSED_CUSTOM_CALL", "1")
      # Raises with a build hint if libedifice_xla.so is missing.
      Edifice.CUDA.Library.ensure_loaded!()
      prev_backend = Nx.default_backend(@cuda)

      on_exit(fn ->
        Nx.default_backend(prev_backend)

        if prev,
          do: System.put_env("EDIFICE_FUSED_CUSTOM_CALL", prev),
          else: System.delete_env("EDIFICE_FUSED_CUSTOM_CALL")
      end)

      :ok
    end

    # The GRU handlers were retired on 2026-10-26 (slower than XLA's compiled
    # fallback at every size; docs/kernel_tier_benchmarks.md). With the tier
    # on, the GRU scan must still compile its fallback, so the GRU model case
    # below is a no-regression check, not a kernel parity check.
    test "gru_scan ignores the tier and compiles the fallback" do
      {wx, r, h0, bhn} = gru_inputs(64)

      fused =
        Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

      ref =
        Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

      got = fused.(wx, r, h0, bhn) |> Nx.backend_copy(Nx.BinaryBackend)
      want = forced_fallback(fn -> ref.(wx, r, h0, bhn) end) |> Nx.backend_copy(Nx.BinaryBackend)
      assert_close(got, want, atol: 1.0e-6)
      refute FusedScan.custom_call_available?(:gru_scan)
    end

    for state <- [8, 32] do
      test "selective_scan forward + gradients at state=#{state}" do
        {x, dt, a, b, c} = selective_inputs(unquote(state))

        fwd_fused =
          Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
            compiler: EXLA
          )

        fwd_ref =
          Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
            compiler: EXLA
          )

        got = fwd_fused.(x, dt, a, b, c) |> Nx.backend_copy(Nx.BinaryBackend)

        want =
          forced_fallback(fn -> fwd_ref.(x, dt, a, b, c) end) |> Nx.backend_copy(Nx.BinaryBackend)

        assert_close(got, want, atol: 1.0e-5, rtol: 1.0e-4)

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

        got = Nx.Defn.jit(grad_fused, compiler: EXLA).(x, dt, b, c, a) |> copy_all()

        want =
          forced_fallback(fn -> Nx.Defn.jit(grad_ref, compiler: EXLA).(x, dt, b, c, a) end)
          |> copy_all()

        for {g, w} <- Enum.zip(Tuple.to_list(got), Tuple.to_list(want)) do
          assert_close(g, w, atol: 1.0e-5, rtol: 1.0e-4)
        end
      end
    end

    test "the protocol lowers exactly the ops in Edifice.Block.native_ops/0" do
      client = %{platform: :cuda}
      f32 = Nx.template({@b, @t, 3 * @h}, {:f, 32})
      r = Nx.template({@h, 3 * @h}, {:f, 32})
      h0 = Nx.template({@b, @h}, {:f, 32})
      bhn = Nx.template({@h}, {:f, 32})
      x = Nx.template({@b, @t, @h}, {:f, 32})
      a = Nx.template({@h, 16}, {:f, 32})
      bc = Nx.template({@b, @t, 16}, {:f, 32})

      operands = %{
        fused_selective_scan: [x, x, a, bc, bc],
        fused_selective_scan_backward: [x, x, a, bc, bc, x]
      }

      outputs = %{
        fused_selective_scan: x,
        fused_selective_scan_backward: {x, x, bc, bc}
      }

      row_major = fn %{shape: shape} ->
        rank = tuple_size(shape)
        "dense<[#{Enum.join((rank - 1)..0//-1, ", ")}]> : tensor<#{rank}xindex>"
      end

      for op <- Edifice.Block.native_ops() do
        block = %Edifice.Block.FusedOp{op: op}

        assert {:ok, %EXLA.CustomCall.Spec{attributes: attrs}} =
                 EXLA.CustomCall.call(block, outputs[op], operands[op], client),
               "#{op}"

        # Every operand and result is pinned row-major: XLA may otherwise
        # pick a layout the kernel cannot read (T-major GRU output at B>1).
        {_, operand_layouts} = List.keyfind(attrs, "operand_layouts", 0)
        {_, result_layouts} = List.keyfind(attrs, "result_layouts", 0)
        assert operand_layouts == "[" <> Enum.map_join(operands[op], ", ", row_major) <> "]"

        assert result_layouts ==
                 "[" <>
                   Enum.map_join(Nx.Defn.Composite.flatten_list([outputs[op]]), ", ", row_major) <>
                   "]"
      end

      assert :skip ==
               EXLA.CustomCall.call(
                 %Edifice.Block.FusedOp{op: :fused_mingru_scan},
                 nil,
                 [x, x, h0],
                 client
               )

      assert :skip ==
               EXLA.CustomCall.call(
                 %Edifice.Block.FusedOp{op: :fused_gru_scan},
                 nil,
                 [f32, r, h0, bhn],
                 client
               )

      assert :skip ==
               EXLA.CustomCall.call(
                 %Edifice.Block.FusedOp{op: :fused_selective_scan},
                 nil,
                 operands.fused_selective_scan,
                 %{platform: :host}
               )
    end

    # Regression: inside a real model XLA assigned a custom-call result a
    # T-major layout (its consumer was a reduce over H plus a last-step
    # slice), so the 2-layer GRU model was off by ~7 at batch 32 while every
    # parameter-fed op-level case, and batch 1, matched. The layout pin in
    # the Spec fixes it; this exercises the whole Axon graph at batch 4. The
    # GRU model stays in the loop as a no-regression check now that its
    # scan always compiles the fallback.
    for arch <- [:gru, :mamba] do
      test "#{arch} 2-layer model at batch 4 matches the fallback, forward and gradients" do
        {b, t, h} = {4, 24, 64}
        extra = if unquote(arch) == :mamba, do: [state_size: 16], else: []
        opts = [embed_dim: h, hidden_size: h, num_layers: 2, seq_len: t, dropout: 0.0] ++ extra
        {x, _} = Nx.Random.uniform(Nx.Random.key(3), -1.0, 1.0, shape: {b, t, h}, type: :f32)
        input = %{"state_sequence" => x}

        build = fn ->
          {init_fn, predict_fn} =
            Edifice.build(unquote(arch), opts) |> Axon.build(mode: :inference, seed: 0)

          {predict_fn, init_fn.(Nx.template({b, t, h}, :f32), Axon.ModelState.empty())}
        end

        # The fallback build decides its layer graph without the tier.
        {pf_ref, params} = forced_fallback(build)
        {pf_fused, _} = build.()
        loss = fn pf, p, x -> pf.(p, x) |> Nx.pow(2) |> Nx.sum() end
        hp = [compiler: EXLA, precision: :highest]

        # Distinct closures: EXLA keys its cache on the closure, so the
        # same wrapper around two builds would share one executable.
        fwd_ref = Nx.Defn.jit(fn p, x -> pf_ref.(p, x) end, hp)
        fwd_fused = Nx.Defn.jit(fn p, x -> pf_fused.(p, x) end, hp)
        grad_ref = Nx.Defn.jit(fn p, x -> Nx.Defn.grad(p, &loss.(pf_ref, &1, x)) end, hp)
        grad_fused = Nx.Defn.jit(fn p, x -> Nx.Defn.grad(p, &loss.(pf_fused, &1, x)) end, hp)

        want =
          forced_fallback(fn -> fwd_ref.(params, input) end) |> Nx.backend_copy(Nx.BinaryBackend)

        got = fwd_fused.(params, input) |> Nx.backend_copy(Nx.BinaryBackend)
        assert_close(got, want, atol: 1.0e-4, rtol: 1.0e-3)

        want_g = forced_fallback(fn -> grad_ref.(params, input) end)
        got_g = grad_fused.(params, input)

        for {g, w} <-
              Enum.zip(
                Nx.Defn.Composite.flatten_list([got_g]),
                Nx.Defn.Composite.flatten_list([want_g])
              ) do
          assert_close(
            Nx.backend_copy(g, Nx.BinaryBackend),
            Nx.backend_copy(w, Nx.BinaryBackend),
            atol: 1.0e-3,
            rtol: 1.0e-2
          )
        end
      end
    end
  end

  defp forced_fallback(fun) do
    Process.put(:__edifice_force_fallback__, true)

    try do
      fun.()
    after
      Process.delete(:__edifice_force_fallback__)
    end
  end

  defp copy_all(tuple),
    do:
      tuple
      |> Tuple.to_list()
      |> Enum.map(&Nx.backend_copy(&1, Nx.BinaryBackend))
      |> List.to_tuple()

  defp gru_inputs(hidden) do
    key = Nx.Random.key(1234)
    {wx, key} = rand(key, {@b, @t, 3 * hidden}, -0.5, 0.5)
    {r, key} = rand(key, {hidden, 3 * hidden}, -0.2, 0.2)
    {h0, key} = rand(key, {@b, hidden}, -0.5, 0.5)
    {bhn, _} = rand(key, {hidden}, -0.3, 0.3)
    {wx, r, h0, bhn}
  end

  defp selective_inputs(state) do
    key = Nx.Random.key(77)
    {x, key} = rand(key, {@b, @t, @h}, -0.5, 0.5)
    {dt, key} = rand(key, {@b, @t, @h}, 0.001, 0.1)
    {b, key} = rand(key, {@b, @t, state}, -0.5, 0.5)
    {c, _} = rand(key, {@b, @t, state}, -0.5, 0.5)
    {x, dt, canonical_a(state), b, c}
  end

  # The selective-scan fallback hardcodes Mamba's -(1..state) diagonal and
  # ignores `a`; only this A keeps kernel and fallback on the same contract.
  defp canonical_a(state) do
    a_diag = Nx.negate(Nx.add(Nx.iota({state}), 1.0))
    Nx.broadcast(Nx.reshape(a_diag, {1, state}), {@h, state})
  end
end
