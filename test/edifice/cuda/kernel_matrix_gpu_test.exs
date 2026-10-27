defmodule Edifice.CUDA.KernelMatrixGpuTest do
  @moduledoc """
  GPU parity matrix: every fused CUDA kernel reachable from Elixir against
  the pure-Nx reference, on real hardware.

  Excluded by default. Run with the CUDA client and the native libraries built
  (`scripts/check_cuda_kernels.sh` does all of it):

      make -C native/cuda GPU_ARCH=sm_120 && make -C c_src
      mix edifice.cuda.build --arch sm_120
      EDIFICE_TEST_CLIENT=cuda mix test --only cuda_kernels

  Three tiers, three references:

  * **NIF tier** (`priv/libedifice_cuda_nif.so` + `priv/cuda/*.so`): every
    forward kernel, including the five multi-layer block scans, runs on eager
    `{EXLA.Backend, client: :cuda}` tensors and is compared with the same
    `FusedScan` entry point traced under `Nx.Defn.Evaluator` on the
    BinaryBackend (inside `defn` the dispatch always takes the Elixir
    fallback). Kernel and fallback are both sequential f32, so the budget is
    tight.
  * **Custom-call tier** (`libedifice_xla.so` through `EXLA.load_dylib/1`):
    the ops in `Edifice.Block.native_ops/0` (selective scan forward and
    backward), compiled by EXLA with the tier on, against the same code
    compiled with the tier forced off, plus the 2-layer Mamba and GRU models
    at batch 4.
  * **bf16**: the simple scans through the bf16 kernel library, against the
    f32 reference with a bf16-sized budget.

  The backward NIF bindings (`Edifice.CUDA.NIF.fused_*_backward`) are not
  covered: nothing in `FusedScan` calls them, so there is no reachable
  behaviour to pin. Fallback backward math is pinned against autodiff in
  `backward_kernels_test.exs`.
  """
  use ExUnit.Case, async: false
  @moduletag :exla_only
  @moduletag :cuda_kernels
  @moduletag timeout: 300_000

  alias Edifice.CUDA.FusedScan

  @cuda {EXLA.Backend, client: :cuda}

  setup_all do
    unless System.get_env("EDIFICE_TEST_CLIENT") == "cuda" do
      raise "run with EDIFICE_TEST_CLIENT=cuda (the suite pins EXLA to the host client otherwise)"
    end

    unless Edifice.CUDA.NIF.loaded?() do
      raise "priv/libedifice_cuda_nif.so is not loaded; build with `make -C c_src` and `make -C native/cuda`"
    end

    # The NIF allocates with cudaMalloc outside XLA's pool; leave it room.
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

  # Reference: trace through the SAME dispatch entry point; inside defn the
  # tensors are expressions, so every tier check fails and the fallback runs.
  defp reference(fun, args) do
    apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)
  end

  defp on_gpu(fun, args) do
    args = Enum.map(args, &Nx.backend_transfer(&1, @cuda))
    apply(fun, args) |> Nx.backend_copy(Nx.BinaryBackend)
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

  # ── NIF-tier forward matrix ─────────────────────────────────────────

  @b 2
  @t 16
  @h 64
  @heads 2
  @d 16
  @tol [atol: 1.0e-4, rtol: 1.0e-4]

  # {name, generator, tolerance}. The generator returns {fun, args}; fun is
  # the public FusedScan entry point so dispatch itself is under test.
  @cases [
    {"mingru", :gen_two_gate, @tol},
    {"minlstm", :gen_minlstm, @tol},
    {"elu_gru", :gen_elu_gru, @tol},
    {"real_gru", :gen_real_gru, @tol},
    {"diag_linear", :gen_diag_linear, @tol},
    {"linear_scan", :gen_linear_scan, @tol},
    {"liquid", :gen_liquid, @tol},
    {"delta_net_scan", :gen_delta_net, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"gated_delta_net_scan", :gen_gated_delta_net, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"delta_product_scan", :gen_delta_product, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"slstm_scan", :gen_slstm, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"lstm_scan", :gen_lstm, @tol},
    {"gru_scan", :gen_gru, @tol},
    {"gru_scan H=512", :gen_gru_512, @tol},
    {"ttt_scan", :gen_ttt, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"selective_scan", :gen_selective, [atol: 1.0e-5, rtol: 1.0e-4]},
    {"kda_scan", :gen_kda, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"rla_scan (rla)", :gen_rla, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"rla_scan (rdn)", :gen_rdn, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"flash_attention causal", :gen_flash_causal, [atol: 1.0e-4, rtol: 1.0e-3]},
    {"flash_attention full", :gen_flash_full, [atol: 1.0e-4, rtol: 1.0e-3]},
    {"laser_attention", :gen_laser, [atol: 1.0e-4, rtol: 1.0e-3]},
    {"fox_attention", :gen_fox, [atol: 1.0e-4, rtol: 1.0e-3]},
    {"reservoir_scan", :gen_reservoir, @tol},
    {"titans_scan", :gen_titans, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"miras_scan", :gen_miras, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"gsa_scan", :gen_gsa, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"mingru_block", :gen_mingru_block, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"minlstm_block", :gen_minlstm_block, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"linear_block", :gen_linear_block, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"lstm_block", :gen_lstm_block, [atol: 2.0e-4, rtol: 1.0e-3]},
    {"gru_block", :gen_gru_block, [atol: 2.0e-4, rtol: 1.0e-3]}
  ]

  describe "NIF tier (f32) == Elixir fallback" do
    for {name, gen, tol} <- @cases do
      test name do
        {fun, args} =
          apply(__MODULE__, unquote(gen), [Nx.Random.key(unquote(:erlang.phash2(name)))])

        want = reference(fun, args)
        got = on_gpu(fun, args)
        assert_close(got, want, unquote(tol))
      end
    end
  end

  # ── bf16 through the bf16 kernel library ───────────────────────────

  @bf16_cases ~w(gen_two_gate gen_minlstm gen_elu_gru gen_real_gru gen_diag_linear gen_linear_scan gen_liquid gen_lstm gen_gru)a

  describe "NIF tier (bf16) == f32 fallback within bf16 precision" do
    for gen <- @bf16_cases do
      test "#{gen}" do
        {fun, args} = apply(__MODULE__, unquote(gen), [Nx.Random.key(7)])
        want = reference(fun, args)
        got = on_gpu(fun, Enum.map(args, &Nx.as_type(&1, {:bf, 16})))
        # bf16 inputs + bf16 I/O with f32 accumulation: ~3 significant digits.
        assert_close(got, want, atol: 2.0e-2, rtol: 2.0e-2)
      end
    end
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

  # ── input generators (public so the test macros can apply/3 them) ───

  def gen_two_gate(key) do
    {g, key} = rand(key, {@b, @t, @h}, -2.0, 2.0)
    {c, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.mingru/2, [g, c]}
  end

  def gen_minlstm(key) do
    {f, key} = rand(key, {@b, @t, @h}, 0.1, 0.9)
    {i, key} = rand(key, {@b, @t, @h}, 0.1, 0.9)
    {c, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.minlstm/3, [f, i, c]}
  end

  def gen_elu_gru(key) do
    {g, key} = rand(key, {@b, @t, @h}, -2.0, 2.0)
    {c, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.elu_gru/2, [g, c]}
  end

  def gen_real_gru(key) do
    {g, key} = rand(key, {@b, @t, @h}, -2.0, 2.0)
    {c, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.real_gru/2, [g, c]}
  end

  def gen_diag_linear(key) do
    {a, key} = rand(key, {@b, @t, @h}, -2.0, 2.0)
    {b, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.diag_linear/2, [a, b]}
  end

  def gen_linear_scan(key) do
    {a, key} = rand(key, {@b, @t, @h}, 0.0, 0.95)
    {b, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.linear_scan/2, [a, b]}
  end

  def gen_liquid(key) do
    {tau, key} = rand(key, {@b, @t, @h}, 0.5, 2.0)
    {act, _} = rand(key, {@b, @t, @h}, -1.0, 1.0)
    {&FusedScan.liquid/2, [tau, act]}
  end

  def gen_delta_net(key) do
    {q, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    # Kernel contract (fused_delta_rule_scan.cu): beta is per element [B,T,H,d].
    {beta, _} = rand(key, {@b, @t, @heads, @d}, 0.3, 0.7)
    {&FusedScan.delta_net_scan/4, [q, k, v, beta]}
  end

  def gen_gated_delta_net(key) do
    {q, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    # beta per element [B,T,H,d]; alpha per head [B,T,H] (kernel contract).
    {beta, key} = rand(key, {@b, @t, @heads, @d}, 0.3, 0.7)
    {alpha, _} = rand(key, {@b, @t, @heads}, 0.8, 0.99)
    {&FusedScan.gated_delta_net_scan/5, [q, k, v, beta, alpha]}
  end

  def gen_delta_product(key) do
    n_h = 2
    {q, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, n_h, @heads, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, n_h, @heads, @d}, -0.5, 0.5)
    {beta, _} = rand(key, {@b, @t, n_h, @heads}, 0.3, 0.7)
    {&FusedScan.delta_product_scan/4, [q, k, v, beta]}
  end

  def gen_slstm(key) do
    {wx, key} = rand(key, {@b, @t, 4 * @h}, -0.5, 0.5)
    {r, _} = rand(key, {@h, 4 * @h}, -0.2, 0.2)
    {&FusedScan.slstm_scan/2, [wx, r]}
  end

  def gen_lstm(key) do
    {wx, key} = rand(key, {@b, @t, 4 * @h}, -0.5, 0.5)
    {r, _} = rand(key, {@h, 4 * @h}, -0.2, 0.2)
    {&FusedScan.lstm_scan/2, [wx, r]}
  end

  def gen_gru(key), do: gen_gru_h(key, @h)
  def gen_gru_512(key), do: gen_gru_h(key, 512)

  defp gen_gru_h(key, hidden) do
    {wx, key} = rand(key, {@b, @t, 3 * hidden}, -0.5, 0.5)
    {r, key} = rand(key, {hidden, 3 * hidden}, -0.2, 0.2)
    {h0, key} = rand(key, {@b, hidden}, -0.5, 0.5)
    {bhn, _} = rand(key, {hidden}, -0.3, 0.3)
    {&FusedScan.gru_scan/4, [wx, r, h0, bhn]}
  end

  def gen_ttt(key) do
    inner = 16
    {q, key} = rand(key, {@b, @t, inner}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, inner}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, inner}, -0.5, 0.5)
    {eta, key} = rand(key, {@b, @t, inner}, 0.01, 0.1)
    {w0, _} = rand(key, {inner, inner}, -0.1, 0.1)
    g = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {inner})
    bt = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {inner})
    {&FusedScan.ttt_scan/7, [q, k, v, eta, w0, g, bt]}
  end

  def gen_selective(key) do
    state = 16
    {x, key} = rand(key, {@b, @t, @h}, -0.5, 0.5)
    {dt, key} = rand(key, {@b, @t, @h}, 0.001, 0.1)
    {b, key} = rand(key, {@b, @t, state}, -0.5, 0.5)
    {c, _} = rand(key, {@b, @t, state}, -0.5, 0.5)
    {&FusedScan.selective_scan/5, [x, dt, canonical_a(state), b, c]}
  end

  def gen_kda(key) do
    {q, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {alpha, key} = rand(key, {@b, @t, @heads, @d}, -0.3, 0.0)
    {beta, _} = rand(key, {@b, @t, @heads}, 0.3, 0.7)
    {&FusedScan.kda_scan/5, [q, k, v, alpha, beta]}
  end

  def gen_rla(key), do: gen_rla_variant(key, :rla)
  def gen_rdn(key), do: gen_rla_variant(key, :rdn)

  defp gen_rla_variant(key, variant) do
    {q, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {alpha, key} = rand(key, {@b, @t, @heads}, 0.8, 0.95)
    {beta, key} = rand(key, {@b, @t, @heads}, 0.1, 0.3)
    {gamma, _} = rand(key, {@b, @t, @heads}, 0.05, 0.15)

    {fn q, k, v, alpha, beta, gamma ->
       FusedScan.rla_scan(q, k, v, alpha, beta, gamma, variant: variant, clip_threshold: 1.0)
     end, [q, k, v, alpha, beta, gamma]}
  end

  def gen_flash_causal(key), do: gen_flash(key, true)
  def gen_flash_full(key), do: gen_flash(key, false)

  defp gen_flash(key, causal) do
    {q, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {v, _} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {fn q, k, v -> FusedScan.flash_attention(q, k, v, causal: causal) end, [q, k, v]}
  end

  def gen_laser(key) do
    {q, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {v, _} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {fn q, k, v -> FusedScan.laser_attention(q, k, v, causal: true) end, [q, k, v]}
  end

  def gen_fox(key) do
    {q, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {k, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {v, key} = rand(key, {@b, @heads, @t, @d}, -0.5, 0.5)
    {cs, _} = rand(key, {@b, @heads, @t}, -2.0, 0.0)
    {&FusedScan.fox_attention/4, [q, k, v, cs]}
  end

  def gen_reservoir(key) do
    {wx, key} = rand(key, {@b, @t, @h}, -0.5, 0.5)
    {w_res, _} = rand(key, {@h, @h}, -0.3, 0.3)
    {fn wx, w_res -> FusedScan.reservoir_scan(wx, w_res, leak_rate: 0.5) end, [wx, w_res]}
  end

  def gen_titans(key) do
    m = 16
    {combined, _} = rand(key, {@b, @t, 4 * m}, -0.5, 0.5)
    {fn c -> FusedScan.titans_scan(c, memory_size: m, momentum: 0.9) end, [combined]}
  end

  def gen_miras(key) do
    m = 16
    {combined, _} = rand(key, {@b, @t, 5 * m}, -0.5, 0.5)
    {fn c -> FusedScan.miras_scan(c, memory_size: m, momentum: 0.9) end, [combined]}
  end

  def gen_gsa(key) do
    slots = 8
    {q, key} = rand(key, {@b, @t, @heads, @d}, 0.5, 1.5)
    {k_logits, key} = rand(key, {@b, @t, @heads, slots}, -1.0, 1.0)
    {v, key} = rand(key, {@b, @t, @heads, @d}, -0.5, 0.5)
    {alpha, _} = rand(key, {@b, @t, @heads}, 0.8, 0.99)
    k_slot = Nx.divide(Nx.exp(k_logits), Nx.sum(Nx.exp(k_logits), axes: [3], keep_axes: true))
    {&FusedScan.gsa_scan/4, [q, k_slot, v, alpha]}
  end

  # Block scans: packed per-layer weights, layouts from FusedScan's fallbacks.
  @bh 32
  @layers 2

  defp packed(key, segments) do
    {parts, key} =
      Enum.map_reduce(segments, key, fn
        {:w, n}, k -> rand(k, {n}, -0.1, 0.1)
        {:b, n}, k -> rand(k, {n}, -0.01, 0.01)
        {:gamma, n}, k -> {Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {n}), k}
        {:beta, n}, k -> {Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {n}), k}
      end)

    {Nx.concatenate(parts), key}
  end

  defp packed_layers(key, segments) do
    {layers, key} = Enum.map_reduce(1..@layers, key, fn _, k -> packed(k, segments) end)
    {Nx.concatenate(layers), key}
  end

  def gen_mingru_block(key) do
    h = @bh
    {input, key} = rand(key, {@b, @t, h}, -1.0, 1.0)
    {h0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {w, _} = packed_layers(key, w: h * h, b: h, w: h * h, b: h, gamma: h, beta: h)
    {fn i, w, h0 -> FusedScan.mingru_block(i, w, h0, @layers) end, [input, w, h0]}
  end

  def gen_minlstm_block(key) do
    h = @bh
    {input, key} = rand(key, {@b, @t, h}, -1.0, 1.0)
    {h0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {w, _} = packed_layers(key, w: h * h, b: h, w: h * h, b: h, w: h * h, b: h, gamma: h, beta: h)
    {fn i, w, h0 -> FusedScan.minlstm_block(i, w, h0, @layers) end, [input, w, h0]}
  end

  def gen_linear_block(key) do
    h = @bh
    {input, key} = rand(key, {@b, @t, h}, -1.0, 1.0)
    {h0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {w, _} = packed_layers(key, w: h * h, b: h, w: h * h, b: h, gamma: h, beta: h)
    {fn i, w, h0 -> FusedScan.linear_block(i, w, h0, @layers) end, [input, w, h0]}
  end

  def gen_lstm_block(key) do
    h = @bh
    {input, key} = rand(key, {@b, @t, h}, -1.0, 1.0)
    {h0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {c0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {w, _} = packed_layers(key, w: 4 * h * h, b: 4 * h, w: 4 * h * h, gamma: h, beta: h)
    {fn i, w, h0, c0 -> FusedScan.lstm_block(i, w, h0, c0, @layers) end, [input, w, h0, c0]}
  end

  def gen_gru_block(key) do
    h = @bh
    {input, key} = rand(key, {@b, @t, h}, -1.0, 1.0)
    {h0, key} = rand(key, {@b, @layers, h}, -0.1, 0.1)
    {w, _} = packed_layers(key, w: 3 * h * h, b: 3 * h, w: 3 * h * h, gamma: h, beta: h)
    {fn i, w, h0 -> FusedScan.gru_block(i, w, h0, @layers) end, [input, w, h0]}
  end
end
