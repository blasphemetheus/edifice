# NIF Bridge Overhead Profiler
#
# Measures where time is spent in the fused scan path:
#   1. Full model: Elixir scan (on GPU via XLA)
#   2. Full model: CUDA fused scan (NIF bridge)
#   3. Isolated: Nx.to_pointer extraction
#   4. Isolated: NIF kernel call
#   5. Isolated: Nx.from_pointer wrapping
#
# Usage: nix-shell --run "mix run bench/nif_overhead_profile.exs" shell.nix

Nx.default_backend(EXLA.Backend)

defmodule NIFOverheadProfile do
  @warmup 5
  @iters 50

  defp rand(shape) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {t, _} = Nx.Random.uniform(key, -2.0, 2.0, shape: shape)
    t
  end

  defp measure(label, fun) do
    for _ <- 1..@warmup, do: fun.()

    times =
      for _ <- 1..@iters do
        {us, _} = :timer.tc(fun)
        us
      end

    times = Enum.sort(times)
    min = hd(times)
    med = Enum.at(times, div(length(times), 2))
    mean = Enum.sum(times) / length(times)

    IO.puts(
      "  #{String.pad_trailing(label, 45)}" <>
        "min=#{pad(min)}  med=#{pad(med)}  mean=#{pad(mean)}"
    )

    med
  end

  defp pad(us) do
    ms = us / 1000
    String.pad_leading("#{Float.round(ms, 3)} ms", 12)
  end

  def run do
    batch = 1
    seq_len = 32
    hidden = 256

    IO.puts("=" |> String.duplicate(90))
    IO.puts("NIF Bridge Overhead Profile — B=#{batch} T=#{seq_len} H=#{hidden}")
    IO.puts("#{@iters} iterations, #{@warmup} warmup")
    IO.puts("=" |> String.duplicate(90))

    # GPU warmup
    _w = Nx.add(rand({2, 2}), rand({2, 2}))

    model_opts = [
      embed_dim: hidden, hidden_size: hidden,
      num_layers: 2, dropout: 0.0, seq_len: seq_len
    ]

    input = rand({batch, seq_len, hidden})

    # ── Section 1: Full model comparison ────────────────────────────
    IO.puts("\n## Full Model Forward Pass (MinGRU, 2 layers)")
    IO.puts("-" |> String.duplicate(80))

    # Force Elixir fallback by temporarily checking
    gru_model = Edifice.Recurrent.MinGRU.build(model_opts)
    {init_fn, predict_fn} = Axon.build(gru_model)
    params = init_fn.(Nx.template({batch, seq_len, hidden}, :f32), Axon.ModelState.empty())

    # Warmup XLA compilation
    for _ <- 1..5, do: predict_fn.(params, input)

    fused_med = measure("Full model (current dispatch)", fn ->
      predict_fn.(params, input)
    end)

    # ── Section 2: Isolate pointer overhead ─────────────────────────
    IO.puts("\n## Pointer Extraction / Wrapping Overhead")
    IO.puts("-" |> String.duplicate(80))

    gates = rand({batch, seq_len, hidden})
    candidates = rand({batch, seq_len, hidden})

    # Force materialization
    _ = Nx.to_number(gates[0][0][0])

    measure("Nx.to_pointer (gates)", fn ->
      Nx.to_pointer(gates, mode: :local)
    end)

    measure("Nx.to_pointer (candidates)", fn ->
      Nx.to_pointer(candidates, mode: :local)
    end)

    z = Nx.sigmoid(gates)
    h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: EXLA.Backend), {batch, hidden})

    # Force materialization
    _ = Nx.to_number(z[0][0][0])
    _ = Nx.to_number(h0[0][0])

    measure("Nx.sigmoid (XLA)", fn ->
      Nx.sigmoid(gates)
    end)

    z_ptr = Nx.to_pointer(z, mode: :local)
    c_ptr = Nx.to_pointer(candidates, mode: :local)
    h0_ptr = Nx.to_pointer(h0, mode: :local)

    measure("NIF kernel call only", fn ->
      {:ok, _out, _gc} = Edifice.CUDA.NIF.fused_mingru_scan(
        z_ptr.address, c_ptr.address, h0_ptr.address,
        batch, seq_len, hidden
      )
    end)

    {:ok, out_addr, gc_ref} = Edifice.CUDA.NIF.fused_mingru_scan(
      z_ptr.address, c_ptr.address, h0_ptr.address,
      batch, seq_len, hidden
    )

    out_bytes = batch * seq_len * hidden * 4

    measure("Nx.from_pointer (wrap output)", fn ->
      Nx.from_pointer(
        {EXLA.Backend, client: :cuda, device_id: 0},
        %Nx.Pointer{kind: :local, address: out_addr, data_size: out_bytes},
        {:f, 32}, {batch, seq_len, hidden}
      )
    end)

    # Keep gc_ref alive
    _ = gc_ref

    # ── Section 3: End-to-end raw scan comparison ───────────────────
    IO.puts("\n## Raw Scan: Fused vs Elixir (pre-computed gates)")
    IO.puts("-" |> String.duplicate(80))

    elixir_med = measure("Elixir Enum.reduce scan", fn ->
      Edifice.Recurrent.MinGRU.min_gru_scan(gates, candidates)
    end)

    fused_raw_med = measure("CUDA fused (full FusedScan.mingru)", fn ->
      Edifice.CUDA.FusedScan.mingru(gates, candidates)
    end)

    IO.puts("\n  Raw scan speedup: #{Float.round(elixir_med / max(fused_raw_med, 1), 1)}x")
    IO.puts("  Full model median: #{Float.round(fused_med / 1000, 1)} ms")
    IO.puts("  Raw fused median:  #{Float.round(fused_raw_med / 1000, 3)} ms")
    IO.puts("  Overhead ratio:    #{Float.round(fused_med / max(fused_raw_med, 1), 1)}x (model vs raw)")

    IO.puts("\nDone.")
  end
end

NIFOverheadProfile.run()
