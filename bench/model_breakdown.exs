# Model Layer Breakdown — where does MinGRU spend its time?
#
# Builds the model in pieces to isolate:
#   1. Dense projections (gate + candidate)
#   2. Layer norm
#   3. Fused scan
#   4. Residual add
#   5. Full single layer
#   6. Full 2-layer model
#
# Usage: nix-shell --run "mix run bench/model_breakdown.exs" shell.nix

Nx.default_backend(EXLA.Backend)

defmodule ModelBreakdown do
  @warmup 10
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
        {us, _} = :timer.tc(fn ->
          r = fun.()
          # Force materialization to include GPU execution time
          r |> Nx.backend_transfer(Nx.BinaryBackend)
        end)
        us
      end

    times = Enum.sort(times)
    med = Enum.at(times, div(length(times), 2))
    min = hd(times)

    IO.puts(
      "  #{String.pad_trailing(label, 45)}" <>
        "min=#{pad(min)}  med=#{pad(med)}"
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

    IO.puts("=" |> String.duplicate(80))
    IO.puts("MinGRU Model Layer Breakdown — B=#{batch} T=#{seq_len} H=#{hidden}")
    IO.puts("#{@iters} iters, #{@warmup} warmup, Nx.to_number sync after each")
    IO.puts("=" |> String.duplicate(80))

    _w = Nx.add(rand({2, 2}), rand({2, 2}))

    input = rand({batch, seq_len, hidden})

    # ── Individual Axon layer components ────────────────────────────
    IO.puts("\n## Individual Components (Axon layers, compiled)")
    IO.puts("-" |> String.duplicate(70))

    # Layer norm only
    ln_model = Axon.input("x", shape: {nil, nil, hidden}) |> Axon.layer_norm(name: "ln")
    {ln_init, ln_pred} = Axon.build(ln_model)
    ln_params = ln_init.(%{"x" => Nx.template({batch, seq_len, hidden}, :f32)}, Axon.ModelState.empty())
    for _ <- 1..5, do: ln_pred.(ln_params, %{"x" => input})

    measure("LayerNorm only", fn -> ln_pred.(ln_params, %{"x" => input}) end)

    # Dense only (one projection)
    dense_model = Axon.input("x", shape: {nil, nil, hidden}) |> Axon.dense(hidden, name: "d")
    {d_init, d_pred} = Axon.build(dense_model)
    d_params = d_init.(%{"x" => Nx.template({batch, seq_len, hidden}, :f32)}, Axon.ModelState.empty())
    for _ <- 1..5, do: d_pred.(d_params, %{"x" => input})

    measure("Dense(256->256) only", fn -> d_pred.(d_params, %{"x" => input}) end)

    # Two dense projections (gate + candidate)
    two_dense_model =
      Axon.input("x", shape: {nil, nil, hidden})
      |> then(fn inp ->
        a = Axon.dense(inp, hidden, name: "gate")
        b = Axon.dense(inp, hidden, name: "cand")
        Axon.container({a, b})
      end)
    {td_init, td_pred} = Axon.build(two_dense_model)
    td_params = td_init.(%{"x" => Nx.template({batch, seq_len, hidden}, :f32)}, Axon.ModelState.empty())
    for _ <- 1..5, do: td_pred.(td_params, %{"x" => input})

    measure("Two Dense (gate+cand)", fn -> td_pred.(td_params, %{"x" => input}) end)

    # LN + two dense
    ln_dense_model =
      Axon.input("x", shape: {nil, nil, hidden})
      |> then(fn inp ->
        normed = Axon.layer_norm(inp, name: "ln")
        a = Axon.dense(normed, hidden, name: "gate")
        b = Axon.dense(normed, hidden, name: "cand")
        Axon.container({a, b})
      end)
    {ld_init, ld_pred} = Axon.build(ln_dense_model)
    ld_params = ld_init.(%{"x" => Nx.template({batch, seq_len, hidden}, :f32)}, Axon.ModelState.empty())
    for _ <- 1..5, do: ld_pred.(ld_params, %{"x" => input})

    measure("LN + Two Dense", fn -> ld_pred.(ld_params, %{"x" => input}) end)

    # ── Scan alone via Axon.layer ───────────────────────────────────
    IO.puts("\n## Scan via Axon.layer callback")
    IO.puts("-" |> String.duplicate(70))

    scan_model =
      Axon.input("x", shape: {nil, nil, hidden})
      |> then(fn inp ->
        gates = Axon.dense(inp, hidden, name: "gate")
        cands = Axon.dense(inp, hidden, name: "cand")

        Axon.layer(
          fn g, c, _opts -> Edifice.CUDA.FusedScan.mingru(g, c) end,
          [gates, cands],
          name: "scan"
        )
      end)
    {s_init, s_pred} = Axon.build(scan_model)
    s_params = s_init.(%{"x" => Nx.template({batch, seq_len, hidden}, :f32)}, Axon.ModelState.empty())
    for _ <- 1..5, do: s_pred.(s_params, %{"x" => input})

    measure("Dense+Dense+FusedScan (1 layer core)", fn ->
      s_pred.(s_params, %{"x" => input})
    end)

    # ── Full model ──────────────────────────────────────────────────
    IO.puts("\n## Full MinGRU Models")
    IO.puts("-" |> String.duplicate(70))

    for layers <- [1, 2] do
      model_opts = [
        embed_dim: hidden, hidden_size: hidden,
        num_layers: layers, dropout: 0.0, seq_len: seq_len
      ]

      model = Edifice.Recurrent.MinGRU.build(model_opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({batch, seq_len, hidden}, :f32), Axon.ModelState.empty())
      for _ <- 1..5, do: predict_fn.(params, input)

      measure("Full MinGRU (#{layers} layer#{if layers > 1, do: "s"})", fn ->
        predict_fn.(params, input)
      end)
    end

    IO.puts("\nDone.")
  end
end

ModelBreakdown.run()
