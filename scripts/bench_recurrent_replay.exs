# mix run scripts/bench_recurrent_replay.exs --client host --out /tmp/replay.json
# Backbone-only timings: resident weights/frames, warmed executables, synchronized
# final carry. Includes dispatch overhead, excludes embedding and action heads.
defmodule ReplayBench do
  def sync(%Nx.Tensor{} = t), do: Nx.to_binary(t)
  def sync(m) when is_map(m), do: Enum.each(m, fn {_, v} -> sync(v) end)

  def close(a, b) do
    Enum.each(a, fn {k, v} ->
      if is_struct(v, Nx.Tensor),
        do: Nx.Testing.assert_all_close(v, b[k], atol: 1.0e-4, rtol: 1.0e-4),
        else: close(v, b[k])
    end)
  end

  def measure(f, n) do
    for _ <- 1..5, do: sync(f.())

    times =
      for _ <- 1..n do
        {us, _} = :timer.tc(fn -> sync(f.()) end)
        us
      end
      |> Enum.sort()

    %{median_us: Enum.at(times, div(n, 2)), p95_us: Enum.at(times, min(n - 1, floor(n * 0.95)))}
  end

  def run(arch, hidden, window, client, iterations) do
    opts = [
      embed_dim: 264,
      hidden_size: hidden,
      num_layers: 2,
      state_size: 16,
      conv_size: 4,
      window_size: window,
      dropout: 0.0
    ]

    compiler = [compiler: EXLA, client: client, precision: :highest]
    {init, _} = Axon.build(Edifice.build(arch, opts), compiler ++ [seed: 42])
    {x, _} = Nx.Random.normal(Nx.Random.key(42), shape: {1, window, 264})
    params = init.(x, Axon.ModelState.empty()) |> Nx.backend_copy({EXLA.Backend, client: client})

    initial =
      Edifice.init_state(arch, params, opts) |> Nx.backend_copy({EXLA.Backend, client: client})

    module = Edifice.module_for(arch)
    step = Edifice.Stateful.jit_step(module, EXLA, Keyword.delete(compiler, :compiler))
    replay = Edifice.Stateful.jit_replay(module, EXLA, Keyword.delete(compiler, :compiler))
    {_, carry} = step.(params, initial, Nx.take(x, 0, axis: 1))

    results =
      for {kind, sequence, state} <- [
            {:cold_start,
             Nx.broadcast(Nx.slice_along_axis(x, 0, 1, axis: 1), {1, window - 1, 264}), initial},
            {:resync, Nx.slice_along_axis(x, 1, window - 1, axis: 1), initial},
            {:continued_chunk, Nx.slice_along_axis(x, 1, window - 1, axis: 1), carry}
          ] do
        frames = for i <- 0..(window - 2), do: Nx.take(sequence, i, axis: 1)
        old = fn -> Enum.reduce(frames, state, fn f, s -> elem(step.(params, s, f), 1) end) end
        new = fn -> replay.(params, state, sequence) end
        close(old.(), new.())
        sequential = measure(old, iterations)
        compiled = measure(new, iterations)

        %{
          kind: kind,
          sequential: sequential,
          compiled: compiled,
          speedup: sequential.median_us / max(compiled.median_us, 1)
        }
      end

    row = %{architecture: arch, hidden: hidden, window: window, results: results}
    IO.inspect(row)
    row
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [client: :string, iterations: :integer, out: :string])

client =
  case Keyword.get(opts, :client, "host") do
    "host" -> :host
    "cuda" -> :cuda
  end

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, memory_fraction: 0.15]
)

Application.put_env(:exla, :default_client, client)
Nx.default_backend({EXLA.Backend, client: client})
Nx.Defn.default_options(compiler: EXLA, client: client, precision: :highest)
Process.put(:__edifice_force_fallback__, true)
iterations = Keyword.get(opts, :iterations, 40)

results =
  for {arch, hidden, window} <- [
        {:min_gru, 256, 80},
        {:gru, 512, 60},
        {:mamba, 512, 80},
        {:mamba_ssd, 256, 60}
      ],
      do: ReplayBench.run(arch, hidden, window, client, iterations)

File.write!(
  Keyword.fetch!(opts, :out),
  Jason.encode!(
    %{
      client: client,
      precision: "highest",
      embed: 264,
      batch: 1,
      iterations: iterations,
      note:
        "Synthetic weights, resident inputs, backbone-only replay; no embedding/head/IPC time. All final state leaves checked against sequential calls before timing.",
      results: results
    },
    pretty: true
  )
)
