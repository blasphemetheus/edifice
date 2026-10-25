# mix run scripts/profile_recurrent_resync.exs --client cuda --out /tmp/resync.json
# Synchronized stage timings and selected-row replay prototype, resident inputs.
defmodule ResyncProfile do
  def sync(%Nx.Tensor{} = t), do: Nx.to_binary(t)
  def sync(m) when is_map(m), do: Enum.each(m, fn {_, v} -> sync(v) end)
  def map(%Nx.Tensor{} = t, f), do: f.(t)
  def map(m, f), do: Map.new(m, fn {k, v} -> {k, map(v, f)} end)

  def merge(current, replayed, mask) do
    Map.new(current, fn {k, v} ->
      shape = List.to_tuple([Nx.axis_size(v, 0) | List.duplicate(1, Nx.rank(v) - 1)])
      {k, Nx.select(Nx.broadcast(Nx.reshape(mask, shape), Nx.shape(v)), replayed[k], v)}
    end)
  end

  def measure(f, n) do
    for _ <- 1..4, do: sync(f.())

    times =
      for _ <- 1..n do
        {us, _} = :timer.tc(fn -> sync(f.()) end)
        us
      end
      |> Enum.sort()

    %{median_us: Enum.at(times, div(n, 2)), p95_us: Enum.at(times, min(n - 1, floor(n * 0.95)))}
  end

  def run(arch, hidden, window, batch, client, iterations) do
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
    initial_one = Edifice.init_state(arch, params, opts)

    initial =
      map(initial_one, &Nx.broadcast(&1, put_elem(Nx.shape(&1), 0, batch)))
      |> Nx.backend_copy({EXLA.Backend, client: client})

    {sequence, _} = Nx.Random.normal(Nx.Random.key(43), shape: {batch, window - 1, 264})
    sequence = Nx.backend_copy(sequence, {EXLA.Backend, client: client})
    module = Edifice.module_for(arch)
    replay = Edifice.Stateful.jit_replay(module, EXLA, Keyword.delete(compiler, :compiler))
    step = Edifice.Stateful.jit_step(module, EXLA, Keyword.delete(compiler, :compiler))
    {_, current} = step.(params, initial, Nx.take(sequence, 0, axis: 1))
    history = for i <- 0..(window - 2), do: Nx.take(sequence, i, axis: 1)
    # Profile the real list->tensor assembly used by Agent; synchronize each stage.
    assembly = measure(fn -> Nx.stack(history, axis: 1) end, iterations)
    full_replay = measure(fn -> replay.(params, initial, sequence) end, iterations)
    replayed = replay.(params, initial, sequence)
    selected = Edifice.Stateful.jit_replay_rows(module, EXLA, Keyword.delete(compiler, :compiler))

    fused_full =
      Nx.Defn.jit(
        fn p, c, s, f, m ->
          merge(c, Edifice.Stateful.Replay.run(p, s, f, module: module), m)
        end,
        compiler
      )

    history_tuple = List.to_tuple(history)

    full_history =
      Nx.Defn.jit(
        fn p, c, s, h, m ->
          merge(
            c,
            Edifice.Stateful.Replay.run(p, s, Nx.stack(Tuple.to_list(h), axis: 1), module: module),
            m
          )
        end,
        compiler
      )

    rows =
      for count <- Enum.uniq([1, min(3, batch), min(4, batch), batch]) do
        bucket = min(Stream.iterate(1, &(&1 * 2)) |> Enum.find(&(&1 >= count)), batch)
        indices = Nx.tensor(Enum.take(Enum.reverse(0..(batch - 1)), bucket), type: :s32)
        count_tensor = Nx.tensor(count, type: :s32)

        mask =
          Nx.tensor(for(i <- 0..(batch - 1), do: if(i >= batch - count, do: 1, else: 0)),
            type: :u8
          )

        expected = merge(current, replayed, mask)
        actual = selected.(params, current, initial, history_tuple, indices, count_tensor)

        Enum.each(actual, fn {key, value} ->
          Nx.Testing.assert_all_close(value, expected[key], atol: 1.0e-4, rtol: 1.0e-4)

          if count < batch do
            untouched = Nx.slice_along_axis(value, 0, batch - count, axis: 0)
            original = Nx.slice_along_axis(current[key], 0, batch - count, axis: 0)

            if Nx.to_binary(untouched) != Nx.to_binary(original),
              do: raise("unchanged row modified")
          end
        end)

        current_total = fn ->
          result = replay.(params, initial, Nx.stack(history, axis: 1))
          if count == batch, do: result, else: merge(current, result, mask)
        end

        optimized = fn ->
          if count == batch,
            do: replay.(params, initial, history_tuple),
            else: selected.(params, current, initial, history_tuple, indices, count_tensor)
        end

        row = %{
          due: count,
          bucket: bucket,
          eager_merge: measure(fn -> merge(current, replayed, mask) end, iterations),
          current_total: measure(current_total, iterations),
          fused_full:
            measure(fn -> fused_full.(params, current, initial, sequence, mask) end, iterations),
          selected:
            measure(
              fn -> selected.(params, current, initial, sequence, indices, count_tensor) end,
              iterations
            ),
          fused_history_total:
            measure(
              fn -> full_history.(params, current, initial, history_tuple, mask) end,
              iterations
            ),
          optimized_total: measure(optimized, iterations),
          selected_total:
            measure(
              fn ->
                selected.(
                  params,
                  current,
                  initial,
                  Nx.stack(history, axis: 1),
                  indices,
                  count_tensor
                )
              end,
              iterations
            )
        }

        IO.inspect({arch, batch, row})
        row
      end

    %{
      architecture: arch,
      hidden: hidden,
      window: window,
      batch: batch,
      assembly: assembly,
      full_replay: full_replay,
      rows: rows
    }
  end
end

{opts, [], []} =
  OptionParser.parse(System.argv(),
    strict: [client: :string, out: :string, iterations: :integer, batch: :integer]
  )

client =
  case opts[:client] || "host" do
    "host" -> :host
    "cuda" -> :cuda
  end

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, memory_fraction: 0.15, preallocate: false]
)

Application.put_env(:exla, :default_client, client)
Nx.default_backend({EXLA.Backend, client: client})
Nx.Defn.default_options(compiler: EXLA, client: client, precision: :highest)
Process.put(:__edifice_force_fallback__, true)
iterations = opts[:iterations] || 30

results =
  for {arch, hidden, window} <- [
        {:min_gru, 256, 80},
        {:gru, 512, 60},
        {:mamba, 512, 80},
        {:mamba_ssd, 256, 60}
      ],
      do: ResyncProfile.run(arch, hidden, window, opts[:batch] || 32, client, iterations)

File.write!(
  Keyword.fetch!(opts, :out),
  Jason.encode!(
    %{
      client: client,
      precision: "highest",
      iterations: iterations,
      note:
        "Synthetic representative weights. Resident inputs; stage timings synchronize independently and are not additive. Current total includes history assembly, full-batch replay, eager row merge. Selected total includes assembly, fused gather/replay/scatter. Compilation excluded. Selected outputs checked against full replay; untouched rows bitwise identical.",
      results: results
    },
    pretty: true
  )
)
