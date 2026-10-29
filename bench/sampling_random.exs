# devenv shell -- mix run bench/sampling_random.exs host|cuda OUTPUT.json
[client, out] = System.argv()
client = String.to_existing_atom(client)
Application.ensure_all_started(:exla)

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, memory_fraction: 0.02, preallocate: false]
)

Nx.default_backend({EXLA.Backend, client: client})
Logger.configure(level: :warning)

results =
  for shape <- [{1, 8}, {1, 17}, {8, 17}, {128, 17}] do
    reference =
      Nx.Defn.jit(fn key -> Nx.Random.uniform(key, shape: shape) end,
        compiler: EXLA,
        client: client
      )

    unrolled =
      Nx.Defn.jit(fn key -> Edifice.Utils.SamplingRandom.uniform(key, shape: shape) end,
        compiler: EXLA,
        client: client
      )

    key = Nx.Random.key(2026)

    for seed <- [0, 1, 42, 0xFFFFFFFFFFFFFFFF] do
      {x, k} = reference.(Nx.Random.key(seed))
      {y, l} = unrolled.(Nx.Random.key(seed))
      true = Nx.to_binary(x) == Nx.to_binary(y)
      true = Nx.to_binary(k) == Nx.to_binary(l)
    end

    for {label, fun} <- [reference: reference, unrolled: unrolled] do
      times =
        for _ <- 1..200 do
          {us, _} =
            :timer.tc(fn ->
              {x, k} = fun.(key)
              {Nx.to_binary(x), Nx.to_binary(k)}
            end)

          us
        end
        |> Enum.sort()

      result = %{
        shape: Tuple.to_list(shape),
        implementation: label,
        median_us: Enum.at(times, 100),
        p95_us: Enum.at(times, 189)
      }

      IO.inspect(result)
      result
    end
  end

File.write!(
  out,
  Jason.encode!(%{client: client, exact_reference_parity: true, results: List.flatten(results)},
    pretty: true
  )
)
