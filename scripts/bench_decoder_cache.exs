# Run: devenv shell -- mix run scripts/bench_decoder_cache.exs
# Measures host EXLA latency including the cache bounds check and output sync.
alias Edifice.Transformer.DecoderOnly

opts = [
  embed_dim: 64,
  hidden_size: 64,
  num_heads: 4,
  num_kv_heads: 2,
  num_layers: 2,
  seq_len: 60,
  dropout: 0.0
]

model = DecoderOnly.build(opts)
{init, predict} = Axon.build(model, seed: 42, compiler: EXLA, client: :host)
x = Nx.sin(Nx.iota({1, 60, 64}, type: :f32))
input = %{"state_sequence" => x}
params = init.(input, Axon.ModelState.empty())
cache = DecoderOnly.Cache.build(opts ++ [compiler: EXLA, client: :host])
{_, state} = cache.predict.(params, Nx.slice_along_axis(x, 0, 59, axis: 1), cache.init.(1))
frame = Nx.slice_along_axis(x, 59, 1, axis: 1)
{out, _} = cache.predict.(params, frame, state)
expected = predict.(params, input)
Nx.Testing.assert_all_close(Nx.squeeze(out, axes: [1]), expected, atol: 1.0e-5)

measure = fn fun ->
  for _ <- 1..5, do: fun.() |> Nx.sum() |> Nx.to_number()

  timings =
    for _ <- 1..100 do
      {us, _} = :timer.tc(fn -> fun.() |> Nx.sum() |> Nx.to_number() end)
      us
    end
    |> Enum.sort()

  %{median_us: Enum.at(timings, 50), p95_us: Enum.at(timings, 95)}
end

full = measure.(fn -> predict.(params, input) end)

cached =
  measure.(fn ->
    {out, _} = cache.predict.(params, frame, state)
    out
  end)

IO.puts(
  Jason.encode!(%{
    client: "host",
    batch: 1,
    window: 60,
    hidden: 64,
    layers: 2,
    full: full,
    cached: cached,
    median_speedup: full.median_us / cached.median_us
  })
)
