# Run one client at a time; no training jobs should share the benchmark GPU.
# devenv shell -- mix run scripts/bench_recurrent_inference.exs --client host --output /tmp/recurrent-host.json
# Options: --client host|cuda --iterations 200 --embed 516 --arches gru,mamba,min_gru
# Add --aot to compare direct shape-specialized executables with the JIT wrapper.
{cli, _, []} =
  OptionParser.parse(System.argv(),
    strict: [
      client: :string,
      iterations: :integer,
      embed: :integer,
      arches: :string,
      aot: :boolean,
      native: :boolean,
      output: :string
    ]
  )

client =
  case Keyword.get(cli, :client, "host") do
    "host" -> :host
    "cuda" -> :cuda
  end

clients = Application.fetch_env!(:exla, :clients)

Application.put_env(
  :exla,
  :clients,
  Keyword.update!(clients, :cuda, &Keyword.put(&1, :memory_fraction, 0.5))
)

Nx.default_backend({EXLA.Backend, client: client})
Nx.Defn.default_options(compiler: EXLA, client: client, precision: :highest)
# Compare portable implementations by default. Native mode requires an existing
# library and explicitly enabled custom calls; it never builds or replaces EXLA.
native? = Keyword.get(cli, :native, false)

if native? and (client != :cuda or System.get_env("EDIFICE_FUSED_CUSTOM_CALL") != "1"),
  do: raise("--native requires --client cuda and EDIFICE_FUSED_CUSTOM_CALL=1")

Process.put(:__edifice_force_fallback__, not native?)
Process.put(:__edifice_benchmarking__, native?)

defmodule RecurrentInferenceBench do
  def sync(x), do: Nx.to_binary(x)

  def measure(fun, n) do
    for _ <- 1..10, do: sync(fun.())

    times =
      for _ <- 1..n do
        {us, _} = :timer.tc(fn -> sync(fun.()) end)
        us
      end
      |> Enum.sort()

    %{
      median_us: Enum.at(times, div(n, 2)),
      p95_us: Enum.at(times, min(n - 1, floor(n * 0.95))),
      min_us: hd(times),
      iterations: n
    }
  end

  def run({arch, hidden, window, extra}, embed, client, n, aot?) do
    opts =
      [
        embed_dim: embed,
        hidden_size: hidden,
        num_layers: 2,
        seq_len: window,
        window_size: window,
        dropout: 0.0,
        state_size: 16,
        conv_size: 4
      ] ++ extra

    IO.puts(:stderr, "Benchmark #{arch} hidden=#{hidden} window=#{window} #{client}")

    try do
      model =
        if arch == :mlp do
          Edifice.build(:mlp, input_size: embed, hidden_sizes: [hidden, hidden], dropout: 0.0)
        else
          Edifice.build(arch, opts)
        end

      {init, predict} =
        Axon.build(model,
          mode: :inference,
          seed: 42,
          compiler: EXLA,
          client: client,
          precision: :highest
        )

      {x, _} = Nx.Random.uniform(Nx.Random.key(42), -1.0, 1.0, shape: {1, window, embed})
      frame = Nx.take(x, window - 1, axis: 1)
      input = if arch == :mlp, do: frame, else: x

      params =
        init.(input, Axon.ModelState.empty()) |> Nx.backend_copy({EXLA.Backend, client: client})

      {compile_us, expected} =
        :timer.tc(fn ->
          out = predict.(params, input)
          sync(out)
          out
        end)

      base = %{
        architecture: arch,
        hidden: hidden,
        window: window,
        embed: embed,
        full: measure(fn -> predict.(params, input) end, n),
        first_prediction_us: compile_us
      }

      if Edifice.stateful?(arch) do
        module = Edifice.module_for(arch)
        step = Edifice.Stateful.jit_step(module, EXLA, client: client, precision: :highest)

        state0 =
          Edifice.init_state(arch, params, opts)
          |> Nx.backend_copy({EXLA.Backend, client: client})

        frames = for i <- 0..(window - 1), do: Nx.take(x, i, axis: 1)

        {actual, state} =
          Enum.reduce(frames, {nil, state0}, fn f, {_, s} -> step.(params, s, f) end)

        diff = Nx.subtract(actual, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        # Fail closed: never advertise a fast step that does not match the full prefix.
        Nx.Testing.assert_all_close(actual, expected, atol: 1.0e-4, rtol: 1.0e-4)
        {replay_a, _} = step.(params, state, frame)
        restored = Edifice.Stateful.snapshot(state)
        {replay_b, _} = step.(params, restored, frame)
        true = Nx.to_binary(replay_a) == Nx.to_binary(replay_b)
        compiled = measure(fn -> elem(step.(params, state, frame), 0) end, n)
        eager = measure(fn -> elem(module.step(params, state, frame), 0) end, min(n, 20))

        ahead_of_time =
          if aot? do
            executable =
              Nx.Defn.compile(&module.step/3, [params, state, frame],
                compiler: EXLA,
                client: client,
                precision: :highest
              )

            {output, _} = executable.(params, state, frame)
            Nx.Testing.assert_all_close(output, replay_a, atol: 1.0e-4, rtol: 1.0e-4)
            measure(fn -> elem(executable.(params, state, frame), 0) end, n)
          end

        Map.merge(base, %{
          step: compiled,
          eager_step: eager,
          ahead_of_time_step: ahead_of_time,
          max_abs_diff: diff,
          rollback_exact: true,
          full_to_step_speedup: base.full.median_us / max(compiled.median_us, 1)
        })
      else
        Map.put(base, :step, nil)
      end
    rescue
      e ->
        %{
          architecture: arch,
          hidden: hidden,
          window: window,
          error: Exception.format(:error, e, __STACKTRACE__)
        }
    end
  end
end

# Width 516 is raw 264 with two action and two character IDs expanded to 64 each.
# Enhanced Nana/style/previous-action configurations require their own width override.
embed = Keyword.get(cli, :embed, 516)
n = Keyword.get(cli, :iterations, 200)
true = n > 0 and embed > 0

configs = [
  {:gru, 512, 60, []},
  {:mamba, 512, 80, []},
  {:min_gru, 256, 80, []},
  {:gated_delta_net, 512, 60, []},
  {:mlp, 64, 60, []},
  {:mamba_ssd, 512, 60, []},
  {:lstm, 512, 60, []},
  {:gated_ssm, 512, 60, [scan_mode: :causal]}
]

selected = Keyword.get(cli, :arches)

configs =
  if selected,
    do:
      Enum.filter(configs, fn {a, _, _, _} -> Atom.to_string(a) in String.split(selected, ",") end),
    else: configs

rows =
  Enum.map(configs, fn cfg ->
    row = RecurrentInferenceBench.run(cfg, embed, client, n, Keyword.get(cli, :aot, false))
    IO.puts(Jason.encode!(row))
    row
  end)

result = %{
  client: client,
  timestamp: DateTime.to_iso8601(DateTime.utc_now()),
  batch: 1,
  dtype: "f32",
  precision: "highest",
  weights: "seeded synthetic, device resident",
  native_requested: native?,
  native_library_loaded: :persistent_term.get(Edifice.CUDA.Library, nil) == :loaded,
  scope:
    "backbone only; full prefix versus carried state, not equivalent sliding-window deployment",
  system: %{
    schedulers: System.schedulers_online(),
    elixir: System.version(),
    otp: System.otp_release()
  },
  script_sha256: :crypto.hash(:sha256, File.read!(__ENV__.file)) |> Base.encode16(case: :lower),
  rows: rows
}

if cli[:output], do: File.write!(cli[:output], Jason.encode!(result, pretty: true))

if Enum.any?(rows, &Map.has_key?(&1, :error)), do: System.halt(1)
