# AI-generated. Usage and authorship marker: docs/mamba2.md.
# Run in a fresh VM per case; compiled memory is not GPU allocator reservation.
Application.ensure_all_started(:axon)
Application.ensure_all_started(:exla)

Application.put_env(:exla, :clients,
  cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.8]
)

Application.put_env(:exla, :default_client, :cuda)

defmodule Mamba2TrainingBench do
  def sync(tree) do
    Nx.Defn.Composite.traverse(tree, fn t ->
      if match?(%EXLA.Backend{}, t.data) do
        EXLA.DeviceBuffer.read(t.data.buffer, div(elem(Nx.type(t), 1), 8))
      end

      t
    end)

    :ok
  end

  def measure(fun, args) do
    IO.puts("Compiling...")

    {compile_us, executable} =
      :timer.tc(fn -> EXLA.to_executable(fun, args, client: :cuda, precision: :highest) end)

    memory = EXLA.Executable.memory_stats(executable)
    IO.inspect(memory, label: "Compiled memory (bytes)")
    hlo = EXLA.Executable.optimized_hlo(executable)
    fusion_count = length(Regex.scan(~r/\bfusion\(/, hlo))
    run = EXLA.compile(fun, args, client: :cuda, precision: :highest)
    for _ <- 1..3, do: sync(apply(run, args))
    IO.puts("Timing warm execution...")

    times =
      for _ <- 1..10 do
        {us, _} = :timer.tc(fn -> sync(apply(run, args)) end)
        us / 1000
      end

    %{
      compile_ms: compile_us / 1000,
      median_ms: Enum.at(Enum.sort(times), 5),
      max_ms: Enum.max(times),
      compiled_memory: memory,
      xla_fusion_calls: fusion_count
    }
  end

  def run do
    path = System.fetch_env!("BENCH_CONFIG")
    config = path |> File.read!() |> Jason.decode!()
    hidden = config["hidden_size"] || hd(config["hidden_sizes"])
    batch = String.to_integer(System.get_env("BENCH_BATCH", to_string(config["batch_size"])))
    state = String.to_integer(System.get_env("BENCH_STATE", "128"))
    execution = System.get_env("BENCH_EXECUTION", "sequential") |> String.to_existing_atom()

    opts = [
      embed_dim: config["embed_size"],
      hidden_size: hidden,
      num_layers: config["num_layers"],
      state_size: state,
      head_dim: 64,
      execution: execution,
      chunk_size: 32
    ]

    IO.inspect(%{batch: batch, time: config["window_size"], opts: opts}, label: "Case")
    model = Edifice.SSM.Mamba2.build(opts)
    {init, predict} = Axon.build(model, seed: 71)

    {x, _} =
      Nx.Random.normal(Nx.Random.key(71),
        shape: {batch, config["window_size"], config["embed_size"]}
      )

    params = init.(Nx.broadcast(0.0, {1, 1, config["embed_size"]}), Axon.ModelState.empty())
    params = Nx.backend_transfer(params, {EXLA.Backend, client: :cuda})
    x = Nx.backend_transfer(x, {EXLA.Backend, client: :cuda})
    forward = fn p, x -> predict.(p, x) end

    backward = fn p, x ->
      Nx.Defn.value_and_grad(p, fn p ->
        y = predict.(p, x)
        Nx.mean(Nx.multiply(y, Nx.sin(Nx.iota(Nx.shape(y)))))
      end)
    end

    results = %{
      source: path,
      parameter_seed: 71,
      batch: batch,
      sequence: config["window_size"],
      opts: Map.new(opts),
      forward: measure(forward, [params, x]),
      forward_backward: measure(backward, [params, x]),
      device:
        elem(
          System.cmd("nvidia-smi", [
            "--query-gpu=name,utilization.gpu,memory.used",
            "--format=csv,noheader"
          ]),
          0
        ),
      scope: "backbone only; synthetic frames; weighted output loss; no optimizer or policy heads"
    }

    results =
      if System.get_env("BENCH_COMPONENTS") == "1" do
        p = params.data["mamba2_1"]
        inner = hidden * 2
        heads = div(inner, 64)
        length = config["window_size"]
        # Fixed-valued runtime inputs. Time the scan separately from
        # projection/convolution/normalization; this is not an additive profile.
        make = fn shape ->
          Nx.broadcast(0.01, shape) |> Nx.backend_transfer({EXLA.Backend, client: :cuda})
        end

        args = [
          make.({batch, length, heads, 64}),
          make.({batch, length, heads, state}),
          make.({batch, length, heads, state}),
          make.({batch, length, heads}),
          Nx.negate(Nx.exp(p["A_log"])),
          p["D"],
          make.({batch, heads, 64, state})
        ]

        scan = fn x, b, c, dt, a, d, h -> Edifice.SSM.Mamba2SSD.scan(x, b, c, dt, a, d, h, 32) end

        scan_grad = fn x, b, c, dt, a, d, h ->
          Nx.Defn.grad({x, b, c, dt, a, d, h}, fn {x, b, c, dt, a, d, h} ->
            {y, _} = scan.(x, b, c, dt, a, d, h)
            Nx.sum(y)
          end)
        end

        Map.put(results, :scan, %{
          forward: measure(scan, args),
          forward_backward: measure(scan_grad, args)
        })
      else
        results
      end

    File.write!(System.fetch_env!("BENCH_OUT"), Jason.encode!(results, pretty: true))
    IO.inspect(results, label: "Result")
  end
end

Mamba2TrainingBench.run()
