# AI-generated. Usage and authorship marker: docs/mamba2.md.
# Run in a fresh VM per case; compiled memory is not GPU allocator reservation.
Application.ensure_all_started(:crypto)
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

  def measure(fun, args, iterations \\ 10) do
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
      for _ <- 1..iterations do
        {us, _} = :timer.tc(fn -> sync(apply(run, args)) end)
        us / 1000
      end

    %{
      compile_ms: compile_us / 1000,
      median_ms: Enum.at(Enum.sort(times), div(iterations, 2)),
      iterations: iterations,
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
    architecture = System.get_env("BENCH_ARCH", "mamba2_full")

    module =
      case architecture do
        "mamba" -> Edifice.SSM.Mamba
        "mamba2_full" -> Edifice.SSM.Mamba2
      end

    state =
      String.to_integer(
        System.get_env("BENCH_STATE", if(architecture == "mamba", do: "16", else: "128"))
      )

    execution = System.get_env("BENCH_EXECUTION", "sequential") |> String.to_existing_atom()

    opts = [
      embed_dim: config["embed_size"],
      hidden_size: hidden,
      num_layers: config["num_layers"],
      state_size: state,
      seq_len: nil,
      window_size: config["window_size"],
      head_dim: 64,
      execution: execution,
      chunk_size: 32
    ]

    IO.inspect(%{batch: batch, time: config["window_size"], opts: opts}, label: "Case")
    model = module.build(opts)
    {init, predict} = Axon.build(model, seed: 71)

    {x, _} =
      Nx.Random.normal(Nx.Random.key(71),
        shape: {batch, config["window_size"], config["embed_size"]}
      )

    params = init.(Nx.broadcast(0.0, {1, 1, config["embed_size"]}), Axon.ModelState.empty())
    params = Nx.backend_transfer(params, {EXLA.Backend, client: :cuda})
    x = Nx.backend_transfer(x, {EXLA.Backend, client: :cuda})
    Edifice.CUDA.AutoTuneProfiler.start()
    forward = fn p, x -> predict.(p, x) end

    backward = fn p, x ->
      Nx.Defn.value_and_grad(p, fn p ->
        y = predict.(p, x)
        Nx.mean(Nx.multiply(y, Nx.sin(Nx.iota(Nx.shape(y)))))
      end)
    end

    results = %{
      source: path,
      architecture: architecture,
      fused_disabled: System.get_env("EDIFICE_DISABLE_FUSED"),
      autotune: System.get_env("EDIFICE_AUTOTUNE"),
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

    dispatches = :ets.tab2list(:__edifice_autotune_profiler__)
    IO.inspect(dispatches, label: "Dispatches")

    if architecture == "mamba" and System.get_env("BENCH_EXPECT_FUSED") == "1" do
      unless Enum.any?(dispatches, fn {{kernel, path, _, _}, n} ->
               kernel == :selective_scan and path == :custom_call and n > 0
             end),
             do: raise("Mamba never used the requested fused scan")
    end

    results = Map.put(results, :dispatches, inspect(dispatches))

    frame = x |> Nx.slice([0, 0, 0], [1, 1, config["embed_size"]]) |> Nx.squeeze(axes: [1])

    carry =
      module.init_state(params, Keyword.put(opts, :batch_size, 1))
      |> Nx.backend_transfer({EXLA.Backend, client: :cuda})

    step = fn p, s, x -> module.step(p, s, x) end
    results = Map.put(results, :single_bot_step, measure(step, [params, carry, frame], 100))

    results =
      if architecture == "mamba" and System.get_env("BENCH_EXPECT_FUSED") == "1" do
        fused = EXLA.compile(forward, [params, x], client: :cuda, precision: :highest).(params, x)
        System.put_env("EDIFICE_DISABLE_FUSED", "1")
        {_, portable_predict} = Axon.build(module.build(opts), seed: 71)

        portable =
          EXLA.compile(fn p, x -> portable_predict.(p, x) end, [params, x],
            client: :cuda,
            precision: :highest
          ).(params, x)

        error = Nx.subtract(fused, portable) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        unless is_number(error) and error < 1.0e-3,
          do: raise("Fused/portable mismatch: #{inspect(error)}")

        Map.put(results, :portable_forward_max_abs_error, error)
      else
        results
      end

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
