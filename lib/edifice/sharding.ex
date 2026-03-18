defmodule Edifice.Sharding do
  @moduledoc """
  Multi-GPU sharding for Edifice models via `Nx.Mesh` and `EXLA.shard_jit`.

  Provides data-parallel inference and training support that automatically
  splits batches across available GPU devices and gathers results.

  ## Quick Start

      model = Edifice.build(:decoder_only, embed_dim: 256, ...)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template, Axon.ModelState.empty())

      # Data-parallel inference across 4 GPUs
      dp_predict = Edifice.Sharding.data_parallel(predict_fn, params, num_devices: 4)
      output = dp_predict.(input)  # batch split across 4 GPUs, output gathered

  ## How It Works

  Data parallelism splits the batch dimension across devices. Each device
  runs the full model on its batch slice. Shardy (XLA's sharding engine)
  handles any necessary communication (all-reduce for training, broadcast
  for replicated parameters).

  - **Inference**: Params captured in closure (replicated), input batch sharded
  - **Training**: Use `Axon.Loop.trainer` with EXLA compiler + SPMD (future)

  ## Device Discovery

      Edifice.Sharding.device_count()       # auto-detect
      Edifice.Sharding.device_count(:cuda)   # specific client
      Edifice.Sharding.mesh(num_devices: 4)  # create a mesh

  ## Composition

  Composes with FP8 quantization and mixed precision:

      q_predict = Edifice.Quantization.FP8.wrap_inference(predict_fn)
      dp_predict = Edifice.Sharding.data_parallel(q_predict, q_params, num_devices: 4)

  ## Requirements

  Requires EXLA. Multi-GPU requires multiple CUDA devices visible to EXLA.
  For testing, the `:host` client provides virtual multi-device support
  via `XLA_FLAGS=--xla_force_host_platform_device_count=N`.
  """

  require Logger

  @doc """
  Create a data-parallel inference function.

  Returns a function `(input) -> output` that splits the input batch
  across devices, runs `predict_fn` on each shard, and concatenates
  the results.

  ## Parameters

    * `predict_fn` - Compiled Axon prediction function `(params, input) -> output`
    * `params` - Model parameters (captured in closure, replicated on each device)

  ## Options

    * `:num_devices` - Number of devices to use (default: `:auto`)
    * `:client` - EXLA client name (default: auto-detect)
    * `:batch_dim` - Batch dimension index (default: 0)

  ## Examples

      dp_predict = Edifice.Sharding.data_parallel(predict_fn, params, num_devices: 2)
      output = dp_predict.(input_batch)
  """
  @spec data_parallel(function(), map(), keyword()) :: function()
  def data_parallel(predict_fn, params, opts \\ []) do
    ensure_exla!()
    num_devices = resolve_num_devices(opts)
    client = Keyword.get(opts, :client, :host)
    batch_dim = Keyword.get(opts, :batch_dim, 0)

    mesh = %Nx.Mesh{name: "dp", shape: {num_devices}}

    fn input ->
      case input do
        %Nx.Tensor{} ->
          run_data_parallel_tensor(predict_fn, params, input, mesh, batch_dim, num_devices, client)

        %{} = map_input ->
          run_data_parallel_map(predict_fn, params, map_input, mesh, batch_dim, num_devices, client)
      end
    end
  end

  @doc """
  Create a mesh from device count.

  ## Options

    * `:num_devices` - Number of devices (default: `:auto`)
    * `:name` - Mesh name (default: `"mesh"`)
    * `:shape` - Explicit shape tuple for multi-axis meshes (e.g., `{2, 2}`)
  """
  @spec mesh(keyword()) :: Nx.Mesh.t()
  def mesh(opts \\ []) do
    name = Keyword.get(opts, :name, "mesh")

    shape =
      case Keyword.get(opts, :shape) do
        nil ->
          n = resolve_num_devices(opts)
          {n}

        shape when is_tuple(shape) ->
          shape
      end

    %Nx.Mesh{name: name, shape: shape}
  end

  @doc """
  Query the number of available devices for a client.

  ## Examples

      Edifice.Sharding.device_count()        # auto-detect
      Edifice.Sharding.device_count(:cuda)    # CUDA devices
      Edifice.Sharding.device_count(:host)    # host virtual devices
  """
  @spec device_count(atom() | nil) :: pos_integer()
  def device_count(client \\ nil) do
    ensure_exla!()
    client_name = client || EXLA.Client.default_name()
    EXLA.Client.fetch!(client_name).device_count
  end

  @doc """
  Report sharding plan for a model.

  Estimates memory distribution under data parallelism.

  ## Parameters

    * `params` - Model parameters

  ## Options

    * `:num_devices` - Number of devices (default: `:auto`)
  """
  @spec report(map(), keyword()) :: map()
  def report(params, opts \\ []) do
    num_devices = resolve_num_devices(opts)
    data = extract_params(params)

    {total_bytes, tensor_count} =
      flatten_tensors(data)
      |> Enum.reduce({0, 0}, fn tensor, {bytes, count} ->
        {_, bits} = Nx.type(tensor)
        elem_bytes = div(bits, 8)
        {bytes + Nx.size(tensor) * elem_bytes, count + 1}
      end)

    result = %{
      total_bytes: total_bytes,
      tensor_count: tensor_count,
      num_devices: num_devices,
      # Data parallel: params replicated, so per-device = total
      params_per_device_bytes: total_bytes,
      # Activation memory is split across devices (proportional to batch fraction)
      activation_savings: "~#{num_devices}x (batch split)"
    }

    IO.puts("[Sharding] Data Parallel Plan (#{num_devices} devices)")
    IO.puts("  Parameters: #{readable_size(total_bytes)} (#{tensor_count} tensors)")
    IO.puts("  Per-device params: #{readable_size(total_bytes)} (replicated)")
    IO.puts("  Activation memory: ~#{num_devices}x reduction (batch split)")
    IO.puts("  Throughput: ~#{num_devices}x (linear scaling)")

    result
  end

  # ============================================================================
  # Data parallel: single tensor input
  # ============================================================================

  defp run_data_parallel_tensor(predict_fn, params, input, mesh, batch_dim, num_devices, client) do
    batch_size = Nx.axis_size(input, batch_dim)
    validate_batch_divisible!(batch_size, num_devices)
    shard_size = div(batch_size, num_devices)

    # Params captured in closure — replicated on each device
    wrapper = fn input_shard ->
      predict_fn.(params, input_shard)
    end

    input_shardings = [%{batch_dim => [0]}]

    sharded_args =
      for i <- 0..(num_devices - 1) do
        [Nx.slice_along_axis(input, i * shard_size, shard_size, axis: batch_dim)]
      end

    results =
      EXLA.shard_jit(wrapper, mesh,
        input_shardings: input_shardings,
        client: client
      ).(sharded_args)

    gather_results(results, batch_dim)
  end

  # ============================================================================
  # Data parallel: map input (e.g., %{"input" => tensor})
  # ============================================================================

  defp run_data_parallel_map(predict_fn, params, map_input, mesh, batch_dim, num_devices, client) do
    keys = Map.keys(map_input) |> Enum.sort()

    wrapper = build_map_wrapper(predict_fn, params, keys)

    # All map values are sharded on batch dim
    input_shardings = Enum.map(keys, fn _key -> %{batch_dim => [0]} end)

    sharded_args =
      for i <- 0..(num_devices - 1) do
        Enum.map(keys, fn key ->
          tensor = Map.fetch!(map_input, key)
          batch_size = Nx.axis_size(tensor, batch_dim)
          shard_size = div(batch_size, num_devices)
          Nx.slice_along_axis(tensor, i * shard_size, shard_size, axis: batch_dim)
        end)
      end

    results =
      EXLA.shard_jit(wrapper, mesh,
        input_shardings: input_shardings,
        client: client
      ).(sharded_args)

    gather_results(results, batch_dim)
  end

  # Build a fixed-arity function from sorted map keys
  defp build_map_wrapper(predict_fn, params, keys) do
    rebuild = fn args_list ->
      map = keys |> Enum.zip(args_list) |> Map.new()
      predict_fn.(params, map)
    end

    case length(keys) do
      1 -> fn a -> rebuild.([a]) end
      2 -> fn a, b -> rebuild.([a, b]) end
      3 -> fn a, b, c -> rebuild.([a, b, c]) end
      4 -> fn a, b, c, d -> rebuild.([a, b, c, d]) end

      n ->
        raise ArgumentError,
              "Edifice.Sharding.data_parallel does not support models with #{n} input keys " <>
                "(max 4). Input keys: #{inspect(keys)}"
    end
  end

  # ============================================================================
  # Output gathering
  # ============================================================================

  defp gather_results(results, batch_dim) when is_list(results) do
    first = hd(results)

    cond do
      is_struct(first, Nx.Tensor) ->
        results
        |> Enum.map(&Nx.backend_transfer(&1, Nx.BinaryBackend))
        |> Nx.concatenate(axis: batch_dim)

      is_tuple(first) ->
        size = tuple_size(first)

        for i <- 0..(size - 1) do
          results
          |> Enum.map(&(elem(&1, i) |> Nx.backend_transfer(Nx.BinaryBackend)))
          |> Nx.concatenate(axis: batch_dim)
        end
        |> List.to_tuple()

      is_map(first) and not is_struct(first) ->
        map_keys = Map.keys(first)

        Map.new(map_keys, fn key ->
          tensors =
            results
            |> Enum.map(&(Map.fetch!(&1, key) |> Nx.backend_transfer(Nx.BinaryBackend)))

          {key, Nx.concatenate(tensors, axis: batch_dim)}
        end)

      true ->
        results
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp ensure_exla! do
    unless Code.ensure_loaded?(EXLA) do
      raise RuntimeError,
            "Edifice.Sharding requires EXLA. Add {:exla, \"~> 0.11\"} to your deps."
    end
  end

  defp resolve_num_devices(opts) do
    case Keyword.get(opts, :num_devices, :auto) do
      :auto ->
        client = Keyword.get(opts, :client)
        device_count(client)

      n when is_integer(n) and n > 0 ->
        n
    end
  end

  defp validate_batch_divisible!(batch_size, num_devices) do
    if rem(batch_size, num_devices) != 0 do
      raise ArgumentError,
            "batch size #{batch_size} is not divisible by num_devices #{num_devices}. " <>
              "Pad your batch or use a compatible batch size."
    end
  end

  defp extract_params(%Axon.ModelState{data: data}), do: data
  defp extract_params(%{} = map), do: map

  defp flatten_tensors(map) when is_map(map) and not is_struct(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, v} when is_map(v) -> flatten_tensors(v)
      _ -> []
    end)
  end

  defp flatten_tensors(_), do: []

  defp readable_size(n) when n < 1_024, do: "#{n} B"
  defp readable_size(n) when n < 1_048_576, do: "#{Float.round(n / 1_024, 2)} KB"
  defp readable_size(n), do: "#{Float.round(n / 1_048_576, 2)} MB"
end
