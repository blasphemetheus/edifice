defmodule Edifice.Serving.InferenceServer do
  @moduledoc """
  Batched inference server for Edifice models.

  A GenServer that accepts inference requests, accumulates them into
  batches, and dispatches them efficiently. Supports configurable
  batch sizes, timeout-based dispatch, and architecture-aware padding.

  ## Features

  - **Request batching** — Accumulates requests up to `max_batch_size`
  - **Timeout dispatch** — Dispatches partial batches after `batch_timeout_ms`
  - **Padding** — Pads inputs to uniform sequence length within a batch
  - **Async responses** — Callers receive results via `from` reply
  - **Metrics** — Tracks requests served, batches dispatched, total latency

  ## Usage

      # Start server with a compiled model
      model = Edifice.build(:min_gru, embed_dim: 256, hidden_size: 256, seq_len: 64)
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
      params = init_fn.(template, Axon.ModelState.empty())

      {:ok, pid} = InferenceServer.start_link(
        predict_fn: predict_fn,
        params: params,
        max_batch_size: 8,
        batch_timeout_ms: 50
      )

      # Send requests (blocks until batch is processed)
      result = InferenceServer.predict(pid, input_tensor)

      # Get metrics
      InferenceServer.metrics(pid)

  ## Architecture

  ```
  Client 1 ──► predict(input) ─┐
  Client 2 ──► predict(input) ─┤─► batch accumulator ─► dispatch ─► predict_fn
  Client 3 ──► predict(input) ─┘      (timer)            (pad)       (reply)
  ```
  """

  use GenServer

  @default_max_batch 8
  @default_timeout_ms 50

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start the inference server.

  ## Options

    - `:predict_fn` - Compiled prediction function (required)
    - `:params` - Model parameters (required)
    - `:max_batch_size` - Maximum requests per batch (default: 8)
    - `:batch_timeout_ms` - Dispatch partial batches after this many ms (default: 50)
    - `:name` - Optional GenServer name for registration
  """
  def start_link(opts) do
    name = Keyword.get(opts, :name)
    gen_opts = if name, do: [name: name], else: []
    GenServer.start_link(__MODULE__, opts, gen_opts)
  end

  @doc """
  Submit an inference request and wait for the result.

  `input` should be a tensor or map of tensors matching the model's input spec.
  For sequence models, typically `%{"state_sequence" => tensor}`.

  Returns the model output tensor.
  """
  def predict(server, input, timeout \\ 30_000) do
    GenServer.call(server, {:predict, input}, timeout)
  end

  @doc """
  Get server metrics.

  Returns `%{requests: n, batches: n, mean_batch_size: float, mean_latency_ms: float}`.
  """
  def metrics(server) do
    GenServer.call(server, :metrics)
  end

  @doc """
  Stop the server gracefully.
  """
  def stop(server) do
    GenServer.stop(server)
  end

  # ============================================================================
  # GenServer Implementation
  # ============================================================================

  @impl true
  def init(opts) do
    predict_fn = Keyword.fetch!(opts, :predict_fn)
    params = Keyword.fetch!(opts, :params)
    max_batch_size = Keyword.get(opts, :max_batch_size, @default_max_batch)
    batch_timeout_ms = Keyword.get(opts, :batch_timeout_ms, @default_timeout_ms)

    state = %{
      predict_fn: predict_fn,
      params: params,
      max_batch_size: max_batch_size,
      batch_timeout_ms: batch_timeout_ms,
      queue: [],
      timer_ref: nil,
      # Metrics
      total_requests: 0,
      total_batches: 0,
      total_latency_us: 0
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:predict, input}, from, state) do
    queue = [{from, input} | state.queue]
    state = %{state | queue: queue}

    cond do
      length(queue) >= state.max_batch_size ->
        # Batch is full — dispatch immediately
        state = cancel_timer(state)
        state = dispatch_batch(state)
        {:noreply, state}

      state.timer_ref == nil ->
        # Start timeout timer for partial batch
        ref = Process.send_after(self(), :batch_timeout, state.batch_timeout_ms)
        {:noreply, %{state | timer_ref: ref}}

      true ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_call(:metrics, _from, state) do
    metrics = %{
      requests: state.total_requests,
      batches: state.total_batches,
      mean_batch_size:
        if(state.total_batches > 0,
          do: Float.round(state.total_requests / state.total_batches, 2),
          else: 0.0
        ),
      mean_latency_ms:
        if(state.total_batches > 0,
          do: Float.round(state.total_latency_us / state.total_batches / 1_000, 3),
          else: 0.0
        ),
      pending: length(state.queue)
    }

    {:reply, metrics, state}
  end

  @impl true
  def handle_info(:batch_timeout, state) do
    state = %{state | timer_ref: nil}

    if state.queue != [] do
      state = dispatch_batch(state)
      {:noreply, state}
    else
      {:noreply, state}
    end
  end

  # ============================================================================
  # Batch Dispatch
  # ============================================================================

  defp dispatch_batch(%{queue: []} = state), do: state

  defp dispatch_batch(state) do
    %{queue: queue, predict_fn: predict_fn, params: params} = state

    # Take up to max_batch_size requests
    {batch_items, remaining} = Enum.split(queue, state.max_batch_size)
    batch_items = Enum.reverse(batch_items)
    batch_size = length(batch_items)

    # Build batched input
    {froms, inputs} = Enum.unzip(batch_items)
    batched_input = batch_inputs(inputs)

    # Run inference
    {latency_us, output} = :timer.tc(fn -> predict_fn.(params, batched_input) end)

    # Unbatch and reply to each caller
    unbatched = unbatch_output(output, batch_size)
    Enum.zip(froms, unbatched) |> Enum.each(fn {from, result} ->
      GenServer.reply(from, result)
    end)

    %{
      state
      | queue: remaining,
        total_requests: state.total_requests + batch_size,
        total_batches: state.total_batches + 1,
        total_latency_us: state.total_latency_us + latency_us
    }
  end

  # Batch inputs: stack tensors along batch dimension, pad sequences to max length
  defp batch_inputs(inputs) when is_list(inputs) do
    case hd(inputs) do
      %{} = map ->
        # Map of named inputs — batch each key
        keys = Map.keys(map)

        for key <- keys, into: %{} do
          tensors = Enum.map(inputs, &Map.fetch!(&1, key))
          {key, stack_and_pad(tensors)}
        end

      tensor when is_struct(tensor, Nx.Tensor) ->
        stack_and_pad(inputs)
    end
  end

  # Stack tensors, padding shorter sequences to the max length
  defp stack_and_pad(tensors) do
    shapes = Enum.map(tensors, &Nx.shape/1)

    if length(Enum.uniq(shapes)) == 1 do
      # All same shape — simple stack
      Nx.stack(tensors)
    else
      # Different sequence lengths — pad to max
      max_len = shapes |> Enum.map(&elem(&1, 0)) |> Enum.max()
      _rank = tuple_size(hd(shapes))

      padded =
        Enum.map(tensors, fn t ->
          current_len = elem(Nx.shape(t), 0)

          if current_len < max_len do
            pad_shape =
              Nx.shape(t)
              |> Tuple.to_list()
              |> List.replace_at(0, max_len - current_len)
              |> List.to_tuple()

            pad = Nx.broadcast(0.0, pad_shape) |> Nx.as_type(Nx.type(t))
            Nx.concatenate([t, pad], axis: 0)
          else
            t
          end
        end)

      Nx.stack(padded)
    end
  end

  # Unbatch output: split batch dimension into individual results
  defp unbatch_output(output, batch_size) when is_struct(output, Nx.Tensor) do
    for i <- 0..(batch_size - 1) do
      Nx.slice_along_axis(output, i, 1, axis: 0)
    end
  end

  defp unbatch_output(output, batch_size) when is_map(output) do
    # Container output — unbatch each key
    keys = Map.keys(output)

    for i <- 0..(batch_size - 1) do
      for key <- keys, into: %{} do
        tensor = Map.fetch!(output, key)
        {key, Nx.slice_along_axis(tensor, i, 1, axis: 0)}
      end
    end
  end

  defp cancel_timer(%{timer_ref: nil} = state), do: state

  defp cancel_timer(%{timer_ref: ref} = state) do
    Process.cancel_timer(ref)
    %{state | timer_ref: nil}
  end
end
