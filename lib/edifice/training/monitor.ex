defmodule Edifice.Training.Monitor do
  @moduledoc """
  Runtime monitoring utilities for inspecting tensor values during training.

  Uses `Nx.runtime_call/4` to observe tensor values from within `defn`
  computations without breaking the computation graph. All observations
  are passthrough — they return the input tensor unchanged.

  ## Usage in defn

      import Nx.Defn

      defn training_step(params, input, target) do
        output = predict_fn.(params, input)
        loss = loss_fn.(output, target)

        # Log loss value during execution
        loss = Edifice.Training.Monitor.observe(loss, label: "loss")

        # Check for numerical issues
        output = Edifice.Training.Monitor.assert_finite(output, label: "activations")

        loss
      end

  ## Usage with Recipes

      loop =
        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> Edifice.Training.Monitor.attach()

  ## Callbacks

  All callbacks follow the `Nx.runtime_call/4` contract: named captures
  with arity 2 receiving `(tensor, opts)` and returning a tensor matching
  the output template.
  """

  require Logger
  import Nx.Defn

  # ============================================================================
  # Defn-compatible observation functions
  # ============================================================================

  @doc """
  Observe a scalar or tensor value during graph execution.

  Logs the value to the console and returns the tensor unchanged.
  Works inside `defn` — creates a `runtime_call` node in the graph.

  ## Options

    * `:label` - Label for the logged value (default: `"value"`)
    * `:every` - Only log every N calls (default: 1). Uses process
      dictionary counter keyed by label.

  ## Examples

      defn my_step(x) do
        loss = compute_loss(x)
        Edifice.Training.Monitor.observe(loss, label: "loss")
      end
  """
  defn observe(tensor, opts \\ []) do
    Nx.runtime_call(tensor, tensor, opts, &__MODULE__.observe_callback/2)
  end

  @doc """
  Log summary statistics (mean, std, min, max) of a tensor.

  Useful for monitoring activation distributions or gradient magnitudes.
  Returns the tensor unchanged.

  ## Options

    * `:label` - Label for the logged stats (default: `"tensor"`)
    * `:every` - Only log every N calls (default: 1)
  """
  defn observe_stats(tensor, opts \\ []) do
    Nx.runtime_call(tensor, tensor, opts, &__MODULE__.observe_stats_callback/2)
  end

  @doc """
  Assert that a tensor contains no NaN or Inf values.

  Logs a warning if non-finite values are detected. Returns the
  tensor unchanged (does not raise — training continues).

  ## Options

    * `:label` - Label identifying the tensor (default: `"tensor"`)
    * `:halt` - If `true`, raises on NaN/Inf instead of warning (default: `false`)
  """
  defn assert_finite(tensor, opts \\ []) do
    Nx.runtime_call(tensor, tensor, opts, &__MODULE__.assert_finite_callback/2)
  end

  @doc """
  Compute and observe the global norm of a gradient map.

  Takes a flat map of gradient tensors (as from `value_and_grad`),
  computes the global L2 norm, and logs it. Returns the gradient map
  unchanged.

  This is meant to be called outside `defn` on materialized gradients,
  since gradient maps are not valid `defn` arguments.

  ## Options

    * `:label` - Label for the logged norm (default: `"grad_norm"`)
    * `:every` - Only log every N calls (default: 1)
  """
  def observe_grad_norm(grads, opts \\ []) when is_map(grads) do
    label = Keyword.get(opts, :label, "grad_norm")
    every = Keyword.get(opts, :every, 1)

    if should_log?(label, every) do
      norm = grad_global_norm(grads)
      log_value(label, norm)
    end

    grads
  end

  # ============================================================================
  # Axon.Loop integration
  # ============================================================================

  @doc """
  Attach training monitors to an Axon.Loop.

  Adds event handlers that log training metrics using `runtime_call`
  where possible and standard loop events otherwise.

  ## Options

    * `:metrics` - List of metrics to monitor (default: `[:loss, :grad_norm]`)
      * `:loss` - Log loss value each step
      * `:grad_norm` - Log gradient global norm each step
      * `:param_norm` - Log parameter global norm each step
      * `:learning_rate` - Log current learning rate each step
    * `:every` - Log every N steps (default: 1)
    * `:nan_check` - Check for NaN/Inf in loss (default: `false`)
    * `:halt_on_nan` - Stop training on NaN loss (default: `false`)
  """
  def attach(loop, opts \\ []) do
    metrics = Keyword.get(opts, :metrics, [:loss, :grad_norm])
    every = Keyword.get(opts, :every, 1)
    nan_check = Keyword.get(opts, :nan_check, false)
    halt_on_nan = Keyword.get(opts, :halt_on_nan, false)

    loop
    |> maybe_attach_step_monitor(metrics, every)
    |> maybe_attach_nan_check(nan_check, halt_on_nan)
  end

  # ============================================================================
  # Callbacks (named captures for runtime_call — must be public, arity 2)
  # ============================================================================

  @doc false
  def observe_callback(tensor, opts) do
    label = Keyword.get(opts, :label, "value")
    every = Keyword.get(opts, :every, 1)

    if should_log?(label, every) do
      value = scalar_value(tensor)
      log_value(label, value)
    end

    tensor
  end

  @doc false
  def observe_stats_callback(tensor, opts) do
    label = Keyword.get(opts, :label, "tensor")
    every = Keyword.get(opts, :every, 1)

    if should_log?(label, every) do
      mean = tensor |> Nx.mean() |> Nx.to_number()
      std = tensor |> Nx.standard_deviation() |> Nx.to_number()
      min = tensor |> Nx.reduce_min() |> Nx.to_number()
      max = tensor |> Nx.reduce_max() |> Nx.to_number()

      Logger.info(
        "[Monitor] #{label}: mean=#{format_float(mean)} std=#{format_float(std)} " <>
          "min=#{format_float(min)} max=#{format_float(max)}"
      )
    end

    tensor
  end

  @doc false
  def assert_finite_callback(tensor, opts) do
    label = Keyword.get(opts, :label, "tensor")
    halt = Keyword.get(opts, :halt, false)

    has_nan = tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    has_inf = tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 1

    if has_nan or has_inf do
      msg = "[Monitor] #{label}: NaN=#{has_nan} Inf=#{has_inf}"

      if halt do
        raise msg
      else
        Logger.warning(msg)
      end
    end

    tensor
  end

  # ============================================================================
  # Private helpers
  # ============================================================================

  defp should_log?(label, 1), do: bump_counter(label) || true

  defp should_log?(label, every) do
    count = bump_counter(label)
    rem(count, every) == 0
  end

  defp bump_counter(label) do
    key = {__MODULE__, :counter, label}
    count = Process.get(key, 0) + 1
    Process.put(key, count)
    count
  end

  defp scalar_value(tensor) do
    if Nx.size(tensor) == 1 do
      Nx.to_number(tensor)
    else
      Nx.mean(tensor) |> Nx.to_number()
    end
  end

  defp grad_global_norm(grads) do
    grads
    |> flatten_params()
    |> Enum.map(fn tensor ->
      tensor
      |> Nx.as_type(:f32)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.to_number()
    end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, v} when is_map(v) -> flatten_params(v)
      _ -> []
    end)
  end

  defp log_value(label, value) when is_float(value) do
    Logger.info("[Monitor] #{label}: #{format_float(value)}")
  end

  defp log_value(label, value) do
    Logger.info("[Monitor] #{label}: #{inspect(value)}")
  end

  defp format_float(value) when is_float(value) do
    :erlang.float_to_binary(value, decimals: 6)
  end

  defp format_float(value), do: inspect(value)

  defp maybe_attach_step_monitor(loop, metrics, every) do
    Axon.Loop.handle_event(loop, :iteration_completed, fn state ->
      step = state.iteration

      if rem(step, every) == 0 do
        parts =
          Enum.flat_map(metrics, fn
            :loss ->
              loss = get_in(state.step_state, [:loss])
              if loss, do: ["loss=#{format_float(Nx.to_number(loss))}"], else: []

            :grad_norm ->
              grads = get_in(state.step_state, [:gradients])

              if grads do
                norm = grad_global_norm(grads)
                ["grad_norm=#{format_float(norm)}"]
              else
                []
              end

            :param_norm ->
              params = get_in(state.step_state, [:model_state])

              if params do
                data =
                  case params do
                    %Axon.ModelState{data: d} -> d
                    %{} -> params
                  end

                norm = grad_global_norm(data)
                ["param_norm=#{format_float(norm)}"]
              else
                []
              end

            _ ->
              []
          end)

        if parts != [] do
          Logger.info("[Monitor] step=#{step} #{Enum.join(parts, " ")}")
        end
      end

      {:continue, state}
    end)
  end

  defp maybe_attach_nan_check(loop, false, _halt), do: loop

  defp maybe_attach_nan_check(loop, true, halt_on_nan) do
    Axon.Loop.handle_event(loop, :iteration_completed, fn state ->
      loss = get_in(state.step_state, [:loss])

      if loss do
        loss_val = Nx.to_number(loss)

        cond do
          loss_val != loss_val ->
            # NaN: NaN != NaN is true
            msg = "[Monitor] NaN loss detected at step #{state.iteration}"

            if halt_on_nan do
              Logger.error(msg <> " — halting training")
              {:halt_loop, state}
            else
              Logger.warning(msg)
              {:continue, state}
            end

          loss_val == :infinity or loss_val == :neg_infinity ->
            msg = "[Monitor] Inf loss detected at step #{state.iteration}"

            if halt_on_nan do
              Logger.error(msg <> " — halting training")
              {:halt_loop, state}
            else
              Logger.warning(msg)
              {:continue, state}
            end

          true ->
            {:continue, state}
        end
      else
        {:continue, state}
      end
    end)
  end
end
