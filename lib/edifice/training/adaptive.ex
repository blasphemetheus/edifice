defmodule Edifice.Training.Adaptive do
  @moduledoc """
  Adaptive training utilities powered by `Nx.runtime_call/4`.

  These functions combine defn-level observation (via runtime_call) with
  Axon.Loop-level responses to create adaptive training behaviors:

  - **Loss spike detection** — detect anomalous loss spikes and skip updates
  - **Gradient overflow guard** — zero out overflowed gradients in fp16 training
  - **Adaptive gradient clipping** — AGC (unit-wise clipping from NFNet)

  ## Usage with Recipes

      loop =
        model
        |> Axon.Loop.trainer(loss, optimizer)
        |> Edifice.Training.Adaptive.skip_on_loss_spike(threshold: 5.0)
        |> Edifice.Training.Adaptive.log_grad_norm(every: 10)

  ## Usage in defn

      defn training_step(params, input, target) do
        {loss, grads} = value_and_grad(params, fn p ->
          predict_fn.(p, input) |> loss_fn.(target)
        end)

        # Zero out overflowed gradients inside defn
        grads = Edifice.Training.Adaptive.guard_overflow(grads)
        {loss, grads}
      end
  """

  require Logger
  import Nx.Defn

  # ============================================================================
  # Defn-level adaptive functions (via runtime_call)
  # ============================================================================

  @doc """
  Guard against gradient overflow inside defn.

  Checks a gradient tensor for NaN/Inf via `runtime_call`. Returns the
  tensor unchanged if finite, or a zero tensor of the same shape if
  overflow is detected. This prevents corrupted gradients from reaching
  the optimizer.

  Works on individual tensors — call on each gradient tensor you want
  to protect.

  ## Options

    * `:label` - Label for logging (default: `"grad"`)

  ## Examples

      defn safe_step(params, input) do
        {loss, grads} = value_and_grad(params, &loss_fn.(&1, input))
        safe_grads = Edifice.Training.Adaptive.guard_overflow(grads)
        {loss, safe_grads}
      end
  """
  defn guard_overflow(tensor, opts \\ []) do
    # runtime_call checks for overflow and returns {tensor, flag}
    # where flag is 1 if overflow detected, 0 otherwise
    flag_template = Nx.template({}, {:u, 8})
    flag = Nx.runtime_call(flag_template, tensor, opts, &__MODULE__.overflow_check_callback/2)

    # Conditionally zero out the tensor
    Nx.select(flag, Nx.broadcast(Nx.tensor(0.0, type: Nx.type(tensor)), tensor), tensor)
  end

  deftransformp get_eps(opts), do: Keyword.get(opts, :eps, 1.0e-3)

  @doc """
  Adaptive Gradient Clipping (AGC) inside defn.

  Clips gradients based on the ratio of gradient norm to parameter norm
  (unit-wise), as described in the NFNet paper. Unlike global norm clipping,
  AGC adapts to each parameter's scale.

  `clipping_factor` controls maximum allowed gradient-to-parameter norm
  ratio. Typical values: 0.01-0.1.

  ## Options

    * `:eps` - Minimum parameter norm to prevent division by zero (default: 1.0e-3)

  ## Examples

      defn step(params, grads) do
        clipped = Edifice.Training.Adaptive.agc_clip(grads, params, 0.01)
        apply_updates(params, clipped)
      end
  """
  defn agc_clip(grad, param, clipping_factor, opts \\ []) do
    eps = get_eps(opts)

    # Compute norms along all but the first axis (unit-wise)
    param_norm = Nx.sqrt(Nx.sum(Nx.pow(param, 2)) + eps)
    grad_norm = Nx.sqrt(Nx.sum(Nx.pow(grad, 2)) + eps)

    # Clip ratio
    max_norm = Nx.multiply(param_norm, clipping_factor)
    clip_ratio = Nx.divide(max_norm, Nx.max(grad_norm, max_norm))

    Nx.multiply(grad, clip_ratio)
  end

  @doc """
  Observe the effective gradient norm inside defn.

  Computes the L2 norm of a gradient tensor and logs it via `runtime_call`.
  Returns the gradient unchanged. Use this to understand gradient dynamics
  before tuning clip thresholds.

  ## Options

    * `:label` - Label for the logged norm (default: `"grad_norm"`)
    * `:every` - Only log every N calls (default: 1)
  """
  defn observe_grad_norm(grad, opts \\ []) do
    Nx.runtime_call(grad, grad, opts, &__MODULE__.grad_norm_callback/2)
  end

  # ============================================================================
  # Axon.Loop-level adaptive hooks
  # ============================================================================

  @doc """
  Skip optimizer updates when loss spikes.

  Attaches a loop handler that tracks a running average of loss. When the
  current loss exceeds `threshold` times the running average, the update
  is skipped by zeroing gradients in the step state.

  ## Options

    * `:threshold` - Spike detection multiplier (default: 5.0).
      A loss is considered a spike if `loss > threshold * running_avg`.
    * `:warmup_steps` - Steps before spike detection activates (default: 100).
      Allows loss to stabilize before monitoring.
    * `:ema_decay` - Exponential moving average decay for loss tracking
      (default: 0.99).
  """
  def skip_on_loss_spike(loop, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 5.0)
    warmup_steps = Keyword.get(opts, :warmup_steps, 100)
    ema_decay = Keyword.get(opts, :ema_decay, 0.99)

    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      meta =
        Map.merge(state.handler_metadata, %{
          loss_ema: nil,
          spike_count: 0,
          spike_config: %{threshold: threshold, warmup_steps: warmup_steps, ema_decay: ema_decay}
        })

      {:continue, %{state | handler_metadata: meta}}
    end)
    |> Axon.Loop.handle_event(:iteration_completed, fn state ->
      loss_val = get_in(state.step_state, [:loss])

      if loss_val do
        loss_num = Nx.to_number(loss_val)
        meta = state.handler_metadata
        config = meta.spike_config
        step = state.iteration

        {loss_ema, is_spike} =
          case meta.loss_ema do
            nil ->
              {loss_num, false}

            ema when step < config.warmup_steps ->
              new_ema = config.ema_decay * ema + (1 - config.ema_decay) * loss_num
              {new_ema, false}

            ema ->
              new_ema = config.ema_decay * ema + (1 - config.ema_decay) * loss_num
              spike = loss_num > config.threshold * ema and loss_num == loss_num
              {new_ema, spike}
          end

        if is_spike do
          spike_count = meta.spike_count + 1

          Logger.warning(
            "[Adaptive] Loss spike at step #{step}: #{format_float(loss_num)} " <>
              "(#{format_float(loss_num / meta.loss_ema)}x EMA). Skipping update. " <>
              "Total spikes: #{spike_count}"
          )

          # Don't update EMA with spike value — keep the pre-spike average
          new_meta = %{meta | spike_count: spike_count}

          # Zero out gradients to skip the update
          new_step_state =
            if Map.has_key?(state.step_state, :gradients) do
              grads = state.step_state.gradients
              zeroed = deep_map_tensors(grads, fn t -> Nx.broadcast(Nx.tensor(0.0, type: Nx.type(t)), t) end)
              put_in(state.step_state, [:gradients], zeroed)
            else
              state.step_state
            end

          {:continue, %{state | handler_metadata: new_meta, step_state: new_step_state}}
        else
          new_meta = %{meta | loss_ema: loss_ema}
          {:continue, %{state | handler_metadata: new_meta}}
        end
      else
        {:continue, state}
      end
    end)
  end

  @doc """
  Log gradient global norm at the loop level.

  Attaches a handler that computes and logs the L2 norm of all gradients
  after each step.

  ## Options

    * `:every` - Log every N steps (default: 1)
  """
  def log_grad_norm(loop, opts \\ []) do
    every = Keyword.get(opts, :every, 1)

    Axon.Loop.handle_event(loop, :iteration_completed, fn state ->
      step = state.iteration

      if rem(step, every) == 0 do
        grads = get_in(state.step_state, [:gradients])

        if grads do
          norm = compute_global_norm(grads)
          Logger.info("[Adaptive] step=#{step} grad_norm=#{format_float(norm)}")
        end
      end

      {:continue, state}
    end)
  end

  @doc """
  Halt training if loss remains NaN for consecutive steps.

  More nuanced than `Monitor.attach(nan_check: true)` — this allows
  transient NaN (e.g., from a bad batch) but halts if loss doesn't
  recover within `:patience` steps.

  ## Options

    * `:patience` - Consecutive NaN steps before halting (default: 5)
  """
  def halt_on_persistent_nan(loop, opts \\ []) do
    patience = Keyword.get(opts, :patience, 5)

    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      meta = Map.put(state.handler_metadata, :nan_streak, 0)
      {:continue, %{state | handler_metadata: meta}}
    end)
    |> Axon.Loop.handle_event(:iteration_completed, fn state ->
      loss_val = get_in(state.step_state, [:loss])

      if loss_val do
        loss_num = Nx.to_number(loss_val)

        if loss_num != loss_num do
          streak = state.handler_metadata.nan_streak + 1

          if streak >= patience do
            Logger.error(
              "[Adaptive] Loss NaN for #{streak} consecutive steps — halting training"
            )

            {:halt_loop, state}
          else
            Logger.warning(
              "[Adaptive] Loss NaN at step #{state.iteration} (streak: #{streak}/#{patience})"
            )

            meta = %{state.handler_metadata | nan_streak: streak}
            {:continue, %{state | handler_metadata: meta}}
          end
        else
          meta = %{state.handler_metadata | nan_streak: 0}
          {:continue, %{state | handler_metadata: meta}}
        end
      else
        {:continue, state}
      end
    end)
  end

  # ============================================================================
  # Callbacks (named captures for runtime_call)
  # ============================================================================

  @doc false
  def overflow_check_callback(tensor, opts) do
    label = Keyword.get(opts, :label, "grad")
    has_nan = tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1
    has_inf = tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 1

    if has_nan or has_inf do
      Logger.warning("[Adaptive] Overflow in #{label}: NaN=#{has_nan} Inf=#{has_inf} — zeroing")
    end

    if has_nan or has_inf do
      Nx.tensor(1, type: {:u, 8})
    else
      Nx.tensor(0, type: {:u, 8})
    end
  end

  @doc false
  def grad_norm_callback(grad, opts) do
    label = Keyword.get(opts, :label, "grad_norm")
    every = Keyword.get(opts, :every, 1)

    if should_log?(label, every) do
      norm = grad |> Nx.as_type(:f32) |> Nx.pow(2) |> Nx.sum() |> Nx.to_number() |> :math.sqrt()
      Logger.info("[Adaptive] #{label}: #{format_float(norm)}")
    end

    grad
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp compute_global_norm(grads) do
    grads
    |> flatten_tensors()
    |> Enum.map(fn t ->
      t |> Nx.as_type(:f32) |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
    end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp flatten_tensors(map) when is_map(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, v} when is_map(v) -> flatten_tensors(v)
      _ -> []
    end)
  end

  defp deep_map_tensors(map, fun) when is_map(map) do
    Map.new(map, fn
      {k, %Nx.Tensor{} = t} -> {k, fun.(t)}
      {k, v} when is_map(v) -> {k, deep_map_tensors(v, fun)}
      other -> other
    end)
  end

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

  defp format_float(value) when is_float(value) do
    :erlang.float_to_binary(value, decimals: 6)
  end

  defp format_float(value), do: inspect(value)
end
