defmodule Edifice.Interpretability.SAETrainer do
  @moduledoc """
  Shared trainer for the SAE/transcoder family (INTERP_AUDIT remediation
  steps 1–2: the family had losses but no fit path anywhere).

  Full-batch gradient descent with momentum in a single jitted train step
  (the `Edifice.Interpretability.Probe` house pattern), with the two
  constraints every SAE objective needs to be non-degenerate:

  - **Decoder unit-norm renorm after every update** (projected gradient).
    Without it, any L1 objective is degenerate: the optimizer shrinks the
    codes and grows the decoder instead of becoming sparse.
  - **f32 loss entry** — already enforced inside every module's `loss/4`.

  ## BatchTopK production extras (audit step 2)

  - **Inference threshold**: during fit, tracks the running mean of the
    minimum selected activation per step. Feature firing under batch
    top-k depends on batch composition; rebuilding for inference with
    `inference_threshold: result.threshold` makes firing deterministic
    per-sample (required for ground-truth feature scoring).
  - **AuxK dead-feature loss** (`aux_k:`/`aux_coeff:` opts): features
    whose firing-rate EMA falls below `dead_eps` are revived by an
    auxiliary reconstruction of the residual from the top `aux_k` dead
    pre-activations (Gao et al. 2024). Applies to any module whose
    container exposes `pre_acts` (all of them), but matters most for
    batch top-k.

  ## Usage

      acts = ...                              # {n, d} activation matrix
      result = SAETrainer.fit(:batch_top_k_sae, acts,
        dict_size: 512, batch_k: 64, steps: 500, compiler: EXLA)

      # Deterministic inference encoder:
      model = Edifice.build(:batch_top_k_sae,
        input_size: result.build_opts[:input_size],
        dict_size: 512,
        inference_threshold: result.threshold)

  Pass `compiler: EXLA` for real workloads; the default
  `Nx.Defn.Evaluator` never touches an accelerator. Full-batch only —
  minibatching is a follow-up (slice activations outside if needed).
  """

  require Logger

  @trainer_keys [
    :steps,
    :lr,
    :momentum,
    :l1_coeff,
    :compiler,
    :seed,
    :aux_k,
    :aux_coeff,
    :dead_eps,
    :targets
  ]

  @doc """
  Fit an SAE/transcoder on an `{n, d}` activation matrix.

  `arch_or_module` is a registry atom (`:sae`, `:batch_top_k_sae`, ...)
  or the module itself. Trainer options (all others are passed to the
  module's `build/1`; `:input_size` defaults from the data):

    - `:steps` - gradient steps (default 300)
    - `:lr` - learning rate (default 1.0e-3)
    - `:momentum` - momentum coefficient (default 0.9)
    - `:l1_coeff` - forwarded to the module's `loss/4` (default 1.0e-3)
    - `:compiler` - `Nx.Defn` compiler (default `Nx.Defn.Evaluator`)
    - `:seed` - init seed (default 42)
    - `:aux_k` - AuxK dead-feature budget per sample (default 0 = off)
    - `:aux_coeff` - AuxK loss weight (default 1/32)
    - `:dead_eps` - firing-rate EMA below which a feature counts as dead
      (default 1.0e-3)

  Returns `%{params, model, threshold, history, dead_count, module,
  build_opts}` — `params` is an `Axon.ModelState` usable with any build
  of the same options; `threshold` is the running-mean min selected
  activation (see moduledoc); `history` is the per-step loss list.
  """
  def fit(arch_or_module, activations, opts \\ []) do
    module = resolve_module(arch_or_module)

    steps = Keyword.get(opts, :steps, 300)
    lr = Keyword.get(opts, :lr, 1.0e-3)
    momentum = Keyword.get(opts, :momentum, 0.9)
    l1_coeff = Keyword.get(opts, :l1_coeff, 1.0e-3)
    compiler = Keyword.get(opts, :compiler, Nx.Defn.Evaluator)
    seed = Keyword.get(opts, :seed, 42)
    aux_k = Keyword.get(opts, :aux_k, 0)
    aux_coeff = Keyword.get(opts, :aux_coeff, 1.0 / 32.0)
    dead_eps = Keyword.get(opts, :dead_eps, 1.0e-3)

    x = Nx.as_type(activations, :f32)
    {n, d} = Nx.shape(x)

    # Transcoders map input-space to a DIFFERENT output space; pass the
    # target activations via :targets. Autoencoders reconstruct x itself.
    y = opts |> Keyword.get(:targets, x) |> Nx.as_type(:f32)

    build_opts =
      opts
      |> Keyword.drop(@trainer_keys)
      |> Keyword.put_new(:input_size, d)
      |> Keyword.put(:output, :container)

    _ = seed
    model = module.build(build_opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    input_key = model |> Axon.get_inputs() |> Map.keys() |> hd()
    template = %{input_key => Nx.template({n, d}, :f32)}
    model_state = init_fn.(template, Axon.ModelState.empty())

    decoder_keys = decoder_keys!(model_state.data)
    dict_size = Keyword.get(build_opts, :dict_size, dict_size_of(model_state.data, decoder_keys))

    # The whole update — forward, loss (+AuxK), grads, momentum step,
    # decoder renorm, firing/threshold stats — is ONE jitted executable
    train_step =
      Nx.Defn.jit(
        fn data, momentum_acc, x, y, dead_mask ->
          {loss, grads} =
            Nx.Defn.value_and_grad(data, fn d ->
              out = predict_fn.(%{model_state | data: d}, %{input_key => x})

              base = module.loss(y, out.reconstruction, out.hidden, l1_coeff: l1_coeff)

              # Plain traced fn (not defn): tensor arithmetic must be Nx.*
              if aux_k > 0 do
                Nx.add(base, aux_loss(out, y, d, decoder_keys, dead_mask, aux_k, aux_coeff))
              else
                base
              end
            end)

          momentum_acc = tree_map2(momentum_acc, grads, &Nx.add(Nx.multiply(momentum, &1), &2))
          data = tree_map2(data, momentum_acc, &Nx.subtract(&1, Nx.multiply(lr, &2)))
          data = renorm_decoders(data, decoder_keys)

          out = predict_fn.(%{model_state | data: data}, %{input_key => x})
          fired = out.hidden |> Nx.greater(0.0) |> Nx.as_type(:f32)

          # Min selected activation this step (+inf when nothing fired)
          min_selected =
            Nx.select(Nx.greater(fired, 0.0), out.hidden, Nx.Constants.infinity(:f32))
            |> Nx.reduce_min()

          firing_rate = Nx.mean(fired, axes: [0])

          {data, momentum_acc, loss, min_selected, firing_rate}
        end,
        compiler: compiler
      )

    # Momentum starts at zero, everything alive
    zero_momentum = tree_map(model_state.data, &Nx.multiply(&1, 0.0))
    data0 = renorm_decoders(model_state.data, decoder_keys)
    alive0 = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {dict_size})

    {data, _m, _ema, history_rev, thresholds_rev} =
      Enum.reduce(1..steps, {data0, zero_momentum, alive0, [], []}, fn _i,
                                                                       {data, m, ema, hist,
                                                                        thrs} ->
        dead_mask = Nx.as_type(Nx.less(ema, dead_eps), :f32)
        {data, m, loss, min_sel, firing_rate} = train_step.(data, m, x, y, dead_mask)

        ema = Nx.add(Nx.multiply(0.9, ema), Nx.multiply(0.1, firing_rate))

        min_sel = Nx.to_number(min_sel)
        thrs = if min_sel == :infinity, do: thrs, else: [min_sel | thrs]

        {data, m, ema, [Nx.to_number(loss) | hist], thrs}
      end)

    history = Enum.reverse(history_rev)
    thresholds = Enum.reverse(thresholds_rev)

    threshold =
      case thresholds do
        [] -> nil
        list -> Enum.sum(list) / length(list)
      end

    final_out = predict_fn.(%{model_state | data: data}, %{input_key => x})
    dead_count = final_out.hidden |> Nx.greater(0.0) |> Nx.any(axes: [0]) |> Nx.logical_not() |> Nx.sum() |> Nx.to_number()

    %{
      params: %{model_state | data: data},
      model: model,
      threshold: threshold,
      history: history,
      dead_count: dead_count,
      module: module,
      build_opts: build_opts
    }
  end

  # ============================================================================
  # Internals
  # ============================================================================

  defp resolve_module(module) when is_atom(module) do
    if match?("Elixir." <> _, Atom.to_string(module)) do
      module
    else
      Edifice.module_for(module)
    end
  end

  # AuxK (Gao et al. 2024): reconstruct the (detached) residual from the
  # top aux_k DEAD pre-activations, giving dead features a gradient path
  defp aux_loss(out, x, data, decoder_keys, dead_mask, aux_k, aux_coeff) do
    residual = Nx.Defn.Kernel.stop_grad(Nx.subtract(x, out.reconstruction))

    masked = Nx.multiply(out.pre_acts, Nx.new_axis(dead_mask, 0))

    {top_values, _} = Nx.top_k(masked, k: aux_k)
    kth = Nx.slice_along_axis(top_values, aux_k - 1, 1, axis: 1)

    keep = Nx.logical_and(Nx.greater_equal(masked, kth), Nx.greater(masked, 0.0))
    z_aux = Nx.select(keep, masked, Nx.tensor(0.0, type: Nx.type(masked)))

    # Reconstruct through the (first) decoder kernel, no bias
    [decoder_key | _] = decoder_keys
    aux_recon = Nx.dot(z_aux, data[decoder_key]["kernel"])

    # Zero the whole term when nothing is dead (indicator in-graph)
    any_dead = Nx.any(Nx.greater(dead_mask, 0.0)) |> Nx.as_type(:f32)

    Nx.mean(Nx.pow(Nx.subtract(residual, aux_recon), 2))
    |> Nx.multiply(aux_coeff)
    |> Nx.multiply(any_dead)
  end

  # Project every decoder kernel's rows (feature directions) back to unit
  # L2 norm — the constraint that keeps L1 objectives meaningful
  defp renorm_decoders(data, decoder_keys) do
    Enum.reduce(decoder_keys, data, fn key, acc ->
      kernel = acc[key]["kernel"]

      norms =
        kernel
        |> Nx.pow(2)
        |> Nx.sum(axes: [1], keep_axes: true)
        |> Nx.sqrt()
        |> Nx.max(1.0e-8)

      put_in(acc[key]["kernel"], Nx.divide(kernel, norms))
    end)
  end

  defp decoder_keys!(data) do
    case data |> Map.keys() |> Enum.filter(&String.ends_with?(&1, "decoder")) |> Enum.sort() do
      [] ->
        raise ArgumentError,
              "no decoder layer (name ending in \"decoder\") found in params " <>
                "(available: #{data |> Map.keys() |> Enum.sort() |> Enum.join(", ")})"

      keys ->
        keys
    end
  end

  defp dict_size_of(data, [decoder_key | _]) do
    data[decoder_key]["kernel"] |> Nx.shape() |> elem(0)
  end

  defp tree_map(%Nx.Tensor{} = t, fun), do: fun.(t)

  defp tree_map(%{} = map, fun) when not is_struct(map),
    do: Map.new(map, fn {k, v} -> {k, tree_map(v, fun)} end)

  defp tree_map2(%Nx.Tensor{} = a, b, fun), do: fun.(a, b)

  defp tree_map2(%{} = a, b, fun) when not is_struct(a) do
    Map.new(a, fn {k, v} -> {k, tree_map2(v, Map.fetch!(b, k), fun)} end)
  end
end
