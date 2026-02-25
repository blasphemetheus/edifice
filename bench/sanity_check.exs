# Architecture Sanity Check
#
# Deeper health validation than the sweep: checks each architecture for
# NaN/Inf outputs, dead outputs (all zeros), flat outputs (near-zero variance),
# and gradient flow (can a loss backprop through the model).
#
# Usage:
#   mix run bench/sanity_check.exs                        # all architectures
#   CHECK_ONLY=mamba,var,trellis mix run bench/sanity_check.exs  # specific ones
#   CHECK_FAMILY=generative mix run bench/sanity_check.exs       # by family
#
# Requires EXLA compiled (EXLA_TARGET=host for CPU, or CUDA).

Nx.default_backend(EXLA.Backend)

# Load specs from FullSweep without triggering its run()
System.put_env("EDIFICE_NO_AUTORUN", "1")
Code.require_file("bench/full_sweep.exs")

defmodule SanityCheck do
  @doc """
  Run sanity checks on architecture specs from FullSweep.
  """
  def run do
    all_specs = FullSweep.specs()

    # Filter by CHECK_ONLY (comma-separated names) or CHECK_FAMILY
    specs = filter_specs(all_specs)
    total = length(specs)

    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Sanity Check — #{total} architectures")
    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 28)}" <>
        "#{String.pad_trailing("Family", 15)}" <>
        "#{String.pad_trailing("NaN", 6)}" <>
        "#{String.pad_trailing("Inf", 6)}" <>
        "#{String.pad_trailing("Zero", 6)}" <>
        "#{String.pad_trailing("Flat", 6)}" <>
        "#{String.pad_trailing("Grad", 6)}" <>
        "Verdict"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 78))

    results =
      for {name, family, build_fn, input_fn} <- specs do
        result = check_one(name, family, build_fn, input_fn)
        :erlang.garbage_collect()
        result
      end

    IO.puts("")
    IO.puts("  " <> String.duplicate("-", 78))

    # Summary
    {ok, failed} = Enum.split_with(results, fn r -> r.status != :error end)
    healthy = Enum.filter(ok, fn r -> r.verdict == :healthy end)
    warnings = Enum.filter(ok, fn r -> r.verdict != :healthy end)

    IO.puts("")
    IO.puts("  SUMMARY: #{length(healthy)} healthy, #{length(warnings)} warnings, #{length(failed)} errors")
    IO.puts("")

    if warnings != [] do
      IO.puts("  WARNINGS:")

      for r <- warnings do
        issues = Enum.join(r.issues, ", ")
        IO.puts("    #{String.pad_trailing(to_string(r.name), 28)} #{issues}")
      end

      IO.puts("")
    end

    if failed != [] do
      IO.puts("  ERRORS:")

      for r <- failed do
        IO.puts("    #{String.pad_trailing(to_string(r.name), 28)} #{r.error}")
      end

      IO.puts("")
    end
  end

  defp filter_specs(all_specs) do
    cond do
      (only = System.get_env("CHECK_ONLY")) && only != "" ->
        names =
          only
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> Enum.map(&String.to_atom/1)
          |> MapSet.new()

        Enum.filter(all_specs, fn {name, _, _, _} -> name in names end)

      (family = System.get_env("CHECK_FAMILY")) && family != "" ->
        families =
          family
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> MapSet.new()

        Enum.filter(all_specs, fn {_, fam, _, _} -> fam in families end)

      true ->
        all_specs
    end
  end

  defp check_one(name, family, build_fn, input_fn) do
    try do
      model = build_fn.()
      input = input_fn.()

      template =
        case input do
          %{} = map when not is_struct(map) ->
            Map.new(map, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

          tensor ->
            Nx.template(Nx.shape(tensor), Nx.type(tensor))
        end

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Forward pass
      output = predict_fn.(params, input)
      tensors = collect_tensors(output)

      # Check NaN
      has_nan = Enum.any?(tensors, fn t -> Nx.any(Nx.is_nan(t)) |> Nx.to_number() == 1 end)

      # Check Inf
      has_inf = Enum.any?(tensors, fn t -> Nx.any(Nx.is_infinity(t)) |> Nx.to_number() == 1 end)

      # Check all-zero
      all_zero =
        tensors != [] and
          Enum.all?(tensors, fn t -> Nx.all(Nx.equal(t, 0)) |> Nx.to_number() == 1 end)

      # Check low variance (flat output = dead network)
      low_variance =
        not has_nan and not has_inf and not all_zero and tensors != [] and
          Enum.all?(tensors, fn t ->
            flat = Nx.reshape(t, {Nx.size(t)})
            Nx.variance(flat) |> Nx.to_number() < 1.0e-8
          end)

      # Gradient flow check: can we backprop a scalar loss through the model?
      # Returns :ok, :no_params (not a warning — e.g. position encodings), or :fail
      grad_result = check_gradient_flow(predict_fn, params, input)

      # Build verdict
      issues = []
      issues = if has_nan, do: ["NaN output" | issues], else: issues
      issues = if has_inf, do: ["Inf output" | issues], else: issues
      issues = if all_zero, do: ["all-zero output" | issues], else: issues
      issues = if low_variance, do: ["flat output (var<1e-8)" | issues], else: issues
      issues = Enum.reverse(issues)

      # Gradient flow is informational — some architectures (hash lookups,
      # discrete ops) are legitimately non-differentiable by design.
      verdict = if issues == [], do: :healthy, else: :warning

      tag = fn bool -> if bool, do: "X", else: "." end
      grad_tag = case grad_result do
        :ok -> "."
        :no_params -> "-"
        :fail -> "X"
      end

      IO.puts(
        "  #{String.pad_trailing(to_string(name), 28)}" <>
          "#{String.pad_trailing(family, 15)}" <>
          "#{String.pad_trailing(tag.(has_nan), 6)}" <>
          "#{String.pad_trailing(tag.(has_inf), 6)}" <>
          "#{String.pad_trailing(tag.(all_zero), 6)}" <>
          "#{String.pad_trailing(tag.(low_variance), 6)}" <>
          "#{String.pad_trailing(grad_tag, 6)}" <>
          if(verdict == :healthy, do: "OK", else: "WARN")
      )

      %{
        name: name,
        family: family,
        status: :ok,
        verdict: verdict,
        issues: issues,
        has_nan: has_nan,
        has_inf: has_inf,
        all_zero: all_zero,
        low_variance: low_variance,
        grad_result: grad_result
      }
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 60)

        IO.puts(
          "  #{String.pad_trailing(to_string(name), 28)}" <>
            "#{String.pad_trailing(family, 15)}" <>
            "#{String.pad_trailing("-", 6)}" <>
            "#{String.pad_trailing("-", 6)}" <>
            "#{String.pad_trailing("-", 6)}" <>
            "#{String.pad_trailing("-", 6)}" <>
            "#{String.pad_trailing("-", 6)}" <>
            "ERR"
        )

        %{name: name, family: family, status: :error, verdict: :error, error: msg, issues: []}
    end
  end

  # Try to compute gradient of a scalar loss w.r.t. params.
  # Uses Axon.Loop's pattern: wrap value_and_grad inside a single jit call
  # so that trainable params, frozen state, and input are all JIT arguments.
  defp check_gradient_flow(predict_fn, params, input) do
    try do
      trainable = Axon.ModelState.trainable_parameters(params)

      if trainable == %{} do
        :no_params
      else
        # Single JIT function that receives all tensors as arguments:
        # - trainable_params: the parameters to differentiate
        # - full_data: complete params.data (trainable + state + frozen)
        # - inp: model input
        # Inside JIT, value_and_grad traces only through trainable_params.
        step_fn = fn trainable_params, full_data, inp ->
          {_loss, grad} =
            Nx.Defn.value_and_grad(trainable_params, fn tp ->
              merged = deep_merge(full_data, tp)

              ms = %Axon.ModelState{
                data: merged,
                parameters: params.parameters,
                state: params.state,
                frozen_parameters: params.frozen_parameters
              }

              output = predict_fn.(ms, inp)
              sum_container(output)
            end)

          grad
        end

        grad = Nx.Defn.jit(step_fn, on_conflict: :reuse).(trainable, params.data, input)

        # Check that at least some gradient values are non-zero
        grad_tensors = collect_map_tensors(grad)

        has_nonzero =
          grad_tensors != [] and
            Enum.any?(grad_tensors, fn t ->
              Nx.any(Nx.not_equal(t, 0)) |> Nx.to_number() == 1
            end)

        if has_nonzero, do: :ok, else: :fail
      end
    rescue
      e ->
        if System.get_env("CHECK_DEBUG") do
          IO.puts("    [grad debug] #{Exception.message(e) |> String.slice(0, 200)}")
        end

        :fail
    end
  end

  # Deep-merge two nested maps, preferring values from `override`
  defp deep_merge(base, override) when is_map(base) and is_map(override) do
    Map.merge(base, override, fn
      _key, base_val, override_val when is_map(base_val) and is_map(override_val) ->
        deep_merge(base_val, override_val)

      _key, _base_val, override_val ->
        override_val
    end)
  end

  # Sum all tensors in a container (tuple, map, or plain tensor) into a scalar.
  defp sum_container(%Nx.Tensor{} = t), do: Nx.sum(t)

  defp sum_container(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.reduce(Nx.tensor(0.0), fn elem, acc -> Nx.add(acc, sum_container(elem)) end)
  end

  defp sum_container(%{} = map) when not is_struct(map) do
    map
    |> Map.values()
    |> Enum.reduce(Nx.tensor(0.0), fn val, acc -> Nx.add(acc, sum_container(val)) end)
  end

  defp sum_container(_other), do: Nx.tensor(0.0)

  defp collect_map_tensors(map) when is_map(map) do
    Enum.flat_map(map, fn
      {_k, %Nx.Tensor{} = t} -> [t]
      {_k, %{} = nested} -> collect_map_tensors(nested)
      _ -> []
    end)
  end

  # Flatten output structure to list of tensors
  defp collect_tensors(%Nx.Tensor{} = t), do: [t]

  defp collect_tensors(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.flat_map(&collect_tensors/1)
  end

  defp collect_tensors(%{} = map) when not is_struct(map) do
    map |> Map.values() |> Enum.flat_map(&collect_tensors/1)
  end

  defp collect_tensors(_other), do: []
end

SanityCheck.run()
