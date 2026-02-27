defmodule Mix.Tasks.Edifice.Viz do
  @moduledoc """
  Visualize an Edifice architecture's layer structure.

  ## Usage

      mix edifice.viz mamba
      mix edifice.viz mamba --format tree
      mix edifice.viz mamba --format mermaid
      mix edifice.viz mamba --hidden_dim 128 --num_layers 4
      mix edifice.viz whisper --component encoder

  ## Formats

    * `table` (default) — Keras-style summary with shapes and parameter counts.
    * `tree` — Lightweight ASCII tree showing operation hierarchy.
    * `mermaid` — Raw Mermaid flowchart text for docs or mermaid.live.

  ## Options

    * `--format` / `-f` — Output format: `table`, `tree`, or `mermaid` (default: `table`)
    * `--component` / `-c` — For tuple-returning models: `all` (default), `encoder`,
      `decoder`, `generator`, `discriminator`, or a 1-based index like `1` or `2`

  All remaining `--key value` pairs are passed as build options to the architecture
  module. Values that parse as integers are converted automatically.
  """

  use Mix.Task
  @dialyzer [:no_missing_calls, :no_return]

  @shortdoc "Visualize an Edifice architecture's layer structure"

  @impl Mix.Task
  def run(args) do
    Application.ensure_all_started(:nx)

    {arch_name, format, component, build_opts} = parse_args(args)

    format = parse_format(format)

    # Validate the architecture name
    unless arch_name in Edifice.list_architectures() do
      similar = suggest_similar(arch_name)
      hint = if similar, do: " Did you mean :#{similar}?", else: ""
      available = Edifice.list_architectures() |> Enum.join(", ")
      Mix.raise("Unknown architecture :#{arch_name}.#{hint}\n\nAvailable: #{available}")
    end

    # Build the model
    Mix.shell().info("Building :#{arch_name}...")
    result = Edifice.build(arch_name, build_opts)

    # Select and render
    output = render_result(result, format, component, arch_name)
    Mix.shell().info(output)
  end

  # -------------------------------------------------------------------
  # Format parsing
  # -------------------------------------------------------------------

  defp parse_format("table"), do: :table
  defp parse_format("tree"), do: :tree
  defp parse_format("mermaid"), do: :mermaid

  defp parse_format(other) do
    Mix.raise("Invalid format #{inspect(other)}. Must be table, tree, or mermaid.")
  end

  # -------------------------------------------------------------------
  # Arg parsing
  # -------------------------------------------------------------------

  # Manual arg parsing to support pass-through build options.
  # OptionParser can't handle unknown --key value pairs in strict mode,
  # so we walk the arg list ourselves.
  defp parse_args(args) do
    {arch_name, format, component, build_opts} = do_parse_args(args, nil, "table", "all", [])

    unless arch_name do
      Mix.raise(
        "Usage: mix edifice.viz <architecture> [--format table|tree|mermaid] [--key value ...]"
      )
    end

    {arch_name, format, component, Enum.reverse(build_opts)}
  end

  defp do_parse_args([], arch, fmt, comp, build), do: {arch, fmt, comp, build}

  defp do_parse_args([flag, value | rest], arch, _fmt, comp, build)
       when flag in ["--format", "-f"] do
    do_parse_args(rest, arch, value, comp, build)
  end

  defp do_parse_args([flag, value | rest], arch, fmt, _comp, build)
       when flag in ["--component", "-c"] do
    do_parse_args(rest, arch, fmt, value, build)
  end

  defp do_parse_args([<<"--", key::binary>>, value | rest], arch, fmt, comp, build) do
    do_parse_args(rest, arch, fmt, comp, [{String.to_atom(key), coerce_value(value)} | build])
  end

  defp do_parse_args([<<"-", key::binary>>, value | rest], arch, fmt, comp, build)
       when byte_size(key) > 1 do
    do_parse_args(rest, arch, fmt, comp, [{String.to_atom(key), coerce_value(value)} | build])
  end

  defp do_parse_args([positional | rest], nil, fmt, comp, build) do
    do_parse_args(rest, String.to_atom(positional), fmt, comp, build)
  end

  defp do_parse_args([_positional | rest], arch, fmt, comp, build) do
    do_parse_args(rest, arch, fmt, comp, build)
  end

  defp coerce_value(val) do
    case Integer.parse(val) do
      {int, ""} ->
        int

      _ ->
        case Float.parse(val) do
          {f, ""} -> f
          _ -> String.to_atom(val)
        end
    end
  end

  # -------------------------------------------------------------------
  # Component selection + rendering
  # -------------------------------------------------------------------

  defp render_result(%Axon{} = model, format, _component, arch_name) do
    Edifice.Display.format_build_result(model, format, name: to_string(arch_name))
  end

  defp render_result(tuple, format, component, arch_name) when is_tuple(tuple) do
    elements = Tuple.to_list(tuple)

    case component do
      "all" ->
        labels = guess_labels(arch_name, length(elements))

        elements
        |> Enum.zip(labels)
        |> Enum.map_join("\n\n", fn {model, label} ->
          "=== #{label} ===\n" <>
            Edifice.Display.format_build_result(model, format, name: label)
        end)

      idx when idx in ["1", "2", "3", "4"] ->
        i = String.to_integer(idx) - 1
        select_component(elements, i, format, arch_name)

      name ->
        labels = guess_labels(arch_name, length(elements)) |> Enum.map(&String.downcase/1)
        idx = Enum.find_index(labels, &(&1 == String.downcase(name)))

        if idx do
          select_component(elements, idx, format, arch_name)
        else
          Mix.raise("Unknown component #{inspect(name)}. Available: #{Enum.join(labels, ", ")}")
        end
    end
  end

  defp select_component(elements, idx, format, arch_name) do
    if idx >= 0 and idx < length(elements) do
      Edifice.Display.format_build_result(Enum.at(elements, idx), format,
        name: to_string(arch_name)
      )
    else
      Mix.raise("Component index out of range (1..#{length(elements)})")
    end
  end

  # Best-effort labels for known tuple-returning architectures
  @tuple_labels %{
    {:vae, 2} => ["Encoder", "Decoder"],
    {:vq_vae, 2} => ["Encoder", "Decoder"],
    {:mae, 2} => ["Encoder", "Decoder"],
    {:act, 2} => ["Encoder", "Decoder"],
    {:whisper, 2} => ["Encoder", "Decoder"],
    {:gan, 2} => ["Generator", "Discriminator"],
    {:simclr, 2} => ["Backbone", "Projection"],
    {:byol, 2} => ["Online", "Target"],
    {:barlow_twins, 2} => ["Backbone", "Projection"],
    {:vicreg, 2} => ["Backbone", "Projection"],
    {:jepa, 2} => ["Context Encoder", "Predictor"],
    {:temporal_jepa, 2} => ["Context Encoder", "Predictor"],
    {:normalizing_flow, 2} => ["Flow Model", "Log Det"],
    {:speculative_decoding, 2} => ["Draft", "Verifier"],
    {:world_model, 3} => ["Encoder", "Dynamics", "Reward"],
    {:byte_latent_transformer, 3} => ["Encoder", "Latent Transformer", "Decoder"],
    {:world_model, 4} => ["Encoder", "Dynamics", "Reward", "Decoder"]
  }

  defp guess_labels(arch_name, count) do
    Map.get(@tuple_labels, {arch_name, count}, Enum.map(1..count, &"Component #{&1}"))
  end

  # -------------------------------------------------------------------
  # Fuzzy matching
  # -------------------------------------------------------------------

  defp suggest_similar(name) do
    name_str = to_string(name)

    Edifice.list_architectures()
    |> Enum.map(fn arch -> {arch, String.jaro_distance(name_str, to_string(arch))} end)
    |> Enum.filter(fn {_, dist} -> dist > 0.7 end)
    |> Enum.sort_by(fn {_, dist} -> dist end, :desc)
    |> case do
      [{arch, _} | _] -> arch
      [] -> nil
    end
  end
end
