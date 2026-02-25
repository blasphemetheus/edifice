defmodule Mix.Tasks.Edifice.ExportGguf do
  @moduledoc """
  Export an Edifice model checkpoint to GGUF format.

  GGUF (GPT-Generated Unified Format) is the binary format used by llama.cpp
  and Ollama for efficient CPU/GPU inference.

  ## Usage

      mix edifice.export_gguf --checkpoint path/to/checkpoint.bin --output model.gguf

  ## Options

    * `--checkpoint` or `-c` - Path to the Edifice checkpoint file (required)
    * `--output` or `-o` - Output GGUF file path (required)
    * `--quantization` or `-q` - Quantization type: f32, f16, q8_0 (default: q8_0)
    * `--architecture` or `-a` - Target architecture name (default: llama)
    * `--config` - Path to model config JSON (optional, will try to infer from checkpoint)
    * `--num-layers` - Number of layers if not in config (default: 4)
    * `--hidden-size` - Hidden size if not in config (default: 256)
    * `--num-heads` - Number of attention heads if not in config (default: 8)
    * `--num-kv-heads` - Number of KV heads if not in config (optional, defaults to num-heads)
    * `--context-length` - Context length (default: 2048)
    * `--vocab-size` - Vocabulary size (default: 32000)
    * `--name` - Model name for metadata (default: "edifice_model")

  ## Examples

      # Export with Q8_0 quantization (default)
      mix edifice.export_gguf -c model.axon -o model.gguf

      # Export with F16 precision
      mix edifice.export_gguf -c model.axon -o model.gguf -q f16

      # Export with explicit config
      mix edifice.export_gguf -c model.axon -o model.gguf \\
        --num-layers 12 --hidden-size 768 --num-heads 12

  """

  use Mix.Task

  @shortdoc "Export Edifice model to GGUF format"

  @switches [
    checkpoint: :string,
    output: :string,
    quantization: :string,
    architecture: :string,
    config: :string,
    num_layers: :integer,
    hidden_size: :integer,
    num_heads: :integer,
    num_kv_heads: :integer,
    context_length: :integer,
    vocab_size: :integer,
    name: :string
  ]

  @aliases [
    c: :checkpoint,
    o: :output,
    q: :quantization,
    a: :architecture
  ]

  @impl Mix.Task
  def run(args) do
    # Start necessary applications
    Application.ensure_all_started(:nx)

    {opts, _remaining, invalid} = OptionParser.parse(args, switches: @switches, aliases: @aliases)

    if invalid != [] do
      Mix.raise("Invalid options: #{inspect(invalid)}")
    end

    checkpoint_path = opts[:checkpoint]
    output_path = opts[:output]

    if is_nil(checkpoint_path) do
      Mix.raise("--checkpoint is required. Use --help for usage.")
    end

    if is_nil(output_path) do
      Mix.raise("--output is required. Use --help for usage.")
    end

    unless File.exists?(checkpoint_path) do
      Mix.raise("Checkpoint file not found: #{checkpoint_path}")
    end

    # Parse quantization option
    quantization =
      case opts[:quantization] do
        nil -> :q8_0
        "f32" -> :f32
        "f16" -> :f16
        "q8_0" -> :q8_0
        other -> Mix.raise("Invalid quantization type: #{other}. Must be f32, f16, or q8_0")
      end

    # Load checkpoint
    Mix.shell().info("Loading checkpoint from #{checkpoint_path}...")
    params = load_checkpoint(checkpoint_path)

    # Build config from options or config file
    config = build_config(opts, params)

    Mix.shell().info("Exporting to GGUF with #{quantization} quantization...")
    Mix.shell().info("  Architecture: #{config[:architecture] || "llama"}")
    Mix.shell().info("  Layers: #{config[:num_layers]}")
    Mix.shell().info("  Hidden size: #{config[:hidden_size]}")
    Mix.shell().info("  Heads: #{config[:num_heads]} (KV: #{config[:num_kv_heads]})")

    export_opts = [
      quantization: quantization,
      architecture: opts[:architecture] || "llama"
    ]

    case Edifice.Export.GGUF.export(params, config, output_path, export_opts) do
      :ok ->
        file_size = File.stat!(output_path).size
        Mix.shell().info("Successfully exported to #{output_path} (#{format_size(file_size)})")

      {:error, reason} ->
        Mix.raise("Export failed: #{inspect(reason)}")
    end
  end

  defp load_checkpoint(path) do
    binary = File.read!(path)

    case Path.extname(path) do
      ".nx" ->
        # Nx.serialize format
        Nx.deserialize(binary)

      _ext ->
        # Most formats use Erlang binary term serialization
        # This works for .axon, .bin, .etf, and most custom formats
        :erlang.binary_to_term(binary)
    end
  end

  defp build_config(opts, params) do
    # Try to load from config file if provided
    base_config =
      if opts[:config] do
        opts[:config]
        |> File.read!()
        |> Jason.decode!(keys: :atoms)
      else
        %{}
      end

    # Infer num_layers from params if not specified
    inferred_layers = infer_num_layers(params)

    # Merge with CLI options (CLI takes precedence)
    %{
      num_layers: opts[:num_layers] || base_config[:num_layers] || inferred_layers || 4,
      hidden_size: opts[:hidden_size] || base_config[:hidden_size] || 256,
      num_heads: opts[:num_heads] || base_config[:num_heads] || 8,
      num_kv_heads:
        opts[:num_kv_heads] || base_config[:num_kv_heads] || opts[:num_heads] ||
          base_config[:num_heads] || 8,
      context_length: opts[:context_length] || base_config[:context_length] || 2048,
      vocab_size: opts[:vocab_size] || base_config[:vocab_size] || 32000,
      name: opts[:name] || base_config[:name] || "edifice_model"
    }
  end

  # Try to infer number of layers from parameter names
  defp infer_num_layers(params) do
    flat_params = flatten_param_names(params)

    layer_nums =
      flat_params
      |> Enum.flat_map(fn name ->
        case Regex.run(~r/block_(\d+)/, name) do
          [_, num] -> [String.to_integer(num)]
          _ -> []
        end
      end)

    if Enum.empty?(layer_nums), do: nil, else: Enum.max(layer_nums)
  end

  defp flatten_param_names(%Axon.ModelState{data: data}) do
    flatten_param_names(data)
  end

  defp flatten_param_names(params) when is_map(params) do
    Enum.flat_map(params, fn {key, value} ->
      case value do
        %Nx.Tensor{} ->
          [to_string(key)]

        nested when is_map(nested) ->
          nested
          |> flatten_param_names()
          |> Enum.map(&"#{key}.#{&1}")

        _ ->
          []
      end
    end)
  end

  defp format_size(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_size(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"

  defp format_size(bytes) when bytes < 1024 * 1024 * 1024,
    do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"

  defp format_size(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"
end
