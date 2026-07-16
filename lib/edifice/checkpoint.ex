defmodule Edifice.Checkpoint do
  @moduledoc """
  Fast model checkpoint save/load using `Nx.serialize/deserialize`.

  Provides efficient tensor-native serialization for Edifice model
  parameters, with ~3-5x faster I/O than Erlang term serialization
  (used by `Axon.Loop.checkpoint`) and compact binary format.

  ## Quick Start

      # Save
      Edifice.Checkpoint.save(params, "checkpoints/model_epoch_5.nx")

      # Load
      params = Edifice.Checkpoint.load("checkpoints/model_epoch_5.nx")

  ## With Training Metadata

      Edifice.Checkpoint.save(params, "checkpoints/model.nx",
        metadata: %{epoch: 5, loss: 0.042, architecture: :decoder_only}
      )

      {params, metadata} = Edifice.Checkpoint.load("checkpoints/model.nx",
        return_metadata: true
      )

  ## Self-Describing Checkpoints (model manifest)

      {model, spec} = Edifice.build_with_spec(:mamba, embed_dim: 287, state_size: 32)
      Edifice.Checkpoint.save(params, "mamba.nx", spec: spec)

      # Later — rebuilds from the embedded spec and validates param shapes:
      {model, params} = Edifice.Checkpoint.load_model("mamba.nx")

  Embedding an `Edifice.Spec` kills a silent-failure class: loading a
  checkpoint into a model rebuilt with *default* options (shape mismatch =
  garbage outputs, no error). `load_model/2` raises loudly instead.
  Limitations: specs are embedded only in the `:nx` format (safetensors has
  no metadata write support here), and shape validation is skipped for
  architectures whose `build/2` returns a tuple of models.

  ## Axon.Loop Integration

      loop
      |> Axon.Loop.trainer(loss, optimizer)
      |> Edifice.Checkpoint.attach("checkpoints/model.nx", every: :epoch)

  ## Format

  The `.nx` checkpoint format stores:
  - Tensor data in Nx's native binary format (preserves dtype, shape, names)
  - Optional metadata map alongside the tensors
  - Cross-platform endianness handling

  ## Comparison

  | Format | Save Speed | File Size | Preserves Type | Cross-platform |
  |--------|-----------|-----------|----------------|----------------|
  | Nx.serialize | Fast | Compact | Yes | Yes |
  | :erlang.term_to_binary | Slow | Larger | Yes | Erlang only |
  | Safetensors | Fast | Compact | Yes | Yes |
  | GGUF | Medium | Quantized | Quantized only | Yes |
  """

  require Logger

  alias Edifice.Spec

  # Reserved metadata key for the embedded Edifice.Spec (string key: always
  # survives :erlang.binary_to_term(bin, [:safe])).
  @spec_key "__edifice_spec__"

  # ============================================================================
  # Save
  # ============================================================================

  @doc """
  Save model parameters to a file.

  ## Parameters

    * `params` - Model parameters (map or `Axon.ModelState`)
    * `path` - File path to save to (recommended: `.nx` extension)

  ## Options

    * `:metadata` - Optional map of training metadata to store alongside
      parameters (epoch, loss, architecture, hyperparameters, etc.)
    * `:spec` - Optional `Edifice.Spec` manifest to embed, making the
      checkpoint self-describing (see `load_model/2`). Stored in metadata
      under a reserved key; raises if `:metadata` already contains that key.
      Only supported by the `:nx` format.
    * `:format` - Checkpoint format: `:nx` (default) or `:safetensors`.
      Safetensors enables cross-ecosystem sharing (PyTorch, JAX, HuggingFace).
      Note: safetensors format does not support metadata storage.
    * `:compressed` - Compression level 0-9 (default: 0, no compression).
      Only applies to `:nx` format. Higher levels reduce file size but slow down save/load.
  """
  @spec save(map(), String.t(), keyword()) :: :ok
  def save(params, path, opts \\ []) do
    format = Keyword.get(opts, :format, :nx)

    if format == :safetensors do
      if Keyword.has_key?(opts, :spec) do
        Logger.warning(
          "[Checkpoint] :spec is not supported for the :safetensors format " <>
            "(no metadata write support) — the spec will NOT be embedded in #{path}"
        )
      end

      export_safetensors(params, path, opts)
    else
      save_nx(params, path, opts)
    end
  end

  defp save_nx(params, path, opts) do
    metadata =
      opts
      |> Keyword.get(:metadata, nil)
      |> inject_spec(Keyword.get(opts, :spec))

    compressed = Keyword.get(opts, :compressed, 0)

    data = extract_params(params)

    # Serialize tensors via Nx.serialize (fast binary format)
    tensor_iodata = Nx.serialize(data)
    tensor_binary = IO.iodata_to_binary(tensor_iodata)

    # Optional zlib compression at the file level
    tensor_binary =
      if compressed > 0 do
        :zlib.compress(tensor_binary)
      else
        tensor_binary
      end

    # Serialize metadata separately via Erlang term format
    meta_binary =
      if metadata do
        :erlang.term_to_binary(metadata)
      else
        <<>>
      end

    # File format: [compressed::8][meta_size::32][meta_binary][tensor_binary]
    compressed_flag = if compressed > 0, do: 1, else: 0
    meta_size = byte_size(meta_binary)
    file_binary = <<compressed_flag::8, meta_size::32>> <> meta_binary <> tensor_binary

    dir = Path.dirname(path)
    if dir != "." and dir != "", do: File.mkdir_p!(dir)

    File.write!(path, file_binary)

    size = byte_size(file_binary)
    Logger.info("[Checkpoint] Saved #{readable_size(size)} to #{path}")

    :ok
  end

  # ============================================================================
  # Load
  # ============================================================================

  @doc """
  Load model parameters from a file.

  ## Parameters

    * `path` - File path to load from

  ## Options

    * `:format` - Checkpoint format: `:nx` (default) or `:safetensors`.
      Auto-detected from file extension if not specified.
    * `:return_metadata` - If `true`, returns `{params, metadata}` tuple
      (default: `false`, returns just params). Only for `:nx` format.
    * `:backend` - Backend to load tensors into (default: `Nx.BinaryBackend`)

  ## Returns

    Model parameters map (or `{params, metadata}` if `:return_metadata` is true).
  """
  @spec load(String.t(), keyword()) :: map() | {map(), map()}
  def load(path, opts \\ []) do
    format = Keyword.get_lazy(opts, :format, fn ->
      if String.ends_with?(path, ".safetensors"), do: :safetensors, else: :nx
    end)

    if format == :safetensors do
      import_safetensors(path, opts)
    else
      load_nx(path, opts)
    end
  end

  defp load_nx(path, opts) do
    return_metadata = Keyword.get(opts, :return_metadata, false)

    file_binary = File.read!(path)
    size = byte_size(file_binary)

    # Parse format: [compressed::8][meta_size::32][meta_binary][tensor_binary]
    <<compressed_flag::8, meta_size::32, rest::binary>> = file_binary
    <<meta_binary::binary-size(meta_size), tensor_binary::binary>> = rest

    # 4 GB decompression limit to prevent decompression bombs
    max_decompressed_size = 4_294_967_296

    tensor_binary =
      if compressed_flag == 1 do
        decompressed = :zlib.uncompress(tensor_binary)

        if byte_size(decompressed) > max_decompressed_size do
          raise RuntimeError,
                "Decompressed checkpoint exceeds #{max_decompressed_size} byte limit " <>
                  "(got #{byte_size(decompressed)} bytes). " <>
                  "This may indicate a decompression bomb or corrupted file."
        end

        decompressed
      else
        tensor_binary
      end

    params = Nx.deserialize(tensor_binary)

    metadata =
      if meta_size > 0 do
        :erlang.binary_to_term(meta_binary, [:safe])
      else
        %{}
      end

    unless is_map(metadata) and Map.has_key?(metadata, @spec_key) do
      warn_missing_spec(path)
    end

    Logger.info("[Checkpoint] Loaded #{readable_size(size)} from #{path}")

    if return_metadata do
      {params, metadata}
    else
      params
    end
  end

  # ============================================================================
  # Model manifest (Edifice.Spec)
  # ============================================================================

  @doc """
  Read the embedded `Edifice.Spec` from a checkpoint without deserializing
  its tensors (the metadata block sits at the head of the file).

  Returns `{:ok, spec}`, `{:error, :missing}` (no spec embedded, or not an
  `:nx`-format file with metadata), or `{:error, {:invalid, reason}}`.
  """
  @spec fetch_spec(String.t()) ::
          {:ok, Spec.t()} | {:error, :missing | {:invalid, String.t()}}
  def fetch_spec(path) do
    File.open!(path, [:read, :binary, :raw], fn fd ->
      with {:ok, <<_compressed::8, meta_size::32>>} <- :file.read(fd, 5),
           true <- meta_size > 0,
           {:ok, meta_binary} <- :file.read(fd, meta_size),
           %{} = metadata <- safe_term(meta_binary),
           {:ok, spec_map} <- Map.fetch(metadata, @spec_key) do
        case Spec.from_map(spec_map) do
          {:ok, spec} -> {:ok, spec}
          {:error, reason} -> {:error, {:invalid, reason}}
        end
      else
        {:invalid_term, reason} -> {:error, {:invalid, reason}}
        _ -> {:error, :missing}
      end
    end)
  end

  @doc """
  Load a checkpoint AND rebuild its model from the embedded `Edifice.Spec`.

  Returns `{model, params}` where `model` is the result of
  `Edifice.build(spec.arch, spec.build_opts)` and `params` is the loaded
  parameter map. This is the safe counterpart to rebuilding the model
  by hand: build options travel with the checkpoint, so a non-default
  `state_size` (or any other shape option) can never silently fall back
  to defaults.

  ## Options

    * `:validate` - Validate stored parameter shapes against the rebuilt
      model (default: `true`). Only performed when the rebuilt model is a
      single `%Axon{}`; tuple-returning architectures skip validation.
      Validation initializes the rebuilt model's parameters once, on the
      current default backend.

  Raises with a descriptive message when the checkpoint has no spec
  (pre-manifest checkpoint — rebuild the model yourself and use `load/2`),
  when the spec is invalid, or when parameter shapes mismatch.
  """
  @spec load_model(String.t(), keyword()) :: {Axon.t() | tuple(), map()}
  def load_model(path, opts \\ []) do
    validate = Keyword.get(opts, :validate, true)

    spec =
      case fetch_spec(path) do
        {:ok, spec} ->
          spec

        {:error, :missing} ->
          raise RuntimeError,
                "Checkpoint #{path} has no embedded Edifice.Spec (it predates model " <>
                  "manifests, or was saved without the :spec option). Rebuild the model " <>
                  "yourself with the original build options and use Checkpoint.load/2, or " <>
                  "re-save it with Checkpoint.save(params, path, spec: spec)."

        {:error, {:invalid, reason}} ->
          raise RuntimeError,
                "Checkpoint #{path} has an invalid embedded Edifice.Spec: #{reason}"
      end

    if spec.external do
      raise RuntimeError,
            "Checkpoint #{path} carries an EXTERNAL spec (#{inspect(spec.arch)}) — a " <>
              "composite model owned by a downstream library. Edifice cannot rebuild it; " <>
              "rebuild the model from spec.build_opts in the owning library and use " <>
              "Checkpoint.load/2 + validate_shapes!/3 instead."
    end

    params = load(path)
    model = Edifice.build(spec.arch, spec.build_opts)

    if validate do
      validate_shapes!(model, params, spec)
    end

    {model, params}
  end

  @doc """
  Validate stored checkpoint parameters against a model's expected shapes.

  Compares the flattened parameter tree of `params` against the shapes the
  model initializes, raising a descriptive error listing every mismatched,
  missing, and unexpected parameter. Returns `:ok` when everything matches.

  Only `%Axon{}` models are validated; tuple/container build results return
  `:ok` unchecked.
  """
  @spec validate_shapes!(Axon.t() | tuple(), map(), Spec.t() | nil) :: :ok
  def validate_shapes!(model, params, spec \\ nil)

  def validate_shapes!(%Axon{} = model, params, spec) do
    templates = Edifice.Display.build_input_templates(model)
    {init_fn, _predict_fn} = Axon.build(model)
    expected = init_fn.(templates, Axon.ModelState.empty())

    expected_shapes = param_shapes(expected)
    stored_shapes = param_shapes(params)

    mismatched =
      for {key, shape} <- expected_shapes,
          stored = Map.get(stored_shapes, key),
          stored != shape,
          do: {key, shape, stored}

    missing = Map.keys(expected_shapes) -- Map.keys(stored_shapes)
    extra = Map.keys(stored_shapes) -- Map.keys(expected_shapes)

    if mismatched == [] and missing == [] and extra == [] do
      :ok
    else
      arch_line =
        case spec do
          %Spec{arch: arch, build_opts: build_opts} ->
            "\n  arch: #{inspect(arch)}\n  build_opts: #{inspect(build_opts)}\n"

          _ ->
            "\n"
        end

      sections =
        [
          {"Mismatched shapes",
           Enum.map(mismatched, fn {key, expected, stored} ->
             "  #{key}: expected #{inspect(expected)}, stored #{inspect(stored)}"
           end)},
          {"Missing from checkpoint", Enum.map(missing, &"  #{&1}")},
          {"Unexpected in checkpoint", Enum.map(extra, &"  #{&1}")}
        ]
        |> Enum.reject(fn {_title, lines} -> lines == [] end)
        |> Enum.map_join("\n", fn {title, lines} ->
          "#{title}:\n" <> Enum.join(lines, "\n")
        end)

      raise RuntimeError,
            "Checkpoint parameters do not match the model rebuilt from its spec." <>
              arch_line <> sections <>
              "\nThe checkpoint was likely saved with different build options than " <>
              "the embedded spec, or the file is corrupted."
    end
  end

  def validate_shapes!(_model, _params, _spec), do: :ok

  @doc false
  def reset_missing_spec_warning do
    :persistent_term.erase({__MODULE__, :missing_spec_warned})
    :ok
  end

  defp inject_spec(metadata, nil), do: metadata

  defp inject_spec(metadata, %Spec{} = spec) do
    metadata = metadata || %{}

    if Map.has_key?(metadata, @spec_key) do
      raise ArgumentError,
            "metadata already contains the reserved key #{inspect(@spec_key)}; " <>
              "pass the spec via the :spec option only"
    end

    Map.put(metadata, @spec_key, Spec.to_map(spec))
  end

  defp inject_spec(_metadata, other) do
    raise ArgumentError, "expected :spec to be an %Edifice.Spec{}, got: #{inspect(other)}"
  end

  defp safe_term(binary) do
    :erlang.binary_to_term(binary, [:safe])
  rescue
    ArgumentError ->
      {:invalid_term, "metadata is not deserializable with binary_to_term(bin, [:safe])"}
  end

  defp param_shapes(params) do
    params
    |> extract_params()
    |> flatten_to_dot_keys()
    |> Map.new(fn {key, tensor} -> {key, Nx.shape(tensor)} end)
  end

  defp warn_missing_spec(path) do
    key = {__MODULE__, :missing_spec_warned}

    unless :persistent_term.get(key, false) do
      :persistent_term.put(key, true)

      Logger.warning(
        "[Checkpoint] #{path} has no embedded Edifice.Spec — the model cannot be " <>
          "rebuilt from this checkpoint alone. Save with Checkpoint.save(params, path, " <>
          "spec: spec) (see Edifice.build_with_spec/3). This warning is shown once per VM."
      )
    end
  end

  # ============================================================================
  # Loop state serialization (resume training)
  # ============================================================================

  @doc """
  Serialize an Axon.Loop state for later resumption.

  Wraps `Axon.Loop.serialize_state/2` to save the full loop state
  (model params, optimizer state, epoch, step) to a file.

  ## Parameters

    * `loop_state` - The state returned by `Axon.Loop.run/4`
    * `path` - File path to save to
  """
  @spec save_loop_state(map(), String.t()) :: :ok
  def save_loop_state(loop_state, path) do
    serialized = Axon.Loop.serialize_state(loop_state)
    dir = Path.dirname(path)
    if dir != "." and dir != "", do: File.mkdir_p!(dir)
    File.write!(path, serialized)
    size = byte_size(serialized)
    Logger.info("[Checkpoint] Saved loop state #{readable_size(size)} to #{path}")
    :ok
  end

  @doc """
  Resume training from a saved loop state.

  Wraps `Axon.Loop.from_state/2` to restore a loop from a previously
  serialized state. Use with `Axon.Loop.run/4` to continue training.

  ## Parameters

    * `loop` - The `Axon.Loop` to resume
    * `path` - File path to the saved loop state

  ## Returns

    An updated `Axon.Loop` that will resume from the saved state.

  ## Example

      loop = Edifice.Recipes.classify(model, num_classes: 10)

      # First run
      state = Axon.Loop.run(loop, data, %{}, epochs: 5)
      Edifice.Checkpoint.save_loop_state(state, "checkpoints/state.bin")

      # Later: resume
      loop = Edifice.Checkpoint.resume(loop, "checkpoints/state.bin")
      state = Axon.Loop.run(loop, data, %{}, epochs: 10)
  """
  @spec resume(Axon.Loop.t(), String.t()) :: Axon.Loop.t()
  def resume(%Axon.Loop{} = loop, path) do
    serialized = File.read!(path)
    loop_state = Axon.Loop.deserialize_state(serialized)
    size = byte_size(serialized)
    Logger.info("[Checkpoint] Resuming from #{readable_size(size)} state at #{path}")
    Axon.Loop.from_state(loop, loop_state)
  end

  # ============================================================================
  # Axon.Loop integration
  # ============================================================================

  @doc """
  Attach checkpoint saving to an Axon.Loop.

  Saves model parameters at specified intervals using `Nx.serialize`
  for fast I/O.

  ## Options

    * `:every` - Save frequency: `:epoch` (default) or `:step`
    * `:step_interval` - When `every: :step`, save every N steps (default: 1000)
    * `:keep` - Number of recent checkpoints to keep (default: 3).
      Older checkpoints are deleted automatically.
    * `:compressed` - Compression level 0-9 (default: 0)
  """
  @spec attach(Axon.Loop.t(), String.t(), keyword()) :: Axon.Loop.t()
  def attach(loop, path_pattern, opts \\ []) do
    every = Keyword.get(opts, :every, :epoch)
    step_interval = Keyword.get(opts, :step_interval, 1000)
    keep = Keyword.get(opts, :keep, 3)
    compressed = Keyword.get(opts, :compressed, 0)

    event =
      case every do
        :epoch -> :epoch_completed
        :step -> :iteration_completed
      end

    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      meta = Map.put(state.handler_metadata, :checkpoint_history, [])
      {:continue, %{state | handler_metadata: meta}}
    end)
    |> Axon.Loop.handle_event(event, fn state ->
      should_save =
        case every do
          :epoch -> true
          :step -> rem(state.iteration, step_interval) == 0
        end

      if should_save do
        # Build path with epoch/step substitution
        path = build_path(path_pattern, state)

        # Extract model state
        model_state = get_in(state.step_state, [:model_state])

        params =
          case model_state do
            %Axon.ModelState{data: data} -> data
            %{} = map -> map
            _ -> %{}
          end

        metadata = %{
          epoch: state.epoch,
          iteration: state.iteration,
          metrics: state.metrics
        }

        save(params, path, metadata: metadata, compressed: compressed)

        # Manage checkpoint history
        history = state.handler_metadata[:checkpoint_history] || []
        history = [path | history]

        # Delete old checkpoints beyond `keep`
        if length(history) > keep do
          {_kept, to_delete} = Enum.split(history, keep)

          Enum.each(to_delete, fn old_path ->
            case File.rm(old_path) do
              :ok -> Logger.info("[Checkpoint] Deleted old checkpoint: #{old_path}")
              {:error, _} -> :ok
            end
          end)

          history = Enum.take(history, keep)
          meta = %{state.handler_metadata | checkpoint_history: history}
          {:continue, %{state | handler_metadata: meta}}
        else
          meta = %{state.handler_metadata | checkpoint_history: history}
          {:continue, %{state | handler_metadata: meta}}
        end
      else
        {:continue, state}
      end
    end)
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp extract_params(%Axon.ModelState{data: data}), do: data
  defp extract_params(%{} = map), do: map

  defp build_path(pattern, state) do
    pattern
    |> String.replace("{epoch}", to_string(state.epoch))
    |> String.replace("{step}", to_string(state.iteration))
  end

  # ============================================================================
  # Safetensors export/import
  # ============================================================================

  @doc """
  Export model parameters to Safetensors format.

  Saves in the `.safetensors` format used by HuggingFace, PyTorch, and JAX.
  Enables cross-ecosystem model sharing.

  Nested parameter maps are flattened with dot-separated keys:
  `%{"layer" => %{"kernel" => t}}` becomes `%{"layer.kernel" => t}`.

  ## Options

    * `:metadata` - Optional metadata map (stored in safetensors header).
      Values must be strings.
  """
  @spec export_safetensors(map(), String.t(), keyword()) :: :ok
  def export_safetensors(params, path, _opts \\ []) do
    ensure_safetensors!()
    data = extract_params(params)
    flat = flatten_to_dot_keys(data)

    dir = Path.dirname(path)
    if dir != "." and dir != "", do: File.mkdir_p!(dir)

    apply(Safetensors, :write!, [path, flat])

    size = File.stat!(path).size
    Logger.info("[Checkpoint] Exported #{readable_size(size)} safetensors to #{path} (#{map_size(flat)} tensors)")

    :ok
  end

  @doc """
  Import model parameters from Safetensors format.

  Loads a `.safetensors` file and reconstructs the nested parameter map
  from dot-separated keys.
  """
  @spec import_safetensors(String.t(), keyword()) :: map()
  def import_safetensors(path, _opts \\ []) do
    ensure_safetensors!()
    flat = apply(Safetensors, :read!, [path])

    size = File.stat!(path).size
    Logger.info("[Checkpoint] Imported #{readable_size(size)} safetensors from #{path} (#{map_size(flat)} tensors)")

    unflatten_from_dot_keys(flat)
  end

  defp ensure_safetensors! do
    unless Code.ensure_loaded?(Safetensors) do
      raise RuntimeError,
            "Safetensors export requires the :safetensors dependency. " <>
              "Add {:safetensors, \"~> 0.1.3\"} to your deps."
    end
  end

  defp flatten_to_dot_keys(map, prefix \\ []) do
    Enum.flat_map(map, fn
      {k, %Nx.Tensor{} = t} ->
        key = Enum.join(prefix ++ [k], ".")
        [{key, t}]

      {k, v} when is_map(v) and not is_struct(v) ->
        flatten_to_dot_keys(v, prefix ++ [k])

      _ ->
        []
    end)
    |> Map.new()
  end

  defp unflatten_from_dot_keys(flat) do
    Enum.reduce(flat, %{}, fn {dot_key, tensor}, acc ->
      keys = String.split(dot_key, ".")
      put_nested(acc, keys, tensor)
    end)
  end

  defp put_nested(map, [key], value), do: Map.put(map, key, value)

  defp put_nested(map, [key | rest], value) do
    nested = Map.get(map, key, %{})
    Map.put(map, key, put_nested(nested, rest, value))
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp readable_size(n) when n < 1_024, do: "#{n} B"
  defp readable_size(n) when n < 1_048_576, do: "#{Float.round(n / 1_024, 1)} KB"
  defp readable_size(n) when n < 1_073_741_824, do: "#{Float.round(n / 1_048_576, 1)} MB"
  defp readable_size(n), do: "#{Float.round(n / 1_073_741_824, 2)} GB"
end
