defmodule Edifice.Pretrained do
  @moduledoc """
  Load pretrained weights from SafeTensors checkpoints into Axon models.

  This module provides the main entry point for loading pretrained weights.
  It reads a SafeTensors file, maps checkpoint keys to Axon parameter paths
  using a key map module, applies tensor transforms, and returns an
  `Axon.ModelState` ready for use with `Axon.predict/3`.

  ## Prerequisites

  Requires the optional `safetensors` dependency:

      {:safetensors, "~> 0.1.3"}

  ## Usage

      # Define or use an existing key map
      model_state = Edifice.Pretrained.load(MyApp.KeyMaps.ViT, "model.safetensors")

      # Sharded models (multiple shard files)
      model_state = Edifice.Pretrained.load_sharded(MyKeyMap, ["shard-1.safetensors", "shard-2.safetensors"])

      # Use with an Axon model
      result = Axon.predict(model, model_state, input)

  ## Options

    - `:dtype` — Cast all tensors to this type (e.g., `:f32`, `:bf16`).
      Default: `nil` (keep original dtype).
    - `:strict` — When `true` (default), raises if any checkpoint key is not
      handled by the key map. When `false`, unmapped keys are logged and skipped.

  ## Inspecting Checkpoints

  Use `list_keys/1` to see what parameter names a checkpoint contains:

      Edifice.Pretrained.list_keys("model.safetensors")
      #=> ["attention.key.weight", "attention.query.weight", ...]

  """

  require Logger

  alias Edifice.Pretrained.{Config, Hub, Transform}

  @type load_opt ::
          {:dtype, atom()}
          | {:strict, boolean()}

  @type hub_opt ::
          {:revision, String.t()}
          | {:cache_dir, Path.t()}
          | {:force, boolean()}
          | {:token, String.t()}
          | {:dtype, atom()}
          | {:strict, boolean()}
          | {:build_opts, keyword()}

  # Axon defines no @type t for these structs.
  @typep model_state :: %Axon.ModelState{}

  @doc """
  Downloads and loads a pretrained model from HuggingFace Hub.

  Fetches `config.json` to auto-detect the architecture, downloads the
  SafeTensors weights, builds the Edifice model, and loads the weights.

  Returns `{model, model_state}` for most architectures. For Whisper
  (which has separate encoder and decoder), returns
  `{encoder, decoder, model_state}`.

  ## Options

    - `:revision` — Git revision (branch, tag, commit). Default: `"main"`.
    - `:cache_dir` — Override cache directory. Default: `~/.cache/edifice`.
    - `:force` — Re-download even if cached. Default: `false`.
    - `:token` — HuggingFace API token for private/gated models.
    - `:dtype` — Cast all tensors to this type (e.g., `:f32`, `:bf16`).
    - `:strict` — When `true` (default), raises on unmapped checkpoint keys.
    - `:build_opts` — Additional build opts to merge (overrides config-derived opts).

  ## Examples

      # ViT — returns {model, model_state}
      {model, model_state} = Edifice.Pretrained.from_hub("google/vit-base-patch16-224")
      Axon.predict(model, model_state, input)

      # Whisper — returns {encoder, decoder, model_state}
      {encoder, decoder, model_state} = Edifice.Pretrained.from_hub("openai/whisper-base")

  """
  @spec from_hub(String.t(), [hub_opt()]) ::
          {Axon.t(), model_state()} | {Axon.t(), Axon.t(), model_state()}
  def from_hub(repo_id, opts \\ []) do
    # 1. Fetch and parse config.json
    hub_opts = Keyword.take(opts, [:revision, :token])
    config_json = Hub.fetch_config!(repo_id, hub_opts)
    parsed = Config.parse!(config_json)

    # 2. Download weights
    download_opts = Keyword.take(opts, [:revision, :cache_dir, :force, :token])
    paths = Hub.download!(repo_id, download_opts)

    # 3. Build model with config-derived opts, allowing user overrides
    build_opts = Keyword.merge(parsed.build_opts, Keyword.get(opts, :build_opts, []))

    # 4. Load weights
    load_opts = Keyword.take(opts, [:dtype, :strict])
    model_state = load_or_load_sharded(parsed.key_map, paths, load_opts)

    # 5. Build and return appropriate tuple
    case parsed.model_type do
      "whisper" ->
        encoder = Edifice.Audio.Whisper.build_encoder(build_opts)
        decoder = Edifice.Audio.Whisper.build_decoder(build_opts)
        {encoder, decoder, model_state}

      _other ->
        model = parsed.build_fn.(build_opts)
        {model, model_state}
    end
  end

  @doc """
  Loads pretrained weights from a SafeTensors file using the given key map.

  The key map module must implement `Edifice.Pretrained.KeyMap`. Each key in
  the checkpoint is passed through `key_map.map_key/1` to get the Axon path,
  then tensor transforms from `key_map.tensor_transforms/0` are applied.

  Returns an `Axon.ModelState` struct.

  ## Examples

      model_state = Edifice.Pretrained.load(MyKeyMap, "model.safetensors")
      model_state = Edifice.Pretrained.load(MyKeyMap, "model.safetensors", dtype: :f32)

  """
  @spec load(module(), Path.t(), [load_opt()]) :: model_state()
  def load(key_map, path, opts \\ []) do
    ensure_safetensors!()

    checkpoint = apply(Safetensors, :read!, [path])
    do_load(key_map, checkpoint, opts)
  end

  @doc """
  Loads pretrained weights from multiple sharded SafeTensors files.

  Merges all shard files into a single checkpoint map, then processes
  keys using the given key map — same as `load/3` but for sharded models.

  ## Examples

      paths = Edifice.Pretrained.Hub.download!("bigscience/bloom")
      model_state = Edifice.Pretrained.load_sharded(MyKeyMap, paths)

  """
  @spec load_sharded(module(), [Path.t()], [load_opt()]) :: model_state()
  def load_sharded(key_map, paths, opts \\ []) when is_list(paths) do
    ensure_safetensors!()

    checkpoint =
      Enum.reduce(paths, %{}, fn path, acc ->
        Map.merge(acc, apply(Safetensors, :read!, [path]))
      end)

    do_load(key_map, checkpoint, opts)
  end

  defp do_load(key_map, checkpoint, opts) do
    dtype = Keyword.get(opts, :dtype)
    strict = Keyword.get(opts, :strict, true)
    transforms = key_map.tensor_transforms()

    # Build concat lookup: source_key -> {target_key, axis, [source_keys]}
    {concat_source_lookup, concat_targets} = build_concat_lookup(key_map)

    {mapped, unmapped, concat_acc} =
      Enum.reduce(checkpoint, {%{}, [], %{}}, fn {ext_key, tensor},
                                                 {mapped_acc, unmapped_acc, concat_acc} ->
        case key_map.map_key(ext_key) do
          :skip ->
            {mapped_acc, unmapped_acc, concat_acc}

          mapped_key when is_binary(mapped_key) ->
            tensor = Transform.apply_transform(mapped_key, transforms, tensor)
            tensor = if dtype, do: Transform.cast(tensor, dtype), else: tensor

            case Map.get(concat_source_lookup, mapped_key) do
              nil ->
                {Map.put(mapped_acc, mapped_key, tensor), unmapped_acc, concat_acc}

              _target_info ->
                concat_acc = Map.put(concat_acc, mapped_key, tensor)
                {mapped_acc, unmapped_acc, concat_acc}
            end

          :unmapped ->
            {mapped_acc, [ext_key | unmapped_acc], concat_acc}
        end
      end)

    # Resolve concat groups: concatenate accumulated tensors for each target
    mapped = resolve_concat_groups(mapped, concat_acc, concat_targets)

    if strict and unmapped != [] do
      raise ArgumentError,
            "Strict loading failed: #{length(unmapped)} unmapped key(s) in checkpoint: " <>
              Enum.join(Enum.take(Enum.reverse(unmapped), 10), ", ")
    end

    for key <- unmapped do
      Logger.warning("Pretrained: unmapped checkpoint key #{inspect(key)}, skipping")
    end

    mapped
    |> Transform.nest_params()
    |> Axon.ModelState.new()
  end

  @doc """
  Returns a sorted list of parameter key names from a SafeTensors file.

  Useful for inspecting checkpoint contents when developing a key map.
  Does not load tensor data into memory beyond what SafeTensors requires.

  ## Examples

      Edifice.Pretrained.list_keys("model.safetensors")
      #=> ["attention.key.bias", "attention.key.weight", ...]

  """
  @spec list_keys(Path.t()) :: [String.t()]
  def list_keys(path) do
    ensure_safetensors!()

    path
    |> then(&apply(Safetensors, :read!, [&1]))
    |> Map.keys()
    |> Enum.sort()
  end

  # Builds a lookup from source keys to their concat target info.
  # Returns {source_lookup, targets} where:
  #   source_lookup: %{source_key => {target_key, axis, [source_keys]}}
  #   targets: %{target_key => {[source_keys], axis}}
  defp build_concat_lookup(key_map) do
    if function_exported?(key_map, :concat_keys, 0) do
      targets = key_map.concat_keys()

      source_lookup =
        Enum.flat_map(targets, fn {target_key, {source_keys, axis}} ->
          Enum.map(source_keys, fn src -> {src, {target_key, axis, source_keys}} end)
        end)
        |> Map.new()

      {source_lookup, targets}
    else
      {%{}, %{}}
    end
  end

  # Concatenates accumulated partial tensors for each concat target.
  # Skips targets where not all source keys are present in the accumulator.
  defp resolve_concat_groups(mapped, concat_acc, concat_targets) do
    Enum.reduce(concat_targets, mapped, fn {target_key, {source_keys, axis}}, acc ->
      if Enum.all?(source_keys, &Map.has_key?(concat_acc, &1)) do
        tensors = Enum.map(source_keys, &Map.fetch!(concat_acc, &1))
        Map.put(acc, target_key, Nx.concatenate(tensors, axis: axis))
      else
        acc
      end
    end)
  end

  defp load_or_load_sharded(key_map, paths, opts) do
    case paths do
      [single] -> load(key_map, single, opts)
      multiple -> load_sharded(key_map, multiple, opts)
    end
  end

  defp ensure_safetensors! do
    unless Code.ensure_loaded?(Safetensors) do
      raise RuntimeError, """
      The :safetensors package is required for pretrained weight loading.

      Add it to your mix.exs dependencies:

          {:safetensors, "~> 0.1.3"}

      Then run: mix deps.get
      """
    end
  end
end
