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

  alias Edifice.Pretrained.Transform

  @type load_opt ::
          {:dtype, atom()}
          | {:strict, boolean()}

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
  @spec load(module(), Path.t(), [load_opt()]) :: Axon.ModelState.t()
  def load(key_map, path, opts \\ []) do
    ensure_safetensors!()

    dtype = Keyword.get(opts, :dtype)
    strict = Keyword.get(opts, :strict, true)
    transforms = key_map.tensor_transforms()

    checkpoint = Safetensors.read!(path)

    {mapped, unmapped} =
      Enum.reduce(checkpoint, {%{}, []}, fn {ext_key, tensor}, {mapped_acc, unmapped_acc} ->
        case key_map.map_key(ext_key) do
          :skip ->
            {mapped_acc, unmapped_acc}

          mapped_key when is_binary(mapped_key) ->
            tensor = Transform.apply_transform(mapped_key, transforms, tensor)
            tensor = if dtype, do: Transform.cast(tensor, dtype), else: tensor
            {Map.put(mapped_acc, mapped_key, tensor), unmapped_acc}

          :unmapped ->
            {mapped_acc, [ext_key | unmapped_acc]}
        end
      end)

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
    |> Safetensors.read!()
    |> Map.keys()
    |> Enum.sort()
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
