defmodule Edifice.Pretrained.Transform do
  @moduledoc """
  Tensor transform utilities for pretrained weight loading.

  Provides common transformations needed when converting weights between
  frameworks (e.g., PyTorch → Axon). Also includes helpers for converting
  between flat dot-separated key maps and nested maps suitable for
  `Axon.ModelState`.

  ## Key Functions

    - `transpose_linear/1` — Transposes 2D weight matrices from PyTorch's
      `[out_features, in_features]` to Axon's `[in_features, out_features]`.
    - `permute_conv2d/1` — Identity for now (both use OIHW), hook for future formats.
    - `cast/2` — Converts tensor dtype (e.g., fp16 → f32).
    - `nest_params/1` — Converts flat `"a.b.c"` keys to nested maps.
    - `flatten_params/1` — Inverse of `nest_params/1`.
    - `apply_transform/3` — Matches a key against regex patterns and applies
      the first matching transform.

  """

  @doc """
  Transposes a 2D weight matrix from `[out, in]` to `[in, out]`.

  PyTorch stores dense/linear layer weights as `[out_features, in_features]`,
  while Axon expects `[in_features, out_features]`. This function handles
  that conversion.

  Only transposes rank-2 tensors; higher-rank tensors are returned unchanged.

  ## Examples

      iex> t = Nx.iota({3, 2})
      iex> Edifice.Pretrained.Transform.transpose_linear(t)
      #Nx.Tensor<
        s32[2][3]
        ...
      >

  """
  @spec transpose_linear(Nx.Tensor.t()) :: Nx.Tensor.t()
  def transpose_linear(tensor) do
    case Nx.rank(tensor) do
      2 -> Nx.transpose(tensor)
      _ -> tensor
    end
  end

  @doc """
  Permutes a conv2d weight tensor between layout conventions.

  Currently a no-op since both PyTorch and Axon use OIHW layout.
  Provided as a hook for future format support (e.g., TensorFlow's HWIO).
  """
  @spec permute_conv2d(Nx.Tensor.t()) :: Nx.Tensor.t()
  def permute_conv2d(tensor), do: tensor

  @doc """
  Casts a tensor to the given dtype.

  Useful for converting fp16/bf16 checkpoint weights to f32 for inference
  on backends that don't support half precision.

  ## Examples

      iex> t = Nx.tensor([1.0, 2.0], type: :f32)
      iex> Edifice.Pretrained.Transform.cast(t, :bf16) |> Nx.type()
      {:bf, 16}

  """
  @spec cast(Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def cast(tensor, dtype) do
    Nx.as_type(tensor, dtype)
  end

  @doc """
  Converts a flat map with dot-separated keys into a nested map.

  ## Examples

      iex> flat = %{"a.b.c" => 1, "a.b.d" => 2, "a.e" => 3}
      iex> Edifice.Pretrained.Transform.nest_params(flat)
      %{"a" => %{"b" => %{"c" => 1, "d" => 2}, "e" => 3}}

  """
  @spec nest_params(%{String.t() => term()}) :: map()
  def nest_params(flat_map) when is_map(flat_map) do
    Enum.reduce(flat_map, %{}, fn {key, value}, acc ->
      parts = String.split(key, ".")
      deep_put(acc, parts, value)
    end)
  end

  defp deep_put(map, [key], value) do
    Map.put(map, key, value)
  end

  defp deep_put(map, [key | rest], value) do
    inner = Map.get(map, key, %{})
    Map.put(map, key, deep_put(inner, rest, value))
  end

  @doc """
  Converts a nested map into a flat map with dot-separated keys.

  Inverse of `nest_params/1`. Leaf values are `Nx.Tensor` structs or any
  non-map term.

  ## Examples

      iex> nested = %{"a" => %{"b" => 1, "c" => 2}}
      iex> Edifice.Pretrained.Transform.flatten_params(nested) |> Enum.sort()
      [{"a.b", 1}, {"a.c", 2}]

  """
  @spec flatten_params(map()) :: %{String.t() => term()}
  def flatten_params(%Axon.ModelState{data: data}), do: flatten_params(data)

  def flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} ->
        [{to_string(key), tensor}]

      {key, inner} when is_map(inner) ->
        inner
        |> flatten_params()
        |> Enum.map(fn {nested_key, value} ->
          {"#{key}.#{nested_key}", value}
        end)

      {key, value} ->
        [{to_string(key), value}]
    end)
    |> Map.new()
  end

  @doc """
  Applies the first matching transform from a list of `{regex, transform_fn}` pairs.

  Iterates through the patterns and applies the transform function from the
  first regex that matches the given key. If no pattern matches, returns the
  tensor unchanged.

  ## Examples

      iex> patterns = [{~r/\\.kernel$/, &Nx.transpose/1}]
      iex> t = Nx.iota({3, 2})
      iex> Edifice.Pretrained.Transform.apply_transform("dense.kernel", patterns, t)
      #Nx.Tensor<
        s32[2][3]
        ...
      >

  """
  @spec apply_transform(
          String.t(),
          [{Regex.t(), (Nx.Tensor.t() -> Nx.Tensor.t())}],
          Nx.Tensor.t()
        ) ::
          Nx.Tensor.t()
  def apply_transform(key, patterns, tensor) do
    case Enum.find(patterns, fn {regex, _fn} -> Regex.match?(regex, key) end) do
      {_regex, transform_fn} -> transform_fn.(tensor)
      nil -> tensor
    end
  end
end
