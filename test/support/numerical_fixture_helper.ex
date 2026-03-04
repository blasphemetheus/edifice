defmodule Edifice.NumericalFixtureHelper do
  @moduledoc """
  Shared helper for random-weight numerical validation tests.

  Loads PyTorch-generated SafeTensors fixtures and maps weights into
  Axon models using per-architecture key mappings.
  """

  @fixtures_dir Path.join([__DIR__, "..", "fixtures", "numerical"])

  @doc """
  Load a SafeTensors fixture file from the fixtures directory.
  Returns a map of tensor name -> Nx.Tensor.
  """
  def load_fixture(name) do
    path = Path.join(@fixtures_dir, name)

    unless File.exists?(path) do
      raise "Fixture #{name} not found at #{path}. " <>
              "Run: python scripts/generate_random_weight_fixtures.py"
    end

    Safetensors.read!(path)
  end

  @doc """
  Build Axon ModelState from a fixture map and key mapping.

  First initializes the model with default params to get the correct ModelState
  structure, then replaces param tensors with values from the fixture.

  ## Arguments
    - `fixture` - Map of tensor names from SafeTensors
    - `key_mapping` - List of `{pytorch_key, axon_path, transform}` tuples
    - `model` - Axon model
    - `template` - Input template map (e.g. `%{"state_sequence" => Nx.template({2, 8, 32}, :f32)}`)

  ## Transforms
    - `:identity` - Use tensor as-is
    - `:transpose_2d` - Transpose [out, in] -> [in, out] (for nn.Linear weights)
    - `{:sum, key2}` - Sum with another fixture key (for LSTM bias_ih + bias_hh)
  """
  def build_params_from_fixture(fixture, key_mapping, model, template) do
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)
    model_state = init_fn.(template, Axon.ModelState.empty())

    data =
      Enum.reduce(key_mapping, model_state.data, fn {pytorch_key, axon_path, transform}, acc ->
        tensor = Map.fetch!(fixture, "weight.#{pytorch_key}")
        transformed = apply_transform(tensor, transform, fixture)
        put_nested(acc, axon_path, transformed)
      end)

    {%{model_state | data: data}, predict_fn}
  end

  defp apply_transform(tensor, :identity, _fixture), do: tensor
  defp apply_transform(tensor, :transpose_2d, _fixture), do: Nx.transpose(tensor)
  defp apply_transform(tensor, :transpose_conv, _fixture), do: Nx.transpose(tensor, axes: [2, 1, 0])

  defp apply_transform(tensor, {:sum, key2}, fixture) do
    tensor2 = Map.fetch!(fixture, "weight.#{key2}")
    Nx.add(tensor, tensor2)
  end

  defp put_nested(map, [key], value), do: Map.put(map, key, value)

  defp put_nested(map, [key | rest], value) do
    inner = Map.get(map, key, %{})
    Map.put(map, key, put_nested(inner, rest, value))
  end

  @doc """
  Assert two tensors are close within tolerance.
  """
  def assert_all_close(actual, expected, opts \\ []) do
    import ExUnit.Assertions

    atol = Keyword.get(opts, :atol, 1.0e-4)

    assert Nx.shape(actual) == Nx.shape(expected),
           "Shape mismatch: #{inspect(Nx.shape(actual))} vs #{inspect(Nx.shape(expected))}"

    diff = actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Max absolute difference #{diff} exceeds tolerance #{atol}. " <>
             "Mean diff: #{actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.mean() |> Nx.to_number()}"
  end

  @doc """
  Dump all param keys from an Axon ModelState for debugging key mappings.
  Returns a sorted list of dotted paths like ["encoder.kernel", "encoder.bias", ...].
  """
  def dump_param_keys(model_state) do
    model_state
    |> flatten_params()
    |> Enum.map(fn {path, tensor} -> "#{path} #{inspect(Nx.shape(tensor))}" end)
    |> Enum.sort()
  end

  defp flatten_params(%Axon.ModelState{data: data}), do: flatten_params(data, [])
  defp flatten_params(map, prefix) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} ->
        [{Enum.join(prefix ++ [key], "."), tensor}]

      {key, inner} when is_map(inner) ->
        flatten_params(inner, prefix ++ [key])

      _ ->
        []
    end)
  end
end
