defmodule Edifice.TestHelpers do
  @moduledoc """
  Shared test utilities for Edifice architecture tests.
  """

  @doc """
  Assert that a tensor contains no NaN or Inf values.
  Returns the tensor for pipeline use.
  """
  def assert_finite!(output, label \\ "output")

  def assert_finite!(%Nx.Tensor{} = tensor, label) do
    has_nan = Nx.any(Nx.is_nan(tensor)) |> Nx.to_number()
    has_inf = Nx.any(Nx.is_infinity(tensor)) |> Nx.to_number()

    if has_nan == 1, do: raise("#{label} contains NaN values")
    if has_inf == 1, do: raise("#{label} contains Inf values")

    tensor
  end

  # Handle Axon.container outputs (maps of tensors, e.g. VAE encoder %{mu:, log_var:})
  def assert_finite!(container, label) when is_map(container) do
    Enum.each(container, fn {key, value} ->
      assert_finite!(value, "#{label}.#{key}")
    end)

    container
  end

  @doc """
  Build a model, init params, and return {init_fn, predict_fn, params} for a given
  architecture spec. Handles both single Axon models and tuple returns (generative).

  Returns `{predict_fn, params, input}` for the primary model component.
  """
  def build_and_init(model, input_map, mode \\ :inference) do
    {init_fn, predict_fn} = Axon.build(model, mode: mode)

    # Build template from input map
    template =
      Map.new(input_map, fn {name, tensor} ->
        {name, Nx.template(Nx.shape(tensor), Nx.type(tensor))}
      end)

    params = init_fn.(template, Axon.ModelState.empty())
    {predict_fn, params}
  end

  @doc """
  Generate random input tensor with a fixed seed.
  """
  def random_tensor(shape, seed \\ 42) do
    key = Nx.Random.key(seed)
    {tensor, _key} = Nx.Random.uniform(key, shape: shape, type: :f32)
    tensor
  end

  @doc """
  Check that at least some values in a nested param map are non-zero.
  """
  def any_nonzero_params?(params) do
    params
    |> flatten_params()
    |> Enum.any?(fn {_key, tensor} ->
      Nx.any(Nx.not_equal(tensor, 0)) |> Nx.to_number() == 1
    end)
  end

  @doc """
  Flatten a possibly-nested param map into a flat list of {path, tensor} pairs.
  """
  def flatten_params(%Axon.ModelState{data: data}), do: flatten_params(data)

  def flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} ->
        [{key, tensor}]

      {key, inner} when is_map(inner) ->
        flatten_params(inner) |> Enum.map(fn {k, v} -> {"#{key}.#{k}", v} end)

      _ ->
        []
    end)
  end

  def flatten_params(_), do: []
end
