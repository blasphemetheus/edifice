defmodule Edifice.Blocks.HeadNorm do
  @moduledoc """
  Per-token group normalization for multi-head sequence representations.

  Each head is normalized over its own channels, independently of every
  batch row and timestep. Unlike spatial GroupNorm on `{batch, time, channels}`,
  this preserves causality. Affine parameters retain shape `{channels}`.
  """

  def layer(input, num_heads, opts \\ []) do
    name = Keyword.fetch!(opts, :name)

    gamma =
      Axon.param("gamma", fn shape -> {elem(shape, tuple_size(shape) - 1)} end,
        initializer: :ones
      )

    beta =
      Axon.param("beta", fn shape -> {elem(shape, tuple_size(shape) - 1)} end,
        initializer: :zeros
      )

    Axon.layer(&normalize/4, [input, gamma, beta],
      name: name,
      num_heads: num_heads,
      epsilon: Keyword.get(opts, :epsilon, 1.0e-5),
      op_name: :head_norm
    )
  end

  defp normalize(input, gamma, beta, opts) do
    shape = Nx.shape(input)
    channels = elem(shape, tuple_size(shape) - 1)
    heads = opts[:num_heads]

    if heads < 1 or rem(channels, heads) != 0 do
      raise ArgumentError, "channels must be divisible by a positive num_heads"
    end

    grouped =
      Nx.reshape(
        input,
        List.to_tuple(Enum.drop(Tuple.to_list(shape), -1) ++ [heads, div(channels, heads)])
      )

    mean = Nx.mean(grouped, axes: [-1], keep_axes: true)
    centered = Nx.subtract(grouped, mean)
    variance = Nx.mean(Nx.multiply(centered, centered), axes: [-1], keep_axes: true)

    centered
    |> Nx.multiply(Nx.rsqrt(Nx.add(variance, opts[:epsilon])))
    |> Nx.reshape(shape)
    |> Nx.multiply(gamma)
    |> Nx.add(beta)
  end
end
