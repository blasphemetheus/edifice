defmodule Edifice.Utils.SamplingRandom do
  @moduledoc """
  Statically unrolled Threefry2x32-20 for small inference sampling tensors.

  `uniform/2` matches Nx.Random's f32 `[0, 1)` values and returned key, including
  its split order and odd-sized counter padding. The fixed twenty rounds are
  expanded during tracing so GPU inference does not launch nested RNG loops.
  This specializes the default uniform draw; use `Nx.Random` for other types,
  intervals, or distributions. Compile-time work is larger than the rolled RNG.

  Algorithm: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011).
  Stream layout follows `Nx.Random.uniform/2`.
  """
  import Nx.Defn

  @doc "Draw f32 uniforms of `:shape` (default `{}`) and return the next u32 key."
  defn uniform(key, opts \\ []) do
    opts = keyword!(opts, shape: {})
    validate_key!(key)
    keys = bits(key, {2, 2})
    words = bits(keys[1], opts[:shape])

    values =
      words
      |> Nx.right_shift(9)
      |> Nx.bitwise_or(Nx.tensor(0x3F800000, type: :u32))
      |> Nx.bitcast(:f32)
      |> Nx.subtract(1.0)
      |> stop_grad()

    {values, stop_grad(keys[0])}
  end

  deftransformp validate_key!(key) do
    unless Nx.shape(key) == {2} and Nx.type(key) == {:u, 32},
      do: raise(ArgumentError, "expected a u32 PRNG key of shape {2}")
  end

  deftransformp bits(key, shape) do
    size = Nx.size(shape)
    padded = size + rem(size, 2)
    counters = Nx.iota({padded}, type: :u32)

    counters =
      if padded == size,
        do: counters,
        else: Nx.put_slice(counters, [size], Nx.tensor([0], type: :u32))

    counters = Nx.reshape(counters, {2, div(padded, 2)})
    k0 = key[0]
    k1 = key[1]
    parity = Nx.bitwise_xor(Nx.bitwise_xor(k0, k1), Nx.tensor(0x1BD11BDA, type: :u32))
    schedule = {k0, k1, parity}
    rotations = {13, 15, 26, 6, 17, 29, 16, 24}

    {a, b} =
      Enum.reduce(0..19, {Nx.add(counters[0], k0), Nx.add(counters[1], k1)}, fn round, {a, b} ->
        rotation = elem(rotations, rem(round, 8))
        a = Nx.add(a, b)
        rotated = Nx.bitwise_or(Nx.left_shift(b, rotation), Nx.right_shift(b, 32 - rotation))
        b = Nx.bitwise_xor(rotated, a)

        if rem(round + 1, 4) == 0 do
          injection = div(round + 1, 4)
          a = Nx.add(a, elem(schedule, rem(injection, 3)))

          b =
            Nx.add(
              Nx.add(b, elem(schedule, rem(injection + 1, 3))),
              Nx.tensor(injection, type: :u32)
            )

          {a, b}
        else
          {a, b}
        end
      end)

    Nx.concatenate([a, b]) |> Nx.slice([0], [size]) |> Nx.reshape(shape)
  end
end
