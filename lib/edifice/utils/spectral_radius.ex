defmodule Edifice.Utils.SpectralRadius do
  @moduledoc false

  # Build-time real nonsymmetric eigensolver: orthogonal Hessenberg reduction,
  # then implicit double-shift QR. Deflated 2x2 blocks retain complex pairs.
  # This runs on host floats, independent of Nx's default compiler/backend.
  # No eigenvectors are needed. Private atomics storage avoids copying an
  # entire immutable matrix for every scalar update and is garbage collected.
  def compute(matrix) do
    {n, n} = Nx.shape(matrix)
    values = Nx.to_flat_list(matrix)

    unless Enum.all?(values, &is_number/1),
      do: raise(ArgumentError, "spectral radius requires a finite real matrix")

    scale = Enum.reduce(values, 0.0, &max(abs(&1), &2))

    if scale == 0 or acyclic?(Enum.chunk_every(values, n)) do
      0.0
    else
      storage = :atomics.new(n * n, signed: false)
      a = {storage, n}

      values
      |> Enum.with_index()
      |> Enum.each(fn {value, i} -> put(a, div(i, n), rem(i, n), value / scale) end)

      hessenberg(a, n)
      qr(a, n - 1, 0, 0.0) * scale
    end
  end

  # A directed acyclic support graph is exactly nilpotent. Detect it before
  # orthogonal reduction: roundoff can split repeated zero eigenvalues into
  # tiny nonzero roots, which would otherwise cause enormous rescaling.
  defp acyclic?(rows) do
    edges =
      rows
      |> Enum.map(fn row ->
        row |> Enum.with_index() |> Enum.flat_map(fn {x, j} -> if x == 0, do: [], else: [j] end)
      end)
      |> List.to_tuple()

    degrees =
      rows
      |> Enum.with_index()
      |> Map.new(fn {_, i} -> {i, 0} end)
      |> then(fn initial ->
        Enum.reduce(Tuple.to_list(edges), initial, fn row, acc ->
          Enum.reduce(row, acc, fn j, acc -> Map.update!(acc, j, &(&1 + 1)) end)
        end)
      end)

    ready = for {i, 0} <- degrees, do: i
    remove_sources(ready, degrees, edges, 0) == length(rows)
  end

  defp remove_sources([], _degrees, _edges, count), do: count

  defp remove_sources([i | queue], degrees, edges, count) do
    {degrees, queue} =
      Enum.reduce(elem(edges, i), {degrees, queue}, fn j, {degrees, queue} ->
        remaining = degrees[j] - 1
        {Map.put(degrees, j, remaining), if(remaining == 0, do: [j | queue], else: queue)}
      end)

    remove_sources(queue, degrees, edges, count + 1)
  end

  defp get({storage, n}, i, j) do
    bits = :atomics.get(storage, i * n + j + 1)
    <<value::float-64>> = <<bits::unsigned-64>>
    value
  end

  defp put({storage, n}, i, j, value) do
    <<bits::unsigned-64>> = <<value::float-64>>
    :atomics.put(storage, i * n + j + 1, bits)
  end

  defp hessenberg(a, n) when n > 2 do
    for k <- 0..(n - 3) do
      rows = Enum.to_list((k + 1)..(n - 1))
      v = reflector(Enum.map(rows, &get(a, &1, k)))
      left(a, rows, k..(n - 1), v)
      right(a, 0..(n - 1), rows, v)
      for i <- (k + 2)..(n - 1), do: put(a, i, k, 0.0)
    end
  end

  defp hessenberg(_, _), do: :ok

  defp reflector(values) do
    scale = Enum.reduce(values, 0.0, &max(abs(&1), &2))

    if scale == 0 do
      Enum.map(values, fn _ -> 0.0 end)
    else
      [first | rest] = Enum.map(values, &(&1 / scale))
      norm = :math.sqrt(Enum.reduce([first | rest], 0.0, &(&1 * &1 + &2)))
      vector = [first + if(first >= 0, do: norm, else: -norm) | rest]
      length = :math.sqrt(Enum.reduce(vector, 0.0, &(&1 * &1 + &2)))
      Enum.map(vector, &(&1 / length))
    end
  end

  defp left(a, rows, columns, v) do
    indexed = Enum.zip(rows, v)

    for j <- columns do
      dot = 2 * Enum.reduce(indexed, 0.0, fn {i, x}, acc -> acc + x * get(a, i, j) end)
      for {i, x} <- indexed, do: put(a, i, j, get(a, i, j) - dot * x)
    end
  end

  defp right(a, rows, columns, v) do
    indexed = Enum.zip(columns, v)

    for i <- rows do
      dot = 2 * Enum.reduce(indexed, 0.0, fn {j, x}, acc -> acc + x * get(a, i, j) end)
      for {j, x} <- indexed, do: put(a, i, j, get(a, i, j) - dot * x)
    end
  end

  defp qr(_a, hi, _iterations, radius) when hi < 0, do: radius

  defp qr(a, hi, iterations, radius) do
    lo =
      if hi == 0 do
        0
      else
        Enum.find(hi..1//-1, 0, fn i ->
          abs(get(a, i, i - 1)) <=
            1.0e-13 * (abs(get(a, i - 1, i - 1)) + abs(get(a, i, i)))
        end)
      end

    cond do
      lo == hi ->
        qr(a, hi - 1, 0, max(radius, abs(get(a, hi, hi))))

      lo == hi - 1 ->
        qr(a, hi - 2, 0, max(radius, pair_radius(a, hi)))

      iterations >= 200 ->
        raise ArithmeticError, "spectral-radius QR failed to converge; reservoir was not scaled"

      true ->
        step(a, lo, hi, iterations)
        qr(a, hi, iterations + 1, radius)
    end
  end

  defp pair_radius(a, hi) do
    x = get(a, hi - 1, hi - 1)
    y = get(a, hi, hi)
    product = get(a, hi, hi - 1) * get(a, hi - 1, hi)
    midpoint = (x + y) / 2
    discriminant = (x - y) * (x - y) / 4 + product

    if discriminant < 0,
      do: :math.sqrt(max(x * y - product, 0.0)),
      else: abs(midpoint) + :math.sqrt(discriminant)
  end

  defp step(a, lo, hi, iterations) do
    x = get(a, hi - 1, hi - 1)
    y = get(a, hi, hi)

    {trace, determinant} =
      if iterations > 0 and rem(iterations, 20) == 0 do
        shift = y + abs(get(a, hi, hi - 1)) + abs(get(a, hi - 1, hi - 2))
        {2 * shift, shift * shift}
      else
        {x + y, x * y - get(a, hi, hi - 1) * get(a, hi - 1, hi)}
      end

    h = get(a, lo, lo)
    sub = get(a, lo + 1, lo)

    initial = [
      h * h + get(a, lo, lo + 1) * sub - trace * h + determinant,
      sub * (h + get(a, lo + 1, lo + 1) - trace),
      sub * get(a, lo + 2, lo + 1)
    ]

    for k <- lo..(hi - 1) do
      rows = Enum.to_list(k..min(k + 2, hi))
      values = if k == lo, do: initial, else: Enum.map(rows, &get(a, &1, k - 1))
      v = reflector(values)
      left(a, rows, max(lo, k - 1)..hi, v)
      right(a, lo..min(hi, k + 3), rows, v)

      if k > lo do
        for i <- (k + 1)..min(k + 2, hi), do: put(a, i, k - 1, 0.0)
      end
    end
  end
end
