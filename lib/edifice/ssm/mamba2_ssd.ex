defmodule Edifice.SSM.Mamba2SSD do
  @moduledoc false
  # AI-generated. Canonical SSD equations and provenance: docs/mamba2.md.
  # Matrix products handle time within each chunk. Only chunk boundaries carry
  # recurrent state; no [batch,time,heads,width,state] history is materialized.
  def scan(x, b, c, dt, a, d, initial, chunk) do
    unless is_integer(chunk) and chunk > 0,
      do: raise(ArgumentError, "chunk_size must be positive")

    length = Nx.axis_size(x, 1)
    batch = Nx.axis_size(x, 0)
    heads = Nx.axis_size(x, 2)
    width = Nx.axis_size(x, 3)
    groups = Nx.axis_size(b, 2)
    per_group = div(heads, groups)
    state_size = Nx.axis_size(b, 3)

    {ys, final} =
      Enum.map_reduce(0..div(length - 1, chunk), initial, fn i, h ->
        size = min(chunk, length - i * chunk)

        slice = fn t ->
          t |> Nx.slice_along_axis(i * chunk, size, axis: 1) |> Nx.transpose(axes: [0, 2, 1, 3])
        end

        xc = slice.(x)
        bc = slice.(b)
        cc = slice.(c)

        delta =
          dt |> Nx.slice_along_axis(i * chunk, size, axis: 1) |> Nx.transpose(axes: [0, 2, 1])

        da = Nx.multiply(delta, Nx.reshape(a, {1, :auto, 1}))
        cumulative = Nx.cumulative_sum(da, axis: 2)
        # Sum only intervening decay terms, avoiding cancellation from subtracting
        # large prefix sums. Mask before exp so no upper-triangle overflow occurs.
        row = Nx.iota({size, size}, axis: 0)
        col = Nx.iota({size, size}, axis: 1)
        square = {Nx.axis_size(x, 0), Nx.axis_size(x, 2), size, size}
        lower = Nx.broadcast(Nx.greater(row, col), square)
        causal = Nx.broadcast(Nx.greater_equal(row, col), square)
        terms = Nx.select(lower, Nx.new_axis(da, -1), 0.0)
        segments = Nx.cumulative_sum(terms, axis: 2)
        decay = Nx.exp(Nx.select(causal, segments, :neg_infinity))
        # Keep shared B/C grouped through the matrix products. Expanding them to
        # every head repeats work and creates large activation/gradient buffers.
        cb = Nx.dot(cc, [3], [0, 1], bc, [3], [0, 1]) |> Nx.new_axis(2)

        drive =
          Nx.multiply(xc, Nx.new_axis(delta, -1))
          |> Nx.reshape({batch, groups, per_group, size, width})

        transfer = Nx.multiply(cb, Nx.reshape(decay, {batch, groups, per_group, size, size}))

        local =
          Nx.dot(transfer, [4], [0, 1, 2], drive, [3], [0, 1, 2])
          |> Nx.reshape({batch, heads, size, width})

        grouped_h = Nx.reshape(h, {batch, groups, per_group, width, state_size})

        carried =
          Nx.dot(grouped_h, [4], [0, 1], cc, [3], [0, 1])
          |> Nx.transpose(axes: [0, 1, 2, 4, 3])
          |> Nx.reshape({batch, heads, size, width})
          |> Nx.multiply(Nx.new_axis(Nx.exp(cumulative), -1))

        y = Nx.add(local, carried) |> Nx.add(Nx.multiply(xc, Nx.reshape(d, {1, :auto, 1, 1})))
        last = Nx.slice_along_axis(cumulative, size - 1, 1, axis: 2)

        weighted =
          Nx.multiply(
            drive,
            Nx.reshape(Nx.exp(Nx.subtract(last, cumulative)), {batch, groups, per_group, size, 1})
          )

        update =
          Nx.dot(weighted, [3], [0, 1], bc, [2], [0, 1])
          |> Nx.reshape({batch, heads, width, state_size})

        h = Nx.add(Nx.multiply(h, Nx.new_axis(Nx.exp(last), -1)), update)
        {Nx.transpose(y, axes: [0, 2, 1, 3]), h}
      end)

    {Nx.concatenate(ys, axis: 1), final}
  end
end
