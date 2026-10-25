defmodule Edifice.Attention.SparseAttention do
  @moduledoc false

  # Portable gathered attention. All tensors use [batch, heads, time, dim].
  # Routing is discrete; gradients flow through the selected keys and values.
  def validate_positive!(opts) do
    Enum.each(opts, fn {name, value} ->
      unless is_integer(value) and value > 0 do
        raise ArgumentError, "#{name} must be a positive integer, got: #{inspect(value)}"
      end
    end)
  end

  def block_means(x, size) do
    {batch, heads, time, dim} = Nx.shape(x)
    blocks = div(time + size - 1, size)
    padded = Nx.pad(x, 0, [{0, 0, 0}, {0, 0, 0}, {0, blocks * size - time, 0}, {0, 0, 0}])
    counts = Nx.min(Nx.subtract(time, Nx.multiply(Nx.iota({blocks}), size)), size)

    padded
    |> Nx.reshape({batch, heads, blocks, size, dim})
    |> Nx.sum(axes: [3])
    |> Nx.divide(Nx.reshape(counts, {1, 1, blocks, 1}))
  end

  def route(q, k, size, topk, causal) do
    # Route in f32 even when the model uses a lower precision.
    means = block_means(Nx.as_type(k, :f32), size)
    scores = Nx.dot(Nx.as_type(q, :f32), [3], [0, 1], means, [3], [0, 1])
    select_blocks(scores, size, topk, causal)
  end

  def select_blocks(scores, size, topk, causal) do
    {_, _, time, blocks} = Nx.shape(scores)

    scores =
      if causal do
        current = Nx.quotient(Nx.iota({time, 1}, axis: 0), size)
        block = Nx.iota({1, blocks}, axis: 1)
        # Never rank the current block using its mean: it contains future keys.
        # It is always selected and receives a token-level causal mask below.
        prior =
          Nx.select(
            Nx.broadcast(Nx.less(block, current), Nx.shape(scores)),
            scores,
            Nx.Constants.neg_infinity()
          )

        Nx.select(
          Nx.broadcast(Nx.equal(block, current), Nx.shape(scores)),
          Nx.Constants.infinity(),
          prior
        )
      else
        scores
      end

    {_, indices} = Nx.top_k(scores, k: min(topk, blocks))
    indices
  end

  def selected(q, k, v, blocks, size, causal) do
    {batch, heads, time, count} = Nx.shape(blocks)

    indices =
      blocks
      |> Nx.new_axis(-1)
      |> Nx.multiply(size)
      |> Nx.add(Nx.iota({size}))
      |> Nx.reshape({batch, heads, time, count * size})

    valid = Nx.less(indices, time)

    valid =
      if causal,
        do: Nx.logical_and(valid, Nx.less_equal(indices, Nx.iota({1, 1, time, 1}, axis: 2))),
        else: valid

    attend_indices(q, k, v, indices, valid)
  end

  def window(q, k, v, size) do
    {batch, heads, time, _} = Nx.shape(q)
    size = min(size, time)
    indices = Nx.subtract(Nx.iota({time, 1}, axis: 0), Nx.iota({1, size}, axis: 1))
    indices = Nx.broadcast(Nx.reshape(indices, {1, 1, time, size}), {batch, heads, time, size})
    attend_indices(q, k, v, indices, Nx.greater_equal(indices, 0))
  end

  defp attend_indices(q, k, v, indices, valid) do
    {batch, heads, time, dim} = Nx.shape(q)
    offsets = Nx.multiply(Nx.iota({batch, heads, 1, 1}), time)
    indices = Nx.add(Nx.clip(indices, 0, time - 1), offsets)
    keys = Nx.take(Nx.reshape(k, {batch * heads * time, dim}), indices)
    values = Nx.take(Nx.reshape(v, {batch * heads * time, dim}), indices)

    scores =
      Nx.sum(Nx.multiply(Nx.new_axis(q, -2), keys), axes: [-1]) |> Nx.divide(:math.sqrt(dim))

    weights = masked_softmax(scores, valid)
    Nx.sum(Nx.multiply(Nx.new_axis(weights, -1), values), axes: [-2])
  end

  def compressed(q, k, v, size) do
    {_, _, time, dim} = Nx.shape(q)
    keys = block_means(k, size)
    values = block_means(v, size)
    blocks = Nx.axis_size(keys, 2)
    # A pooled block is visible only after a full compression window arrives.
    # A trailing partial window stays invisible, preserving prefix invariance.
    ends = Nx.subtract(Nx.multiply(Nx.add(Nx.iota({1, blocks}, axis: 1), 1), size), 1)
    valid = Nx.greater_equal(Nx.iota({time, 1}, axis: 0), ends)
    scores = Nx.dot(q, [3], [0, 1], keys, [3], [0, 1]) |> Nx.divide(:math.sqrt(dim))
    weights = masked_softmax(scores, valid)
    {Nx.dot(weights, [3], [0, 1], values, [2], [0, 1]), weights}
  end

  def compression_scores(weights, time, compression, size) do
    compressed = Nx.axis_size(weights, 3)
    blocks = div(time + size - 1, size)
    # Only gather compression windows overlapping each selection block. A dense
    # compressed-to-selected mapping would introduce a cubic matrix multiply.
    starts_b = Nx.multiply(Nx.iota({blocks, 1}, axis: 0), size)
    count = div(size + compression - 1, compression) + 1
    indices = Nx.add(Nx.quotient(starts_b, compression), Nx.iota({1, count}, axis: 1))
    starts_c = Nx.multiply(indices, compression)
    ends_c = Nx.min(Nx.add(starts_c, compression), time)
    ends_b = Nx.min(Nx.add(starts_b, size), time)
    overlap = Nx.max(Nx.subtract(Nx.min(ends_c, ends_b), Nx.max(starts_c, starts_b)), 0)
    allocation = Nx.divide(overlap, Nx.max(Nx.subtract(ends_c, starts_c), 1))

    weights
    |> Nx.take(Nx.min(indices, compressed - 1), axis: 3)
    |> Nx.multiply(allocation)
    |> Nx.sum(axes: [-1])
  end

  # All-masked rows (early compressed queries) return zero, including gradients.
  def masked_softmax(scores, valid) do
    valid = Nx.broadcast(valid, Nx.shape(scores))
    masked = Nx.select(valid, scores, Nx.Constants.neg_infinity(Nx.type(scores)))
    maximum = Nx.reduce_max(masked, axes: [-1], keep_axes: true)
    maximum = Nx.select(Nx.any(valid, axes: [-1], keep_axes: true), maximum, 0)
    exp = Nx.exp(Nx.subtract(masked, maximum))
    Nx.divide(exp, Nx.max(Nx.sum(exp, axes: [-1], keep_axes: true), 1.0))
  end
end
