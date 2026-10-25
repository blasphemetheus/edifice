defmodule Edifice.Attention.SparseAttentionTest do
  use ExUnit.Case, async: true
  import Edifice.TestHelpers
  import Nx.Testing
  alias Edifice.Attention.{MoBA, NSA, SparseAttention}

  defp tensor(time), do: Nx.sin(Nx.iota({2, 2, time, 3}, type: :f32))
  defp prefix(x, n), do: Nx.slice_along_axis(x, 0, n, axis: 2)

  defp moba(q, k, v, size, topk, causal \\ true) do
    blocks = SparseAttention.route(q, k, size, topk, causal)
    SparseAttention.selected(q, k, v, blocks, size, causal)
  end

  defp nsa(q, k, v, gates \\ Nx.tensor([0.0, 0.0, 0.0])) do
    NSA.attention(q, k, v, gates,
      compression_ratio: 3,
      block_size: 4,
      num_selected_blocks: 2,
      window_size: 3
    )
  end

  defp dense(q, k, v, valid) do
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), :math.sqrt(Nx.axis_size(q, 3)))
    Nx.dot(SparseAttention.masked_softmax(scores, valid), [3], [0, 1], v, [2], [0, 1])
  end

  test "full selection equals dense attention for tails, short sequences and noncausal mode" do
    for time <- [1, 3, 9], causal <- [true, false] do
      q = tensor(time)
      k = Nx.cos(q)
      v = Nx.multiply(q, 2)

      valid =
        if causal,
          do: Nx.greater_equal(Nx.iota({time, 1}, axis: 0), Nx.iota({1, time}, axis: 1)),
          else: Nx.broadcast(1, {time, time})

      assert_all_close(moba(q, k, v, 4, 20, causal), dense(q, k, v, valid))
    end
  end

  test "MoBA and NSA every prefix is invariant to future keys, values and queries" do
    q = tensor(11)
    k = Nx.cos(q)
    v = Nx.multiply(q, 7)

    for fun <- [&moba(&1, &2, &3, 4, 2), &nsa/3], n <- 1..10 do
      assert_all_close(prefix(fun.(q, k, v), n), fun.(prefix(q, n), prefix(k, n), prefix(v, n)))
    end
  end

  test "ties select exact cardinality and current block is mandatory" do
    q = Nx.broadcast(0.0, {1, 1, 13, 2})
    selected = SparseAttention.route(q, q, 4, 2, true)
    assert Nx.shape(selected) == {1, 1, 13, 2}

    for {blocks, position} <-
          selected |> Nx.to_flat_list() |> Enum.chunk_every(2) |> Enum.with_index() do
      assert length(Enum.uniq(blocks)) == 2
      assert div(position, 4) in blocks
      if position >= 4, do: assert(Enum.all?(blocks, &(&1 <= div(position, 4))))
    end
  end

  test "selected attention retrieves a distant block instead of only the query block" do
    q = Nx.broadcast(1.0, {1, 1, 8, 1})
    k = Nx.reshape(Nx.tensor([10.0, 10, -10, -10, -10, -10, -10, -10]), {1, 1, 8, 1})
    v = Nx.reshape(Nx.tensor([7.0, 7, 0, 0, 0, 0, 0, 0]), {1, 1, 8, 1})
    output = moba(q, k, v, 2, 2)
    assert_in_delta Nx.to_number(output[0][0][7][0]), 7.0, 1.0e-5
  end

  test "local gather matches dense sliding window outputs and gradients" do
    q = tensor(7)
    k = Nx.cos(q)
    v = Nx.multiply(q, 2)
    distance = Nx.subtract(Nx.iota({7, 1}, axis: 0), Nx.iota({1, 7}, axis: 1))
    valid = Nx.logical_and(Nx.greater_equal(distance, 0), Nx.less(distance, 3))
    assert_all_close(SparseAttention.window(q, k, v, 3), dense(q, k, v, valid))
    valid = Nx.backend_transfer(valid, Nx.BinaryBackend)
    a = Nx.Defn.grad(fn {q, k, v} -> Nx.sum(SparseAttention.window(q, k, v, 3)) end).({q, k, v})
    b = Nx.Defn.grad(fn {q, k, v} -> Nx.sum(dense(q, k, v, valid)) end).({q, k, v})
    for {x, y} <- Enum.zip(Tuple.to_list(a), Tuple.to_list(b)), do: assert_all_close(x, y)
  end

  test "selected attention gradients equal the dense full-selection reference" do
    q = tensor(5)
    k = Nx.cos(q)
    v = Nx.multiply(q, 2)
    valid = Nx.greater_equal(Nx.iota({5, 1}, axis: 0), Nx.iota({1, 5}, axis: 1))
    valid = Nx.backend_transfer(valid, Nx.BinaryBackend)
    a = Nx.Defn.grad(fn {q, k, v} -> Nx.sum(moba(q, k, v, 3, 2)) end).({q, k, v})
    b = Nx.Defn.grad(fn {q, k, v} -> Nx.sum(dense(q, k, v, valid)) end).({q, k, v})
    for {x, y} <- Enum.zip(Tuple.to_list(a), Tuple.to_list(b)), do: assert_all_close(x, y)
  end

  test "NSA early compressed rows and large gates have finite gradients" do
    q = tensor(7)

    {output, gradients} =
      Nx.Defn.value_and_grad(fn {q, k, v, g} -> Nx.sum(nsa(q, k, v, g)) end).(
        {q, Nx.cos(q), q, Nx.tensor([1000.0, 999, 998])}
      )

    assert_finite!(output)
    for x <- Tuple.to_list(gradients), do: assert_finite!(x)
  end

  test "pooling includes every tail token with its actual count" do
    x = Nx.reshape(Nx.tensor([1.0, 3, 5, 7, 9]), {1, 1, 5, 1})
    assert Nx.to_flat_list(SparseAttention.block_means(x, 3)) == [3.0, 8.0]
  end

  test "invalid sparse budgets fail at model construction" do
    assert_raise ArgumentError, ~r/topk/, fn -> MoBA.build(embed_dim: 8, topk: 0) end

    assert_raise ArgumentError, ~r/divisible/, fn ->
      MoBA.build(embed_dim: 8, hidden_size: 7, num_heads: 2)
    end

    assert_raise ArgumentError, ~r/compression_ratio/, fn ->
      NSA.build(embed_dim: 8, compression_ratio: 0)
    end
  end

  test "NSA selection budget changes distant retrieval" do
    q = Nx.broadcast(1.0, {1, 1, 8, 1})
    k = Nx.reshape(Nx.tensor([10.0, 10, -10, -10, -10, -10, -10, -10]), {1, 1, 8, 1})
    v = Nx.reshape(Nx.tensor([7.0, 7, 0, 0, 0, 0, 0, 0]), {1, 1, 8, 1})
    gates = Nx.tensor([-1000.0, 0, -1000])

    run = fn budget ->
      NSA.attention(q, k, v, gates,
        compression_ratio: 2,
        block_size: 2,
        num_selected_blocks: budget,
        window_size: 2
      )
    end

    assert_in_delta Nx.to_number(run.(1)[0][0][7][0]), 0, 1.0e-5
    assert_in_delta Nx.to_number(run.(2)[0][0][7][0]), 7, 1.0e-5
  end

  test "nonaligned compression windows distribute probability by overlap" do
    weights = Nx.reshape(Nx.tensor([0.2, 0.3, 0.5]), {1, 1, 1, 3})
    expected = Nx.reshape(Nx.tensor([0.3, 0.7]), {1, 1, 1, 2})
    assert_all_close(SparseAttention.compression_scores(weights, 7, 3, 4), expected)
  end

  test "sparse budget output and gradients match independently expanded dense mask" do
    inputs = {tensor(11), Nx.cos(tensor(11)), Nx.multiply(tensor(11), 3)}

    reference = fn {q, k, v} ->
      selected = SparseAttention.route(q, k, 3, 2, true)

      mask =
        Nx.equal(Nx.new_axis(selected, -1), Nx.quotient(Nx.iota({11}), 3)) |> Nx.any(axes: [-2])

      causal = Nx.greater_equal(Nx.iota({11, 1}, axis: 0), Nx.iota({1, 11}, axis: 1))
      dense(q, k, v, Nx.logical_and(mask, causal))
    end

    sparse = fn {q, k, v} -> moba(q, k, v, 3, 2) end
    assert_all_close(sparse.(inputs), reference.(inputs))
    a = Nx.Defn.grad(fn xs -> Nx.sum(sparse.(xs)) end).(inputs)
    b = Nx.Defn.grad(fn xs -> Nx.sum(reference.(xs)) end).(inputs)
    for {x, y} <- Enum.zip(Tuple.to_list(a), Tuple.to_list(b)), do: assert_all_close(x, y)
  end
end
