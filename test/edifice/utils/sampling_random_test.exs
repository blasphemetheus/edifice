defmodule Edifice.Utils.SamplingRandomTest do
  use ExUnit.Case, async: true
  alias Edifice.Utils.SamplingRandom

  test "matches Nx uniforms and key transitions exactly across seeds and odd/even shapes" do
    for seed <- [0, 1, 42, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF],
        shape <- [{}, {1}, {3}, {8}, {1, 17}, {8, 17}, {8, 4}, {3, 5, 7}] do
      key = Nx.Random.key(seed)
      {expected, next} = Nx.Random.uniform(key, shape: shape)
      {actual, actual_next} = SamplingRandom.uniform(key, shape: shape)
      assert Nx.type(actual) == {:f, 32}
      assert Nx.to_binary(actual) == Nx.to_binary(expected)
      assert Nx.to_binary(actual_next) == Nx.to_binary(next)
    end
  end

  test "a controller's chained draws retain the reference stream" do
    Enum.reduce(
      [{8, 8}, {8, 17}, {8, 17}, {8, 5}, {8, 17}, {8, 17}, {8, 4}],
      {Nx.Random.key(2026), Nx.Random.key(2026)},
      fn shape, {reference, candidate} ->
        {expected, reference} = Nx.Random.uniform(reference, shape: shape)
        {actual, candidate} = SamplingRandom.uniform(candidate, shape: shape)
        assert Nx.to_binary(actual) == Nx.to_binary(expected)
        assert Nx.to_binary(candidate) == Nx.to_binary(reference)
        {reference, candidate}
      end
    )
  end

  test "rejects invalid key shapes and types" do
    for key <- [Nx.tensor([1, 2]), Nx.tensor([1], type: :u32)] do
      assert_raise ArgumentError, ~r/u32 PRNG key/, fn -> SamplingRandom.uniform(key) end
    end
  end
end
