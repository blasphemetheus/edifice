defmodule Edifice.Serving.SamplingTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias Edifice.Serving.Sampling

  # Logits where token 3 is strongly preferred
  @logits Nx.tensor([[0.1, 0.2, 0.3, 5.0, 0.1, 0.0]])

  describe "sample_adaptive/4 (defn if branching)" do
    test "with top_k=0 samples without filtering" do
      key = Nx.Random.key(42)
      {token, _key} = Sampling.sample_adaptive(@logits, key, Nx.tensor(0.01), Nx.tensor(0))
      # Near-greedy temperature, no top-k: should pick token 3
      assert token |> Nx.squeeze() |> Nx.to_number() == 3
    end

    test "with top_k>0 filters to top candidates" do
      key = Nx.Random.key(42)
      {token, _key} = Sampling.sample_adaptive(@logits, key, Nx.tensor(0.01), Nx.tensor(2))
      # Top-2 keeps only tokens 2 and 3 (highest logits), greedy picks 3
      assert token |> Nx.squeeze() |> Nx.to_number() == 3
    end

    test "runtime top_k branching produces different distributions" do
      key = Nx.Random.key(99)

      # High temperature + no top-k: can sample any token
      tokens_no_filter =
        for seed <- 1..50 do
          {t, _} = Sampling.sample_adaptive(@logits, Nx.Random.key(seed), Nx.tensor(5.0), Nx.tensor(0))
          t |> Nx.squeeze() |> Nx.to_number()
        end

      # High temperature + top-k=1: always picks token 3
      tokens_top1 =
        for seed <- 1..50 do
          {t, _} = Sampling.sample_adaptive(@logits, Nx.Random.key(seed), Nx.tensor(5.0), Nx.tensor(1))
          t |> Nx.squeeze() |> Nx.to_number()
        end

      unique_no_filter = Enum.uniq(tokens_no_filter) |> length()
      unique_top1 = Enum.uniq(tokens_top1) |> length()

      # No filter should produce more variety than top-1
      assert unique_no_filter > unique_top1
      assert unique_top1 == 1
    end
  end

  describe "sample_greedy_or_stochastic/3 (defn if branching)" do
    test "greedy mode (temperature <= 0.01)" do
      key = Nx.Random.key(42)
      {token, _key} = Sampling.sample_greedy_or_stochastic(@logits, key, Nx.tensor(0.001))
      assert token |> Nx.squeeze() |> Nx.to_number() == 3
    end

    test "stochastic mode (temperature > 0.01)" do
      # With high temperature, should produce varied results across seeds
      tokens =
        for seed <- 1..50 do
          key = Nx.Random.key(seed)
          {t, _} = Sampling.sample_greedy_or_stochastic(@logits, key, Nx.tensor(5.0))
          t |> Nx.squeeze() |> Nx.to_number()
        end

      unique = Enum.uniq(tokens) |> length()
      # Should have variety with high temperature
      assert unique > 1
    end

    test "boundary: temperature exactly 0.01 is greedy" do
      key = Nx.Random.key(42)
      {token, _key} = Sampling.sample_greedy_or_stochastic(@logits, key, Nx.tensor(0.01))
      assert token |> Nx.squeeze() |> Nx.to_number() == 3
    end
  end
end
