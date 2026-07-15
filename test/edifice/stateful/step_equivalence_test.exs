defmodule Edifice.StatefulStepEquivalenceTest do
  @moduledoc """
  THE correctness pin for `Edifice.Stateful` (HANDOFF_BOT_RUNTIME.md task 2):
  iterated `step/3` must reproduce the full-sequence forward at every prefix
  length. If this equivalence holds, everything downstream (latency numbers,
  rollback behavior, the bot's frame loop) is trustworthy.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Edifice.StatefulCase

  @moduletag :stateful
  @moduletag timeout: 120_000

  setup do
    # Pin the pure-Nx fallback paths (no fused CUDA kernels) so step and
    # forward compare the same math
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  describe "MinGRU step == forward" do
    test "single layer, embed == hidden" do
      assert_step_matches_forward(:min_gru,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 1,
        dropout: 0.0
      )
    end

    test "three layers, embed != hidden (input projection)" do
      assert_step_matches_forward(:min_gru,
        embed_dim: 12,
        hidden_size: 8,
        num_layers: 3,
        dropout: 0.0
      )
    end

    test "batch of 2" do
      assert_step_matches_forward(:min_gru,
        [embed_dim: 8, hidden_size: 8, num_layers: 2, dropout: 0.0],
        batch: 2
      )
    end

    property "holds across random seeds and sequence lengths" do
      check all(
              seed <- StreamData.integer(0..1_000),
              seq_len <- StreamData.integer(1..10),
              batch <- StreamData.integer(1..2),
              max_runs: 8
            ) do
        assert_step_matches_forward(
          :min_gru,
          [embed_dim: 6, hidden_size: 6, num_layers: 2, dropout: 0.0],
          seed: seed,
          seq_len: seq_len,
          batch: batch
        )
      end
    end
  end
end
