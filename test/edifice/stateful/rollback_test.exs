defmodule Edifice.StatefulRollbackTest do
  @moduledoc """
  The netplay rollback primitive (HANDOFF_BOT_RUNTIME.md task 2): run N
  steps, snapshot at k (through the serialize/deserialize wire format),
  run to N, restore, replay k..N with identical inputs → bitwise-identical
  outputs. Slippi rollback re-simulates frames when remote inputs arrive
  late; a policy that can't do this desyncs from the game it thinks it's
  playing.
  """
  use ExUnit.Case, async: true

  import Edifice.StatefulCase

  @moduletag :stateful

  setup do
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  describe "MinGRU rollback" do
    test "snapshot at 5, replay 6..12 is bitwise-identical" do
      assert_rollback_deterministic(:min_gru,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 2,
        dropout: 0.0
      )
    end

    test "snapshot at 1 (immediately after first frame)" do
      assert_rollback_deterministic(
        :min_gru,
        [embed_dim: 6, hidden_size: 6, num_layers: 1, dropout: 0.0],
        seq_len: 6,
        rollback_at: 1
      )
    end
  end

  describe "Mamba rollback" do
    test "snapshot at 5, replay 6..12 is bitwise-identical" do
      assert_rollback_deterministic(:mamba,
        embed_dim: 8,
        hidden_size: 8,
        state_size: 4,
        num_layers: 2,
        conv_size: 3,
        dropout: 0.0
      )
    end

    test "snapshot inside the conv warm-up window" do
      # rollback_at 2 < conv_size 4: the restored ring buffer still contains
      # left-pad zeros — replay must reproduce them exactly
      assert_rollback_deterministic(
        :mamba,
        [embed_dim: 6, hidden_size: 6, state_size: 4, num_layers: 1, conv_size: 4, dropout: 0.0],
        seq_len: 8,
        rollback_at: 2
      )
    end
  end
end
