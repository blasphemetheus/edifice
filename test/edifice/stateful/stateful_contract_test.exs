defmodule Edifice.StatefulContractTest do
  use ExUnit.Case, async: true

  alias Edifice.Stateful

  @moduletag :stateful

  describe "stateful?/1" do
    test "true for implementing architectures" do
      assert Stateful.stateful?(:min_gru)
      assert Edifice.stateful?(:min_gru)
    end

    test "false for non-stateful architectures" do
      refute Stateful.stateful?(:mlp)
      refute Stateful.stateful?(:decoder_only)
      refute Edifice.stateful?(:mlp)
    end

    test "false for unknown atoms" do
      refute Stateful.stateful?(:not_a_real_arch)
    end

    test "accepts modules directly" do
      assert Stateful.stateful?(Edifice.Recurrent.MinGRU)
      refute Stateful.stateful?(Edifice.Feedforward.MLP)
    end
  end

  describe "Edifice.init_state/3 and Edifice.step/4 dispatch" do
    test "dispatches through the registry" do
      opts = [embed_dim: 6, hidden_size: 6, num_layers: 2]
      {params, _predict, x} = Edifice.StatefulCase.build_forward(:min_gru, opts, 2, 1, 0)

      state = Edifice.init_state(:min_gru, params, opts)
      frame = x |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {out, new_state} = Edifice.step(:min_gru, params, state, frame)

      assert Nx.shape(out) == {1, 6}
      assert Nx.shape(new_state.h) == {1, 2, 6}
    end

    test "raises descriptively for non-stateful architectures" do
      err =
        assert_raise ArgumentError, fn ->
          Edifice.init_state(:mlp, %{}, [])
        end

      assert err.message =~ "does not implement"
      assert err.message =~ "min_gru"
    end
  end

  describe "state container invariants" do
    test "state is a plain Nx container (tensor leaves only)" do
      state = Edifice.init_state(:min_gru, %{}, embed_dim: 4, hidden_size: 4, num_layers: 1)
      assert :ok = Stateful.assert_plain_container!(state)
    end

    test "snapshot/serialize/deserialize round-trip preserves tensors" do
      state = Edifice.init_state(:min_gru, %{}, embed_dim: 4, hidden_size: 8, num_layers: 3)

      snap = Stateful.snapshot(state)
      assert Nx.to_binary(snap.h) == Nx.to_binary(state.h)

      restored = state |> Stateful.serialize() |> Stateful.deserialize()
      assert Nx.to_binary(restored.h) == Nx.to_binary(state.h)
      assert Nx.shape(restored.h) == Nx.shape(state.h)
    end
  end
end
