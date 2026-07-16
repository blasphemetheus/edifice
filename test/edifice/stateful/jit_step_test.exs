defmodule Edifice.StatefulJitStepTest do
  @moduledoc """
  The JIT step path must satisfy the SAME correctness contract as the
  eager path: step == full forward at every prefix, bitwise rollback.
  These run the whole equivalence harness through `step_opts:
  [compiler: ...]` — if JIT tracing breaks an implementation, that's a
  finding, not a test to relax.

  Default runs use `Nx.Defn.Evaluator` (backend-agnostic, GPU-safe);
  the `:exla_only` twins compile through EXLA and run wherever EXLA
  tests are enabled.
  """
  use ExUnit.Case, async: true

  import Edifice.StatefulCase

  alias Edifice.Stateful

  @moduletag :stateful

  @evaluator [compiler: Nx.Defn.Evaluator]

  @archs [
    {:min_gru, [embed_dim: 8, hidden_size: 8, num_layers: 2, dropout: 0.0]},
    {:mamba,
     [embed_dim: 8, hidden_size: 8, state_size: 4, num_layers: 2, conv_size: 3, dropout: 0.0]},
    {:gated_ssm,
     [
       embed_dim: 8,
       hidden_size: 8,
       state_size: 4,
       num_layers: 1,
       conv_size: 3,
       dropout: 0.0,
       scan_mode: :causal
     ]},
    {:gru, [embed_dim: 8, hidden_size: 8, num_layers: 1, dropout: 0.0]},
    {:lstm, [embed_dim: 8, hidden_size: 8, num_layers: 1, dropout: 0.0]}
  ]

  setup do
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  describe "jitted step == full forward (Evaluator)" do
    for {arch, build_opts} <- @archs do
      test "#{arch}" do
        assert_step_matches_forward(unquote(arch), unquote(build_opts),
          step_opts: @evaluator
        )
      end
    end
  end

  describe "jitted rollback (Evaluator)" do
    test "mamba snapshot/replay is bitwise-identical through the jit path" do
      assert_rollback_deterministic(
        :mamba,
        [embed_dim: 6, hidden_size: 6, state_size: 4, num_layers: 1, conv_size: 3, dropout: 0.0],
        step_opts: @evaluator
      )
    end
  end

  describe "jit cache and init_state warmup" do
    test "jit_step returns the cached fun on repeat calls" do
      Stateful.clear_jit_cache()

      fun1 = Stateful.jit_step(Edifice.Recurrent.MinGRU, Nx.Defn.Evaluator)
      fun2 = Stateful.jit_step(Edifice.Recurrent.MinGRU, Nx.Defn.Evaluator)

      assert fun1 == fun2

      Stateful.clear_jit_cache()
      fun3 = Stateful.jit_step(Edifice.Recurrent.MinGRU, Nx.Defn.Evaluator)
      # A fresh compile after clearing — still a function, possibly a new one
      assert is_function(fun3, 3)
    end

    test "init_state with compiler warms the jit but returns a plain state" do
      opts = [embed_dim: 6, hidden_size: 6, num_layers: 1]
      {params, _predict, _x} = build_forward(:min_gru, opts, 2, 1, 0)

      plain = Edifice.init_state(:min_gru, params, opts)
      warmed = Edifice.init_state(:min_gru, params, opts ++ @evaluator)

      # Identical plain-container state — the compiled fun is NOT inside it
      assert Nx.to_binary(plain.h) == Nx.to_binary(warmed.h)
      assert :ok = Stateful.assert_plain_container!(warmed)

      # And it still serializes (the rollback wire-format property)
      restored = warmed |> Stateful.serialize() |> Stateful.deserialize()
      assert Nx.shape(restored.h) == Nx.shape(warmed.h)
    end

    test "init_state warns (not raises) when compiler given without embed_dim" do
      {params, _predict, _x} =
        build_forward(:min_gru, [embed_dim: 6, hidden_size: 6, num_layers: 1], 2, 1, 0)

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          state =
            Edifice.init_state(
              :min_gru,
              params,
              [hidden_size: 6, num_layers: 1] ++ @evaluator
            )

          assert %{h: _} = state
        end)

      assert log =~ "skipping JIT warmup"
    end
  end

  describe "jitted step == full forward (EXLA)" do
    @describetag :exla_only

    for {arch, build_opts} <- @archs do
      test "#{arch}" do
        assert_step_matches_forward(unquote(arch), unquote(build_opts),
          step_opts: [compiler: EXLA]
        )
      end
    end
  end
end
