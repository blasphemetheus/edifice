defmodule Edifice.ProfileStepTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO

  @moduletag :stateful

  setup do
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  describe "run/2 with mode: :step" do
    test "returns step-latency stats and state footprint" do
      result =
        Edifice.Profile.run(:min_gru,
          mode: :step,
          embed_dim: 16,
          hidden_size: 16,
          num_layers: 1,
          dropout: 0.0,
          warmup: 2,
          iterations: 5
        )

      assert result.arch == :min_gru
      assert result.mode == :step

      for key <- [:init_ms, :p50_step_ms, :p95_step_ms, :mean_step_ms, :max_step_ms] do
        assert is_number(result[key]) and result[key] >= 0, "#{key} missing or negative"
      end

      # MinGRU state is %{h: [1, num_layers, hidden]} f32
      assert result.state_bytes == 1 * 1 * 16 * 4

      # Percentiles must be ordered
      assert result.p50_step_ms <= result.p95_step_ms
      assert result.p95_step_ms <= result.max_step_ms

      assert result.param_count > 0
      assert is_binary(result.backend)
      assert result.config.batch_size == 1
    end

    test "gated_ssm gets scan_mode: :causal injected" do
      result =
        Edifice.Profile.run(:gated_ssm,
          mode: :step,
          embed_dim: 8,
          hidden_size: 8,
          state_size: 4,
          num_layers: 1,
          warmup: 1,
          iterations: 3
        )

      assert result.mode == :step
      # state %{h: [1, 1, 16], conv: [1, 1, 3, 16]} f32 = (16 + 48) * 4
      assert result.state_bytes == (16 + 48) * 4
    end

    test "raises descriptively for non-stateful architectures" do
      err =
        assert_raise ArgumentError, fn ->
          Edifice.Profile.run(:mlp, mode: :step, iterations: 1)
        end

      assert err.message =~ "does not implement Edifice.Stateful"
      assert err.message =~ "min_gru"
    end

    test "compare/1 prints the step table without raising" do
      output =
        capture_io(fn ->
          Edifice.Profile.compare(
            archs: [:min_gru, :gru],
            mode: :step,
            embed_dim: 8,
            hidden_size: 8,
            num_layers: 1,
            dropout: 0.0,
            warmup: 1,
            iterations: 3
          )
        end)

      assert output =~ "p50(ms)"
      assert output =~ "State(KB)"
      assert output =~ "min_gru"
      assert output =~ "re-measure on idle GPU"
    end
  end
end
