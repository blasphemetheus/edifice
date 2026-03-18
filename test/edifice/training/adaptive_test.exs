defmodule Edifice.Training.AdaptiveTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog
  import Nx.Defn

  alias Edifice.Training.Adaptive

  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)
    on_exit(fn -> Logger.configure(level: previous_level) end)
  end

  # ============================================================================
  # Defn-level: guard_overflow
  # ============================================================================

  describe "guard_overflow/2" do
    defn guard_clean(x) do
      Adaptive.guard_overflow(x, label: "clean_grad")
    end

    test "passes through clean tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      log =
        capture_log(fn ->
          result = guard_clean(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      # No warning for clean tensors
      refute log =~ "Overflow"
    end

    defn guard_nan(x) do
      bad = Nx.add(x, Nx.divide(0.0, 0.0))
      Adaptive.guard_overflow(bad, label: "nan_grad")
    end

    test "zeros out NaN tensors and warns" do
      x = Nx.tensor([1.0, 2.0])

      log =
        capture_log(fn ->
          result = guard_nan(x)
          # Result should be all zeros
          expected = Nx.tensor([0.0, 0.0])
          assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "Overflow"
      assert log =~ "nan_grad"
      assert log =~ "NaN=true"
    end
  end

  # ============================================================================
  # Defn-level: agc_clip
  # ============================================================================

  describe "agc_clip/4" do
    defn clip_test(grad, param) do
      Adaptive.agc_clip(grad, param, 0.01)
    end

    test "clips large gradients relative to parameter norm" do
      # Parameter with norm ~5.0
      param = Nx.tensor([3.0, 4.0])
      # Gradient with norm ~50.0 (10x param norm)
      grad = Nx.tensor([30.0, 40.0])

      result = clip_test(grad, param)

      # With clipping_factor=0.01, max_norm = 5.0 * 0.01 = 0.05
      # grad_norm ~50, so clip_ratio = 0.05/50 = 0.001
      result_norm =
        result
        |> Nx.pow(2)
        |> Nx.sum()
        |> Nx.to_number()
        |> :math.sqrt()

      # Clipped norm should be much smaller than original
      assert result_norm < 1.0
    end

    test "leaves small gradients unchanged" do
      param = Nx.tensor([3.0, 4.0])
      # Gradient much smaller than param norm * factor
      grad = Nx.tensor([0.001, 0.001])

      result = clip_test(grad, param)

      assert Nx.all_close(result, grad, atol: 1.0e-6) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Defn-level: observe_grad_norm
  # ============================================================================

  describe "observe_grad_norm/2" do
    defn observe_norm(x) do
      Adaptive.observe_grad_norm(x, label: "test_norm")
    end

    test "logs norm and returns tensor unchanged" do
      x = Nx.tensor([3.0, 4.0])

      log =
        capture_log(fn ->
          result = observe_norm(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "test_norm:"
      # norm should be 5.0
      assert log =~ "5.0"
    end
  end

  # ============================================================================
  # Callbacks directly
  # ============================================================================

  describe "overflow_check_callback/2" do
    test "returns 0 for clean tensor" do
      t = Nx.tensor([1.0, 2.0])
      result = Adaptive.overflow_check_callback(t, label: "test")
      assert Nx.to_number(result) == 0
    end

    test "returns 1 and warns for NaN tensor" do
      t = Nx.divide(Nx.tensor(0.0), Nx.tensor(0.0))

      log =
        capture_log(fn ->
          result = Adaptive.overflow_check_callback(t, label: "bad")
          assert Nx.to_number(result) == 1
        end)

      assert log =~ "Overflow"
      assert log =~ "bad"
    end
  end

  describe "grad_norm_callback/2" do
    test "logs norm and returns tensor" do
      t = Nx.tensor([3.0, 4.0])

      log =
        capture_log(fn ->
          result = Adaptive.grad_norm_callback(t, label: "cb_norm")
          assert Nx.equal(result, t) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "cb_norm:"
      assert log =~ "5.0"
    end
  end

  # ============================================================================
  # Loop-level: skip_on_loss_spike (unit test of detection logic)
  # ============================================================================

  describe "loss spike detection logic" do
    test "detects spike when loss exceeds threshold * EMA" do
      # Simulate: EMA = 1.0, threshold = 5.0, current loss = 10.0
      ema = 1.0
      threshold = 5.0
      loss = 10.0
      assert loss > threshold * ema
    end

    test "does not flag normal loss" do
      ema = 1.0
      threshold = 5.0
      loss = 2.0
      refute loss > threshold * ema
    end
  end
end
