defmodule Edifice.Training.MonitorTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog
  import Nx.Defn

  alias Edifice.Training.Monitor

  # Logger is set to :warning in test_helper.exs, so we temporarily
  # lower it for tests that need to capture :info messages.
  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)
    on_exit(fn -> Logger.configure(level: previous_level) end)
  end

  # ============================================================================
  # Defn integration — passthrough behavior
  # ============================================================================

  describe "observe/2 in defn" do
    defn observe_loss(x) do
      loss = Nx.mean(x)
      Monitor.observe(loss, label: "test_loss")
    end

    test "returns tensor unchanged and logs" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      log =
        capture_log(fn ->
          result = observe_loss(x)
          assert Nx.to_number(result) == 2.5
        end)

      assert log =~ "[Monitor] test_loss:"
      assert log =~ "2.5"
    end

    defn observe_passthrough(x) do
      Monitor.observe(x, label: "pass")
    end

    test "passes through multi-dimensional tensors" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

      log =
        capture_log(fn ->
          result = observe_passthrough(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "[Monitor] pass:"
    end
  end

  describe "observe_stats/2 in defn" do
    defn stats_check(x) do
      Monitor.observe_stats(x, label: "activations")
    end

    test "logs mean, std, min, max and returns tensor unchanged" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      log =
        capture_log(fn ->
          result = stats_check(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "activations:"
      assert log =~ "mean="
      assert log =~ "std="
      assert log =~ "min="
      assert log =~ "max="
    end
  end

  describe "assert_finite/2 in defn" do
    defn check_finite(x) do
      Monitor.assert_finite(x, label: "weights")
    end

    test "passes through clean tensors silently" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      log =
        capture_log(fn ->
          result = check_finite(x)
          assert Nx.equal(result, x) |> Nx.all() |> Nx.to_number() == 1
        end)

      refute log =~ "NaN"
      refute log =~ "Inf"
    end

    defn check_finite_nan(x) do
      bad = Nx.divide(0.0, 0.0)
      y = Nx.add(x, bad)
      Monitor.assert_finite(y, label: "nan_test")
    end

    test "warns on NaN" do
      x = Nx.tensor([1.0, 2.0])

      log =
        capture_log(fn ->
          check_finite_nan(x)
        end)

      assert log =~ "nan_test"
      assert log =~ "NaN=true"
    end

    defn check_finite_halt(x) do
      Monitor.assert_finite(x, label: "halt_test", halt: true)
    end

    test "raises when halt: true and tensor has NaN" do
      x = Nx.divide(Nx.tensor(0.0), Nx.tensor(0.0))

      assert_raise RuntimeError, ~r/halt_test/, fn ->
        check_finite_halt(x)
      end
    end
  end

  # ============================================================================
  # Non-defn utilities
  # ============================================================================

  describe "observe_grad_norm/2" do
    test "computes global norm of gradient map" do
      grads = %{
        "layer1" => %{
          "kernel" => Nx.tensor([[3.0, 4.0]]),
          "bias" => Nx.tensor([0.0])
        }
      }

      log =
        capture_log(fn ->
          result = Monitor.observe_grad_norm(grads, label: "grads")
          assert result == grads
        end)

      assert log =~ "grads:"
      # sqrt(9 + 16 + 0) = 5.0
      assert log =~ "5.0"
    end

    test "handles deeply nested param maps" do
      grads = %{
        "encoder" => %{
          "layer_0" => %{
            "kernel" => Nx.tensor([1.0]),
            "bias" => Nx.tensor([0.0])
          }
        }
      }

      log =
        capture_log(fn ->
          Monitor.observe_grad_norm(grads, label: "deep")
        end)

      assert log =~ "deep:"
    end
  end

  describe "every option" do
    defn observe_every(x) do
      Monitor.observe(x, label: "every_test", every: 3)
    end

    test "only logs every N calls" do
      # Reset counter
      Process.delete({Monitor, :counter, "every_test"})
      x = Nx.tensor(1.0)

      logs =
        for _i <- 1..6 do
          capture_log(fn -> observe_every(x) end)
        end

      # Should log on calls 3 and 6 (every: 3)
      non_empty = Enum.filter(logs, &(&1 != ""))
      assert length(non_empty) == 2
    end
  end

  # ============================================================================
  # Callback unit tests (direct invocation, no defn)
  # ============================================================================

  describe "callbacks directly" do
    test "observe_callback logs and returns tensor" do
      t = Nx.tensor(3.14)

      log =
        capture_log(fn ->
          result = Monitor.observe_callback(t, label: "direct")
          assert Nx.to_number(result) == Nx.to_number(t)
        end)

      assert log =~ "[Monitor] direct:"
      assert log =~ "3.14"
    end

    test "observe_stats_callback logs statistics" do
      t = Nx.tensor([0.0, 10.0])

      log =
        capture_log(fn ->
          Monitor.observe_stats_callback(t, label: "stats")
        end)

      assert log =~ "stats:"
      assert log =~ "mean="
      assert log =~ "min="
      assert log =~ "max="
    end

    test "assert_finite_callback passes clean tensor" do
      t = Nx.tensor([1.0, 2.0])

      log =
        capture_log(fn ->
          result = Monitor.assert_finite_callback(t, label: "clean")
          assert result == t
        end)

      refute log =~ "NaN"
    end

    test "assert_finite_callback warns on Inf" do
      # BinaryBackend raises on 1/0, so construct Inf via exp overflow
      t = Nx.tensor(1.0e38) |> Nx.multiply(1.0e38)

      log =
        capture_log(fn ->
          Monitor.assert_finite_callback(t, label: "inf_test")
        end)

      assert log =~ "inf_test"
      assert log =~ "Inf=true"
    end
  end
end
