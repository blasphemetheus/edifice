defmodule Edifice.CUDA.AutoTuneProfilerTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog
  import ExUnit.CaptureIO
  import Nx.Defn

  alias Edifice.CUDA.AutoTuneProfiler

  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)

    on_exit(fn ->
      Logger.configure(level: previous_level)
      if AutoTuneProfiler.active?(), do: AutoTuneProfiler.stop()
    end)
  end

  describe "lifecycle" do
    test "start/stop cycle" do
      assert AutoTuneProfiler.start() == :ok
      assert AutoTuneProfiler.active?()
      assert AutoTuneProfiler.stop() == :ok
      refute AutoTuneProfiler.active?()
    end

    test "double start warns" do
      AutoTuneProfiler.start()

      log =
        capture_log(fn ->
          assert AutoTuneProfiler.start() == :already_running
        end)

      assert log =~ "Already running"
    end

    test "stop when not running" do
      assert AutoTuneProfiler.stop() == :not_running
    end
  end

  describe "record/4" do
    test "records dispatch counts" do
      AutoTuneProfiler.start()

      AutoTuneProfiler.record(:mingru, :custom_call, {4, 60, 64}, {:f, 32})
      AutoTuneProfiler.record(:mingru, :custom_call, {4, 60, 64}, {:f, 32})
      AutoTuneProfiler.record(:mingru, :fallback, {4, 60, 64}, {:f, 32})

      entries = :ets.tab2list(:__edifice_autotune_profiler__)
      assert length(entries) == 2

      custom_call_entry =
        Enum.find(entries, fn {{_k, path, _s, _d}, _c} -> path == :custom_call end)

      assert elem(custom_call_entry, 1) == 2

      fallback_entry =
        Enum.find(entries, fn {{_k, path, _s, _d}, _c} -> path == :fallback end)

      assert elem(fallback_entry, 1) == 1
    end

    test "no-op when profiler is not active" do
      refute AutoTuneProfiler.active?()
      # Should not crash
      AutoTuneProfiler.record(:mingru, :custom_call, {4, 60, 64}, {:f, 32})
    end
  end

  describe "report/0" do
    test "prints dispatch summary" do
      AutoTuneProfiler.start()

      AutoTuneProfiler.record(:mingru, :custom_call, {4, 60, 64}, {:f, 32})
      AutoTuneProfiler.record(:mingru, :custom_call, {4, 60, 64}, {:f, 32})
      AutoTuneProfiler.record(:minlstm, :fallback, {4, 60, 128}, {:bf, 16})

      output = capture_io(fn -> AutoTuneProfiler.report() end)

      assert output =~ "Kernel Dispatch Report"
      assert output =~ "Total dispatches: 3"
      assert output =~ "mingru"
      assert output =~ "minlstm"
      assert output =~ "custom_call"
      assert output =~ "fallback"
    end

    test "prints empty message when no dispatches" do
      AutoTuneProfiler.start()
      output = capture_io(fn -> AutoTuneProfiler.report() end)
      assert output =~ "No dispatches recorded"
    end
  end

  describe "validate_output/3 in defn" do
    defmodule Fallbacks do
      def double(x, _y), do: Nx.multiply(x, 2.0)
    end

    defn validate_matching(x) do
      # "fused" output is just x * 2
      fused = Nx.multiply(x, 2.0)

      AutoTuneProfiler.validate_output(
        fused,
        {x, x},
        kernel: :test_match,
        fallback: &Fallbacks.double/2
      )
    end

    test "passes silently when outputs match" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      log =
        capture_log(fn ->
          result = validate_matching(x)
          expected = Nx.tensor([2.0, 4.0, 6.0])
          assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
        end)

      refute log =~ "mismatch"
    end

    defmodule WrongFallback do
      def wrong(x, _y), do: Nx.multiply(x, 3.0)
    end

    defn validate_mismatch(x) do
      fused = Nx.multiply(x, 2.0)

      AutoTuneProfiler.validate_output(
        fused,
        {x, x},
        kernel: :test_mismatch,
        fallback: &WrongFallback.wrong/2,
        atol: 1.0e-6
      )
    end

    test "warns when outputs diverge" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      log =
        capture_log(fn ->
          result = validate_mismatch(x)
          # Returns fused output regardless
          expected = Nx.tensor([2.0, 4.0, 6.0])
          assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
        end)

      assert log =~ "test_mismatch"
      assert log =~ "mismatch"
      assert log =~ "max_diff"
    end
  end
end
