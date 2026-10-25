defmodule Edifice.Training.MemoryTrackerTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  import ExUnit.CaptureLog

  alias Edifice.Training.MemoryTracker

  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)
    on_exit(fn -> Logger.configure(level: previous_level) end)
  end

  describe "snapshot/1" do
    test "returns memory stats map" do
      stats = MemoryTracker.snapshot(client: :host)

      assert is_map(stats)
      assert is_integer(stats.allocated)
      assert is_integer(stats.peak)
      assert is_map(stats.per_device)
      assert is_integer(stats.timestamp)
      assert stats.allocated >= 0
      assert stats.peak >= 0
    end
  end

  describe "reset_peak/1" do
    test "resets peak counter" do
      assert MemoryTracker.reset_peak(client: :host) == :ok
      stats = MemoryTracker.snapshot(client: :host)
      # After reset, peak should be <= allocated (or 0)
      assert stats.peak >= 0
    end
  end

  describe "measure/2" do
    test "measures peak memory of a computation" do
      log =
        capture_log(fn ->
          {result, mem} =
            MemoryTracker.measure(
              fn ->
                # Allocate a tensor on the EXLA host backend
                Nx.Defn.jit(fn -> Nx.iota({100, 100}, type: :f32) end,
                  compiler: EXLA,
                  client: :host
                ).()
              end,
              client: :host,
              label: "test_compute"
            )

          assert is_struct(result, Nx.Tensor)
          assert is_integer(mem.peak)
          assert is_integer(mem.delta)
          assert is_integer(mem.before_allocated)
          assert is_integer(mem.after_allocated)
        end)

      assert log =~ "[MemoryTracker] test_compute:"
      assert log =~ "peak="
      assert log =~ "delta="
    end

    test "suppresses log when log: false" do
      log =
        capture_log(fn ->
          MemoryTracker.measure(fn -> :ok end, client: :host, log: false)
        end)

      refute log =~ "[MemoryTracker]"
    end
  end

  describe "report/1" do
    test "prints formatted memory report" do
      output =
        capture_io(fn ->
          stats = MemoryTracker.report(client: :host)
          assert is_map(stats)
        end)

      assert output =~ "[MemoryTracker] GPU Memory Report"
      assert output =~ "Allocated:"
      assert output =~ "Peak:"
    end
  end

  describe "compare/3" do
    test "compares memory between two computations" do
      fun_a = fn ->
        Nx.Defn.jit(fn -> Nx.iota({50, 50}, type: :f32) end, compiler: EXLA, client: :host).()
      end

      fun_b = fn ->
        Nx.Defn.jit(fn -> Nx.iota({10, 10}, type: :f32) end, compiler: EXLA, client: :host).()
      end

      output =
        capture_io(fn ->
          result = MemoryTracker.compare({"large", fun_a}, {"small", fun_b}, client: :host)

          assert is_map(result)
          assert result.a.label == "large"
          assert result.b.label == "small"
          assert is_float(result.ratio)
        end)

      assert output =~ "[MemoryTracker] Comparison"
      assert output =~ "large"
      assert output =~ "small"
      assert output =~ "Ratio:"
    end
  end
end
