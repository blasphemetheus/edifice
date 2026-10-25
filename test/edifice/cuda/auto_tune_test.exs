defmodule Edifice.CUDA.AutoTuneTest do
  use ExUnit.Case, async: false

  alias Edifice.CUDA.AutoTune

  setup do
    # Clean up persistent_term cache and process dict before each test
    AutoTune.clear_cache()
    Process.delete(:__edifice_benchmarking__)
    Process.delete(:__edifice_force_fallback__)

    # Clean up env vars
    System.delete_env("EDIFICE_DISABLE_FUSED")
    System.delete_env("EDIFICE_AUTOTUNE")

    on_exit(fn ->
      AutoTune.clear_cache()
      System.delete_env("EDIFICE_DISABLE_FUSED")
      System.delete_env("EDIFICE_AUTOTUNE")
      Process.delete(:__edifice_benchmarking__)
      Process.delete(:__edifice_force_fallback__)
    end)

    :ok
  end

  describe "use_fused?/2 env var overrides" do
    test "EDIFICE_DISABLE_FUSED=1 always returns false" do
      System.put_env("EDIFICE_DISABLE_FUSED", "1")
      tensor = Nx.iota({2, 4, 8})
      refute AutoTune.use_fused?(:mingru, tensor)
    end

    test "EDIFICE_AUTOTUNE=fallback always returns false" do
      System.put_env("EDIFICE_AUTOTUNE", "fallback")
      tensor = Nx.iota({2, 4, 8})
      refute AutoTune.use_fused?(:mingru, tensor)
    end

    test "EDIFICE_AUTOTUNE=0 defers to custom_call_available?" do
      System.put_env("EDIFICE_AUTOTUNE", "0")
      tensor = Nx.iota({2, 4, 8})
      # Result depends on whether EXLA custom call is available
      # Without EXLA, custom_call_available? returns false
      result = AutoTune.use_fused?(:mingru, tensor)
      expected = Edifice.CUDA.FusedScan.custom_call_available?()
      assert result == expected
    end
  end

  describe "persistent_term cache" do
    test "returns cached :fused result" do
      :persistent_term.put({:edifice_autotune, :mingru, 64, {:f, 32}}, :fused)
      tensor = Nx.iota({2, 4, 64}, type: {:f, 32})

      # Only returns true if custom_call is also available
      if Edifice.CUDA.FusedScan.custom_call_available?() do
        assert AutoTune.use_fused?(:mingru, tensor) == true
      end
    end

    test "returns cached :fallback result" do
      :persistent_term.put({:edifice_autotune, :mingru, 64, {:f, 32}}, :fallback)
      tensor = Nx.iota({2, 4, 64}, type: {:f, 32})

      # If custom_call not available, returns false regardless
      # If custom_call available and cached as fallback, returns false
      refute AutoTune.use_fused?(:mingru, tensor)
    end
  end

  describe "extract_key_dim" do
    test "extracts last dim for standard 3D scan kernels" do
      # We test indirectly through the cache key used by use_fused?
      # Pre-populate cache, then check that use_fused? finds it
      :persistent_term.put({:edifice_autotune, :mingru, 128, {:f, 32}}, :fallback)
      tensor = Nx.iota({2, 4, 128}, type: {:f, 32})
      # Should find the cached entry for dim=128
      refute AutoTune.use_fused?(:mingru, tensor)
    end

    test "extracts hidden from 4H for slstm_scan" do
      # slstm_scan tensor is {B, T, 4*H}, key dim should be H
      :persistent_term.put({:edifice_autotune, :slstm_scan, 32, {:f, 32}}, :fallback)
      # 4*32 = 128
      tensor = Nx.iota({2, 4, 128}, type: {:f, 32})
      refute AutoTune.use_fused?(:slstm_scan, tensor)
    end

    test "extracts hidden from 3H for gru_scan" do
      :persistent_term.put({:edifice_autotune, :gru_scan, 32, {:f, 32}}, :fallback)
      # 3*32 = 96
      tensor = Nx.iota({2, 4, 96}, type: {:f, 32})
      refute AutoTune.use_fused?(:gru_scan, tensor)
    end

    test "extracts memory_size from 4M for titans_scan" do
      :persistent_term.put({:edifice_autotune, :titans_scan, 16, {:f, 32}}, :fallback)
      # 4*16 = 64
      tensor = Nx.iota({2, 4, 64}, type: {:f, 32})
      refute AutoTune.use_fused?(:titans_scan, tensor)
    end

    test "extracts memory_size from 5M for miras_scan" do
      :persistent_term.put({:edifice_autotune, :miras_scan, 10, {:f, 32}}, :fallback)
      # 5*10 = 50
      tensor = Nx.iota({2, 4, 50}, type: {:f, 32})
      refute AutoTune.use_fused?(:miras_scan, tensor)
    end

    test "extracts head_dim for 4D attention kernels" do
      :persistent_term.put({:edifice_autotune, :flash_attention, 16, {:f, 32}}, :fallback)
      # head_dim = 16
      tensor = Nx.iota({2, 4, 8, 16}, type: {:f, 32})
      refute AutoTune.use_fused?(:flash_attention, tensor)
    end
  end

  describe "benchmarking guard" do
    test "defers to custom_call_available? during benchmarking" do
      Process.put(:__edifice_benchmarking__, true)
      tensor = Nx.iota({2, 4, 64})
      result = AutoTune.use_fused?(:mingru, tensor)
      expected = Edifice.CUDA.FusedScan.custom_call_available?()
      assert result == expected
    end

    test "force_fallback makes custom_call_available? return false" do
      Process.put(:__edifice_force_fallback__, true)
      refute Edifice.CUDA.FusedScan.custom_call_available?()
    end
  end

  describe "disk cache" do
    @tag :tmp_dir
    test "save and load round-trip", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "autotune.json")

      # Populate cache
      :persistent_term.put({:edifice_autotune, :mingru, 64, {:f, 32}}, :fused)
      :persistent_term.put({:edifice_autotune, :gru_scan, 64, {:f, 32}}, :fallback)

      # Save
      assert :ok = AutoTune.save_disk_cache(path)
      assert File.exists?(path)

      # Clear and verify empty
      AutoTune.clear_cache()

      assert :persistent_term.get({:edifice_autotune, :mingru, 64, {:f, 32}}, :not_cached) ==
               :not_cached

      # Load
      assert {:ok, 2} = AutoTune.load_disk_cache(path)

      assert :persistent_term.get({:edifice_autotune, :mingru, 64, {:f, 32}}, :not_cached) ==
               :fused

      assert :persistent_term.get({:edifice_autotune, :gru_scan, 64, {:f, 32}}, :not_cached) ==
               :fallback
    end

    @tag :tmp_dir
    test "load rejects cache from different GPU", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "autotune.json")

      # Write a cache with a fake GPU name
      data = %{
        "gpu" => "Fake GPU That Doesn't Exist",
        "results" => %{"mingru:64:f32" => "fused"}
      }

      File.write!(path, JSON.encode!(data) |> IO.iodata_to_binary())

      assert {:error, :gpu_mismatch} = AutoTune.load_disk_cache(path)
    end

    test "load returns not_found for missing file" do
      assert {:error, :not_found} = AutoTune.load_disk_cache("/tmp/nonexistent_autotune.json")
    end
  end

  describe "clear_cache/0" do
    test "removes all autotune entries from persistent_term" do
      :persistent_term.put({:edifice_autotune, :mingru, 64, {:f, 32}}, :fused)
      :persistent_term.put({:edifice_autotune, :minlstm, 64, {:f, 32}}, :fallback)

      AutoTune.clear_cache()

      assert :persistent_term.get({:edifice_autotune, :mingru, 64, {:f, 32}}, :not_cached) ==
               :not_cached

      assert :persistent_term.get({:edifice_autotune, :minlstm, 64, {:f, 32}}, :not_cached) ==
               :not_cached
    end
  end

  describe "report/0" do
    test "prints table when cache has entries" do
      :persistent_term.put({:edifice_autotune, :mingru, 64, {:f, 32}}, :fused)
      :persistent_term.put({:edifice_autotune, :gru_scan, 64, {:f, 32}}, :fallback)

      output = ExUnit.CaptureIO.capture_io(fn -> AutoTune.report() end)
      assert output =~ "Cached Results"
      assert output =~ "mingru"
      assert output =~ "gru_scan"
      assert output =~ "fused"
      assert output =~ "fallback"
    end

    test "prints message when cache is empty" do
      output = ExUnit.CaptureIO.capture_io(fn -> AutoTune.report() end)
      assert output =~ "No cached results"
    end
  end

  describe "all_kernels/0" do
    test "returns 29 kernel names" do
      kernels = AutoTune.all_kernels()
      assert length(kernels) == 29
      assert :mingru in kernels
      assert :flash_attention in kernels
      assert :lstm_block in kernels
    end
  end
end
