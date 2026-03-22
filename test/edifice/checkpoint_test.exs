defmodule Edifice.CheckpointTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog

  alias Edifice.Checkpoint

  @params %{
    "dense_0" => %{
      "kernel" => Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
      "bias" => Nx.tensor([0.1, 0.2, 0.3])
    },
    "dense_1" => %{
      "kernel" => Nx.tensor([[0.5, -0.5], [1.0, -1.0], [0.0, 0.0]]),
      "bias" => Nx.tensor([0.0, 0.0])
    }
  }

  setup do
    previous_level = Logger.level()
    Logger.configure(level: :debug)

    tmp_dir = Path.join(System.tmp_dir!(), "edifice_ckpt_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(tmp_dir)

    on_exit(fn ->
      Logger.configure(level: previous_level)
      File.rm_rf!(tmp_dir)
    end)

    %{tmp_dir: tmp_dir}
  end

  describe "save/3 and load/2" do
    test "round-trip preserves parameter values", %{tmp_dir: dir} do
      path = Path.join(dir, "model.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path)
      end)

      loaded =
        capture_log(fn ->
          Checkpoint.load(path)
        end)
        |> then(fn _log ->
          Checkpoint.load(path)
        end)

      # Check all tensors match
      assert Nx.all_close(loaded["dense_0"]["kernel"], @params["dense_0"]["kernel"])
             |> Nx.to_number() == 1

      assert Nx.all_close(loaded["dense_0"]["bias"], @params["dense_0"]["bias"])
             |> Nx.to_number() == 1

      assert Nx.all_close(loaded["dense_1"]["kernel"], @params["dense_1"]["kernel"])
             |> Nx.to_number() == 1
    end

    test "preserves tensor types", %{tmp_dir: dir} do
      bf16_params = %{
        "w" => Nx.as_type(Nx.tensor([1.0, 2.0, 3.0]), {:bf, 16})
      }

      path = Path.join(dir, "bf16.nx")

      capture_log(fn ->
        Checkpoint.save(bf16_params, path)
        loaded = Checkpoint.load(path)
        assert Nx.type(loaded["w"]) == {:bf, 16}
      end)
    end

    test "preserves tensor shapes", %{tmp_dir: dir} do
      path = Path.join(dir, "shapes.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path)
        loaded = Checkpoint.load(path)
        assert Nx.shape(loaded["dense_0"]["kernel"]) == {2, 3}
        assert Nx.shape(loaded["dense_1"]["bias"]) == {2}
      end)
    end

    test "creates parent directories", %{tmp_dir: dir} do
      path = Path.join([dir, "nested", "deep", "model.nx"])

      capture_log(fn ->
        Checkpoint.save(@params, path)
        assert File.exists?(path)
      end)
    end
  end

  describe "metadata" do
    test "save and load with metadata", %{tmp_dir: dir} do
      path = Path.join(dir, "meta.nx")
      metadata = %{epoch: 5, loss: 0.042, architecture: :decoder_only}

      capture_log(fn ->
        Checkpoint.save(@params, path, metadata: metadata)
        {loaded_params, loaded_meta} = Checkpoint.load(path, return_metadata: true)

        assert Nx.all_close(loaded_params["dense_0"]["kernel"], @params["dense_0"]["kernel"])
               |> Nx.to_number() == 1

        assert loaded_meta.epoch == 5
        assert loaded_meta.loss == 0.042
        assert loaded_meta.architecture == :decoder_only
      end)
    end

    test "load without return_metadata discards it", %{tmp_dir: dir} do
      path = Path.join(dir, "meta2.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path, metadata: %{epoch: 1})
        loaded = Checkpoint.load(path)

        # Returns just params, not tuple
        assert is_map(loaded)
        assert Map.has_key?(loaded, "dense_0")
      end)
    end

    test "load without metadata returns empty metadata map", %{tmp_dir: dir} do
      path = Path.join(dir, "nometa.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path)
        {loaded, meta} = Checkpoint.load(path, return_metadata: true)

        assert is_map(loaded)
        assert meta == %{}
      end)
    end
  end

  describe "compression" do
    test "compressed checkpoint is smaller", %{tmp_dir: dir} do
      path_plain = Path.join(dir, "plain.nx")
      path_compressed = Path.join(dir, "compressed.nx")

      # Use larger tensors so compression has something to work with
      big_params = %{"w" => Nx.broadcast(0.0, {100, 100})}

      capture_log(fn ->
        Checkpoint.save(big_params, path_plain)
        Checkpoint.save(big_params, path_compressed, compressed: 6)
      end)

      plain_size = File.stat!(path_plain).size
      compressed_size = File.stat!(path_compressed).size

      assert compressed_size < plain_size
    end

    test "compressed round-trip preserves values", %{tmp_dir: dir} do
      path = Path.join(dir, "compressed.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path, compressed: 6)
        loaded = Checkpoint.load(path)

        assert Nx.all_close(loaded["dense_0"]["kernel"], @params["dense_0"]["kernel"])
               |> Nx.to_number() == 1
      end)
    end
  end

  describe "FP8 quantized params" do
    test "saves and loads dequantized FP8 parameters", %{tmp_dir: dir} do
      q_params = Edifice.Quantization.FP8.quantize(@params)
      deq_params = Edifice.Quantization.FP8.dequantize(q_params)
      path = Path.join(dir, "fp8_deq.nx")

      capture_log(fn ->
        Checkpoint.save(deq_params, path)
        loaded = Checkpoint.load(path)

        assert Nx.all_close(loaded["dense_0"]["kernel"], deq_params["dense_0"]["kernel"])
               |> Nx.to_number() == 1
      end)
    end
  end

  describe "safetensors format" do
    test "save and load with format: :safetensors", %{tmp_dir: dir} do
      path = Path.join(dir, "model.safetensors")

      capture_log(fn ->
        Checkpoint.save(@params, path, format: :safetensors)
        loaded = Checkpoint.load(path)

        assert Nx.all_close(loaded["dense_0"]["kernel"], @params["dense_0"]["kernel"])
               |> Nx.to_number() == 1

        assert Nx.all_close(loaded["dense_1"]["bias"], @params["dense_1"]["bias"])
               |> Nx.to_number() == 1
      end)
    end

    test "auto-detects safetensors from extension", %{tmp_dir: dir} do
      path = Path.join(dir, "auto.safetensors")

      capture_log(fn ->
        Checkpoint.save(@params, path, format: :safetensors)
        # load auto-detects from .safetensors extension
        loaded = Checkpoint.load(path)

        assert is_map(loaded)
        assert Map.has_key?(loaded, "dense_0")
      end)
    end

    test "preserves nested structure via dot keys", %{tmp_dir: dir} do
      path = Path.join(dir, "nested.safetensors")

      capture_log(fn ->
        Checkpoint.save(@params, path, format: :safetensors)
        loaded = Checkpoint.load(path)

        # Should reconstruct nested map from dot-separated keys
        assert is_map(loaded["dense_0"])
        assert is_struct(loaded["dense_0"]["kernel"], Nx.Tensor)
        assert Nx.shape(loaded["dense_0"]["kernel"]) == {2, 3}
      end)
    end

    test "export_safetensors and import_safetensors directly", %{tmp_dir: dir} do
      path = Path.join(dir, "direct.safetensors")

      capture_log(fn ->
        Checkpoint.export_safetensors(@params, path)
        loaded = Checkpoint.import_safetensors(path)

        assert Nx.all_close(loaded["dense_0"]["kernel"], @params["dense_0"]["kernel"])
               |> Nx.to_number() == 1
      end)
    end
  end
end
