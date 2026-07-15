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

  # The motivating bug: SSM shape params (e.g. non-default state_size) not
  # threaded through checkpoint load, so inference silently rebuilt with
  # DEFAULT shapes — garbage outputs, no error. The manifest tests pin the fix.

  @mamba_opts [embed_dim: 8, hidden_size: 8, state_size: 4, num_layers: 1, window_size: 4]

  defp build_and_init(arch, opts) do
    {model, spec} = Edifice.build_with_spec(arch, opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({1, opts[:window_size], opts[:embed_dim]}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    {model, spec, params, predict_fn}
  end

  # capture_log swallows the return value; this variant keeps it
  defp capture_log_result(fun) do
    parent = self()
    capture_log(fn -> send(parent, {:result, fun.()}) end)

    receive do
      {:result, result} -> result
    end
  end

  describe "model manifest (Edifice.Spec)" do
    test "round-trip preserves non-default build opts via fetch_spec", %{tmp_dir: dir} do
      path = Path.join(dir, "spec_roundtrip.nx")
      {_model, spec, params, _predict} = build_and_init(:mamba, @mamba_opts)

      capture_log(fn -> Checkpoint.save(params, path, spec: spec) end)

      assert {:ok, restored} = Checkpoint.fetch_spec(path)
      assert restored.arch == :mamba
      assert restored.build_opts[:state_size] == 4
      assert restored.build_opts[:num_layers] == 1
    end

    test "the mamba scenario: load_model rebuilds with non-default shapes, no opts passed",
         %{tmp_dir: dir} do
      path = Path.join(dir, "mamba_s4.nx")
      {_model, spec, params, predict_fn} = build_and_init(:mamba, @mamba_opts)

      capture_log(fn -> Checkpoint.save(params, path, spec: spec) end)

      {model2, params2} = capture_log_result(fn -> Checkpoint.load_model(path) end)

      assert %Axon{} = model2

      # Same params + same input through the rebuilt model must reproduce the
      # original model's outputs exactly (deterministic forward, atol 1e-6)
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {1, 4, 8})

      {_init2, predict2} = Axon.build(model2)
      out1 = predict_fn.(params, input)
      out2 = predict2.(Axon.ModelState.new(params2), input)

      assert Nx.all_close(out1, out2, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "shape mismatch raises loudly with both shapes named", %{tmp_dir: dir} do
      path = Path.join(dir, "mamba_tampered.nx")
      {_model, _spec, params, _predict} = build_and_init(:mamba, @mamba_opts)

      # Tamper: a spec claiming the DEFAULT state_size (16) while the stored
      # params were built with state_size 4 — exactly the silent-garbage case
      lying_spec =
        Edifice.Spec.new(:mamba, Keyword.put(@mamba_opts, :state_size, 16))

      capture_log(fn -> Checkpoint.save(params, path, spec: lying_spec) end)

      err =
        assert_raise RuntimeError, fn ->
          capture_log(fn -> Checkpoint.load_model(path) end)
        end

      assert err.message =~ "Mismatched shapes"
      assert err.message =~ "state_size: 16"
      assert err.message =~ "expected"
      assert err.message =~ "stored"
    end

    test "validate: false skips shape validation", %{tmp_dir: dir} do
      path = Path.join(dir, "mamba_novalidate.nx")
      {_model, _spec, params, _predict} = build_and_init(:mamba, @mamba_opts)

      lying_spec = Edifice.Spec.new(:mamba, Keyword.put(@mamba_opts, :state_size, 16))
      capture_log(fn -> Checkpoint.save(params, path, spec: lying_spec) end)

      {model, loaded} =
        capture_log_result(fn -> Checkpoint.load_model(path, validate: false) end)

      assert %Axon{} = model
      assert is_map(loaded)
    end

    test "spec-less checkpoint warns once per VM on load", %{tmp_dir: dir} do
      path = Path.join(dir, "legacy.nx")
      capture_log(fn -> Checkpoint.save(@params, path) end)

      Checkpoint.reset_missing_spec_warning()

      first = capture_log(fn -> Checkpoint.load(path) end)
      assert first =~ "no embedded Edifice.Spec"

      second = capture_log(fn -> Checkpoint.load(path) end)
      refute second =~ "no embedded Edifice.Spec"
    end

    test "load_model on a spec-less checkpoint raises descriptively", %{tmp_dir: dir} do
      path = Path.join(dir, "legacy2.nx")
      capture_log(fn -> Checkpoint.save(@params, path) end)

      assert_raise RuntimeError, ~r/no embedded Edifice.Spec/, fn ->
        capture_log(fn -> Checkpoint.load_model(path) end)
      end

      assert {:error, :missing} = Checkpoint.fetch_spec(path)
    end

    test "corrupted spec: load_model raises, plain load still returns params",
         %{tmp_dir: dir} do
      path = Path.join(dir, "corrupt_spec.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path, metadata: %{"__edifice_spec__" => %{"nonsense" => true}})
      end)

      assert {:error, {:invalid, _reason}} = Checkpoint.fetch_spec(path)

      assert_raise RuntimeError, ~r/invalid embedded Edifice.Spec/, fn ->
        capture_log(fn -> Checkpoint.load_model(path) end)
      end

      capture_log(fn ->
        loaded = Checkpoint.load(path)
        assert Map.has_key?(loaded, "dense_0")
      end)
    end

    test "spec coexists with user metadata", %{tmp_dir: dir} do
      path = Path.join(dir, "spec_meta.nx")
      spec = Edifice.Spec.new(:mlp, input_size: 4)

      capture_log(fn ->
        Checkpoint.save(@params, path, spec: spec, metadata: %{epoch: 3})
        {_params, meta} = Checkpoint.load(path, return_metadata: true)

        assert meta.epoch == 3
        assert Map.has_key?(meta, "__edifice_spec__")
      end)
    end

    test "reserved metadata key collision raises", %{tmp_dir: dir} do
      path = Path.join(dir, "collision.nx")
      spec = Edifice.Spec.new(:mlp, input_size: 4)

      assert_raise ArgumentError, ~r/reserved key/, fn ->
        Checkpoint.save(@params, path,
          spec: spec,
          metadata: %{"__edifice_spec__" => %{}}
        )
      end
    end

    test "old checkpoints load exactly as before (backward compat)", %{tmp_dir: dir} do
      path = Path.join(dir, "compat.nx")

      capture_log(fn ->
        Checkpoint.save(@params, path, metadata: %{epoch: 1})
        {loaded, meta} = Checkpoint.load(path, return_metadata: true)

        assert Nx.all_close(loaded["dense_0"]["kernel"], @params["dense_0"]["kernel"])
               |> Nx.to_number() == 1

        assert meta == %{epoch: 1}
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
