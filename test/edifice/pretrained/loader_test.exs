defmodule Edifice.PretrainedTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  # A simple test key map that maps PyTorch-style keys to Axon-style keys
  defmodule TestKeyMap do
    @behaviour Edifice.Pretrained.KeyMap

    @impl true
    def map_key("layer.0.weight"), do: "block_0.dense.kernel"
    def map_key("layer.0.bias"), do: "block_0.dense.bias"
    def map_key("layer.1.weight"), do: "block_1.dense.kernel"
    def map_key("norm.weight"), do: "norm.scale"
    def map_key("cls_token"), do: :skip
    def map_key(_), do: :unmapped

    @impl true
    def tensor_transforms do
      [
        {~r/\.kernel$/, &Edifice.Pretrained.Transform.transpose_linear/1}
      ]
    end
  end

  # A key map that maps everything (no unmapped keys possible)
  defmodule LenientKeyMap do
    @behaviour Edifice.Pretrained.KeyMap

    @impl true
    def map_key("known_key"), do: "mapped.key"
    def map_key(_unknown), do: :skip

    @impl true
    def tensor_transforms, do: []
  end

  setup do
    # Verify safetensors is available
    assert Code.ensure_loaded?(Safetensors),
           "safetensors package must be available for these tests"

    :ok
  end

  defp write_fixture(tensors, context \\ %{}) do
    path =
      Path.join(
        System.tmp_dir!(),
        "edifice_test_#{System.unique_integer([:positive])}.safetensors"
      )

    Safetensors.write!(path, tensors)

    if Map.has_key?(context, :registered?) do
      ExUnit.Callbacks.on_exit(fn -> File.rm(path) end)
    end

    path
  end

  describe "load/3" do
    test "loads and maps checkpoint keys to Axon paths" do
      tensors = %{
        "layer.0.weight" => Nx.iota({4, 3}, type: :f32),
        "layer.0.bias" => Nx.tensor([1.0, 2.0, 3.0, 4.0]),
        "norm.weight" => Nx.tensor([1.0, 1.0, 1.0]),
        "cls_token" => Nx.tensor([0.0])
      }

      path = write_fixture(tensors)

      model_state = Edifice.Pretrained.load(TestKeyMap, path, strict: false)

      assert %Axon.ModelState{} = model_state

      # Verify nested structure
      data = model_state.data
      assert is_map(data["block_0"])
      assert is_map(data["block_0"]["dense"])
      assert %Nx.Tensor{} = data["block_0"]["dense"]["kernel"]
      assert %Nx.Tensor{} = data["norm"]["scale"]

      File.rm(path)
    end

    test "applies tensor transforms (transpose) to mapped keys" do
      weight = Nx.iota({4, 3}, type: :f32)

      tensors = %{
        "layer.0.weight" => weight,
        "layer.0.bias" => Nx.tensor([1.0, 2.0, 3.0, 4.0]),
        "norm.weight" => Nx.tensor([1.0])
      }

      path = write_fixture(tensors)

      model_state = Edifice.Pretrained.load(TestKeyMap, path, strict: false)
      kernel = model_state.data["block_0"]["dense"]["kernel"]

      # Weight was [4, 3] in checkpoint, should be [3, 4] after transpose
      assert Nx.shape(kernel) == {3, 4}

      File.rm(path)
    end

    test "casts tensors when :dtype option is provided" do
      tensors = %{
        "layer.0.weight" => Nx.iota({4, 3}, type: :f32),
        "layer.0.bias" => Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32),
        "norm.weight" => Nx.tensor([1.0], type: :f32)
      }

      path = write_fixture(tensors)

      model_state = Edifice.Pretrained.load(TestKeyMap, path, dtype: :bf16, strict: false)
      kernel = model_state.data["block_0"]["dense"]["kernel"]
      assert Nx.type(kernel) == {:bf, 16}

      bias = model_state.data["block_0"]["dense"]["bias"]
      assert Nx.type(bias) == {:bf, 16}

      File.rm(path)
    end

    test "raises on unmapped keys in strict mode (default)" do
      tensors = %{
        "layer.0.weight" => Nx.iota({4, 3}, type: :f32),
        "unknown_key" => Nx.tensor([1.0])
      }

      path = write_fixture(tensors)

      assert_raise ArgumentError, ~r/unmapped key/, fn ->
        Edifice.Pretrained.load(TestKeyMap, path)
      end

      File.rm(path)
    end

    test "logs warnings for unmapped keys in lenient mode" do
      tensors = %{
        "layer.0.weight" => Nx.iota({4, 3}, type: :f32),
        "unknown_key" => Nx.tensor([1.0])
      }

      path = write_fixture(tensors)

      log =
        capture_log(fn ->
          Edifice.Pretrained.load(TestKeyMap, path, strict: false)
        end)

      assert log =~ "unmapped checkpoint key"
      assert log =~ "unknown_key"

      File.rm(path)
    end

    test "excludes :skip keys from result" do
      tensors = %{
        "layer.0.weight" => Nx.iota({4, 3}, type: :f32),
        "layer.0.bias" => Nx.tensor([1.0, 2.0, 3.0, 4.0]),
        "norm.weight" => Nx.tensor([1.0]),
        "cls_token" => Nx.tensor([0.0, 0.0, 0.0])
      }

      path = write_fixture(tensors)

      model_state = Edifice.Pretrained.load(TestKeyMap, path, strict: false)
      flat = Edifice.Pretrained.Transform.flatten_params(model_state)

      # cls_token was :skip, should not appear
      refute Map.has_key?(flat, "cls_token")
      refute Map.has_key?(flat, "cls_token.kernel")

      File.rm(path)
    end

    test "returns a valid Axon.ModelState struct" do
      tensors = %{
        "known_key" => Nx.tensor([1.0, 2.0, 3.0])
      }

      path = write_fixture(tensors)

      model_state = Edifice.Pretrained.load(LenientKeyMap, path)
      assert %Axon.ModelState{} = model_state
      assert is_map(model_state.data)

      File.rm(path)
    end
  end

  describe "list_keys/1" do
    test "returns sorted key names from a safetensors file" do
      tensors = %{
        "zebra.weight" => Nx.tensor([1.0]),
        "alpha.bias" => Nx.tensor([2.0]),
        "middle.scale" => Nx.tensor([3.0])
      }

      path = write_fixture(tensors)

      keys = Edifice.Pretrained.list_keys(path)

      assert keys == ["alpha.bias", "middle.scale", "zebra.weight"]

      File.rm(path)
    end
  end
end
