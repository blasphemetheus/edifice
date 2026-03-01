defmodule Edifice.Pretrained.HubTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.Hub

  describe "cache_path/2" do
    test "returns default cache path for repo" do
      path = Hub.cache_path("google/vit-base-patch16-224")
      assert path == Path.expand("~/.cache/edifice/google/vit-base-patch16-224")
    end

    test "respects :cache_dir option" do
      path = Hub.cache_path("google/vit-base-patch16-224", cache_dir: "/tmp/models")
      assert path == "/tmp/models/google/vit-base-patch16-224"
    end

    test "handles nested org/model paths" do
      path = Hub.cache_path("bigscience/bloom")
      assert String.ends_with?(path, "bigscience/bloom")
    end
  end

  describe "parse_shard_index/1" do
    test "extracts unique sorted filenames from weight_map" do
      json = Jason.encode!(%{
        "metadata" => %{"total_size" => 100},
        "weight_map" => %{
          "layer.0.weight" => "model-00002-of-00003.safetensors",
          "layer.0.bias" => "model-00002-of-00003.safetensors",
          "layer.1.weight" => "model-00001-of-00003.safetensors",
          "embed.weight" => "model-00003-of-00003.safetensors"
        }
      })

      filenames = Hub.parse_shard_index(json)

      assert filenames == [
               "model-00001-of-00003.safetensors",
               "model-00002-of-00003.safetensors",
               "model-00003-of-00003.safetensors"
             ]
    end

    test "deduplicates filenames" do
      json = Jason.encode!(%{
        "weight_map" => %{
          "a" => "shard-1.safetensors",
          "b" => "shard-1.safetensors",
          "c" => "shard-2.safetensors"
        }
      })

      filenames = Hub.parse_shard_index(json)
      assert filenames == ["shard-1.safetensors", "shard-2.safetensors"]
    end
  end

  describe "download/2 caching" do
    test "detects cached files via file existence" do
      # Create a temp dir simulating a cached model
      tmp = Path.join(System.tmp_dir!(), "edifice_hub_test_#{System.unique_integer([:positive])}")
      repo_dir = Path.join(tmp, "test-org/test-model")
      File.mkdir_p!(repo_dir)

      cached_file = Path.join(repo_dir, "model.safetensors")
      File.write!(cached_file, "fake safetensors data")

      assert File.exists?(cached_file)

      # Clean up
      File.rm_rf!(tmp)
    end
  end

  describe "download/2 integration" do
    @describetag :external
    @describetag timeout: 120_000

    test "downloads tiny model from HuggingFace Hub" do
      tmp = Path.join(System.tmp_dir!(), "edifice_hub_int_#{System.unique_integer([:positive])}")

      assert {:ok, [path]} =
               Hub.download("hf-internal-testing/tiny-random-vit",
                 cache_dir: tmp
               )

      assert File.exists?(path)
      assert String.ends_with?(path, ".safetensors")

      # Verify it's a valid safetensors file (should be loadable)
      assert Code.ensure_loaded?(Safetensors)
      checkpoint = Safetensors.read!(path)
      assert is_map(checkpoint)
      assert map_size(checkpoint) > 0

      # Second call should hit cache (no re-download)
      assert {:ok, [^path]} =
               Hub.download("hf-internal-testing/tiny-random-vit",
                 cache_dir: tmp
               )

      File.rm_rf!(tmp)
    end
  end
end
