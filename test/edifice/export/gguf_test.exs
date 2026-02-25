defmodule Edifice.Export.GGUFTest do
  use ExUnit.Case, async: true

  alias Edifice.Export.GGUF

  @moduletag :gguf

  # GGUF magic number: 'GGUF' in little-endian
  @magic 0x46554747
  @version 3

  describe "export/4" do
    test "exports a tiny model to GGUF file with correct magic bytes" do
      # Create minimal model params
      params = create_tiny_model_params()

      config = %{
        num_layers: 2,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        context_length: 512,
        vocab_size: 1000,
        name: "test_model"
      }

      output_path = Path.join(System.tmp_dir!(), "test_model_#{:rand.uniform(100_000)}.gguf")

      try do
        assert :ok = GGUF.export(params, config, output_path, quantization: :f32)

        # Verify file exists
        assert File.exists?(output_path)

        # Read and verify magic bytes and version
        {:ok, file} = File.open(output_path, [:read, :binary])
        header = IO.binread(file, 16)
        File.close(file)

        <<magic::little-32, version::little-32, tensor_count::little-64>> = header

        assert magic == @magic
        assert version == @version
        assert tensor_count > 0
      after
        File.rm(output_path)
      end
    end

    test "exports with Q8_0 quantization" do
      params = create_tiny_model_params()

      config = %{
        num_layers: 2,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2
      }

      output_path = Path.join(System.tmp_dir!(), "test_q8_#{:rand.uniform(100_000)}.gguf")

      try do
        assert :ok = GGUF.export(params, config, output_path, quantization: :q8_0)
        assert File.exists?(output_path)

        # Q8_0 file should be smaller than F32
        q8_size = File.stat!(output_path).size

        f32_path = Path.join(System.tmp_dir!(), "test_f32_#{:rand.uniform(100_000)}.gguf")
        GGUF.export(params, config, f32_path, quantization: :f32)
        f32_size = File.stat!(f32_path).size
        File.rm(f32_path)

        # Q8_0 should be smaller (roughly 1/4 size for data, plus overhead)
        assert q8_size < f32_size
      after
        File.rm(output_path)
      end
    end

    test "exports with F16 quantization" do
      params = create_tiny_model_params()

      config = %{
        num_layers: 1,
        hidden_size: 32,
        num_heads: 2,
        num_kv_heads: 2
      }

      output_path = Path.join(System.tmp_dir!(), "test_f16_#{:rand.uniform(100_000)}.gguf")

      try do
        assert :ok = GGUF.export(params, config, output_path, quantization: :f16)
        assert File.exists?(output_path)

        # Verify header
        {:ok, file} = File.open(output_path, [:read, :binary])
        <<magic::little-32, version::little-32, _::binary>> = IO.binread(file, 16)
        File.close(file)

        assert magic == @magic
        assert version == @version
      after
        File.rm(output_path)
      end
    end
  end

  describe "encode_metadata/2" do
    test "encodes required metadata fields" do
      config = %{
        hidden_size: 256,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 6,
        context_length: 2048,
        vocab_size: 32_000,
        name: "test_model"
      }

      binary = GGUF.encode_metadata(config, architecture: "llama")

      # Should be non-empty binary
      assert is_binary(binary)
      assert byte_size(binary) > 0

      # Should contain the architecture string
      assert binary =~ "general.architecture"
      assert binary =~ "llama"
      assert binary =~ "test_model"
    end

    test "uses default values when config is minimal" do
      config = %{}

      binary = GGUF.encode_metadata(config, [])

      assert is_binary(binary)
      assert byte_size(binary) > 0
      assert binary =~ "general.architecture"
    end
  end

  describe "encode_tensor_info/2" do
    test "encodes tensor info with correct shape" do
      tensor = Nx.iota({64, 128}, type: :f32)
      tensors = [{"test.weight", tensor}]

      {info_binary, tensor_infos} = GGUF.encode_tensor_info(tensors, :f32)

      assert is_binary(info_binary)
      assert length(tensor_infos) == 1

      [{name, shape, ggml_type, offset}] = tensor_infos
      assert name == "test.weight"
      assert shape == [64, 128]
      # F32
      assert ggml_type == 0
      assert offset == 0
    end

    test "encodes multiple tensors with correct offsets" do
      t1 = Nx.iota({32, 32}, type: :f32)
      t2 = Nx.iota({64, 64}, type: :f32)
      tensors = [{"tensor1.weight", t1}, {"tensor2.weight", t2}]

      {_info_binary, tensor_infos} = GGUF.encode_tensor_info(tensors, :f32)

      assert length(tensor_infos) == 2

      [{_, _, _, offset1}, {_, _, _, offset2}] = tensor_infos
      assert offset1 == 0
      # Second tensor offset should be after first tensor data + padding
      assert offset2 > 0
    end
  end

  describe "encode_tensors/2" do
    test "encodes F32 tensors correctly" do
      tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      tensors = [{"test.weight", tensor}]

      data = GGUF.encode_tensors(tensors, :f32)

      assert is_binary(data)
      # 4 floats * 4 bytes = 16 bytes, plus padding to 32
      assert byte_size(data) >= 16
    end

    test "encodes F16 tensors correctly" do
      tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
      tensors = [{"test.weight", tensor}]

      data = GGUF.encode_tensors(tensors, :f16)

      assert is_binary(data)
      # 4 floats * 2 bytes = 8 bytes, plus padding
      assert byte_size(data) >= 8
    end

    test "encodes Q8_0 tensors correctly" do
      # Create a tensor with 64 elements (2 blocks of 32)
      tensor = Nx.iota({64}, type: :f32)
      tensors = [{"test.weight", tensor}]

      data = GGUF.encode_tensors(tensors, :q8_0)

      assert is_binary(data)
      # 2 blocks * (2 bytes scale + 32 bytes data) = 68 bytes, plus padding
      assert byte_size(data) >= 68
    end
  end

  describe "quantize_q8_0/1" do
    test "quantizes a block of 32 values" do
      # Create exactly 32 values
      tensor = Nx.tensor(Enum.to_list(1..32), type: :f32)

      binary = GGUF.quantize_q8_0(tensor)

      # One block: 2 bytes (f16 scale) + 32 bytes (int8 values) = 34 bytes
      assert byte_size(binary) == 34
    end

    test "pads to multiple of 32" do
      # Create 40 values (needs padding to 64)
      tensor = Nx.iota({40}, type: :f32)

      binary = GGUF.quantize_q8_0(tensor)

      # 2 blocks * 34 bytes = 68 bytes
      assert byte_size(binary) == 68
    end

    test "handles all zeros" do
      tensor = Nx.broadcast(0.0, {32})

      binary = GGUF.quantize_q8_0(tensor)

      # Should not crash, produces valid output
      assert byte_size(binary) == 34
    end

    test "handles large values" do
      tensor = Nx.broadcast(1.0e6, {32})

      binary = GGUF.quantize_q8_0(tensor)

      assert byte_size(binary) == 34
    end
  end

  describe "map_param_names/2" do
    test "maps input projection to token embedding" do
      params = %{
        "input_projection" => %{
          "kernel" => Nx.iota({64, 128}, type: :f32)
        }
      }

      result = GGUF.map_param_names(params, 1)

      names = Enum.map(result, fn {name, _} -> name end)
      assert "token_embd.weight" in names
    end

    test "maps decoder block params to GGUF blk format" do
      params = %{
        "decoder_block_1_attn_q_proj" => %{
          "kernel" => Nx.iota({64, 64}, type: :f32)
        },
        "decoder_block_1_attn_k_proj" => %{
          "kernel" => Nx.iota({64, 32}, type: :f32)
        },
        "decoder_block_1_ffn_gate" => %{
          "kernel" => Nx.iota({64, 256}, type: :f32)
        }
      }

      result = GGUF.map_param_names(params, 1)

      names = Enum.map(result, fn {name, _} -> name end)
      assert "blk.0.attn_q.weight" in names
      assert "blk.0.attn_k.weight" in names
      assert "blk.0.ffn_gate.weight" in names
    end

    test "maps final norm correctly" do
      params = %{
        "final_norm" => %{
          "gamma" => Nx.iota({64}, type: :f32)
        }
      }

      result = GGUF.map_param_names(params, 1)

      names = Enum.map(result, fn {name, _} -> name end)
      assert "output_norm.weight" in names
    end

    test "handles nested Axon.ModelState structure" do
      params = %Axon.ModelState{
        data: %{
          "input_projection" => %{
            "kernel" => Nx.iota({32, 64}, type: :f32)
          }
        },
        state: %{},
        parameters: %{}
      }

      result = GGUF.map_param_names(params, 1)

      names = Enum.map(result, fn {name, _} -> name end)
      assert "token_embd.weight" in names
    end
  end

  describe "round-trip integration" do
    test "exports a complete tiny decoder model" do
      # Build an actual decoder_only model
      model =
        Edifice.build(:decoder_only,
          embed_dim: 32,
          hidden_size: 32,
          num_heads: 2,
          num_kv_heads: 1,
          num_layers: 2,
          window_size: 8
        )

      # Initialize params
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), Axon.ModelState.empty())

      config = %{
        num_layers: 2,
        hidden_size: 32,
        num_heads: 2,
        num_kv_heads: 1,
        context_length: 512,
        vocab_size: 1000,
        name: "tiny_decoder"
      }

      output_path = Path.join(System.tmp_dir!(), "decoder_#{:rand.uniform(100_000)}.gguf")

      try do
        # Export
        assert :ok = GGUF.export(params, config, output_path, quantization: :q8_0)

        # Verify the file
        assert File.exists?(output_path)

        {:ok, file} = File.open(output_path, [:read, :binary])
        header = IO.binread(file, 24)
        File.close(file)

        <<magic::little-32, version::little-32, tensor_count::little-64,
          metadata_count::little-64>> =
          header

        assert magic == @magic
        assert version == @version
        assert tensor_count > 0
        assert metadata_count > 0

        # The file should have a reasonable size
        file_size = File.stat!(output_path).size
        assert file_size > 1000
      after
        File.rm(output_path)
      end
    end
  end

  # Helper to create minimal model params for testing
  defp create_tiny_model_params do
    %{
      "input_projection" => %{
        "kernel" => Nx.iota({32, 64}, type: :f32),
        "bias" => Nx.iota({64}, type: :f32)
      },
      "decoder_block_1_attn_norm_gamma" => Nx.broadcast(1.0, {64}),
      "decoder_block_1_attn_q_proj" => %{
        "kernel" => Nx.iota({64, 64}, type: :f32)
      },
      "decoder_block_1_attn_k_proj" => %{
        "kernel" => Nx.iota({64, 32}, type: :f32)
      },
      "decoder_block_1_attn_v_proj" => %{
        "kernel" => Nx.iota({64, 32}, type: :f32)
      },
      "decoder_block_1_attn_out_proj" => %{
        "kernel" => Nx.iota({64, 64}, type: :f32)
      },
      "decoder_block_1_ffn_norm_gamma" => Nx.broadcast(1.0, {64}),
      "decoder_block_1_ffn_gate" => %{
        "kernel" => Nx.iota({64, 172}, type: :f32)
      },
      "decoder_block_1_ffn_up" => %{
        "kernel" => Nx.iota({64, 172}, type: :f32)
      },
      "decoder_block_1_ffn_down" => %{
        "kernel" => Nx.iota({172, 64}, type: :f32)
      },
      "decoder_block_2_attn_norm_gamma" => Nx.broadcast(1.0, {64}),
      "decoder_block_2_attn_q_proj" => %{
        "kernel" => Nx.iota({64, 64}, type: :f32)
      },
      "decoder_block_2_attn_k_proj" => %{
        "kernel" => Nx.iota({64, 32}, type: :f32)
      },
      "decoder_block_2_attn_v_proj" => %{
        "kernel" => Nx.iota({64, 32}, type: :f32)
      },
      "decoder_block_2_attn_out_proj" => %{
        "kernel" => Nx.iota({64, 64}, type: :f32)
      },
      "decoder_block_2_ffn_norm_gamma" => Nx.broadcast(1.0, {64}),
      "decoder_block_2_ffn_gate" => %{
        "kernel" => Nx.iota({64, 172}, type: :f32)
      },
      "decoder_block_2_ffn_up" => %{
        "kernel" => Nx.iota({64, 172}, type: :f32)
      },
      "decoder_block_2_ffn_down" => %{
        "kernel" => Nx.iota({172, 64}, type: :f32)
      },
      "final_norm" => %{
        "gamma" => Nx.broadcast(1.0, {64})
      }
    }
  end
end
