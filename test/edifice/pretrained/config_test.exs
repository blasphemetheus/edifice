defmodule Edifice.Pretrained.ConfigTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.Config

  describe "supported_model_types/0" do
    test "returns sorted list of supported types" do
      types = Config.supported_model_types()
      assert types == ["convnext", "detr", "resnet", "vit", "whisper"]
    end
  end

  describe "parse/1 with ViT config" do
    @vit_config Jason.encode!(%{
                  "model_type" => "vit",
                  "image_size" => 224,
                  "patch_size" => 16,
                  "num_channels" => 3,
                  "hidden_size" => 768,
                  "num_hidden_layers" => 12,
                  "num_attention_heads" => 12,
                  "intermediate_size" => 3072,
                  "hidden_dropout_prob" => 0.0,
                  "num_labels" => 1000
                })

    test "parses ViT config correctly" do
      assert {:ok, parsed} = Config.parse(@vit_config)
      assert parsed.model_type == "vit"
      assert parsed.key_map == Edifice.Pretrained.KeyMaps.ViT
      assert is_function(parsed.build_fn, 1)

      opts = parsed.build_opts
      assert opts[:image_size] == 224
      assert opts[:patch_size] == 16
      assert opts[:in_channels] == 3
      assert opts[:embed_dim] == 768
      assert opts[:depth] == 12
      assert opts[:num_heads] == 12
      assert opts[:mlp_ratio] == 4.0
      assert opts[:dropout] == 0.0
      assert opts[:num_classes] == 1000
    end

    test "computes mlp_ratio from intermediate_size / hidden_size" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "hidden_size" => 1024,
          "num_hidden_layers" => 24,
          "num_attention_heads" => 16,
          "intermediate_size" => 4096
        })

      assert {:ok, parsed} = Config.parse(config)
      assert parsed.build_opts[:mlp_ratio] == 4.0
      assert parsed.build_opts[:embed_dim] == 1024
      assert parsed.build_opts[:depth] == 24
      assert parsed.build_opts[:num_heads] == 16
    end

    test "num_classes is nil when num_labels absent" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "hidden_size" => 768,
          "num_hidden_layers" => 12,
          "num_attention_heads" => 12
        })

      assert {:ok, parsed} = Config.parse(config)
      refute Keyword.has_key?(parsed.build_opts, :num_classes)
    end

    test "uses defaults for optional fields" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "hidden_size" => 768,
          "num_hidden_layers" => 12,
          "num_attention_heads" => 12
        })

      assert {:ok, parsed} = Config.parse(config)
      assert parsed.build_opts[:image_size] == 224
      assert parsed.build_opts[:patch_size] == 16
      assert parsed.build_opts[:in_channels] == 3
      assert parsed.build_opts[:dropout] == 0.0
    end

    test "returns error when hidden_size is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "num_hidden_layers" => 12,
          "num_attention_heads" => 12
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "hidden_size"
    end

    test "returns error when num_hidden_layers is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "hidden_size" => 768,
          "num_attention_heads" => 12
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "num_hidden_layers"
    end
  end

  describe "parse/1 with Whisper config" do
    @whisper_config Jason.encode!(%{
                      "model_type" => "whisper",
                      "num_mel_bins" => 80,
                      "max_source_positions" => 1500,
                      "d_model" => 512,
                      "encoder_layers" => 6,
                      "decoder_layers" => 6,
                      "encoder_attention_heads" => 8,
                      "decoder_attention_heads" => 8,
                      "encoder_ffn_dim" => 2048,
                      "vocab_size" => 51_865,
                      "max_target_positions" => 448,
                      "dropout" => 0.0
                    })

    test "parses Whisper config correctly" do
      assert {:ok, parsed} = Config.parse(@whisper_config)
      assert parsed.model_type == "whisper"
      assert parsed.key_map == Edifice.Pretrained.KeyMaps.Whisper
      assert parsed.build_fn == nil

      opts = parsed.build_opts
      assert opts[:n_mels] == 80
      assert opts[:max_audio_len] == 1500
      assert opts[:hidden_dim] == 512
      assert opts[:encoder_layers] == 6
      assert opts[:decoder_layers] == 6
      assert opts[:num_heads] == 8
      assert opts[:ffn_dim] == 2048
      assert opts[:vocab_size] == 51_865
      assert opts[:max_dec_len] == 448
      assert opts[:dropout] == 0.0
    end

    test "returns error when d_model is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "whisper",
          "encoder_layers" => 6,
          "decoder_layers" => 6,
          "vocab_size" => 51_865
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "d_model"
    end

    test "returns error when vocab_size is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "whisper",
          "d_model" => 512,
          "encoder_layers" => 6,
          "decoder_layers" => 6
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "vocab_size"
    end
  end

  describe "parse/1 with ConvNeXt config" do
    @convnext_config Jason.encode!(%{
                       "model_type" => "convnext",
                       "image_size" => 224,
                       "patch_size" => 4,
                       "num_channels" => 3,
                       "depths" => [3, 3, 9, 3],
                       "hidden_sizes" => [96, 192, 384, 768],
                       "num_labels" => 1000,
                       "drop_path_rate" => 0.1
                     })

    test "parses ConvNeXt config correctly" do
      assert {:ok, parsed} = Config.parse(@convnext_config)
      assert parsed.model_type == "convnext"
      assert parsed.key_map == Edifice.Pretrained.KeyMaps.ConvNeXt
      assert is_function(parsed.build_fn, 1)

      opts = parsed.build_opts
      assert opts[:image_size] == 224
      assert opts[:patch_size] == 4
      assert opts[:in_channels] == 3
      assert opts[:depths] == [3, 3, 9, 3]
      assert opts[:dims] == [96, 192, 384, 768]
      assert opts[:num_classes] == 1000
      assert opts[:dropout] == 0.1
    end

    test "returns error when depths is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "convnext",
          "hidden_sizes" => [96, 192, 384, 768]
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "depths"
    end

    test "returns error when hidden_sizes is missing" do
      config =
        Jason.encode!(%{
          "model_type" => "convnext",
          "depths" => [3, 3, 9, 3]
        })

      assert {:error, msg} = Config.parse(config)
      assert msg =~ "hidden_sizes"
    end

    test "num_classes is nil when num_labels absent" do
      config =
        Jason.encode!(%{
          "model_type" => "convnext",
          "depths" => [3, 3, 9, 3],
          "hidden_sizes" => [96, 192, 384, 768]
        })

      assert {:ok, parsed} = Config.parse(config)
      refute Keyword.has_key?(parsed.build_opts, :num_classes)
    end
  end

  describe "parse/1 error cases" do
    test "returns error for unknown model_type" do
      config = Jason.encode!(%{"model_type" => "llama"})
      assert {:error, msg} = Config.parse(config)
      assert msg =~ "unsupported model_type"
      assert msg =~ "llama"
      assert msg =~ "vit"
    end

    test "returns error for missing model_type" do
      config = Jason.encode!(%{"hidden_size" => 768})
      assert {:error, msg} = Config.parse(config)
      assert msg =~ "model_type"
    end

    test "returns error for invalid JSON" do
      assert {:error, _} = Config.parse("not valid json{")
    end
  end

  describe "parse!/1" do
    test "returns parsed config on success" do
      config =
        Jason.encode!(%{
          "model_type" => "vit",
          "hidden_size" => 768,
          "num_hidden_layers" => 12,
          "num_attention_heads" => 12
        })

      parsed = Config.parse!(config)
      assert parsed.model_type == "vit"
    end

    test "raises on error" do
      assert_raise ArgumentError, ~r/unsupported model_type/, fn ->
        Config.parse!(Jason.encode!(%{"model_type" => "unknown"}))
      end
    end
  end
end
