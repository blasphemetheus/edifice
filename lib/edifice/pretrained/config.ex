defmodule Edifice.Pretrained.Config do
  @moduledoc """
  Parses HuggingFace `config.json` and returns Edifice build opts.

  Maps `model_type` strings from HuggingFace config files to the appropriate
  key map module, build function, and translated build options.

  ## Usage

      {:ok, parsed} = Config.parse(json_string)
      # parsed.key_map    => Edifice.Pretrained.KeyMaps.ViT
      # parsed.build_fn   => &Edifice.Vision.ViT.build/1
      # parsed.build_opts  => [image_size: 224, patch_size: 16, ...]
      # parsed.model_type  => "vit"

  ## Supported model types

      Config.supported_model_types()
      #=> ["convnext", "detr", "resnet", "vit", "whisper"]

  """

  alias Edifice.Pretrained.KeyMaps

  @registry %{
    "vit" => {KeyMaps.ViT, &Edifice.Vision.ViT.build/1, &__MODULE__.map_vit_config/1},
    "convnext" =>
      {KeyMaps.ConvNeXt, &Edifice.Vision.ConvNeXt.build/1, &__MODULE__.map_convnext_config/1},
    "detr" => {KeyMaps.DETR, &Edifice.Detection.DETR.build/1, &__MODULE__.map_detr_config/1},
    "resnet" =>
      {KeyMaps.ResNet, &Edifice.Convolutional.ResNet.build/1, &__MODULE__.map_resnet_config/1},
    "whisper" => {KeyMaps.Whisper, nil, &__MODULE__.map_whisper_config/1}
  }

  @type parsed :: %{
          model_type: String.t(),
          key_map: module(),
          build_fn: (keyword() -> Axon.t()) | nil,
          build_opts: keyword()
        }

  @doc """
  Parses a HuggingFace `config.json` string and returns Edifice build configuration.

  Returns `{:ok, parsed}` where `parsed` contains:

    - `:model_type` — the HuggingFace model type string
    - `:key_map` — the key map module for weight loading
    - `:build_fn` — the build function (nil for Whisper which has encoder/decoder)
    - `:build_opts` — keyword list of Edifice build options

  Returns `{:error, reason}` if the model type is unsupported or required fields
  are missing.

  ## Examples

      {:ok, parsed} = Config.parse(~s({"model_type": "vit", "hidden_size": 768, ...}))
      parsed.key_map
      #=> Edifice.Pretrained.KeyMaps.ViT

  """
  @spec parse(String.t()) :: {:ok, parsed()} | {:error, String.t()}
  def parse(json_string) do
    with {:ok, config} <- Jason.decode(json_string),
         {:ok, model_type} <- fetch_model_type(config),
         {:ok, {key_map, build_fn, mapper}} <- lookup_registry(model_type),
         {:ok, build_opts} <- mapper.(config) do
      {:ok,
       %{
         model_type: model_type,
         key_map: key_map,
         build_fn: build_fn,
         build_opts: build_opts
       }}
    end
  end

  @doc """
  Like `parse/1` but raises on error.
  """
  @spec parse!(String.t()) :: parsed()
  def parse!(json_string) do
    case parse(json_string) do
      {:ok, parsed} -> parsed
      {:error, reason} -> raise ArgumentError, "Failed to parse config.json: #{reason}"
    end
  end

  @doc """
  Returns the list of supported HuggingFace `model_type` strings.

  ## Examples

      Config.supported_model_types()
      #=> ["convnext", "detr", "resnet", "vit", "whisper"]

  """
  @spec supported_model_types() :: [String.t()]
  def supported_model_types do
    @registry |> Map.keys() |> Enum.sort()
  end

  # -- Private helpers --

  defp fetch_model_type(config) do
    case Map.get(config, "model_type") do
      nil -> {:error, "missing \"model_type\" field in config.json"}
      model_type when is_binary(model_type) -> {:ok, model_type}
    end
  end

  defp lookup_registry(model_type) do
    case Map.get(@registry, model_type) do
      nil ->
        supported = supported_model_types() |> Enum.join(", ")

        {:error,
         "unsupported model_type #{inspect(model_type)}. " <>
           "Supported types: #{supported}"}

      entry ->
        {:ok, entry}
    end
  end

  # -- Config mappers --

  @doc false
  @spec map_vit_config(map()) :: {:ok, keyword()} | {:error, String.t()}
  def map_vit_config(config) do
    with {:ok, hidden_size} <- require_field(config, "hidden_size"),
         {:ok, num_hidden_layers} <- require_field(config, "num_hidden_layers"),
         {:ok, num_attention_heads} <- require_field(config, "num_attention_heads") do
      intermediate_size = config["intermediate_size"] || hidden_size * 4

      opts =
        [
          image_size: config["image_size"] || 224,
          patch_size: config["patch_size"] || 16,
          in_channels: config["num_channels"] || 3,
          embed_dim: hidden_size,
          depth: num_hidden_layers,
          num_heads: num_attention_heads,
          mlp_ratio: intermediate_size / hidden_size,
          layer_norm_epsilon: config["layer_norm_eps"] || 1.0e-12,
          dropout: config["hidden_dropout_prob"] || 0.0
        ]
        |> maybe_put(:num_classes, config["num_labels"])

      {:ok, opts}
    end
  end

  @doc false
  @spec map_whisper_config(map()) :: {:ok, keyword()} | {:error, String.t()}
  def map_whisper_config(config) do
    with {:ok, d_model} <- require_field(config, "d_model"),
         {:ok, encoder_layers} <- require_field(config, "encoder_layers"),
         {:ok, decoder_layers} <- require_field(config, "decoder_layers"),
         {:ok, vocab_size} <- require_field(config, "vocab_size") do
      opts = [
        n_mels: config["num_mel_bins"] || 80,
        max_audio_len: 2 * (config["max_source_positions"] || 1500),
        hidden_dim: d_model,
        encoder_layers: encoder_layers,
        decoder_layers: decoder_layers,
        num_heads: config["encoder_attention_heads"] || 8,
        ffn_dim: config["encoder_ffn_dim"] || d_model * 4,
        vocab_size: vocab_size,
        max_dec_len: config["max_target_positions"] || 448,
        dropout: config["dropout"] || 0.0
      ]

      {:ok, opts}
    end
  end

  @doc false
  @spec map_convnext_config(map()) :: {:ok, keyword()} | {:error, String.t()}
  def map_convnext_config(config) do
    with {:ok, depths} <- require_field(config, "depths"),
         {:ok, hidden_sizes} <- require_field(config, "hidden_sizes") do
      opts =
        [
          image_size: config["image_size"] || 224,
          patch_size: config["patch_size"] || 4,
          in_channels: config["num_channels"] || 3,
          depths: depths,
          dims: hidden_sizes,
          layer_norm_epsilon: config["layer_norm_eps"] || 1.0e-6,
          dropout: config["drop_path_rate"] || 0.0
        ]
        |> maybe_put(:num_classes, config["num_labels"])

      {:ok, opts}
    end
  end

  @doc false
  @spec map_detr_config(map()) :: {:ok, keyword()} | {:error, String.t()}
  def map_detr_config(config) do
    with {:ok, hidden_dim} <- require_field(config, "d_model") do
      opts = [
        image_size: config["image_size"] || 800,
        in_channels: config["num_channels"] || 3,
        hidden_dim: hidden_dim,
        num_heads: config["encoder_attention_heads"] || 8,
        num_encoder_layers: config["encoder_layers"] || 6,
        num_decoder_layers: config["decoder_layers"] || 6,
        ffn_dim: config["encoder_ffn_dim"] || 2048,
        num_queries: config["num_queries"] || 100,
        num_classes: config["num_labels"] || 91,
        dropout: config["dropout"] || 0.1,
        backbone: :resnet50,
        norm_position: :post
      ]

      {:ok, opts}
    end
  end

  @doc false
  @spec map_resnet_config(map()) :: {:ok, keyword()} | {:error, String.t()}
  def map_resnet_config(config) do
    with {:ok, depths} <- require_field(config, "depths"),
         {:ok, _hidden_sizes} <- require_field(config, "hidden_sizes") do
      image_size = config["image_size"] || 224
      num_channels = config["num_channels"] || 3

      layer_type =
        case config["layer_type"] do
          "bottleneck" -> :bottleneck
          "basic" -> :residual
          _ -> :residual
        end

      opts =
        [
          input_shape: {nil, image_size, image_size, num_channels},
          block_sizes: depths,
          block_type: layer_type,
          initial_channels: config["embedding_size"] || 64
        ]
        |> maybe_put(:num_classes, config["num_labels"])

      {:ok, opts}
    end
  end

  defp require_field(config, field) do
    case Map.get(config, field) do
      nil -> {:error, "missing required field #{inspect(field)} in config.json"}
      value -> {:ok, value}
    end
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
