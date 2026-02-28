defmodule Edifice.Vision.Backbone do
  @moduledoc """
  Shared behaviour for vision modules that can serve as feature extractors.

  Any vision module that produces `[batch, feature_dim]` feature vectors from
  images can adopt this behaviour. This enables downstream consumers (detection,
  segmentation, multimodal) to swap backbones without knowing internals.

  ## Callbacks

    - `build_backbone/1` — Returns an Axon model that outputs `[batch, feature_dim]`
      feature vectors (no classifier head).
    - `feature_size/1` — Returns the feature dimension as a positive integer.
    - `input_shape/1` — Returns the expected input shape, e.g. `{nil, 3, 224, 224}`.

  ## Usage

  Adopting modules use `use Edifice.Vision.Backbone` to get default
  implementations of `input_shape/1` (reads `:in_channels` and `:image_size`
  from opts) which can be overridden.

      defmodule Edifice.Vision.ViT do
        use Edifice.Vision.Backbone

        @impl Edifice.Vision.Backbone
        def build_backbone(opts) do
          opts |> Keyword.delete(:num_classes) |> build()
        end

        @impl Edifice.Vision.Backbone
        def feature_size(opts) do
          Keyword.get(opts, :embed_dim, 768)
        end
      end

  ## Dispatch Helper

      # Build any backbone by module reference
      model = Edifice.Vision.Backbone.build_backbone(Edifice.Vision.ViT,
        image_size: 224, patch_size: 16, embed_dim: 768, depth: 12, num_heads: 12
      )

  ## Adopters

  ViT, DeiT, Swin, ConvNeXt, MLPMixer, PoolFormer, FocalNet, MetaFormer,
  EfficientViT, MambaVision, DINOv2, DINOv3.
  """

  @doc """
  Build a feature extractor model (no classifier head).

  Returns an Axon model that outputs `[batch, feature_dim]` feature vectors.
  """
  @callback build_backbone(opts :: keyword()) :: Axon.t()

  @doc """
  Return the feature dimension for the given options.
  """
  @callback feature_size(opts :: keyword()) :: pos_integer()

  @doc """
  Return the expected input shape for the given options.

  Typically `{nil, in_channels, image_size, image_size}` (NCHW).
  """
  @callback input_shape(opts :: keyword()) :: tuple()

  @doc false
  defmacro __using__(_opts) do
    quote do
      @behaviour Edifice.Vision.Backbone

      @impl Edifice.Vision.Backbone
      def input_shape(opts \\ []) do
        in_channels = Keyword.get(opts, :in_channels, 3)
        image_size = Keyword.get(opts, :image_size, 224)
        {nil, in_channels, image_size, image_size}
      end

      defoverridable input_shape: 1
    end
  end

  @doc """
  Build a backbone by dispatching to the given module's `build_backbone/1`.

  ## Examples

      model = Edifice.Vision.Backbone.build_backbone(Edifice.Vision.ViT,
        image_size: 32, patch_size: 8, embed_dim: 64, depth: 2, num_heads: 2
      )
  """
  @spec build_backbone(module(), keyword()) :: Axon.t()
  def build_backbone(module, opts \\ []) do
    module.build_backbone(opts)
  end

  @doc """
  Get the feature size by dispatching to the given module's `feature_size/1`.
  """
  @spec feature_size(module(), keyword()) :: pos_integer()
  def feature_size(module, opts \\ []) do
    module.feature_size(opts)
  end

  @doc """
  Get the input shape by dispatching to the given module's `input_shape/1`.
  """
  @spec input_shape(module(), keyword()) :: tuple()
  def input_shape(module, opts \\ []) do
    module.input_shape(opts)
  end
end
