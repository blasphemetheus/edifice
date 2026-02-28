defmodule Edifice.Pretrained.KeyMap do
  @moduledoc """
  Behaviour for mapping external checkpoint keys to Axon parameter paths.

  Each pretrained checkpoint format (HuggingFace, timm, etc.) uses different
  naming conventions for model parameters. A key map module translates those
  external names into the dot-separated paths that Axon uses internally.

  ## Callbacks

    - `map_key/1` — Maps a single external key to an Axon parameter path string,
      or returns `:skip` to exclude the parameter from loading.
    - `tensor_transforms/0` — Returns a list of `{regex, transform_fn}` pairs.
      After key mapping, the first regex that matches the **mapped** (Axon) key
      determines which transform is applied to the tensor value.

  ## Example

      defmodule MyApp.KeyMaps.ViT do
        @behaviour Edifice.Pretrained.KeyMap

        @impl true
        def map_key("cls_token"), do: "cls_token.kernel"
        def map_key("patch_embed.proj.weight"), do: "patch_embed.kernel"
        def map_key("patch_embed.proj.bias"), do: "patch_embed.bias"
        def map_key(_key), do: :skip

        @impl true
        def tensor_transforms do
          [
            {~r/\\.kernel$/, &Edifice.Pretrained.Transform.transpose_linear/1}
          ]
        end
      end

  """

  @doc """
  Maps an external checkpoint key to an Axon parameter path.

  Returns a dot-separated string like `"block_0_norm1.scale"`, `:skip`
  to intentionally exclude the parameter, or `:unmapped` to signal that
  the key is not recognized (triggers an error in strict mode).
  """
  @callback map_key(external_key :: String.t()) :: String.t() | :skip | :unmapped

  @doc """
  Returns a list of `{regex, transform_fn}` pairs for post-mapping tensor transforms.

  Each pair consists of a compiled `Regex` and a function `Nx.Tensor.t() -> Nx.Tensor.t()`.
  After key mapping, the loader checks the **mapped** key against each regex in order
  and applies the transform from the first match. If no regex matches, the tensor is
  used as-is.

  ## Example

      def tensor_transforms do
        [
          {~r/\\.kernel$/, &Edifice.Pretrained.Transform.transpose_linear/1},
          {~r/\\.scale$/, &Function.identity/1}
        ]
      end

  """
  @callback tensor_transforms() :: [{Regex.t(), (Nx.Tensor.t() -> Nx.Tensor.t())}]
end
