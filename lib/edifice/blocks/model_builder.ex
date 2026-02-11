defmodule Edifice.Blocks.ModelBuilder do
  @moduledoc """
  High-level model building utilities for sequence and vision architectures.

  Provides standardized model skeletons that handle input creation, projection,
  block stacking, final normalization, and output extraction. Architecture-specific
  logic is provided via block builder callbacks.

  ## Sequence Model

  ```
  Input [batch, seq_len, embed_size]
    -> Optional projection to hidden_size
    -> Stack N blocks (via block_builder callback)
    -> Final LayerNorm
    -> Output extraction (last_timestep / all / mean_pool)
  ```

  ## Vision Model

  ```
  Input [batch, channels, height, width]
    -> Patch embedding
    -> Stack N blocks (via block_builder callback)
    -> Final LayerNorm
    -> Pooling (cls_token / mean_pool)
    -> Optional classifier head
  ```

  ## Usage

      # Build a sequence model with custom blocks
      model = ModelBuilder.build_sequence_model(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        block_builder: fn input, opts -> MyBlock.layer(input, opts) end
      )

  ## Design

  Generalizes the pattern from `Edifice.SSM.Common.build_model/2` to work
  with any block type (SSM, attention, MLP mixer, etc.).
  """

  require Axon

  @doc """
  Build a sequence processing model.

  ## Options
    - `:embed_size` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: embed_size)
    - `:num_layers` - Number of blocks to stack (required)
    - `:block_builder` - Function `(input, opts) -> Axon.t()` that builds one block (required)
    - `:seq_len` - Expected sequence length for JIT optimization (default: 60)
    - `:output_mode` - Output extraction: `:last_timestep`, `:all`, `:mean_pool` (default: :last_timestep)
    - `:final_norm` - Whether to apply final layer norm (default: true)
    - `:dropout` - Dropout rate between blocks (default: 0.0)

  ## Returns

  An Axon model. Output shape depends on `:output_mode`:
    - `:last_timestep` -> `[batch, hidden_size]`
    - `:all` -> `[batch, seq_len, hidden_size]`
    - `:mean_pool` -> `[batch, hidden_size]`
  """
  @spec build_sequence_model(keyword()) :: Axon.t()
  def build_sequence_model(opts) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, embed_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    block_builder = Keyword.fetch!(opts, :block_builder)
    seq_len = Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))
    output_mode = Keyword.get(opts, :output_mode, :last_timestep)
    final_norm = Keyword.get(opts, :final_norm, true)
    dropout = Keyword.get(opts, :dropout, 0.0)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = block_builder.(acc, Keyword.put(opts, :layer_idx, layer_idx))

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "inter_block_dropout_#{layer_idx}")
        else
          block
        end
      end)

    # Final normalization
    x = if final_norm, do: Axon.layer_norm(x, name: "final_norm"), else: x

    # Output extraction
    extract_output(x, output_mode)
  end

  @doc """
  Build a vision model with patch embedding.

  ## Options
    - `:image_size` - Input image size (square, default: 224)
    - `:patch_size` - Patch size (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_size` - Hidden dimension (required)
    - `:num_layers` - Number of blocks to stack (required)
    - `:block_builder` - Function `(input, opts) -> Axon.t()` that builds one block (required)
    - `:num_classes` - If provided, adds a classifier head
    - `:output_mode` - Pooling mode: `:mean_pool`, `:cls_token` (default: :mean_pool)
    - `:final_norm` - Whether to apply final layer norm (default: true)

  ## Returns

  An Axon model outputting `[batch, hidden_size]` or `[batch, num_classes]`.
  """
  @spec build_vision_model(keyword()) :: Axon.t()
  def build_vision_model(opts) do
    image_size = Keyword.get(opts, :image_size, 224)
    patch_size = Keyword.get(opts, :patch_size, 16)
    in_channels = Keyword.get(opts, :in_channels, 3)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    block_builder = Keyword.fetch!(opts, :block_builder)
    num_classes = Keyword.get(opts, :num_classes)
    final_norm = Keyword.get(opts, :final_norm, true)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding
    x =
      Edifice.Blocks.PatchEmbed.layer(input,
        embed_dim: hidden_size,
        patch_size: patch_size,
        in_channels: in_channels,
        name: "patch_embed"
      )

    # Stack blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block_builder.(acc, Keyword.put(opts, :layer_idx, layer_idx))
      end)

    # Final normalization
    x = if final_norm, do: Axon.layer_norm(x, name: "final_norm"), else: x

    # Pooling: mean over patches
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [1])
        end,
        name: "mean_pool"
      )

    # Optional classifier head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # Output extraction helpers

  defp extract_output(x, :last_timestep) do
    Axon.nx(
      x,
      fn tensor ->
        seq_len = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp extract_output(x, :all), do: x

  defp extract_output(x, :mean_pool) do
    Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1]) end, name: "mean_pool")
  end
end
