defmodule Edifice.Blocks.TransformerBlock do
  @moduledoc """
  Composable transformer block with configurable attention/mixing function.

  Implements the standard pre-norm transformer block pattern:

      norm -> attention_fn -> residual -> norm -> FFN -> residual

  The caller provides the attention/mixing function as a callback, making this
  block reusable across GQA, FNet, Performer, Nystromformer, LinearTransformer,
  and any future attention variant.

  ## Architecture

  ```
  Input
    |
    +---> LayerNorm/RMSNorm -> attention_fn(x) ---+
    |                                              |
    +<----- Residual + Dropout <-------------------+
    |
    +---> LayerNorm/RMSNorm -> FFN(x) ------------+
    |                                              |
    +<----- Residual + Dropout <-------------------+
    |
  Output
  ```

  ## Usage

      # Single block with custom attention
      block = TransformerBlock.layer(input,
        attention_fn: fn x, name -> build_my_attention(x, name) end,
        hidden_size: 256,
        name: "block_1"
      )

      # Stack N blocks
      output = TransformerBlock.stack(input, 4,
        attention_fn: fn x, name -> build_my_attention(x, name) end,
        hidden_size: 256,
        name: "transformer"
      )

  ## 3-Sublayer Variant (Encoder-Decoder)

  For encoder-decoder architectures, `layer/3` inserts a cross-attention
  sublayer between self-attention and FFN:

  ```
  Input   Memory
    |       |
    +---> LayerNorm/RMSNorm -> attention_fn(x) ---+
    |                                              |
    +<----- Residual + Dropout <-------------------+
    |       |
    +---> LayerNorm/RMSNorm -> cross_attention_fn(x, memory) ---+
    |                                                           |
    +<----- Residual + Dropout <--------------------------------+
    |
    +---> LayerNorm/RMSNorm -> FFN(x) ------------+
    |                                              |
    +<----- Residual + Dropout <-------------------+
    |
  Output
  ```

  ## Design

  Follows the callback pattern established by `Edifice.SSM.Common.build_block/3`,
  where the caller provides the core mixing/attention function and this module
  handles the surrounding structure (normalization, residuals, FFN, dropout).
  """

  alias Edifice.Blocks.FFN

  @doc """
  Build a single transformer block (2 sublayers: self-attention + FFN).

  ## Options
    - `:attention_fn` - Function `(input, name) -> Axon.t()` that builds the
      attention/mixing sublayer (required)
    - `:hidden_size` - Hidden dimension (required)
    - `:ffn_type` - FFN variant: `:standard` or `:gated` (default: :standard)
    - `:ffn_expansion` - FFN expansion factor (default: 4)
    - `:custom_ffn` - Custom FFN callback `(input, name) -> Axon.t()` that
      replaces the standard FFN sublayer. When provided, `:ffn_type` and
      `:ffn_expansion` are ignored. Used by KAT and other architectures
      that need non-standard feed-forward networks.
    - `:norm` - Normalization type: `:layer_norm` or `:rms_norm` (default: :layer_norm)
    - `:norm_position` - Where to normalize: `:pre` or `:post` (default: :pre)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Block name prefix (default: "transformer_block")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts) do
    attention_fn = Keyword.fetch!(opts, :attention_fn)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    ffn_type = Keyword.get(opts, :ffn_type, :standard)
    ffn_expansion = Keyword.get(opts, :ffn_expansion, 4)
    custom_ffn = Keyword.get(opts, :custom_ffn)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "transformer_block")

    # 1. Attention sublayer: norm -> attention_fn -> dropout -> residual
    attn_normed = apply_norm(input, opts, "#{name}_attn_norm")
    attn_out = attention_fn.(attn_normed, "#{name}_attn")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    after_attn = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer: norm -> FFN -> dropout -> residual
    ffn_normed = apply_norm(after_attn, opts, "#{name}_ffn_norm")

    ffn_out =
      if custom_ffn do
        custom_ffn.(ffn_normed, "#{name}_ffn")
      else
        case ffn_type do
          :gated ->
            FFN.gated_layer(ffn_normed,
              hidden_size: hidden_size,
              name: "#{name}_ffn"
            )

          _standard ->
            FFN.layer(ffn_normed,
              hidden_size: hidden_size,
              expansion_factor: ffn_expansion,
              name: "#{name}_ffn"
            )
        end
      end

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_dropout")
    Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build a single encoder-decoder transformer block (3 sublayers).

  Inserts a cross-attention sublayer between self-attention and FFN.
  Used by Whisper, ACT, DETR, and RT-DETR decoders.

  ## Options

  Same as `layer/2` plus:
    - `:cross_attention_fn` - Function `(query, memory, name) -> Axon.t()` that
      builds the cross-attention sublayer (required)
  """
  @spec layer(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(input, memory, opts) do
    attention_fn = Keyword.fetch!(opts, :attention_fn)
    cross_attention_fn = Keyword.fetch!(opts, :cross_attention_fn)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    ffn_type = Keyword.get(opts, :ffn_type, :standard)
    ffn_expansion = Keyword.get(opts, :ffn_expansion, 4)
    custom_ffn = Keyword.get(opts, :custom_ffn)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "transformer_block")

    # 1. Self-attention sublayer: norm -> attention_fn -> dropout -> residual
    attn_normed = apply_norm(input, opts, "#{name}_attn_norm")
    attn_out = attention_fn.(attn_normed, "#{name}_attn")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    after_attn = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. Cross-attention sublayer: norm -> cross_attention_fn -> dropout -> residual
    cross_normed = apply_norm(after_attn, opts, "#{name}_cross_attn_norm")
    cross_out = cross_attention_fn.(cross_normed, memory, "#{name}_cross_attn")
    cross_out = maybe_dropout(cross_out, dropout, "#{name}_cross_attn_dropout")
    after_cross = Axon.add(after_attn, cross_out, name: "#{name}_cross_attn_residual")

    # 3. FFN sublayer: norm -> FFN -> dropout -> residual
    ffn_normed = apply_norm(after_cross, opts, "#{name}_ffn_norm")

    ffn_out =
      if custom_ffn do
        custom_ffn.(ffn_normed, "#{name}_ffn")
      else
        case ffn_type do
          :gated ->
            FFN.gated_layer(ffn_normed,
              hidden_size: hidden_size,
              name: "#{name}_ffn"
            )

          _standard ->
            FFN.layer(ffn_normed,
              hidden_size: hidden_size,
              expansion_factor: ffn_expansion,
              name: "#{name}_ffn"
            )
        end
      end

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_dropout")
    Axon.add(after_cross, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Stack N transformer blocks with auto-naming (2-sublayer variant).

  ## Options

  Same as `layer/2` plus:
    - First argument is the input Axon node
    - Second argument is the number of layers to stack

  ## Returns

  The output of the final block (same shape as input).
  """
  @spec stack(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def stack(input, num_layers, opts) do
    base_name = Keyword.get(opts, :name, "transformer")

    Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
      layer(acc, Keyword.put(opts, :name, "#{base_name}_block_#{layer_idx}"))
    end)
  end

  @doc """
  Stack N encoder-decoder transformer blocks with auto-naming (3-sublayer variant).

  Each block receives the same `memory` input for cross-attention.

  ## Options

  Same as `layer/3`.
  """
  @spec stack(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def stack(input, memory, num_layers, opts) do
    base_name = Keyword.get(opts, :name, "transformer")

    Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
      layer(acc, memory, Keyword.put(opts, :name, "#{base_name}_block_#{layer_idx}"))
    end)
  end

  # Normalization helper
  defp apply_norm(input, opts, name) do
    case Keyword.get(opts, :norm, :layer_norm) do
      :rms_norm ->
        hidden_size = Keyword.fetch!(opts, :hidden_size)
        Edifice.Blocks.RMSNorm.layer(input, hidden_size: hidden_size, name: name)

      _layer_norm ->
        Axon.layer_norm(input, name: name)
    end
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
