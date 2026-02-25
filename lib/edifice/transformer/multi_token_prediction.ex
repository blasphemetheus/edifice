defmodule Edifice.Transformer.MultiTokenPrediction do
  @moduledoc """
  Multi-Token Prediction (MTP) — predict multiple future tokens simultaneously.

  Wraps a backbone transformer (DecoderOnly by default) with multiple
  independent prediction heads. Each head projects the backbone's hidden
  states to vocabulary logits for a different future position.

  ## Key Innovation: Parallel Next-Token Heads

  Instead of predicting only the next token, MTP attaches N independent
  dense layers to the backbone output, each predicting a different future
  position. This provides richer training signal and enables speculative
  decoding at inference time.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Backbone (output_mode: :all)
        |
  [batch, seq_len, hidden_size]
        |
  +-- Head 1: dense(vocab_size) -> pred_1 [batch, seq_len, vocab_size]
  +-- Head 2: dense(vocab_size) -> pred_2 [batch, seq_len, vocab_size]
  +-- ...
  +-- Head N: dense(vocab_size) -> pred_N [batch, seq_len, vocab_size]
        |
  Axon.container(%{pred_1: h1, pred_2: h2, ..., pred_N: hN})
  ```

  ## Usage

      model = MultiTokenPrediction.build(
        embed_dim: 256,
        vocab_size: 32000,
        num_predictions: 4
      )

  ## References

  - "Better & Faster Large Language Models via Multi-token Prediction"
    (Gloeckle et al., 2024) — https://arxiv.org/abs/2404.19737
  """

  alias Edifice.Attention.GQA
  alias Edifice.Blocks.TransformerBlock

  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_num_predictions 4
  @default_dropout 0.1

  @doc """
  Build a Multi-Token Prediction model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:vocab_size` - Vocabulary size for each prediction head (required)
    - `:hidden_size` - Backbone hidden dimension (default: 256)
    - `:num_layers` - Number of backbone transformer layers (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_kv_heads` - Number of key/value heads for GQA (default: 2)
    - `:num_predictions` - Number of future tokens to predict (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An `Axon.container` with keys `:pred_1` through `:pred_N`, each
    shaped `[batch, seq_len, vocab_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:vocab_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:num_predictions, pos_integer()}
          | {:dropout, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    num_predictions = Keyword.get(opts, :num_predictions, @default_num_predictions)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Build backbone with output_mode: :all to get all timesteps
    backbone =
      build_backbone(
        embed_dim: embed_dim,
        hidden_size: hidden_size,
        num_layers: num_layers,
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        dropout: dropout,
        seq_len: Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))
      )

    # N independent prediction heads
    heads =
      for i <- 1..num_predictions, into: %{} do
        head =
          backbone
          |> Axon.dense(vocab_size, name: "pred_head_#{i}")

        {String.to_atom("pred_#{i}"), head}
      end

    Axon.container(heads)
  end

  # Build a DecoderOnly-style backbone that outputs all timesteps
  defp build_backbone(opts) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_heads = Keyword.fetch!(opts, :num_heads)
    num_kv_heads = Keyword.fetch!(opts, :num_kv_heads)
    dropout = Keyword.fetch!(opts, :dropout)
    seq_len = Keyword.get(opts, :seq_len, 60)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        name = "mtp_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          GQA.build_gqa_attention(x_in,
            hidden_size: hidden_size,
            num_heads: num_heads,
            num_kv_heads: num_kv_heads,
            rope: true,
            name: attn_name
          )
        end

        TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: hidden_size,
          norm: :rms_norm,
          ffn_type: :gated,
          dropout: dropout,
          name: name
        )
      end)

    # Final norm — keep all timesteps (no last_timestep extraction)
    Axon.layer_norm(x, name: "final_norm")
  end

  @doc "Get the output size of the model (hidden_size of backbone)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
