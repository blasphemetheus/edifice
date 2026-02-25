defmodule Edifice.Meta.SpeculativeHead do
  @moduledoc """
  Speculative Head — multi-head parallel draft with per-head MLPs (Medusa/EAGLE).

  Each speculative head is an independent MLP that predicts a different future
  token position from the backbone's hidden states. Unlike the simple linear
  heads in MultiTokenPrediction, each head here has its own hidden layer with
  SiLU activation for greater expressiveness.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Backbone (GQA + RoPE + SwiGLU, all timesteps)
        |
  [batch, seq_len, hidden_size]
        |
  +-- Head 1: dense(head_hidden) -> silu -> dense(vocab_size) -> pred_1
  +-- Head 2: dense(head_hidden) -> silu -> dense(vocab_size) -> pred_2
  +-- ...
  +-- Head N: dense(head_hidden) -> silu -> dense(vocab_size) -> pred_N
        |
  Axon.container(%{pred_1: h1, ..., pred_N: hN})
  ```

  ## Usage

      model = SpeculativeHead.build(
        embed_dim: 256,
        vocab_size: 32000,
        num_predictions: 4,
        head_hidden: 128
      )

      # Re-exports accept_reject for inference
      accepted = SpeculativeHead.accept_reject(draft_tokens, verifier_tokens)

  ## References

  - Cai et al., "Medusa: Simple LLM Inference Acceleration Framework" (2024)
  - Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
  """

  alias Edifice.Attention.GQA
  alias Edifice.Blocks.TransformerBlock

  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_num_predictions 4
  @default_head_hidden 128
  @default_dropout 0.1

  @doc """
  Build a Speculative Head model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:vocab_size` - Vocabulary size for each prediction head (required)
    - `:hidden_size` - Backbone hidden dimension (default: 256)
    - `:num_layers` - Number of backbone transformer layers (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_kv_heads` - Number of KV heads for GQA (default: 2)
    - `:num_predictions` - Number of speculative heads (default: 4)
    - `:head_hidden` - Hidden dimension per head MLP (default: 128)
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
          | {:head_hidden, pos_integer()}
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
    head_hidden = Keyword.get(opts, :head_hidden, @default_head_hidden)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

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

    # N independent speculative heads, each with its own MLP
    heads =
      for i <- 1..num_predictions, into: %{} do
        head =
          backbone
          |> Axon.dense(head_hidden, name: "spec_head_#{i}_dense1")
          |> Axon.activation(:silu, name: "spec_head_#{i}_act")
          |> Axon.dense(vocab_size, name: "spec_head_#{i}_dense2")

        {String.to_atom("pred_#{i}"), head}
      end

    Axon.container(heads)
  end

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
        name = "spec_block_#{layer_idx}"

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

    Axon.layer_norm(x, name: "final_norm")
  end

  @doc """
  Re-export accept_reject from SpeculativeDecoding.

  Compares draft tokens to verifier tokens and returns the number of
  consecutively accepted tokens per batch element.
  """
  @spec accept_reject(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defdelegate accept_reject(draft_tokens, verifier_tokens),
    to: Edifice.Meta.SpeculativeDecoding

  @doc "Get the output size of the model (hidden_size of backbone)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      num_heads: 4,
      num_kv_heads: 2,
      num_predictions: 4,
      head_hidden: 128,
      dropout: 0.1
    ]
  end
end
