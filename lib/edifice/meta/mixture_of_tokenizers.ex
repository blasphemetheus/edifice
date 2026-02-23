defmodule Edifice.Meta.MixtureOfTokenizers do
  @moduledoc """
  Mixture of Tokenizers — multiple parallel embedding pathways with learned routing.

  Uses N separate tokenizer embedding pathways, each with a different vocabulary
  size and embedding dimension, combined via learned soft routing weights. This
  allows the model to dynamically select the best tokenization granularity for
  each position.

  ## Architecture

  ```
  Input [batch, seq_len]
        |
  +-- Tokenizer 1: embedding(vocab_1, embed_1) -> dense(hidden_size) --+
  +-- Tokenizer 2: embedding(vocab_2, embed_2) -> dense(hidden_size) --+
  +-- ...                                                              +
  +-- Tokenizer N: embedding(vocab_N, embed_N) -> dense(hidden_size) --+
        |
  Router: shared_embed -> dense(N) -> softmax -> weights [batch, seq_len, N]
        |
  Weighted sum -> [batch, seq_len, hidden_size]
        |
  Transformer blocks -> final norm -> last timestep
        |
  [batch, hidden_size]
  ```

  ## Usage

      model = MixtureOfTokenizers.build(
        hidden_size: 256,
        num_tokenizers: 4,
        tokenizer_vocab_sizes: [256, 512, 1024, 2048],
        tokenizer_embed_dims: [32, 64, 128, 256]
      )

  ## References

  - "Mixture-of-Tokenizers" (Pham et al., 2024) — multi-granularity tokenization
  """

  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Attention.GQA

  @default_hidden_size 256
  @default_num_tokenizers 4
  @default_vocab_sizes [256, 512, 1024, 2048]
  @default_embed_dims [32, 64, 128, 256]
  @default_num_layers 4
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Mixture of Tokenizers model.

  ## Options

    - `:hidden_size` - Output hidden dimension (default: 256)
    - `:num_tokenizers` - Number of parallel tokenizer pathways (default: 4)
    - `:tokenizer_vocab_sizes` - List of vocab sizes per tokenizer (default: [256, 512, 1024, 2048])
    - `:tokenizer_embed_dims` - List of embedding dims per tokenizer (default: [32, 64, 128, 256])
    - `:num_layers` - Number of transformer layers after fusion (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_kv_heads` - Number of KV heads for GQA (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:num_tokenizers, pos_integer()}
          | {:tokenizer_vocab_sizes, [pos_integer()]}
          | {:tokenizer_embed_dims, [pos_integer()]}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_tokenizers = Keyword.get(opts, :num_tokenizers, @default_num_tokenizers)
    vocab_sizes = Keyword.get(opts, :tokenizer_vocab_sizes, @default_vocab_sizes)
    embed_dims = Keyword.get(opts, :tokenizer_embed_dims, @default_embed_dims)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # We accept a 3D float input [batch, seq_len, embed_dim] since the framework
    # uses continuous embeddings. Each "tokenizer" pathway takes the same input
    # through different projection widths.
    embed_dim = Keyword.get(opts, :embed_dim, hd(embed_dims))
    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Build N tokenizer pathways: each projects to its own embed_dim, then up to hidden_size
    tokenizer_outputs =
      Enum.zip([0..(num_tokenizers - 1), vocab_sizes, embed_dims])
      |> Enum.map(fn {i, _vocab_size, tok_embed_dim} ->
        input
        |> Axon.dense(tok_embed_dim, name: "tokenizer_#{i}_embed")
        |> Axon.dense(hidden_size, name: "tokenizer_#{i}_proj")
      end)

    # Router: input -> dense(num_tokenizers) -> softmax
    router_weights =
      input
      |> Axon.dense(hidden_size, name: "router_shared")
      |> Axon.activation(:silu, name: "router_act")
      |> Axon.dense(num_tokenizers, name: "router_logits")
      |> Axon.activation(:softmax, name: "router_softmax")

    # Weighted sum of tokenizer outputs using router weights
    # For each tokenizer i, extract weight slice i and multiply with output i, then sum
    fused = build_weighted_sum(tokenizer_outputs, router_weights)

    # Transformer blocks on fused representation
    x =
      Enum.reduce(1..num_layers, fused, fn layer_idx, acc ->
        name = "mot_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          GQA.build_gqa_attention(x_in,
            hidden_size: hidden_size,
            num_heads: num_heads,
            num_kv_heads: num_kv_heads,
            rope: true,
            name: attn_name
          )
        end

        block = TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: hidden_size,
          norm: :rms_norm,
          ffn_type: :gated,
          dropout: dropout,
          name: name
        )

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "mot_dropout_#{layer_idx}")
        else
          block
        end
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    # Last timestep
    Axon.nx(
      x,
      fn tensor ->
        seq_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_actual - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # Build weighted sum using Axon graph ops (multiply + add per tokenizer)
  defp build_weighted_sum(tokenizer_outputs, router_weights) do
    # For each tokenizer i, extract weight slice and multiply
    weighted_outputs =
      tokenizer_outputs
      |> Enum.with_index()
      |> Enum.map(fn {tok_out, i} ->
        # Extract weight for tokenizer i: [batch, seq_len, 1]
        weight_i =
          Axon.nx(
            router_weights,
            fn w -> Nx.slice_along_axis(w, i, 1, axis: 2) end,
            name: "router_weight_#{i}"
          )

        Axon.multiply(tok_out, weight_i, name: "weighted_tok_#{i}")
      end)

    # Sum all weighted outputs
    Enum.reduce(weighted_outputs, fn out, acc ->
      Axon.add(acc, out, name: "fuse_add")
    end)
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_tokenizers: 4,
      tokenizer_vocab_sizes: [256, 512, 1024, 2048],
      tokenizer_embed_dims: [32, 64, 128, 256],
      num_layers: 4,
      dropout: 0.1
    ]
  end
end
