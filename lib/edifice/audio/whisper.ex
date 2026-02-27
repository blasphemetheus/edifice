defmodule Edifice.Audio.Whisper do
  @moduledoc """
  Whisper: Encoder-Decoder Automatic Speech Recognition.

  OpenAI's Whisper is a transformer-based encoder-decoder model trained on
  680,000 hours of multilingual speech data. It processes log-mel spectrograms
  through a convolutional stem and transformer encoder, then autoregressively
  decodes text tokens via a transformer decoder with cross-attention.

  ## Architecture

  ```
  Mel spectrogram [batch, n_mels, audio_len]
        |
  +-----v-----------------------+
  | Conv1D Stem                  |  Conv(n_mels→d, k=3, s=1) + GELU
  |                              |  Conv(d→d, k=3, s=2) + GELU
  +-----+-----------------------+  Downsamples time by 2x
        |
  + Sinusoidal Positional Encoding
        |
  +-----v-----------------------+
  | Encoder Blocks x N           |  LayerNorm → Self-Attn → Residual
  |                              |  LayerNorm → FFN → Residual
  +-----+-----------------------+
        |
  LayerNorm → encoder output [batch, audio_len/2, hidden_dim]
        |
        +--------------------------------------------------+
        |                                                  |
  Token IDs [batch, dec_len]                               |
        |                                                  |
  Token Embedding + Learned Positional Embedding            |
        |                                                  |
  +-----v-----------------------+                          |
  | Decoder Blocks x N           |  LN → Causal Self-Attn → Residual
  |                              |  LN → Cross-Attn(encoder) → Residual
  |                              |  LN → FFN → Residual
  +-----+-----------------------+
        |
  LayerNorm → Dense(vocab_size)
        |
  Logits [batch, dec_len, vocab_size]
  ```

  ## Usage

      # Build encoder and decoder
      {encoder, decoder} = Whisper.build(
        hidden_dim: 512,
        encoder_layers: 6,
        decoder_layers: 6,
        num_heads: 8
      )

      # Encoder: mel spectrogram → encoder output
      {enc_init, enc_predict} = Axon.build(encoder, mode: :inference)
      enc_params = enc_init.(mel_input, Axon.ModelState.empty())
      encoder_output = enc_predict.(enc_params, mel_input)

      # Decoder: token IDs + encoder output → logits
      {dec_init, dec_predict} = Axon.build(decoder, mode: :inference)
      dec_params = dec_init.(decoder_inputs, Axon.ModelState.empty())
      logits = dec_predict.(dec_params, %{"token_ids" => tokens, "encoder_output" => encoder_output})

  ## Model Sizes

  The defaults correspond to Whisper base. Common configurations:

  | Model  | hidden_dim | encoder_layers | decoder_layers | num_heads |
  |--------|-----------|----------------|----------------|-----------|
  | tiny   | 384       | 4              | 4              | 6         |
  | base   | 512       | 6              | 6              | 8         |
  | small  | 768       | 12             | 12             | 12        |
  | medium | 1024      | 24             | 24             | 16        |
  | large  | 1280      | 32             | 32             | 20        |

  ## References

  - Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
    (OpenAI, 2023) — https://arxiv.org/abs/2212.04356
  """

  alias Edifice.Blocks.{CrossAttention, FFN, SinusoidalPE, TransformerBlock}
  alias Edifice.Utils.FusedOps

  # Whisper base defaults
  @default_n_mels 80
  @default_max_audio_len 1500
  @default_hidden_dim 512
  @default_encoder_layers 6
  @default_decoder_layers 6
  @default_num_heads 8
  @default_ffn_dim 2048
  @default_vocab_size 51_865
  @default_max_dec_len 448
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:n_mels, pos_integer()}
          | {:max_audio_len, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:encoder_layers, pos_integer()}
          | {:decoder_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:ffn_dim, pos_integer()}
          | {:vocab_size, pos_integer()}
          | {:max_dec_len, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build the Whisper encoder and decoder.

  Returns `{encoder, decoder}` where:
  - `encoder` takes `%{"mel_spectrogram" => [batch, n_mels, audio_len]}` and
    outputs `[batch, audio_len/2, hidden_dim]`
  - `decoder` takes `%{"token_ids" => [batch, dec_len], "encoder_output" => [batch, enc_len, hidden_dim]}`
    and outputs `[batch, dec_len, vocab_size]`

  See module docs for available options.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    {build_encoder(opts), build_decoder(opts)}
  end

  @doc """
  Build only the Whisper encoder.

  Takes a mel spectrogram and produces encoder hidden states.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    n_mels = Keyword.get(opts, :n_mels, @default_n_mels)
    max_audio_len = Keyword.get(opts, :max_audio_len, @default_max_audio_len)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    encoder_layers = Keyword.get(opts, :encoder_layers, @default_encoder_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    ffn_dim = Keyword.get(opts, :ffn_dim, @default_ffn_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Input: [batch, n_mels, audio_len] (channels-first, standard for mel spectrograms)
    input = Axon.input("mel_spectrogram", shape: {nil, n_mels, max_audio_len})

    # Transpose to channels-last for Axon conv: [batch, audio_len, n_mels]
    x = Axon.nx(input, fn t -> Nx.transpose(t, axes: [0, 2, 1]) end, name: "enc_transpose")

    # Conv1D stem: two 1D convolutions
    # Conv1(n_mels → hidden_dim, kernel=3, stride=1, padding=same) + GELU
    x =
      x
      |> Axon.conv(hidden_dim, kernel_size: {3}, padding: :same, name: "enc_conv1")
      |> Axon.activation(:gelu, name: "enc_conv1_act")

    # Conv2(hidden_dim → hidden_dim, kernel=3, stride=2, padding=same) + GELU
    x =
      x
      |> Axon.conv(hidden_dim,
        kernel_size: {3},
        strides: 2,
        padding: :same,
        name: "enc_conv2"
      )
      |> Axon.activation(:gelu, name: "enc_conv2_act")

    # Sinusoidal positional encoding
    x = SinusoidalPE.layer(x, dim: hidden_dim, name: "enc_pos")

    # Encoder transformer blocks (pre-norm self-attention + FFN)
    x =
      TransformerBlock.stack(x, encoder_layers,
        attention_fn: fn input_node, name ->
          self_attention(input_node, hidden_dim, num_heads, name)
        end,
        hidden_size: hidden_dim,
        ffn_type: :standard,
        custom_ffn: fn input_node, name ->
          FFN.layer(input_node,
            hidden_size: hidden_dim,
            inner_size: ffn_dim,
            dropout: dropout,
            name: name
          )
        end,
        dropout: dropout,
        name: "enc"
      )

    # Final layer norm
    Axon.layer_norm(x, name: "enc_final_norm")
  end

  @doc """
  Build only the Whisper decoder.

  Takes token IDs and encoder output, produces token logits.
  """
  @spec build_decoder([build_opt()]) :: Axon.t()
  def build_decoder(opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    decoder_layers = Keyword.get(opts, :decoder_layers, @default_decoder_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    ffn_dim = Keyword.get(opts, :ffn_dim, @default_ffn_dim)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    max_dec_len = Keyword.get(opts, :max_dec_len, @default_max_dec_len)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Inputs
    token_ids = Axon.input("token_ids", shape: {nil, max_dec_len})
    encoder_output = Axon.input("encoder_output", shape: {nil, nil, hidden_dim})

    # Token embedding
    token_embed = Axon.embedding(token_ids, vocab_size, hidden_dim, name: "dec_token_embed")

    # Learned positional embedding: generate position indices and embed
    pos_embed_table =
      Axon.embedding(
        Axon.nx(
          token_ids,
          fn t ->
            seq_len = Nx.axis_size(t, 1)

            Nx.iota({1, seq_len}, axis: 1, type: :s64)
            |> Nx.broadcast({Nx.axis_size(t, 0), seq_len})
          end,
          name: "dec_pos_indices"
        ),
        max_dec_len,
        hidden_dim,
        name: "dec_pos_embed"
      )

    x = Axon.add(token_embed, pos_embed_table, name: "dec_embed_sum")

    # Decoder transformer blocks (self-attn + cross-attn + FFN)
    x =
      TransformerBlock.stack(x, encoder_output, decoder_layers,
        attention_fn: fn input_node, name ->
          causal_self_attention(input_node, hidden_dim, num_heads, name)
        end,
        cross_attention_fn: fn q_normed, enc_out, name ->
          CrossAttention.layer(q_normed, enc_out,
            hidden_size: hidden_dim,
            num_heads: num_heads,
            name: name
          )
        end,
        hidden_size: hidden_dim,
        custom_ffn: fn input_node, name ->
          FFN.layer(input_node,
            hidden_size: hidden_dim,
            inner_size: ffn_dim,
            name: name
          )
        end,
        dropout: dropout,
        name: "dec"
      )

    # Final layer norm + projection to vocab
    x
    |> Axon.layer_norm(name: "dec_final_norm")
    |> Axon.dense(vocab_size, name: "dec_output_proj")
  end

  @doc """
  Get the output vocabulary size.

  ## Options
    - `:vocab_size` - Token vocabulary size (default: 51865)
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :vocab_size, @default_vocab_size)
  end

  # ============================================================================
  # Attention Helpers
  # ============================================================================

  # Standard multi-head self-attention (bidirectional, for encoder).
  @spec self_attention(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp self_attention(input, hidden_dim, num_heads, name) do
    q = Axon.dense(input, hidden_dim, name: "#{name}_q")
    k = Axon.dense(input, hidden_dim, name: "#{name}_k")
    v = Axon.dense(input, hidden_dim, name: "#{name}_v")

    head_dim = div(hidden_dim, num_heads)

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          compute_mha(q_t, k_t, v_t, num_heads, head_dim, nil)
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :self_attention
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  # Causal multi-head self-attention (for decoder).
  @spec causal_self_attention(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defp causal_self_attention(input, hidden_dim, num_heads, name) do
    q = Axon.dense(input, hidden_dim, name: "#{name}_q")
    k = Axon.dense(input, hidden_dim, name: "#{name}_k")
    v = Axon.dense(input, hidden_dim, name: "#{name}_v")

    head_dim = div(hidden_dim, num_heads)

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          seq_len = Nx.axis_size(q_t, 1)
          mask = Edifice.Blocks.CausalMask.causal(seq_len)
          compute_mha(q_t, k_t, v_t, num_heads, head_dim, mask)
        end,
        [q, k, v],
        name: "#{name}_compute",
        op_name: :causal_self_attention
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out")
  end

  # Shared multi-head attention computation. Handles reshaping to heads,
  # scaled dot-product attention with optional mask, and reshaping back.
  @spec compute_mha(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer(),
          Nx.Tensor.t() | nil
        ) ::
          Nx.Tensor.t()
  defp compute_mha(q, k, v, num_heads, head_dim, mask) do
    {batch, q_len, _} = Nx.shape(q)
    {_, kv_len, _} = Nx.shape(k)

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, q_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, kv_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Apply causal mask if provided
    scores =
      if mask do
        # mask: [q_len, kv_len] → broadcast to [batch, heads, q_len, kv_len]
        mask =
          mask
          |> Nx.reshape({1, 1, q_len, kv_len})
          |> Nx.broadcast({batch, num_heads, q_len, kv_len})

        neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
        Nx.select(mask, scores, neg_inf)
      else
        scores
      end

    weights = FusedOps.fused_softmax(scores)

    # Apply to values: [batch, heads, q_len, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, q_len, hidden_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, q_len, num_heads * head_dim})
  end
end
