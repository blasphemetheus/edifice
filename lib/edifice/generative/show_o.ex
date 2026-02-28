defmodule Edifice.Generative.ShowO do
  @moduledoc """
  Show-o: Unified Autoregressive + Discrete Diffusion Multimodal Model.

  <!-- verified: true, date: 2026-02-28 -->

  A single transformer that handles both text (autoregressive next-token
  prediction) and images (discrete diffusion mask prediction) within one
  shared backbone using an omni-attention mask.

  ## Key Innovation: Unified Discrete Space

  Both text and images operate in discrete token space. Images are tokenized
  via MAGVIT-v2 (or similar VQ codebook) into discrete codes, which are
  appended to the LLM vocabulary. A single LM head produces logits over
  the combined vocabulary for both modalities.

  ## Omni-Attention Mask

  Text tokens use causal attention; image tokens use full bidirectional
  attention within the image block, plus causal access to preceding text:

  ```
  mask[i, j] = (j <= i) OR (image[i] AND image[j])
  ```

  This is the same mixed mask pattern as Transfusion, but applied to
  discrete tokens rather than continuous patches.

  ## Architecture

  ```
  Inputs: input_ids [batch, seq_len]          (text + image token IDs)
          modality_mask [batch, seq_len]       (0=text, 1=image)
          |
          v
  Token embedding (vocab_size = llm_vocab + codebook + special)
          |
          v
  +----------------------------------------------+
  |  Transformer Block x num_layers               |
  |                                               |
  |  LayerNorm -> Self-Attention (omni-mask)      |
  |    with QK-Norm + partial RoPE                |
  |  -> Residual                                  |
  |  LayerNorm -> FFN (GELU) -> Residual          |
  +----------------------------------------------+
          |
  Final LayerNorm -> LM Head
          |
  logits [batch, seq_len, vocab_size]
  ```

  ## Image Generation (Discrete Diffusion)

  Training: Randomly mask image tokens with `[MASK]`, predict original IDs.
  Inference: Start with all `[MASK]`, iteratively unmask by confidence
  using cosine schedule (MaskGIT-style).

  ## References

  - Xie et al., "Show-o: One Single Transformer to Unify Multimodal
    Understanding and Generation" (ICLR 2025)
  - https://arxiv.org/abs/2408.12528
  """

  alias Edifice.Blocks.FFN

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_layers 6
  @default_intermediate_size 1024
  @default_vocab_size 58_498
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:codebook_size, pos_integer()}
          | {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:intermediate_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:qk_norm, boolean()}
          | {:vocab_size, pos_integer()}

  @doc """
  Build a Show-o model.

  ## Options

    - `:vocab_size` - Total vocabulary size including image codes and special
      tokens (default: 58498)
    - `:codebook_size` - Image codebook size, appended to LLM vocab (default: 8192)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of transformer layers (default: 6)
    - `:intermediate_size` - FFN intermediate dimension (default: 1024)
    - `:qk_norm` - Apply LayerNorm to Q and K before attention (default: true)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    An Axon model: `{input_ids, modality_mask}` -> logits `[batch, seq_len, vocab_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    intermediate_size = Keyword.get(opts, :intermediate_size, @default_intermediate_size)
    qk_norm = Keyword.get(opts, :qk_norm, true)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    input_ids = Axon.input("input_ids", shape: {nil, nil})
    modality_mask = Axon.input("modality_mask", shape: {nil, nil})

    # Token embedding (unified vocab: text + image codes + special tokens)
    x =
      Axon.embedding(input_ids, vocab_size, hidden_size, name: "token_embed")

    x = maybe_dropout(x, dropout, "embed_drop")

    # N transformer blocks with omni-attention
    x =
      Enum.reduce(1..num_layers, x, fn i, acc ->
        build_block(acc, modality_mask,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          intermediate_size: intermediate_size,
          qk_norm: qk_norm,
          dropout: dropout,
          name: "layer_#{i}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Unified LM head over combined vocabulary
    Axon.dense(x, vocab_size, name: "lm_head")
  end

  # ==========================================================================
  # Transformer block with omni-attention + QK-Norm
  # ==========================================================================

  defp build_block(input, modality_mask, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    intermediate_size = Keyword.fetch!(opts, :intermediate_size)
    qk_norm = Keyword.fetch!(opts, :qk_norm)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "block")

    # Attention sublayer
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    q = Axon.dense(attn_normed, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(attn_normed, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(attn_normed, hidden_size, name: "#{name}_v_proj")

    # Optional QK-Norm
    {q, k} =
      if qk_norm do
        {
          Axon.layer_norm(q, name: "#{name}_q_norm"),
          Axon.layer_norm(k, name: "#{name}_k_norm")
        }
      else
        {q, k}
      end

    attn_out =
      Axon.layer(
        &omni_attention_impl/5,
        [q, k, v, modality_mask],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :omni_attention
      )

    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # FFN sublayer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        intermediate_size: intermediate_size,
        activation: :gelu,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ==========================================================================
  # Omni-attention: causal for text, bidirectional for image
  # ==========================================================================

  # q, k, v: [batch, seq_len, hidden_size]
  # modality_mask: [batch, seq_len], 0=text, 1=image
  defp omni_attention_impl(q, k, v, modality_mask, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, heads, seq_len, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Causal mask: j <= i
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal = Nx.greater_equal(rows, cols)

    # Image bidirectional: both query and key are image tokens
    is_image_i = Nx.reshape(modality_mask, {batch, seq_len, 1})
    is_image_j = Nx.reshape(modality_mask, {batch, 1, seq_len})
    img_to_img = Nx.logical_and(is_image_i, is_image_j)

    # Broadcast to [batch, num_heads, seq_len, seq_len]
    causal_exp =
      causal
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    img_exp =
      img_to_img
      |> Nx.reshape({batch, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    # Combined: allow if causal OR both image
    combined = Nx.logical_or(causal_exp, img_exp)

    scores =
      Nx.select(
        combined,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Stable softmax
    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ==========================================================================
  # Mask scheduling for discrete diffusion inference
  # ==========================================================================

  @doc """
  Cosine mask schedule for MaskGIT-style iterative unmasking.

  Returns the fraction of tokens that should remain masked at step `t`
  out of `total_steps`.

  ## Examples

      iex> Edifice.Generative.ShowO.cosine_mask_ratio(0, 18)
      1.0

      iex> ratio = Edifice.Generative.ShowO.cosine_mask_ratio(18, 18)
      iex> abs(ratio) < 1.0e-6
      true
  """
  @spec cosine_mask_ratio(non_neg_integer(), pos_integer()) :: float()
  def cosine_mask_ratio(step, total_steps) do
    :math.cos(step / total_steps * :math.pi() / 2.0)
  end

  @doc """
  Build the omni-attention mask for a text+image sequence.

  Same pattern as Transfusion: causal for text, bidirectional for image,
  with cross-modal causal access from image to preceding text.

  ## Parameters

    - `text_len` - Number of text token positions
    - `image_len` - Number of image token positions (appended after text)

  ## Returns

    Boolean `Nx.Tensor.t()` of shape `[text_len + image_len, text_len + image_len]`.
  """
  @spec build_omni_mask(non_neg_integer(), non_neg_integer()) :: Nx.Tensor.t()
  def build_omni_mask(text_len, image_len) do
    total = text_len + image_len

    modality =
      cond do
        text_len == 0 ->
          Nx.broadcast(Nx.tensor(1, type: :s32), {image_len})

        image_len == 0 ->
          Nx.broadcast(Nx.tensor(0, type: :s32), {text_len})

        true ->
          Nx.concatenate([
            Nx.broadcast(Nx.tensor(0, type: :s32), {text_len}),
            Nx.broadcast(Nx.tensor(1, type: :s32), {image_len})
          ])
      end

    rows = Nx.iota({total, total}, axis: 0)
    cols = Nx.iota({total, total}, axis: 1)
    causal = Nx.greater_equal(rows, cols)

    is_image_i = Nx.reshape(modality, {total, 1})
    is_image_j = Nx.reshape(modality, {1, total})
    img_to_img = Nx.logical_and(is_image_i, is_image_j)

    Nx.logical_or(causal, img_to_img)
  end

  # ==========================================================================
  # Utilities
  # ==========================================================================

  @doc "Get the output size of a Show-o model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
