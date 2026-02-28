defmodule Edifice.Vision.Janus do
  @moduledoc """
  Janus — Unified Multimodal Understanding and Generation.

  <!-- verified: true, date: 2026-02-28 -->

  Decouples visual encoding for understanding and generation: a ViT-style
  encoder processes images for comprehension, while a separate VQ-codebook
  embedding handles discrete image generation tokens. Both feed into a
  shared autoregressive LLM backbone (decoder-only Transformer).

  ## Architecture

  ```
  Understanding path:
    Image [batch, n_patches, patch_dim]
        |
    ViT Encoder (N layers of self-attention)
        |
    MLP Aligner (proj to LLM hidden)
        |
    visual_tokens [batch, n_patches, hidden]
        |
    Interleave with text tokens -> LLM

  Generation path:
    VQ token IDs [batch, n_tokens]
        |
    Codebook Embedding
        |
    MLP Aligner (proj to LLM hidden)
        |
    image_tokens [batch, n_tokens, hidden]
        |
    Interleave with text tokens -> LLM -> Gen Head
        |
    vision_head: Dense -> GELU -> Dense(codebook_size)
        |
    codebook_logits [batch, seq, codebook_size]
  ```

  ## Returns

  `{understanding_encoder, generation_head}` tuple:
  - Understanding encoder: `[batch, n_patches, patch_dim]` -> `[batch, n_patches, hidden]`
  - Generation head: `[batch, seq, hidden]` -> `[batch, seq, codebook_size]`

  The LLM backbone is assumed external (e.g. `Edifice.Transformer.DecoderOnly`).
  These modules provide the visual interface layers.

  ## References

  - Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal
    Understanding and Generation" (CVPR 2025)
  - https://arxiv.org/abs/2410.13848
  """

  alias Edifice.Blocks.{SDPA, TransformerBlock}

  @default_hidden_size 2048
  @default_patch_dim 1024
  @default_num_patches 576
  @default_vit_layers 6
  @default_vit_heads 16
  @default_vit_ffn_mult 4
  @default_aligner_depth 2
  @default_codebook_size 16_384
  @default_gen_embed_dim 8
  @default_gen_head_intermediate 1024

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:aligner_depth, pos_integer()}
          | {:codebook_size, pos_integer()}
          | {:gen_embed_dim, pos_integer()}
          | {:gen_head_intermediate, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_patches, pos_integer()}
          | {:patch_dim, pos_integer()}
          | {:vit_ffn_mult, pos_integer()}
          | {:vit_heads, pos_integer()}
          | {:vit_layers, pos_integer()}

  @doc """
  Build Janus visual interface layers.

  ## Options

    - `:hidden_size` - LLM hidden dimension (default: 2048)
    - `:patch_dim` - ViT patch feature dimension (default: 1024)
    - `:num_patches` - Number of image patches (default: 576)
    - `:vit_layers` - ViT encoder depth (default: 6)
    - `:vit_heads` - ViT attention heads (default: 16)
    - `:vit_ffn_mult` - ViT FFN multiplier (default: 4)
    - `:aligner_depth` - MLP aligner depth (default: 2)
    - `:codebook_size` - VQ codebook entries (default: 16384)
    - `:gen_embed_dim` - Generation token embed dim (default: 8)
    - `:gen_head_intermediate` - Gen head hidden dim (default: 1024)

  ## Returns

    `{understanding_encoder, generation_head}` — two Axon models.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    hidden = Keyword.get(opts, :hidden_size, @default_hidden_size)
    patch_dim = Keyword.get(opts, :patch_dim, @default_patch_dim)
    num_patches = Keyword.get(opts, :num_patches, @default_num_patches)
    vit_layers = Keyword.get(opts, :vit_layers, @default_vit_layers)
    vit_heads = Keyword.get(opts, :vit_heads, @default_vit_heads)
    vit_ffn_mult = Keyword.get(opts, :vit_ffn_mult, @default_vit_ffn_mult)
    aligner_depth = Keyword.get(opts, :aligner_depth, @default_aligner_depth)
    codebook_size = Keyword.get(opts, :codebook_size, @default_codebook_size)
    gen_embed_dim = Keyword.get(opts, :gen_embed_dim, @default_gen_embed_dim)

    gen_head_intermediate =
      Keyword.get(opts, :gen_head_intermediate, @default_gen_head_intermediate)

    understanding =
      build_understanding_encoder(
        patch_dim,
        num_patches,
        hidden,
        vit_layers,
        vit_heads,
        vit_ffn_mult,
        aligner_depth
      )

    gen_head =
      build_generation_head(
        hidden,
        codebook_size,
        gen_embed_dim,
        gen_head_intermediate,
        aligner_depth
      )

    {understanding, gen_head}
  end

  @doc "Get the output size for the generation head."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :codebook_size, @default_codebook_size)
  end

  # ===========================================================================
  # Understanding Encoder: ViT + MLP Aligner
  # ===========================================================================

  defp build_understanding_encoder(
         patch_dim,
         num_patches,
         hidden,
         layers,
         heads,
         ffn_mult,
         aligner_depth
       ) do
    input = Axon.input("patches", shape: {nil, num_patches, patch_dim})

    head_dim = div(patch_dim, heads)

    # Simple self-attention based ViT encoder
    x =
      Enum.reduce(0..(layers - 1), input, fn i, acc ->
        TransformerBlock.layer(acc,
          attention_fn: fn inp, name ->
            self_attention(inp, patch_dim, heads, head_dim, name)
          end,
          hidden_size: patch_dim,
          ffn_expansion: ffn_mult,
          name: "vit_#{i}"
        )
      end)

    # MLP aligner: project ViT features to LLM hidden dimension
    mlp_aligner(x, patch_dim, hidden, aligner_depth, "understand_aligner")
  end

  # ===========================================================================
  # Generation Head: Hidden -> Codebook Logits
  # ===========================================================================

  defp build_generation_head(hidden, codebook_size, _gen_embed_dim, intermediate, _aligner_depth) do
    input = Axon.input("hidden_states", shape: {nil, nil, hidden})

    # vision_head: Dense -> GELU -> Dense(codebook_size)
    input
    |> Axon.dense(intermediate, name: "gen_head_proj")
    |> Axon.activation(:gelu, name: "gen_head_gelu")
    |> Axon.dense(codebook_size, name: "gen_head_logits")
  end

  # ===========================================================================
  # MLP Aligner (mlp_gelu projector)
  # ===========================================================================

  defp mlp_aligner(x, _in_dim, out_dim, depth, name) do
    x = Axon.dense(x, out_dim, name: "#{name}_proj_0")

    Enum.reduce(1..(depth - 1)//1, x, fn i, acc ->
      acc
      |> Axon.activation(:gelu, name: "#{name}_gelu_#{i}")
      |> Axon.dense(out_dim, name: "#{name}_proj_#{i}")
    end)
  end

  # ===========================================================================
  # Self-Attention (Q/K/V projections + SDPA)
  # ===========================================================================

  defp self_attention(input, dim, heads, head_dim, name) do
    q = Axon.dense(input, dim, name: "#{name}_q")
    k = Axon.dense(input, dim, name: "#{name}_k")
    v = Axon.dense(input, dim, name: "#{name}_v")

    attended =
      Axon.layer(
        fn q_t, k_t, v_t, _opts ->
          SDPA.compute(q_t, k_t, v_t, heads, head_dim)
        end,
        [q, k, v],
        name: "#{name}_sdpa",
        op_name: :self_attention
      )

    Axon.dense(attended, dim, name: "#{name}_out")
  end
end
