defmodule Edifice.Generative.MMDiT do
  @moduledoc """
  MMDiT: Multimodal Diffusion Transformer.

  Implements the MMDiT architecture from "Scaling Rectified Flow Transformers
  for High-Resolution Image Synthesis" (Esser et al., 2024), used in Stable
  Diffusion 3 and FLUX.1. Replaces DiT's single-stream design with dual
  parallel streams (one per modality) connected via joint self-attention.

  ## Key Innovation: Joint Attention (Not Cross-Attention)

  Instead of cross-attention between modalities, MMDiT concatenates Q, K, V
  from both streams along the sequence dimension, runs a single standard
  self-attention, then splits the output back. This allows all four
  interaction directions (I2I, I2T, T2I, T2T) in every layer.

  ```
  Image stream:  img_q, img_k, img_v = img_attn(AdaLN(img))
  Text stream:   txt_q, txt_k, txt_v = txt_attn(AdaLN(txt))

  Combined:      q = cat([txt_q, img_q], dim=seq)
                 k = cat([txt_k, img_k], dim=seq)
                 v = cat([txt_v, img_v], dim=seq)

  Joint attn:    out = softmax(q @ k^T / sqrt(d)) @ v

  Split back:    txt_out = out[:, :txt_len]
                 img_out = out[:, txt_len:]
  ```

  ## Architecture

  ```
  Image Latent [batch, img_tokens, img_dim]     Text Embed [batch, txt_tokens, txt_dim]
        |                                              |
        v                                              v
  [Image Projection]                            [Text Projection]
        |                                              |
        v                                              v
  +-------------------------------------------------------------------+
  |  Double-Stream Block x depth                                       |
  |                                                                    |
  |  img_stream:  AdaLN(vec) -> QKV_img ---+                          |
  |  txt_stream:  AdaLN(vec) -> QKV_txt ---+-> Joint Attention        |
  |                                        |                          |
  |  img_stream:  gate * proj(img_attn) + residual -> AdaLN -> MLP   |
  |  txt_stream:  gate * proj(txt_attn) + residual -> AdaLN -> MLP   |
  +-------------------------------------------------------------------+
        |
        v
  [Final Norm + Linear] -> output [batch, img_tokens, img_dim]
  ```

  ## Conditioning (AdaLN-Zero)

  Each stream has separate modulation weights. The conditioning vector is:
  `vec = timestep_mlp(sinusoidal(t)) + pooled_text_mlp(pooled_text)`

  Each modulation produces 6 parameters per stream:
  (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)

  ## Usage

      model = MMDiT.build(
        img_dim: 16,           # VAE latent channels (flattened patch dim)
        txt_dim: 4096,         # Text encoder hidden dim (e.g., T5-XXL)
        hidden_size: 1536,     # Joint hidden dimension
        depth: 24,             # Number of double-stream blocks
        num_heads: 24,
        img_tokens: 256,       # Number of image patches
        txt_tokens: 77         # Max text sequence length
      )

  ## References
  - SD3: https://arxiv.org/abs/2403.03206
  - FLUX.1: https://github.com/black-forest-labs/flux
  """

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_hidden_size 768
  @default_depth 12
  @default_num_heads 12
  @default_mlp_ratio 4
  @default_img_tokens 64
  @default_txt_tokens 32

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an MMDiT model for multimodal diffusion.

  ## Options

  **Modality dimensions (at least one pair required):**
    - `:img_dim` - Image patch feature dimension (required)
    - `:txt_dim` - Text token feature dimension (required)
    - `:img_tokens` - Number of image tokens/patches (default: 64)
    - `:txt_tokens` - Max text tokens (default: 32)

  **Architecture:**
    - `:hidden_size` - Joint hidden dimension (default: 768)
    - `:depth` - Number of double-stream blocks (default: 12)
    - `:num_heads` - Attention heads (default: 12)
    - `:mlp_ratio` - MLP expansion ratio (default: 4)
    - `:cond_dim` - Conditioning vector dimension (default: hidden_size)

  ## Returns
    An Axon model: (img_latent, txt_embed, timestep, pooled_text) -> denoised_img
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:cond_dim, pos_integer()}
          | {:depth, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:img_dim, pos_integer()}
          | {:img_tokens, pos_integer()}
          | {:mlp_ratio, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:txt_dim, pos_integer()}
          | {:txt_tokens, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    img_dim = Keyword.fetch!(opts, :img_dim)
    txt_dim = Keyword.fetch!(opts, :txt_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    depth = Keyword.get(opts, :depth, @default_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    img_tokens = Keyword.get(opts, :img_tokens, @default_img_tokens)
    txt_tokens = Keyword.get(opts, :txt_tokens, @default_txt_tokens)
    cond_dim = Keyword.get(opts, :cond_dim, hidden_size)

    # Inputs
    img_input = Axon.input("img_latent", shape: {nil, img_tokens, img_dim})
    txt_input = Axon.input("txt_embed", shape: {nil, txt_tokens, txt_dim})
    timestep = Axon.input("timestep", shape: {nil})
    pooled_text = Axon.input("pooled_text", shape: {nil, cond_dim})

    # Build conditioning vector: vec = time_mlp(sinusoidal(t)) + text_mlp(pooled)
    time_embed = build_time_embedding(timestep, hidden_size)
    text_cond = Axon.dense(pooled_text, hidden_size, name: "pooled_text_mlp")
    vec = Axon.add(time_embed, text_cond, name: "cond_vec")

    # Project modalities to shared hidden dimension
    img = Axon.dense(img_input, hidden_size, name: "img_proj")
    txt = Axon.dense(txt_input, hidden_size, name: "txt_proj")

    # Stack double-stream blocks
    {img, txt} =
      Enum.reduce(1..depth, {img, txt}, fn block_idx, {img_acc, txt_acc} ->
        build_double_stream_block(
          img_acc,
          txt_acc,
          vec,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          txt_tokens: txt_tokens,
          name: "mmdit_block_#{block_idx}"
        )
      end)

    # Final layer: only image stream continues
    # AdaLN modulation + linear projection back to img_dim
    _txt = txt
    img = Axon.layer_norm(img, name: "final_norm")
    Axon.dense(img, img_dim, name: "final_proj")
  end

  # ============================================================================
  # Double-Stream Block
  # ============================================================================

  defp build_double_stream_block(img, txt, vec, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    mlp_ratio = Keyword.fetch!(opts, :mlp_ratio)
    txt_tokens = Keyword.fetch!(opts, :txt_tokens)
    name = Keyword.fetch!(opts, :name)

    head_dim = div(hidden_size, num_heads)
    ffn_dim = hidden_size * mlp_ratio

    # ---- Per-stream modulation from conditioning vector ----
    # Each produces 6 values: (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    img_mod = build_modulation(vec, hidden_size, "#{name}_img_mod")
    txt_mod = build_modulation(vec, hidden_size, "#{name}_txt_mod")

    # ---- Attention sub-block ----
    # Modulated layer norm (no learned affine): (1 + scale) * LN(x) + shift
    img_normed = build_adaln(img, img_mod, :attn, "#{name}_img_attn_norm")
    txt_normed = build_adaln(txt, txt_mod, :attn, "#{name}_txt_attn_norm")

    # Per-stream QKV projections
    img_qkv = Axon.dense(img_normed, hidden_size * 3, name: "#{name}_img_qkv")
    txt_qkv = Axon.dense(txt_normed, hidden_size * 3, name: "#{name}_txt_qkv")

    # Joint attention: concatenate along sequence dim, attend, split back
    joint_attn_out =
      Axon.layer(
        &joint_attention_impl/3,
        [img_qkv, txt_qkv],
        name: "#{name}_joint_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        txt_tokens: txt_tokens,
        op_name: :joint_attention
      )

    # Split joint attention output back to img and txt portions
    {img_attn_out, txt_attn_out} = split_joint_output(joint_attn_out, txt_tokens, name)

    # Per-stream output projections with gating
    img_attn_proj = Axon.dense(img_attn_out, hidden_size, name: "#{name}_img_attn_proj")
    txt_attn_proj = Axon.dense(txt_attn_out, hidden_size, name: "#{name}_txt_attn_proj")

    img_attn_gated = apply_gate(img_attn_proj, img_mod, :attn, "#{name}_img_attn_gate")
    txt_attn_gated = apply_gate(txt_attn_proj, txt_mod, :attn, "#{name}_txt_attn_gate")

    # Residual
    img = Axon.add(img, img_attn_gated, name: "#{name}_img_attn_res")
    txt = Axon.add(txt, txt_attn_gated, name: "#{name}_txt_attn_res")

    # ---- MLP sub-block ----
    img_mlp_in = build_adaln(img, img_mod, :mlp, "#{name}_img_mlp_norm")
    txt_mlp_in = build_adaln(txt, txt_mod, :mlp, "#{name}_txt_mlp_norm")

    img_mlp =
      img_mlp_in
      |> Axon.dense(ffn_dim, name: "#{name}_img_mlp_up")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_img_mlp_down")

    txt_mlp =
      txt_mlp_in
      |> Axon.dense(ffn_dim, name: "#{name}_txt_mlp_up")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_txt_mlp_down")

    img_mlp_gated = apply_gate(img_mlp, img_mod, :mlp, "#{name}_img_mlp_gate")
    txt_mlp_gated = apply_gate(txt_mlp, txt_mod, :mlp, "#{name}_txt_mlp_gate")

    img = Axon.add(img, img_mlp_gated, name: "#{name}_img_mlp_res")
    txt = Axon.add(txt, txt_mlp_gated, name: "#{name}_txt_mlp_res")

    {img, txt}
  end

  # ============================================================================
  # AdaLN Modulation
  # ============================================================================

  # Build modulation: vec -> SiLU -> Linear -> 6 * hidden_size
  # Returns a tensor of shape [batch, 6 * hidden_size]
  defp build_modulation(vec, hidden_size, name) do
    vec
    |> Axon.activation(:silu, name: "#{name}_silu")
    |> Axon.dense(hidden_size * 6, name: "#{name}_proj")
  end

  # Apply AdaLN: (1 + scale) * LayerNorm(x) + shift
  # mod_type is :attn (uses params 0-2) or :mlp (uses params 3-5)
  defp build_adaln(x, mod, mod_type, name) do
    normed = Axon.layer_norm(x, name: "#{name}_ln")

    Axon.layer(
      &adaln_modulate_impl/3,
      [normed, mod],
      name: name,
      mod_type: mod_type,
      op_name: :adaln_modulate
    )
  end

  defp adaln_modulate_impl(normed, mod, opts) do
    mod_type = opts[:mod_type]
    hidden_size = Nx.axis_size(normed, 2)

    # mod: [batch, 6 * hidden_size] -> chunk into 6 parts
    # :attn uses indices 0,1 for shift,scale; :mlp uses indices 3,4
    offset =
      case mod_type do
        :attn -> 0
        :mlp -> 3
      end

    shift = Nx.slice_along_axis(mod, offset * hidden_size, hidden_size, axis: 1)
    scale = Nx.slice_along_axis(mod, (offset + 1) * hidden_size, hidden_size, axis: 1)

    # Broadcast [batch, hidden] -> [batch, 1, hidden] for seq dim
    shift = Nx.new_axis(shift, 1)
    scale = Nx.new_axis(scale, 1)

    # (1 + scale) * normed + shift
    Nx.add(Nx.multiply(Nx.add(1.0, scale), normed), shift)
  end

  # Apply gating: gate * x
  defp apply_gate(x, mod, mod_type, name) do
    Axon.layer(
      &gate_impl/3,
      [x, mod],
      name: name,
      mod_type: mod_type,
      op_name: :adaln_gate
    )
  end

  defp gate_impl(x, mod, opts) do
    mod_type = opts[:mod_type]
    hidden_size = Nx.axis_size(x, 2)

    # gate is at index 2 (attn) or 5 (mlp)
    offset =
      case mod_type do
        :attn -> 2
        :mlp -> 5
      end

    gate = Nx.slice_along_axis(mod, offset * hidden_size, hidden_size, axis: 1)
    gate = Nx.new_axis(gate, 1)

    Nx.multiply(gate, x)
  end

  # ============================================================================
  # Joint Attention
  # ============================================================================

  defp joint_attention_impl(img_qkv, txt_qkv, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    hidden_size = num_heads * head_dim

    batch_size = Nx.axis_size(img_qkv, 0)
    img_len = Nx.axis_size(img_qkv, 1)
    txt_len = Nx.axis_size(txt_qkv, 1)

    # Split QKV for each stream
    img_q = Nx.slice_along_axis(img_qkv, 0, hidden_size, axis: 2)
    img_k = Nx.slice_along_axis(img_qkv, hidden_size, hidden_size, axis: 2)
    img_v = Nx.slice_along_axis(img_qkv, hidden_size * 2, hidden_size, axis: 2)

    txt_q = Nx.slice_along_axis(txt_qkv, 0, hidden_size, axis: 2)
    txt_k = Nx.slice_along_axis(txt_qkv, hidden_size, hidden_size, axis: 2)
    txt_v = Nx.slice_along_axis(txt_qkv, hidden_size * 2, hidden_size, axis: 2)

    # Concatenate along sequence dimension
    q = Nx.concatenate([txt_q, img_q], axis: 1)
    k = Nx.concatenate([txt_k, img_k], axis: 1)
    v = Nx.concatenate([txt_v, img_v], axis: 1)

    total_len = txt_len + img_len

    # Reshape to [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q =
      q
      |> Nx.reshape({batch_size, total_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k =
      k
      |> Nx.reshape({batch_size, total_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch_size, total_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.rsqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    attn_weights = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)

    attn_weights =
      Nx.exp(Nx.subtract(attn_weights, Nx.reduce_max(attn_weights, axes: [3], keep_axes: true)))

    attn_weights = Nx.divide(attn_weights, Nx.sum(attn_weights, axes: [3], keep_axes: true))

    # Apply to values
    out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])
    # [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    out |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch_size, total_len, hidden_size})
  end

  # Split the joint attention output back into txt and img portions
  defp split_joint_output(joint_out, txt_tokens, name) do
    txt_out =
      Axon.nx(joint_out, fn x -> Nx.slice_along_axis(x, 0, txt_tokens, axis: 1) end,
        name: "#{name}_split_txt"
      )

    img_out =
      Axon.nx(
        joint_out,
        fn x ->
          total = Nx.axis_size(x, 1)
          Nx.slice_along_axis(x, txt_tokens, total - txt_tokens, axis: 1)
        end,
        name: "#{name}_split_img"
      )

    {img_out, txt_out}
  end

  # ============================================================================
  # Time Embedding
  # ============================================================================

  defp build_time_embedding(timestep, hidden_size) do
    Axon.layer(
      &time_embedding_impl/2,
      [timestep],
      name: "time_embed",
      hidden_size: hidden_size,
      op_name: :time_embed
    )
  end

  defp time_embedding_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim)
        )
      )

    t_expanded = Nx.new_axis(t, -1)
    angles = Nx.multiply(t_expanded, freqs)
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: -1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an MMDiT model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    img_dim = Keyword.get(opts, :img_dim, 16)
    img_tokens = Keyword.get(opts, :img_tokens, @default_img_tokens)
    img_tokens * img_dim
  end

  @doc """
  Get recommended defaults for MMDiT.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      img_dim: 16,
      txt_dim: 4096,
      hidden_size: 1536,
      depth: 24,
      num_heads: 24,
      mlp_ratio: 4,
      img_tokens: 256,
      txt_tokens: 77
    ]
  end
end
