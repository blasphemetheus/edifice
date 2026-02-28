defmodule Edifice.Vision.DINOv3 do
  @moduledoc """
  DINOv3: Self-supervised vision backbone with axial RoPE and text alignment.

  Implements DINOv3 from "DINOv3: Visual Foundation Models with Scalable
  Self-Supervised Pre-Training" (Meta AI, 2025). Major upgrade over DINOv2
  with axial 2D RoPE, LayerScale, iBOT patch-level distillation, and
  Sinkhorn-Knopp centering.

  ## Key Innovations over DINOv2

  - **Axial 2D RoPE**: Rotary position embedding split across spatial axes
    (theta=100.0), replacing learnable position embeddings. Only applied to
    patch tokens, not CLS or register tokens.
  - **LayerScale**: Learnable per-channel scaling on attention and FFN outputs
    (initialized to 1.0). Stabilizes deep ViT training.
  - **Asymmetric QKV bias**: Q and V projections use bias, K does not.
  - **SwiGLU FFN**: Optional gated FFN for larger model variants.
  - **iBOT loss**: Patch-level self-distillation alongside CLS-level DINO loss.
  - **Sinkhorn-Knopp centering**: Doubly-stochastic assignment normalization
    for teacher outputs (replaces simple mean centering).
  - **Gram anchoring**: Regularizes prototype weight Gram matrices.

  ## Architecture

  ```
  Image [batch, C, H, W]
        |
  +================+
  |  Patch Embed   |  (patch_size=16, not 14)
  +================+
        |
  +================+
  | [CLS] + [REG]  |  (register tokens, no position embedding)
  | + Patch Tokens |
  +================+
        |
  +================+
  | Transformer    |  Pre-LN, Axial 2D RoPE on patches only,
  | Blocks x N     |  LayerScale, SwiGLU or MLP FFN
  +================+
        |
  +---------+---------+
  |                   |
  CLS token       Patch tokens
  |                   |
  DINO Head        iBOT Head
  (MLP+L2+protos)  (MLP+L2+protos)
  |                   |
  DINO logits      iBOT logits
  ```

  ## Usage

      # Build student and teacher
      {student, teacher} = DINOv3.build(
        image_size: 224,
        patch_size: 16,
        embed_dim: 384,
        num_heads: 6,
        num_layers: 12
      )

      # Output is %{dino: [batch, num_dino_protos], ibot: [batch, num_patches, num_ibot_protos]}

      # Compute losses
      dino_loss = DINOv3.dino_loss(student_dino, teacher_dino)
      ibot_loss = DINOv3.ibot_loss(student_ibot, teacher_ibot, mask)

  ## References

  - "DINOv3: Visual Foundation Models with Scalable Self-Supervised Pre-Training"
  - DINOv2: https://arxiv.org/abs/2304.07193
  - iBOT: https://arxiv.org/abs/2111.07832
  """

  @behaviour Edifice.Vision.Backbone

  import Nx.Defn

  alias Edifice.Blocks.{PatchEmbed, SwiGLU}

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_embed_dim 384
  @default_num_heads 6
  @default_num_layers 12
  @default_mlp_ratio 4.0
  @default_num_register_tokens 4
  @default_ffn_type :mlp
  @default_rope_theta 100.0
  @default_dino_hidden_dim 2048
  @default_dino_bottleneck_dim 256
  @default_dino_num_prototypes 65_536
  @default_ibot_hidden_dim 2048
  @default_ibot_bottleneck_dim 256
  @default_ibot_num_prototypes 8192
  @default_student_temp 0.1
  @default_teacher_temp 0.04
  @default_momentum 0.996

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build both student and teacher DINOv3 networks.

  Returns `{student_model, teacher_model}` tuple. Each model outputs a
  container `%{dino: logits, ibot: patch_logits}` when heads are included.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Embedding dimension (default: 384)
    - `:num_heads` - Number of attention heads (default: 6)
    - `:num_layers` - Number of transformer blocks (default: 12)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_register_tokens` - Number of register tokens (default: 4)
    - `:ffn_type` - Feed-forward type: `:mlp` or `:swiglu` (default: `:mlp`)
    - `:rope_theta` - RoPE base frequency (default: 100.0)
    - `:dino_hidden_dim` - DINO head hidden dimension (default: 2048)
    - `:dino_bottleneck_dim` - DINO head bottleneck dimension (default: 256)
    - `:dino_num_prototypes` - DINO prototype count (default: 65536)
    - `:ibot_hidden_dim` - iBOT head hidden dimension (default: 2048)
    - `:ibot_bottleneck_dim` - iBOT head bottleneck dimension (default: 256)
    - `:ibot_num_prototypes` - iBOT prototype count (default: 8192)
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dino_bottleneck_dim, pos_integer()}
          | {:dino_hidden_dim, pos_integer()}
          | {:dino_num_prototypes, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:ffn_type, :mlp | :swiglu}
          | {:ibot_bottleneck_dim, pos_integer()}
          | {:ibot_hidden_dim, pos_integer()}
          | {:ibot_num_prototypes, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_register_tokens, non_neg_integer()}
          | {:patch_size, pos_integer()}
          | {:rope_theta, float()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    student = build_backbone(Keyword.put(opts, :prefix, "student"))
    teacher = build_backbone(Keyword.put(opts, :prefix, "teacher"))
    {student, teacher}
  end

  @doc """
  Build a single DINOv3 backbone (ViT with axial RoPE + projection heads).

  When called via the `Edifice.Vision.Backbone` behaviour, pass
  `include_head: false` to get raw `[batch, embed_dim]` features.

  ## Options

    Same as `build/1`, plus:
    - `:prefix` - Layer name prefix ("student" or "teacher")
    - `:include_head` - Whether to include projection heads (default: true)

  When `include_head: true`, returns `Axon.container(%{dino: ..., ibot: ...})`.
  When `include_head: false`, returns CLS token features `[batch, embed_dim]`.
  """
  @impl Edifice.Vision.Backbone
  @spec build_backbone(keyword()) :: Axon.t()
  def build_backbone(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    num_register_tokens = Keyword.get(opts, :num_register_tokens, @default_num_register_tokens)
    ffn_type = Keyword.get(opts, :ffn_type, @default_ffn_type)
    rope_theta = Keyword.get(opts, :rope_theta, @default_rope_theta)
    dino_hidden = Keyword.get(opts, :dino_hidden_dim, @default_dino_hidden_dim)
    dino_bottleneck = Keyword.get(opts, :dino_bottleneck_dim, @default_dino_bottleneck_dim)
    dino_prototypes = Keyword.get(opts, :dino_num_prototypes, @default_dino_num_prototypes)
    ibot_hidden = Keyword.get(opts, :ibot_hidden_dim, @default_ibot_hidden_dim)
    ibot_bottleneck = Keyword.get(opts, :ibot_bottleneck_dim, @default_ibot_bottleneck_dim)
    ibot_prototypes = Keyword.get(opts, :ibot_num_prototypes, @default_ibot_num_prototypes)
    prefix = Keyword.get(opts, :prefix, "dino3")
    include_head = Keyword.get(opts, :include_head, true)

    num_patches = PatchEmbed.num_patches(image_size, patch_size)
    grid_h = div(image_size, patch_size)
    grid_w = div(image_size, patch_size)
    num_special = 1 + num_register_tokens

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding: [batch, num_patches, embed_dim]
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: embed_dim,
        name: "#{prefix}_patch_embed"
      )

    # Prepend CLS token and register tokens: [batch, 1 + num_reg + num_patches, embed_dim]
    x = prepend_special_tokens(x, embed_dim, num_register_tokens, name: "#{prefix}_tokens")

    # NO learnable position embedding — axial 2D RoPE replaces it

    # Transformer blocks with axial RoPE + LayerScale
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        transformer_block(acc,
          embed_dim: embed_dim,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          ffn_type: ffn_type,
          num_special: num_special,
          grid_h: grid_h,
          grid_w: grid_w,
          rope_theta: rope_theta,
          name: "#{prefix}_block_#{idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "#{prefix}_final_norm")

    if include_head do
      # Extract CLS token: [batch, embed_dim]
      cls =
        Axon.nx(
          x,
          fn tensor ->
            Nx.slice_along_axis(tensor, 0, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "#{prefix}_extract_cls"
        )

      # DINO head on CLS token
      dino_out =
        build_projection_head(cls,
          hidden_dim: dino_hidden,
          bottleneck_dim: dino_bottleneck,
          num_prototypes: dino_prototypes,
          name: "#{prefix}_dino_head"
        )

      # Extract patch tokens: [batch, num_patches, embed_dim]
      patches =
        Axon.nx(
          x,
          fn tensor ->
            Nx.slice_along_axis(tensor, num_special, num_patches, axis: 1)
          end,
          name: "#{prefix}_extract_patches"
        )

      # iBOT head on patch tokens (applied per-patch)
      ibot_out =
        build_projection_head(patches,
          hidden_dim: ibot_hidden,
          bottleneck_dim: ibot_bottleneck,
          num_prototypes: ibot_prototypes,
          name: "#{prefix}_ibot_head"
        )

      Axon.container(%{dino: dino_out, ibot: ibot_out})
    else
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, 1, axis: 1) |> Nx.squeeze(axes: [1])
        end,
        name: "#{prefix}_extract_cls"
      )
    end
  end

  # ============================================================================
  # Special Tokens (CLS + Register) — same as DINOv2
  # ============================================================================

  defp prepend_special_tokens(input, embed_dim, num_register_tokens, opts) do
    name = Keyword.get(opts, :name, "tokens")

    # CLS token via dense projection of constant
    cls_source =
      Axon.nx(input, fn _tensor -> Nx.broadcast(1.0, {1, 1}) end, name: "#{name}_cls_src")

    cls_proj = Axon.dense(cls_source, embed_dim, name: "#{name}_cls_proj")

    if num_register_tokens > 0 do
      reg_source =
        Axon.nx(
          input,
          fn _tensor ->
            Nx.iota({1, num_register_tokens}, axis: 1) |> Nx.divide(num_register_tokens)
          end,
          name: "#{name}_reg_src"
        )

      reg_proj =
        Axon.dense(reg_source, num_register_tokens * embed_dim, name: "#{name}_reg_proj")

      Axon.layer(
        &prepend_tokens_impl/4,
        [input, cls_proj, reg_proj],
        name: "#{name}_prepend",
        num_register_tokens: num_register_tokens,
        op_name: :prepend_tokens
      )
    else
      Axon.layer(
        &prepend_cls_impl/3,
        [input, cls_proj],
        name: "#{name}_prepend_cls",
        op_name: :prepend_cls
      )
    end
  end

  defp prepend_tokens_impl(patches, cls_token, reg_tokens, opts) do
    batch_size = Nx.axis_size(patches, 0)
    embed_dim = Nx.axis_size(patches, 2)
    num_register_tokens = opts[:num_register_tokens]

    cls = Nx.reshape(cls_token, {1, 1, embed_dim}) |> Nx.broadcast({batch_size, 1, embed_dim})

    reg =
      Nx.reshape(reg_tokens, {1, num_register_tokens, embed_dim})
      |> Nx.broadcast({batch_size, num_register_tokens, embed_dim})

    Nx.concatenate([cls, reg, patches], axis: 1)
  end

  defp prepend_cls_impl(patches, cls_token, _opts) do
    batch_size = Nx.axis_size(patches, 0)
    embed_dim = Nx.axis_size(cls_token, 1)

    cls = Nx.reshape(cls_token, {1, 1, embed_dim}) |> Nx.broadcast({batch_size, 1, embed_dim})
    Nx.concatenate([cls, patches], axis: 1)
  end

  # ============================================================================
  # Transformer Block (Axial RoPE + LayerScale)
  # ============================================================================

  defp transformer_block(input, opts) do
    name = Keyword.fetch!(opts, :name)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_heads = Keyword.fetch!(opts, :num_heads)
    mlp_ratio = Keyword.fetch!(opts, :mlp_ratio)
    ffn_type = Keyword.fetch!(opts, :ffn_type)
    num_special = Keyword.fetch!(opts, :num_special)
    grid_h = Keyword.fetch!(opts, :grid_h)
    grid_w = Keyword.fetch!(opts, :grid_w)
    rope_theta = Keyword.fetch!(opts, :rope_theta)

    head_dim = div(embed_dim, num_heads)
    mlp_hidden = round(embed_dim * mlp_ratio)

    # Pre-norm self-attention
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    # Asymmetric QKV: Q with bias, K without bias, V with bias
    q = Axon.dense(normed, embed_dim, name: "#{name}_q_proj")
    k = Axon.dense(normed, embed_dim, use_bias: false, name: "#{name}_k_proj")
    v = Axon.dense(normed, embed_dim, name: "#{name}_v_proj")

    # Self-attention with axial 2D RoPE on patch tokens
    attended =
      Axon.layer(
        &axial_rope_attention_impl/4,
        [q, k, v],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        num_special: num_special,
        grid_h: grid_h,
        grid_w: grid_w,
        rope_theta: rope_theta,
        op_name: :axial_rope_attention
      )

    # Output projection + LayerScale
    attn_out = Axon.dense(attended, embed_dim, name: "#{name}_out_proj")
    attn_scaled = layer_scale(attn_out, embed_dim, name: "#{name}_ls1")

    x = Axon.add(input, attn_scaled, name: "#{name}_res1")

    # Pre-norm FFN
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      case ffn_type do
        :swiglu ->
          SwiGLU.layer(normed2, hidden_size: embed_dim, name: "#{name}_ffn")

        :mlp ->
          normed2
          |> Axon.dense(mlp_hidden, name: "#{name}_ffn_fc1")
          |> Axon.activation(:gelu, name: "#{name}_ffn_gelu")
          |> Axon.dense(embed_dim, name: "#{name}_ffn_fc2")
      end

    # LayerScale + residual
    ffn_scaled = layer_scale(ffn, embed_dim, name: "#{name}_ls2")
    Axon.add(x, ffn_scaled, name: "#{name}_res2")
  end

  # ============================================================================
  # LayerScale
  # ============================================================================

  # Learnable per-channel scaling initialized to 1.0.
  # Implemented as dense(1→dim) with kernel initialized to ones.
  defp layer_scale(input, dim, opts) do
    name = Keyword.get(opts, :name, "layer_scale")

    gamma_src =
      Axon.nx(input, fn _ -> Nx.broadcast(1.0, {1, 1}) end, name: "#{name}_src")

    gamma =
      Axon.dense(gamma_src, dim,
        use_bias: false,
        kernel_initializer: :ones,
        name: "#{name}_gamma"
      )

    Axon.layer(
      fn x, g, _opts ->
        # g: [1, dim], x: [batch, seq, dim] — broadcasts across batch and seq
        Nx.multiply(x, g)
      end,
      [input, gamma],
      name: name,
      op_name: :layer_scale
    )
  end

  # ============================================================================
  # Attention with Axial 2D RoPE
  # ============================================================================

  # Q, K, V: [batch, seq, hidden] → attended: [batch, seq, hidden]
  # RoPE is applied only to patch tokens (not CLS/register tokens).
  defp axial_rope_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    num_special = opts[:num_special]
    grid_h = opts[:grid_h]
    grid_w = opts[:grid_w]
    rope_theta = opts[:rope_theta]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)
    num_patches = grid_h * grid_w

    # Reshape to [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Split special tokens and patch tokens along seq axis
    q_special = Nx.slice_along_axis(q, 0, num_special, axis: 2)
    q_patch = Nx.slice_along_axis(q, num_special, num_patches, axis: 2)
    k_special = Nx.slice_along_axis(k, 0, num_special, axis: 2)
    k_patch = Nx.slice_along_axis(k, num_special, num_patches, axis: 2)

    # Apply axial 2D RoPE to patch tokens
    {q_patch_rot, k_patch_rot} =
      apply_axial_2d_rope(q_patch, k_patch, grid_h, grid_w, head_dim, rope_theta)

    # Reassemble: [special, rotated_patches]
    q = Nx.concatenate([q_special, q_patch_rot], axis: 2)
    k = Nx.concatenate([k_special, k_patch_rot], axis: 2)

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Numerically stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply to values
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Apply axial 2D RoPE to patch Q and K tensors.
  # Splits head_dim into two halves: first half rotated by y-position,
  # second half rotated by x-position.
  # q, k: [batch, heads, num_patches, head_dim]
  defp apply_axial_2d_rope(q, k, grid_h, grid_w, head_dim, theta) do
    half_dim = div(head_dim, 2)
    quarter_dim = div(half_dim, 2)
    num_patches = grid_h * grid_w

    # Frequencies: theta^(-2i/half_dim) for each axis
    freqs =
      Nx.pow(
        theta,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({quarter_dim})), half_dim))
      )
      |> Nx.as_type(Nx.type(q))

    # Y positions: [0,0,...,1,1,...,h-1,h-1,...] each repeated w times
    y_pos =
      Nx.iota({grid_h, grid_w}, axis: 0)
      |> Nx.reshape({num_patches})
      |> Nx.as_type(Nx.type(q))

    # X positions: [0,1,...,w-1,0,1,...,w-1,...] repeated h times
    x_pos =
      Nx.iota({grid_h, grid_w}, axis: 1)
      |> Nx.reshape({num_patches})
      |> Nx.as_type(Nx.type(q))

    # Angles: [num_patches, quarter_dim]
    y_angles = Nx.outer(y_pos, freqs)
    x_angles = Nx.outer(x_pos, freqs)

    # cos/sin tables: [1, 1, num_patches, quarter_dim] for broadcasting
    y_cos = Nx.cos(y_angles) |> Nx.reshape({1, 1, num_patches, quarter_dim})
    y_sin = Nx.sin(y_angles) |> Nx.reshape({1, 1, num_patches, quarter_dim})
    x_cos = Nx.cos(x_angles) |> Nx.reshape({1, 1, num_patches, quarter_dim})
    x_sin = Nx.sin(x_angles) |> Nx.reshape({1, 1, num_patches, quarter_dim})

    # Split into y-half and x-half
    q_y = Nx.slice_along_axis(q, 0, half_dim, axis: 3)
    q_x = Nx.slice_along_axis(q, half_dim, half_dim, axis: 3)
    k_y = Nx.slice_along_axis(k, 0, half_dim, axis: 3)
    k_x = Nx.slice_along_axis(k, half_dim, half_dim, axis: 3)

    # Apply 1D RoPE to y-half (rotary half formulation)
    {q_y_rot, k_y_rot} = rotate_half_pair(q_y, k_y, y_cos, y_sin, quarter_dim)

    # Apply 1D RoPE to x-half
    {q_x_rot, k_x_rot} = rotate_half_pair(q_x, k_x, x_cos, x_sin, quarter_dim)

    # Reassemble: [y_half_rotated, x_half_rotated]
    q_rot = Nx.concatenate([q_y_rot, q_x_rot], axis: 3)
    k_rot = Nx.concatenate([k_y_rot, k_x_rot], axis: 3)

    {q_rot, k_rot}
  end

  # Apply rotary half formulation to a pair of Q/K tensors.
  # Each tensor has shape [batch, heads, seq, dim], cos/sin have [1, 1, seq, dim/2].
  defp rotate_half_pair(q, k, cos_table, sin_table, half) do
    q1 = Nx.slice_along_axis(q, 0, half, axis: 3)
    q2 = Nx.slice_along_axis(q, half, half, axis: 3)

    q_rot =
      Nx.concatenate(
        [
          Nx.subtract(Nx.multiply(q1, cos_table), Nx.multiply(q2, sin_table)),
          Nx.add(Nx.multiply(q1, sin_table), Nx.multiply(q2, cos_table))
        ],
        axis: 3
      )

    k1 = Nx.slice_along_axis(k, 0, half, axis: 3)
    k2 = Nx.slice_along_axis(k, half, half, axis: 3)

    k_rot =
      Nx.concatenate(
        [
          Nx.subtract(Nx.multiply(k1, cos_table), Nx.multiply(k2, sin_table)),
          Nx.add(Nx.multiply(k1, sin_table), Nx.multiply(k2, cos_table))
        ],
        axis: 3
      )

    {q_rot, k_rot}
  end

  # ============================================================================
  # Projection Heads
  # ============================================================================

  # Builds a DINO/iBOT projection head: MLP → L2 norm → prototype projection.
  # Works on both [batch, dim] (CLS) and [batch, seq, dim] (patches).
  defp build_projection_head(input, opts) do
    hidden_dim = Keyword.fetch!(opts, :hidden_dim)
    bottleneck_dim = Keyword.fetch!(opts, :bottleneck_dim)
    num_prototypes = Keyword.fetch!(opts, :num_prototypes)
    name = Keyword.get(opts, :name, "head")

    x =
      input
      |> Axon.dense(hidden_dim, name: "#{name}_fc1")
      |> Axon.activation(:gelu, name: "#{name}_gelu1")
      |> Axon.dense(hidden_dim, name: "#{name}_fc2")
      |> Axon.activation(:gelu, name: "#{name}_gelu2")
      |> Axon.dense(bottleneck_dim, name: "#{name}_fc3")

    # L2 normalize
    x = Axon.nx(x, &l2_normalize_tensor/1, name: "#{name}_l2_norm")

    # Prototype projection (no bias)
    Axon.dense(x, num_prototypes, use_bias: false, name: "#{name}_prototypes")
  end

  defp l2_normalize_tensor(x) do
    norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(x, x), axes: [-1], keep_axes: true), 1.0e-8))
    Nx.divide(x, norm)
  end

  # ============================================================================
  # DINO Loss (with Sinkhorn-Knopp centering)
  # ============================================================================

  @doc """
  Compute the DINO self-distillation loss with Sinkhorn-Knopp centering.

  Cross-entropy between student and teacher CLS token distributions.
  Teacher targets are computed via Sinkhorn-Knopp normalization for
  balanced prototype assignment.

  ## Parameters

    - `student_out` - Student logits: [batch, num_prototypes]
    - `teacher_out` - Teacher logits: [batch, num_prototypes]

  ## Options

    - `:student_temp` - Student temperature (default: 0.1)
    - `:teacher_temp` - Teacher temperature (default: 0.04)

  ## Returns

    Scalar loss tensor.
  """
  @spec dino_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def dino_loss(student_out, teacher_out, opts \\ []) do
    student_temp = Keyword.get(opts, :student_temp, @default_student_temp)
    teacher_temp = Keyword.get(opts, :teacher_temp, @default_teacher_temp)
    dino_loss_impl(student_out, teacher_out, student_temp, teacher_temp)
  end

  defnp dino_loss_impl(student_out, teacher_out, student_temp, teacher_temp) do
    # Teacher: Sinkhorn-Knopp soft assignments
    teacher_probs = sinkhorn_knopp_impl(teacher_out, teacher_temp)

    # Student: log-softmax with temperature
    student_log_probs = log_softmax_with_temp(student_out, student_temp)

    # Cross-entropy
    -Nx.mean(Nx.sum(teacher_probs * student_log_probs, axes: [1]))
  end

  # ============================================================================
  # iBOT Loss (patch-level self-distillation)
  # ============================================================================

  @doc """
  Compute the iBOT patch-level self-distillation loss.

  Cross-entropy between student and teacher patch token predictions,
  applied only to masked patch positions.

  ## Parameters

    - `student_patch` - Student patch logits: [batch, num_patches, num_prototypes]
    - `teacher_patch` - Teacher patch logits: [batch, num_patches, num_prototypes]
    - `mask` - Boolean mask: [batch, num_patches] where `true` = masked (predict)

  ## Options

    - `:student_temp` - Student temperature (default: 0.1)
    - `:teacher_temp` - Teacher temperature (default: 0.04)

  ## Returns

    Scalar loss tensor.
  """
  @spec ibot_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def ibot_loss(student_patch, teacher_patch, mask, opts \\ []) do
    student_temp = Keyword.get(opts, :student_temp, @default_student_temp)
    teacher_temp = Keyword.get(opts, :teacher_temp, @default_teacher_temp)
    ibot_loss_impl(student_patch, teacher_patch, mask, student_temp, teacher_temp)
  end

  defnp ibot_loss_impl(student_patch, teacher_patch, mask, student_temp, teacher_temp) do
    batch = Nx.axis_size(student_patch, 0)
    num_patches = Nx.axis_size(student_patch, 1)
    num_protos = Nx.axis_size(student_patch, 2)

    # Flatten to [batch * num_patches, num_prototypes]
    s_flat = Nx.reshape(student_patch, {batch * num_patches, num_protos})
    t_flat = Nx.reshape(teacher_patch, {batch * num_patches, num_protos})
    mask_flat = Nx.reshape(mask, {batch * num_patches})

    # Teacher: softmax with temperature
    t_probs = softmax_with_temp(t_flat, teacher_temp)

    # Student: log-softmax with temperature
    s_log_probs = log_softmax_with_temp(s_flat, student_temp)

    # Per-token cross-entropy
    per_token_loss = -Nx.sum(t_probs * s_log_probs, axes: [1])

    # Apply mask (only masked tokens contribute)
    masked_loss = per_token_loss * mask_flat
    num_masked = Nx.sum(mask_flat) + 1.0e-8

    Nx.sum(masked_loss) / num_masked
  end

  # ============================================================================
  # Gram Anchoring Loss
  # ============================================================================

  @doc """
  Compute the Gram anchoring loss for prototype weight regularization.

  Penalizes deviation of the prototype Gram matrix from an anchor,
  preventing prototype collapse or drift during training.

  ## Parameters

    - `prototypes` - Current prototype weights: [num_prototypes, dim]
    - `anchor` - Fixed anchor prototype weights: [num_prototypes, dim]

  ## Returns

    Scalar loss tensor.
  """
  @spec gram_anchoring_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn gram_anchoring_loss(prototypes, anchor) do
    # L2 normalize rows
    p_norm = l2_normalize_rows(prototypes)
    a_norm = l2_normalize_rows(anchor)

    # Gram matrices: [K, K]
    gram_p = Nx.dot(p_norm, [1], p_norm, [1])
    gram_a = Nx.dot(a_norm, [1], a_norm, [1])

    # Frobenius norm of difference
    diff = gram_p - gram_a
    Nx.mean(diff * diff)
  end

  defnp l2_normalize_rows(x) do
    norm = Nx.sqrt(Nx.sum(x * x, axes: [1], keep_axes: true) + 1.0e-8)
    x / norm
  end

  # ============================================================================
  # KoLeo Regularizer
  # ============================================================================

  @doc """
  Compute KoLeo regularizer (Kozachenko-Leonenko entropy estimator).

  Encourages uniform distribution of patch representations in feature space.

  ## Parameters

    - `patch_tokens` - Patch token representations: [batch, num_patches, embed_dim]

  ## Returns

    Scalar KoLeo loss (negative, to maximize entropy).
  """
  @spec koleo_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn koleo_loss(patch_tokens) do
    batch = Nx.axis_size(patch_tokens, 0)
    num_patches = Nx.axis_size(patch_tokens, 1)
    embed_dim = Nx.axis_size(patch_tokens, 2)

    flat = Nx.reshape(patch_tokens, {batch * num_patches, embed_dim})
    flat_normed = l2_normalize_rows(flat)

    # Pairwise cosine similarities
    similarities = Nx.dot(flat_normed, [1], flat_normed, [1])

    # Mask out self-similarity
    n = batch * num_patches
    mask = Nx.eye(n)
    similarities = similarities - mask * 1.0e10

    # Nearest neighbor distance
    max_sim = Nx.reduce_max(similarities, axes: [1])
    min_dist = 1.0 - max_sim + 1.0e-8

    -Nx.mean(Nx.log(min_dist))
  end

  # ============================================================================
  # EMA Teacher Update
  # ============================================================================

  @doc """
  Update teacher network parameters via exponential moving average.

  teacher = momentum * teacher + (1 - momentum) * student

  ## Parameters

    - `student_params` - Student network parameters (map of tensors)
    - `teacher_params` - Teacher network parameters (map of tensors)

  ## Options

    - `:momentum` - EMA momentum (default: 0.996)
  """
  @spec update_teacher(map(), map(), keyword()) :: map()
  def update_teacher(student_params, teacher_params, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, @default_momentum)

    Map.new(teacher_params, fn {key, teacher_val} ->
      student_key = String.replace(key, "teacher_", "student_", global: false)

      updated =
        case Map.fetch(student_params, student_key) do
          {:ok, student_val} -> ema_blend(student_val, teacher_val, momentum)
          :error -> teacher_val
        end

      {key, updated}
    end)
  end

  defp ema_blend(student, teacher, momentum)
       when is_map(student) and not is_struct(student) and is_map(teacher) and
              not is_struct(teacher) do
    Map.new(teacher, fn {k, t_v} ->
      case Map.fetch(student, k) do
        {:ok, s_v} -> {k, ema_blend(s_v, t_v, momentum)}
        :error -> {k, t_v}
      end
    end)
  end

  defp ema_blend(student, teacher, momentum) do
    Nx.add(Nx.multiply(momentum, teacher), Nx.multiply(1.0 - momentum, student))
  end

  # ============================================================================
  # Sinkhorn-Knopp Centering
  # ============================================================================

  # Computes doubly-stochastic soft assignments via 3 iterations of
  # row/column normalization. Input: [batch, K] logits. Output: [batch, K] probs.
  defnp sinkhorn_knopp_impl(logits, temp) do
    # Exponentiate and transpose: [K, batch]
    q = Nx.exp(logits / temp) |> Nx.transpose()

    k = Nx.axis_size(q, 0)
    b = Nx.axis_size(q, 1)

    # Global normalization
    q = q / (Nx.sum(q) + 1.0e-8)

    # 3 Sinkhorn iterations (unrolled)
    q = q / (Nx.sum(q, axes: [1], keep_axes: true) * k + 1.0e-8)
    q = q / (Nx.sum(q, axes: [0], keep_axes: true) * b + 1.0e-8)

    q = q / (Nx.sum(q, axes: [1], keep_axes: true) * k + 1.0e-8)
    q = q / (Nx.sum(q, axes: [0], keep_axes: true) * b + 1.0e-8)

    q = q / (Nx.sum(q, axes: [1], keep_axes: true) * k + 1.0e-8)
    q = q / (Nx.sum(q, axes: [0], keep_axes: true) * b + 1.0e-8)

    # Transpose back and scale: [batch, K]
    Nx.transpose(q) * b
  end

  defnp softmax_with_temp(x, temp) do
    x_scaled = x / temp
    max_x = Nx.reduce_max(x_scaled, axes: [1], keep_axes: true)
    exp_x = Nx.exp(x_scaled - max_x)
    exp_x / (Nx.sum(exp_x, axes: [1], keep_axes: true) + 1.0e-8)
  end

  defnp log_softmax_with_temp(x, temp) do
    x_scaled = x / temp
    max_x = Nx.reduce_max(x_scaled, axes: [1], keep_axes: true)
    x_scaled - max_x - Nx.log(Nx.sum(Nx.exp(x_scaled - max_x), axes: [1], keep_axes: true))
  end

  # ============================================================================
  # Model Size Presets
  # ============================================================================

  @doc """
  Get recommended defaults for different model sizes.
  """
  @spec recommended_defaults(atom()) :: keyword()
  def recommended_defaults(size \\ :small) do
    case size do
      :small ->
        [
          embed_dim: 384,
          num_heads: 6,
          num_layers: 12,
          patch_size: 16,
          num_register_tokens: 4,
          ffn_type: :mlp
        ]

      :base ->
        [
          embed_dim: 768,
          num_heads: 12,
          num_layers: 12,
          patch_size: 16,
          num_register_tokens: 4,
          ffn_type: :mlp
        ]

      :large ->
        [
          embed_dim: 1024,
          num_heads: 16,
          num_layers: 24,
          patch_size: 16,
          num_register_tokens: 4,
          ffn_type: :swiglu
        ]

      :huge ->
        [
          embed_dim: 1280,
          num_heads: 20,
          num_layers: 32,
          patch_size: 16,
          num_register_tokens: 4,
          ffn_type: :swiglu
        ]

      :giant ->
        [
          embed_dim: 4096,
          num_heads: 32,
          num_layers: 40,
          patch_size: 16,
          num_register_tokens: 4,
          ffn_type: :swiglu
        ]
    end
  end

  # ============================================================================
  # Backbone Behaviour
  # ============================================================================

  @impl Edifice.Vision.Backbone
  @spec feature_size(keyword()) :: pos_integer()
  def feature_size(opts \\ []) do
    Keyword.get(opts, :embed_dim, @default_embed_dim)
  end

  @impl Edifice.Vision.Backbone
  @spec input_shape(keyword()) :: tuple()
  def input_shape(opts \\ []) do
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    {nil, in_channels, image_size, image_size}
  end
end
