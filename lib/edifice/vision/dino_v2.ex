defmodule Edifice.Vision.DINOv2 do
  @moduledoc """
  DINOv2: Self-supervised vision backbone via self-distillation.

  Implements DINOv2 from "DINOv2: Learning Robust Visual Features without
  Supervision" (Oquab et al., Meta 2023). Learns powerful visual representations
  through self-distillation without labels, using a student-teacher framework
  with masked patch prediction.

  ## Key Innovations

  - **Self-distillation**: Student network learns to match teacher's output distribution
  - **EMA teacher**: Teacher parameters are exponential moving average of student
  - **Register tokens**: Learnable tokens (beyond CLS) that improve attention maps
  - **KoLeo regularizer**: Maximizes entropy of patch token distribution
  - **No labels needed**: Completely self-supervised pretraining

  ## Architecture

  ```
  Student (trained)                 Teacher (EMA, no grad)
        |                                 |
  Augmented/Masked Image            Full Image
        |                                 |
        v                                 v
  +================+              +================+
  |  Patch Embed   |              |  Patch Embed   |
  +================+              +================+
        |                                 |
        v                                 v
  +================+              +================+
  | [CLS] + [REG]  |              | [CLS] + [REG]  |  (register tokens)
  | + Patch Tokens |              | + Patch Tokens |
  +================+              +================+
        |                                 |
        v                                 v
  +================+              +================+
  | Position Embed |              | Position Embed |
  +================+              +================+
        |                                 |
        v                                 v
  +================+              +================+
  | Transformer    |              | Transformer    |
  | Blocks x N     |              | Blocks x N     |
  +================+              +================+
        |                                 |
        v                                 v
  Extract CLS token              Extract CLS token
        |                                 |
        v                                 v
  +================+              +================+
  |   DINO Head    |              |   DINO Head    |
  | (MLP + L2 norm)|              | (MLP + L2 norm)|
  +================+              +================+
        |                                 |
        v                                 v
      Student                          Teacher
   Distribution                     Distribution
        |                                 |
        +---------> DINO Loss <-----------+
                  (cross-entropy)
  ```

  ## DINO Loss

  Cross-entropy between student and teacher output distributions:
    - Teacher outputs are centered and sharpened (temperature τ_t)
    - Student outputs are just sharpened (temperature τ_s)
    - Teacher centering prevents collapse to uniform distribution

  ## Usage

      # Build student and teacher ViT backbones
      {student, teacher} = DINOv2.build(
        image_size: 224,
        patch_size: 14,
        embed_dim: 384,
        num_heads: 6,
        num_layers: 12,
        num_register_tokens: 4
      )

      # Compute DINO loss
      loss = DINOv2.dino_loss(student_out, teacher_out,
        student_temp: 0.1,
        teacher_temp: 0.04,
        center: center_tensor
      )

      # Update teacher via EMA after each step
      teacher_params = DINOv2.update_teacher(student_params, teacher_params, momentum: 0.996)

  ## References

  - Paper: "DINOv2: Learning Robust Visual Features without Supervision"
  - arXiv: https://arxiv.org/abs/2304.07193
  - Original DINO: "Emerging Properties in Self-Supervised Vision Transformers" (2021)
  """

  import Nx.Defn

  alias Edifice.Blocks.PatchEmbed

  @default_image_size 224
  @default_patch_size 14
  @default_in_channels 3
  @default_embed_dim 384
  @default_num_heads 6
  @default_num_layers 12
  @default_mlp_ratio 4.0
  @default_num_register_tokens 4
  @default_head_hidden_dim 2048
  @default_head_bottleneck_dim 256
  @default_head_output_dim 65_536
  @default_student_temp 0.1
  @default_teacher_temp 0.04
  @default_momentum 0.996

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build both student and teacher DINOv2 networks.

  Returns `{student_model, teacher_model}` tuple. The teacher should be
  updated via EMA after each training step using `update_teacher/3`.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 14)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Embedding dimension (default: 384)
    - `:num_heads` - Number of attention heads (default: 6)
    - `:num_layers` - Number of transformer blocks (default: 12)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_register_tokens` - Number of register tokens (default: 4)
    - `:head_hidden_dim` - DINO head hidden dimension (default: 2048)
    - `:head_bottleneck_dim` - DINO head bottleneck dimension (default: 256)
    - `:head_output_dim` - DINO head output dimension (default: 65536)

  ## Returns

    `{student_model, teacher_model}` tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:head_bottleneck_dim, pos_integer()}
          | {:head_hidden_dim, pos_integer()}
          | {:head_output_dim, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_register_tokens, non_neg_integer()}
          | {:patch_size, pos_integer()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    student = build_backbone(Keyword.put(opts, :prefix, "student"))
    teacher = build_backbone(Keyword.put(opts, :prefix, "teacher"))
    {student, teacher}
  end

  @doc """
  Build a single DINOv2 backbone (ViT with register tokens + DINO head).

  ## Options

    Same as `build/1`, plus:
    - `:prefix` - Layer name prefix ("student" or "teacher")
    - `:include_head` - Whether to include DINO head (default: true)
  """
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
    head_hidden = Keyword.get(opts, :head_hidden_dim, @default_head_hidden_dim)
    head_bottleneck = Keyword.get(opts, :head_bottleneck_dim, @default_head_bottleneck_dim)
    head_output = Keyword.get(opts, :head_output_dim, @default_head_output_dim)
    prefix = Keyword.get(opts, :prefix, "dino")
    include_head = Keyword.get(opts, :include_head, true)

    num_patches = PatchEmbed.num_patches(image_size, patch_size)
    mlp_hidden = round(embed_dim * mlp_ratio)

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

    # Total sequence length: CLS + registers + patches
    seq_len = 1 + num_register_tokens + num_patches

    # Add learnable position embeddings
    x = add_position_embedding(x, seq_len, embed_dim, name: "#{prefix}_pos_embed")

    # Transformer blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        transformer_block(acc, embed_dim, num_heads, mlp_hidden, name: "#{prefix}_block_#{idx}")
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "#{prefix}_final_norm")

    # Extract CLS token: [batch, embed_dim]
    cls_token =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "#{prefix}_extract_cls"
      )

    # DINO projection head
    if include_head do
      build_dino_head(cls_token,
        hidden_dim: head_hidden,
        bottleneck_dim: head_bottleneck,
        output_dim: head_output,
        name: "#{prefix}_head"
      )
    else
      cls_token
    end
  end

  # Build the DINO projection head: MLP -> bottleneck -> L2 norm -> output
  defp build_dino_head(input, opts) do
    hidden_dim = Keyword.fetch!(opts, :hidden_dim)
    bottleneck_dim = Keyword.fetch!(opts, :bottleneck_dim)
    output_dim = Keyword.fetch!(opts, :output_dim)
    name = Keyword.get(opts, :name, "dino_head")

    # 3-layer MLP with GELU
    x =
      input
      |> Axon.dense(hidden_dim, name: "#{name}_fc1")
      |> Axon.activation(:gelu, name: "#{name}_gelu1")
      |> Axon.dense(hidden_dim, name: "#{name}_fc2")
      |> Axon.activation(:gelu, name: "#{name}_gelu2")
      |> Axon.dense(bottleneck_dim, name: "#{name}_fc3")

    # L2 normalize the bottleneck features
    x =
      Axon.nx(x, fn tensor -> l2_normalize_tensor(tensor) end, name: "#{name}_l2_norm")

    # Final projection to output dimension
    Axon.dense(x, output_dim, use_bias: false, name: "#{name}_last_layer")
  end

  defp l2_normalize_tensor(x) do
    norm = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(x, x), axes: [-1], keep_axes: true), 1.0e-8))
    Nx.divide(x, norm)
  end

  # ============================================================================
  # Special Tokens (CLS + Register)
  # ============================================================================

  defp prepend_special_tokens(input, embed_dim, num_register_tokens, opts) do
    name = Keyword.get(opts, :name, "tokens")

    # Create learnable CLS token via dense projection
    cls_source =
      Axon.nx(input, fn _tensor -> Nx.broadcast(1.0, {1, 1}) end, name: "#{name}_cls_src")

    cls_proj = Axon.dense(cls_source, embed_dim, name: "#{name}_cls_proj")

    # Create learnable register tokens (if any)
    if num_register_tokens > 0 do
      reg_source =
        Axon.nx(
          input,
          fn _tensor ->
            Nx.iota({1, num_register_tokens}, axis: 1) |> Nx.divide(num_register_tokens)
          end,
          name: "#{name}_reg_src"
        )

      reg_proj = Axon.dense(reg_source, num_register_tokens * embed_dim, name: "#{name}_reg_proj")

      # Concatenate: [CLS, registers, patches]
      Axon.layer(
        &prepend_tokens_impl/4,
        [input, cls_proj, reg_proj],
        name: "#{name}_prepend",
        num_register_tokens: num_register_tokens,
        op_name: :prepend_tokens
      )
    else
      # Just CLS token
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

    # Expand CLS token: [1, embed_dim] -> [batch, 1, embed_dim]
    cls = Nx.reshape(cls_token, {1, 1, embed_dim})
    cls = Nx.broadcast(cls, {batch_size, 1, embed_dim})

    # Reshape register tokens: [1, num_reg * embed_dim] -> [batch, num_reg, embed_dim]
    reg = Nx.reshape(reg_tokens, {1, num_register_tokens, embed_dim})
    reg = Nx.broadcast(reg, {batch_size, num_register_tokens, embed_dim})

    Nx.concatenate([cls, reg, patches], axis: 1)
  end

  defp prepend_cls_impl(patches, cls_token, _opts) do
    batch_size = Nx.axis_size(patches, 0)
    embed_dim = Nx.axis_size(cls_token, 1)

    cls = Nx.reshape(cls_token, {1, 1, embed_dim})
    cls = Nx.broadcast(cls, {batch_size, 1, embed_dim})

    Nx.concatenate([cls, patches], axis: 1)
  end

  defp add_position_embedding(input, seq_len, embed_dim, opts) do
    name = Keyword.get(opts, :name, "pos_embed")

    pos_source =
      Axon.nx(input, fn _tensor -> Nx.iota({1, seq_len}, axis: 1) |> Nx.divide(seq_len) end,
        name: "#{name}_src"
      )

    pos_proj = Axon.dense(pos_source, embed_dim, name: "#{name}_proj")

    Axon.layer(
      &add_embedding_impl/3,
      [input, pos_proj],
      name: "#{name}_add",
      op_name: :add_pos_embed
    )
  end

  defp add_embedding_impl(input, pos_embed, _opts) do
    Nx.add(input, pos_embed)
  end

  # ============================================================================
  # Transformer Block
  # ============================================================================

  defp transformer_block(input, embed_dim, num_heads, mlp_hidden, opts) do
    name = Keyword.get(opts, :name, "block")

    # Pre-norm self-attention
    normed = Axon.layer_norm(input, name: "#{name}_norm1")
    attended = self_attention(normed, embed_dim, num_heads, name: "#{name}_attn")
    x = Axon.add(input, attended, name: "#{name}_residual1")

    # Pre-norm MLP
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      normed2
      |> Axon.dense(mlp_hidden, name: "#{name}_mlp_fc1")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> Axon.dense(embed_dim, name: "#{name}_mlp_fc2")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  defp self_attention(input, embed_dim, num_heads, opts) do
    name = Keyword.get(opts, :name, "attn")
    head_dim = div(embed_dim, num_heads)

    qkv = Axon.dense(input, embed_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.layer(
        &mha_impl/2,
        [qkv],
        name: "#{name}_compute",
        embed_dim: embed_dim,
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :multi_head_attention
      )

    Axon.dense(attended, embed_dim, name: "#{name}_proj")
  end

  defp mha_impl(qkv, opts) do
    embed_dim = opts[:embed_dim]
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(qkv, 0)
    seq_len = Nx.axis_size(qkv, 1)

    # Split into Q, K, V
    q = Nx.slice_along_axis(qkv, 0, embed_dim, axis: 2)
    k = Nx.slice_along_axis(qkv, embed_dim, embed_dim, axis: 2)
    v = Nx.slice_along_axis(qkv, embed_dim * 2, embed_dim, axis: 2)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply attention to values
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # DINO Loss
  # ============================================================================

  @doc """
  Compute the DINO self-distillation loss.

  Cross-entropy between student and teacher output distributions, where:
  - Teacher outputs are centered (prevents collapse) and sharpened
  - Student outputs are sharpened with higher temperature

  ## Parameters

    - `student_out` - Student network output: [batch, output_dim]
    - `teacher_out` - Teacher network output: [batch, output_dim]

  ## Options

    - `:student_temp` - Student temperature (default: 0.1)
    - `:teacher_temp` - Teacher temperature (default: 0.04)
    - `:center` - Running center for teacher outputs (optional, zeros if not provided)

  ## Returns

    Scalar loss tensor.
  """
  @spec dino_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def dino_loss(student_out, teacher_out, opts \\ []) do
    student_temp = Keyword.get(opts, :student_temp, @default_student_temp)
    teacher_temp = Keyword.get(opts, :teacher_temp, @default_teacher_temp)
    output_dim = Nx.axis_size(teacher_out, 1)
    center = Keyword.get_lazy(opts, :center, fn -> Nx.broadcast(0.0, {output_dim}) end)

    dino_loss_impl(student_out, teacher_out, student_temp, teacher_temp, center)
  end

  defnp dino_loss_impl(student_out, teacher_out, student_temp, teacher_temp, center) do
    # Teacher: center and sharpen
    teacher_centered = teacher_out - center
    teacher_probs = softmax_with_temp(teacher_centered, teacher_temp)

    # Student: just sharpen
    student_log_probs = log_softmax_with_temp(student_out, student_temp)

    # Cross-entropy loss
    -Nx.mean(Nx.sum(teacher_probs * student_log_probs, axes: [1]))
  end

  defnp softmax_with_temp(x, temp) do
    x_scaled = x / temp
    max_x = Nx.reduce_max(x_scaled, axes: [1], keep_axes: true)
    exp_x = Nx.exp(x_scaled - max_x)
    exp_x / Nx.sum(exp_x, axes: [1], keep_axes: true)
  end

  defnp log_softmax_with_temp(x, temp) do
    x_scaled = x / temp
    max_x = Nx.reduce_max(x_scaled, axes: [1], keep_axes: true)
    x_scaled - max_x - Nx.log(Nx.sum(Nx.exp(x_scaled - max_x), axes: [1], keep_axes: true))
  end

  @doc """
  Update the running center for teacher outputs.

  The center is an exponential moving average of teacher outputs,
  used to prevent collapse to uniform distribution.

  ## Parameters

    - `teacher_out` - Current batch teacher output: [batch, output_dim]
    - `center` - Current center: [output_dim]

  ## Options

    - `:momentum` - Center update momentum (default: 0.9)

  ## Returns

    Updated center tensor.
  """
  @spec update_center(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def update_center(teacher_out, center, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, 0.9)
    update_center_impl(teacher_out, center, momentum)
  end

  defnp update_center_impl(teacher_out, center, momentum) do
    batch_center = Nx.mean(teacher_out, axes: [0])
    momentum * center + (1.0 - momentum) * batch_center
  end

  # ============================================================================
  # KoLeo Regularizer
  # ============================================================================

  @doc """
  Compute KoLeo regularizer (Kozachenko-Leonenko entropy estimator).

  Encourages uniform distribution of patch representations in feature space.
  KoLeo = mean(log(distance to nearest neighbor)).

  ## Parameters

    - `patch_tokens` - Patch token representations: [batch, num_patches, embed_dim]

  ## Returns

    Scalar KoLeo loss (negative, to maximize entropy).
  """
  @spec koleo_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn koleo_loss(patch_tokens) do
    # Flatten batch and patches: [batch * num_patches, embed_dim]
    batch = Nx.axis_size(patch_tokens, 0)
    num_patches = Nx.axis_size(patch_tokens, 1)
    embed_dim = Nx.axis_size(patch_tokens, 2)

    flat = Nx.reshape(patch_tokens, {batch * num_patches, embed_dim})

    # L2 normalize
    norm = Nx.sqrt(Nx.sum(flat * flat, axes: [1], keep_axes: true) + 1.0e-8)
    flat_normed = flat / norm

    # Pairwise distances (using dot product for normalized vectors)
    similarities = Nx.dot(flat_normed, [1], flat_normed, [1])

    # Set diagonal to -inf so self isn't nearest neighbor
    n = batch * num_patches
    mask = Nx.eye(n)
    similarities = similarities - mask * 1.0e10

    # Find max similarity (nearest neighbor)
    max_sim = Nx.reduce_max(similarities, axes: [1])

    # Convert to distance (1 - similarity for normalized vectors)
    min_dist = 1.0 - max_sim + 1.0e-8

    # KoLeo: mean log distance
    -Nx.mean(Nx.log(min_dist))
  end

  # ============================================================================
  # EMA Update
  # ============================================================================

  @doc """
  Update teacher network parameters via exponential moving average.

  teacher = momentum * teacher + (1 - momentum) * student

  ## Parameters

    - `student_params` - Student network parameters (map of tensors)
    - `teacher_params` - Teacher network parameters (map of tensors)

  ## Options

    - `:momentum` - EMA momentum (default: 0.996)

  ## Returns

    Updated teacher parameters.
  """
  @spec update_teacher(map(), map(), keyword()) :: map()
  def update_teacher(student_params, teacher_params, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, @default_momentum)

    Map.new(teacher_params, fn {key, teacher_val} ->
      student_key = String.replace(key, "teacher_", "student_", global: false)

      updated =
        case Map.fetch(student_params, student_key) do
          {:ok, student_val} ->
            ema_blend(student_val, teacher_val, momentum)

          :error ->
            teacher_val
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
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a DINOv2 model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :head_output_dim, @default_head_output_dim)
  end

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
          patch_size: 14,
          num_register_tokens: 4
        ]

      :base ->
        [
          embed_dim: 768,
          num_heads: 12,
          num_layers: 12,
          patch_size: 14,
          num_register_tokens: 4
        ]

      :large ->
        [
          embed_dim: 1024,
          num_heads: 16,
          num_layers: 24,
          patch_size: 14,
          num_register_tokens: 4
        ]

      :giant ->
        [
          embed_dim: 1536,
          num_heads: 24,
          num_layers: 40,
          patch_size: 14,
          num_register_tokens: 4
        ]
    end
  end
end
