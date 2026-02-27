defmodule Edifice.Generative.LinearDiT do
  @moduledoc """
  Linear DiT / SANA: Diffusion Transformer with Linear Attention.

  Implements DiT (Diffusion Transformer) architecture with linear attention
  replacing the quadratic softmax attention. This achieves comparable image
  quality at dramatically reduced computational cost.

  ## Key Innovation: Linear Attention in Diffusion

  Standard DiT uses O(N²) softmax attention, which becomes prohibitive for
  high-resolution images. Linear DiT replaces this with O(N) linear attention
  using kernel feature maps, enabling:

  - 100x speedup for high-resolution generation
  - Same quality as quadratic DiT
  - Scalable to 4K+ resolution images

  ## Architecture

  ```
  Input [batch, num_patches, patch_dim]
        |
        v
  +---------------------------+
  | Patchify + Position Embed |
  +---------------------------+
        |
        v
  +---------------------------+
  | Linear DiT Block x depth  |
  |  AdaLN-Zero(condition)    |
  |  Linear Attention         |  <- O(N) instead of O(N²)
  |  Residual                 |
  |  AdaLN-Zero(condition)    |
  |  MLP                      |
  |  Residual                 |
  +---------------------------+
        |
        v
  | Final AdaLN + Linear     |
        |
        v
  Output [batch, num_patches, patch_dim]
  ```

  ## Linear Attention Mechanism

  Standard: `Attn(Q,K,V) = softmax(QK^T/sqrt(d)) * V`  [O(N²)]

  Linear: `Attn(Q,K,V) = phi(Q) * (phi(K)^T * V) / (phi(Q) * sum(phi(K)))` [O(N)]

  Where phi(x) = ELU(x) + 1 ensures non-negative attention weights.

  ## Usage

      model = LinearDiT.build(
        input_dim: 64,
        hidden_size: 512,
        num_layers: 12,
        num_heads: 8,
        patch_size: 2
      )

  ## References

  - SANA: "Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer" (2024)
  - DiT: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
  - Linear Attention: "Transformers are RNNs" (Katharopoulos et al., 2020)
  """

  @default_hidden_size 512
  @default_num_layers 12
  @default_num_heads 8
  @default_mlp_ratio 4.0
  @default_num_steps 1000
  @default_patch_size 2

  @doc """
  Build a Linear DiT model for diffusion denoising with linear attention.

  ## Options

    - `:input_dim` - Input/output feature dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 512)
    - `:num_layers` - Number of DiT blocks (default: 12)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_classes` - Number of classes for conditioning (optional, nil = unconditional)
    - `:num_steps` - Number of diffusion timesteps (default: 1000)
    - `:patch_size` - Patch size for spatial inputs (default: 2)

  ## Returns

    An Axon model that predicts noise given (noisy_input, timestep, [class]).
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_steps, pos_integer()}
          | {:patch_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    num_classes = Keyword.get(opts, :num_classes, nil)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)

    # Inputs
    noisy_input = Axon.input("noisy_input", shape: {nil, input_dim})
    timestep = Axon.input("timestep", shape: {nil})

    # Timestep embedding: sinusoidal -> MLP
    time_embed = build_timestep_mlp(timestep, hidden_size, num_steps)

    # Optional class conditioning
    condition =
      if num_classes do
        class_label = Axon.input("class_label", shape: {nil})
        class_embed = build_class_embedding(class_label, hidden_size, num_classes)
        Axon.add(time_embed, class_embed, name: "condition_combine")
      else
        time_embed
      end

    # Project input to hidden dimension
    x = Axon.dense(noisy_input, hidden_size, name: "input_embed")

    # Add learnable position embedding
    x = Axon.bias(x, name: "pos_embed")

    # Linear DiT blocks
    x =
      Enum.reduce(1..num_layers, x, fn block_idx, acc ->
        build_linear_dit_block(acc, condition,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          name: "linear_dit_block_#{block_idx}"
        )
      end)

    # Final layer: AdaLN + linear projection
    x = Axon.layer_norm(x, name: "final_norm")
    Axon.dense(x, input_dim, name: "output_proj")
  end

  @doc """
  Build a single Linear DiT block with AdaLN-Zero conditioning and linear attention.
  """
  @spec build_linear_dit_block(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def build_linear_dit_block(input, condition, opts) do
    alias Edifice.Blocks.AdaptiveNorm

    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "linear_dit_block")
    mlp_dim = round(hidden_size * mlp_ratio)

    # AdaLN parameters for attention sub-layer: shift1, scale1, gate1
    adaln_attn = Axon.dense(condition, hidden_size * 3, name: "#{name}_adaln_attn")

    # AdaLN-Zero modulated attention
    x_modulated =
      input
      |> Axon.layer_norm(name: "#{name}_attn_norm")
      |> AdaptiveNorm.modulate(adaln_attn,
        hidden_size: hidden_size,
        offset: 0,
        name: "#{name}_attn_mod"
      )

    # Linear self-attention (the key difference from standard DiT)
    attn_out = build_linear_attention(x_modulated, hidden_size, num_heads, "#{name}_attn")

    attn_out =
      AdaptiveNorm.gate(attn_out, adaln_attn,
        hidden_size: hidden_size,
        gate_index: 2,
        name: "#{name}_attn_gate"
      )

    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # AdaLN parameters for MLP sub-layer: shift2, scale2, gate2
    adaln_mlp = Axon.dense(condition, hidden_size * 3, name: "#{name}_adaln_mlp")

    x_modulated2 =
      x
      |> Axon.layer_norm(name: "#{name}_mlp_norm")
      |> AdaptiveNorm.modulate(adaln_mlp,
        hidden_size: hidden_size,
        offset: 0,
        name: "#{name}_mlp_mod"
      )

    # MLP
    mlp_out = Axon.dense(x_modulated2, mlp_dim, name: "#{name}_mlp_up")
    mlp_out = Axon.activation(mlp_out, :gelu, name: "#{name}_mlp_gelu")
    mlp_out = Axon.dense(mlp_out, hidden_size, name: "#{name}_mlp_down")

    mlp_out =
      AdaptiveNorm.gate(mlp_out, adaln_mlp,
        hidden_size: hidden_size,
        gate_index: 2,
        name: "#{name}_mlp_gate"
      )

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # ============================================================================
  # Linear Attention
  # ============================================================================

  defp build_linear_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &linear_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :linear_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # Linear attention using ELU+1 feature map
  # Bidirectional (non-causal) since DiT denoises all patches at once
  defp linear_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    eps = 1.0e-6

    rank = Nx.rank(q)

    # Handle both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
    {q, k, v, was_2d} =
      if rank == 2 do
        {Nx.new_axis(q, 1), Nx.new_axis(k, 1), Nx.new_axis(v, 1), true}
      else
        {q, k, v, false}
      end

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply ELU+1 feature map for non-negative features
    q_feat = elu_plus_one(q)
    k_feat = elu_plus_one(k)

    # Global (bidirectional) linear attention:
    # KV = sum over all positions: phi(K)^T * V  [batch, heads, head_dim, head_dim]
    # K_sum = sum over all positions: phi(K)    [batch, heads, head_dim]
    kv = Nx.dot(Nx.transpose(k_feat, axes: [0, 1, 3, 2]), [3], [0, 1], v, [2], [0, 1])
    k_sum = Nx.sum(k_feat, axes: [2])

    # Output: phi(Q) @ KV / (phi(Q) . K_sum)
    numerator = Nx.dot(q_feat, [3], [0, 1], kv, [2], [0, 1])
    # k_sum is [batch, heads, head_dim], expand to [batch, heads, head_dim, 1]
    # so we can dot with q_feat [batch, heads, seq, head_dim] along head_dim axis
    k_sum_expanded = Nx.new_axis(k_sum, 3)
    denominator = Nx.dot(q_feat, [3], [0, 1], k_sum_expanded, [2], [0, 1])
    denominator = Nx.add(denominator, eps)

    output = Nx.divide(numerator, denominator)

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output =
      output
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, num_heads * head_dim})

    # Remove seq dim if input was 2D
    if was_2d do
      Nx.squeeze(output, axes: [1])
    else
      output
    end
  end

  # ELU+1 feature map: ensures non-negative attention weights
  defp elu_plus_one(x) do
    positive = Nx.max(x, 0.0)
    negative = Nx.exp(Nx.min(x, 0.0))
    Nx.add(positive, negative)
  end

  # ============================================================================
  # Timestep and Class Conditioning
  # ============================================================================

  defp build_timestep_mlp(timestep, hidden_size, num_steps) do
    Edifice.Blocks.SinusoidalPE.timestep_layer(timestep,
      hidden_size: hidden_size,
      num_steps: num_steps
    )
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp build_class_embedding(class_label, hidden_size, num_classes) do
    one_hot =
      Axon.layer(
        &class_one_hot_impl/2,
        [class_label],
        name: "class_one_hot",
        num_classes: num_classes,
        op_name: :one_hot
      )

    one_hot
    |> Axon.dense(hidden_size, name: "class_embed_1")
    |> Axon.activation(:silu, name: "class_embed_silu")
    |> Axon.dense(hidden_size, name: "class_embed_2")
  end

  defp class_one_hot_impl(labels, opts) do
    num_classes = opts[:num_classes]

    Nx.equal(
      Nx.new_axis(Nx.as_type(labels, :s64), 1),
      Nx.iota({1, num_classes})
    )
    |> Nx.as_type(:f32)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Linear DiT model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_dim, 64)
  end

  @doc """
  Calculate approximate parameter count for a Linear DiT model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    input_dim = Keyword.get(opts, :input_dim, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    mlp_dim = round(hidden_size * mlp_ratio)

    # Per block: 2 adaln projections + attention (QKV + out) + MLP (up + down)
    adaln = 2 * hidden_size * (hidden_size * 3)
    attn = 4 * hidden_size * hidden_size
    mlp = hidden_size * mlp_dim + mlp_dim * hidden_size
    per_block = adaln + attn + mlp

    # Input/output projections + time MLP
    io = input_dim * hidden_size + hidden_size * input_dim
    time_mlp = hidden_size * hidden_size + hidden_size * hidden_size

    io + time_mlp + num_layers * per_block
  end

  @doc """
  Get recommended defaults for Linear DiT / SANA.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      input_dim: 64,
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      num_heads: @default_num_heads,
      mlp_ratio: @default_mlp_ratio,
      num_steps: @default_num_steps,
      patch_size: @default_patch_size
    ]
  end
end
