defmodule Edifice.Generative.DiT do
  @moduledoc """
  DiT: Diffusion Transformer.

  Implements the DiT architecture from "Scalable Diffusion Models with
  Transformers" (Peebles & Xie, ICCV 2023). Replaces the traditional
  U-Net backbone in diffusion models with a Transformer, using
  Adaptive Layer Normalization (AdaLN-Zero) for timestep and class
  conditioning.

  ## Key Innovation: AdaLN-Zero Conditioning

  Instead of cross-attention for conditioning (expensive), DiT modulates
  LayerNorm parameters based on the conditioning signal:

  ```
  # Standard LayerNorm:
  y = gamma * normalize(x) + beta

  # AdaLN-Zero:
  gamma, beta, alpha = MLP(condition)    # Learned modulation
  y = gamma * normalize(x) + beta       # Modulated norm
  y = alpha * y                         # Scale (initialized to zero)
  ```

  Initializing alpha to zero means each DiT block starts as an identity
  function, enabling stable deep training.

  ## Architecture

  ```
  Input [batch, input_dim]
        |
        v
  +--------------------------+
  | Patchify + Position Embed|
  +--------------------------+
        |
        v
  +--------------------------+
  | DiT Block x depth        |
  |  AdaLN-Zero(cond)        |
  |  Self-Attention          |
  |  Residual                |
  |  AdaLN-Zero(cond)        |
  |  MLP                     |
  |  Residual                |
  +--------------------------+
        |
        v
  | Final AdaLN + Linear    |
        |
        v
  Output [batch, input_dim]  (predicted noise or v-prediction)
  ```

  ## Conditioning

  ```
  Timestep t -----> Sinusoidal Embed --> MLP --+
                                               |--> condition vector
  Class label c --> Embedding ----------> MLP --+
  ```

  ## Usage

      model = DiT.build(
        input_dim: 64,
        hidden_size: 256,
        depth: 6,
        num_heads: 4
      )

  ## Reference

  - Paper: "Scalable Diffusion Models with Transformers"
  - arXiv: https://arxiv.org/abs/2212.09748
  """

  require Axon

  @default_hidden_size 256
  @default_depth 6
  @default_num_heads 4
  @default_mlp_ratio 4.0
  @default_num_steps 1000

  @doc """
  Build a DiT model for diffusion denoising.

  ## Options

    - `:input_dim` - Input/output feature dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:depth` - Number of DiT blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_classes` - Number of classes for conditioning (optional, nil = unconditional)
    - `:num_steps` - Number of diffusion timesteps (default: 1000)

  ## Returns

    An Axon model that predicts noise given (noisy_input, timestep, [class]).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    depth = Keyword.get(opts, :depth, @default_depth)
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

    # Add learnable position embedding (for single-token, this is a bias)
    x = Axon.bias(x, name: "pos_embed")

    # DiT blocks
    x =
      Enum.reduce(1..depth, x, fn block_idx, acc ->
        build_dit_block(acc, condition,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          name: "dit_block_#{block_idx}"
        )
      end)

    # Final layer: AdaLN + linear projection to input_dim
    x = Axon.layer_norm(x, name: "final_norm")
    Axon.dense(x, input_dim, name: "output_proj")
  end

  @doc """
  Build a single DiT block with AdaLN-Zero conditioning.
  """
  @spec build_dit_block(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def build_dit_block(input, condition, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "dit_block")
    mlp_dim = round(hidden_size * mlp_ratio)

    # AdaLN parameters for attention sub-layer: gamma1, beta1, alpha1
    adaln_attn = Axon.dense(condition, hidden_size * 3, name: "#{name}_adaln_attn")

    # AdaLN-Zero modulated attention
    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")

    x_modulated = Axon.layer(
      &adaln_modulate_impl/3,
      [x_norm, adaln_attn],
      name: "#{name}_attn_mod",
      hidden_size: hidden_size,
      op_name: :adaln_modulate
    )

    # Self-attention (simplified: dense Q, K, V projections + output)
    attn_out = build_self_attention(x_modulated, hidden_size, num_heads, "#{name}_attn")

    # Scale by alpha (gating)
    alpha_attn = Axon.nx(
      adaln_attn,
      fn params ->
        Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: 1)
      end,
      name: "#{name}_alpha_attn"
    )
    attn_out = Axon.multiply(attn_out, alpha_attn, name: "#{name}_attn_gate")

    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # AdaLN parameters for MLP sub-layer: gamma2, beta2, alpha2
    adaln_mlp = Axon.dense(condition, hidden_size * 3, name: "#{name}_adaln_mlp")

    x_norm2 = Axon.layer_norm(x, name: "#{name}_mlp_norm")

    x_modulated2 = Axon.layer(
      &adaln_modulate_impl/3,
      [x_norm2, adaln_mlp],
      name: "#{name}_mlp_mod",
      hidden_size: hidden_size,
      op_name: :adaln_modulate
    )

    # MLP
    mlp_out = Axon.dense(x_modulated2, mlp_dim, name: "#{name}_mlp_up")
    mlp_out = Axon.activation(mlp_out, :gelu, name: "#{name}_mlp_gelu")
    mlp_out = Axon.dense(mlp_out, hidden_size, name: "#{name}_mlp_down")

    # Scale by alpha
    alpha_mlp = Axon.nx(
      adaln_mlp,
      fn params ->
        Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: 1)
      end,
      name: "#{name}_alpha_mlp"
    )
    mlp_out = Axon.multiply(mlp_out, alpha_mlp, name: "#{name}_mlp_gate")

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # AdaLN modulation: gamma * x + beta
  defp adaln_modulate_impl(x, adaln_params, opts) do
    hidden_size = opts[:hidden_size]

    gamma = Nx.slice_along_axis(adaln_params, 0, hidden_size, axis: 1)
    beta = Nx.slice_along_axis(adaln_params, hidden_size, hidden_size, axis: 1)

    # gamma acts as scale (1 + gamma for stability)
    Nx.add(Nx.multiply(Nx.add(1.0, gamma), x), beta)
  end

  # Simplified self-attention for 2D input [batch, hidden_size]
  defp build_self_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    # For single-token input, attention is just a weighted projection
    # For multi-token, we would reshape to [batch, num_heads, seq, head_dim]
    # Here with [batch, hidden_size], attention degenerates to gating
    attn_out = Axon.layer(
      &single_token_attention_impl/4,
      [q, k, v],
      name: "#{name}_compute",
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :self_attention
    )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # For single-token: attention simplifies to sigmoid gating of value
  defp single_token_attention_impl(q, k, v, _opts) do
    # score = softmax(q * k / sqrt(d)) * v
    # For single token, softmax(scalar) = 1, so output = v
    # We add a learned gating for expressivity
    gate = Nx.sigmoid(Nx.sum(Nx.multiply(q, k), axes: [-1], keep_axes: true))
    Nx.multiply(gate, v)
  end

  # Sinusoidal timestep embedding -> MLP
  defp build_timestep_mlp(timestep, hidden_size, num_steps) do
    embed = Axon.layer(
      &sinusoidal_embed_impl/2,
      [timestep],
      name: "time_sinusoidal",
      hidden_size: hidden_size,
      num_steps: num_steps,
      op_name: :sinusoidal_embed
    )

    embed
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp sinusoidal_embed_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    num_steps = opts[:num_steps]
    half_dim = div(hidden_size, 2)

    t_norm = Nx.divide(Nx.as_type(t, :f32), num_steps)
    freqs = Nx.exp(
      Nx.multiply(
        Nx.negate(Nx.log(Nx.tensor(10000.0))),
        Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
      )
    )

    t_expanded = Nx.new_axis(t_norm, 1)
    angles = Nx.multiply(t_expanded, Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # Class embedding: lookup table -> MLP
  defp build_class_embedding(class_label, hidden_size, num_classes) do
    # Use one-hot encoding since Axon doesn't have nn.Embedding
    one_hot = Axon.layer(
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
    ) |> Nx.as_type(:f32)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a DiT model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_dim, 64)
  end

  @doc """
  Calculate approximate parameter count for a DiT model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    input_dim = Keyword.get(opts, :input_dim, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    depth = Keyword.get(opts, :depth, @default_depth)
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

    io + time_mlp + depth * per_block
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      input_dim: 64,
      hidden_size: 256,
      depth: 6,
      num_heads: 4,
      mlp_ratio: 4.0,
      num_steps: 1000
    ]
  end
end
