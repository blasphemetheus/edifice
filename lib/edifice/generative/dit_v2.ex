defmodule Edifice.Generative.DiTv2 do
  @moduledoc """
  DiT v2: Improved Diffusion Transformer with Unified AdaLN and QK-Norm.

  Implements improvements from SD3/Flux to the Diffusion Transformer,
  incorporating best practices from large-scale diffusion model training.

  ## Key Improvements over DiT v1

  1. **Unified AdaLN**: Single projection produces all 6 modulation parameters
     (gamma1, beta1, alpha1, gamma2, beta2, alpha2) instead of two separate
     projections. This is more parameter-efficient and allows better
     conditioning coordination.

  2. **QK-Norm**: Applies RMSNorm to Q and K before computing attention scores.
     This stabilizes attention at scale and prevents entropy collapse in
     deep models.

  3. **Multi-condition fusion**: Supports additive fusion of multiple condition
     signals (e.g., timestep + class + text embedding), each projected
     independently before combining.

  4. **RMSNorm instead of LayerNorm**: Uses RMSNorm throughout for faster
     computation and comparable quality.

  ## Architecture

  ```
  Inputs: noisy_input [batch, input_dim], timestep [batch]
        |
        v
  +--------------------------+
  | Condition Embedding      |
  |  time_embed + class_embed|
  +--------------------------+
        |
        v
  +--------------------------+
  | DiT v2 Block x depth     |
  |  Unified AdaLN(cond)     |
  |    -> 6 modulation params|
  |  RMSNorm(Q), RMSNorm(K)  |
  |  Self-Attention           |
  |  alpha1 * attn + residual |
  |  AdaLN MLP                |
  |  alpha2 * mlp + residual  |
  +--------------------------+
        |
        v
  | Final Norm + Linear      |
        |
        v
  Output [batch, input_dim]
  ```

  ## Usage

      model = DiTv2.build(
        input_dim: 64,
        hidden_size: 256,
        depth: 6,
        num_heads: 4
      )

  ## References

  - Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)
  - Esser et al., "Scaling Rectified Flow Transformers" (SD3, 2024)
  """

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:depth, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_classes, pos_integer() | nil}
          | {:num_steps, pos_integer()}
          | {:dropout, float()}

  @default_hidden_size 256
  @default_depth 6
  @default_num_heads 4
  @default_mlp_ratio 4.0
  @default_num_steps 1000

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a DiT v2 model for diffusion denoising.

  ## Options

    - `:input_dim` - Input/output feature dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:depth` - Number of DiT v2 blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_classes` - Number of classes for conditioning (optional)
    - `:num_steps` - Number of diffusion timesteps (default: 1000)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    An Axon model that predicts noise given (noisy_input, timestep, [class]).
  """
  @spec build([build_opt()]) :: Axon.t()
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

    # Timestep embedding
    time_embed = build_timestep_mlp(timestep, hidden_size, num_steps)

    # Optional class conditioning (additive multi-condition fusion)
    condition =
      if num_classes do
        class_label = Axon.input("class_label", shape: {nil})
        class_embed = build_class_embedding(class_label, hidden_size, num_classes)
        Axon.add(time_embed, class_embed, name: "condition_combine")
      else
        time_embed
      end

    # Project input
    x = Axon.dense(noisy_input, hidden_size, name: "input_embed")
    x = Axon.bias(x, name: "pos_embed")

    # DiT v2 blocks
    x =
      Enum.reduce(1..depth, x, fn block_idx, acc ->
        build_dit_v2_block(acc, condition,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          name: "dit_v2_block_#{block_idx}"
        )
      end)

    # Final RMSNorm + output projection
    x = build_rms_norm(x, hidden_size, "final_rms_norm")
    Axon.dense(x, input_dim, name: "output_proj")
  end

  # ============================================================================
  # DiT v2 Block with Unified AdaLN and QK-Norm
  # ============================================================================

  defp build_dit_v2_block(input, condition, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "dit_v2_block")
    mlp_dim = round(hidden_size * mlp_ratio)

    # Unified AdaLN: single projection -> 6 modulation params
    # gamma1, beta1, alpha1, gamma2, beta2, alpha2
    adaln_params = Axon.dense(condition, hidden_size * 6, name: "#{name}_unified_adaln")

    # --- Attention sub-layer ---
    x_norm = build_rms_norm(input, hidden_size, "#{name}_attn_rms_norm")

    # Modulate with gamma1, beta1
    x_modulated =
      Axon.layer(
        &adaln_modulate_impl/3,
        [x_norm, adaln_params],
        name: "#{name}_attn_mod",
        hidden_size: hidden_size,
        offset: 0,
        op_name: :adaln_modulate
      )

    # Self-attention with QK-Norm
    attn_out =
      build_qk_norm_attention(x_modulated, hidden_size, num_heads, "#{name}_attn")

    # Scale by alpha1 (gate)
    alpha_attn =
      Axon.nx(
        adaln_params,
        fn params ->
          Nx.slice_along_axis(params, hidden_size * 2, hidden_size, axis: 1)
        end,
        name: "#{name}_alpha_attn"
      )

    attn_out = Axon.multiply(attn_out, alpha_attn, name: "#{name}_attn_gate")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # --- MLP sub-layer ---
    x_norm2 = build_rms_norm(x, hidden_size, "#{name}_mlp_rms_norm")

    # Modulate with gamma2, beta2
    x_modulated2 =
      Axon.layer(
        &adaln_modulate_impl/3,
        [x_norm2, adaln_params],
        name: "#{name}_mlp_mod",
        hidden_size: hidden_size,
        offset: 3,
        op_name: :adaln_modulate
      )

    # MLP
    mlp_out = Axon.dense(x_modulated2, mlp_dim, name: "#{name}_mlp_up")
    mlp_out = Axon.activation(mlp_out, :gelu, name: "#{name}_mlp_gelu")
    mlp_out = Axon.dense(mlp_out, hidden_size, name: "#{name}_mlp_down")

    # Scale by alpha2
    alpha_mlp =
      Axon.nx(
        adaln_params,
        fn params ->
          Nx.slice_along_axis(params, hidden_size * 5, hidden_size, axis: 1)
        end,
        name: "#{name}_alpha_mlp"
      )

    mlp_out = Axon.multiply(mlp_out, alpha_mlp, name: "#{name}_mlp_gate")
    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # AdaLN modulation: (1 + gamma) * x + beta
  # offset: 0 for attn (gamma1, beta1), 3 for mlp (gamma2, beta2)
  defp adaln_modulate_impl(x, adaln_params, opts) do
    hidden_size = opts[:hidden_size]
    offset = opts[:offset] || 0

    gamma = Nx.slice_along_axis(adaln_params, offset * hidden_size, hidden_size, axis: 1)
    beta = Nx.slice_along_axis(adaln_params, (offset + 1) * hidden_size, hidden_size, axis: 1)

    Nx.add(Nx.multiply(Nx.add(1.0, gamma), x), beta)
  end

  # Self-attention with QK-Norm (RMSNorm applied to Q and K)
  defp build_qk_norm_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    # QK-Norm: RMSNorm on Q and K before attention
    q = build_rms_norm(q, hidden_size, "#{name}_q_norm")
    k = build_rms_norm(k, hidden_size, "#{name}_k_norm")

    attn_out =
      Axon.layer(
        &multi_head_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :self_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp multi_head_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    rank = Nx.rank(q)

    {q, k, v, was_2d} =
      if rank == 2 do
        {Nx.new_axis(q, 1), Nx.new_axis(k, 1), Nx.new_axis(v, 1), true}
      else
        {q, k, v, false}
      end

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output =
      output
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, num_heads * head_dim})

    if was_2d do
      Nx.squeeze(output, axes: [1])
    else
      output
    end
  end

  # RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
  defp build_rms_norm(input, hidden_size, name) do
    gamma =
      Axon.param("#{name}_gamma", {hidden_size},
        initializer: fn shape, _opts -> Nx.broadcast(Nx.tensor(1.0), shape) end
      )

    Axon.layer(
      fn x, g, _opts ->
        variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
        normalized = Nx.divide(x, Nx.sqrt(Nx.add(variance, 1.0e-6)))
        Nx.multiply(normalized, g)
      end,
      [input, gamma],
      name: name,
      op_name: :rms_norm
    )
  end

  # Timestep embedding MLP
  defp build_timestep_mlp(timestep, hidden_size, num_steps) do
    embed =
      Axon.layer(
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

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    t_expanded = Nx.new_axis(t_norm, 1)
    angles = Nx.multiply(t_expanded, Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  defp build_class_embedding(class_label, hidden_size, num_classes) do
    one_hot =
      Axon.layer(
        fn labels, opts ->
          nc = opts[:num_classes]

          Nx.equal(Nx.new_axis(Nx.as_type(labels, :s64), 1), Nx.iota({1, nc}))
          |> Nx.as_type(:f32)
        end,
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

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a DiT v2 model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_dim, 64)
  end

  @doc "Get recommended defaults."
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
