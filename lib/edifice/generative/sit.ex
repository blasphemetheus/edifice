defmodule Edifice.Generative.SiT do
  @moduledoc """
  SiT: Scalable Interpolant Transformer.

  Implements SiT from "Scalable Interpolant Transformers" (Ma et al., 2024).
  Generalizes DiT by learning the interpolant between noise and data, rather
  than predicting just the score or velocity with a fixed schedule.

  ## Key Innovation: Learnable Interpolant

  Instead of a fixed diffusion schedule, SiT uses:

      I(t) = α(t) · x + β(t) · ε

  where α and β are schedules (default linear: α(t)=1-t, β(t)=t). The model
  predicts the interpolant velocity:

      v(x, t) = dI/dt = α'(t) · x + β'(t) · ε

  This subsumes both DDPM (score prediction) and flow matching (velocity
  prediction) as special cases depending on the choice of α, β.

  ## Architecture

  Same transformer backbone as DiT:

  ```
  Input [batch, input_dim]
        |
        v
  +--------------------------+
  | Input Embed + Pos Embed  |
  +--------------------------+
        |
        v
  +--------------------------+
  | SiT Block x depth        |
  |  AdaLN-Zero(time_cond)   |
  |  Self-Attention          |
  |  Residual                |
  |  AdaLN-Zero(time_cond)   |
  |  MLP                     |
  |  Residual                |
  +--------------------------+
        |
        v
  | Final Norm + Linear     |
        |
        v
  Output [batch, input_dim]  (predicted velocity)
  ```

  ## Interpolant Schedules

  - Linear (default): α(t) = 1-t, β(t) = t  →  simple, matches flow matching
  - Cosine: α(t) = cos(πt/2), β(t) = sin(πt/2)  →  smoother transitions
  - Custom: user-provided α(t), β(t) functions

  ## Usage

      model = SiT.build(
        input_dim: 64,
        hidden_size: 256,
        num_layers: 6,
        num_heads: 4
      )

      # Training: sample time, compute interpolant and target velocity
      t = SiT.sample_interpolant_time(batch_size)
      loss = SiT.sit_loss(predicted_velocity, target_velocity)

  ## References

  - Paper: "Scalable Interpolant Transformers"
  - arXiv: https://arxiv.org/abs/2401.08740
  """

  import Nx.Defn

  @default_hidden_size 256
  @default_depth 6
  @default_num_heads 4
  @default_mlp_ratio 4.0
  @default_num_steps 1000

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a SiT model for interpolant-based generation.

  ## Options

    - `:input_dim` - Input/output feature dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_layers` - Number of SiT blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_classes` - Number of classes for conditioning (optional)
    - `:num_steps` - Number of timesteps for embedding (default: 1000)

  ## Returns

    An Axon model that predicts velocity given (noisy_input, timestep, [class]).
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:depth, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, pos_integer()}
          | {:num_steps, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    depth = Keyword.get(opts, :num_layers, Keyword.get(opts, :depth, @default_depth))
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

    # SiT blocks (same architecture as DiT blocks with AdaLN-Zero)
    x =
      Enum.reduce(1..depth, x, fn block_idx, acc ->
        build_sit_block(acc, condition,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          name: "sit_block_#{block_idx}"
        )
      end)

    # Final layer: norm + linear projection to input_dim (velocity prediction)
    x = Axon.layer_norm(x, name: "final_norm")
    Axon.dense(x, input_dim, name: "velocity_proj")
  end

  # SiT block: identical to DiT block with AdaLN-Zero conditioning
  defp build_sit_block(input, condition, opts) do
    alias Edifice.Blocks.AdaptiveNorm

    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "sit_block")
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

    # Self-attention
    attn_out = build_self_attention(x_modulated, hidden_size, num_heads, "#{name}_attn")

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

  # Multi-head self-attention supporting both 2D and 3D inputs
  defp build_self_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

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

    # Add seq dim if 2D: [batch, hidden] -> [batch, 1, hidden]
    {q, k, v, was_2d} =
      if rank == 2 do
        {Nx.new_axis(q, 1), Nx.new_axis(k, 1), Nx.new_axis(v, 1), true}
      else
        {q, k, v, false}
      end

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
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

  # ============================================================================
  # Timestep & Conditioning
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
  # Interpolant Utilities
  # ============================================================================

  @doc """
  Sample interpolant time t ~ Uniform(0, 1).

  ## Parameters

    - `batch_size` - Number of samples

  ## Options

    - `:key` - Random key (default: `Nx.Random.key(System.system_time())`)

  ## Returns

    `{t, new_key}` where t has shape `{batch_size}`.
  """
  @spec sample_interpolant_time(pos_integer(), keyword()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def sample_interpolant_time(batch_size, opts \\ []) do
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(System.system_time()) end)
    Nx.Random.uniform(key, shape: {batch_size})
  end

  @doc """
  Compute the linear interpolant between data and noise.

  I(t) = (1 - t) · x + t · ε  (linear schedule)

  ## Parameters

    - `x` - Clean data: [batch, dim]
    - `noise` - Random noise: [batch, dim]
    - `t` - Interpolant time: [batch] (values in [0, 1])

  ## Returns

    Interpolated tensor with same shape as x.
  """
  @spec linear_interpolant(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn linear_interpolant(x, noise, t) do
    t_expanded = Nx.new_axis(t, 1)
    (1.0 - t_expanded) * x + t_expanded * noise
  end

  @doc """
  Compute the target velocity for the linear interpolant.

  v(t) = dI/dt = -x + ε  (linear schedule derivative)

  ## Parameters

    - `x` - Clean data: [batch, dim]
    - `noise` - Random noise: [batch, dim]

  ## Returns

    Target velocity tensor with same shape as x.
  """
  @spec linear_velocity(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn linear_velocity(x, noise) do
    # For linear interpolant: α(t)=1-t, β(t)=t
    # α'(t) = -1, β'(t) = 1
    # v = α'(t)·x + β'(t)·ε = -x + ε
    noise - x
  end

  @doc """
  Compute the cosine interpolant between data and noise.

  I(t) = cos(πt/2) · x + sin(πt/2) · ε

  ## Parameters

    - `x` - Clean data: [batch, dim]
    - `noise` - Random noise: [batch, dim]
    - `t` - Interpolant time: [batch] (values in [0, 1])

  ## Returns

    Interpolated tensor with same shape as x.
  """
  @spec cosine_interpolant(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn cosine_interpolant(x, noise, t) do
    t_expanded = Nx.new_axis(t, 1)
    alpha = Nx.cos(Nx.Constants.pi() * t_expanded / 2.0)
    beta = Nx.sin(Nx.Constants.pi() * t_expanded / 2.0)
    alpha * x + beta * noise
  end

  @doc """
  Compute the target velocity for the cosine interpolant.

  v(t) = -(π/2)·sin(πt/2)·x + (π/2)·cos(πt/2)·ε

  ## Parameters

    - `x` - Clean data: [batch, dim]
    - `noise` - Random noise: [batch, dim]
    - `t` - Interpolant time: [batch] (values in [0, 1])

  ## Returns

    Target velocity tensor with same shape as x.
  """
  @spec cosine_velocity(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn cosine_velocity(x, noise, t) do
    t_expanded = Nx.new_axis(t, 1)
    half_pi = Nx.Constants.pi() / 2.0
    d_alpha = -half_pi * Nx.sin(half_pi * t_expanded)
    d_beta = half_pi * Nx.cos(half_pi * t_expanded)
    d_alpha * x + d_beta * noise
  end

  # ============================================================================
  # Loss
  # ============================================================================

  @doc """
  Compute the SiT loss (MSE between predicted and target velocity).

  ## Parameters

    - `pred_velocity` - Model-predicted velocity: [batch, dim]
    - `target_velocity` - Target velocity: [batch, dim]

  ## Returns

    Scalar MSE loss tensor.
  """
  @spec sit_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn sit_loss(pred_velocity, target_velocity) do
    diff = pred_velocity - target_velocity
    Nx.mean(diff * diff)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a SiT model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_dim, 64)
  end
end
