defmodule Edifice.Generative.DeepFlow do
  @moduledoc """
  DeepFlow: Deeply Supervised Flow-Based Generative Model.

  Implements DeepFlow from "Deeply Supervised Flow-Based Generative Models"
  (Shin et al., ByteDance Seed, ICCV 2025). Extends SiT/DiT with branch
  partitioning, intermediate velocity heads (deep supervision), and VeRA
  (Velocity Refiner with Acceleration) blocks between branches.

  ## Key Innovation

  Standard flow models predict velocity only from the final layer, wasting
  intermediate representations. DeepFlow partitions the transformer into K
  branches, each with its own velocity head. Between branches, VeRA blocks
  use second-order dynamics (acceleration) to refine velocity features.
  Achieves 8x faster convergence than SiT on ImageNet-256.

  ## Architecture

  ```
  Input: noisy_latent [batch, C, H, W] + timestep [batch]
        |
  +--------------------------------------------------+
  | Patch Embedding + Position Embedding              |
  | + AdaLN timestep conditioning                     |
  +--------------------------------------------------+
        |
  +--------------------------------------------------+
  | Branch 1: DiT Blocks [0..L/K-1]                  |
  +--- Velocity Head 1 ----> v_1 (deep supervision)  |
        |                                             |
  | VeRA Block (acc MLP + time-gap + cross-attn)     |
        |                                             |
  | Branch 2: DiT Blocks [L/K..2L/K-1]              |
  +--- Velocity Head 2 ----> v_2                     |
        |                                             |
  | ... repeat VeRA + Branch ...                     |
        |                                             |
  | Branch K: DiT Blocks [(K-1)*L/K..L-1]           |
  +--- Velocity Head K ----> v_K (final output)      |
  +--------------------------------------------------+
        |
  Output: %{velocity: v_K, branch_velocities: [v_1..v_K]}
  ```

  ## VeRA Block

  Between adjacent branches, a Velocity Refiner with Acceleration:
  1. **ACC MLP**: 2-layer MLP generating acceleration features from velocity
  2. **Time-gap conditioning**: AdaLN-Zero modulation from timestep delta
  3. **Cross-space attention**: Q from velocity features, K/V from spatial

  ## Usage

      model = DeepFlow.build(
        input_size: 32,
        hidden_size: 256,
        num_layers: 12,
        num_heads: 8,
        num_branches: 4
      )

  ## References

  - Shin et al., "Deeply Supervised Flow-Based Generative Models"
    (ICCV 2025) — https://arxiv.org/abs/2503.14494
  """

  @default_hidden_size 256
  @default_num_layers 12
  @default_num_heads 8
  @default_num_branches 4
  @default_mlp_ratio 4.0
  @default_patch_size 2
  @default_in_channels 4

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:input_size, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_branches, pos_integer()}
          | {:num_classes, non_neg_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:patch_size, pos_integer()}

  @doc """
  Build a DeepFlow model with branched deep supervision and VeRA.

  ## Options

    - `:input_size` - Spatial size of input latent (required, e.g. 32)
    - `:patch_size` - Patch size for patchification (default: 2)
    - `:in_channels` - Input channels (default: 4)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_layers` - Total transformer blocks (default: 12)
    - `:num_heads` - Attention heads (default: 8)
    - `:num_branches` - Number of supervised branches K (default: 4)
    - `:num_classes` - Class conditioning (default: 0, unconditional)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)

  ## Returns

    An Axon container `%{velocity: final, branch_velocities: [v_1..v_K]}`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_branches = Keyword.get(opts, :num_branches, @default_num_branches)
    num_classes = Keyword.get(opts, :num_classes, 0)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)

    num_patches = div(input_size, patch_size) * div(input_size, patch_size)
    patch_dim = patch_size * patch_size * in_channels

    # Inputs
    noisy_latent = Axon.input("noisy_latent", shape: {nil, num_patches, patch_dim})
    timestep = Axon.input("timestep", shape: {nil})

    # Patch embedding
    x = Axon.dense(noisy_latent, hidden_size, name: "patch_embed")
    x = Axon.bias(x, name: "pos_embed")

    # Timestep conditioning
    time_embed = build_timestep_mlp(timestep, hidden_size)

    # Optional class conditioning
    condition =
      if num_classes > 0 do
        class_label = Axon.input("class_label", shape: {nil})
        class_embed = build_class_embedding(class_label, hidden_size, num_classes)
        Axon.add(time_embed, class_embed, name: "condition_combine")
      else
        time_embed
      end

    # Partition layers into branches
    branch_sizes = partition_layers(num_layers, num_branches)

    # Build branches with VeRA blocks between them
    {_final, branch_velocities} =
      branch_sizes
      |> Enum.with_index(1)
      |> Enum.reduce({x, []}, fn {branch_layer_count, branch_idx}, {acc, velocities} ->
        # Stack of DiT blocks for this branch
        start_layer = Enum.sum(Enum.take(branch_sizes, branch_idx - 1))

        acc =
          Enum.reduce(1..branch_layer_count, acc, fn layer_offset, inner ->
            layer_idx = start_layer + layer_offset

            build_dit_block(inner, condition,
              hidden_size: hidden_size,
              num_heads: num_heads,
              mlp_ratio: mlp_ratio,
              name: "dit_block_#{layer_idx}"
            )
          end)

        # Velocity head for this branch
        vel = build_velocity_head(acc, hidden_size, patch_dim, "vel_head_#{branch_idx}")

        # VeRA block between branches (not after last)
        acc =
          if branch_idx < num_branches do
            build_vera_block(acc, x, condition,
              hidden_size: hidden_size,
              num_heads: num_heads,
              name: "vera_#{branch_idx}"
            )
          else
            acc
          end

        {acc, velocities ++ [vel]}
      end)

    # Return container with final velocity and all branch velocities
    final_velocity = List.last(branch_velocities)

    Axon.container(%{
      velocity: final_velocity,
      branch_velocities: List.to_tuple(branch_velocities)
    })
  end

  # ===========================================================================
  # DiT Block (AdaLN-Zero conditioned)
  # ===========================================================================

  defp build_dit_block(input, condition, opts) do
    alias Edifice.Blocks.AdaptiveNorm

    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    mlp_ratio = opts[:mlp_ratio]
    name = opts[:name]
    mlp_dim = round(hidden_size * mlp_ratio)

    # AdaLN: 6 params (shift, scale, gate for attn and MLP)
    adaln_params =
      condition
      |> Axon.activation(:silu, name: "#{name}_adaln_silu")
      |> Axon.dense(hidden_size * 6, name: "#{name}_adaln_proj")

    # Attention sub-layer
    x_mod =
      input
      |> Axon.layer_norm(name: "#{name}_attn_norm")
      |> AdaptiveNorm.modulate(adaln_params,
        hidden_size: hidden_size,
        offset: 0,
        name: "#{name}_attn_mod"
      )

    attn_out = build_self_attention(x_mod, hidden_size, num_heads, "#{name}_attn")

    attn_out =
      AdaptiveNorm.gate(attn_out, adaln_params,
        hidden_size: hidden_size,
        gate_index: 2,
        name: "#{name}_attn_gate"
      )

    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # MLP sub-layer
    adaln_mlp =
      condition
      |> Axon.activation(:silu, name: "#{name}_adaln_mlp_silu")
      |> Axon.dense(hidden_size * 3, name: "#{name}_adaln_mlp_proj")

    x_mod2 =
      x
      |> Axon.layer_norm(name: "#{name}_mlp_norm")
      |> AdaptiveNorm.modulate(adaln_mlp,
        hidden_size: hidden_size,
        offset: 0,
        name: "#{name}_mlp_mod"
      )

    mlp_out = Axon.dense(x_mod2, mlp_dim, name: "#{name}_mlp_up")
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

  # ===========================================================================
  # VeRA Block (Velocity Refiner with Acceleration)
  # ===========================================================================

  defp build_vera_block(velocity_features, spatial_features, condition, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    name = opts[:name]

    # 1. ACC MLP: acceleration from velocity
    acc_feat =
      velocity_features
      |> Axon.dense(hidden_size, name: "#{name}_acc_up")
      |> Axon.activation(:gelu, name: "#{name}_acc_gelu")
      |> Axon.dense(hidden_size, name: "#{name}_acc_down")

    # 2. Concatenate velocity + acceleration, condition on timestep
    combined =
      Axon.layer(
        fn v, a, _opts -> Nx.concatenate([v, a], axis: -1) end,
        [velocity_features, acc_feat],
        name: "#{name}_va_concat",
        op_name: :concatenate
      )

    # Time-gap modulated features
    adaln_vera =
      condition
      |> Axon.activation(:silu, name: "#{name}_adaln_silu")
      |> Axon.dense(hidden_size * 2, name: "#{name}_adaln_proj")

    h = Axon.dense(combined, hidden_size, name: "#{name}_va_proj")

    h =
      h
      |> Axon.layer_norm(name: "#{name}_va_norm")
      |> Edifice.Blocks.AdaptiveNorm.modulate(adaln_vera,
        hidden_size: hidden_size,
        offset: 0,
        name: "#{name}_va_mod"
      )

    # 3. Cross-space attention: Q from velocity, K/V from spatial
    build_cross_attention(h, spatial_features, hidden_size, num_heads, "#{name}_cross_attn")
  end

  # ===========================================================================
  # Velocity Head
  # ===========================================================================

  defp build_velocity_head(input, _hidden_size, output_dim, name) do
    input
    |> Axon.layer_norm(name: "#{name}_norm")
    |> Axon.dense(output_dim, name: "#{name}_proj")
  end

  # ===========================================================================
  # Self-Attention
  # ===========================================================================

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

  # ===========================================================================
  # Cross-Attention
  # ===========================================================================

  defp build_cross_attention(query_input, kv_input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(query_input, hidden_size, name: "#{name}_q")
    k = Axon.dense(kv_input, hidden_size, name: "#{name}_k")
    v = Axon.dense(kv_input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &multi_head_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :cross_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp multi_head_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    q_seq = Nx.axis_size(q, 1)
    kv_seq = Nx.axis_size(k, 1)

    q = q |> Nx.reshape({batch, q_seq, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, kv_seq, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, kv_seq, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, q_seq, num_heads * head_dim})
  end

  # ===========================================================================
  # Timestep & Conditioning
  # ===========================================================================

  defp build_timestep_mlp(timestep, hidden_size) do
    Edifice.Blocks.SinusoidalPE.timestep_layer(timestep,
      hidden_size: hidden_size,
      num_steps: 1000
    )
    |> Axon.dense(hidden_size, name: "time_mlp_1")
    |> Axon.activation(:silu, name: "time_mlp_silu")
    |> Axon.dense(hidden_size, name: "time_mlp_2")
  end

  defp build_class_embedding(class_label, hidden_size, num_classes) do
    one_hot =
      Axon.layer(
        fn labels, opts ->
          nc = opts[:num_classes]

          Nx.equal(
            Nx.new_axis(Nx.as_type(labels, :s64), 1),
            Nx.iota({1, nc})
          )
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

  # ===========================================================================
  # Helpers
  # ===========================================================================

  defp partition_layers(num_layers, num_branches) do
    base = div(num_layers, num_branches)
    remainder = rem(num_layers, num_branches)

    # Last branch gets extra layers if not evenly divisible
    Enum.map(1..num_branches, fn i ->
      if i == num_branches, do: base + remainder, else: base
    end)
  end

  @doc """
  Get the output size of a DeepFlow model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    input_size = Keyword.get(opts, :input_size, 32)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)

    div(input_size, patch_size) * div(input_size, patch_size) *
      (patch_size * patch_size * in_channels)
  end
end
