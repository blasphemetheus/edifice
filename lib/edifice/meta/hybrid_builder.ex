defmodule Edifice.Meta.HybridBuilder do
  @moduledoc """
  Configurable Hybrid Builder — flexible hybrid architecture composition.

  A meta-module that builds hybrid sequence models with arbitrary layer
  schedules, going beyond the fixed `attention_every: N` pattern. Supports
  ratio-based specification, explicit schedules, parallel SSM+attention
  blocks, and multi-backbone mixing.

  ## Scheduling Modes

  | Mode | Option | Example |
  |------|--------|---------|
  | Ratio | `ratio: {9, 1}` | 9 backbone : 1 attention |
  | Every-N | `attention_every: 4` | Same as existing Hybrid |
  | Explicit | `schedule: [:mamba, :mamba, :attn, ...]` | Full control |
  | Parallel | `mode: :parallel` | SSM+attention in parallel per block |

  ## Architecture (Interleaved Mode)

  ```
  Input [batch, seq_len, embed_dim]
        │
  ┌─────┴──────────────────────────────┐
  │  schedule[0] block                  │  backbone or attention
  ├────────────────────────────────────┤
  │  schedule[1] block                  │  backbone or attention
  ├────────────────────────────────────┤
  │  ...                                │
  └────────────────────────────────────┘
        │
  [batch, hidden_size]
  ```

  ## Architecture (Parallel Mode)

  ```
  Input [batch, seq_len, embed_dim]
        │
  ┌─────┴──────────────────────────────┐
  │  Per block:                         │
  │    norm(x) → SSM path              │
  │    norm(x) → Attention path        │
  │    gate * ssm + (1 - gate) * attn  │
  │    + FFN + residual                │
  └────────────────────────────────────┘
        │
  [batch, hidden_size]
  ```

  ## Usage

      # Ratio-based: 90% Mamba, 10% attention (10 layers → 9 Mamba + 1 attn)
      model = HybridBuilder.build(
        embed_dim: 256,
        hidden_size: 256,
        num_layers: 10,
        ratio: {9, 1}
      )

      # Explicit schedule with multi-backbone
      model = HybridBuilder.build(
        embed_dim: 256,
        hidden_size: 256,
        schedule: [:mamba, :mamba, :gru, :attn, :mamba, :mamba, :gru, :attn]
      )

      # Parallel mode (Hymba-style, all layers have both SSM + attention)
      model = HybridBuilder.build(
        embed_dim: 256,
        hidden_size: 256,
        num_layers: 6,
        mode: :parallel
      )

  ## References

  - Jamba (AI21, 2024) — sequential Mamba+attention interleaving
  - Zamba (Zyphra, 2024) — shared attention layer
  - Hymba (NVIDIA, 2024) — parallel Mamba+attention per block
  - Nemotron-H (NVIDIA, 2025) — 90:10 SSM:attention ratio
  """

  alias Edifice.Blocks.FFN
  alias Edifice.SSM.Hybrid

  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4

  @doc """
  Build a configurable hybrid model.

  ## Options

  **Scheduling (mutually exclusive — first match wins):**
    - `:schedule` - Explicit layer schedule as list of atoms.
      Valid entries: `:attn`, `:mamba`, `:gru`, `:rwkv`, `:delta_net`,
      `:gated_delta_net`, `:griffin_lru`.
    - `:ratio` - `{backbone_count, attn_count}` tuple. Layers are distributed
      so attention is evenly spaced. E.g., `{9, 1}` with 10 layers → 9 backbone + 1 attn.
    - `:attention_every` - Insert attention every N layers (fallback, same as Hybrid).

  **Mode:**
    - `:mode` - `:interleaved` (default) or `:parallel`.
      Parallel mode runs SSM + attention in every block with learned gating.

  **Architecture:**
    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Total number of layers (default: 6). Ignored if `:schedule` is given.
    - `:backbone` - Default backbone type (default: `:mamba`)

  **SSM-specific (for Mamba backbone):**
    - `:state_size` - SSM state dimension (default: 16)
    - `:expand_factor` - Mamba expansion factor (default: 2)
    - `:conv_size` - Causal conv kernel size (default: 4)

  **Attention-specific:**
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:window_size` - Attention window size (default: 60)

  **General:**
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Fixed sequence length (default: window_size)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:backbone, atom()}
          | {:mode, :interleaved | :parallel}
          | {:schedule, [atom()]}
          | {:ratio, {pos_integer(), pos_integer()}}
          | {:attention_every, pos_integer()}
          | {:state_size, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:window_size, pos_integer()}
          | {:dropout, float()}
          | {:seq_len, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    mode = Keyword.get(opts, :mode, :interleaved)

    case mode do
      :parallel -> build_parallel(opts)
      :interleaved -> build_interleaved(opts)
    end
  end

  # ============================================================================
  # Interleaved Mode
  # ============================================================================

  defp build_interleaved(opts) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    schedule = resolve_schedule(opts)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Build backbone opts
    backbone_opts = [
      hidden_size: hidden_size,
      state_size: Keyword.get(opts, :state_size, @default_state_size),
      expand_factor: Keyword.get(opts, :expand_factor, @default_expand_factor),
      conv_size: Keyword.get(opts, :conv_size, @default_conv_size),
      dropout: dropout,
      pre_norm: true,
      num_heads: num_heads
    ]

    attn_hidden_dim = num_heads * head_dim

    # Pre-compute mask
    precomputed_mask =
      if seq_len do
        Edifice.Attention.MultiHead.window_mask(seq_len, window_size)
        |> Nx.backend_copy(Nx.BinaryBackend)
      else
        nil
      end

    x =
      schedule
      |> Enum.with_index(1)
      |> Enum.reduce(x, fn {layer_type, layer_idx}, acc ->
        if layer_type == :attn do
          Hybrid.build_attention_layer(acc,
            hidden_size: hidden_size,
            attn_hidden_dim: attn_hidden_dim,
            num_heads: num_heads,
            head_dim: head_dim,
            dropout: dropout,
            use_sliding_window: true,
            window_size: window_size,
            precomputed_mask: precomputed_mask,
            pre_norm: true,
            qk_layernorm: true,
            name: "layer_#{layer_idx}_attn"
          )
        else
          Hybrid.build_backbone_layer(
            acc,
            layer_type,
            Keyword.put(backbone_opts, :name, "layer_#{layer_idx}_#{layer_type}")
          )
        end
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_actual - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # Parallel Mode (Hymba-style)
  # ============================================================================

  defp build_parallel(opts) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    block_opts = [
      hidden_size: hidden_size,
      num_heads: num_heads,
      head_dim: head_dim,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      dropout: dropout
    ]

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_parallel_block(acc, Keyword.put(block_opts, :layer_idx, layer_idx))
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_actual - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_parallel_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    state_size = Keyword.fetch!(opts, :state_size)
    expand_factor = Keyword.fetch!(opts, :expand_factor)
    conv_size = Keyword.fetch!(opts, :conv_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    layer_idx = Keyword.fetch!(opts, :layer_idx)
    name = "parallel_#{layer_idx}"

    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # SSM path (Mamba-style selective scan)
    ssm_out = build_ssm_path(normed, hidden_size, state_size, expand_factor, conv_size, name)

    # Attention path
    attn_out = build_attention_path(normed, hidden_size, num_heads, head_dim, name)

    # Learned gate: sigmoid per dimension
    gate =
      normed
      |> Axon.dense(hidden_size, name: "#{name}_gate_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_gate")

    # Fused: gate * ssm + (1 - gate) * attn
    fused =
      Axon.layer(
        fn ssm, attn, g, _opts ->
          Nx.add(
            Nx.multiply(g, ssm),
            Nx.multiply(Nx.subtract(1.0, g), attn)
          )
        end,
        [ssm_out, attn_out, gate],
        name: "#{name}_fusion",
        op_name: :gated_fusion
      )

    after_fusion = Axon.add(input, fused, name: "#{name}_residual")

    # FFN sub-layer
    ffn_normed = Axon.layer_norm(after_fusion, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        expansion_factor: 4,
        name: "#{name}_ffn"
      )

    result = Axon.add(after_fusion, ffn_out, name: "#{name}_ffn_residual")

    if dropout > 0 do
      Axon.dropout(result, rate: dropout, name: "#{name}_dropout")
    else
      result
    end
  end

  # SSM path: expand → split x/z → causal conv → silu → selective scan → gate → project
  defp build_ssm_path(input, hidden_size, state_size, expand_factor, conv_size, name) do
    inner_size = hidden_size * expand_factor

    xz = Axon.dense(input, inner_size * 2, name: "#{name}_ssm_in_proj")

    x_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, 0, inner_size, axis: 2) end,
        name: "#{name}_ssm_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2) end,
        name: "#{name}_ssm_z_split"
      )

    # Causal depthwise conv
    x_conv =
      Axon.conv(x_branch, inner_size,
        kernel_size: {conv_size},
        padding: [{conv_size - 1, 0}],
        feature_group_size: inner_size,
        name: "#{name}_ssm_conv"
      )

    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_ssm_silu")

    # Simplified selective scan
    ssm_out = build_selective_scan(x_activated, inner_size, state_size, "#{name}_ssm")

    # Gate with z branch
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_ssm_z_gate")
    gated = Axon.multiply(ssm_out, z_activated, name: "#{name}_ssm_gated")

    Axon.dense(gated, hidden_size, name: "#{name}_ssm_out_proj")
  end

  defp build_selective_scan(input, hidden_size, state_size, name) do
    b_proj = Axon.dense(input, state_size, name: "#{name}_b_proj")
    c_proj = Axon.dense(input, state_size, name: "#{name}_c_proj")

    dt_proj =
      input
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.nx(fn t -> Nx.log(Nx.add(Nx.exp(t), 1.0)) end, name: "#{name}_dt_softplus")

    Axon.layer(
      &selective_scan_impl/5,
      [input, b_proj, c_proj, dt_proj],
      name: "#{name}_scan",
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :selective_scan
    )
  end

  defp selective_scan_impl(x, b, c, dt, opts) do
    hidden_size = opts[:hidden_size]
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    gate = Nx.sigmoid(Nx.mean(dt, axes: [2], keep_axes: true))
    bc_product = Nx.multiply(b, c)
    bc_gate = Nx.sigmoid(Nx.sum(bc_product, axes: [2], keep_axes: true))

    ema_coeff = Nx.mean(gate, axes: [2], keep_axes: true)
    ema_coeff = Nx.broadcast(ema_coeff, {batch_size, seq_len, hidden_size})
    gated_broadcast = Nx.broadcast(Nx.multiply(gate, bc_gate), {batch_size, seq_len, hidden_size})
    output = Nx.multiply(gated_broadcast, x)

    cumulative_ema(output, ema_coeff)
  end

  defp cumulative_ema(x, alpha) do
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)
    hidden_size = Nx.axis_size(x, 2)

    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        a_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), Nx.multiply(Nx.subtract(1.0, a_t), x_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Multi-head causal attention path
  defp build_attention_path(input, hidden_size, num_heads, head_dim, name) do
    attn_dim = num_heads * head_dim

    q = Axon.dense(input, attn_dim, name: "#{name}_attn_q")
    k = Axon.dense(input, attn_dim, name: "#{name}_attn_k")
    v = Axon.dense(input, attn_dim, name: "#{name}_attn_v")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q, k, v],
        name: "#{name}_attn_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :causal_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_attn_out_proj")
  end

  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q_heads =
      q
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k_heads =
      k
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v_heads =
      v
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q_heads, [3], [0, 1], k_heads, [3], [0, 1]) |> Nx.divide(scale)

    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols) |> Nx.new_axis(0) |> Nx.new_axis(0)
    mask = Nx.broadcast(mask, {batch_size, num_heads, seq_len, seq_len})

    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
    scores = Nx.select(mask, scores, neg_inf)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v_heads, [2], [0, 1])

    Nx.transpose(output, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Schedule Resolution
  # ============================================================================

  @doc """
  Resolve the layer schedule from options.

  Priority: `:schedule` > `:ratio` > `:attention_every` > default (3:1).

  ## Examples

      iex> HybridBuilder.resolve_schedule(schedule: [:mamba, :mamba, :attn])
      [:mamba, :mamba, :attn]

      iex> HybridBuilder.resolve_schedule(num_layers: 10, ratio: {9, 1}, backbone: :mamba)
      [:mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :attn]

      iex> HybridBuilder.resolve_schedule(num_layers: 6, attention_every: 3, backbone: :gru)
      [:gru, :gru, :attn, :gru, :gru, :attn]
  """
  @spec resolve_schedule(keyword()) :: [atom()]
  def resolve_schedule(opts) do
    cond do
      Keyword.has_key?(opts, :schedule) ->
        Keyword.fetch!(opts, :schedule)

      Keyword.has_key?(opts, :ratio) ->
        resolve_ratio_schedule(opts)

      Keyword.has_key?(opts, :attention_every) ->
        resolve_every_schedule(opts)

      true ->
        # Default: 3:1 backbone:attention
        resolve_every_schedule(Keyword.put_new(opts, :attention_every, 4))
    end
  end

  defp resolve_ratio_schedule(opts) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    {backbone_ratio, attn_ratio} = Keyword.fetch!(opts, :ratio)
    backbone = Keyword.get(opts, :backbone, :mamba)

    total_ratio = backbone_ratio + attn_ratio
    num_attn = max(round(num_layers * attn_ratio / total_ratio), 1)
    num_attn = min(num_attn, num_layers)

    # Evenly space attention layers
    attn_positions = distribute_positions(num_layers, num_attn)

    Enum.map(0..(num_layers - 1), fn idx ->
      if idx in attn_positions, do: :attn, else: backbone
    end)
  end

  defp resolve_every_schedule(opts) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.fetch!(opts, :attention_every)
    backbone = Keyword.get(opts, :backbone, :mamba)

    Enum.map(1..num_layers, fn idx ->
      if rem(idx, attention_every) == 0, do: :attn, else: backbone
    end)
  end

  # Evenly distribute N positions across L slots, preferring later positions
  defp distribute_positions(total_layers, num_positions) do
    if num_positions >= total_layers do
      MapSet.new(0..(total_layers - 1))
    else
      # Place attention at evenly spaced intervals, biased toward end
      spacing = total_layers / num_positions

      Enum.map(0..(num_positions - 1), fn i ->
        # Position: round((i + 1) * spacing) - 1, clamped to valid range
        pos = round((i + 1) * spacing) - 1
        min(pos, total_layers - 1)
      end)
      |> MapSet.new()
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of the model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Describe the layer pattern for a given configuration.

  Useful for debugging and visualization.

  ## Examples

      iex> HybridBuilder.describe_schedule(num_layers: 10, ratio: {9, 1})
      %{
        schedule: [:mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :mamba, :attn],
        num_backbone: 9,
        num_attention: 1,
        backbone_pct: 90.0,
        mode: :interleaved
      }
  """
  @spec describe_schedule(keyword()) :: map()
  def describe_schedule(opts) do
    mode = Keyword.get(opts, :mode, :interleaved)

    case mode do
      :parallel ->
        num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

        %{
          schedule: List.duplicate(:parallel, num_layers),
          num_backbone: num_layers,
          num_attention: num_layers,
          backbone_pct: 50.0,
          mode: :parallel
        }

      :interleaved ->
        schedule = resolve_schedule(opts)
        num_attn = Enum.count(schedule, &(&1 == :attn))
        num_backbone = length(schedule) - num_attn
        total = length(schedule)
        backbone_pct = if total > 0, do: Float.round(num_backbone / total * 100, 1), else: 0.0

        %{
          schedule: schedule,
          num_backbone: num_backbone,
          num_attention: num_attn,
          backbone_pct: backbone_pct,
          mode: :interleaved
        }
    end
  end

  @doc """
  Get recommended defaults for common hybrid patterns.

  ## Patterns

    - `:nemotron_h` — 90:10 Mamba:attention (Nemotron-H style)
    - `:jamba` — 3:1 interleaved (Jamba style)
    - `:parallel` — Hymba-style parallel in every block
    - `:minimal_attn` — Backbone-heavy with rare attention

  ## Examples

      HybridBuilder.build(HybridBuilder.preset(:nemotron_h) ++ [embed_dim: 256])
  """
  @spec preset(atom()) :: keyword()
  def preset(:nemotron_h) do
    [
      mode: :interleaved,
      num_layers: 10,
      ratio: {9, 1},
      backbone: :mamba,
      hidden_size: 256,
      num_heads: 4,
      head_dim: 64,
      dropout: 0.1
    ]
  end

  def preset(:jamba) do
    [
      mode: :interleaved,
      num_layers: 6,
      attention_every: 3,
      backbone: :mamba,
      hidden_size: 256,
      num_heads: 4,
      head_dim: 64,
      dropout: 0.1
    ]
  end

  def preset(:parallel) do
    [
      mode: :parallel,
      num_layers: 6,
      hidden_size: 256,
      num_heads: 4,
      head_dim: 64,
      dropout: 0.1
    ]
  end

  def preset(:minimal_attn) do
    [
      mode: :interleaved,
      num_layers: 12,
      ratio: {11, 1},
      backbone: :mamba,
      hidden_size: 256,
      num_heads: 4,
      head_dim: 64,
      dropout: 0.1
    ]
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 4,
      head_dim: 64,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      window_size: 60,
      dropout: 0.1,
      backbone: :mamba,
      mode: :interleaved,
      attention_every: 4
    ]
  end
end
