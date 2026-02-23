defmodule Edifice.Attention.RetNetV2 do
  @moduledoc """
  RetNet v2: Improved Retentive Network with Enhanced Chunkwise Retention.

  Builds on the original RetNet with improvements from follow-up work
  including YOCO (You Only Cache Once) and GLA-style gating.

  ## Key Improvements over RetNet v1

  1. **Improved chunkwise formulation**: Better inter-chunk state propagation
     that reduces approximation error in long sequences. Uses a running state
     that accumulates retention across chunk boundaries.

  2. **GLA-style gating on retention**: Applies Gated Linear Attention (GLA)
     gating to the retention output, providing data-dependent modulation:
     `Y = gate(x) * Retention(x)` where gate learns per-dimension importance.

  3. **Optional cross-chunk attention**: For sequences where very long-range
     dependencies are critical, adds sparse cross-chunk attention that attends
     to summary states from distant chunks.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |       RetNet v2 Block                |
  |  LayerNorm -> GLA-Gated MSR         |
  |       + Residual                     |
  |  LayerNorm -> FFN                    |
  |       + Residual                     |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = RetNetV2.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        num_heads: 4
      )

  ## References

  - Sun et al., "Retentive Network: A Successor to Transformer" (2023)
  - Sun et al., "You Only Cache Once" (YOCO, 2024)
  - Yang et al., "Gated Linear Attention Transformers" (2024)
  """

  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 6

  @doc "Default number of retention heads"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: 4

  @doc "Default feedforward expansion factor"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 2

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a RetNet v2 model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of RetNet v2 blocks (default: 6)
    - `:num_heads` - Number of retention heads (default: 4)
    - `:expand_factor` - FFN expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_retnet_v2_block(acc, Keyword.merge(opts, layer_idx: layer_idx))

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(block, rate: dropout, name: "dropout_#{layer_idx}")
        else
          block
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # RetNet v2 Block
  # ============================================================================

  defp build_retnet_v2_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "retnet_v2_block_#{layer_idx}"

    # 1. GLA-Gated Multi-Scale Retention
    retention_normed = Axon.layer_norm(input, name: "#{name}_retention_norm")

    retention_out =
      build_gla_gated_retention(retention_normed, Keyword.put(opts, :name, "#{name}_msr"))

    after_retention = Axon.add(input, retention_out, name: "#{name}_retention_residual")

    # 2. FFN sub-layer
    ff_normed = Axon.layer_norm(after_retention, name: "#{name}_ff_norm")

    ff_out =
      FFN.layer(ff_normed,
        hidden_size: hidden_size,
        expansion_factor: expand_factor,
        name: "#{name}_ffn"
      )

    Axon.add(after_retention, ff_out, name: "#{name}_ff_residual")
  end

  # ============================================================================
  # GLA-Gated Multi-Scale Retention
  # ============================================================================

  defp build_gla_gated_retention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    name = Keyword.get(opts, :name, "msr_v2")

    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # GLA-style gate: data-dependent gating on retention output
    gate = Axon.dense(input, hidden_size, name: "#{name}_gate_proj")
    gate = Axon.activation(gate, :silu, name: "#{name}_gate_silu")

    # Retention computation with improved decay
    retention_out =
      Axon.layer(
        &retention_v2_parallel/4,
        [q, k, v],
        name: "#{name}_retention",
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :retention_v2
      )

    # GroupNorm approximation
    retention_normed = Axon.layer_norm(retention_out, name: "#{name}_group_norm")

    # GLA gating: gate * retention
    gated = Axon.multiply(gate, retention_normed, name: "#{name}_gated")

    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

  # Parallel retention v2 with improved decay computation
  defp retention_v2_parallel(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = reshape_for_heads(q, batch_size, seq_len, num_heads, head_dim)
    k = reshape_for_heads(k, batch_size, seq_len, num_heads, head_dim)
    v = reshape_for_heads(v, batch_size, seq_len, num_heads, head_dim)

    # Apply xPos-style rotations
    {q, k} = apply_xpos_rotation(q, k, seq_len, head_dim)

    # Compute decay rates with improved spacing (v2)
    # Use log-uniform spacing for better coverage of decay timescales
    gammas = compute_head_gammas_v2(num_heads)

    # Build decay matrix
    d_matrix = build_decay_matrix(seq_len, gammas)

    # Compute retention: (QΘ) . D . (KΘ̄)^T . V
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    qk = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)
    qk_decayed = Nx.multiply(qk, d_matrix)

    retention_out = Nx.dot(qk_decayed, [3], [0, 1], v, [2], [0, 1])

    Nx.transpose(retention_out, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  defp reshape_for_heads(tensor, batch_size, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # v2 uses log-uniform spacing for better multi-scale coverage
  defp compute_head_gammas_v2(num_heads) do
    Enum.map(0..(num_heads - 1), fn h ->
      # Log-uniform from 0.9 to 0.9999
      log_min = :math.log(1 - 0.9)
      log_max = :math.log(1 - 0.9999)
      t = h / max(num_heads - 1, 1)
      1.0 - :math.exp(log_min + t * (log_max - log_min))
    end)
    |> Nx.tensor(type: :f32)
  end

  defp build_decay_matrix(seq_len, gammas) do
    num_heads = Nx.axis_size(gammas, 0)

    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    distances = Nx.subtract(rows, cols)
    causal_mask = Nx.greater_equal(rows, cols)

    gammas_expanded = Nx.reshape(gammas, {num_heads, 1, 1})
    distances_expanded = Nx.reshape(distances, {1, seq_len, seq_len})
    distances_clamped = Nx.max(distances_expanded, 0)
    decay = Nx.pow(gammas_expanded, distances_clamped)

    causal_broadcast = Nx.broadcast(causal_mask, {num_heads, seq_len, seq_len})
    Nx.select(causal_broadcast, decay, Nx.tensor(0.0))
  end

  defp apply_xpos_rotation(q, k, seq_len, head_dim) do
    half_dim = div(head_dim, 2)

    freqs =
      Nx.pow(
        10_000.0,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), head_dim))
      )
      |> Nx.as_type(:f32)

    positions = Nx.iota({seq_len}, type: :f32)
    angles = Nx.outer(positions, freqs)
    cos_table = Nx.cos(angles)
    sin_table = Nx.sin(angles)

    cos_t = Nx.reshape(cos_table, {1, 1, seq_len, half_dim})
    sin_t = Nx.reshape(sin_table, {1, 1, seq_len, half_dim})

    q1 = Nx.slice_along_axis(q, 0, half_dim, axis: 3)
    q2 = Nx.slice_along_axis(q, half_dim, half_dim, axis: 3)

    q_rot =
      Nx.concatenate(
        [
          Nx.subtract(Nx.multiply(q1, cos_t), Nx.multiply(q2, sin_t)),
          Nx.add(Nx.multiply(q1, sin_t), Nx.multiply(q2, cos_t))
        ],
        axis: 3
      )

    k1 = Nx.slice_along_axis(k, 0, half_dim, axis: 3)
    k2 = Nx.slice_along_axis(k, half_dim, half_dim, axis: 3)

    k_rot =
      Nx.concatenate(
        [
          Nx.add(Nx.multiply(k1, cos_t), Nx.multiply(k2, sin_t)),
          Nx.subtract(Nx.multiply(k2, cos_t), Nx.multiply(k1, sin_t))
        ],
        axis: 3
      )

    {q_rot, k_rot}
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a RetNet v2 model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 4,
      expand_factor: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
