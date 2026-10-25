defmodule Edifice.Attention.RLA do
  @moduledoc """
  RLA/RDN: Residual Linear Attention with error-correcting dual-state recurrence.

  Maintains two state matrices -- base state S and residual state R -- where R
  tracks and corrects retrieval errors from S. This dual-state approach gives
  the model an explicit error-correction mechanism that improves associative
  memory accuracy over single-state methods like DeltaNet.

  ## Variants

  - `:rla` -- Moving average state updates with residual error tracking
  - `:rdn` -- Delta-rule state updates (Householder-like error correction on both states)

  ## Key Innovations

  - **Dual state**: Base state S stores primary associations, residual state R corrects errors
  - **Error clipping**: Residual errors are clipped for training stability
  - **Three gates**: alpha (decay), beta (base learning rate), gamma (residual learning rate)
  - **SiLU + L2-norm**: Applied to Q and K for stable retrieval

  ## Equations

  ```
  # Per-timestep update (RLA variant):
  retrieval = S_{t-1} @ k_t
  r_t = clip(v_t - retrieval, -c, c)
  S_t = alpha_t * S_{t-1} + beta_t * outer(v_t, k_t)
  R_t = alpha_t * R_{t-1} + gamma_t * outer(r_t, k_t)
  o_t = (S_t + R_t) @ q_t

  # Per-timestep update (RDN variant / delta rule):
  retrieval_s = S_{t-1} @ k_t
  retrieval_r = R_{t-1} @ k_t
  r_t = clip(v_t - retrieval_s, -c, c)
  S_t = alpha_t * S_{t-1} + beta_t * outer(v_t - retrieval_s, k_t)
  R_t = alpha_t * R_{t-1} + gamma_t * outer(r_t - retrieval_r, k_t)
  o_t = (S_t + R_t) @ q_t
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +------------------------------------------+
  |      RLA/RDN Block                        |
  |  Pre-norm                                 |
  |  Project to Q, K, V + 3 gates             |
  |  SiLU + L2-norm on Q, K                   |
  |  For each timestep:                       |
  |    retrieval = S @ k                      |
  |    r = clip(v - retrieval)                |
  |    S = alpha*S + beta*outer(v, k)  [RLA]  |
  |    R = alpha*R + gamma*outer(r, k) [RLA]  |
  |    output = (S + R) @ q                   |
  |  Output projection + residual             |
  |  FFN + residual                           |
  +------------------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = RLA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4,
        variant: :rla,
        dropout: 0.1
      )

      # RDN (delta-rule) variant:
      model = RLA.build(
        embed_dim: 287,
        hidden_size: 256,
        variant: :rdn
      )

  ## References

  - "Residual Linear Attention" (2025 preprint)
  - Extends DeltaNet with dual-state error correction
  """

  alias Edifice.Blocks.FFN

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_clip_threshold 1.0
  @default_dropout 0.1
  @default_window_size 60
  @norm_eps 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an RLA/RDN model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of RLA blocks (default: 4)
    - `:variant` - `:rla` (moving average) or `:rdn` (delta rule) (default: `:rla`)
    - `:clip_threshold` - Clipping bound for residual errors (default: 1.0)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:seq_len` - Alias for window_size

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:clip_threshold, float()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:variant, :rla | :rdn}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    variant = Keyword.get(opts, :variant, :rla)
    clip_threshold = Keyword.get(opts, :clip_threshold, @default_clip_threshold)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack RLA blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_rla_block(
          acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          variant: variant,
          clip_threshold: clip_threshold,
          dropout: dropout,
          name: "rla_block_#{layer_idx}",
          is_last: layer_idx == num_layers
        )
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # RLA Block
  # ============================================================================

  defp build_rla_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    variant = Keyword.fetch!(opts, :variant)
    clip_threshold = Keyword.fetch!(opts, :clip_threshold)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "rla_block")
    is_last = Keyword.get(opts, :is_last, false)

    # 1. Attention sublayer: pre-norm -> RLA recurrence -> output proj -> residual
    normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    # Project to Q, K, V (each hidden_size) + 3 scalar gates per head (each num_heads)
    # Gates are projected as hidden_size each for per-head-dim granularity,
    # then reduced to per-head scalars inside the scan.
    # Total: Q + K + V + alpha + beta + gamma = 6 * hidden_size
    projections = Axon.dense(normed, hidden_size * 6, name: "#{name}_qkvabg_proj")

    # Apply dual-state recurrence
    recurrence_output =
      Axon.nx(
        projections,
        fn proj ->
          rla_scan(proj, hidden_size, num_heads, variant, clip_threshold)
        end,
        name: "#{name}_recurrence"
      )

    # Output projection
    attn_out = Axon.dense(recurrence_output, hidden_size, name: "#{name}_attn_out_proj")

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer: norm -> FFN -> residual
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        name: "#{name}_ffn"
      )

    ffn_out =
      if is_last do
        ffn_out
      else
        maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
      end

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # RLA/RDN Sequential Scan
  # ============================================================================

  defp rla_scan(projections, hidden_size, num_heads, variant, clip_threshold) do
    # projections: [batch, seq_len, hidden_size * 6]
    batch_size = Nx.axis_size(projections, 0)
    seq_len = Nx.axis_size(projections, 1)
    head_dim = div(hidden_size, num_heads)

    # Split into Q, K, V, alpha_pre, beta_pre, gamma_pre
    q_all = Nx.slice_along_axis(projections, 0, hidden_size, axis: 2)
    k_all = Nx.slice_along_axis(projections, hidden_size, hidden_size, axis: 2)
    v_all = Nx.slice_along_axis(projections, hidden_size * 2, hidden_size, axis: 2)
    alpha_pre = Nx.slice_along_axis(projections, hidden_size * 3, hidden_size, axis: 2)
    beta_pre = Nx.slice_along_axis(projections, hidden_size * 4, hidden_size, axis: 2)
    gamma_pre = Nx.slice_along_axis(projections, hidden_size * 5, hidden_size, axis: 2)

    # Reshape to multi-head: [batch, seq, num_heads, head_dim]
    q_all = Nx.reshape(q_all, {batch_size, seq_len, num_heads, head_dim})
    k_all = Nx.reshape(k_all, {batch_size, seq_len, num_heads, head_dim})
    v_all = Nx.reshape(v_all, {batch_size, seq_len, num_heads, head_dim})

    # Apply SiLU + L2-normalize on Q and K
    q_all = silu_l2_normalize(q_all)
    k_all = silu_l2_normalize(k_all)

    # Sigmoid gates: [batch, seq, num_heads, head_dim]
    alpha = Nx.sigmoid(Nx.reshape(alpha_pre, {batch_size, seq_len, num_heads, head_dim}))
    beta = Nx.sigmoid(Nx.reshape(beta_pre, {batch_size, seq_len, num_heads, head_dim}))
    gamma = Nx.sigmoid(Nx.reshape(gamma_pre, {batch_size, seq_len, num_heads, head_dim}))

    # Reduce gates to per-head scalars: [B, T, H, d] -> [B, T, H, 1, 1]
    alpha_scalar = reduce_gate_to_scalar(alpha)
    beta_scalar = reduce_gate_to_scalar(beta)
    gamma_scalar = reduce_gate_to_scalar(gamma)

    # Fused RLA scan via CUDA dispatch (3-tier: custom call -> NIF -> Elixir)
    # Returns [B, T, H, d]
    scan_output =
      Edifice.CUDA.FusedScan.rla_scan(
        q_all,
        k_all,
        v_all,
        alpha_scalar,
        beta_scalar,
        gamma_scalar,
        variant: variant,
        clip_threshold: clip_threshold
      )

    # Flatten heads: [batch, seq_len, num_heads * head_dim]
    Nx.reshape(scan_output, {batch_size, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  # Apply SiLU activation then L2-normalize per head dimension.
  # Input: [batch, seq, num_heads, head_dim]
  # Output: [batch, seq, num_heads, head_dim]
  defp silu_l2_normalize(x) do
    # SiLU: x * sigmoid(x)
    activated = Nx.multiply(x, Nx.sigmoid(x))

    # L2 normalize along head_dim (last axis)
    norm = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(activated, 2), axes: [-1], keep_axes: true), @norm_eps))
    Nx.divide(activated, norm)
  end

  # Reduce per-head-dim gate to scalar per head (full sequence): [B, T, H, d] -> [B, T, H, 1, 1]
  defp reduce_gate_to_scalar(gate) do
    Nx.mean(gate, axes: [3])
    |> Nx.new_axis(3)
    |> Nx.new_axis(4)
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an RLA/RDN model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_layers: 4,
      variant: :rla,
      clip_threshold: 1.0,
      dropout: 0.1,
      window_size: 60
    ]
  end
end
