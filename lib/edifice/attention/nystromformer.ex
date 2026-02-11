defmodule Edifice.Attention.Nystromformer do
  @moduledoc """
  Nystromformer: Nystrom-based approximation for O(N) attention.

  Approximates the full softmax attention matrix using the Nystrom method
  with landmark points. Instead of computing the full N x N attention matrix,
  it samples M landmark points and reconstructs the attention through them.

  ## Key Innovation: Nystrom Approximation

  The Nystrom method approximates a matrix using a subset of its columns/rows:

  ```
  Full attention:    A = softmax(QK^T / sqrt(d))
  Nystrom approx:    A ~ F1 * pinv(F2) * F3

  Where:
    landmarks = downsample(K, M)     # M landmark points
    F1 = softmax(Q @ landmarks^T)    # [N, M] queries-to-landmarks
    F2 = softmax(landmarks @ landmarks^T)  # [M, M] landmarks-to-landmarks
    F3 = softmax(landmarks @ K^T)    # [M, N] landmarks-to-keys
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |  Nystromformer Block                 |
  |                                      |
  |  LayerNorm                           |
  |    -> Q, K, V projections            |
  |    -> Select M landmarks (avg pool)  |
  |    -> Q-to-landmark attention [N,M]  |
  |    -> Landmark kernel [M,M]          |
  |    -> Landmark-to-K attention [M,N]  |
  |    -> Reconstruct: F1*F2^{-1}*F3*V  |
  |  -> Residual                         |
  |                                      |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Component | Standard | Nystromformer |
  |-----------|----------|---------------|
  | Attention | O(N^2) | O(N*M) |
  | Memory | O(N^2) | O(N*M + M^2) |
  | Kernel inv | - | O(M^3) |

  Where M = num_landmarks << N. Typically M = 32-64 is sufficient.

  ## Usage

      model = Nystromformer.build(
        embed_size: 287,
        hidden_size: 256,
        num_landmarks: 32,
        num_layers: 4,
        num_heads: 4
      )

  ## References
  - Paper: "Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention"
    (Xiong et al., AAAI 2021)
  """

  require Axon

  alias Edifice.Utils.FusedOps

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_landmarks 32
  @default_num_layers 4
  @default_num_heads 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Nystromformer model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_landmarks` - Number of Nystrom landmark points M (default: 32)
    - `:num_layers` - Number of Nystromformer blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    _dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack Nystromformer blocks (dropout applied inside blocks)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_nystrom_block(
          acc,
          Keyword.merge(opts, [name: "nystrom_block_#{layer_idx}"])
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single Nystromformer block.

  Each block has:
  1. LayerNorm -> Nystrom Attention -> Residual
  2. LayerNorm -> FFN -> Residual
  """
  @spec build_nystrom_block(Axon.t(), keyword()) :: Axon.t()
  def build_nystrom_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_landmarks = Keyword.get(opts, :num_landmarks, @default_num_landmarks)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "nystrom_block")

    head_dim = div(hidden_size, num_heads)

    # 1. Nystrom attention branch
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    # Q, K, V projections
    q_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_v_proj")

    # Apply Nystrom attention
    attn_out = Axon.layer(
      &nystrom_attention_impl/4,
      [q_proj, k_proj, v_proj],
      name: "#{name}_nystrom_attn",
      num_heads: num_heads,
      head_dim: head_dim,
      num_landmarks: num_landmarks,
      op_name: :nystrom_attention
    )

    # Output projection + dropout
    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")

    attn_out =
      if dropout > 0 do
        Axon.dropout(attn_out, rate: dropout, name: "#{name}_attn_dropout")
      else
        attn_out
      end

    # Residual
    after_attn = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN branch
    ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")
    ffn_out = build_ffn(ffn_normed, hidden_size, "#{name}_ffn")

    ffn_out =
      if dropout > 0 do
        Axon.dropout(ffn_out, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn_out
      end

    Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
  end

  # Nystrom attention implementation
  # Approximates softmax(QK^T) using landmark points
  defp nystrom_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    num_landmarks = opts[:num_landmarks]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Compute landmarks via segment-mean (average pooling)
    # Divide sequence into num_landmarks segments and average each
    segment_size = max(div(seq_len, num_landmarks), 1)
    actual_landmarks = min(num_landmarks, seq_len)

    # Compute landmarks by reshaping and averaging segments
    # Pad if needed to make seq_len divisible by actual_landmarks
    padded_len = actual_landmarks * segment_size

    # If sequence is shorter than landmarks, use all positions as landmarks
    {q_landmarks, k_landmarks} =
      if seq_len <= actual_landmarks do
        {q, k}
      else
        # Take first padded_len positions, reshape into segments, average
        q_trunc = Nx.slice_along_axis(q, 0, padded_len, axis: 2)
        k_trunc = Nx.slice_along_axis(k, 0, padded_len, axis: 2)

        q_segments = Nx.reshape(q_trunc, {batch, num_heads, actual_landmarks, segment_size, head_dim})
        k_segments = Nx.reshape(k_trunc, {batch, num_heads, actual_landmarks, segment_size, head_dim})

        {Nx.mean(q_segments, axes: [3]), Nx.mean(k_segments, axes: [3])}
      end

    # F1: queries-to-landmarks attention [batch, heads, N, M]
    f1_scores = Nx.dot(q, [3], [0, 1], k_landmarks, [3], [0, 1]) |> Nx.divide(scale)
    f1 = FusedOps.fused_softmax(f1_scores)

    # F2: landmarks-to-landmarks kernel [batch, heads, M, M]
    f2_scores = Nx.dot(q_landmarks, [3], [0, 1], k_landmarks, [3], [0, 1]) |> Nx.divide(scale)
    f2 = FusedOps.fused_softmax(f2_scores)

    # F3: landmarks-to-keys attention [batch, heads, M, N]
    f3_scores = Nx.dot(q_landmarks, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)
    f3 = FusedOps.fused_softmax(f3_scores)

    # Pseudo-inverse of F2 via iterative Newton method
    # pinv(F2) ~ F2^T * (F2 * F2^T)^{-1} approximated iteratively
    f2_inv = iterative_pinv(f2, num_iterations: 6)

    # Reconstruct: output = F1 * pinv(F2) * F3 * V
    # Step 1: F3 * V -> [batch, heads, M, head_dim]
    f3_v = Nx.dot(f3, [3], [0, 1], v, [2], [0, 1])

    # Step 2: pinv(F2) * (F3 * V) -> [batch, heads, M, head_dim]
    f2_inv_f3_v = Nx.dot(f2_inv, [3], [0, 1], f3_v, [2], [0, 1])

    # Step 3: F1 * (pinv(F2) * F3 * V) -> [batch, heads, N, head_dim]
    output = Nx.dot(f1, [3], [0, 1], f2_inv_f3_v, [2], [0, 1])

    # Reshape back: [batch, seq, num_heads * head_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Iterative pseudo-inverse using Newton's method
  # Converges to pinv(A) through: Z_{i+1} = Z_i * (2I - A * Z_i)
  defp iterative_pinv(matrix, opts) do
    num_iterations = Keyword.get(opts, :num_iterations, 6)

    # Initialize: Z_0 = A^T / ||A||^2
    matrix_t = Nx.transpose(matrix, axes: [0, 1, 3, 2])
    norm_sq = Nx.sum(Nx.multiply(matrix, matrix), axes: [2, 3], keep_axes: true)
    z = Nx.divide(matrix_t, Nx.add(norm_sq, 1.0e-6))

    # Identity matrix for the [M, M] dimension
    m = Nx.axis_size(matrix, 2)
    eye = Nx.eye(m) |> Nx.as_type(Nx.type(matrix))

    # Iterate: Z = Z * (2I - A * Z)
    Enum.reduce(1..num_iterations, z, fn _iter, z_acc ->
      # A * Z: [batch, heads, M, M]
      az = Nx.dot(matrix, [3], [0, 1], z_acc, [2], [0, 1])
      # 2I - A * Z
      correction = Nx.subtract(Nx.multiply(2.0, eye), az)
      # Z * correction
      Nx.dot(z_acc, [3], [0, 1], correction, [2], [0, 1])
    end)
  end

  # Feed-forward network
  defp build_ffn(input, hidden_size, name) do
    inner_size = hidden_size * 4

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Nystromformer model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Nystromformer model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer:
    # Attention: Q + K + V + output projections
    attn_params = hidden_size * hidden_size * 4

    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
      inner_size * hidden_size

    per_layer = attn_params + ffn_params

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_landmarks: 32,
      num_layers: 4,
      num_heads: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
