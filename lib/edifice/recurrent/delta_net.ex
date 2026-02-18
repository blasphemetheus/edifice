defmodule Edifice.Recurrent.DeltaNet do
  @moduledoc """
  DeltaNet - Linear Attention with Delta Rule.

  Implements linear attention with the delta rule update from
  "Linear Transformers with Learnable Kernel Functions are Better
  In-Context Models" (Schlag et al., 2021) and subsequent work.

  DeltaNet maintains an associative memory matrix S that is updated
  using the delta rule, which corrects previous associations rather
  than blindly accumulating them. This gives it superior retrieval
  accuracy compared to standard linear attention.

  ## Key Innovations

  - **Delta rule update**: S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
  - **Error-correcting**: Subtracts the current retrieval S_{t-1} k_t before adding
  - **Learnable beta**: Controls update rate per-token via a gate
  - **Linear complexity**: O(d^2) memory vs O(n*d) for softmax attention

  ## Equations

  ```
  q_t = W_q x_t                          # Query projection
  k_t = W_k x_t                          # Key projection (L2 normalized)
  v_t = W_v x_t                          # Value projection
  beta_t = sigmoid(W_beta x_t)           # Update gate
  S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) * k_t^T   # Delta rule
  o_t = S_t q_t                          # Output retrieval
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +----------------------------------+
  |      DeltaNet Layer              |
  |  Project to Q, K, V, beta        |
  |  For each timestep:              |
  |    error = v - S @ k             |
  |    S += beta * error * k^T       |
  |    output = S @ q                |
  +----------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = DeltaNet.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2102.11174
  - Delta rule RNNs: https://arxiv.org/abs/2310.01655
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.1

  @doc "Epsilon for normalization"
  @spec norm_eps() :: float()
  def norm_eps, do: 1.0e-6

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc "Default number of attention heads"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: 4

  @doc """
  Build a DeltaNet model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of independent delta rule heads (default: 4)
    - `:num_layers` - Number of DeltaNet layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
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

    # Stack DeltaNet layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_delta_net_layer(acc, hidden_size, num_heads, "delta_net_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
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
  # DeltaNet Layer
  # ============================================================================

  defp build_delta_net_layer(input, hidden_size, num_heads, name) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to Q, K, V, beta: 4 * hidden_size total
    projections = Axon.dense(normed, hidden_size * 4, name: "#{name}_qkvb_proj")

    # Apply multi-head delta rule recurrence
    recurrence_output =
      Axon.nx(
        projections,
        fn proj ->
          delta_net_scan(proj, hidden_size, num_heads)
        end,
        name: "#{name}_recurrence"
      )

    # Output projection
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  defp delta_net_scan(projections, hidden_size, num_heads) do
    # projections: [batch, seq_len, hidden_size * 4]
    batch_size = Nx.axis_size(projections, 0)
    seq_len = Nx.axis_size(projections, 1)
    head_dim = div(hidden_size, num_heads)

    # Split into Q, K, V, beta
    q_all = Nx.slice_along_axis(projections, 0, hidden_size, axis: 2)
    k_all = Nx.slice_along_axis(projections, hidden_size, hidden_size, axis: 2)
    v_all = Nx.slice_along_axis(projections, hidden_size * 2, hidden_size, axis: 2)
    beta_pre = Nx.slice_along_axis(projections, hidden_size * 3, hidden_size, axis: 2)

    # Beta gate: controls update rate
    beta = Nx.sigmoid(beta_pre)

    # Reshape to multi-head: [batch, seq, num_heads, head_dim]
    q_all = Nx.reshape(q_all, {batch_size, seq_len, num_heads, head_dim})
    k_all = Nx.reshape(k_all, {batch_size, seq_len, num_heads, head_dim})
    v_all = Nx.reshape(v_all, {batch_size, seq_len, num_heads, head_dim})
    beta = Nx.reshape(beta, {batch_size, seq_len, num_heads, head_dim})

    # L2 normalize keys per head for stable memory updates
    k_norm = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(k_all, 2), axes: [3], keep_axes: true), norm_eps()))
    k_normalized = Nx.divide(k_all, k_norm)

    # Initialize per-head memory: [batch, num_heads, head_dim, head_dim]
    s_init = Nx.broadcast(0.0, {batch_size, num_heads, head_dim, head_dim})

    # Sequential scan with delta rule — vectorized across heads
    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {s_init, []}, fn t, {s_prev, acc} ->
        # [batch, num_heads, head_dim]
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_normalized, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Retrieval: S @ k per head: [batch, num_heads, head_dim]
        # s_prev: [batch, H, d, d], k_t: [batch, H, d] -> [batch, H, d]
        retrieval =
          Nx.dot(s_prev, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1])

        retrieval = Nx.squeeze(retrieval, axes: [3])

        # Error: v_t - retrieval
        error = Nx.subtract(v_t, retrieval)

        # Delta update: S += beta * error * k^T (outer product per head)
        # beta_t * error: [batch, H, d]
        scaled_error = Nx.multiply(beta_t, error)
        # outer: [batch, H, d, 1] * [batch, H, 1, d] -> [batch, H, d, d]
        update = Nx.multiply(Nx.new_axis(scaled_error, 3), Nx.new_axis(k_t, 2))

        s_t = Nx.add(s_prev, update)

        # Output: S @ q per head: [batch, num_heads, head_dim]
        o_t = Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, 3), [2], [0, 1])
        o_t = Nx.squeeze(o_t, axes: [3])

        # Flatten heads: [batch, num_heads * head_dim]
        o_flat = Nx.reshape(o_t, {batch_size, num_heads * head_dim})

        {s_t, [o_flat | acc]}
      end)

    # [batch, seq_len, hidden_size]
    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a DeltaNet model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end
end
