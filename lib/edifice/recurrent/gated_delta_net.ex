defmodule Edifice.Recurrent.GatedDeltaNet do
  @moduledoc """
  Gated DeltaNet - Linear Attention with Gated Delta Rule.

  Extends DeltaNet with a data-dependent gating mechanism that modulates
  the state matrix between timesteps. Where vanilla DeltaNet always retains
  all of S_{t-1} (modulated only by the delta correction), Gated DeltaNet
  introduces a forget gate alpha_t that controls how much of the previous
  state to retain before applying the delta update.

  This gives the model explicit control over memory erasure, which is
  critical for tasks that require forgetting stale associations.

  ## Key Innovations

  - **Gated state transition**: S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
  - **Data-dependent forgetting**: alpha_t = sigmoid(W_alpha x_t) controls memory decay
  - **Short convolution**: Optional causal convolution before Q/K/V projections for local context
  - **Swish gate on output**: Gated output projection for expressivity

  ## Equations

  ```
  q_t = W_q x_t                              # Query projection
  k_t = W_k x_t                              # Key projection (L2 normalized)
  v_t = W_v x_t                              # Value projection
  beta_t = sigmoid(W_beta x_t)               # Update gate (write strength)
  alpha_t = sigmoid(W_alpha x_t)             # Forget gate (retention)
  S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) * k_t^T  # Gated delta rule
  o_t = swish(W_g x_t) * (S_t q_t)          # Gated output
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +----------------------------------------------+
  |      Gated DeltaNet Layer                     |
  |  Short Conv (optional) for local context      |
  |  Project to Q, K, V, beta, alpha, gate        |
  |  For each timestep:                           |
  |    S = alpha * S + beta * (v - S@k) * k^T     |
  |    output = swish(gate) * (S @ q)             |
  +----------------------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Compared to DeltaNet

  | Aspect | DeltaNet | Gated DeltaNet |
  |--------|----------|----------------|
  | State update | S + beta * error * k^T | alpha * S + beta * error * k^T |
  | Forgetting | Implicit (via delta correction) | Explicit (alpha gate) |
  | Output gating | None | Swish gate |
  | Local context | None | Optional short convolution |
  | Expressivity | Lower | Higher (data-dependent dynamics) |

  ## Usage

      model = GatedDeltaNet.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        use_short_conv: true,
        dropout: 0.1
      )

  ## References
  - "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang et al., 2024)
  - https://arxiv.org/abs/2412.06464
  - Adopted by Qwen3-Next and Kimi Linear (Moonshot AI)
  """

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_dropout 0.1
  @default_conv_size 4
  @norm_eps 1.0e-6

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: @default_hidden_size

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: @default_num_layers

  @doc "Default number of attention heads"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: @default_num_heads

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: @default_dropout

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Gated DeltaNet model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of independent gated delta rule heads (default: 4)
    - `:num_layers` - Number of Gated DeltaNet layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:use_short_conv` - Use short causal convolution before projections (default: true)
    - `:conv_size` - Kernel size for short convolution (default: 4)
    - `:window_size` - Expected sequence length (default: 60)
    - `:seq_len` - Alias for window_size

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:conv_size, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:use_short_conv, boolean()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_short_conv = Keyword.get(opts, :use_short_conv, true)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
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

    # Stack Gated DeltaNet layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer =
          build_gated_delta_net_layer(
            acc,
            hidden_size,
            num_heads,
            use_short_conv,
            conv_size,
            "gated_delta_net_#{layer_idx}"
          )

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

  @doc """
  Build a single Gated DeltaNet block that can be used as a backbone layer
  in hybrid architectures.

  Takes input of shape [batch, seq_len, hidden_size] and returns the same shape.
  Includes pre-norm and residual connection.

  ## Options
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_heads` - Number of heads (default: 4)
    - `:use_short_conv` - Use short causal convolution (default: true)
    - `:conv_size` - Convolution kernel size (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:name` - Layer name prefix (default: "gated_delta_net_block")
  """
  @spec build_block(Axon.t(), keyword()) :: Axon.t()
  def build_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    use_short_conv = Keyword.get(opts, :use_short_conv, true)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    name = Keyword.get(opts, :name, "gated_delta_net_block")

    build_gated_delta_net_layer(input, hidden_size, num_heads, use_short_conv, conv_size, name)
  end

  # ============================================================================
  # Gated DeltaNet Layer
  # ============================================================================

  defp build_gated_delta_net_layer(input, hidden_size, num_heads, use_short_conv, conv_size, name) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Optional short causal convolution for local context mixing
    conv_input =
      if use_short_conv do
        build_short_conv(normed, hidden_size, conv_size, name)
      else
        normed
      end

    # Project to Q, K, V, beta (update), alpha (forget), gate: 5*hidden + hidden
    # Q, K, V, beta: 4 * hidden_size (same as DeltaNet)
    # alpha (forget gate): hidden_size
    # gate (output gate): hidden_size
    projections = Axon.dense(conv_input, hidden_size * 6, name: "#{name}_qkvbag_proj")

    # Apply gated delta rule recurrence
    recurrence_output =
      Axon.nx(
        projections,
        fn proj ->
          gated_delta_net_scan(proj, hidden_size, num_heads)
        end,
        name: "#{name}_recurrence"
      )

    # Output projection
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Short causal convolution: applies a 1D depthwise convolution along the
  # sequence dimension for local context mixing before Q/K/V projections.
  # This helps the model capture local patterns that complement the global
  # associative memory.
  defp build_short_conv(input, hidden_size, conv_size, name) do
    # Input: [batch, seq, hidden]. Axon.conv default is channels-last,
    # so [batch, spatial, channels] maps directly to our layout.

    # Depthwise 1D convolution with causal padding (pad left, no pad right)
    conved =
      Axon.conv(input, hidden_size,
        kernel_size: conv_size,
        padding: [{conv_size - 1, 0}],
        feature_group_size: hidden_size,
        name: "#{name}_short_conv"
      )

    # Apply SiLU activation: x * sigmoid(x)
    Axon.activation(conved, :silu, name: "#{name}_conv_silu")
  end

  defp gated_delta_net_scan(projections, hidden_size, num_heads) do
    # projections: [batch, seq_len, hidden_size * 6]
    batch_size = Nx.axis_size(projections, 0)
    seq_len = Nx.axis_size(projections, 1)
    head_dim = div(hidden_size, num_heads)

    # Split into Q, K, V, beta, alpha, gate
    q_all = Nx.slice_along_axis(projections, 0, hidden_size, axis: 2)
    k_all = Nx.slice_along_axis(projections, hidden_size, hidden_size, axis: 2)
    v_all = Nx.slice_along_axis(projections, hidden_size * 2, hidden_size, axis: 2)
    beta_pre = Nx.slice_along_axis(projections, hidden_size * 3, hidden_size, axis: 2)
    alpha_pre = Nx.slice_along_axis(projections, hidden_size * 4, hidden_size, axis: 2)
    gate_pre = Nx.slice_along_axis(projections, hidden_size * 5, hidden_size, axis: 2)

    # Gates: beta controls write strength, alpha controls memory retention
    beta = Nx.sigmoid(beta_pre)
    alpha = Nx.sigmoid(alpha_pre)
    # Output gate: swish(gate) = gate * sigmoid(gate)
    gate = Nx.multiply(gate_pre, Nx.sigmoid(gate_pre))

    # Reshape to multi-head: [batch, seq, num_heads, head_dim]
    q_all = Nx.reshape(q_all, {batch_size, seq_len, num_heads, head_dim})
    k_all = Nx.reshape(k_all, {batch_size, seq_len, num_heads, head_dim})
    v_all = Nx.reshape(v_all, {batch_size, seq_len, num_heads, head_dim})
    beta = Nx.reshape(beta, {batch_size, seq_len, num_heads, head_dim})
    alpha = Nx.reshape(alpha, {batch_size, seq_len, num_heads, head_dim})

    # L2 normalize keys per head for stable memory updates
    k_norm =
      Nx.sqrt(Nx.add(Nx.sum(Nx.pow(k_all, 2), axes: [3], keep_axes: true), @norm_eps))

    k_normalized = Nx.divide(k_all, k_norm)

    # Initialize per-head memory: [batch, num_heads, head_dim, head_dim]
    s_init = Nx.broadcast(0.0, {batch_size, num_heads, head_dim, head_dim})

    # Sequential scan with gated delta rule — vectorized across heads
    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {s_init, []}, fn t, {s_prev, acc} ->
        # Extract timestep t: [batch, num_heads, head_dim]
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_normalized, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Gated state retention: alpha controls how much of previous state to keep
        # alpha_t is [batch, num_heads, head_dim], need to broadcast to [batch, H, d, d]
        # We use the mean across head_dim as a scalar gate per head
        alpha_scalar =
          Nx.mean(alpha_t, axes: [2])
          |> Nx.new_axis(2)
          |> Nx.new_axis(3)

        s_gated = Nx.multiply(alpha_scalar, s_prev)

        # Retrieval: S @ k per head: [batch, num_heads, head_dim]
        retrieval =
          Nx.dot(s_gated, [3], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1])
          |> Nx.squeeze(axes: [3])

        # Error: v_t - retrieval
        error = Nx.subtract(v_t, retrieval)

        # Delta update: S = alpha*S + beta * error * k^T (outer product per head)
        scaled_error = Nx.multiply(beta_t, error)
        update = Nx.multiply(Nx.new_axis(scaled_error, 3), Nx.new_axis(k_t, 2))

        s_t = Nx.add(s_gated, update)

        # Output: S @ q per head
        o_t =
          Nx.dot(s_t, [3], [0, 1], Nx.new_axis(q_t, 3), [2], [0, 1])
          |> Nx.squeeze(axes: [3])

        # Flatten heads: [batch, num_heads * head_dim]
        o_flat = Nx.reshape(o_t, {batch_size, num_heads * head_dim})

        {s_t, [o_flat | acc]}
      end)

    # Stack and apply output gate
    # output: [batch, seq_len, hidden_size]
    raw_output = output_list |> Enum.reverse() |> Nx.stack(axis: 1)

    # Apply swish output gate: gate is [batch, seq_len, hidden_size]
    Nx.multiply(gate, raw_output)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Gated DeltaNet model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
