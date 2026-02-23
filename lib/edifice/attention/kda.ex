defmodule Edifice.Attention.KDA do
  @moduledoc """
  KDA: Kimi Delta Attention.

  Implements the KDA mechanism from "Kimi Linear: An Expressive, Efficient
  Attention Architecture" (Moonshot AI, 2025). KDA extends Gated DeltaNet
  with **channel-wise (per-dimension) decay gating**, allowing different
  semantic dimensions to persist at different rates.

  ## Key Innovation: Channel-Wise Decay

  Where Gated DeltaNet uses a scalar alpha per head (all channels decay
  at the same rate), KDA uses a vector alpha in R^{d_k} — each channel
  gets its own independent decay rate.

  ```
  Gated DeltaNet:  S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_t k_t) k_t^T
                   alpha_t is scalar per head

  KDA:             S_t = (I - beta_t * k_t k_t^T) * Diag(alpha_t) * S_{t-1}
                         + beta_t * k_t v_t^T
                   alpha_t is vector per channel (d_k dims)
  ```

  This means syntax cues can persist while recency signals decay rapidly,
  or vice versa — the model learns per-channel memory dynamics.

  ## Architecture

  ```
  Input [batch, seq_len, hidden_size]
        |
        v
  [Pre-LayerNorm]
        |
  +-----+------+--------+--------+--------+
  |     |      |        |        |        |
  Q     K      V     Alpha    Beta    Gate
  |     |      |    (channel) (scalar) (output)
  |     |      |        |        |
  [Short Conv + SiLU]   |        |
  |     |      |        |        |
  [L2 Normalize Q, K]   |        |
  |     |      |        |        |
  +-----+------+--------+--------+
        |
  [KDA Recurrence]
        |
  [RMSNorm * sigmoid(Gate)]
        |
  [Output Projection]
        |
  [Residual Connection]
        |
        v
  Output [batch, seq_len, hidden_size]
  ```

  ## Alpha Gate Production

  The channel-wise decay is produced by a low-rank MLP:
  `alpha_t = sigmoid(W_up * SiLU(W_down * x_t))`

  Stored in log-space for numerical stability.

  ## Kimi Linear Hybrid

  In the full Kimi Linear model, KDA layers are interleaved with MLA
  (Multi-head Latent Attention) at a 3:1 ratio.

  ## Usage

      model = KDA.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4
      )

  ## References
  - Paper: https://arxiv.org/abs/2510.26692
  - Code: https://github.com/MoonshotAI/Kimi-Linear
  - FLA: https://github.com/fla-org/flash-linear-attention
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

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a KDA model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of independent KDA heads (default: 4)
    - `:num_layers` - Number of KDA layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:use_short_conv` - Use short convolution before Q/K/V (default: true)
    - `:conv_size` - Short convolution kernel size (default: 4)
    - `:window_size` / `:seq_len` - Expected sequence length (default: 60)

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

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack KDA layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_kda_layer(acc, hidden_size, num_heads, use_short_conv, conv_size, "kda_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep
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
  Build a single KDA block for use as a hybrid backbone layer.

  Takes [batch, seq_len, hidden_size] and returns the same shape.
  """
  @spec build_block(Axon.t(), keyword()) :: Axon.t()
  def build_block(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    use_short_conv = Keyword.get(opts, :use_short_conv, true)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    name = Keyword.get(opts, :name, "kda_block")

    build_kda_layer(input, hidden_size, num_heads, use_short_conv, conv_size, name)
  end

  # ============================================================================
  # KDA Layer
  # ============================================================================

  defp build_kda_layer(input, hidden_size, num_heads, use_short_conv, conv_size, name) do
    head_dim = div(hidden_size, num_heads)

    # Pre-norm
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Q, K, V projections
    q_proj = Axon.dense(normed, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(normed, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(normed, hidden_size, name: "#{name}_v_proj")

    # Optional short convolution on Q, K, V
    {q, k, v} =
      if use_short_conv do
        q = build_short_conv(q_proj, hidden_size, conv_size, "#{name}_q")
        k = build_short_conv(k_proj, hidden_size, conv_size, "#{name}_k")
        v = build_short_conv(v_proj, hidden_size, conv_size, "#{name}_v")
        {q, k, v}
      else
        q = Axon.activation(q_proj, :silu, name: "#{name}_q_silu")
        k = Axon.activation(k_proj, :silu, name: "#{name}_k_silu")
        v = Axon.activation(v_proj, :silu, name: "#{name}_v_silu")
        {q, k, v}
      end

    # Channel-wise alpha gate: low-rank MLP
    # hidden_size -> bottleneck (head_dim) -> hidden_size
    alpha_gate =
      normed
      |> Axon.dense(head_dim, name: "#{name}_alpha_down")
      |> Axon.activation(:silu, name: "#{name}_alpha_silu")
      |> Axon.dense(hidden_size, name: "#{name}_alpha_up")

    # Beta gate: scalar per head
    beta_gate = Axon.dense(normed, num_heads, name: "#{name}_beta_proj")

    # Output gate: low-rank MLP with sigmoid
    output_gate =
      normed
      |> Axon.dense(head_dim, name: "#{name}_gate_down")
      |> Axon.activation(:silu, name: "#{name}_gate_silu")
      |> Axon.dense(hidden_size, name: "#{name}_gate_up")

    # KDA recurrence
    recurrence_output =
      Axon.layer(
        &kda_recurrence_impl/7,
        [q, k, v, alpha_gate, beta_gate, output_gate],
        name: "#{name}_recurrence",
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :kda_recurrence
      )

    # Output projection
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Short causal convolution with SiLU activation
  defp build_short_conv(input, hidden_size, conv_size, name) do
    input
    |> Axon.conv(hidden_size,
      kernel_size: conv_size,
      padding: [{conv_size - 1, 0}],
      feature_group_size: hidden_size,
      name: "#{name}_conv"
    )
    |> Axon.activation(:silu, name: "#{name}_conv_silu")
  end

  # ============================================================================
  # KDA Recurrence Implementation
  # ============================================================================

  defp kda_recurrence_impl(q, k, v, alpha_pre, beta_pre, gate_pre, opts) do
    _hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, num_heads, head_dim]
    q_all = Nx.reshape(q, {batch_size, seq_len, num_heads, head_dim})
    k_all = Nx.reshape(k, {batch_size, seq_len, num_heads, head_dim})
    v_all = Nx.reshape(v, {batch_size, seq_len, num_heads, head_dim})

    # L2 normalize Q and K for stability
    q_all = l2_normalize(q_all, axis: 3)
    k_all = l2_normalize(k_all, axis: 3)

    # Alpha: channel-wise decay in log-space
    # alpha_pre: [batch, seq, hidden_size] -> [batch, seq, num_heads, head_dim]
    alpha_log = Nx.reshape(alpha_pre, {batch_size, seq_len, num_heads, head_dim})
    # Apply sigmoid to get decay in (0, 1), then take log for log-space
    alpha_log = Nx.log(Nx.add(Nx.sigmoid(alpha_log), 1.0e-8))

    # Beta: scalar per head, sigmoid
    # beta_pre: [batch, seq, num_heads]
    beta = Nx.sigmoid(beta_pre)

    # Output gate: sigmoid (not SiLU — validated by KDA ablation)
    gate = Nx.sigmoid(gate_pre)

    # Initialize state: [batch, num_heads, head_dim, head_dim]
    s_init = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch_size, num_heads, head_dim, head_dim})

    # Sequential KDA scan
    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {s_init, []}, fn t, {s_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_t = Nx.slice_along_axis(alpha_log, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        beta_t = Nx.slice_along_axis(beta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Channel-wise decay: exp(alpha_log) per channel
        # alpha_t: [batch, num_heads, head_dim] -> broadcast to [batch, heads, d_k, d_v]
        decay = Nx.exp(alpha_t) |> Nx.new_axis(3)
        s_decayed = Nx.multiply(decay, s_prev)

        # Delta rule update: S = (I - beta * k k^T) * decayed_S + beta * k v^T
        # beta_t: [batch, num_heads] -> [batch, num_heads, 1, 1]
        beta_bc = beta_t |> Nx.new_axis(2) |> Nx.new_axis(3)

        # (I - beta * k k^T) * S: erase old value for key k_t before writing new
        # S @ k: [batch, heads, d_v] (what k_t currently retrieves)
        sk = Nx.dot(s_decayed, [2], [0, 1], Nx.new_axis(k_t, 3), [2], [0, 1])
             |> Nx.squeeze(axes: [3])
        # correction = beta * k * (k^T S)^T = beta * outer(k, S^T k)
        erase = Nx.multiply(beta_bc, Nx.multiply(Nx.new_axis(k_t, 3), Nx.new_axis(sk, 2)))

        s_after_erase = Nx.subtract(s_decayed, erase)

        # Add: beta * k v^T
        kv = Nx.multiply(Nx.new_axis(k_t, 3), Nx.new_axis(v_t, 2))
        s_t = Nx.add(s_after_erase, Nx.multiply(beta_bc, kv))

        # Output: h_t = S_t^T @ q_t
        o_t =
          Nx.dot(s_t, [2], [0, 1], Nx.new_axis(q_t, 3), [2], [0, 1])
          |> Nx.squeeze(axes: [3])

        # Flatten heads: [batch, num_heads * head_dim]
        o_flat = Nx.reshape(o_t, {batch_size, num_heads * head_dim})

        {s_t, [o_flat | acc]}
      end)

    # Stack outputs: [batch, seq_len, hidden_size]
    raw_output = output_list |> Enum.reverse() |> Nx.stack(axis: 1)

    # Apply RMSNorm + sigmoid gate
    normed = rms_norm(raw_output)
    Nx.multiply(gate, normed)
  end

  # L2 normalization along a given axis
  defp l2_normalize(tensor, opts) do
    axis = Keyword.fetch!(opts, :axis)
    norm = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(tensor, 2), axes: [axis], keep_axes: true), @norm_eps))
    Nx.divide(tensor, norm)
  end

  # RMSNorm: x / sqrt(mean(x^2) + eps)
  defp rms_norm(x) do
    ms = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(ms, @norm_eps)))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a KDA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
