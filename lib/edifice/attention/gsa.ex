defmodule Edifice.Attention.GSA do
  @moduledoc """
  GSA: Gated Slot Attention with low-rank memory bottleneck.

  GSA is a two-pass variant of Gated Linear Attention (GLA) that uses a
  slot memory bottleneck to reduce per-head state from {d, d} to {m, d},
  where m is the number of slots and m << d. This makes it significantly
  more memory-efficient than GLA while retaining competitive performance.

  ## Key Innovation: Slot Memory Bottleneck

  Instead of maintaining a full d x d state matrix per head (as in GLA),
  GSA maintains m slot vectors of dimension d. Keys are projected to an
  m-dimensional slot space, creating a low-rank approximation:

  ```
  Write: slot_mem[h,s,:] = alpha * slot_mem[h,s,:] + (1-alpha) * k_slot[h,s] * v[h,:]
  Read:  scores = softmax(slot_mem @ q), output = scores^T @ slot_mem
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +------------------------------------------+
  |  GSA Block                                |
  |                                           |
  |  Q, K, V projections + alpha gate         |
  |  K -> slot projection [H*m]               |
  |         |                                 |
  |  Sequential scan:                         |
  |    Write: update slot memory with gating  |
  |    Read: softmax attention over slots     |
  |         |                                 |
  |  Output projection + residual             |
  |  SwiGLU FFN + residual                    |
  +------------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | GLA | GSA |
  |--------|-----|-----|
  | State per head | O(d^2) | O(m*d) |
  | Time | O(L*d^2) | O(L*m*d) |
  | Space | O(d^2) | O(m*d) |

  ## Usage

      model = GSA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        num_heads: 4,
        num_slots: 16
      )

  ## Reference

  - Paper: "Gated Slot Attention for Efficient Linear-Time Sequence Modeling" (NeurIPS 2024)
  - Repository: https://github.com/fla-org/flash-linear-attention
  """

  alias Edifice.Blocks.FFN

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_num_slots 16
  @default_damping 8
  @default_expand_factor 2
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a GSA model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of GSA blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_slots` - Number of memory slots per head (default: 16)
    - `:damping` - Gate damping temperature tau (default: 8)
    - `:expand_factor` - FFN expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:damping, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_slots, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_slots = Keyword.get(opts, :num_slots, @default_num_slots)
    damping = Keyword.get(opts, :damping, @default_damping)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack GSA blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_gsa_block(
          acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          num_slots: num_slots,
          damping: damping,
          expand_factor: expand_factor,
          dropout: dropout,
          name: "gsa_block_#{layer_idx}"
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
  Build a single GSA block.

  Each block has:
  1. Gated Slot Attention layer (pre-norm + attention + output proj + residual)
  2. Gated FFN (norm + SwiGLU + residual)
  """
  @spec build_gsa_block(Axon.t(), keyword()) :: Axon.t()
  def build_gsa_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_slots = Keyword.get(opts, :num_slots, @default_num_slots)
    damping = Keyword.get(opts, :damping, @default_damping)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gsa_block")

    # Gated Slot Attention
    x =
      build_gated_slot_attention(input,
        hidden_size: hidden_size,
        num_heads: num_heads,
        num_slots: num_slots,
        damping: damping,
        dropout: dropout,
        name: "#{name}_attention"
      )

    # Gated FFN (norm + SwiGLU + residual)
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.gated_layer(ffn_normed,
        hidden_size: hidden_size,
        inner_size: hidden_size * expand_factor,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the Gated Slot Attention layer.

  Components:
  1. Pre-norm
  2. Q, K, V projections with SiLU activation on Q and K
  3. K slot projection (head_dim -> num_slots)
  4. Alpha gate with damped sigmoid
  5. Two-pass sequential scan over timesteps
  6. Output projection + dropout + residual
  """
  @spec build_gated_slot_attention(Axon.t(), keyword()) :: Axon.t()
  def build_gated_slot_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_slots = Keyword.get(opts, :num_slots, @default_num_slots)
    damping = Keyword.get(opts, :damping, @default_damping)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gsa")

    head_dim = div(hidden_size, num_heads)
    attn_dim = num_heads * head_dim

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Q, K, V projections with SiLU activation on Q and K
    q_proj = Axon.dense(x, attn_dim, name: "#{name}_q_proj")
    q_proj = Axon.activation(q_proj, :silu, name: "#{name}_q_silu")

    k_proj = Axon.dense(x, attn_dim, name: "#{name}_k_proj")
    k_proj = Axon.activation(k_proj, :silu, name: "#{name}_k_silu")

    v_proj = Axon.dense(x, attn_dim, name: "#{name}_v_proj")

    # K slot projection: project each head's key from head_dim to num_slots
    # Total: num_heads * num_slots output dims
    k_slot_proj = Axon.dense(k_proj, num_heads * num_slots, name: "#{name}_k_slot_proj")

    # Alpha gate logits: one per head
    alpha_proj = Axon.dense(x, num_heads, name: "#{name}_alpha_proj")

    # Two-pass gated slot attention via sequential scan
    output =
      Axon.layer(
        &gsa_scan_impl/5,
        [q_proj, k_slot_proj, v_proj, alpha_proj],
        name: "#{name}_scan",
        num_heads: num_heads,
        num_slots: num_slots,
        head_dim: head_dim,
        damping: damping,
        op_name: :gated_slot_attention
      )

    # Output projection
    output = Axon.dense(output, hidden_size, name: "#{name}_output")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Two-pass gated slot attention implementation using sequential scan.
  #
  # q: [batch, seq_len, num_heads * head_dim]
  # k_slot: [batch, seq_len, num_heads * num_slots]
  # v: [batch, seq_len, num_heads * head_dim]
  # alpha_logits: [batch, seq_len, num_heads]
  #
  # Returns: [batch, seq_len, num_heads * head_dim]
  defp gsa_scan_impl(q, k_slot, v, alpha_logits, opts) do
    num_heads = opts[:num_heads]
    num_slots = opts[:num_slots]
    head_dim = opts[:head_dim]
    damping = opts[:damping]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head format
    # q: [batch, seq_len, H, d]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    # k_slot: [batch, seq_len, H, m]
    k_slot = Nx.reshape(k_slot, {batch, seq_len, num_heads, num_slots})
    # v: [batch, seq_len, H, d]
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})

    # Apply ELU+1 feature map to Q for positive features
    # ELU+1(x) = 1 + x if x > 0, exp(x) if x <= 0
    q = elu_plus_one(q)

    # Apply softmax to k_slot over slot dimension for write distribution
    k_slot = stable_softmax(k_slot, 3)

    # Compute damped sigmoid gate: alpha = sigmoid(logit)^(1/tau)
    # alpha_logits: [batch, seq_len, H]
    tau = Nx.tensor(damping, type: :f32)
    inv_tau = Nx.divide(1.0, tau)
    alpha = Nx.pow(Nx.sigmoid(alpha_logits), inv_tau)

    # Initialize slot memory: [batch, H, m, d]
    slot_mem = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, num_heads, num_slots, head_dim})

    # Sequential scan over timesteps
    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {slot_mem, []}, fn t, {mem, acc} ->
        # Extract timestep slices: squeeze out the seq dim
        # q_t: [batch, H, d]
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        # k_slot_t: [batch, H, m]
        k_slot_t = Nx.slice_along_axis(k_slot, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        # v_t: [batch, H, d]
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        # alpha_t: [batch, H]
        alpha_t = Nx.slice_along_axis(alpha, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Broadcast alpha for slot memory: [batch, H, 1, 1]
        alpha_broadcast = alpha_t |> Nx.new_axis(2) |> Nx.new_axis(3)

        # Write pass: update slot memory
        # Outer product of slot keys and values: k_slot_t:[B,H,m,1] * v_t:[B,H,1,d] -> [B,H,m,d]
        kv_outer = Nx.multiply(Nx.new_axis(k_slot_t, 3), Nx.new_axis(v_t, 2))

        # Gated update: mem = alpha * mem + (1 - alpha) * kv_outer
        mem_new =
          Nx.add(
            Nx.multiply(alpha_broadcast, mem),
            Nx.multiply(Nx.subtract(1.0, alpha_broadcast), kv_outer)
          )

        # Read pass: compute attention scores over slots
        # scores = mem @ q_t: [B,H,m,d] @ [B,H,d,1] -> [B,H,m]
        q_t_expanded = Nx.new_axis(q_t, 3)

        scores =
          Nx.dot(mem_new, [3], [0, 1], q_t_expanded, [2], [0, 1])
          |> Nx.squeeze(axes: [3])

        # Softmax over slots: [B, H, m]
        p = stable_softmax(scores, 2)

        # Weighted read: sum_s(p[s] * mem[s,:]) -> [B, H, d]
        # p:[B,H,m,1] * mem:[B,H,m,d] -> sum over m -> [B,H,d]
        output_t = Nx.sum(Nx.multiply(Nx.new_axis(p, 3), mem_new), axes: [2])

        # Flatten heads: [batch, H * d]
        o_flat = Nx.reshape(output_t, {batch, num_heads * head_dim})

        {mem_new, [o_flat | acc]}
      end)

    # Stack timesteps: [batch, seq_len, H * d]
    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ELU+1 feature map: ensures strictly positive outputs for valid attention.
  # ELU+1(x) = 1 + x if x > 0, exp(x) if x <= 0
  defp elu_plus_one(x) do
    Nx.add(1.0, Nx.select(Nx.greater(x, 0.0), x, Nx.subtract(Nx.exp(x), 1.0)))
  end

  # Numerically stable softmax along the given axis.
  defp stable_softmax(x, axis) do
    max_val = Nx.reduce_max(x, axes: [axis], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_val))
    sum_exp = Nx.sum(exp_x, axes: [axis], keep_axes: true)
    Nx.divide(exp_x, Nx.add(sum_exp, 1.0e-8))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a GSA model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a GSA model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_slots = Keyword.get(opts, :num_slots, @default_num_slots)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)

    head_dim = div(hidden_size, num_heads)
    attn_dim = num_heads * head_dim
    inner_size = hidden_size * expand_factor

    # Per layer:
    # Attention:
    #   - Q, K, V projections: 3 * hidden * attn_dim
    #   - K slot projection: attn_dim * (num_heads * num_slots)
    #   - Alpha projection: hidden * num_heads
    #   - Output projection: attn_dim * hidden
    attention_params =
      3 * hidden_size * attn_dim +
        attn_dim * (num_heads * num_slots) +
        hidden_size * num_heads +
        attn_dim * hidden_size

    # FFN (GLU style):
    #   - Gate, Up projections: 2 * hidden * inner
    #   - Down projection: inner * hidden
    ffn_params =
      2 * hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = attention_params + ffn_params

    # Input projection
    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      num_heads: 4,
      num_slots: 16,
      damping: 8,
      expand_factor: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
