defmodule Edifice.Recurrent.DeltaProduct do
  @moduledoc """
  DeltaProduct: Multi-step DeltaNet via Householder Products.

  Extends DeltaNet/GatedDeltaNet by taking n_h gradient steps per token
  instead of one. Each step applies a generalized Householder transformation
  `(I - beta * k * k^T)` to the state matrix, producing rank-n_h state
  transitions that can express rotations and complex dynamics.

  ## Key Idea

  DeltaNet performs one online gradient step per token on an associative
  recall loss, yielding rank-1 updates. DeltaProduct takes n_h steps with
  different (k, v, beta) projections, producing:

      A(x_i) = product_{j=1}^{n_h} (I - beta_{i,j} * k_{i,j} * k_{i,j}^T)

  This product of Householder transformations has rank-n_h, enabling:
  - Rotations (products of two reflections)
  - State tracking of permutation groups
  - Better language modeling quality

  ## Architecture

  Uses the same outer structure as GatedDeltaNet (norm, projections, recurrence,
  gated output). The key difference: K, V, and beta are projected n_h times
  (once per step), while Q is projected once (shared across steps).

  The implementation interleaves the n_h steps into an expanded sequence of
  length T*n_h and runs the standard delta rule recurrence, extracting every
  n_h-th output.

  ```
  Input [batch, seq_len, embed_dim]
        |
  +-------------------------------------------------+
  | DeltaProduct Layer                              |
  | Q: [batch, seq, h*d]     (shared across steps) |
  | K: [batch, seq, n_h*h*d] (per-step keys)       |
  | V: [batch, seq, n_h*h*d] (per-step values)     |
  | beta: [batch, seq, n_h*h] (per-step scalars)   |
  |                                                  |
  | Interleave to length T*n_h, run delta scan,     |
  | extract every n_h-th output                     |
  +-------------------------------------------------+
        |
  Output [batch, seq_len, hidden_size]
  ```

  ## Usage

      model = DeltaProduct.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_householder: 2,
        num_layers: 4
      )

  ## Reference

  - Keller et al., "DeltaProduct: Improving State-Tracking in Linear RNNs
    via Householder Products" (NeurIPS 2025)
  """

  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_num_householder 2
  @default_dropout 0.1
  @default_conv_size 4
  @norm_eps 1.0e-6

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_householder, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:use_short_conv, boolean()}
          | {:conv_size, pos_integer()}
          | {:allow_neg_eigval, boolean()}
          | {:window_size, pos_integer()}
          | {:seq_len, pos_integer()}

  @doc """
  Build a DeltaProduct model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_householder` - Number of Householder steps per token (default: 2)
    - `:num_layers` - Number of layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:use_short_conv` - Use causal convolution (default: true)
    - `:conv_size` - Conv kernel size (default: 4)
    - `:allow_neg_eigval` - Beta range [0,2] for rotations (default: true)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.

  ## Examples

      iex> model = Edifice.Recurrent.DeltaProduct.build(embed_dim: 32, hidden_size: 16, num_heads: 2, num_householder: 2, num_layers: 1)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_householder = Keyword.get(opts, :num_householder, @default_num_householder)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_short_conv = Keyword.get(opts, :use_short_conv, true)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    allow_neg_eigval = Keyword.get(opts, :allow_neg_eigval, true)
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
        layer =
          build_delta_product_layer(
            acc,
            hidden_size: hidden_size,
            num_heads: num_heads,
            num_householder: num_householder,
            use_short_conv: use_short_conv,
            conv_size: conv_size,
            allow_neg_eigval: allow_neg_eigval,
            name: "delta_product_#{layer_idx}"
          )

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_delta_product_layer(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    num_householder = Keyword.fetch!(opts, :num_householder)
    use_short_conv = Keyword.get(opts, :use_short_conv, true)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    allow_neg_eigval = Keyword.get(opts, :allow_neg_eigval, true)
    name = Keyword.get(opts, :name, "delta_product")

    head_dim = div(hidden_size, num_heads)

    # Pre-norm
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Optional short convolution for local context
    conv_input =
      if use_short_conv do
        Axon.conv(normed, hidden_size,
          kernel_size: {conv_size},
          padding: [{conv_size - 1, 0}],
          feature_group_size: hidden_size,
          name: "#{name}_short_conv"
        )
        |> Axon.activation(:silu, name: "#{name}_conv_act")
      else
        normed
      end

    # Q projection (shared, not per-step): [batch, seq, hidden_size]
    q_proj = Axon.dense(conv_input, hidden_size, name: "#{name}_q_proj")

    # K projection (per-step): [batch, seq, hidden_size * n_h]
    k_proj = Axon.dense(conv_input, hidden_size * num_householder, name: "#{name}_k_proj")

    # V projection (per-step): [batch, seq, hidden_size * n_h]
    v_proj = Axon.dense(conv_input, hidden_size * num_householder, name: "#{name}_v_proj")

    # Beta projection (per-step, scalar per head): [batch, seq, num_heads * n_h]
    beta_proj = Axon.dense(conv_input, num_heads * num_householder, name: "#{name}_beta_proj")

    # Output gate
    gate_proj = Axon.dense(conv_input, hidden_size, name: "#{name}_gate_proj")
    gate = Axon.activation(gate_proj, :silu, name: "#{name}_gate_act")

    # DeltaProduct recurrence
    recurrence_output =
      Axon.layer(
        &delta_product_recurrence/6,
        [q_proj, k_proj, v_proj, beta_proj, gate],
        name: "#{name}_recurrence",
        num_heads: num_heads,
        head_dim: head_dim,
        num_householder: num_householder,
        allow_neg_eigval: allow_neg_eigval,
        op_name: :delta_product
      )

    # Output projection
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  defp delta_product_recurrence(q, k_all, v_all, beta_all, gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    num_householder = opts[:num_householder]
    allow_neg_eigval = opts[:allow_neg_eigval]

    {batch, seq_len, _hidden} = Nx.shape(q)
    hidden_size = num_heads * head_dim

    # Reshape Q to multi-head: [batch, seq, num_heads, head_dim]
    q_heads = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})

    # Reshape K to per-step multi-head: [batch, seq, n_h, num_heads, head_dim]
    k_heads =
      k_all
      |> Nx.reshape({batch, seq_len, num_householder, num_heads, head_dim})

    # Reshape V similarly
    v_heads =
      v_all
      |> Nx.reshape({batch, seq_len, num_householder, num_heads, head_dim})

    # Beta: [batch, seq, n_h, num_heads]
    beta_raw =
      beta_all
      |> Nx.reshape({batch, seq_len, num_householder, num_heads})

    # Apply sigmoid (or 2*sigmoid for allow_neg_eigval)
    beta =
      if allow_neg_eigval do
        Nx.multiply(2.0, Nx.sigmoid(beta_raw))
      else
        Nx.sigmoid(beta_raw)
      end

    # Run per-head recurrence with interleaved Householder steps
    # Process each head independently
    # For simplicity, flatten heads into batch dimension
    # q: [batch * num_heads, seq, head_dim]
    q_flat =
      Nx.reshape(
        Nx.transpose(q_heads, axes: [0, 2, 1, 3]),
        {batch * num_heads, seq_len, head_dim}
      )

    # k: [batch * num_heads, seq, n_h, head_dim]
    k_flat =
      Nx.reshape(
        Nx.transpose(k_heads, axes: [0, 3, 1, 2, 4]),
        {batch * num_heads, seq_len, num_householder, head_dim}
      )

    # v: [batch * num_heads, seq, n_h, head_dim]
    v_flat =
      Nx.reshape(
        Nx.transpose(v_heads, axes: [0, 3, 1, 2, 4]),
        {batch * num_heads, seq_len, num_householder, head_dim}
      )

    # beta: [batch * num_heads, seq, n_h]
    beta_flat =
      Nx.reshape(
        Nx.transpose(beta, axes: [0, 3, 1, 2]),
        {batch * num_heads, seq_len, num_householder}
      )

    # Sequential recurrence (DeltaProduct is inherently sequential per-token,
    # but the n_h steps within each token are also sequential)
    output =
      delta_product_sequential(
        q_flat,
        k_flat,
        v_flat,
        beta_flat,
        seq_len,
        num_householder,
        head_dim
      )

    # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden_size]
    output_heads = Nx.reshape(output, {batch, num_heads, seq_len, head_dim})

    output_seq =
      Nx.reshape(Nx.transpose(output_heads, axes: [0, 2, 1, 3]), {batch, seq_len, hidden_size})

    # Apply output gate
    Nx.multiply(output_seq, gate)
  end

  defp delta_product_sequential(q, k, v, beta, seq_len, num_householder, head_dim) do
    # q: [bh, seq, head_dim]
    # k: [bh, seq, n_h, head_dim]
    # v: [bh, seq, n_h, head_dim]
    # beta: [bh, seq, n_h]
    bh = Nx.axis_size(q, 0)

    # State matrix S: [bh, head_dim, head_dim] initialized to zeros
    s_init = Nx.broadcast(0.0, {bh, head_dim, head_dim})

    # Process each timestep
    {_s_final, outputs} =
      Enum.reduce(0..(seq_len - 1)//1, {s_init, []}, fn t, {s_prev, acc} ->
        # For each Householder step j = 0..n_h-1:
        # S = (I - beta_{t,j} * k_{t,j} k_{t,j}^T) * S + beta_{t,j} * k_{t,j} * v_{t,j}^T
        s_updated =
          Enum.reduce(0..(num_householder - 1)//1, s_prev, fn j, s_acc ->
            # k_{t,j}: [bh, head_dim]
            k_tj =
              Nx.squeeze(
                Nx.slice_along_axis(Nx.slice_along_axis(k, t, 1, axis: 1), j, 1, axis: 2),
                axes: [1, 2]
              )

            v_tj =
              Nx.squeeze(
                Nx.slice_along_axis(Nx.slice_along_axis(v, t, 1, axis: 1), j, 1, axis: 2),
                axes: [1, 2]
              )

            beta_tj =
              Nx.squeeze(
                Nx.slice_along_axis(Nx.slice_along_axis(beta, t, 1, axis: 1), j, 1, axis: 2),
                axes: [1, 2]
              )

            # L2 normalize key
            k_norm =
              Nx.sqrt(
                Nx.add(Nx.sum(Nx.multiply(k_tj, k_tj), axes: [1], keep_axes: true), @norm_eps)
              )

            k_normalized = Nx.divide(k_tj, k_norm)

            # beta_tj: [bh] -> [bh, 1, 1] for broadcasting
            beta_expanded = Nx.reshape(beta_tj, {bh, 1, 1})

            # k k^T: [bh, head_dim, head_dim]
            k_col = Nx.new_axis(k_normalized, 2)
            k_row = Nx.new_axis(k_normalized, 1)
            kk_t = Nx.dot(k_col, [2], [0], k_row, [1], [0])

            # S = (I - beta * k k^T) * S + beta * k * v^T
            # = S - beta * (k k^T) * S + beta * k * v^T
            # First term: S - beta * (k k^T @ S)
            kkt_s = Nx.dot(kk_t, [2], [0], s_acc, [1], [0])
            s_after_decay = Nx.subtract(s_acc, Nx.multiply(beta_expanded, kkt_s))

            # Second term: + beta * k * v^T
            v_row = Nx.new_axis(v_tj, 1)
            kv_t = Nx.dot(k_col, [2], [0], v_row, [1], [0])
            Nx.add(s_after_decay, Nx.multiply(beta_expanded, kv_t))
          end)

        # Output: o_t = S_t @ q_t
        # q_t: [bh, head_dim]
        q_t = Nx.squeeze(Nx.slice_along_axis(q, t, 1, axis: 1), axes: [1])
        # S_t @ q_t: [bh, head_dim, head_dim] @ [bh, head_dim] -> [bh, head_dim]
        o_t = Nx.dot(s_updated, [2], [0], Nx.new_axis(q_t, 2), [1], [0])
        o_t = Nx.squeeze(o_t, axes: [2])

        # Apply RMS norm per output
        rms =
          Nx.sqrt(Nx.add(Nx.mean(Nx.multiply(o_t, o_t), axes: [1], keep_axes: true), @norm_eps))

        o_t_normed = Nx.divide(o_t, rms)

        {s_updated, [o_t_normed | acc]}
      end)

    # Stack outputs: [bh, seq, head_dim]
    outputs
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a DeltaProduct model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_householder: 2,
      num_layers: 4,
      dropout: 0.1,
      use_short_conv: true,
      conv_size: 4,
      allow_neg_eigval: true
    ]
  end
end
