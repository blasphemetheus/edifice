defmodule Edifice.Transformer.NemotronH do
  @moduledoc """
  Nemotron-H: NVIDIA's Hybrid Mamba-Transformer Architecture.

  Nemotron-H is a hybrid language model that combines 90% Mamba2 (SSD) layers with
  10% full attention layers. This design achieves Transformer-level quality while
  maintaining linear inference cost from the SSM components.

  ## Key Innovation: Hybrid Layer Mixing

  Rather than using all-attention or all-SSM, Nemotron-H interleaves them:
  - 90% of layers use Mamba2 (State Space Duality) for efficient linear-time processing
  - 10% of layers use full multi-head attention for global reasoning
  - Attention blocks placed at regular intervals (every 10th layer by default)

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Shared Embedding Projection]
        |
        v
  +========================================+
  |            Layer 0 (Mamba2)            |
  |  RMSNorm -> Mamba2 SSD -> Residual     |
  |  RMSNorm -> SwiGLU FFN -> Residual     |
  +========================================+
        |
       ... (Mamba2 layers 1-8)
        |
  +========================================+
  |           Layer 9 (Attention)          |
  |  RMSNorm -> MultiHead Attn -> Residual |
  |  RMSNorm -> SwiGLU FFN -> Residual     |
  +========================================+
        |
       ... (pattern repeats)
        |
        v
  [Final RMSNorm]
        |
        v
  [Output Projection (tied weights)]
        |
        v
  Output [batch, hidden_dim]
  ```

  ## Mamba2 (SSD) Blocks

  Use State Space Duality from Mamba-2:
  - Chunked matmul for tensor core utilization
  - Selective state space with input-dependent parameters
  - Depthwise convolution + gating

  ## Attention Blocks

  Standard multi-head attention with:
  - Grouped Query Attention (optional)
  - RoPE position embeddings (optional)
  - Causal masking

  ## Usage

      model = NemotronH.build(
        embed_dim: 287,
        hidden_dim: 2048,
        num_layers: 32,
        attention_every_n: 10,
        num_heads: 16
      )

  ## References

  - Paper: "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer
    Language Models" (NVIDIA, 2025)
  - Mamba-2: "Transformers are SSMs" (Gu & Dao, 2024)
  """

  alias Edifice.Blocks.{FFN, RMSNorm, TransformerBlock}
  alias Edifice.SSM.Common, as: SSMCommon

  # Default hyperparameters (Nemotron-H style)
  @default_hidden_dim 2048
  @default_num_layers 32
  @default_attention_every_n 10
  @default_num_heads 16
  @default_num_kv_heads 4
  @default_mamba_d_state 64
  @default_mamba_d_conv 4
  @default_mamba_expand 2
  @default_dropout 0.0
  @default_chunk_size 16

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:attention_every_n, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:mamba_d_state, pos_integer()}
          | {:mamba_d_conv, pos_integer()}
          | {:mamba_expand, pos_integer()}
          | {:dropout, float()}
          | {:rope, boolean()}
          | {:window_size, pos_integer()}
          | {:seq_len, pos_integer()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Nemotron-H hybrid model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_dim` - Model hidden dimension (default: 2048)
    - `:num_layers` - Total number of layers (default: 32)
    - `:attention_every_n` - Place attention at every Nth layer (default: 10)
    - `:num_heads` - Number of attention heads (default: 16)
    - `:num_kv_heads` - Number of KV heads for GQA (default: 4)
    - `:mamba_d_state` - Mamba SSM state dimension (default: 64)
    - `:mamba_d_conv` - Mamba convolution kernel size (default: 4)
    - `:mamba_expand` - Mamba expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:rope` - Apply RoPE to attention layers (default: false)
    - `:window_size` / `:seq_len` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_dim]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_dim do
        Axon.dense(input, hidden_dim, name: "input_projection")
      else
        input
      end

    # Stack hybrid blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn layer_idx, acc ->
        block_opts = Keyword.put(opts, :layer_idx, layer_idx)
        block = nemotron_block(acc, layer_idx, block_opts)

        # Add residual connection + optional dropout
        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers - 1 do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Final layer norm
    x = RMSNorm.layer(x, hidden_size: hidden_dim, name: "final_norm")

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
  Build a single Nemotron-H block.

  Dispatches to either Mamba2 or attention based on the block index.
  Attention blocks are placed at positions where `block_idx % attention_every_n == (attention_every_n - 1)`.

  ## Parameters

    - `input` - Input Axon node
    - `block_idx` - 0-indexed block position
    - `opts` - Model options

  ## Returns

    Block output (before residual connection).
  """
  @spec nemotron_block(Axon.t(), non_neg_integer(), keyword()) :: Axon.t()
  def nemotron_block(input, block_idx, opts) do
    attention_every_n = Keyword.get(opts, :attention_every_n, @default_attention_every_n)

    # Place attention at every Nth layer (0-indexed: positions 9, 19, 29, ...)
    is_attention_layer = rem(block_idx + 1, attention_every_n) == 0

    if is_attention_layer do
      build_attention_block(input, opts)
    else
      build_mamba_block(input, opts)
    end
  end

  @doc """
  Build a Mamba2 (SSD) block with RMSNorm and SwiGLU FFN.

  Architecture: RMSNorm -> Mamba2 -> (no residual here, handled by caller)
                RMSNorm -> SwiGLU FFN -> (no residual here)

  ## Options

    Same as `build/1`, plus `:layer_idx` for naming.
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    d_state = Keyword.get(opts, :mamba_d_state, @default_mamba_d_state)
    d_conv = Keyword.get(opts, :mamba_d_conv, @default_mamba_d_conv)
    expand = Keyword.get(opts, :mamba_expand, @default_mamba_expand)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    layer_idx = Keyword.get(opts, :layer_idx, 0)
    name = "mamba_block_#{layer_idx}"

    inner_size = hidden_dim * expand

    # First sublayer: RMSNorm -> Mamba2 SSD
    normed = RMSNorm.layer(input, hidden_size: hidden_dim, name: "#{name}_norm1")

    # Mamba2 block using SSD algorithm
    mamba_out =
      build_mamba2_sublayer(normed, hidden_dim, inner_size, d_state, d_conv, chunk_size, name)

    # First residual
    x = Axon.add(input, mamba_out, name: "#{name}_residual1")

    # Second sublayer: RMSNorm -> SwiGLU FFN
    normed2 = RMSNorm.layer(x, hidden_size: hidden_dim, name: "#{name}_norm2")
    ffn_out = FFN.gated_layer(normed2, hidden_size: hidden_dim, name: "#{name}_ffn")

    # Second residual (but we return just the FFN output, residual added by caller)
    # Actually, looking at the architecture more carefully, the block should include
    # both sublayers with their internal residuals, and the outer residual wraps the whole block.
    # Let me restructure: the block output should be the delta to add to input.

    # For Nemotron-H, each block is: input + (mamba sublayer) + (ffn sublayer)
    # So we return the sum of mamba_out and ffn_out
    Axon.add(mamba_out, ffn_out, name: "#{name}_combined")
  end

  # Build the Mamba2 (SSD) sublayer
  defp build_mamba2_sublayer(input, hidden_dim, inner_size, d_state, d_conv, chunk_size, name) do
    dt_rank = max(div(hidden_dim, 16), 1)

    # Project to 2x inner_size for x and z branches
    xz = Axon.dense(input, inner_size * 2, name: "#{name}_in_proj")

    # Split into x (SSM path) and z (gating path)
    x_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
        end,
        name: "#{name}_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
        end,
        name: "#{name}_z_split"
      )

    # X branch: Depthwise Conv1D -> SiLU -> SSM
    x_conv = SSMCommon.build_depthwise_conv1d(x_branch, inner_size, d_conv, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    # SSM using SSD-style computation
    x_ssm = build_ssd_ssm(x_activated, inner_size, d_state, dt_rank, chunk_size, name)

    # Z branch: SiLU activation (gating)
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")

    # Multiply x_ssm * z (gated output)
    gated = Axon.multiply(x_ssm, z_activated, name: "#{name}_gated")

    # Project back to hidden_dim
    Axon.dense(gated, hidden_dim, name: "#{name}_out_proj")
  end

  # Build the SSD (State Space Duality) SSM
  defp build_ssd_ssm(input, inner_size, d_state, dt_rank, chunk_size, name) do
    # SSM parameter projections
    # B and C: [batch, seq_len, d_state]
    bc_proj = Axon.dense(input, d_state * 2, name: "#{name}_ssm_bc")

    b_matrix =
      Axon.nx(
        bc_proj,
        fn tensor -> Nx.slice_along_axis(tensor, 0, d_state, axis: 2) end,
        name: "#{name}_ssm_B"
      )

    c_matrix =
      Axon.nx(
        bc_proj,
        fn tensor -> Nx.slice_along_axis(tensor, d_state, d_state, axis: 2) end,
        name: "#{name}_ssm_C"
      )

    # Delta (dt) projection through low-rank bottleneck
    dt_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_ssm_dt_rank")
      |> Axon.dense(inner_size, name: "#{name}_ssm_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_ssm_dt_softplus")

    # Apply SSD scan
    Axon.layer(
      &ssd_scan_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: "#{name}_ssm_compute",
      state_size: d_state,
      hidden_size: inner_size,
      chunk_size: chunk_size,
      op_name: :nemotron_ssd
    )
  end

  # SSD scan implementation
  defp ssd_scan_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    chunk_size = opts[:chunk_size] || @default_chunk_size

    # Discretize SSM parameters
    {a_bar, bx} = SSMCommon.discretize_ssm(x, b, dt, state_size)

    # Apply SSD chunked scan
    h = ssd_chunked_scan(a_bar, bx, chunk_size)

    # Compute output
    SSMCommon.compute_ssm_output(h, c)
  end

  # Chunked SSD scan for efficient processing
  defp ssd_chunked_scan(a, b, chunk_size) do
    seq_len = Nx.axis_size(a, 1)

    if seq_len <= chunk_size do
      SSMCommon.blelloch_scan(a, b)
    else
      # Multi-chunk processing
      batch = Nx.axis_size(a, 0)
      hidden = Nx.axis_size(a, 2)
      state = Nx.axis_size(a, 3)

      num_chunks = div(seq_len, chunk_size)
      remainder = rem(seq_len, chunk_size)

      # Process chunks
      chunk_outputs =
        Enum.map(0..(num_chunks - 1), fn chunk_idx ->
          start = chunk_idx * chunk_size

          a_chunk = Nx.slice_along_axis(a, start, chunk_size, axis: 1)
          b_chunk = Nx.slice_along_axis(b, start, chunk_size, axis: 1)

          SSMCommon.blelloch_scan(a_chunk, b_chunk)
        end)

      # Handle remainder
      chunk_outputs =
        if remainder > 0 do
          start = num_chunks * chunk_size
          a_rem = Nx.slice_along_axis(a, start, remainder, axis: 1)
          b_rem = Nx.slice_along_axis(b, start, remainder, axis: 1)
          chunk_outputs ++ [SSMCommon.blelloch_scan(a_rem, b_rem)]
        else
          chunk_outputs
        end

      # Get final states for inter-chunk propagation
      chunk_final_states =
        Enum.map(chunk_outputs, fn chunk_h ->
          chunk_len = Nx.axis_size(chunk_h, 1)
          Nx.slice_along_axis(chunk_h, chunk_len - 1, 1, axis: 1)
        end)

      # Inter-chunk state propagation
      {_, propagated_outputs} =
        Enum.reduce(
          Enum.with_index(chunk_outputs),
          {Nx.broadcast(0.0, {batch, 1, hidden, state}), []},
          fn {chunk_h, idx}, {running_state, acc} ->
            if idx == 0 do
              new_running = Enum.at(chunk_final_states, idx)
              {new_running, acc ++ [chunk_h]}
            else
              a_chunk = slice_a_chunk(a, idx, chunk_outputs, num_chunks, chunk_size, remainder)

              a_cumprods = compute_cumulative_products(a_chunk)
              state_contribution = Nx.multiply(a_cumprods, running_state)
              adjusted_chunk = Nx.add(chunk_h, state_contribution)

              chunk_len = Nx.axis_size(chunk_h, 1)
              chunk_final = Nx.slice_along_axis(adjusted_chunk, chunk_len - 1, 1, axis: 1)

              {chunk_final, acc ++ [adjusted_chunk]}
            end
          end
        )

      Nx.concatenate(propagated_outputs, axis: 1)
    end
  end

  defp slice_a_chunk(a, idx, chunk_outputs, num_chunks, chunk_size, remainder) do
    if idx == length(chunk_outputs) - 1 and remainder > 0 do
      start = num_chunks * chunk_size
      Nx.slice_along_axis(a, start, remainder, axis: 1)
    else
      start = idx * chunk_size
      Nx.slice_along_axis(a, start, chunk_size, axis: 1)
    end
  end

  # Compute cumulative products along sequence dimension
  defp compute_cumulative_products(a) do
    seq_len = Nx.axis_size(a, 1)

    {_, cumprods} =
      Enum.reduce(0..(seq_len - 1), {nil, []}, fn t, {prev_prod, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)

        new_prod =
          if prev_prod == nil do
            a_t
          else
            Nx.multiply(prev_prod, a_t)
          end

        {new_prod, acc ++ [new_prod]}
      end)

    Nx.concatenate(cumprods, axis: 1)
  end

  @doc """
  Build an attention block with RMSNorm and SwiGLU FFN.

  Architecture: RMSNorm -> MultiHead Attention -> (residual handled by caller)
                RMSNorm -> SwiGLU FFN -> (residual handled by caller)

  ## Options

    Same as `build/1`, plus `:layer_idx` for naming.
  """
  @spec build_attention_block(Axon.t(), keyword()) :: Axon.t()
  def build_attention_block(input, opts) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    use_rope = Keyword.get(opts, :rope, false)
    layer_idx = Keyword.get(opts, :layer_idx, 0)
    name = "attn_block_#{layer_idx}"

    # Use TransformerBlock for the attention layer
    attn_fn = fn x, attn_name ->
      build_gqa_attention(x, hidden_dim, num_heads, num_kv_heads, use_rope, attn_name)
    end

    # First sublayer: RMSNorm -> Attention
    attn_out =
      TransformerBlock.layer(input,
        attention_fn: attn_fn,
        hidden_size: hidden_dim,
        ffn_type: :gated,
        norm: :rms_norm,
        name: name
      )

    # TransformerBlock already handles both sublayers with residuals internally,
    # so we need to compute the delta (output - input) for consistency with our
    # outer residual pattern
    Axon.subtract(attn_out, input, name: "#{name}_delta")
  end

  # Build GQA attention layer
  defp build_gqa_attention(input, hidden_dim, num_heads, num_kv_heads, use_rope, name) do
    head_dim = div(hidden_dim, num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Q, K, V projections
    q_proj = Axon.dense(input, q_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, kv_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, kv_dim, name: "#{name}_v_proj")

    # Compute attention
    output =
      Axon.layer(
        &gqa_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        head_dim: head_dim,
        rope: use_rope,
        op_name: :nemotron_gqa
      )

    # Output projection
    Axon.dense(output, hidden_dim, name: "#{name}_out_proj")
  end

  # GQA attention implementation
  defp gqa_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    num_kv_heads = opts[:num_kv_heads]
    head_dim = opts[:head_dim]
    use_rope = opts[:rope] || false

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)
    heads_per_group = div(num_heads, num_kv_heads)

    # Reshape Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    q =
      q
      |> Nx.reshape({batch, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Reshape K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
    k =
      k
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v =
      v
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply RoPE if enabled
    {q, k} =
      if use_rope do
        q_flat = Nx.reshape(q, {batch * num_heads, seq_len, head_dim})
        k_flat = Nx.reshape(k, {batch * num_kv_heads, seq_len, head_dim})

        {q_rot, k_rot} = Edifice.Blocks.RoPE.apply_rotary(q_flat, k_flat)

        q_out = Nx.reshape(q_rot, {batch, num_heads, seq_len, head_dim})
        k_out = Nx.reshape(k_rot, {batch, num_kv_heads, seq_len, head_dim})
        {q_out, k_out}
      else
        {q, k}
      end

    # Repeat K, V for each group of Q heads
    k =
      k
      |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
      |> Nx.broadcast({batch, num_kv_heads, heads_per_group, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    v =
      v
      |> Nx.reshape({batch, num_kv_heads, 1, seq_len, head_dim})
      |> Nx.broadcast({batch, num_kv_heads, heads_per_group, seq_len, head_dim})
      |> Nx.reshape({batch, num_heads, seq_len, head_dim})

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      Nx.broadcast(
        Nx.reshape(causal_mask, {1, 1, seq_len, seq_len}),
        {batch, num_heads, seq_len, seq_len}
      )

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

    # Weighted sum
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Nemotron-H model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_dim, @default_hidden_dim)
  end

  @doc """
  Calculate approximate parameter count for a Nemotron-H model.
  """
  @spec param_count(keyword()) :: pos_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every_n = Keyword.get(opts, :attention_every_n, @default_attention_every_n)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    d_state = Keyword.get(opts, :mamba_d_state, @default_mamba_d_state)
    expand = Keyword.get(opts, :mamba_expand, @default_mamba_expand)
    d_conv = Keyword.get(opts, :mamba_d_conv, @default_mamba_d_conv)

    num_attention_layers = div(num_layers, attention_every_n)
    num_mamba_layers = num_layers - num_attention_layers

    inner_size = hidden_dim * expand
    dt_rank = max(div(hidden_dim, 16), 1)
    head_dim = div(hidden_dim, num_heads)

    # Mamba layer params
    # in_proj (2 * inner)
    # depthwise conv
    # BC projection
    # dt projection
    # out_proj
    # FFN (gated: 3 * hidden * inner)
    mamba_per_layer =
      hidden_dim * (2 * inner_size) +
        d_conv * inner_size +
        inner_size * (2 * d_state) +
        inner_size * dt_rank + dt_rank * inner_size +
        inner_size * hidden_dim +
        3 * hidden_dim * inner_size

    # Attention layer params
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Q, K, V projections
    # out projection
    # FFN (gated)
    attn_per_layer =
      hidden_dim * q_dim + hidden_dim * kv_dim * 2 +
        q_dim * hidden_dim +
        3 * hidden_dim * inner_size

    input_proj = if embed_dim != hidden_dim, do: embed_dim * hidden_dim, else: 0

    input_proj + num_mamba_layers * mamba_per_layer + num_attention_layers * attn_per_layer
  end

  @doc """
  Recommended default configuration for Nemotron-H.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_dim: 2048,
      num_layers: 32,
      attention_every_n: 10,
      num_heads: 16,
      num_kv_heads: 4,
      mamba_d_state: 64,
      mamba_d_conv: 4,
      mamba_expand: 2,
      window_size: 60,
      dropout: 0.0
    ]
  end

  @doc """
  Get small model configuration (for testing/prototyping).
  """
  @spec small_config() :: keyword()
  def small_config do
    [
      hidden_dim: 256,
      num_layers: 8,
      attention_every_n: 4,
      num_heads: 4,
      num_kv_heads: 2,
      mamba_d_state: 16,
      mamba_d_conv: 4,
      mamba_expand: 2
    ]
  end
end
