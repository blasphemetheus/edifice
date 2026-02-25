defmodule Edifice.SSM.Hymba do
  @moduledoc """
  Hymba: Hybrid-head Architecture with Parallel Mamba + Attention.

  Implements the Hymba architecture from "Hymba: A Hybrid-head Architecture
  for Small Language Models" (NVIDIA, 2024). Unlike sequential hybrid models
  (Jamba, Zamba), Hymba runs Mamba and attention **in parallel** within each
  block, with learnable gated fusion.

  ## Key Innovations

  1. **Parallel Mamba + Attention**: Both paths process the same input
     simultaneously, and outputs are combined via a learnable gate:
     `output = gate * mamba_out + (1 - gate) * attn_out`

  2. **Learnable Meta Tokens**: K learnable vectors prepended to K/V in
     the attention path. These serve as "summarizers" that compress global
     context, reducing the effective attention complexity while maintaining
     long-range access.

  3. **Cross-layer meta token propagation**: Meta token states are updated
     across layers, accumulating information throughout the network.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |         Hymba Block                  |
  |                                      |
  |  +--------+    +------------------+  |
  |  | Mamba   |    | Attention       |  |
  |  | (SSM)   |    | + Meta Tokens   |  |
  |  +----+----+    +--------+--------+  |
  |       |                  |           |
  |       v                  v           |
  |  gate * mamba + (1-gate) * attn      |
  |            |                         |
  |            v                         |
  |       residual + FFN                 |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Compared to Other Hybrids

  | Model | Mamba + Attention | Pattern |
  |-------|-------------------|---------|
  | Jamba | Alternating | Sequential layers |
  | Zamba | Shared attention | Interleaved |
  | Hymba | Parallel heads | Within each block |

  ## Usage

      model = Hymba.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        num_meta_tokens: 4
      )

  ## References

  - Dong et al., "Hymba: A Hybrid-head Architecture for Small Language Models"
    (NVIDIA, 2024)
  - https://arxiv.org/abs/2411.13676
  """

  alias Edifice.SSM.Common
  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_meta_tokens, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default SSM state dimension"
  @spec default_state_size() :: pos_integer()
  def default_state_size, do: 16

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default number of attention heads"
  @spec default_num_heads() :: pos_integer()
  def default_num_heads, do: 4

  @doc "Default number of learnable meta tokens"
  @spec default_num_meta_tokens() :: pos_integer()
  def default_num_meta_tokens, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Hymba model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension (default: 16)
    - `:num_layers` - Number of Hymba blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_meta_tokens` - Learnable meta tokens for attention (default: 4)
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
        block = build_hymba_block(acc, Keyword.merge(opts, layer_idx: layer_idx))

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
  # Hymba Block (Parallel Mamba + Attention with Gated Fusion)
  # ============================================================================

  defp build_hymba_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    state_size = Keyword.get(opts, :state_size, default_state_size())
    num_heads = Keyword.get(opts, :num_heads, default_num_heads())
    num_meta = Keyword.get(opts, :num_meta_tokens, default_num_meta_tokens())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "hymba_block_#{layer_idx}"

    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # --- Mamba path (SSM) ---
    mamba_out = build_mamba_path(normed, hidden_size, state_size, name)

    # --- Attention path (with meta tokens) ---
    attn_out = build_attention_with_meta(normed, hidden_size, num_heads, num_meta, name)

    # --- Gated fusion ---
    # gate: sigmoid learned gate per dimension
    gate =
      normed
      |> Axon.dense(hidden_size, name: "#{name}_fusion_gate_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_fusion_gate")

    # output = gate * mamba + (1 - gate) * attn
    fused =
      Axon.layer(
        fn mamba, attn, g, _opts ->
          Nx.add(
            Nx.multiply(g, mamba),
            Nx.multiply(Nx.subtract(1.0, g), attn)
          )
        end,
        [mamba_out, attn_out, gate],
        name: "#{name}_fusion",
        op_name: :gated_fusion
      )

    # Residual + FFN
    after_fusion = Axon.add(input, fused, name: "#{name}_residual")

    ffn_normed = Axon.layer_norm(after_fusion, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        expansion_factor: 4,
        name: "#{name}_ffn"
      )

    Axon.add(after_fusion, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Mamba Path (simplified SSM)
  # ============================================================================

  defp build_mamba_path(input, hidden_size, state_size, name) do
    expand_factor = 2
    inner_size = hidden_size * expand_factor

    # Project to inner dimensions
    xz = Axon.dense(input, inner_size * 2, name: "#{name}_mamba_in_proj")

    x_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, 0, inner_size, axis: 2) end,
        name: "#{name}_mamba_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor -> Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2) end,
        name: "#{name}_mamba_z_split"
      )

    # Depthwise conv + SiLU
    x_conv =
      Axon.conv(x_branch, inner_size,
        kernel_size: {4},
        padding: [{3, 0}],
        feature_group_size: inner_size,
        name: "#{name}_mamba_conv"
      )

    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_mamba_silu")

    # SSM scan
    ssm_out = build_ssm_scan(x_activated, inner_size, state_size, "#{name}_mamba_ssm")

    # Gate and project
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_mamba_gate")
    gated = Axon.multiply(ssm_out, z_activated, name: "#{name}_mamba_gated")
    Axon.dense(gated, hidden_size, name: "#{name}_mamba_out_proj")
  end

  defp build_ssm_scan(input, hidden_size, state_size, name) do
    {b_matrix, c_matrix, dt_proj} =
      Common.build_ssm_projections(input,
        hidden_size: hidden_size,
        state_size: state_size,
        name: name
      )

    Axon.layer(
      &ssm_scan_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: "#{name}_scan",
      state_size: state_size,
      op_name: :ssm_scan
    )
  end

  defp ssm_scan_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)
    h = Common.blelloch_scan(a_bar, bx)
    Common.compute_ssm_output(h, c)
  end

  # ============================================================================
  # Attention Path with Meta Tokens
  # ============================================================================

  defp build_attention_with_meta(input, hidden_size, num_heads, num_meta, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_attn_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_attn_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_attn_v")

    # Learnable meta tokens: [num_meta, hidden_size]
    meta_k =
      Axon.param("#{name}_meta_k", {num_meta, hidden_size}, initializer: :glorot_uniform)

    meta_v =
      Axon.param("#{name}_meta_v", {num_meta, hidden_size}, initializer: :glorot_uniform)

    # Attention with meta tokens prepended to K, V
    attn_out =
      Axon.layer(
        &attention_with_meta_impl/6,
        [q, k, v, meta_k, meta_v],
        name: "#{name}_attn_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        num_meta: num_meta,
        op_name: :attention_with_meta
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_attn_out_proj")
  end

  defp attention_with_meta_impl(q, k, v, meta_k, meta_v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    num_meta = opts[:num_meta]

    batch_size = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Prepend meta tokens to K, V
    # meta_k: [num_meta, hidden_size] -> [batch, num_meta, hidden_size]
    meta_k_batch =
      Nx.broadcast(Nx.new_axis(meta_k, 0), {batch_size, num_meta, num_heads * head_dim})

    meta_v_batch =
      Nx.broadcast(Nx.new_axis(meta_v, 0), {batch_size, num_meta, num_heads * head_dim})

    k_with_meta = Nx.concatenate([meta_k_batch, k], axis: 1)
    v_with_meta = Nx.concatenate([meta_v_batch, v], axis: 1)

    kv_len = seq_len + num_meta

    # Reshape for multi-head attention
    q_heads =
      q
      |> Nx.reshape({batch_size, seq_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k_heads =
      k_with_meta
      |> Nx.reshape({batch_size, kv_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v_heads =
      v_with_meta
      |> Nx.reshape({batch_size, kv_len, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q_heads, [3], [0, 1], k_heads, [3], [0, 1]) |> Nx.divide(scale)

    # Causal mask: queries attend to meta tokens + past/current regular tokens
    # Meta tokens (columns 0..num_meta-1) are always visible
    # Regular tokens (columns num_meta..kv_len-1) have causal structure
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, kv_len})

    # Meta tokens always visible (cols < num_meta)
    meta_visible = Nx.less(cols, num_meta)
    # Regular tokens: causal (query_pos >= key_pos - num_meta)
    regular_causal = Nx.greater_equal(Nx.add(rows, num_meta), cols)
    causal_mask = Nx.logical_or(meta_visible, regular_causal)
    causal_mask = Nx.reshape(causal_mask, {1, 1, seq_len, kv_len})

    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
    scores = Nx.select(Nx.broadcast(causal_mask, Nx.shape(scores)), scores, neg_inf)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply to values
    output = Nx.dot(attn_weights, [3], [0, 1], v_heads, [2], [0, 1])

    # Reshape back
    Nx.transpose(output, axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a Hymba model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      num_layers: 4,
      num_heads: 4,
      num_meta_tokens: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
