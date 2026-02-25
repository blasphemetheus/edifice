defmodule Edifice.Attention.DiffTransformer do
  @moduledoc """
  Differential Transformer V2: simplified noise-cancelling attention.

  Instead of a single softmax attention map per head, the Differential Transformer
  computes two independent attention maps and subtracts them. Shared noise patterns
  (tokens that attract attention universally) cancel out, amplifying the
  signal-to-noise ratio — analogous to differential amplifiers in electronics.

  ## Key Innovation

  For each head, Q and K are split into two halves:

  ```
  A1 = softmax(Q1 @ K1^T / sqrt(d/2))
  A2 = softmax(Q2 @ K2^T / sqrt(d/2))

  DiffAttn = (A1 - lambda * A2) @ V
  ```

  Where `lambda` is a simple learnable scalar (initialized to layer-dependent value).

  ## V2 Simplifications

  - Lambda is now a single learnable scalar per layer (vs 4-vector parameterization)
  - Removed per-head GroupNorm/SubLayerNorm
  - Uses simple RMSNorm before output projection

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +--------------------------------------+
  |   DiffTransformer Block (x N)        |
  |                                      |
  |   LayerNorm -> Diff Attention        |
  |     Q -> [Q1, Q2], K -> [K1, K2]    |
  |     A1 = softmax(Q1K1^T/s)          |
  |     A2 = softmax(Q2K2^T/s)          |
  |     out = (A1 - lambda*A2) @ V      |
  |     RMSNorm                          |
  |   -> Residual                        |
  |   LayerNorm -> FFN -> Residual       |
  +--------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = DiffTransformer.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6
      )

  ## References

  - "Differential Transformer" (Ye et al., Microsoft Research, 2024)
  - "Differential Transformer V2" (2025) - simplified parameterization
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Utils.FusedOps

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Differential Transformer V2 model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of differential attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "diff_block_#{layer_idx}"

          # lambda_init is depth-dependent per the paper
          lambda_init = 0.8 - 0.6 * :math.exp(-0.3 * max(layer_idx - 1, 0))

          attn_fn = fn x, attn_name ->
            build_diff_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              lambda_init: lambda_init,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  @doc """
  Build the differential attention layer (V2 simplified).

  Projects to Q, K, V, splits Q/K into two halves, computes dual
  softmax attention maps and subtracts them with learnable lambda scalar,
  then applies RMSNorm.
  """
  @spec build_diff_attention(Axon.t(), keyword()) :: Axon.t()
  def build_diff_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    lambda_init = Keyword.get(opts, :lambda_init, 0.8)
    name = Keyword.get(opts, :name, "diff_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K projected to 2x hidden_size (split into 2*num_heads sub-heads of head_dim)
    # V stays at hidden_size (num_heads heads of head_dim)
    q_proj = Axon.dense(input, hidden_size * 2, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size * 2, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # V2: Simple learnable lambda scalar (initialized to layer-dependent value)
    lambda_param =
      Axon.param("#{name}_lambda", {}, initializer: fn _, _ -> Nx.tensor(lambda_init) end)

    # RMSNorm scale parameter
    rms_scale = Axon.param("#{name}_rms_scale", {hidden_size}, initializer: :ones)

    output =
      Axon.layer(
        &diff_attention_impl/6,
        [q_proj, k_proj, v_proj, lambda_param, rms_scale],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :diff_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # V2 Differential attention implementation - simplified
  # Q: [batch, seq, 2*hidden], K: [batch, seq, 2*hidden], V: [batch, seq, hidden]
  # lambda: scalar, rms_scale: [hidden]
  defp diff_attention_impl(q, k, v, lambda, rms_scale, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape Q to [batch, 2*heads, seq, head_dim] then split into Q1, Q2
    q = Nx.reshape(q, {batch, seq_len, 2 * num_heads, head_dim})
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    q1 = Nx.slice_along_axis(q, 0, num_heads, axis: 1)
    q2 = Nx.slice_along_axis(q, num_heads, num_heads, axis: 1)

    # Reshape K similarly: [batch, 2*heads, seq, head_dim] -> K1, K2
    k = Nx.reshape(k, {batch, seq_len, 2 * num_heads, head_dim})
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    k1 = Nx.slice_along_axis(k, 0, num_heads, axis: 1)
    k2 = Nx.slice_along_axis(k, num_heads, num_heads, axis: 1)

    # Reshape V to [batch, heads, seq, head_dim]
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # Scale factor: sqrt(head_dim)
    scale = Nx.sqrt(head_dim) |> Nx.as_type(Nx.type(q))

    # Compute two attention maps
    # A1 = softmax(Q1 @ K1^T / sqrt(d))
    scores1 = Nx.dot(q1, [3], [0, 1], k1, [3], [0, 1])
    scores1 = Nx.divide(scores1, scale)

    # Apply causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores1 = Nx.select(causal_mask, scores1, Nx.broadcast(-1.0e9, Nx.shape(scores1)))
    attn1 = FusedOps.fused_softmax(scores1)

    # A2 = softmax(Q2 @ K2^T / sqrt(d))
    scores2 = Nx.dot(q2, [3], [0, 1], k2, [3], [0, 1])
    scores2 = Nx.divide(scores2, scale)
    scores2 = Nx.select(causal_mask, scores2, Nx.broadcast(-1.0e9, Nx.shape(scores2)))
    attn2 = FusedOps.fused_softmax(scores2)

    # V2: Lambda is now a simple learnable scalar
    # Differential attention: (A1 - lambda * A2) @ V
    diff_attn = Nx.subtract(attn1, Nx.multiply(lambda, attn2))
    output = Nx.dot(diff_attn, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output = Nx.transpose(output, axes: [0, 2, 1, 3])
    output = Nx.reshape(output, {batch, seq_len, num_heads * head_dim})

    # V2: Simple RMSNorm (no per-head GroupNorm)
    rms = Nx.sqrt(Nx.mean(Nx.multiply(output, output), axes: [-1], keep_axes: true))
    output = Nx.divide(output, Nx.add(rms, 1.0e-6))
    Nx.multiply(output, rms_scale)
  end

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
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
      num_layers: 6,
      dropout: 0.1
    ]
  end
end
