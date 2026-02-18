defmodule Edifice.Attention.Performer do
  @moduledoc """
  Performer: Fast Attention Via Positive Orthogonal Random Features (FAVOR+).

  Performer approximates softmax attention using random feature maps,
  achieving O(N) time and space complexity. The FAVOR+ mechanism uses
  orthogonal random features to approximate the exponential kernel.

  ## Key Innovation: FAVOR+ Random Feature Attention

  Standard attention: softmax(QK^T/sqrt(d)) * V  -- O(N^2)

  FAVOR+ approximates exp(QK^T) using random features:
  ```
  exp(q^T k) ~ phi(q)^T phi(k)

  Where phi(x) = exp(-||x||^2 / 2) / sqrt(m) * [exp(w_1^T x), ..., exp(w_m^T x)]
  w_1, ..., w_m ~ iid N(0, I) (orthogonalized)
  ```

  This allows rewriting attention as:
  ```
  Attn(Q,K,V) = D^{-1} * phi(Q) * (phi(K)^T * V)
  D = diag(phi(Q) * phi(K)^T * 1)
  ```

  Computing phi(K)^T * V is O(N*d*m) instead of O(N^2*d).

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |  Performer Block                     |
  |                                      |
  |  LayerNorm                           |
  |    -> Q, K, V projections            |
  |    -> Random feature map phi(Q,K)    |
  |       (orthogonal random features)   |
  |    -> KV = phi(K)^T * V   [d, d]    |
  |    -> out = phi(Q) * KV    [N, d]   |
  |    -> normalize by D                 |
  |  -> Residual                         |
  |                                      |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Component | Standard | Performer |
  |-----------|----------|-----------|
  | Time | O(N^2 * d) | O(N * d * m) |
  | Space | O(N^2 + N*d) | O(N * (d+m)) |
  | Random features | - | m (default 64) |

  Where m = num_features controls approximation quality vs speed tradeoff.

  ## Usage

      model = Performer.build(
        embed_dim: 287,
        hidden_size: 256,
        num_features: 64,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: "Rethinking Attention with Performers" (Choromanski et al., ICLR 2021)
  - FAVOR+: Fast Attention Via positive Orthogonal Random features
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_features 64
  @default_num_layers 4
  @default_num_heads 4
  @default_dropout 0.1

  @doc """
  Build a Performer model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_features` - Number of random features m for FAVOR+ (default: 64)
    - `:num_layers` - Number of Performer blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:num_features, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_features = Keyword.get(opts, :num_features, @default_num_features)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          name = "performer_block_#{block_opts[:layer_idx]}"

          attn_fn = fn x, attn_name ->
            build_favor_attention(x, hidden_size, num_heads, head_dim, num_features, attn_name)
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

  # Build the FAVOR+ attention sublayer (Q/K/V projections + attention + output proj)
  defp build_favor_attention(input, hidden_size, num_heads, head_dim, num_features, name) do
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Precompute orthogonal random features at build time via QR decomposition
    omega = generate_orthogonal_features(head_dim, num_features)
    omega_node = Axon.constant(omega)

    attn_out =
      Axon.layer(
        &favor_plus_attention_impl/5,
        [q_proj, k_proj, v_proj, omega_node],
        name: "#{name}_favor_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        num_features: num_features,
        op_name: :favor_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # FAVOR+ attention implementation
  # Uses random feature maps to approximate softmax attention in O(N*d*m)
  defp favor_plus_attention_impl(q, k, v, omega, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    num_features = opts[:num_features]
    eps = 1.0e-6

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scale Q, K by 1/sqrt(d) before feature map (Choromanski et al.)
    # This ensures the kernel approximation quality is dimension-independent
    d_scale = Nx.rsqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    q_scaled = Nx.multiply(q, d_scale)
    k_scaled = Nx.multiply(k, d_scale)

    # Apply FAVOR+ feature map with precomputed orthogonal random features
    # phi(x) = exp(x @ omega^T - ||x||^2/2) / sqrt(m)
    q_prime = favor_feature_map(q_scaled, omega, num_features)
    k_prime = favor_feature_map(k_scaled, omega, num_features)

    # Causal FAVOR+ attention using cumulative sums
    # K'^T * V outer products: [batch, heads, seq, num_features, head_dim]
    k_expanded = Nx.new_axis(k_prime, 4)
    v_expanded = Nx.new_axis(v, 3)
    kv = Nx.multiply(k_expanded, v_expanded)

    # Cumulative sums for causal attention
    kv_cumsum = Nx.cumulative_sum(kv, axis: 2)
    k_cumsum = Nx.cumulative_sum(k_prime, axis: 2)

    # Compute output: phi(Q) @ KV_cumsum
    q_expanded = Nx.new_axis(q_prime, 4)
    numerator = Nx.sum(Nx.multiply(q_expanded, kv_cumsum), axes: [3])

    # Normalizer: phi(Q) . sum(phi(K))
    denominator = Nx.sum(Nx.multiply(q_prime, k_cumsum), axes: [3], keep_axes: true)

    # Normalized output
    output = Nx.divide(numerator, Nx.add(denominator, eps))

    # Reshape back: [batch, seq, num_heads * head_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  @doc """
  Generate orthogonal random features for FAVOR+ via QR decomposition.

  Returns a `[num_features, head_dim]` matrix with orthogonal rows (within blocks
  of size head_dim). Multiple orthogonal blocks are concatenated if num_features > head_dim.
  """
  # Nx.LinAlg.qr/1 typespec says Nx.Tensor.t() but actually returns {q, r} tuple
  @dialyzer {:nowarn_function, generate_orthogonal_features: 3}
  @spec generate_orthogonal_features(pos_integer(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def generate_orthogonal_features(head_dim, num_features, opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    key = Nx.Random.key(seed)

    # Generate blocks of orthogonal features via QR decomposition
    num_full_blocks = div(num_features, head_dim)
    remainder = rem(num_features, head_dim)

    {blocks, key} =
      Enum.reduce(1..max(num_full_blocks, 0), {[], key}, fn _, {acc, k} ->
        {mat, k} = Nx.Random.normal(k, shape: {head_dim, head_dim}, type: :f32)
        {q, _r} = Nx.LinAlg.qr(mat)
        {[q | acc], k}
      end)

    blocks = Enum.reverse(blocks)

    blocks =
      if remainder > 0 do
        {mat, _} = Nx.Random.normal(key, shape: {head_dim, head_dim}, type: :f32)
        {q, _r} = Nx.LinAlg.qr(mat)
        partial = Nx.slice(q, [0, 0], [remainder, head_dim])
        blocks ++ [partial]
      else
        blocks
      end

    # Concatenate all blocks: [num_features, head_dim]
    Nx.concatenate(blocks, axis: 0)
  end

  # FAVOR+ feature map: phi(x) = exp(x @ omega^T - ||x||^2/2) / sqrt(m)
  # Returns non-negative random features for approximating softmax kernel
  defp favor_feature_map(x, omega, num_features) do
    # x: [batch, heads, seq, head_dim]
    # omega: [num_features, head_dim]

    # x @ omega^T: [batch, heads, seq, num_features]
    projection = Nx.dot(x, [3], omega, [1])

    # ||x||^2 / 2
    x_norm_sq = Nx.sum(Nx.multiply(x, x), axes: [3], keep_axes: true) |> Nx.divide(2.0)

    # phi(x) = exp(x @ omega^T - ||x||^2/2) / sqrt(m)
    # This ensures non-negative features and approximates exp(q^T k)
    scale = Nx.sqrt(Nx.tensor(num_features, type: Nx.type(x)))
    Nx.divide(Nx.exp(Nx.subtract(projection, x_norm_sq)), scale)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Performer model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Performer model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer:
    # Attention: Q + K + V + output projections (random features are not learnable)
    attn_params = hidden_size * hidden_size * 4

    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = attn_params + ffn_params

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
      num_features: 64,
      num_layers: 4,
      num_heads: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
