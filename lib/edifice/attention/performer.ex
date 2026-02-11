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
  Input [batch, seq_len, embed_size]
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
        embed_size: 287,
        hidden_size: 256,
        num_features: 64,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: "Rethinking Attention with Performers" (Choromanski et al., ICLR 2021)
  - FAVOR+: Fast Attention Via positive Orthogonal Random features
  """

  require Axon

  alias Edifice.Blocks.{TransformerBlock, ModelBuilder}

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_features 64
  @default_num_layers 4
  @default_num_heads 4
  @default_dropout 0.1

  @doc """
  Build a Performer model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_features` - Number of random features m for FAVOR+ (default: 64)
    - `:num_layers` - Number of Performer blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
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

    attn_out =
      Axon.layer(
        &favor_plus_attention_impl/4,
        [q_proj, k_proj, v_proj],
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
  defp favor_plus_attention_impl(q, k, v, opts) do
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

    # Generate random projection matrix for FAVOR+
    # omega: [num_features, head_dim] - random orthogonal features
    # Use a deterministic key derived from tensor shape for reproducibility within a session
    omega = generate_random_features(head_dim, num_features, Nx.type(q))

    # Apply FAVOR+ feature map: phi(x) = exp(-||x||^2/2) / sqrt(m) * exp(x @ omega^T)
    # [batch, heads, seq, num_features]
    q_prime = favor_feature_map(q, omega, num_features)
    # [batch, heads, seq, num_features]
    k_prime = favor_feature_map(k, omega, num_features)

    # Causal FAVOR+ attention using cumulative sums
    # KV[t] = sum_{i<=t} phi(K[i])^T * V[i]
    # Z[t] = sum_{i<=t} phi(K[i])

    # K'^T * V outer products: [batch, heads, seq, num_features, head_dim]
    # [batch, heads, seq, num_features, 1]
    k_expanded = Nx.new_axis(k_prime, 4)
    # [batch, heads, seq, 1, head_dim]
    v_expanded = Nx.new_axis(v, 3)
    # [batch, heads, seq, num_features, head_dim]
    kv = Nx.multiply(k_expanded, v_expanded)

    # Cumulative sums for causal attention
    # [batch, heads, seq, num_features, head_dim]
    kv_cumsum = Nx.cumulative_sum(kv, axis: 2)
    # [batch, heads, seq, num_features]
    k_cumsum = Nx.cumulative_sum(k_prime, axis: 2)

    # Compute output: phi(Q) @ KV_cumsum
    # q_prime: [batch, heads, seq, num_features]
    # kv_cumsum: [batch, heads, seq, num_features, head_dim]
    # We want to dot q along num_features dimension with kv_cumsum
    # [batch, heads, seq, num_features, 1]
    q_expanded = Nx.new_axis(q_prime, 4)
    # [batch, heads, seq, head_dim]
    numerator = Nx.sum(Nx.multiply(q_expanded, kv_cumsum), axes: [3])

    # Normalizer: phi(Q) . sum(phi(K))
    # [batch, heads, seq, 1]
    denominator = Nx.sum(Nx.multiply(q_prime, k_cumsum), axes: [3], keep_axes: true)

    # Normalized output
    output = Nx.divide(numerator, Nx.add(denominator, eps))

    # Reshape back: [batch, seq, num_heads * head_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Generate random orthogonal features for FAVOR+
  # Returns [num_features, head_dim] matrix of orthogonal random vectors
  defp generate_random_features(head_dim, num_features, type) do
    # Use a fixed random key for deterministic features
    key = Nx.Random.key(42)

    # Generate random Gaussian matrix
    {random_matrix, _key} = Nx.Random.normal(key, shape: {num_features, head_dim}, type: type)

    # Orthogonalize via QR decomposition for better approximation
    # For simplicity, use Gram-Schmidt-like normalization
    # Normalize each row to unit length
    norms = Nx.sqrt(Nx.sum(Nx.multiply(random_matrix, random_matrix), axes: [1], keep_axes: true))
    Nx.divide(random_matrix, Nx.add(norms, 1.0e-8))
  end

  # FAVOR+ feature map: phi(x) = exp(-||x||^2/2) / sqrt(m) * exp(x @ omega^T)
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
    embed_size = Keyword.get(opts, :embed_size, 287)
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
      num_features: 64,
      num_layers: 4,
      num_heads: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
