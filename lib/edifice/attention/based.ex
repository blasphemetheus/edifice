defmodule Edifice.Attention.Based do
  @moduledoc """
  Based: Linear attention with Taylor expansion feature map.

  Replaces the quadratic softmax(QK^T) attention with a linear approximation
  using Taylor-expanded feature maps. Instead of computing the full attention
  matrix, Based projects Q and K through a polynomial feature map phi(x) and
  computes attention in linear time.

  ## Key Innovation

  The Taylor feature map approximates softmax attention:
  - phi(x) = [1, x, x^2/sqrt(2!), ...] for Taylor order N
  - Linear attention: output = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))
  - This avoids the O(n^2) softmax(QK^T) computation

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +--------------------------------------+
  |   Based Block (x num_layers)         |
  |                                      |
  |   LayerNorm -> Based Linear Attn     |
  |     Q, K projections + Taylor phi()  |
  |     Linear attention via phi(Q/K)    |
  |   -> Residual                        |
  |   LayerNorm -> FFN -> Residual       |
  +--------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Mechanism | Time | Space |
  |-----------|------|-------|
  | Softmax attention | O(n^2 d) | O(n^2) |
  | Based (Taylor) | O(n d^2 p) | O(d^2 p) |

  Where p = Taylor order, typically 2-3.

  ## Usage

      model = Based.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        taylor_order: 2,
        num_layers: 4
      )

  ## References
  - "Simple linear attention language models balance the recall-throughput tradeoff"
    (Arora et al., 2024)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_taylor_order 2
  @default_num_layers 4
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:taylor_order, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Based linear attention model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:taylor_order` - Order of Taylor expansion for feature map (default: 2)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    taylor_order = Keyword.get(opts, :taylor_order, @default_taylor_order)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          name = "based_block_#{block_opts[:layer_idx]}"

          attn_fn = fn x, attn_name ->
            build_based_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              taylor_order: taylor_order,
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
  Build the Based linear attention layer with Taylor feature map.

  Projects to Q, K, V, applies Taylor feature map to Q and K,
  then computes linear attention.
  """
  @spec build_based_attention(Axon.t(), keyword()) :: Axon.t()
  def build_based_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    taylor_order = Keyword.get(opts, :taylor_order, @default_taylor_order)
    name = Keyword.get(opts, :name, "based_attn")

    head_dim = div(hidden_size, num_heads)

    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    output =
      Axon.layer(
        &based_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        taylor_order: taylor_order,
        op_name: :based_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Based linear attention implementation
  defp based_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    taylor_order = opts[:taylor_order]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Apply Taylor feature map to Q and K
    # phi(x) = concat([1, x, x^2/sqrt(2)]) for order=2
    phi_q = taylor_feature_map(q, taylor_order)
    phi_k = taylor_feature_map(k, taylor_order)

    # phi shapes: [batch, heads, seq, expanded_dim]
    # Linear attention: output = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum_phi_K)

    # KV: phi(K)^T @ V -> [batch, heads, expanded_dim, head_dim]
    # phi_k: [batch, heads, seq, expanded_dim], v: [batch, heads, seq, head_dim]
    # Contract on seq (axis 2), batch on [batch, heads] (axes 0, 1)
    kv = Nx.dot(phi_k, [2], [0, 1], v, [2], [0, 1])

    # numerator: phi(Q) @ KV -> [batch, heads, seq, head_dim]
    # phi_q is [batch, heads, seq, expanded_dim]
    # kv is [batch, heads, expanded_dim, head_dim]
    # Nx.dot(phi_q, [3], [0, 1], kv, [2], [0, 1]) => contract expanded_dim, batch on [batch, heads]
    # Result: [batch, heads, seq, head_dim]
    numerator = Nx.dot(phi_q, [3], [0, 1], kv, [2], [0, 1])

    # denominator: phi(Q) @ sum(phi(K), axis=seq)
    # sum_phi_k: [batch, heads, expanded_dim]
    sum_phi_k = Nx.sum(phi_k, axes: [2])

    # phi_q: [batch, heads, seq, expanded_dim]
    # sum_phi_k: [batch, heads, expanded_dim]
    # For each (batch, head, seq_pos): dot(phi_q[..., seq_pos, :], sum_phi_k[..., :])
    # = sum over expanded_dim
    denominator =
      Nx.dot(phi_q, [3], [0, 1], Nx.new_axis(sum_phi_k, 3), [2], [0, 1])
      |> Nx.squeeze(axes: [3])

    # denominator: [batch, heads, seq] - add small epsilon for stability
    denominator = Nx.add(denominator, 1.0e-6)

    # output: [batch, heads, seq, head_dim]
    output = Nx.divide(numerator, Nx.new_axis(denominator, -1))

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Taylor feature map: phi(x) = concat([1, x, x^2/sqrt(2!), ...])
  # Input: [batch, heads, seq, head_dim]
  # Output: [batch, heads, seq, expanded_dim] where expanded_dim = 1 + head_dim * taylor_order
  defp taylor_feature_map(x, taylor_order) do
    {batch, heads, seq_len, _head_dim} = Nx.shape(x)

    # Constant term: ones [batch, heads, seq, 1]
    ones = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(x)), {batch, heads, seq_len, 1})

    # Build polynomial terms
    terms =
      Enum.reduce(1..taylor_order, [ones], fn order, acc ->
        # x^order / sqrt(order!)
        power = Nx.pow(x, order)
        factorial = Enum.reduce(1..order, 1.0, fn i, acc_f -> acc_f * i end)
        scale = :math.sqrt(factorial)
        term = Nx.divide(power, scale)
        acc ++ [term]
      end)

    # Concatenate all terms: [batch, heads, seq, 1 + head_dim * order]
    Nx.concatenate(terms, axis: 3)
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  @doc """
  Get the output size of a Based model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
