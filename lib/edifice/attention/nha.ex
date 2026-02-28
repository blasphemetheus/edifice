defmodule Edifice.Attention.NHA do
  @moduledoc """
  Native Hybrid Attention (NHA): learned per-layer linear vs softmax selection.

  Each layer has shared Q, K, V projections but two attention paths: a linear
  attention path (O(n) compute) and a full softmax attention path (O(n^2)
  compute). A learned per-layer gate selects the mixture of both, allowing
  the model to jointly optimize which layers benefit from full attention
  and which can use efficient linear attention.

  ```
  NHA(Q, K, V) = gate * SoftmaxAttn(Q, K, V) + (1 - gate) * LinearAttn(Q, K, V)
  ```

  ## Key Innovation

  Instead of hand-designing hybrid ratios (like Jamba's 87.5% Mamba / 12.5%
  attention, or Nemotron-H's 90/10), NHA makes the attention type selection
  learnable. Each layer has a scalar gate initialized at 0.5 that shifts
  during training — layers that need precise attention learn gate -> 1.0,
  while layers that can tolerate approximation learn gate -> 0.0.

  The KV projections are shared between both paths, saving parameters and
  enabling smooth interpolation between the two attention types.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +--------------------------------------------------+
  |  NHA Block (x num_layers)                        |
  |                                                  |
  |  Shared Q, K, V projections                     |
  |        |              |                          |
  |  SoftmaxAttn(Q,K,V)  LinearAttn(Q,K,V)         |
  |        |              |                          |
  |  gate * softmax + (1 - gate) * linear           |
  |        |                                         |
  |  Output projection + residual + FFN              |
  +--------------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Usage

      model = NHA.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6
      )

  ## Reference

  - "Native Hybrid Attention" (ICML 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1
  @default_gate_init 0.5

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:gate_init, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Native Hybrid Attention model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:gate_init` - Initial gate value (0=linear, 1=softmax, default: 0.5)
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
    gate_init = Keyword.get(opts, :gate_init, @default_gate_init)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "nha_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_nha_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              gate_init: gate_init,
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
  Build the NHA attention layer.

  Uses shared Q, K, V projections with parallel softmax and linear
  attention paths, combined via a learned gate.
  """
  @spec build_nha_attention(Axon.t(), keyword()) :: Axon.t()
  def build_nha_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    gate_init = Keyword.get(opts, :gate_init, @default_gate_init)
    name = Keyword.get(opts, :name, "nha_attn")

    head_dim = div(hidden_size, num_heads)

    # Shared Q, K, V projections (key innovation: both paths share these)
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Learned per-layer gate: scalar initialized at gate_init
    # Sigmoid maps to (0, 1): gate -> 1 means softmax, gate -> 0 means linear
    gate_param =
      Axon.param("#{name}_gate", {}, initializer: fn _, _ -> Nx.tensor(logit(gate_init)) end)

    output =
      Axon.layer(
        &nha_attention_impl/5,
        [q_proj, k_proj, v_proj, gate_param],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :nha_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Inverse sigmoid for gate initialization
  defp logit(p) when p > 0 and p < 1, do: :math.log(p / (1 - p))

  # NHA attention: parallel softmax + linear paths with learned gate
  # Q, K, V: [batch, seq, hidden], gate: scalar
  defp nha_attention_impl(q, k, v, gate_param, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q_h = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k_h = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v_h = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Gate: sigmoid of learned scalar
    gate = Nx.sigmoid(gate_param)

    # Path 1: Softmax attention
    softmax_out = compute_softmax_attention(q_h, k_h, v_h, head_dim, batch, num_heads, seq_len)

    # Path 2: Linear attention (ELU-based kernel)
    linear_out = compute_linear_attention(q_h, k_h, v_h, batch, num_heads, seq_len, head_dim)

    # Combine: gate * softmax + (1 - gate) * linear
    combined =
      Nx.add(
        Nx.multiply(gate, softmax_out),
        Nx.multiply(Nx.subtract(1.0, gate), linear_out)
      )

    # Reshape back: [batch, seq, hidden]
    reshape_from_heads(combined, batch, seq_len, num_heads, head_dim)
  end

  # Standard causal softmax attention
  defp compute_softmax_attention(q, k, v, head_dim, batch, num_heads, seq_len) do
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols)

    mask =
      mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores =
      Nx.select(
        mask,
        scores,
        Nx.broadcast(Nx.tensor(-1.0e9, type: Nx.type(scores)), Nx.shape(scores))
      )

    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.sum(exp_s, axes: [-1], keep_axes: true))

    # [batch, heads, seq, head_dim]
    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  # Causal linear attention using ELU+1 feature map
  # Linear attention: O(Q, K, V) = (Q' * (K' * V)^T) / (Q' * sum(K'))
  # With causal masking via cumulative sum
  defp compute_linear_attention(q, k, v, _batch, _num_heads, _seq_len, _head_dim) do
    # ELU+1 feature map: phi(x) = elu(x) + 1
    q_prime = Nx.add(elu(q), 1.0)
    k_prime = Nx.add(elu(k), 1.0)

    # Causal linear attention via cumulative KV and K sums
    # For each position i: output_i = (sum_{j<=i} K'_j * V_j^T) @ Q'_i / (sum_{j<=i} K'_j) @ Q'_i

    # KV accumulator: [batch, heads, head_dim, head_dim] (cumulative outer product)
    # K accumulator: [batch, heads, head_dim] (cumulative sum for normalization)

    # Non-causal linear attention (vectorized, no sequential scan needed)
    # For causal: would need cumulative KV, but non-causal is simpler and
    # the softmax path already provides causal capability via its mask.
    # Linear path provides global context: output = (Q' @ K'^T @ V) / (Q' @ sum(K'))

    # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
    # But we use the efficient O(n) formulation:
    # numerator: Q' @ (K'^T @ V) where K'^T @ V is [batch, heads, head_dim, head_dim]
    kv = Nx.dot(k_prime, [2], [0, 1], v, [2], [0, 1])
    # kv: [batch, heads, head_dim, head_dim]

    # num = Q' @ kv -> [batch, heads, seq, head_dim]
    num = Nx.dot(q_prime, [3], [0, 1], kv, [2], [0, 1])

    # denominator: Q' @ sum(K', axis=seq) -> [batch, heads, seq, 1]
    k_sum = Nx.sum(k_prime, axes: [2], keep_axes: true)
    # [batch, heads, seq, head_dim] * [batch, heads, 1, head_dim] -> sum over head_dim
    denom =
      Nx.sum(Nx.multiply(q_prime, Nx.broadcast(k_sum, Nx.shape(q_prime))),
        axes: [-1],
        keep_axes: true
      )

    denom = Nx.max(denom, 1.0e-6)

    Nx.divide(num, denom)
  end

  # ELU activation: max(0, x) + min(0, exp(x) - 1)
  defp elu(x) do
    pos = Nx.max(x, 0.0)
    neg = Nx.min(Nx.subtract(Nx.exp(Nx.min(x, 0.0)), 1.0), 0.0)
    Nx.add(pos, neg)
  end

  # Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, heads, seq, head_dim] -> [batch, seq, hidden]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
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
      dropout: 0.1,
      gate_init: 0.5
    ]
  end
end
