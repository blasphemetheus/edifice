defmodule Edifice.Attention.FoX do
  @moduledoc """
  FoX: Forgetting Transformer with learnable per-head forget gates.

  Augments standard softmax attention with a learned sigmoid forget gate
  that applies exponential decay to attention scores based on temporal distance:

  ```
  FoX(Q, K, V) = softmax(QK^T / sqrt(d) + F) * V
  ```

  Where `F[i,j] = sum_{t=j+1}^{i} log(sigmoid(f_t))` is the cumulative
  log-forget bias. Each head has a per-position forget gate `f_t` that
  controls how quickly past tokens are forgotten.

  ## Key Innovation

  The forget gate is additive on the attention logits (before softmax), not
  multiplicative on the weights. This means:

  1. During training, gates near 1.0 recover standard attention
  2. During inference, gates enable bounded KV cache (streaming)
  3. The gate can be fused into FlashAttention with minimal overhead

  This unifies the transformer/RNN divide: full attention during training,
  bounded memory during inference.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------------------+
  |  FoX Block (x num_layers)               |
  |                                          |
  |  Q, K, V projections                    |
  |  f_t = sigmoid(linear(x_t))  per head   |
  |  F[i,j] = cumsum(log(f_{j+1..i}))      |
  |  scores = QK^T / sqrt(d) + F           |
  |  weights = softmax(scores + causal_mask)|
  |  output = weights @ V                   |
  |  + Residual + FFN                       |
  +------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Usage

      model = FoX.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6
      )

  ## Reference

  - Pramod et al., "FoX: Forgetting Transformer" (ICLR 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

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
  Build a FoX (Forgetting Transformer) model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
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
          name = "fox_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_fox_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
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
  Build the FoX attention layer.

  Projects to Q, K, V plus per-head forget gates, computes cumulative
  forget bias, and applies standard softmax attention with the bias.
  """
  @spec build_fox_attention(Axon.t(), keyword()) :: Axon.t()
  def build_fox_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    name = Keyword.get(opts, :name, "fox_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Forget gate projection: [batch, seq, num_heads]
    # Each head gets its own per-position forget scalar
    forget_proj = Axon.dense(input, num_heads, name: "#{name}_forget_proj")

    output =
      Axon.layer(
        &fox_attention_impl/5,
        [q_proj, k_proj, v_proj, forget_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :fox_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # FoX attention implementation (dispatches through 3-tier CUDA pipeline)
  # Q, K, V: [batch, seq, hidden], forget_logits: [batch, seq, num_heads]
  defp fox_attention_impl(q, k, v, forget_logits, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Compute cumulative log-forget: cs = cumsum(log(sigmoid(f)))
    # forget_logits: [batch, seq, num_heads] -> [batch, heads, seq]
    forget_logits =
      forget_logits
      |> Nx.transpose(axes: [0, 2, 1])

    # log(sigmoid(x)) = x - softplus(x)
    log_forget = Nx.subtract(forget_logits, softplus(forget_logits))

    # Cumulative sum: cs[i] = sum_{t=0}^{i} log_forget[t]
    # cs: [batch, heads, seq]
    cs = Nx.cumulative_sum(log_forget, axis: -1)

    # Dispatch through fused FoX attention (handles forget bias + causal internally)
    attn_out = Edifice.CUDA.FusedScan.fox_attention(q, k, v, cs)

    # Reshape back: [batch, seq, hidden]
    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  # build_forget_bias/2, log_sum_exp/1 removed — now handled by FusedScan.fox_attention

  # Numerically stable softplus: log(1 + exp(x))
  defp softplus(x) do
    Nx.log1p(Nx.exp(x))
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
      dropout: 0.1
    ]
  end
end
