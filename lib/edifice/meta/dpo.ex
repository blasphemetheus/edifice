defmodule Edifice.Meta.DPO do
  @moduledoc """
  DPO: Direct Preference Optimization.

  Implements DPO from "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model" (Rafailov et al., NeurIPS 2023). DPO eliminates
  the need for a separate reward model in RLHF by directly optimizing the
  policy from preference pairs.

  ## Key Innovation: Implicit Reward Model

  Standard RLHF requires:
  1. Train a reward model on preference data
  2. Use RL (PPO) to optimize policy against reward model

  DPO shows that for a given reward function r(x,y), the optimal policy is:

  ```
  pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
  ```

  Inverting this gives the implicit reward:

  ```
  r(y|x) = beta * log(pi*(y|x)/pi_ref(y|x)) + beta*log(Z(x))
  ```

  Substituting into the Bradley-Terry preference model and simplifying yields
  the DPO loss:

  ```
  L_DPO = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x))
                              - log(pi(y_l|x)/pi_ref(y_l|x)))))
  ```

  Where:
  - y_w is the preferred (winning) response
  - y_l is the dispreferred (losing) response
  - pi is the policy being trained
  - pi_ref is the frozen reference policy
  - beta controls KL regularization strength

  ## Architecture

  DPO wraps any sequence model (typically a decoder-only transformer):

  ```
  Input prompt x
        |
        +-------+-------+
        |               |
        v               v
  +------------+   +------------+
  | Policy     |   | Reference  |  (frozen copy)
  | pi(y|x)    |   | pi_ref(y|x)|
  +------------+   +------------+
        |               |
        v               v
  log_probs_pi    log_probs_ref
        |               |
        +-------+-------+
                |
                v
        DPO Loss Computation
  ```

  ## Usage

      # Build a DPO-wrapped policy
      policy = DPO.build(
        backbone: :decoder_only,
        hidden_size: 512,
        num_layers: 6,
        vocab_size: 32000
      )

      # Compute DPO loss
      loss = DPO.loss(
        policy_logprobs_chosen,
        policy_logprobs_rejected,
        ref_logprobs_chosen,
        ref_logprobs_rejected,
        beta: 0.1
      )

  ## Reference

  - Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  - Authors: Rafailov et al.
  - arXiv: https://arxiv.org/abs/2305.18290
  - NeurIPS 2023
  """

  @default_hidden_size 512
  @default_num_layers 6
  @default_num_heads 8
  @default_vocab_size 32_000
  @default_beta 0.1

  @doc """
  Build a DPO policy model.

  The model wraps a decoder-only backbone for language modeling.
  During training, you'll need to maintain a frozen copy as the reference.

  ## Options

    - `:hidden_size` - Hidden dimension (default: 512)
    - `:num_layers` - Number of transformer layers (default: 6)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:vocab_size` - Vocabulary size (default: 32000)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:max_seq_len` - Maximum sequence length (default: 2048)

  ## Returns

    An Axon model that outputs logits over the vocabulary.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:max_seq_len, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:vocab_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    dropout = Keyword.get(opts, :dropout, 0.1)
    max_seq_len = Keyword.get(opts, :max_seq_len, 2048)

    # Input: token indices [batch, seq_len]
    input = Axon.input("tokens", shape: {nil, nil})

    # Token embedding
    embedded = build_token_embedding(input, vocab_size, hidden_size, "token_embed")

    # Add position embeddings
    x = add_position_embedding(embedded, hidden_size, max_seq_len)

    # Transformer blocks (decoder-only with causal attention)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_decoder_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          dropout: dropout,
          name: "decoder_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Output logits
    Axon.dense(x, vocab_size, name: "lm_head")
  end

  @doc """
  Compute the DPO loss given policy and reference log probabilities.

  ## Parameters

    - `policy_logprobs_chosen` - Log probs from policy for chosen responses [batch]
    - `policy_logprobs_rejected` - Log probs from policy for rejected responses [batch]
    - `ref_logprobs_chosen` - Log probs from reference for chosen responses [batch]
    - `ref_logprobs_rejected` - Log probs from reference for rejected responses [batch]
    - `opts` - Options including `:beta` (default: 0.1)

  ## Returns

    Scalar loss value.

  ## Formula

  ```
  L = -log(sigmoid(beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))))
  ```
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def loss(
        policy_logprobs_chosen,
        policy_logprobs_rejected,
        ref_logprobs_chosen,
        ref_logprobs_rejected,
        opts \\ []
      ) do
    beta = Keyword.get(opts, :beta, @default_beta)

    # Compute log ratios
    # log(pi(y_w|x) / pi_ref(y_w|x)) = log_pi_w - log_ref_w
    log_ratio_chosen = Nx.subtract(policy_logprobs_chosen, ref_logprobs_chosen)

    # log(pi(y_l|x) / pi_ref(y_l|x)) = log_pi_l - log_ref_l
    log_ratio_rejected = Nx.subtract(policy_logprobs_rejected, ref_logprobs_rejected)

    # DPO objective: -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
    logits = Nx.multiply(beta, Nx.subtract(log_ratio_chosen, log_ratio_rejected))

    # Binary cross-entropy with logits (stable computation)
    # -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
    losses = Nx.log1p(Nx.exp(Nx.negate(logits)))

    # Mean over batch
    Nx.mean(losses)
  end

  @doc """
  Compute per-token log probabilities from logits.

  ## Parameters

    - `logits` - Model output logits [batch, seq_len, vocab_size]
    - `targets` - Target token indices [batch, seq_len]
    - `mask` - Optional mask for padding [batch, seq_len] (1 for valid, 0 for padding)

  ## Returns

    Per-sequence log probability sums [batch].
  """
  @spec compute_logprobs(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def compute_logprobs(logits, targets, mask \\ nil) do
    # Log softmax for numerical stability
    log_probs = log_softmax(logits)

    # Gather log probs for target tokens
    batch_size = Nx.axis_size(logits, 0)
    seq_len = Nx.axis_size(logits, 1)
    vocab_size = Nx.axis_size(logits, 2)

    # Create indices for gathering
    # We want log_probs[b, t, targets[b, t]] for each b, t
    target_log_probs = gather_logprobs(log_probs, targets, batch_size, seq_len, vocab_size)

    # Apply mask if provided
    target_log_probs =
      if mask do
        Nx.multiply(target_log_probs, mask)
      else
        target_log_probs
      end

    # Sum over sequence dimension
    Nx.sum(target_log_probs, axes: [1])
  end

  # ============================================================================
  # Model Building Helpers
  # ============================================================================

  defp build_token_embedding(input, vocab_size, hidden_size, name) do
    Axon.layer(
      &token_embedding_impl/2,
      [input],
      name: name,
      vocab_size: vocab_size,
      hidden_size: hidden_size,
      op_name: :token_embed
    )
  end

  defp token_embedding_impl(indices, opts) do
    vocab_size = opts[:vocab_size]
    hidden_size = opts[:hidden_size]

    # One-hot and project (simplified embedding)
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s64), -1),
        Nx.iota({1, 1, vocab_size})
      )
      |> Nx.as_type(:f32)

    # Simple learned projection (in practice this would use a learnable embedding table)
    # Here we use a deterministic projection for structure
    proj = Nx.iota({vocab_size, hidden_size}, type: :f32)
    proj = Nx.divide(proj, Nx.sqrt(Nx.tensor(vocab_size * hidden_size, type: :f32)))

    Nx.dot(one_hot, proj)
  end

  defp add_position_embedding(x, hidden_size, max_seq_len) do
    Axon.layer(
      &position_embedding_impl/2,
      [x],
      name: "pos_embed",
      hidden_size: hidden_size,
      max_seq_len: max_seq_len,
      op_name: :pos_embed
    )
  end

  defp position_embedding_impl(x, opts) do
    hidden_size = opts[:hidden_size]
    max_seq_len = opts[:max_seq_len]

    seq_len = Nx.axis_size(x, 1)
    actual_len = min(seq_len, max_seq_len)

    # Sinusoidal position embeddings
    positions = Nx.iota({actual_len}, type: :f32)
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    angles = Nx.outer(positions, freqs)
    pos_embed = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
    pos_embed = Nx.reshape(pos_embed, {1, actual_len, hidden_size})

    Nx.add(x, pos_embed)
  end

  defp build_decoder_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads)
    dropout = Keyword.get(opts, :dropout)
    name = Keyword.get(opts, :name)
    mlp_dim = hidden_size * 4

    # Pre-norm attention
    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = build_causal_attention(x_norm, hidden_size, num_heads, "#{name}_attn")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # Pre-norm MLP
    x_norm2 = Axon.layer_norm(x, name: "#{name}_mlp_norm")

    mlp_out =
      x_norm2
      |> Axon.dense(mlp_dim, name: "#{name}_mlp_up")
      |> Axon.activation(:gelu, name: "#{name}_mlp_act")
      |> Axon.dense(hidden_size, name: "#{name}_mlp_down")
      |> maybe_dropout(dropout, "#{name}_mlp_dropout")

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  defp build_causal_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :causal_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Causal mask
    mask = causal_mask(seq_len, Nx.type(scores))
    scores = Nx.add(scores, mask)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply attention
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp causal_mask(seq_len, type) do
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    mask = Nx.greater(cols, rows)
    Nx.select(mask, Nx.tensor(-1.0e9, type: type), Nx.tensor(0.0, type: type))
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Log Probability Helpers
  # ============================================================================

  defp log_softmax(logits) do
    max_logits = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logits)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end

  defp gather_logprobs(log_probs, targets, batch_size, seq_len, vocab_size) do
    # Flatten for gathering
    log_probs_flat = Nx.reshape(log_probs, {batch_size * seq_len, vocab_size})
    targets_flat = Nx.reshape(targets, {batch_size * seq_len})

    # One-hot selection
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(targets_flat, :s64), 1),
        Nx.iota({1, vocab_size})
      )
      |> Nx.as_type(:f32)

    # Gather: sum over vocab dimension to select the target log prob
    gathered = Nx.sum(Nx.multiply(log_probs_flat, one_hot), axes: [1])

    # Reshape back to [batch, seq]
    Nx.reshape(gathered, {batch_size, seq_len})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the default beta parameter for DPO.
  """
  @spec default_beta() :: float()
  def default_beta, do: @default_beta

  @doc """
  Get recommended defaults for DPO.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      num_heads: @default_num_heads,
      vocab_size: @default_vocab_size,
      beta: @default_beta
    ]
  end
end
