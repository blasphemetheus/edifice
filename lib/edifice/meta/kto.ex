defmodule Edifice.Meta.KTO do
  @moduledoc """
  KTO: Kahneman-Tversky Optimization for RLHF from binary feedback.

  KTO aligns language models using only binary thumbs-up / thumbs-down
  labels — no preference pairs required. It is based on Kahneman-Tversky
  Prospect Theory, modelling how humans perceive gains (desirable outputs)
  and losses (undesirable outputs) asymmetrically.

  ## Key Innovation Over DPO

  DPO needs *paired* (chosen, rejected) responses to the same prompt.
  KTO only needs a single response labelled desirable (1) or undesirable
  (0), which is far easier to collect at scale.

  ## Loss Formula

  For each sample with policy log-probability `log π(y|x)` and
  reference `log π_ref(y|x)`:

  ```
  KL  = β · (log π(y|x) − log π_ref(y|x))
  z   = E_batch[KL]     # partition-function estimate

  # Prospect-theory value functions
  desirable:    v = σ(KL − z)   → utility of a good response
  undesirable:  v = σ(−KL − z)  → disutility of a bad response

  # Per-sample loss (minimise negative utility)
  L = −λ_D · σ(KL − z)       if label = 1 (desirable)
  L = −λ_U · σ(−KL − z)      if label = 0 (undesirable)
  ```

  `λ_D` and `λ_U` allow weighting desirable vs. undesirable feedback.

  ## Architecture

  KTO wraps any sequence model; `build/1` constructs a decoder-only
  transformer for language modelling (identical backbone to DPO).

  ```
  Prompt x
      |
      +--------+---------+
      |                  |
      v                  v
  +----------+     +----------+
  |  Policy  |     |  Ref     |  (frozen)
  | π(y|x)   |     | π_ref(y|x)|
  +----------+     +----------+
      |                  |
  log_pi             log_ref
      |                  |
      +--------+---------+
               |
               v
        KTO Loss + binary label
  ```

  ## Usage

      policy = KTO.build(vocab_size: 32000, hidden_size: 512, num_layers: 6)

      loss = KTO.kto_loss(
        policy_logprobs,
        ref_logprobs,
        labels,
        beta: 0.1,
        desirable_weight: 1.0,
        undesirable_weight: 1.0
      )

  ## References

  - Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2023)
  - https://arxiv.org/abs/2402.01306
  """

  @default_hidden_size 512
  @default_num_layers 6
  @default_num_heads 8
  @default_vocab_size 32_000
  @default_beta 0.1
  @default_desirable_weight 1.0
  @default_undesirable_weight 1.0

  @doc """
  Build a KTO policy model (decoder-only language model).

  ## Options

    - `:hidden_size` - Model hidden dimension (default: 512)
    - `:num_layers` - Number of transformer layers (default: 6)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:vocab_size` - Vocabulary size (default: 32000)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:max_seq_len` - Maximum sequence length for positional encoding (default: 2048)

  ## Returns

    An Axon model taking token indices `[batch, seq_len]` and returning
    logits `[batch, seq_len, vocab_size]`.
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

    input = Axon.input("tokens", shape: {nil, nil})

    x = token_embed(input, vocab_size, hidden_size)
    x = pos_embed(x, hidden_size, max_seq_len)

    x =
      Enum.reduce(1..num_layers, x, fn idx, acc ->
        decoder_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          dropout: dropout,
          name: "block_#{idx}"
        )
      end)

    x = Axon.layer_norm(x, name: "final_norm")
    Axon.dense(x, vocab_size, name: "lm_head")
  end

  @doc """
  Compute the KTO loss from binary preference labels.

  ## Parameters

    - `policy_logprobs` - Log-probs from the policy being trained `[batch]`
    - `ref_logprobs` - Log-probs from the frozen reference model `[batch]`
    - `labels` - Binary labels `[batch]`; 1 = desirable, 0 = undesirable

  ## Options

    - `:beta` - KL regularisation strength (default: 0.1)
    - `:desirable_weight` - Loss weight λ_D for desirable samples (default: 1.0)
    - `:undesirable_weight` - Loss weight λ_U for undesirable samples (default: 1.0)

  ## Returns

    Scalar loss value.

  ## Formula

  ```
  KL     = β · (log π − log π_ref)
  z_ref  = mean(KL)            # batch estimate of partition function
  reward = KL − z_ref

  loss_i = −λ_D · σ(reward_i)     if label_i = 1
  loss_i = −λ_U · σ(−reward_i)    if label_i = 0
  L = mean(loss_i)
  ```
  """
  @typedoc "Options for `kto_loss/4`."
  @type kto_loss_opt ::
          {:beta, float()}
          | {:desirable_weight, float()}
          | {:undesirable_weight, float()}

  @spec kto_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), [kto_loss_opt()]) ::
          Nx.Tensor.t()
  def kto_loss(policy_logprobs, ref_logprobs, labels, opts \\ []) do
    beta = Keyword.get(opts, :beta, @default_beta)
    lambda_d = Keyword.get(opts, :desirable_weight, @default_desirable_weight)
    lambda_u = Keyword.get(opts, :undesirable_weight, @default_undesirable_weight)

    # Scaled KL divergence: how much does the policy deviate from reference?
    # Positive KL → policy assigns more probability than reference (good for desirable)
    kl = Nx.multiply(beta, Nx.subtract(policy_logprobs, ref_logprobs))

    # Estimate the partition function z_ref from the current batch
    # This is the prospect-theory reference point against which gains/losses are measured
    z_ref = Nx.mean(kl)

    # Reward: KL relative to the batch baseline
    reward = Nx.subtract(kl, z_ref)

    labels_f = Nx.as_type(labels, :f32)

    # Prospect-theory value functions (sigmoid as bounded utility)
    # Desirable: σ(reward)  — positive when policy > reference
    # Undesirable: σ(−reward) — positive when policy < reference
    desirable_val = Nx.multiply(lambda_d, Nx.sigmoid(reward))
    undesirable_val = Nx.multiply(lambda_u, Nx.sigmoid(Nx.negate(reward)))

    # Select value function by label
    per_sample_utility =
      Nx.add(
        Nx.multiply(labels_f, desirable_val),
        Nx.multiply(Nx.subtract(1.0, labels_f), undesirable_val)
      )

    # Minimise negative expected utility
    Nx.negate(Nx.mean(per_sample_utility))
  end

  @doc """
  Compute per-sequence log-probabilities from logits and target tokens.

  ## Parameters

    - `logits` - Model output `[batch, seq_len, vocab_size]`
    - `targets` - Target token indices `[batch, seq_len]`
    - `mask` - Optional padding mask `[batch, seq_len]`; 1 = valid token

  ## Returns

    Per-sequence log-probability sums `[batch]`.
  """
  @spec compute_logprobs(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def compute_logprobs(logits, targets, mask \\ nil) do
    batch_size = Nx.axis_size(logits, 0)
    seq_len = Nx.axis_size(logits, 1)
    vocab_size = Nx.axis_size(logits, 2)

    log_probs = log_softmax(logits)
    lp_flat = Nx.reshape(log_probs, {batch_size * seq_len, vocab_size})
    tgt_flat = Nx.reshape(targets, {batch_size * seq_len})

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(tgt_flat, :s64), 1),
        Nx.iota({1, vocab_size})
      )
      |> Nx.as_type(:f32)

    target_lp = Nx.sum(Nx.multiply(lp_flat, one_hot), axes: [1])
    target_lp = Nx.reshape(target_lp, {batch_size, seq_len})

    target_lp =
      if mask do
        Nx.multiply(target_lp, Nx.as_type(mask, :f32))
      else
        target_lp
      end

    Nx.sum(target_lp, axes: [1])
  end

  @doc "Default beta (KL regularisation coefficient) for KTO."
  @spec default_beta() :: float()
  def default_beta, do: @default_beta

  # ============================================================================
  # Private: model-building helpers
  # ============================================================================

  defp token_embed(input, vocab_size, hidden_size) do
    Axon.layer(
      &token_embed_impl/2,
      [input],
      name: "token_embed",
      vocab_size: vocab_size,
      hidden_size: hidden_size,
      op_name: :kto_token_embed
    )
  end

  defp token_embed_impl(indices, opts) do
    vocab_size = opts[:vocab_size]
    hidden_size = opts[:hidden_size]

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s64), -1),
        Nx.iota({1, 1, vocab_size})
      )
      |> Nx.as_type(:f32)

    proj = Nx.divide(Nx.iota({vocab_size, hidden_size}, type: :f32), :math.sqrt(vocab_size))
    Nx.dot(one_hot, proj)
  end

  defp pos_embed(x, hidden_size, max_seq_len) do
    Axon.layer(
      &pos_embed_impl/2,
      [x],
      name: "pos_embed",
      hidden_size: hidden_size,
      max_seq_len: max_seq_len,
      op_name: :kto_pos_embed
    )
  end

  defp pos_embed_impl(x, opts) do
    hidden_size = opts[:hidden_size]
    max_seq_len = opts[:max_seq_len]

    seq_len = min(Nx.axis_size(x, 1), max_seq_len)
    half = div(hidden_size, 2)
    positions = Nx.iota({seq_len}, type: :f32)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half}, type: :f32), max(half - 1, 1))
        )
      )

    angles = Nx.outer(positions, freqs)
    pe = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
    pe = Nx.reshape(pe, {1, seq_len, hidden_size})
    Nx.add(x, pe)
  end

  defp decoder_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.fetch!(opts, :name)
    mlp_dim = hidden_size * 4

    # Pre-norm causal self-attention
    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn = causal_mha(x_norm, hidden_size, num_heads, "#{name}_attn")
    attn = maybe_dropout(attn, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn, name: "#{name}_attn_res")

    # Pre-norm FFN
    x_norm2 = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn =
      x_norm2
      |> Axon.dense(mlp_dim, name: "#{name}_ffn_up")
      |> Axon.activation(:gelu, name: "#{name}_ffn_act")
      |> Axon.dense(hidden_size, name: "#{name}_ffn_down")
      |> maybe_dropout(dropout, "#{name}_ffn_drop")

    Axon.add(x, ffn, name: "#{name}_ffn_res")
  end

  defp causal_mha(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    out =
      Axon.layer(
        &causal_attn_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :kto_causal_attn
      )

    Axon.dense(out, hidden_size, name: "#{name}_out")
  end

  defp causal_attn_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Causal mask: upper triangular set to -inf
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    causal = Nx.select(Nx.greater(cols, rows), Nx.tensor(-1.0e9), Nx.tensor(0.0))
    scores = Nx.add(scores, causal)

    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  defp log_softmax(logits) do
    max_l = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_l)
    log_sum = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum)
  end
end
