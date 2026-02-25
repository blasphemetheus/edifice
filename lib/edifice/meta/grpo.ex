defmodule Edifice.Meta.GRPO do
  @moduledoc """
  GRPO: Group Relative Policy Optimization.

  Implements GRPO from DeepSeek's RLHF methodology. GRPO simplifies RLHF by:
  1. Sampling G completions per prompt
  2. Ranking completions within each group
  3. Using group-relative advantages (no critic network needed)

  ## Key Innovation: Group-Relative Advantage

  Standard PPO requires a value function (critic) to compute advantages:
  ```
  A(s,a) = R - V(s)  # Advantage = Return - Value estimate
  ```

  GRPO eliminates the critic by using group-relative normalization:
  ```
  For each prompt, sample G responses with rewards r_1, ..., r_G
  Normalize: A_i = (r_i - mean(r)) / std(r)
  ```

  This works because:
  - Within a group, the mean reward is a natural baseline
  - Standard deviation normalizes the scale
  - No learned value function needed

  ## Algorithm

  ```
  For each prompt x:
      1. Sample G responses: y_1, ..., y_G ~ pi(y|x)
      2. Get rewards: r_1, ..., r_G = Reward(x, y_i)
      3. Compute advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
      4. Policy gradient: L = -sum(A_i * log pi(y_i|x))
  ```

  ## Architecture

  ```
  Prompt x [batch]
        |
        v
  +------------------+
  | Sample G times   |  -> G responses per prompt
  +------------------+
        |
        v
  +------------------+
  | Reward Model     |  -> Rewards r_1, ..., r_G
  +------------------+
        |
        v
  +------------------+
  | Group Normalize  |  -> Advantages A_1, ..., A_G
  +------------------+
        |
        v
  +------------------+
  | Policy Gradient  |  -> Update policy
  +------------------+
  ```

  ## Usage

      # Build a GRPO policy
      policy = GRPO.build(
        hidden_size: 512,
        num_layers: 6,
        vocab_size: 32000
      )

      # Compute group-relative advantages
      advantages = GRPO.compute_advantages(group_rewards)

      # Compute policy gradient loss
      loss = GRPO.loss(log_probs, advantages, mask)

  ## Reference

  - DeepSeek-R1 Technical Report (2024)
  - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
  """

  @default_hidden_size 512
  @default_num_layers 6
  @default_num_heads 8
  @default_vocab_size 32_000
  @default_group_size 8

  @doc """
  Build a GRPO policy model.

  Uses the same architecture as DPO (decoder-only transformer).

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

    # Transformer blocks
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
  Compute group-relative advantages from rewards.

  For each group of G responses to the same prompt, compute:
  ```
  A_i = (r_i - mean(r)) / (std(r) + eps)
  ```

  ## Parameters

    - `rewards` - Reward tensor [batch, group_size] or [batch * group_size]
    - `opts` - Options:
      - `:group_size` - Number of responses per prompt (default: inferred or 8)
      - `:eps` - Epsilon for numerical stability (default: 1e-8)

  ## Returns

    Advantages tensor with same shape as rewards.
  """
  @spec compute_advantages(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def compute_advantages(rewards, opts \\ []) do
    eps = Keyword.get(opts, :eps, 1.0e-8)

    case Nx.rank(rewards) do
      1 ->
        # Flat tensor: reshape to [batch, group_size]
        group_size = Keyword.get(opts, :group_size, @default_group_size)
        total = Nx.axis_size(rewards, 0)
        batch_size = div(total, group_size)

        rewards_2d = Nx.reshape(rewards, {batch_size, group_size})
        advantages_2d = compute_advantages_2d(rewards_2d, eps)
        Nx.reshape(advantages_2d, {total})

      2 ->
        # Already [batch, group_size]
        compute_advantages_2d(rewards, eps)

      _ ->
        raise ArgumentError, "rewards must be 1D or 2D tensor"
    end
  end

  defp compute_advantages_2d(rewards, eps) do
    # rewards: [batch, group_size]
    # Compute mean and std within each group (along axis 1)
    mean = Nx.mean(rewards, axes: [1], keep_axes: true)
    variance = Nx.mean(Nx.pow(Nx.subtract(rewards, mean), 2), axes: [1], keep_axes: true)
    std = Nx.sqrt(Nx.add(variance, eps))

    # Normalize
    Nx.divide(Nx.subtract(rewards, mean), std)
  end

  @doc """
  Compute the GRPO policy gradient loss.

  ## Parameters

    - `log_probs` - Per-sequence log probabilities [batch] or [batch, group_size]
    - `advantages` - Group-relative advantages [batch] or [batch, group_size]
    - `opts` - Options:
      - `:clip_range` - PPO-style clipping range (optional, default: nil for no clipping)
      - `:old_log_probs` - Old policy log probs for PPO clipping (required if clip_range set)

  ## Returns

    Scalar loss value.

  ## Formula

  Without clipping:
  ```
  L = -mean(A * log_pi)
  ```

  With clipping (PPO-style):
  ```
  ratio = exp(log_pi - old_log_pi)
  L = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
  ```
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def loss(log_probs, advantages, opts \\ []) do
    clip_range = Keyword.get(opts, :clip_range, nil)

    if clip_range do
      # PPO-style clipping
      old_log_probs = Keyword.fetch!(opts, :old_log_probs)
      ppo_loss(log_probs, old_log_probs, advantages, clip_range)
    else
      # Simple policy gradient
      simple_policy_gradient_loss(log_probs, advantages)
    end
  end

  defp simple_policy_gradient_loss(log_probs, advantages) do
    # L = -mean(A * log_pi)
    # Flatten if needed
    log_probs_flat = Nx.flatten(log_probs)
    advantages_flat = Nx.flatten(advantages)

    weighted = Nx.multiply(advantages_flat, log_probs_flat)
    Nx.negate(Nx.mean(weighted))
  end

  defp ppo_loss(log_probs, old_log_probs, advantages, clip_range) do
    # PPO clipped objective
    log_probs_flat = Nx.flatten(log_probs)
    old_log_probs_flat = Nx.flatten(old_log_probs)
    advantages_flat = Nx.flatten(advantages)

    # Importance ratio: pi_new / pi_old = exp(log_pi_new - log_pi_old)
    ratio = Nx.exp(Nx.subtract(log_probs_flat, old_log_probs_flat))

    # Clipped ratio
    clipped_ratio = Nx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Surrogate objectives
    surr1 = Nx.multiply(ratio, advantages_flat)
    surr2 = Nx.multiply(clipped_ratio, advantages_flat)

    # Take minimum (pessimistic bound)
    # For positive advantages: min(ratio*A, clip(ratio)*A)
    # For negative advantages: this becomes max in terms of loss
    surr =
      Nx.select(
        Nx.greater_equal(advantages_flat, 0.0),
        Nx.min(surr1, surr2),
        Nx.max(surr1, surr2)
      )

    Nx.negate(Nx.mean(surr))
  end

  @doc """
  Compute per-token log probabilities from logits.

  Same as DPO.compute_logprobs/3.

  ## Parameters

    - `logits` - Model output logits [batch, seq_len, vocab_size]
    - `targets` - Target token indices [batch, seq_len]
    - `mask` - Optional mask for padding [batch, seq_len]

  ## Returns

    Per-sequence log probability sums [batch].
  """
  @spec compute_logprobs(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def compute_logprobs(logits, targets, mask \\ nil) do
    # Log softmax
    log_probs = log_softmax(logits)

    batch_size = Nx.axis_size(logits, 0)
    seq_len = Nx.axis_size(logits, 1)
    vocab_size = Nx.axis_size(logits, 2)

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

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s64), -1),
        Nx.iota({1, 1, vocab_size})
      )
      |> Nx.as_type(:f32)

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

    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = build_causal_attention(x_norm, hidden_size, num_heads, "#{name}_attn")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

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

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    mask = causal_mask(seq_len, Nx.type(scores))
    scores = Nx.add(scores, mask)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

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
    log_probs_flat = Nx.reshape(log_probs, {batch_size * seq_len, vocab_size})
    targets_flat = Nx.reshape(targets, {batch_size * seq_len})

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(targets_flat, :s64), 1),
        Nx.iota({1, vocab_size})
      )
      |> Nx.as_type(:f32)

    gathered = Nx.sum(Nx.multiply(log_probs_flat, one_hot), axes: [1])
    Nx.reshape(gathered, {batch_size, seq_len})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the default group size for GRPO.
  """
  @spec default_group_size() :: pos_integer()
  def default_group_size, do: @default_group_size

  @doc """
  Get recommended defaults for GRPO.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      num_heads: @default_num_heads,
      vocab_size: @default_vocab_size,
      group_size: @default_group_size
    ]
  end
end
