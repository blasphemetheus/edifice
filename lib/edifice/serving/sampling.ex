defmodule Edifice.Serving.Sampling do
  @moduledoc """
  Token sampling strategies for autoregressive generation.

  Provides temperature scaling, top-k filtering, top-p (nucleus) filtering,
  and categorical sampling — all in pure Nx for EXLA JIT compatibility.
  """

  import Nx.Defn

  @doc """
  Sample token IDs from logits using the specified strategy.

  ## Options

    - `:temperature` - Scaling factor (default: 1.0). Lower = more deterministic.
    - `:top_k` - Keep only top-k logits (default: 0 = disabled)
    - `:top_p` - Nucleus sampling threshold (default: 1.0 = disabled)
    - `:seed` - Random seed (default: 42)

  ## Parameters

    - `logits` - `[batch, vocab_size]` tensor of raw logits

  ## Returns

    `{token_ids, new_key}` where token_ids is `[batch]` and new_key is the updated PRNG key.
  """
  def sample(logits, key, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    top_k = Keyword.get(opts, :top_k, 0)
    top_p = Keyword.get(opts, :top_p, 1.0)

    logits = apply_temperature(logits, temperature)
    logits = if top_k > 0, do: apply_top_k(logits, top_k), else: logits
    logits = if top_p < 1.0, do: apply_top_p(logits, top_p), else: logits

    categorical_sample(logits, key)
  end

  @doc """
  Greedy decoding — return the argmax token ID.
  """
  defn greedy(logits) do
    Nx.argmax(logits, axis: -1)
  end

  @doc """
  Apply temperature scaling to logits.
  """
  defn apply_temperature(logits, temperature) do
    Nx.divide(logits, Nx.max(temperature, 1.0e-8))
  end

  @doc """
  Top-k filtering: mask all logits outside the top-k highest values.

  Uses a threshold approach: sort descending, find the k-th value,
  then mask everything below it in the original logits.
  """
  defn apply_top_k(logits, k) do
    vocab_size = Nx.axis_size(logits, -1)
    sorted = Nx.sort(logits, axis: -1, direction: :desc)

    # Positions 0..k-1 in sorted order are kept
    positions = Nx.iota({1, vocab_size}, axis: 1)
    keep_sorted = Nx.less(positions, k)

    # Find threshold: smallest kept value (use +inf for non-kept so min ignores them)
    large_pos = Nx.tensor(1.0e30, type: Nx.type(logits))
    threshold_vals = Nx.select(keep_sorted, sorted, large_pos)
    threshold = Nx.reduce_min(threshold_vals, axes: [-1], keep_axes: true)

    # Apply threshold to original unsorted logits
    neg_inf = Nx.Constants.neg_infinity(Nx.type(logits))
    mask = Nx.greater_equal(logits, threshold)
    Nx.select(mask, logits, neg_inf)
  end

  @doc """
  Top-p (nucleus) filtering: mask logits outside the smallest set
  whose cumulative probability exceeds `p`.

  Uses a threshold approach: find the cutoff logit value in sorted space,
  then apply it to the unsorted logits (avoids scatter/unsort).
  """
  defn apply_top_p(logits, p) do
    sorted_logits = Nx.sort(logits, axis: -1, direction: :desc)

    # Compute cumulative softmax probabilities
    probs = Nx.exp(sorted_logits - Nx.reduce_max(sorted_logits, axes: [-1], keep_axes: true))
    probs = probs / Nx.sum(probs, axes: [-1], keep_axes: true)
    cumulative_probs = Nx.cumulative_sum(probs, axis: -1)

    # Shift cumulative by 1 to include the boundary token
    shifted_cum =
      Nx.concatenate(
        [
          Nx.broadcast(0.0, {Nx.axis_size(logits, 0), 1}),
          cumulative_probs[[.., 0..-2//1]]
        ],
        axis: -1
      )

    sorted_mask = Nx.less_equal(shifted_cum, p)

    # Find threshold: minimum sorted logit value in the kept set.
    # For non-kept positions, substitute +inf so reduce_min ignores them.
    large_pos = Nx.tensor(1.0e30, type: Nx.type(logits))
    threshold_candidates = Nx.select(sorted_mask, sorted_logits, large_pos)
    threshold = Nx.reduce_min(threshold_candidates, axes: [-1], keep_axes: true)

    # Apply threshold to original unsorted logits
    neg_inf = Nx.Constants.neg_infinity(Nx.type(logits))
    final_mask = Nx.greater_equal(logits, threshold)
    Nx.select(final_mask, logits, neg_inf)
  end

  @doc """
  Categorical sampling from logits using Gumbel-max trick.

  Returns `{token_ids, new_key}`.
  """
  defn categorical_sample(logits, key) do
    {gumbel_noise, new_key} = Nx.Random.uniform(key, shape: Nx.shape(logits), type: Nx.type(logits))
    # Gumbel-max trick: argmax(logits - log(-log(U))) = sample from softmax(logits)
    gumbel = -Nx.log(-Nx.log(Nx.max(gumbel_noise, 1.0e-8)))
    token_ids = Nx.argmax(Nx.add(logits, gumbel), axis: -1)
    {token_ids, new_key}
  end

  @doc """
  Adaptive sampling with runtime-determined strategy.

  Unlike `sample/3` which decides top-k/top-p at compile time,
  this function uses `Nx.Defn.Kernel.if` to branch on tensor
  values at runtime. Useful when sampling parameters vary per
  step (e.g., adaptive temperature based on entropy).

  All parameters are tensors (not Elixir values), so branching
  happens on-device via XLA conditional.

  ## Parameters

    * `logits` - `[batch, vocab_size]` raw logits
    * `key` - PRNG key
    * `temperature` - Scalar tensor (> 0)
    * `top_k` - Scalar tensor (0 = disabled)

  ## Returns

    `{token_ids, new_key}`
  """
  defn sample_adaptive(logits, key, temperature, top_k) do
    # Temperature scaling (always applied)
    scaled = Nx.divide(Nx.as_type(logits, :f32), Nx.max(temperature, 1.0e-8))

    # Top-k filtering: only if top_k > 0 (runtime branch)
    filtered =
      if Nx.greater(top_k, 0) do
        apply_top_k(scaled, top_k)
      else
        scaled
      end

    categorical_sample(filtered, key)
  end

  @doc """
  Greedy or sampled decoding based on a runtime temperature tensor.

  Uses `Nx.Defn.Kernel.if` to branch on-device:
  - temperature <= 0.01: greedy (argmax)
  - temperature > 0.01: Gumbel-max sampling

  ## Parameters

    * `logits` - `[batch, vocab_size]` raw logits
    * `key` - PRNG key
    * `temperature` - Scalar tensor

  ## Returns

    `{token_ids, new_key}`
  """
  defn sample_greedy_or_stochastic(logits, key, temperature) do
    if Nx.less_equal(temperature, 0.01) do
      {Nx.argmax(logits, axis: -1), key}
    else
      scaled = Nx.divide(Nx.as_type(logits, :f32), temperature)
      categorical_sample(scaled, key)
    end
  end
end
