defmodule Edifice.Inference.Medusa do
  @moduledoc """
  Medusa: Multi-Head Speculative Decoding for 2-3x inference speedup.

  Medusa attaches K lightweight "draft heads" to an existing LM backbone.
  Each head is a small MLP that predicts one future token position from the
  last hidden state, sharing the base model's vocabulary (embedding table).
  At inference the K heads generate a tree of candidate continuations, which
  are verified in a single forward pass via "tree attention". Tokens are
  accepted greedily along the tree path that agrees with the base model's
  distribution.

  ## Motivation

  Standard speculative decoding (e.g. `SpeculativeDecoding`) requires a
  separate small draft model. Medusa eliminates this overhead by co-training
  K heads that are already aligned with the base model's hidden states.
  Because all K candidate tokens for all positions are verified in one
  batched forward pass, the wall-clock cost grows sub-linearly with sequence
  length.

  ## Architecture

  ```
  Base model (frozen or fine-tuned)
        |
  last hidden state [batch, hidden_dim]
        |
  +----------- K Medusa heads (parallel) -----------+
  | Head k: dense(hidden_dim) -> SiLU -> dense(vocab_size) |
  +--------------------------------------------------+
        |                |             |
  head_logits_1   head_logits_2  ...  head_logits_K
        |
  build_tree_candidates/2
        |
  tree of candidate token seqs (top-k per head, combined)
        |
  tree_decoding_mask/1  ← causal attention mask for tree positions
        |
  one base-model forward pass (tree attention)
        |
  accept/reject each candidate path
  ```

  ## Usage

      model = Medusa.build(
        base_hidden_dim: 256,
        vocab_size: 32_000,
        num_medusa_heads: 4
      )

      # At inference: generate candidates then verify
      {cands, tree} = Medusa.build_tree_candidates(head_logits, top_k: 5)
      mask = Medusa.tree_decoding_mask(tree)

  ## References

  - Cai et al., "Medusa: Simple Framework for Accelerating LLM Inference
    with Multiple Decoding Heads" (2024) — https://arxiv.org/abs/2401.10774
  """

  @default_num_medusa_heads 4
  @default_medusa_num_layers 1
  @default_top_k 5

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:base_hidden_dim, pos_integer()}
          | {:vocab_size, pos_integer()}
          | {:num_medusa_heads, pos_integer()}
          | {:medusa_num_layers, pos_integer()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build Medusa draft heads as an Axon container.

  Returns an `Axon.container` map with keys `:head_1` through `:head_K`,
  each shaped `[batch, vocab_size]`.  The input is the last hidden state
  from the base model.

  ## Options

    - `:base_hidden_dim` - Hidden dimension of the base model (required)
    - `:vocab_size` - Vocabulary size, shared with base model (required)
    - `:num_medusa_heads` - Number of speculative heads K (default: 4)
    - `:medusa_num_layers` - Dense layers per head, 1 = simple linear (default: 1)

  ## Returns

    An `Axon.container` map `%{head_1: ..., ..., head_K: ...}`, each
    `[batch, vocab_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    base_hidden_dim = Keyword.fetch!(opts, :base_hidden_dim)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    num_medusa_heads = Keyword.get(opts, :num_medusa_heads, @default_num_medusa_heads)
    medusa_num_layers = Keyword.get(opts, :medusa_num_layers, @default_medusa_num_layers)

    # Input: last hidden state from the base model [batch, base_hidden_dim]
    hidden = Axon.input("hidden_states", shape: {nil, base_hidden_dim})

    heads =
      for k <- 1..num_medusa_heads, into: %{} do
        head_logits = build_medusa_head(hidden, base_hidden_dim, vocab_size, medusa_num_layers, k)
        {String.to_atom("head_#{k}"), head_logits}
      end

    Axon.container(heads)
  end

  @doc """
  Build K Medusa head logits from hidden states.

  A helper that runs all K heads on `hidden_states` and returns a list of
  logit tensors (each `[batch, vocab_size]`).  Useful when composing Medusa
  heads into a larger model graph.

  ## Parameters

    - `hidden_states` - Axon node `[batch, hidden_dim]`
    - `opts` - Same as `build/1`

  ## Returns

    `Axon.container` map with keys `:head_1`..`:head_K`.
  """
  @spec medusa_heads(Axon.t(), keyword()) :: Axon.t()
  def medusa_heads(hidden_states, opts) do
    base_hidden_dim = Keyword.fetch!(opts, :base_hidden_dim)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    num_medusa_heads = Keyword.get(opts, :num_medusa_heads, @default_num_medusa_heads)
    medusa_num_layers = Keyword.get(opts, :medusa_num_layers, @default_medusa_num_layers)

    heads =
      for k <- 1..num_medusa_heads, into: %{} do
        head_logits =
          build_medusa_head(hidden_states, base_hidden_dim, vocab_size, medusa_num_layers, k)

        {String.to_atom("head_#{k}"), head_logits}
      end

    Axon.container(heads)
  end

  # Build one Medusa head: (dense -> SiLU)^{num_layers-1} -> dense
  defp build_medusa_head(input, hidden_dim, vocab_size, num_layers, k) do
    # If num_layers == 1, just a single linear projection (original Medusa paper)
    # If num_layers > 1, add intermediate dense+SiLU layers
    trunk =
      if num_layers > 1 do
        Enum.reduce(1..(num_layers - 1), input, fn layer_idx, acc ->
          acc
          |> Axon.dense(hidden_dim, name: "medusa_head_#{k}_hidden_#{layer_idx}")
          |> Axon.activation(:silu, name: "medusa_head_#{k}_act_#{layer_idx}")
        end)
      else
        input
      end

    # Final projection to vocab
    Axon.dense(trunk, vocab_size, name: "medusa_head_#{k}_logits")
  end

  # ============================================================================
  # Inference Utilities (pure Nx, not Axon graph)
  # ============================================================================

  @doc """
  Generate a tree of candidate token continuations from K head logits.

  Each head independently picks its top-`top_k` tokens. The candidates are
  combined into a flat list of token sequences `[1..K]` for tree verification.
  In the full Medusa algorithm these form a Cartesian-product tree; here we
  return:

  - `candidates` — tensor `[num_candidates, K]` of token ID sequences, where
    each row is one path of length K through the tree.
  - `tree_indices` — 1-D integer tensor of length `num_candidates` giving each
    candidate's index into the flattened token tree (used to build the mask).

  The simplest tree structure takes the top-k tokens from head 1, top-k from
  head 2, ..., and enumerates all `top_k^K` combinations (clamped to a
  maximum of `top_k * K` candidates for efficiency).

  ## Parameters

    - `head_logits` - Map `%{head_1: tensor, ..., head_K: tensor}`, each
      `[batch, vocab_size]`. Only the first batch element is used.

  ## Options

    - `:top_k` - Number of top tokens to consider per head (default: 5)

  ## Returns

    `{candidates, tree_indices}` where `candidates` is `[num_cands, K]` and
    `tree_indices` is `[num_cands]`.
  """
  @spec build_tree_candidates(%{atom() => Nx.Tensor.t()}, keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def build_tree_candidates(head_logits, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    num_heads = map_size(head_logits)

    # Gather top-k token IDs for each head from the first batch element
    top_tokens_per_head =
      for k <- 1..num_heads do
        logits_k = head_logits[String.to_atom("head_#{k}")]
        # Slice first batch element: [vocab_size]
        logits_1 = Nx.squeeze(Nx.slice_along_axis(logits_k, 0, 1, axis: 0))
        top_k_ids(logits_1, top_k)
      end

    # Build all combinations: take top-k from head 1 cross top-k from head 2...
    # For efficiency, enumerate only direct product (no pruning needed for small K)
    combinations = cartesian_combinations(top_tokens_per_head, top_k)

    num_cands = Nx.axis_size(combinations, 0)
    tree_indices = Nx.iota({num_cands})

    {combinations, tree_indices}
  end

  @doc """
  Build a tree attention mask for verifying all candidates in one pass.

  Given tree candidates `[num_cands, K]`, each candidate is a length-K path.
  The mask ensures that when the base model processes all K token positions
  for all `num_cands` candidates, each position can only attend to its
  ancestors in the tree (causal within each path, no cross-path attention).

  Returns a boolean mask `[num_cands * K, num_cands * K]` where entry
  `[i, j]` is `true` iff position `j` is an ancestor of position `i`
  (including self).

  ## Parameters

    - `tree_candidates` - Token candidate tensor `[num_cands, K]`

  ## Returns

    Boolean mask `[num_cands * K, num_cands * K]`.
  """
  @spec tree_decoding_mask(Nx.Tensor.t()) :: Nx.Tensor.t()
  def tree_decoding_mask(tree_candidates) do
    {num_cands, k} = Nx.shape(tree_candidates)
    total_positions = num_cands * k

    # Build position indices: position p belongs to candidate c = p div K, depth d = p mod K
    # A position at (c, d) can attend to (c, 0), (c, 1), ..., (c, d) — its own path prefix
    # We encode: for positions i and j, mask[i,j] = true iff
    #   same_candidate(i, j) AND depth(j) <= depth(i)
    i_idx = Nx.iota({total_positions, 1})
    j_idx = Nx.iota({1, total_positions})

    i_cand = Nx.quotient(i_idx, k)
    j_cand = Nx.quotient(j_idx, k)
    i_depth = Nx.remainder(i_idx, k)
    j_depth = Nx.remainder(j_idx, k)

    same_candidate = Nx.equal(i_cand, j_cand)
    causal = Nx.less_equal(j_depth, i_depth)

    Nx.logical_and(same_candidate, causal)
  end

  @doc "Get output size (vocab_size passed through heads)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts) do
    Keyword.fetch!(opts, :vocab_size)
  end

  # ============================================================================
  # Private helpers
  # ============================================================================

  # Return top-k token indices for a 1-D logits tensor.
  defp top_k_ids(logits, k) do
    vocab_size = Nx.axis_size(logits, 0)
    actual_k = min(k, vocab_size)
    # Argsort descending and take first k
    sorted_indices = Nx.argsort(logits, direction: :desc)
    Nx.slice_along_axis(sorted_indices, 0, actual_k, axis: 0)
  end

  # Build Cartesian product of top-k token lists across heads.
  # Returns [num_cands, num_heads] where num_cands = top_k^num_heads (clamped).
  defp cartesian_combinations(top_tokens_list, top_k) do
    num_heads = length(top_tokens_list)

    # Enumerate all combinations as Elixir lists of indices, then stack
    index_combinations = cartesian_indices(top_k, num_heads)

    # For each combination, look up the actual token IDs
    rows =
      Enum.map(index_combinations, fn indices ->
        Enum.zip(top_tokens_list, indices)
        |> Enum.map(fn {tokens, idx} ->
          # Slice returns shape {1}, squeeze to scalar before converting
          tokens
          |> Nx.slice_along_axis(idx, 1, axis: 0)
          |> Nx.squeeze()
          |> Nx.to_number()
        end)
      end)

    rows
    |> Enum.map(&Nx.tensor(&1, type: :s64))
    |> Nx.stack()
  end

  # Generate all combinations of indices [0..top_k-1] repeated num_heads times.
  # Capped at top_k * 8 to avoid exponential explosion for large K.
  defp cartesian_indices(top_k, num_heads) do
    max_cands = top_k * max(num_heads, 8)
    all = do_cartesian(top_k, num_heads)
    Enum.take(all, max_cands)
  end

  defp do_cartesian(_top_k, 0), do: [[]]

  defp do_cartesian(top_k, num_heads) do
    rest = do_cartesian(top_k, num_heads - 1)

    for i <- 0..(top_k - 1), suffix <- rest do
      [i | suffix]
    end
  end
end
