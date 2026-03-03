defmodule Edifice.RL.PointerNetwork do
  @moduledoc """
  Attention-based pointer network for entity selection.

  Given an agent's hidden state (query) and a set of entity representations
  (keys), produces a categorical distribution over entities. Used for
  selecting targets, units, or game objects from a variable-size set.

  ## Architecture

  ```
  Query: agent hidden  [batch, hidden_size]
  Keys:  entity set    [batch, num_entities, entity_dim]
        |
        v
  ┌─────────────────────────────────────────┐
  │  Project query → [batch, 1, key_dim]     │
  │  Project keys  → [batch, N, key_dim]     │
  │                                          │
  │  scores = query · keys^T / sqrt(key_dim) │
  │  logits = [batch, num_entities]           │
  └─────────────────────────────────────────┘
        |
        v
  Selection logits [batch, num_entities]
  (apply softmax externally for probabilities)
  ```

  Optionally applies a mask to exclude invalid entities (padding, dead units).

  ## Usage

      model = PointerNetwork.build(
        hidden_size: 256,
        entity_dim: 64,
        key_dim: 64
      )

      output = predict_fn.(params, %{
        "query" => agent_hidden,    # [batch, 256]
        "keys" => entity_reprs,     # [batch, num_entities, 64]
      })
      # output = [batch, num_entities] (logits)

  ## Game AI Context

  AlphaStar uses pointer networks to select which unit to target from
  thousands of candidates. For Melee:

  - **1v1**: Point at opponent, platforms, ledges, projectiles
  - **Doubles/FFA**: Select which opponent to target
  - **Items**: Choose which item to pick up or avoid

  Pair with `EntityEncoder` — its per-entity output feeds directly into
  this module's keys.

  ## References

  - Vinyals et al., "Pointer Networks" (NeurIPS 2015)
  - Vinyals et al., "Grandmaster level in StarCraft II" (AlphaStar, 2019)
  """

  @default_key_dim 64
  @default_temperature 1.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:entity_dim, pos_integer()}
          | {:key_dim, pos_integer()}
          | {:temperature, float()}
          | {:use_mask, boolean()}

  @doc """
  Build a pointer network.

  ## Options

    - `:hidden_size` - Agent hidden state dimension (required)
    - `:entity_dim` - Per-entity representation dimension (required)
    - `:key_dim` - Projection dimension for query-key matching (default: #{@default_key_dim})
    - `:temperature` - Softmax temperature for logit scaling (default: #{@default_temperature})
    - `:use_mask` - If true, expects a `"mask"` input to mask out invalid entities (default: false)

  ## Inputs

    - `"query"` — Agent hidden state `[batch, hidden_size]`
    - `"keys"` — Entity representations `[batch, num_entities, entity_dim]`
    - `"mask"` — (optional, if `use_mask: true`) Binary mask `[batch, num_entities]`,
      1 for valid entities, 0 for padding

  ## Returns

    An Axon model outputting selection logits `[batch, num_entities]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    entity_dim = Keyword.fetch!(opts, :entity_dim)
    key_dim = Keyword.get(opts, :key_dim, @default_key_dim)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    use_mask = Keyword.get(opts, :use_mask, false)

    # Inputs
    query = Axon.input("query", shape: {nil, hidden_size})
    keys = Axon.input("keys", shape: {nil, nil, entity_dim})

    # Project query and keys to shared dimension
    q_proj = Axon.dense(query, key_dim, name: "pointer_q_proj")
    k_proj = Axon.dense(keys, key_dim, name: "pointer_k_proj")

    # Compute attention scores
    if use_mask do
      mask = Axon.input("mask", shape: {nil, nil})

      Axon.layer(
        fn q, k, m, _opts ->
          # q: [batch, key_dim] → [batch, 1, key_dim]
          q_expanded = Nx.new_axis(q, 1)
          scale = :math.sqrt(key_dim)

          # [batch, 1, key_dim] · [batch, num_entities, key_dim]^T → [batch, 1, num_entities]
          scores = Nx.dot(q_expanded, [2], [0], k, [2], [0]) |> Nx.divide(scale)

          # Squeeze → [batch, num_entities]
          logits = Nx.squeeze(scores, axes: [1])

          # Apply temperature
          logits = Nx.divide(logits, temperature)

          # Mask: set invalid positions to large negative
          mask_penalty = Nx.multiply(Nx.subtract(1.0, m), -1.0e9)
          Nx.add(logits, mask_penalty)
        end,
        [q_proj, k_proj, mask],
        name: "pointer_scores"
      )
    else
      Axon.layer(
        fn q, k, _opts ->
          q_expanded = Nx.new_axis(q, 1)
          scale = :math.sqrt(key_dim)
          scores = Nx.dot(q_expanded, [2], [0], k, [2], [0]) |> Nx.divide(scale)
          logits = Nx.squeeze(scores, axes: [1])
          Nx.divide(logits, temperature)
        end,
        [q_proj, k_proj],
        name: "pointer_scores"
      )
    end
  end

  @doc """
  Return the output size (num_entities, which is dynamic).
  """
  @spec output_size(keyword()) :: :dynamic
  def output_size(_opts \\ []), do: :dynamic

  @doc """
  Recommended defaults for game AI entity targeting.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      entity_dim: 64,
      key_dim: 64,
      temperature: 1.0,
      use_mask: true
    ]
  end
end
