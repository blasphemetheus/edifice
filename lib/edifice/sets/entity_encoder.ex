defmodule Edifice.Sets.EntityEncoder do
  @moduledoc """
  Type-conditioned entity encoder for heterogeneous game entities.

  Encodes a set of entities (players, opponents, projectiles, items, etc.)
  where each entity has a type ID and a feature vector. Each entity type gets
  a learned embedding that conditions the feature projection. Self-attention
  over entities enables relational reasoning, and the output includes both a
  global summary vector and per-entity representations.

  ## Architecture

  ```
  Entity Features [batch, num_entities, entity_dim]
  Entity Types    [batch, num_entities]  (integer type IDs)
        |
        v
  ┌─────────────────────────────────────────┐
  │  Type Embedding [num_types, type_embed]  │
  │  + Feature Projection                    │
  │  → [batch, num_entities, hidden]         │
  └─────────────────────────────────────────┘
        |
        v
  ┌─────────────────────────────────────────┐
  │  N layers of Self-Attention + FFN        │
  │  (entities attend to each other)         │
  └─────────────────────────────────────────┘
        |
        ├──→ Pool (mean/attention) → global [batch, hidden]
        │
        └──→ Per-entity [batch, num_entities, hidden]
  ```

  ## Usage

      model = EntityEncoder.build(
        entity_dim: 16,
        num_types: 5,
        hidden_size: 64,
        num_heads: 4,
        num_layers: 2,
        pool_mode: :mean
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(templates, Axon.ModelState.empty())

      output = predict_fn.(params, %{
        "entity_features" => features,  # [batch, num_entities, entity_dim]
        "entity_types" => types         # [batch, num_entities] integer
      })

      # output = %{global: [batch, hidden], entities: [batch, num_entities, hidden]}

  ## Game AI Context

  Used in AlphaStar (StarCraft II) and OpenAI Five (Dota 2) for encoding
  heterogeneous game objects. Each unit type gets a separate embedding so the
  model can distinguish "self character" from "opponent" from "projectile"
  even when they share the same feature space.

  Pair with `PointerNetwork` for entity-level target selection.

  ## References

  - Vinyals et al., "Grandmaster level in StarCraft II using multi-agent RL" (2019)
  - Berner et al., "Dota 2 with Large Scale Deep RL" (OpenAI Five, 2019)
  - Lee et al., "Set Transformer" (2019)
  """

  @default_hidden_size 64
  @default_num_types 8
  @default_type_embed_dim 16
  @default_num_heads 4
  @default_num_layers 2
  @default_dropout 0.0
  @default_pool_mode :mean

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:entity_dim, pos_integer()}
          | {:num_types, pos_integer()}
          | {:type_embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:pool_mode, :mean | :max | :attention}

  @doc """
  Build an entity encoder model.

  ## Options

    - `:entity_dim` - Dimension of each entity's feature vector (required)
    - `:num_types` - Number of distinct entity types (default: #{@default_num_types})
    - `:type_embed_dim` - Dimension of type embeddings (default: #{@default_type_embed_dim})
    - `:hidden_size` - Hidden dimension after projection (default: #{@default_hidden_size})
    - `:num_heads` - Self-attention heads (default: #{@default_num_heads})
    - `:num_layers` - Self-attention + FFN layers (default: #{@default_num_layers})
    - `:dropout` - Dropout rate (default: #{@default_dropout})
    - `:pool_mode` - Global pooling: `:mean`, `:max`, or `:attention` (default: `:mean`)

  ## Returns

    An Axon model outputting `%{global: [batch, hidden], entities: [batch, num_entities, hidden]}`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    entity_dim = Keyword.fetch!(opts, :entity_dim)
    num_types = Keyword.get(opts, :num_types, @default_num_types)
    type_embed_dim = Keyword.get(opts, :type_embed_dim, @default_type_embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool_mode = Keyword.get(opts, :pool_mode, @default_pool_mode)

    # Inputs
    # [batch, num_entities, entity_dim]
    features = Axon.input("entity_features", shape: {nil, nil, entity_dim})
    # [batch, num_entities] — integer type IDs
    types = Axon.input("entity_types", shape: {nil, nil})

    # Type embedding lookup: [batch, num_entities] → [batch, num_entities, type_embed_dim]
    type_embeds = Axon.embedding(types, num_types, type_embed_dim, name: "type_embedding")

    # Concatenate features + type embedding, then project to hidden_size
    combined = Axon.concatenate([features, type_embeds], axis: 2, name: "entity_concat")

    projected =
      Axon.dense(combined, hidden_size, name: "entity_projection")
      |> Axon.activation(:gelu)

    # Self-attention layers over entities
    entity_repr =
      Enum.reduce(0..(num_layers - 1), projected, fn i, acc ->
        build_entity_attention_block(acc, hidden_size, num_heads, dropout, i)
      end)

    entity_repr = Axon.layer_norm(entity_repr, name: "entity_final_norm")

    # Global pooling
    global = build_pool(entity_repr, hidden_size, pool_mode)

    # Output both global and per-entity representations
    Axon.container(%{global: global, entities: entity_repr})
  end

  @doc """
  Return the output size of the global representation.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Recommended defaults for game AI entity encoding.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      entity_dim: 16,
      num_types: 8,
      type_embed_dim: 16,
      hidden_size: 64,
      num_heads: 4,
      num_layers: 2,
      dropout: 0.1,
      pool_mode: :mean
    ]
  end

  # ============================================================================
  # Private
  # ============================================================================

  # Self-attention + FFN block over entities (pre-norm style)
  defp build_entity_attention_block(input, hidden_size, num_heads, dropout, layer_idx) do
    name = "entity_attn_#{layer_idx}"
    head_dim = div(hidden_size, num_heads)

    # Pre-norm self-attention
    normed = Axon.layer_norm(input, name: "#{name}_ln1")

    q = Axon.dense(normed, hidden_size, name: "#{name}_q")
    k = Axon.dense(normed, hidden_size, name: "#{name}_k")
    v = Axon.dense(normed, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        fn q_val, k_val, v_val, _opts ->
          {b, n, _} = Nx.shape(q_val)
          scale = Nx.tensor(:math.sqrt(head_dim), type: Nx.type(q_val))

          # Reshape to [batch, num_entities, num_heads, head_dim]
          q_h = Nx.reshape(q_val, {b, n, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
          k_h = Nx.reshape(k_val, {b, n, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
          v_h = Nx.reshape(v_val, {b, n, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

          # [batch, heads, n, n]
          scores = Nx.dot(q_h, [3], [0, 1], k_h, [3], [0, 1]) |> Nx.divide(scale)
          weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [3], keep_axes: true)))
          weights = Nx.divide(weights, Nx.sum(weights, axes: [3], keep_axes: true))

          # [batch, heads, n, head_dim]
          out = Nx.dot(weights, [3], [0, 1], v_h, [2], [0, 1])
          Nx.transpose(out, axes: [0, 2, 1, 3]) |> Nx.reshape({b, n, hidden_size})
        end,
        [q, k, v],
        name: "#{name}_sdpa"
      )

    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
    attn_out = Axon.dropout(attn_out, rate: dropout, name: "#{name}_attn_drop")

    # Residual
    after_attn = Axon.add(input, attn_out, name: "#{name}_res1")

    # Pre-norm FFN
    normed2 = Axon.layer_norm(after_attn, name: "#{name}_ln2")

    ffn =
      normed2
      |> Axon.dense(hidden_size * 4, name: "#{name}_ffn1")
      |> Axon.activation(:gelu)
      |> Axon.dropout(rate: dropout, name: "#{name}_ffn_drop")
      |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    Axon.add(after_attn, ffn, name: "#{name}_res2")
  end

  defp build_pool(entity_repr, _hidden_size, :mean) do
    Axon.nx(entity_repr, fn x -> Nx.mean(x, axes: [1]) end, name: "entity_mean_pool")
  end

  defp build_pool(entity_repr, _hidden_size, :max) do
    Axon.nx(entity_repr, fn x -> Nx.reduce_max(x, axes: [1]) end, name: "entity_max_pool")
  end

  defp build_pool(entity_repr, _hidden_size, :attention) do
    # Learned attention pooling: a query vector attends over entities
    # Score each entity, weighted sum
    scores = Axon.dense(entity_repr, 1, name: "attn_pool_score")

    Axon.layer(
      fn repr, score, _opts ->
        weights = Nx.exp(Nx.subtract(score, Nx.reduce_max(score, axes: [1], keep_axes: true)))
        weights = Nx.divide(weights, Nx.sum(weights, axes: [1], keep_axes: true))
        # [batch, num_entities, hidden] * [batch, num_entities, 1] → sum → [batch, hidden]
        Nx.sum(Nx.multiply(repr, weights), axes: [1])
      end,
      [entity_repr, scores],
      name: "entity_attn_pool"
    )
  end
end
