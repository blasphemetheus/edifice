defmodule Edifice.Memory.MemoryNetwork do
  @moduledoc """
  End-to-End Memory Networks (Sukhbaatar et al., 2015).

  Memory Networks perform iterative reasoning over a set of memory slots
  by repeatedly attending to (reading from) memory and updating an internal
  query state. Each "hop" refines the query, enabling multi-step inference.

  ## How It Works

  Given a query q and memories M:
  1. Compute attention: p = softmax(q^T * A * M)
  2. Read memory: o = sum(p_i * C * m_i)
  3. Update query: q' = q + o
  4. Repeat for K hops

  Different embedding matrices A (for attention) and C (for output) at each
  hop allow the network to focus on different aspects of memory.

  ## Architecture

  ```
  Query [batch, input_dim]     Memories [batch, num_memories, input_dim]
        |                              |
        v                              v
  +-----------+                 +-------------+
  |  Embed A  |                 |   Embed A   |  (attention embedding)
  +-----------+                 +-------------+
        |                              |
        +--------> Attention <---------+
                      |
                +-------------+
                |   Embed C   |  (output embedding)
                +-------------+
                      |
                      v
              Weighted Sum = o
                      |
                      v
              q' = q + o      (update query)
                      |
                      v
              (repeat K hops)
                      |
                      v
              Output [batch, output_dim]
  ```

  ## Usage

      model = MemoryNetwork.build(
        input_dim: 128,
        memory_dim: 128,
        num_hops: 3
      )

  ## References
  - Sukhbaatar et al., "End-To-End Memory Networks" (2015)
  - https://arxiv.org/abs/1503.08895
  """

  @default_num_hops 3
  @default_memory_dim 128

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an End-to-End Memory Network.

  ## Options
    - `:input_dim` - Input/query feature dimension (required)
    - `:memory_dim` - Internal memory embedding dimension (default: 128)
    - `:num_hops` - Number of memory reading iterations (default: 3)
    - `:output_dim` - Output dimension (default: same as input_dim)
    - `:num_memories` - Expected number of memory slots, nil for dynamic (default: nil)

  ## Returns
    An Axon model taking query `[batch, input_dim]` and memories
    `[batch, num_memories, input_dim]`, producing `[batch, output_dim]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:memory_dim, pos_integer()}
          | {:num_hops, pos_integer()}
          | {:num_memories, pos_integer()}
          | {:output_dim, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    memory_dim = Keyword.get(opts, :memory_dim, @default_memory_dim)
    num_hops = Keyword.get(opts, :num_hops, @default_num_hops)
    output_dim = Keyword.get(opts, :output_dim, input_dim)
    num_memories = Keyword.get(opts, :num_memories, nil)

    # Inputs
    query = Axon.input("query", shape: {nil, input_dim})
    memories = Axon.input("memories", shape: {nil, num_memories, input_dim})

    # Project query to memory dimension
    q = Axon.dense(query, memory_dim, name: "query_proj")

    # Multi-hop memory reading
    q = build_multi_hop(q, memories, num_hops: num_hops, memory_dim: memory_dim)

    # Final output projection
    Axon.dense(q, output_dim, name: "output_proj")
  end

  @doc """
  Perform a single memory hop: attend to memories, read, update query.

  One hop consists of:
  1. Compute attention weights over memories using current query
  2. Read a weighted sum from memories (output embedding)
  3. Add the read result to the query (residual update)

  ## Parameters
    - `query` - Current query state `[batch, memory_dim]`
    - `memories` - Memory slots `[batch, num_memories, input_dim]`

  ## Options
    - `:memory_dim` - Memory embedding dimension
    - `:hop_idx` - Index of this hop (for naming)
    - `:name` - Layer name prefix

  ## Returns
    Updated query `[batch, memory_dim]`
  """
  @spec memory_hop(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def memory_hop(query, memories, opts \\ []) do
    memory_dim = Keyword.get(opts, :memory_dim, @default_memory_dim)
    hop_idx = Keyword.get(opts, :hop_idx, 0)
    name = Keyword.get(opts, :name, "hop_#{hop_idx}")

    # Attention embedding: project memories for computing attention scores
    # memories: [batch, num_memories, input_dim] -> [batch, num_memories, memory_dim]
    mem_attention = Axon.dense(memories, memory_dim, name: "#{name}_embed_a")

    # Output embedding: project memories for reading content
    mem_output = Axon.dense(memories, memory_dim, name: "#{name}_embed_c")

    # Compute attention and read memory
    Axon.layer(
      &memory_hop_impl/4,
      [query, mem_attention, mem_output],
      name: name,
      op_name: :memory_hop
    )
  end

  @doc """
  Stack multiple memory hops for iterative memory reading.

  Each hop refines the query by attending to memories and incorporating
  the read result. Later hops can focus on different memory aspects
  because the query has been updated by previous hops.

  ## Parameters
    - `query` - Initial query `[batch, memory_dim]`
    - `memories` - Memory slots `[batch, num_memories, input_dim]`

  ## Options
    - `:num_hops` - Number of hops (default: 3)
    - `:memory_dim` - Memory embedding dimension (default: 128)

  ## Returns
    Final query after all hops `[batch, memory_dim]`
  """
  @spec build_multi_hop(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def build_multi_hop(query, memories, opts \\ []) do
    num_hops = Keyword.get(opts, :num_hops, @default_num_hops)
    memory_dim = Keyword.get(opts, :memory_dim, @default_memory_dim)

    Enum.reduce(0..(num_hops - 1), query, fn hop_idx, q ->
      memory_hop(q, memories,
        memory_dim: memory_dim,
        hop_idx: hop_idx,
        name: "hop_#{hop_idx}"
      )
    end)
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Single memory hop: attention + read + residual update
  defp memory_hop_impl(query, mem_attention, mem_output, _opts) do
    # query: [batch, memory_dim]
    # mem_attention: [batch, num_memories, memory_dim]
    # mem_output: [batch, num_memories, memory_dim]

    # Attention scores: query^T * mem_attention
    # query: [batch, memory_dim] -> [batch, 1, memory_dim]
    query_expanded = Nx.new_axis(query, 1)

    # scores: [batch, num_memories]
    scores = Nx.sum(Nx.multiply(query_expanded, mem_attention), axes: [2])

    # Softmax attention weights
    max_score = Nx.reduce_max(scores, axes: [1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_score))
    weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [1], keep_axes: true))

    # Weighted read from output embedding: [batch, memory_dim]
    # weights: [batch, num_memories] -> [batch, num_memories, 1]
    weights_expanded = Nx.new_axis(weights, 2)
    read_result = Nx.sum(Nx.multiply(weights_expanded, mem_output), axes: [1])

    # Residual update: q' = q + o
    Nx.add(query, read_result)
  end
end
