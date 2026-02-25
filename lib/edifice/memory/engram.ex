defmodule Edifice.Memory.Engram do
  @moduledoc """
  Engram: O(1) Hash-Based Associative Memory via Locality-Sensitive Hashing.

  <!-- verified: true, date: 2026-02-23 -->

  Engram implements a fast key-value memory that stores and retrieves values
  in amortised O(1) time using Locality-Sensitive Hashing (LSH). Multiple
  independent hash tables reduce collision probability, and exponential
  moving-average (EMA) writes allow smooth interpolation of stored values.

  ## Motivation

  Classical associative memories (NTM, MemoryNetwork) perform O(N) attention
  over all N memory slots per query. Engram instead hashes each query into a
  small number of buckets, making both reads and writes O(hash_bits × key_dim)
  independent of the number of stored memories.

  ## Architecture

  ```
  Query key [batch, key_dim]
       |
       v
  +---------------------------+
  | LSH Hash (per table t):   |
  |   project = W_t @ key     |  W_t ∈ R^{hash_bits × key_dim}
  |   bits = sign(project)    |  {0,1}^hash_bits
  |   bucket = bits → int     |  0..num_buckets-1
  +---------------------------+
       |
       v  (for each of num_tables tables)
  +---------------------------+
  | Memory Slots              |
  |   [num_tables, num_buckets, value_dim]
  +---------------------------+
       |
       v
  Retrieve slot per table → average across tables
       |
       v
  Retrieved value [batch, value_dim]
  ```

  ## Hashing

  For `num_buckets = 256 = 2^8`:
  - `hash_bits = 8` random projection vectors per table
  - `W` shape: `[num_tables, hash_bits, key_dim]`
  - `bucket = sum(sign(W @ key) >= 0) * [1, 2, 4, ..., 2^{hash_bits-1}]`

  ## Write (EMA update)

  ```
  memory[t, hash(key)] ← decay × memory[t, hash(key)] + (1 − decay) × value
  ```

  ## Usage

      # Create memory state
      mem = Engram.new(key_dim: 32, value_dim: 64)

      # Read
      result = Engram.engram_read(mem, query)   # [batch, value_dim]

      # Write (returns updated memory)
      mem = Engram.engram_write(mem, key, value, decay: 0.99)

      # Build an Axon model for differentiable reads
      model = Engram.build(key_dim: 32, value_dim: 64)

  ## References

  - Andoni & Indyk, "Near-Optimal Hashing Algorithms for Approximate Nearest
    Neighbor in High Dimensions" (FOCS 2006)
  - Locality-Sensitive Hashing for associative memory is explored in
    "Reformer: The Efficient Transformer" (Kitaev et al., 2020)
  """

  @default_num_buckets 256
  @default_num_tables 4

  @doc """
  Build an Axon model for differentiable Engram reads.

  The model takes a query and the current memory slots as inputs; the
  LSH projection matrices are trainable parameters (useful when the
  hash function itself should be optimised end-to-end).

  ## Options

    - `:key_dim` - Query/key dimension (required)
    - `:value_dim` - Value / slot dimension (required)
    - `:num_buckets` - Number of hash buckets; must be a power of 2 (default: 256)
    - `:num_tables` - Number of independent hash tables (default: 4)

  ## Returns

    An Axon model with inputs:
    - `"query"` — `[batch, key_dim]`
    - `"memory_slots"` — `[num_tables, num_buckets, value_dim]`

    Returns retrieved values `[batch, value_dim]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:key_dim, pos_integer()}
          | {:num_buckets, pos_integer()}
          | {:num_tables, pos_integer()}
          | {:value_dim, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    key_dim = Keyword.fetch!(opts, :key_dim)
    value_dim = Keyword.fetch!(opts, :value_dim)
    num_buckets = Keyword.get(opts, :num_buckets, @default_num_buckets)
    num_tables = Keyword.get(opts, :num_tables, @default_num_tables)
    hash_bits = compute_hash_bits(num_buckets)

    query = Axon.input("query", shape: {nil, key_dim})
    memory_slots = Axon.input("memory_slots", shape: {num_tables, num_buckets, value_dim})

    # Learnable LSH projection matrices — one per table
    hash_matrices =
      Axon.param("hash_matrices", {num_tables, hash_bits, key_dim}, initializer: :glorot_uniform)

    Axon.layer(
      &engram_lookup_impl/4,
      [query, memory_slots, hash_matrices],
      name: "engram_lookup",
      num_tables: num_tables,
      hash_bits: hash_bits,
      op_name: :engram_lookup
    )
  end

  @doc """
  Initialise a fresh Engram memory state.

  Creates random LSH projection matrices (normalised) and zero-filled
  memory slots.

  ## Options

    - `:key_dim` - Key dimension (required)
    - `:value_dim` - Value dimension (required)
    - `:num_buckets` - Number of hash buckets (default: 256)
    - `:num_tables` - Number of hash tables (default: 4)
    - `:seed` - Random seed for reproducibility (default: 0)

  ## Returns

    A map `%{hash_matrices: Nx.Tensor.t(), slots: Nx.Tensor.t()}`.
  """
  @spec new([build_opt() | {:seed, non_neg_integer()}]) :: %{
          hash_matrices: Nx.Tensor.t(),
          slots: Nx.Tensor.t()
        }
  def new(opts \\ []) do
    key_dim = Keyword.fetch!(opts, :key_dim)
    value_dim = Keyword.fetch!(opts, :value_dim)
    num_buckets = Keyword.get(opts, :num_buckets, @default_num_buckets)
    num_tables = Keyword.get(opts, :num_tables, @default_num_tables)
    seed = Keyword.get(opts, :seed, 0)
    hash_bits = compute_hash_bits(num_buckets)

    key = Nx.Random.key(seed)
    {raw, _} = Nx.Random.normal(key, shape: {num_tables, hash_bits, key_dim})

    # Normalise rows so each projection vector has unit length
    norms = Nx.sqrt(Nx.sum(Nx.pow(raw, 2), axes: [2], keep_axes: true))
    hash_matrices = Nx.divide(raw, Nx.add(norms, 1.0e-8))

    slots = Nx.broadcast(Nx.tensor(0.0), {num_tables, num_buckets, value_dim})

    %{hash_matrices: hash_matrices, slots: slots}
  end

  @doc """
  Read from Engram memory using LSH-based lookup.

  ## Parameters

    - `memory` - Memory state map with `:hash_matrices` and `:slots`
    - `query` - Query tensor `[batch, key_dim]` or `[key_dim]`

  ## Returns

    Retrieved value `[batch, value_dim]` (averaged across tables).
  """
  @spec engram_read(%{hash_matrices: Nx.Tensor.t(), slots: Nx.Tensor.t()}, Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def engram_read(%{hash_matrices: hash_mats, slots: slots}, query) do
    # Ensure query is 2D [batch, key_dim]
    query = if Nx.rank(query) == 1, do: Nx.new_axis(query, 0), else: query

    batch_size = Nx.axis_size(query, 0)
    num_tables = Nx.axis_size(hash_mats, 0)
    hash_bits = Nx.axis_size(hash_mats, 1)
    value_dim = Nx.axis_size(slots, 2)

    bucket_indices = compute_bucket_indices(query, hash_mats, batch_size, hash_bits, num_tables)

    results = gather_slots(slots, bucket_indices, batch_size, value_dim)
    # Average retrieved values across tables: [T, B, V] → [B, V]
    Nx.mean(results, axes: [0])
  end

  @doc """
  Write a key-value pair into Engram memory using EMA updates.

  Each hash table independently maps the key to a bucket and applies:

      slot[t, bucket] ← decay × slot[t, bucket] + (1 − decay) × value

  ## Parameters

    - `memory` - Memory state map with `:hash_matrices` and `:slots`
    - `key` - Key tensor `[key_dim]` or `[1, key_dim]`
    - `value` - Value tensor `[value_dim]` or `[1, value_dim]`

  ## Options

    - `:decay` - EMA decay coefficient (default: 0.99)

  ## Returns

    Updated memory state map.
  """
  @spec engram_write(
          %{hash_matrices: Nx.Tensor.t(), slots: Nx.Tensor.t()},
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: %{hash_matrices: Nx.Tensor.t(), slots: Nx.Tensor.t()}
  def engram_write(%{hash_matrices: hash_mats, slots: slots} = memory, key, value, opts \\ []) do
    decay = Keyword.get(opts, :decay, 0.99)

    key_1d = if Nx.rank(key) == 1, do: key, else: Nx.squeeze(key)
    value_1d = if Nx.rank(value) == 1, do: value, else: Nx.squeeze(value)

    num_tables = Nx.axis_size(hash_mats, 0)
    hash_bits = Nx.axis_size(hash_mats, 1)

    # Compute bucket index for each table: [T]
    bucket_indices = compute_bucket_indices_1d(key_1d, hash_mats, hash_bits, num_tables)

    new_slots = update_slots(slots, bucket_indices, value_1d, decay, num_tables)
    %{memory | slots: new_slots}
  end

  # ============================================================================
  # Private: hashing and slot operations
  # ============================================================================

  # Number of bits needed to address num_buckets buckets (must be power of 2).
  defp compute_hash_bits(num_buckets), do: trunc(:math.log2(num_buckets))

  # Compute bucket indices for a batched query.
  # query: [B, D], hash_mats: [T, H, D]
  # Returns bucket_indices [T, B] (integer)
  defp compute_bucket_indices(query, hash_mats, batch_size, hash_bits, num_tables) do
    key_dim = Nx.axis_size(query, 1)

    # Expand query for batched matmul: [B, D] → [T, B, D]
    query_bc = Nx.broadcast(Nx.new_axis(query, 0), {num_tables, batch_size, key_dim})

    # Batched matmul: [T, B, D] × [T, H, D] (contract D, batch T) → [T, B, H]
    projections = Nx.dot(query_bc, [2], [0], hash_mats, [2], [0])

    bits_to_buckets(projections, hash_bits)
  end

  # Compute bucket indices for a single key (no batch).
  # key: [D], hash_mats: [T, H, D]
  # Returns bucket_indices [T] (integer)
  defp compute_bucket_indices_1d(key, hash_mats, hash_bits, num_tables) do
    key_dim = Nx.axis_size(key, 0)

    # Flatten hash_mats to [T*H, D], dot with key [D] → [T*H]
    proj_flat = Nx.dot(Nx.reshape(hash_mats, {num_tables * hash_bits, key_dim}), key)
    projections = Nx.reshape(proj_flat, {num_tables, 1, hash_bits})

    bucket_indices_tb = bits_to_buckets(projections, hash_bits)
    # Squeeze batch dimension: [T, 1] → [T]
    Nx.squeeze(bucket_indices_tb, axes: [1])
  end

  # Convert signed projections to integer bucket indices.
  # projections: [T, B, H], returns [T, B]
  defp bits_to_buckets(projections, hash_bits) do
    # Map sign: ≥0 → 1, <0 → 0
    bits = Nx.greater_equal(projections, 0.0) |> Nx.as_type(:s32)

    # Powers of 2: [1, 2, 4, ..., 2^{hash_bits-1}]
    powers =
      Nx.pow(Nx.tensor(2, type: :s32), Nx.iota({hash_bits}, type: :s32))
      |> Nx.reshape({1, 1, hash_bits})

    # Binary vector → integer: sum(bits * powers) along H dimension
    Nx.sum(Nx.multiply(bits, powers), axes: [2])
  end

  # Gather slot values for each (table, batch) bucket index.
  # slots: [T, num_buckets, V], bucket_indices: [T, B]
  # Returns [T, B, V]
  defp gather_slots(slots, bucket_indices, _batch_size, _value_dim) do
    num_buckets = Nx.axis_size(slots, 1)

    # One-hot selection matrix: [T, B, num_buckets]
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(bucket_indices, :s32), 2),
        Nx.reshape(Nx.iota({num_buckets}), {1, 1, num_buckets})
      )
      |> Nx.as_type(:f32)

    # Batched gather: [T, B, num_buckets] × [T, num_buckets, V] → [T, B, V]
    # Contract over num_buckets, batch over T
    Nx.dot(one_hot, [2], [0], slots, [1], [0])
  end

  # Apply EMA updates for each table.
  # slots: [T, num_buckets, V], bucket_indices: [T], value: [V]
  # Returns updated slots [T, num_buckets, V]
  defp update_slots(slots, bucket_indices, value, decay, num_tables) do
    num_buckets = Nx.axis_size(slots, 1)
    value_dim = Nx.axis_size(slots, 2)

    # Iterate over tables at the Elixir level (num_tables is small, typically 4)
    Enum.reduce(0..(num_tables - 1), slots, fn t, acc ->
      bucket_idx = Nx.to_number(acc_index(bucket_indices, t))

      # Build a [T, num_buckets, 1] position mask for (t, bucket_idx)
      t_mask = Nx.equal(Nx.iota({num_tables, 1, 1}), t) |> Nx.as_type(:f32)
      b_mask = Nx.equal(Nx.iota({1, num_buckets, 1}), bucket_idx) |> Nx.as_type(:f32)
      pos_mask = Nx.multiply(t_mask, b_mask)

      # Broadcast updated value: [1, 1, V] → [T, num_buckets, V]
      updated_bc =
        Nx.broadcast(Nx.reshape(value, {1, 1, value_dim}), {num_tables, num_buckets, value_dim})

      # EMA: decay * current + (1-decay) * new_value, applied only at (t, bucket_idx)
      Nx.add(
        Nx.multiply(acc, Nx.subtract(1.0, pos_mask)),
        Nx.multiply(
          Nx.add(Nx.multiply(decay, acc), Nx.multiply(1.0 - decay, updated_bc)),
          pos_mask
        )
      )
    end)
  end

  # Helper: get scalar value at integer index from a 1-D tensor.
  defp acc_index(tensor, idx) do
    Nx.slice_along_axis(tensor, idx, 1, axis: 0) |> Nx.squeeze()
  end

  # Axon layer implementation for differentiable read.
  defp engram_lookup_impl(query, memory_slots, hash_matrices, opts) do
    num_tables = opts[:num_tables]
    hash_bits = opts[:hash_bits]
    batch_size = Nx.axis_size(query, 0)
    value_dim = Nx.axis_size(memory_slots, 2)

    bucket_indices =
      compute_bucket_indices(query, hash_matrices, batch_size, hash_bits, num_tables)

    results = gather_slots(memory_slots, bucket_indices, batch_size, value_dim)
    Nx.mean(results, axes: [0])
  end
end
