defmodule Edifice.Blocks.RoPE do
  @moduledoc """
  Rotary Position Embedding (RoPE).

  Encodes position information by rotating query and key vectors in pairs
  of dimensions. This provides relative position awareness without explicit
  position embeddings, and naturally extrapolates to longer sequences.

  ## How It Works

  RoPE rotates each pair of dimensions (2i, 2i+1) by angle theta_i * position:

      [cos(m*theta_i)  -sin(m*theta_i)] [q_{2i}  ]
      [sin(m*theta_i)   cos(m*theta_i)] [q_{2i+1}]

  where m is position and theta_i = base^(-2i/d).

  The inner product between rotated Q and K at positions m and n depends only
  on (m - n), giving relative position sensitivity.

  ## Usage

      # Apply RoPE to query and key tensors
      {q_rotated, k_rotated} = RoPE.apply_rotary(query, key, seq_len: 128)

      # As an Axon layer
      rotated = RoPE.layer(input, dim: 64, seq_len: 128)

  ## References
  - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
  - https://arxiv.org/abs/2104.09864
  """
  import Nx.Defn

  @default_base 10_000.0

  @doc """
  Build precomputed frequency table for RoPE.

  Returns cosine and sine tables of shape [max_seq_len, dim/2].
  """
  @spec precompute_freqs(pos_integer(), pos_integer(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def precompute_freqs(dim, max_seq_len, opts \\ []) do
    base = Keyword.get(opts, :base, @default_base)

    half_dim = div(dim, 2)
    # theta_i = base^(-2i/dim)
    freqs =
      Nx.pow(
        base,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), dim))
      )
      |> Nx.as_type(:f32)

    # positions: [0, 1, ..., max_seq_len-1]
    positions = Nx.iota({max_seq_len}, type: :f32)

    # angles: [max_seq_len, half_dim]
    angles = Nx.outer(positions, freqs)

    {Nx.cos(angles), Nx.sin(angles)}
  end

  @doc """
  Apply rotary position embedding to Q and K tensors.

  ## Parameters
    - `query` - Query tensor [batch, seq_len, dim]
    - `key` - Key tensor [batch, seq_len, dim]

  ## Options
    - `:seq_len` - Sequence length (inferred from tensor if not provided)
    - `:base` - RoPE base frequency (default: 10000.0)

  ## Returns
    `{rotated_query, rotated_key}` with same shapes as input.
  """
  defn apply_rotary(query, key, _opts \\ []) do
    dim = Nx.axis_size(query, 2)
    seq_len = Nx.axis_size(query, 1)
    half_dim = div(dim, 2)

    # Compute frequency table
    freqs =
      Nx.pow(
        10_000.0,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), dim))
      )
      |> Nx.as_type(Nx.type(query))

    positions = Nx.iota({seq_len}) |> Nx.as_type(Nx.type(query))
    angles = Nx.outer(positions, freqs)

    cos_table = Nx.cos(angles) |> Nx.new_axis(0)
    sin_table = Nx.sin(angles) |> Nx.new_axis(0)

    q_rotated = rotate_half(query, cos_table, sin_table, half_dim)
    k_rotated = rotate_half(key, cos_table, sin_table, half_dim)

    {q_rotated, k_rotated}
  end

  defnp rotate_half(x, cos_table, sin_table, half_dim) do
    x1 = Nx.slice_along_axis(x, 0, half_dim, axis: 2)
    x2 = Nx.slice_along_axis(x, half_dim, half_dim, axis: 2)

    rotated1 = Nx.subtract(Nx.multiply(x1, cos_table), Nx.multiply(x2, sin_table))
    rotated2 = Nx.add(Nx.multiply(x1, sin_table), Nx.multiply(x2, cos_table))

    Nx.concatenate([rotated1, rotated2], axis: 2)
  end

  @doc """
  Build an Axon layer that applies RoPE to the input.

  ## Options
    - `:dim` - Feature dimension (required, must be even)
    - `:name` - Layer name prefix (default: "rope")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    _dim = Keyword.fetch!(opts, :dim)
    name = Keyword.get(opts, :name, "rope")

    Axon.nx(
      input,
      fn tensor ->
        dim = Nx.axis_size(tensor, 2)
        seq_len = Nx.axis_size(tensor, 1)
        half_dim = div(dim, 2)

        freqs =
          Nx.pow(
            10_000.0,
            Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), dim))
          )
          |> Nx.as_type(Nx.type(tensor))

        positions = Nx.iota({seq_len}) |> Nx.as_type(Nx.type(tensor))
        angles = Nx.outer(positions, freqs)
        cos_t = Nx.cos(angles) |> Nx.new_axis(0)
        sin_t = Nx.sin(angles) |> Nx.new_axis(0)

        x1 = Nx.slice_along_axis(tensor, 0, half_dim, axis: 2)
        x2 = Nx.slice_along_axis(tensor, half_dim, half_dim, axis: 2)

        r1 = Nx.subtract(Nx.multiply(x1, cos_t), Nx.multiply(x2, sin_t))
        r2 = Nx.add(Nx.multiply(x1, sin_t), Nx.multiply(x2, cos_t))

        Nx.concatenate([r1, r2], axis: 2)
      end,
      name: name
    )
  end
end
