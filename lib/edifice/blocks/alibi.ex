defmodule Edifice.Blocks.ALiBi do
  @moduledoc """
  Attention with Linear Biases (ALiBi).

  Replaces positional embeddings with a simple linear bias added to attention
  scores. Each attention head gets a different slope, creating head-specific
  position sensitivity. ALiBi provides strong length extrapolation without
  any learned position parameters.

  ## Formula

      attention(Q, K) = softmax(QK^T / sqrt(d) + m * distance_matrix)

  where m is a head-specific slope and distance_matrix[i,j] = -(|i - j|).

  ## Slope Schedule

  Slopes are geometric: m_i = 2^(-8i/n_heads) for i = 1..n_heads.
  Lower heads get steeper slopes (more local), higher heads get gentler
  slopes (more global).

  ## Usage

      # Get ALiBi bias matrix for attention
      bias = ALiBi.compute_bias(seq_len: 128, num_heads: 8)

      # Add to attention scores before softmax
      scores = scores + bias

  ## References
  - "Train Short, Test Long" (Press et al., 2022)
  - https://arxiv.org/abs/2108.12409
  """

  @doc """
  Compute ALiBi slopes for each attention head.

  Returns tensor of shape [num_heads] with geometric slopes.
  """
  @spec compute_slopes(pos_integer()) :: Nx.Tensor.t()
  def compute_slopes(num_heads) do
    # Geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    ratio = 8.0 / num_heads

    Enum.map(1..num_heads, fn i ->
      :math.pow(2.0, -ratio * i)
    end)
    |> Nx.tensor(type: :f32)
  end

  @doc """
  Compute ALiBi bias matrix for a given sequence length and number of heads.

  Returns bias of shape [num_heads, seq_len, seq_len] to add to attention scores.

  ## Options
    - `:seq_len` - Sequence length (required)
    - `:num_heads` - Number of attention heads (required)
    - `:causal` - Use causal (lower-triangular) distances (default: true)
  """
  @spec compute_bias(keyword()) :: Nx.Tensor.t()
  def compute_bias(opts) do
    seq_len = Keyword.fetch!(opts, :seq_len)
    num_heads = Keyword.fetch!(opts, :num_heads)
    causal = Keyword.get(opts, :causal, true)

    slopes = compute_slopes(num_heads)

    # Distance matrix: -(|i - j|) or -(max(0, j - i)) for causal
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    distances =
      if causal do
        # Causal: only look back, mask future with large negative
        diff = Nx.subtract(cols, rows)
        # Where j > i (future), use -inf; otherwise use -(i - j)
        causal_mask = Nx.greater(diff, 0)
        neg_dist = Nx.negate(Nx.subtract(rows, cols))

        Nx.select(causal_mask, Nx.broadcast(-1.0e9, {seq_len, seq_len}), neg_dist)
      else
        Nx.negate(Nx.abs(Nx.subtract(rows, cols)))
      end

    # slopes: [num_heads] -> [num_heads, 1, 1]
    slopes_expanded = Nx.reshape(slopes, {num_heads, 1, 1})

    # bias: [num_heads, seq_len, seq_len]
    Nx.multiply(slopes_expanded, Nx.broadcast(distances, {num_heads, seq_len, seq_len}))
  end
end
