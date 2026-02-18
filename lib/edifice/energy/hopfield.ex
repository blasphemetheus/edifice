defmodule Edifice.Energy.Hopfield do
  @moduledoc """
  Modern Continuous Hopfield Network (Ramsauer et al., 2020).

  Classical Hopfield networks store binary patterns and recall them via
  energy minimization. Modern Hopfield networks replace the quadratic
  energy with an exponential interaction function, yielding:

  1. Exponentially many stored patterns (vs polynomial in classical)
  2. Single-step convergence for retrieval
  3. Mathematical equivalence to attention with softmax

  ## Key Insight

  The update rule `softmax(beta * X * Y^T) * Y` is exactly the attention
  mechanism with query X, key Y, value Y, and inverse temperature beta.
  Higher beta -> sharper retrieval (more like nearest neighbor).
  Lower beta -> softer retrieval (more like averaging).

  ## Architecture

  ```
  Query X [batch, seq_len, input_dim]
       |
       v
  +----------------------------+
  |  Similarity: beta * X * Y^T |
  +----------------------------+
       |
       v
  +----------------------------+
  |       softmax(scores)       |
  +----------------------------+
       |
       v
  +----------------------------+
  |    Retrieval: weights * Y   |
  +----------------------------+
       |
       v
  Output [batch, seq_len, pattern_dim]
  ```

  ## Usage

      # Build a Hopfield layer
      model = Hopfield.build(input_dim: 128, num_patterns: 64, pattern_dim: 128)

      # Build an associative memory
      model = Hopfield.build_associative_memory(
        input_dim: 256,
        num_patterns: 128,
        pattern_dim: 256,
        beta: 2.0,
        num_heads: 4
      )

  ## References
  - Ramsauer et al., "Hopfield Networks is All You Need" (2020)
  - https://arxiv.org/abs/2008.02217
  """
  import Nx.Defn

  @default_beta 1.0
  @default_num_patterns 64
  @default_pattern_dim 128

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Modern Hopfield layer as an Axon model.

  Stores learnable patterns and retrieves them via exponential
  similarity (equivalent to attention).

  ## Options
    - `:input_dim` - Input feature dimension (required)
    - `:num_patterns` - Number of stored patterns N (default: 64)
    - `:pattern_dim` - Dimension of each pattern M (default: 128)
    - `:beta` - Inverse temperature for softmax sharpness (default: 1.0)

  ## Returns
    An Axon model: `[batch, input_dim]` -> `[batch, pattern_dim]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:beta, float()}
          | {:input_dim, pos_integer()}
          | {:num_patterns, pos_integer()}
          | {:pattern_dim, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    num_patterns = Keyword.get(opts, :num_patterns, @default_num_patterns)
    pattern_dim = Keyword.get(opts, :pattern_dim, @default_pattern_dim)
    beta = Keyword.get(opts, :beta, @default_beta)

    input = Axon.input("input", shape: {nil, input_dim})

    hopfield_layer(input,
      num_patterns: num_patterns,
      pattern_dim: pattern_dim,
      beta: beta,
      name: "hopfield"
    )
  end

  @doc """
  Apply a single Hopfield attention layer.

  Computes: `softmax(beta * X * Y^T) * Y`

  where Y are stored (learnable) patterns. The patterns are implemented as
  a dense projection to num_patterns (computing similarity scores), followed
  by a dense projection from num_patterns to pattern_dim (weighted retrieval).

  ## Parameters
    - `input` - Axon node with shape `[batch, input_dim]` or `[batch, seq_len, input_dim]`

  ## Options
    - `:num_patterns` - Number of stored patterns (default: 64)
    - `:pattern_dim` - Dimension of each pattern (default: 128)
    - `:beta` - Inverse temperature (default: 1.0)
    - `:name` - Layer name prefix (default: "hopfield")

  ## Returns
    An Axon node with shape `[batch, pattern_dim]` or `[batch, seq_len, pattern_dim]`
  """
  @spec hopfield_layer(Axon.t(), keyword()) :: Axon.t()
  def hopfield_layer(input, opts \\ []) do
    num_patterns = Keyword.get(opts, :num_patterns, @default_num_patterns)
    pattern_dim = Keyword.get(opts, :pattern_dim, @default_pattern_dim)
    beta = Keyword.get(opts, :beta, @default_beta)
    name = Keyword.get(opts, :name, "hopfield")

    # Stored pattern matrix Y: [num_patterns, pattern_dim]
    # The SAME Y is used for both similarity (X@Y^T) and retrieval (weights@Y),
    # which is the key insight: Modern Hopfield = attention with K=V=Y.
    patterns =
      Axon.param("#{name}_patterns", {num_patterns, pattern_dim}, initializer: :glorot_uniform)

    # Project input to pattern_dim for compatibility
    x_proj = Axon.dense(input, pattern_dim, name: "#{name}_query_proj")

    # Hopfield retrieval: softmax(beta * X @ Y^T) @ Y
    Axon.layer(
      &hopfield_retrieve_impl/3,
      [x_proj, patterns],
      name: "#{name}_retrieve",
      beta: beta,
      op_name: :hopfield_retrieve
    )
  end

  # Hopfield retrieval: softmax(beta * X @ Y^T) @ Y
  defp hopfield_retrieve_impl(x, patterns, opts) do
    beta = opts[:beta] || 1.0

    # x: [batch, pattern_dim] or [batch, seq_len, pattern_dim]
    # patterns: [num_patterns, pattern_dim]

    # Similarity scores: X @ Y^T -> [batch, ..., num_patterns]
    scores = Nx.dot(x, [-1], patterns, [1])
    scaled = Nx.multiply(scores, beta)

    # Stable softmax
    max_s = Nx.reduce_max(scaled, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scaled, max_s))
    weights = Nx.divide(exp_s, Nx.sum(exp_s, axes: [-1], keep_axes: true))

    # Retrieval: weights @ Y -> [batch, ..., pattern_dim]
    Nx.dot(weights, [-1], patterns, [0])
  end

  @doc """
  Build a Hopfield-based associative memory network.

  Multi-layer architecture with multiple Hopfield heads for robust
  pattern storage and retrieval.

  ## Options
    - `:input_dim` - Input feature dimension (required)
    - `:num_patterns` - Number of stored patterns per head (default: 64)
    - `:pattern_dim` - Dimension of each pattern (default: 128)
    - `:beta` - Inverse temperature (default: 1.0)
    - `:num_heads` - Number of parallel Hopfield heads (default: 1)
    - `:hidden_size` - Hidden dimension for projection layers (default: 256)
    - `:num_layers` - Number of Hopfield layers (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    An Axon model: `[batch, input_dim]` -> `[batch, hidden_size]`
  """
  @spec build_associative_memory(keyword()) :: Axon.t()
  def build_associative_memory(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    num_patterns = Keyword.get(opts, :num_patterns, @default_num_patterns)
    pattern_dim = Keyword.get(opts, :pattern_dim, @default_pattern_dim)
    beta = Keyword.get(opts, :beta, @default_beta)
    num_heads = Keyword.get(opts, :num_heads, 1)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 2)
    dropout = Keyword.get(opts, :dropout, 0.1)

    input = Axon.input("input", shape: {nil, input_dim})

    # Project input to hidden dimension
    x = Axon.dense(input, hidden_size, name: "input_proj")
    x = Axon.layer_norm(x, name: "input_norm")

    # Stack Hopfield layers with residual connections
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Multi-head Hopfield: run multiple heads and concatenate
        head_outputs =
          Enum.map(1..num_heads, fn head_idx ->
            head_dim = div(hidden_size, num_heads)

            # Project to per-head dimension
            head_input =
              Axon.dense(acc, head_dim, name: "layer_#{layer_idx}_head_#{head_idx}_proj")

            hopfield_layer(head_input,
              num_patterns: num_patterns,
              pattern_dim: div(pattern_dim, num_heads),
              beta: beta,
              name: "layer_#{layer_idx}_head_#{head_idx}"
            )
          end)

        # Concatenate heads
        hopfield_out =
          if num_heads == 1 do
            hd(head_outputs)
          else
            Axon.concatenate(head_outputs, name: "layer_#{layer_idx}_concat")
          end

        # Project back to hidden_size if pattern_dim differs
        hopfield_out = Axon.dense(hopfield_out, hidden_size, name: "layer_#{layer_idx}_out_proj")

        hopfield_out =
          Axon.dropout(hopfield_out,
            rate: dropout,
            name: "layer_#{layer_idx}_dropout"
          )

        # Residual connection + layer norm
        residual = Axon.add(acc, hopfield_out, name: "layer_#{layer_idx}_residual")
        Axon.layer_norm(residual, name: "layer_#{layer_idx}_norm")
      end)

    x
  end

  # ============================================================================
  # Numerical Helpers
  # ============================================================================

  @doc """
  Compute the Hopfield energy for a state and stored patterns.

  Energy: E(x) = -beta * log(sum_i exp(beta * x^T * y_i))

  Lower energy = better match to stored patterns.
  This is a pure numerical function for analysis/debugging.

  ## Parameters
    - `query` - Query state `[batch, dim]`
    - `patterns` - Stored patterns `[num_patterns, dim]`
    - `beta` - Inverse temperature

  ## Returns
    Energy values `[batch]`
  """
  @spec energy(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn energy(query, patterns, beta) do
    # scores: [batch, num_patterns]
    scores = Nx.dot(query, [1], patterns, [1])
    scaled = beta * scores

    # log-sum-exp for numerical stability
    max_score = Nx.reduce_max(scaled, axes: [1], keep_axes: true)
    lse = max_score + Nx.log(Nx.sum(Nx.exp(scaled - max_score), axes: [1], keep_axes: true))

    -Nx.squeeze(lse, axes: [1])
  end
end
