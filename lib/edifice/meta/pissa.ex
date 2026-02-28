defmodule Edifice.Meta.PiSSA do
  @moduledoc """
  PiSSA: Principal Singular Values and Singular Vectors Adaptation.

  PiSSA is structurally identical to LoRA but uses SVD-based initialization.
  Instead of random A + zero B (where the delta starts at zero), PiSSA
  initializes A and B from the top-r principal SVD components of the
  pretrained weight matrix. The residual `W_res = W - BA` is frozen.

  ## Key Insight

  LoRA fine-tunes a random low-rank subspace, requiring many steps to
  discover the important directions. PiSSA starts from the most important
  directions immediately:

  ```
  Given pretrained W, compute SVD: W = U * Sigma * V^T

  Top-r principal components:
    A = sqrt(Sigma_r) * V_r^T    [rank, input_size]
    B = U_r * sqrt(Sigma_r)      [output_size, rank]

  Frozen residual:
    W_res = W - B * A

  Forward:
    output = W_res * x + (alpha/rank) * B(A(x))
  ```

  ## Usage

      # Standalone adapter (random initialization for architecture testing)
      model = PiSSA.build(input_size: 768, output_size: 768, rank: 8)

      # Decompose a pretrained weight matrix for SVD-initialized adapter
      {a_init, b_init, w_residual} = PiSSA.decompose(weight_matrix, rank: 8)

      # Wrap an existing dense layer
      adapted = PiSSA.wrap(input, original, rank: 8, alpha: 16.0, name: "pissa")

  ## References

  - Meng et al., "PiSSA: Principal Singular Values and Singular Vectors
    Adaptation of Large Language Models" (NeurIPS 2024 Spotlight)
  - https://arxiv.org/abs/2404.02948
  """

  @default_rank 8
  @default_alpha 16.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:alpha, float()}
          | {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:rank, pos_integer()}

  @doc """
  Build a standalone PiSSA adapter layer.

  When no pretrained weights are provided, uses Gaussian initialization
  for A (simulating principal components) and zero initialization for B.
  For SVD-initialized adapters, use `decompose/2` to obtain initializers
  and pass via `:a_initializer` and `:b_initializer`.

  ## Options

    - `:input_size` - Input dimension (required)
    - `:output_size` - Output dimension (required)
    - `:rank` - Low-rank dimension (default: 8)
    - `:alpha` - Scaling factor (default: 16.0)
    - `:name` - Layer name prefix (default: "pissa")
    - `:a_initializer` - Custom initializer for A matrix (default: lecun_normal)
    - `:b_initializer` - Custom initializer for B matrix (default: zeros)

  ## Returns

    An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "pissa")

    input = Axon.input("input", shape: {nil, input_size})

    pissa_delta(input, output_size,
      rank: rank,
      alpha: alpha,
      name: name,
      a_initializer: Keyword.get(opts, :a_initializer),
      b_initializer: Keyword.get(opts, :b_initializer)
    )
  end

  @doc """
  Wrap an existing dense layer with a PiSSA adapter.

  The output is the sum of the original (frozen residual) layer and the
  low-rank adaptation: `W_res*x + (alpha/rank) * B(A(x))`.

  ## Parameters

    - `input` - The Axon input node
    - `original` - The original Axon dense layer output

  ## Options

    - `:rank` - Low-rank dimension (default: 8)
    - `:alpha` - Scaling factor (default: 16.0)
    - `:name` - Layer name prefix (default: "pissa")
    - `:output_size` - Required output dimension
    - `:a_initializer` - Custom initializer for A (default: lecun_normal)
    - `:b_initializer` - Custom initializer for B (default: zeros)

  ## Returns

    An Axon node with the adapted output.
  """
  @spec wrap(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def wrap(input, original, opts \\ []) do
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "pissa")

    delta =
      pissa_delta(input, output_size,
        rank: rank,
        alpha: alpha,
        name: name,
        a_initializer: Keyword.get(opts, :a_initializer),
        b_initializer: Keyword.get(opts, :b_initializer)
      )

    Axon.add(original, delta, name: "#{name}_adapted")
  end

  @doc """
  Build a PiSSA delta: the low-rank component `(alpha/rank) * B(A(x))`.

  Structurally identical to LoRA delta but with different default
  initialization (lecun_normal for A instead of small normal, reflecting
  that A captures principal directions).

  ## Parameters

    - `input` - Axon input node
    - `output_size` - Target output dimension

  ## Options

    - `:rank` - Low-rank dimension (default: 8)
    - `:alpha` - Scaling factor (default: 16.0)
    - `:name` - Layer name prefix
    - `:a_initializer` - Custom initializer for A
    - `:b_initializer` - Custom initializer for B
  """
  @spec pissa_delta(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def pissa_delta(input, output_size, opts \\ []) do
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "pissa")
    scale = alpha / rank

    # A: down-project (captures principal input directions)
    # Default: lecun_normal (PiSSA's A represents principal singular vectors)
    a_init = Keyword.get(opts, :a_initializer) || Axon.Initializers.lecun_normal()

    # B: up-project (captures principal output directions)
    # Default: zeros for clean start; with decompose/2, uses SVD values
    b_init = Keyword.get(opts, :b_initializer) || Axon.Initializers.zeros()

    a_proj =
      Axon.dense(input, rank,
        name: "#{name}_A",
        use_bias: false,
        kernel_initializer: a_init
      )

    b_proj =
      Axon.dense(a_proj, output_size,
        name: "#{name}_B",
        use_bias: false,
        kernel_initializer: b_init
      )

    Axon.nx(b_proj, fn x -> Nx.multiply(x, scale) end, name: "#{name}_scale")
  end

  @doc """
  Decompose a pretrained weight matrix for PiSSA initialization.

  Computes the top-r SVD of the weight matrix and returns initializer
  tensors for A, B, and the frozen residual W_res.

  ## Parameters

    - `weight` - Pretrained weight matrix `[output_size, input_size]`

  ## Options

    - `:rank` - Number of principal components to extract (default: 8)

  ## Returns

    `{a_init, b_init, w_residual}` where:
    - `a_init` - `[input_size, rank]` — transpose of `sqrt(Sigma_r) * V_r^T`
    - `b_init` - `[rank, output_size]` — transpose of `U_r * sqrt(Sigma_r)`
    - `w_residual` - `[output_size, input_size]` — frozen residual
  """
  @spec decompose(Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def decompose(weight, opts \\ []) do
    rank = Keyword.get(opts, :rank, @default_rank)

    # SVD: weight = U * diag(S) * V^T
    {u, s, vt} = Nx.LinAlg.svd(weight)

    # Extract top-r components
    u_r = Nx.slice_along_axis(u, 0, rank, axis: 1)
    s_r = Nx.slice_along_axis(s, 0, rank, axis: 0)
    vt_r = Nx.slice_along_axis(vt, 0, rank, axis: 0)

    sqrt_s = Nx.sqrt(s_r)

    # B_init = U_r * sqrt(Sigma_r) -> [output_size, rank]
    # For Axon dense kernel: [rank, output_size], so transpose
    b_matrix = Nx.multiply(u_r, Nx.new_axis(sqrt_s, 0))
    b_init = Nx.transpose(b_matrix)

    # A_init = sqrt(Sigma_r) * V_r^T -> [rank, input_size]
    # For Axon dense kernel: [input_size, rank], so transpose
    a_matrix = Nx.multiply(Nx.new_axis(sqrt_s, 1), vt_r)
    a_init = Nx.transpose(a_matrix)

    # Residual: W_res = W - B * A (in original [out, in] convention)
    ba = Nx.dot(b_matrix, a_matrix)
    w_residual = Nx.subtract(weight, ba)

    {a_init, b_init, w_residual}
  end

  @doc "Get the output size of a PiSSA layer."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :output_size)
  end

  @doc "Recommended default configuration."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      rank: 8,
      alpha: 16.0
    ]
  end
end
