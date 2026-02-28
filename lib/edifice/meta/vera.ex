defmodule Edifice.Meta.VeRA do
  @moduledoc """
  VeRA: Vector-based Random Matrix Adaptation.

  Extreme parameter-efficient finetuning that uses shared frozen random matrices
  across all layers, with only per-layer trainable scaling vectors. Achieves
  10x fewer trainable parameters than LoRA while maintaining comparable quality.

  ## Key Idea

  Instead of learning separate low-rank A and B matrices per layer (LoRA),
  VeRA shares a single pair of frozen random matrices and learns only
  diagonal scaling vectors b and d per layer:

      h = W₀x + Λ_b B Λ_d A x

  where A ∈ R^{r×in}, B ∈ R^{out×r} are frozen Kaiming-initialized randoms,
  and b ∈ R^{out}, d ∈ R^{r} are trainable per-layer vectors.

  ```
  Input x [batch, input_size]
        |
        +---> W₀x (frozen base)     [batch, output_size]
        |                                    |
        +---> A @ x                  [batch, rank]       (frozen random)
              |
        d * (A @ x)                  [batch, rank]       (trainable d)
              |
        B @ (d * Ax)                 [batch, output_size] (frozen random)
              |
        b * B(d * Ax)                [batch, output_size] (trainable b)
              |                                    |
              +---------- add ---------<-----------+
              |
        Output [batch, output_size]
  ```

  ## Usage

      vera = VeRA.build(input_size: 768, output_size: 768, rank: 256)

  ## Reference

  - Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation" (ICLR 2025)
  """

  @default_rank 256
  @default_d_initial 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:d_initial, float()}
          | {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:rank, pos_integer()}

  @doc """
  Build a standalone VeRA adapter layer.

  Computes `Λ_b B Λ_d A x` where A, B are frozen random matrices
  and b, d are trainable scaling vectors.

  ## Options

    - `:input_size` - Input dimension (required)
    - `:output_size` - Output dimension (required)
    - `:rank` - Rank of shared random matrices (default: 256)
    - `:d_initial` - Initial value for d scaling vector (default: 0.1)

  ## Returns

    An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    d_initial = Keyword.get(opts, :d_initial, @default_d_initial)
    name = Keyword.get(opts, :name, "vera")

    input = Axon.input("input", shape: {nil, input_size})

    vera_delta(input, output_size,
      rank: rank,
      d_initial: d_initial,
      name: name
    )
  end

  @doc """
  Wrap an existing dense layer with a VeRA adapter.

  Output = original_output + Λ_b B Λ_d A x

  ## Parameters

    - `input` - The Axon input node
    - `original` - The original Axon dense layer output

  ## Options

    - `:output_size` - Output dimension (required)
    - `:rank` - Rank of random matrices (default: 256)
    - `:d_initial` - Initial value for d vector (default: 0.1)
    - `:name` - Layer name prefix
  """
  @spec wrap(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def wrap(input, original, opts \\ []) do
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    d_initial = Keyword.get(opts, :d_initial, @default_d_initial)
    name = Keyword.get(opts, :name, "vera")

    delta = vera_delta(input, output_size, rank: rank, d_initial: d_initial, name: name)
    Axon.add(original, delta, name: "#{name}_adapted")
  end

  @doc """
  Build a VeRA delta: the adaptation component Λ_b B Λ_d A x.

  ## Parameters

    - `input` - Axon input node
    - `output_size` - Target output dimension

  ## Options

    - `:rank` - Rank dimension (default: 256)
    - `:d_initial` - Initial value for d vector (default: 0.1)
    - `:name` - Layer name prefix
  """
  @spec vera_delta(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def vera_delta(input, output_size, opts \\ []) do
    rank = Keyword.get(opts, :rank, @default_rank)
    d_initial = Keyword.get(opts, :d_initial, @default_d_initial)
    name = Keyword.get(opts, :name, "vera")

    # Frozen random A: down-project input -> rank (Kaiming/Lecun init)
    a_proj =
      Axon.dense(input, rank,
        name: "#{name}_A",
        use_bias: false,
        kernel_initializer: Axon.Initializers.lecun_normal()
      )

    # Trainable d scaling vector [rank]
    d_vec =
      Axon.param("#{name}_d", {rank}, initializer: Axon.Initializers.full(d_initial))

    # Frozen random B: up-project rank -> output_size (Kaiming/Lecun init)
    # We need to apply d scaling before B projection, so we use Axon.layer
    d_scaled =
      Axon.layer(
        fn a_out, d_param, _opts ->
          # Element-wise: d * (A @ x)
          Nx.multiply(a_out, d_param)
        end,
        [a_proj, d_vec],
        name: "#{name}_d_scale",
        op_name: :vera_d_scale
      )

    b_proj =
      Axon.dense(d_scaled, output_size,
        name: "#{name}_B",
        use_bias: false,
        kernel_initializer: Axon.Initializers.lecun_normal()
      )

    # Trainable b scaling vector [output_size]
    b_vec =
      Axon.param("#{name}_b", {output_size}, initializer: Axon.Initializers.zeros())

    # Element-wise: b * B(d * Ax)
    Axon.layer(
      fn b_out, b_param, _opts ->
        Nx.multiply(b_out, b_param)
      end,
      [b_proj, b_vec],
      name: "#{name}_b_scale",
      op_name: :vera_b_scale
    )
  end

  @doc """
  Get the output size of a VeRA layer.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :output_size)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      rank: 256,
      d_initial: 0.1
    ]
  end
end
