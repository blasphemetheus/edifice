defmodule Edifice.Meta.DoRA do
  @moduledoc """
  DoRA: Weight-Decomposed Low-Rank Adaptation.

  Implements DoRA from "DoRA: Weight-Decomposed Low-Rank Adaptation of Large
  Language Models" (Liu et al., 2024). DoRA decomposes pretrained weights into
  magnitude and direction components, then applies LoRA only to the direction.

  ## Key Innovation: Magnitude-Direction Decomposition

  Standard LoRA modifies the full weight: `W' = W + BA`

  DoRA decomposes W into magnitude m and direction V:
  ```
  W = m * (V / ||V||)
  ```

  Then applies LoRA only to the direction component:
  ```
  W' = m * ((V + BA) / ||V + BA||)
  ```

  Where:
  - `m` is a learnable magnitude vector [output_size]
  - `V` is the original weight direction
  - `BA` is the standard LoRA low-rank update
  - `||.||` is column-wise L2 normalization

  ## Why This Works

  Separating magnitude from direction gives two benefits:
  1. **Direction** captures "what" features are important (adapted by LoRA)
  2. **Magnitude** captures "how much" each feature matters (learned separately)
  3. This mirrors weight normalization, which is known to improve optimization

  ## Architecture

  ```
  Input x [batch, input_size]
        |
        +---> W * x (frozen base)
        |        |
        +---> A * x -> B * (A * x)     (LoRA delta)
        |        |
        |     V + BA                    (direction update)
        |        |
        |     normalize(V + BA)         (unit direction)
        |        |
        |     m * normalized            (apply magnitude)
        |
        v
  Output [batch, output_size]
  ```

  ## LoRA+ Note

  LoRA+ (Hayou et al., 2024) proposes different learning rates for A vs B
  matrices. This is a training configuration choice rather than architectural:
  use a higher learning rate for B (e.g., 5-10x) than for A. We document
  this recommendation but don't enforce it in the graph structure.

  ## Usage

      # Standalone DoRA layer
      dora = DoRA.build(input_size: 768, output_size: 768, rank: 8)

      # Wrap an existing layer with DoRA
      adapted = DoRA.wrap(input, original, rank: 8, name: "dora_attn")

  ## References

  - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
  - https://arxiv.org/abs/2402.09353
  - Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models" (2024)
  """

  @default_rank 8
  @default_alpha 16.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:alpha, float()}
          | {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:rank, pos_integer()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a standalone DoRA adapter layer.

  Computes weight-decomposed adaptation: `m * normalize(V*x + (alpha/rank)*B(A(x)))`.

  ## Options

  - `:input_size` - Input dimension (required)
  - `:output_size` - Output dimension (required)
  - `:rank` - Low-rank dimension (default: 8)
  - `:alpha` - LoRA scaling factor (default: 16.0)
  - `:name` - Layer name prefix (default: "dora")

  ## Returns

  An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "dora")

    input = Axon.input("input", shape: {nil, input_size})

    dora_layer(input, input_size, output_size,
      rank: rank,
      alpha: alpha,
      name: name
    )
  end

  @doc """
  Build a DoRA layer inline (for use in custom architectures).

  ## Parameters

  - `input` - Axon input node
  - `input_size` - Input dimension
  - `output_size` - Output dimension

  ## Options

  - `:rank` - Low-rank dimension (default: 8)
  - `:alpha` - LoRA scaling factor (default: 16.0)
  - `:name` - Layer name prefix (default: "dora")
  """
  @spec dora_layer(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def dora_layer(input, input_size, output_size, opts \\ []) do
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "dora")
    scale = alpha / rank

    # Base weight (frozen direction component V)
    base_weight =
      Axon.param("#{name}_base_weight", {input_size, output_size},
        initializer: :glorot_uniform
      )

    # Learnable magnitude vector m [output_size]
    magnitude =
      Axon.param("#{name}_magnitude", {output_size},
        initializer: &init_magnitude/2
      )

    # LoRA matrices for direction update
    # A: down-project [input_size, rank]
    a_weight =
      Axon.param("#{name}_A", {input_size, rank},
        initializer: Axon.Initializers.normal(scale: 0.02)
      )

    # B: up-project [rank, output_size] (zero-init for clean start)
    b_weight =
      Axon.param("#{name}_B", {rank, output_size},
        initializer: Axon.Initializers.zeros()
      )

    # Compute DoRA: m * normalize(V*x + scale * B(A(x)))
    Axon.layer(
      &dora_impl/6,
      [input, base_weight, magnitude, a_weight, b_weight],
      name: "#{name}_compute",
      scale: scale,
      op_name: :dora
    )
  end

  @doc """
  Wrap an existing dense layer with DoRA adaptation.

  ## Parameters

  - `input` - The Axon input node
  - `original` - The original Axon dense layer output

  ## Options

  - `:output_size` - Output dimension (required)
  - `:rank` - Low-rank dimension (default: 8)
  - `:alpha` - Scaling factor (default: 16.0)
  - `:name` - Layer name prefix (default: "dora")
  """
  @spec wrap(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def wrap(input, original, opts \\ []) do
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    name = Keyword.get(opts, :name, "dora")
    scale = alpha / rank

    # Magnitude for scaling
    magnitude =
      Axon.param("#{name}_magnitude", {output_size},
        initializer: &init_magnitude/2
      )

    # LoRA delta: A down-projects, B up-projects
    a_proj =
      Axon.dense(input, rank,
        name: "#{name}_A",
        use_bias: false,
        kernel_initializer: Axon.Initializers.normal(scale: 0.02)
      )

    b_proj =
      Axon.dense(a_proj, output_size,
        name: "#{name}_B",
        use_bias: false,
        kernel_initializer: Axon.Initializers.zeros()
      )

    lora_delta = Axon.nx(b_proj, fn x -> Nx.multiply(x, scale) end, name: "#{name}_scale")

    # Combine: normalize(original + delta) * magnitude
    Axon.layer(
      fn orig, delta, mag, _opts ->
        combined = Nx.add(orig, delta)
        # Column-wise L2 normalization
        norm = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(combined, 2), axes: [-1], keep_axes: true), 1.0e-8))
        normalized = Nx.divide(combined, norm)
        Nx.multiply(normalized, mag)
      end,
      [original, lora_delta, magnitude],
      name: "#{name}_wrapped",
      op_name: :dora_wrap
    )
  end

  # ============================================================================
  # Implementation
  # ============================================================================

  # DoRA forward: m * normalize(V*x + scale * B(A(x)))
  defp dora_impl(input, base_weight, magnitude, a_weight, b_weight, opts) do
    scale = opts[:scale]

    # Base output: V * x -> [batch, output_size]
    base_out = Nx.dot(input, [1], base_weight, [0])

    # LoRA delta: scale * B(A(x)) -> [batch, output_size]
    a_out = Nx.dot(input, [1], a_weight, [0])
    lora_delta = Nx.multiply(Nx.dot(a_out, [1], b_weight, [0]), scale)

    # Combined direction: V*x + delta
    combined = Nx.add(base_out, lora_delta)

    # Normalize direction (per-sample L2 norm)
    norm = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(combined, 2), axes: [-1], keep_axes: true), 1.0e-8))
    normalized = Nx.divide(combined, norm)

    # Apply magnitude: m * normalized
    Nx.multiply(normalized, magnitude)
  end

  # Initialize magnitude to ones (preserves original scale at init)
  defp init_magnitude(shape, _opts) do
    Nx.broadcast(Nx.tensor(1.0), shape)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a DoRA layer."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :output_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      rank: 8,
      alpha: 16.0
    ]
  end
end
