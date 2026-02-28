defmodule Edifice.Meta.KronLoRA do
  @moduledoc """
  Kron-LoRA: Hybrid Kronecker-LoRA Adapters for parameter-efficient finetuning.

  Factorizes weight updates as a Kronecker product of a small matrix and a
  LoRA-compressed factor. Leverages `rank(A kron B) = rank(A) * rank(B)` to
  achieve high effective rank with ~4x fewer parameters than standard LoRA.

  ## Key Idea

  Instead of LoRA's `delta_W = B * A`, Kron-LoRA factorizes:

      delta_W = A_kron kron (B1 * B2)

  where `A_kron` is a small Kronecker factor and `B1 * B2` is a LoRA-compressed
  matrix. The efficient forward pass avoids materializing the full `delta_W`:

  ```
  Input x [batch, d_in]
        |
  Reshape [batch, d_A1, d_B1]        -- expose Kronecker structure
        |
  B2 @ x [batch, d_A1, r]            -- LoRA down-project
        |
  A_kron mix [batch, r, d_A2]        -- Kronecker mixing
        |
  B1 @ z [batch, d_A2, d_B2]         -- LoRA up-project
        |
  Reshape [batch, d_out]              -- flatten back
        |
  (alpha/r) * delta + W0*x           -- scale and add to frozen
  ```

  Dimension constraints: `d_in = d_A1 * d_B1` and `d_out = d_A2 * d_B2`.

  ## Usage

      # Standalone
      kl = KronLoRA.build(input_size: 4096, output_size: 4096, rank: 8)

      # Wrap existing layer
      adapted = KronLoRA.wrap(input, original,
        input_size: 4096, output_size: 4096, rank: 8, name: "kl_attn")

  ## References

  - Shen, "Kron-LoRA: Hybrid Kronecker-LoRA Adapters" (2025)
  - https://arxiv.org/abs/2508.01961
  """

  @default_rank 8
  @default_alpha 32.0
  @default_d_a1 2
  @default_d_a2 2

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:alpha, float()}
          | {:d_a1, pos_integer()}
          | {:d_a2, pos_integer()}
          | {:input_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:rank, pos_integer()}

  @doc """
  Build a standalone Kron-LoRA adapter layer.

  Computes `(alpha/r) * (A_kron kron (B1 * B2)) x` via three efficient matmuls.

  ## Options

    - `:input_size` - Input dimension (required, must be divisible by `:d_a1`)
    - `:output_size` - Output dimension (required, must be divisible by `:d_a2`)
    - `:rank` - LoRA rank for B factor compression (default: 8)
    - `:alpha` - Scaling factor (default: 32.0)
    - `:d_a1` - Kronecker factor A column dimension (default: 2)
    - `:d_a2` - Kronecker factor A row dimension (default: 2)
    - `:name` - Layer name prefix (default: "kron_lora")

  ## Returns

    An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    output_size = Keyword.fetch!(opts, :output_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    d_a1 = Keyword.get(opts, :d_a1, @default_d_a1)
    d_a2 = Keyword.get(opts, :d_a2, @default_d_a2)
    name = Keyword.get(opts, :name, "kron_lora")

    input = Axon.input("input", shape: {nil, input_size})

    kron_lora_delta(input, output_size,
      input_size: input_size,
      rank: rank,
      alpha: alpha,
      d_a1: d_a1,
      d_a2: d_a2,
      name: name
    )
  end

  @doc """
  Wrap an existing dense layer with a Kron-LoRA adapter.

  Output = original_output + (alpha/r) * (A_kron kron (B1 * B2)) x

  ## Parameters

    - `input` - The Axon input node
    - `original` - The original Axon dense layer output

  ## Options

    - `:input_size` - Input dimension (required)
    - `:output_size` - Output dimension (required)
    - `:rank` - LoRA rank (default: 8)
    - `:alpha` - Scaling factor (default: 32.0)
    - `:d_a1` - Kronecker column dim (default: 2)
    - `:d_a2` - Kronecker row dim (default: 2)
    - `:name` - Layer name prefix (default: "kron_lora")
  """
  @spec wrap(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def wrap(input, original, opts \\ []) do
    output_size = Keyword.fetch!(opts, :output_size)
    input_size = Keyword.fetch!(opts, :input_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    d_a1 = Keyword.get(opts, :d_a1, @default_d_a1)
    d_a2 = Keyword.get(opts, :d_a2, @default_d_a2)
    name = Keyword.get(opts, :name, "kron_lora")

    delta =
      kron_lora_delta(input, output_size,
        input_size: input_size,
        rank: rank,
        alpha: alpha,
        d_a1: d_a1,
        d_a2: d_a2,
        name: name
      )

    Axon.add(original, delta, name: "#{name}_adapted")
  end

  @doc """
  Build a Kron-LoRA delta: the Kronecker-factored low-rank component.

  ## Parameters

    - `input` - Axon input node
    - `output_size` - Target output dimension

  ## Options

    - `:input_size` - Input dimension (required)
    - `:rank` - LoRA rank (default: 8)
    - `:alpha` - Scaling factor (default: 32.0)
    - `:d_a1` - Kronecker column dim (default: 2)
    - `:d_a2` - Kronecker row dim (default: 2)
    - `:name` - Layer name prefix
  """
  @spec kron_lora_delta(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def kron_lora_delta(input, output_size, opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    rank = Keyword.get(opts, :rank, @default_rank)
    alpha = Keyword.get(opts, :alpha, @default_alpha)
    d_a1 = Keyword.get(opts, :d_a1, @default_d_a1)
    d_a2 = Keyword.get(opts, :d_a2, @default_d_a2)
    name = Keyword.get(opts, :name, "kron_lora")
    scale = alpha / rank

    d_b1 = div(input_size, d_a1)
    d_b2 = div(output_size, d_a2)

    if rem(input_size, d_a1) != 0 do
      raise ArgumentError,
            "input_size (#{input_size}) must be divisible by d_a1 (#{d_a1})"
    end

    if rem(output_size, d_a2) != 0 do
      raise ArgumentError,
            "output_size (#{output_size}) must be divisible by d_a2 (#{d_a2})"
    end

    # A_kron: small Kronecker factor {d_A2, d_A1} - Kaiming init
    a_kron =
      Axon.param("#{name}_A_kron", {d_a2, d_a1}, initializer: Axon.Initializers.lecun_normal())

    # B2: LoRA down-project {r, d_B1} - normal init
    b2 =
      Axon.param("#{name}_B2", {rank, d_b1}, initializer: Axon.Initializers.normal(scale: 0.02))

    # B1: LoRA up-project {d_B2, r} - zero init for clean start
    b1 =
      Axon.param("#{name}_B1", {d_b2, rank}, initializer: Axon.Initializers.zeros())

    Axon.layer(
      &kron_lora_forward/5,
      [input, a_kron, b1, b2],
      name: "#{name}_compute",
      d_a1: d_a1,
      d_b1: d_b1,
      d_a2: d_a2,
      d_b2: d_b2,
      scale: scale,
      op_name: :kron_lora
    )
  end

  @doc "Get the output size of a Kron-LoRA layer."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :output_size)
  end

  @doc "Recommended default configuration."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      rank: 8,
      alpha: 32.0,
      d_a1: 2,
      d_a2: 2
    ]
  end

  # Kron-LoRA forward: three-step matmul avoiding full delta_W materialization
  defp kron_lora_forward(input, a_kron, b1, b2, opts) do
    d_a1 = opts[:d_a1]
    d_b1 = opts[:d_b1]
    d_a2 = opts[:d_a2]
    d_b2 = opts[:d_b2]
    scale = opts[:scale]

    # Reshape: {batch, d_in} -> {batch, d_A1, d_B1}
    x = Nx.reshape(input, {:auto, d_a1, d_b1})

    # Step 1: LoRA down-project
    # x: {batch, d_A1, d_B1}, b2: {r, d_B1}
    # Contract d_B1 -> {batch, d_A1, r}
    y1 = Nx.dot(x, [2], b2, [1])

    # Step 2: Kronecker mixing
    # y1: {batch, d_A1, r}, a_kron: {d_A2, d_A1}
    # Contract d_A1 -> {batch, r, d_A2}
    y2 = Nx.dot(y1, [1], a_kron, [1])

    # Step 3: LoRA up-project
    # y2: {batch, r, d_A2}, b1: {d_B2, r}
    # Contract r -> {batch, d_A2, d_B2}
    y3 = Nx.dot(y2, [1], b1, [1])

    # Reshape back: {batch, d_A2 * d_B2} = {batch, d_out}
    out = Nx.reshape(y3, {:auto, d_a2 * d_b2})

    # Scale by alpha/rank
    Nx.multiply(out, scale)
  end
end
