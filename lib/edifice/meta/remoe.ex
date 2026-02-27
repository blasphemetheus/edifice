defmodule Edifice.Meta.ReMoE do
  @moduledoc """
  ReMoE: Fully Differentiable Mixture of Experts with ReLU Routing.

  Replaces the standard TopK + Softmax MoE router with a simple ReLU gate.
  Since ReLU naturally produces sparse outputs (zero for negative inputs),
  it acts as a differentiable sparse gate without the discontinuous TopK
  operation. Sparsity is controlled via adaptive L1 regularization.

  ## Key Innovation: ReLU Routing

  Standard TopK routing is discontinuous — when scores shift between experts,
  the output changes abruptly (e.g., scores (0.51, 0.49) -> weights (0.51, 0)
  creates a jump). ReLU routing is fully differentiable: the transition between
  zero and non-zero is smooth.

  ```
  TopK Router:   softmax(W * x) -> top_k -> sparse weights
  ReMoE Router:  ReLU(W * x)                -> naturally sparse weights
  ```

  ## Architecture

  ```
  Input x: {batch, seq_len, d}
       |
  Router: R(x) = ReLU(W_r * x)  -> {batch, seq_len, E}  (non-negative routing)
       |
  For each expert e where R(x)_e > 0:
       |
  FFN_e(x) * R(x)_e
       |
  Sum over active experts
       |
  Output: {batch, seq_len, d}
  ```

  ## Sparsity Control

  Adaptive L1 regularization drives sparsity toward a target level:

  ```
  L_reg = lambda * mean(||R(x)||_1)
  lambda <- lambda * alpha^{sign(target_sparsity - measured_sparsity)}
  ```

  ## Usage

      model = ReMoE.build(
        input_size: 256,
        num_experts: 8,
        target_active: 2,
        hidden_size: 1024
      )

  ## References

  - Wang et al., "ReMoE: Fully Differentiable Mixture-of-Experts with ReLU
    Routing" (ICLR 2025). arXiv:2412.14711
  """

  import Nx.Defn

  @default_num_experts 8
  @default_target_active 2
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:num_experts, pos_integer()}
          | {:target_active, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build a ReMoE layer.

  ## Options

    - `:input_size` - Input dimension (required)
    - `:hidden_size` - Expert FFN hidden dimension (default: input_size * 4)
    - `:output_size` - Output dimension (default: input_size)
    - `:num_experts` - Number of expert networks (default: 8)
    - `:target_active` - Target active experts per token (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    An Axon model for the ReMoE layer. Input shape: `{batch, seq_len, input_size}`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, input_size * 4)
    output_size = Keyword.get(opts, :output_size, input_size)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "remoe")

    input = Axon.input("remoe_input", shape: {nil, nil, input_size})

    # ReLU router: R(x) = ReLU(W_r * x) -> {batch, seq_len, num_experts}
    router_weights =
      input
      |> Axon.dense(num_experts, name: "#{name}_router")
      |> Axon.relu(name: "#{name}_router_relu")

    # Build expert FFNs
    experts =
      for i <- 0..(num_experts - 1) do
        build_expert(input, hidden_size, output_size, dropout, "#{name}_expert_#{i}")
      end

    # Stack experts: [num_experts, batch, seq_len, output_size]
    experts_stacked =
      Axon.layer(
        build_stack_fn(num_experts),
        experts,
        name: "#{name}_stack_experts",
        op_name: :stack_experts
      )

    # Weighted combination using ReLU routing weights
    Axon.layer(
      &relu_routing_forward/3,
      [router_weights, experts_stacked],
      name: "#{name}_combine",
      op_name: :relu_routing
    )
  end

  @doc """
  Compute L1 sparsity regularization loss.

  Encourages the ReLU router to produce sparse outputs. Use with adaptive
  lambda to hit a target sparsity level.

  ```
  L_reg = mean(||R(x)||_1)
  ```

  ## Parameters

    - `router_weights` - ReLU router output `{batch, seq_len, num_experts}`

  ## Returns

    Scalar regularization loss (multiply by lambda externally).
  """
  @spec sparsity_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn sparsity_loss(router_weights) do
    # L1 norm averaged over batch and sequence
    Nx.mean(Nx.sum(router_weights, axes: [-1]))
  end

  @doc """
  Compute load-balanced sparsity regularization loss.

  Upweights regularization for experts receiving disproportionately many tokens,
  encouraging more uniform expert utilization.

  ```
  f_e = (E / (k * T)) * count(R(x)_e > 0)
  L_reg = mean(sum_e f_e * R(x)_e)
  ```

  ## Parameters

    - `router_weights` - ReLU router output `{batch, seq_len, num_experts}`
    - `target_active` - Target number of active experts per token
  """
  @spec balanced_sparsity_loss(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  defn balanced_sparsity_loss(router_weights, target_active) do
    num_experts = Nx.axis_size(router_weights, 2)
    # Flatten batch and seq: {B*S, E}
    flat = Nx.reshape(router_weights, {:auto, num_experts})
    total_tokens = Nx.axis_size(flat, 0)

    # Count tokens per expert
    active_mask = Nx.greater(flat, 0.0)
    tokens_per_expert = Nx.sum(active_mask, axes: [0])

    # Load factor: E / (k * T) * count
    load_factor =
      Nx.divide(
        Nx.multiply(num_experts, tokens_per_expert),
        Nx.multiply(target_active, total_tokens)
      )

    # Weighted L1: sum_e f_e * R(x)_e, averaged over tokens
    weighted = Nx.multiply(flat, load_factor)
    Nx.mean(Nx.sum(weighted, axes: [-1]))
  end

  @doc """
  Update the adaptive regularization coefficient lambda.

  Increases lambda when sparsity is below target, decreases when above.

  ```
  lambda_new = lambda * alpha^{sign(target_sparsity - measured_sparsity)}
  ```

  ## Parameters

    - `lambda` - Current regularization coefficient
    - `router_weights` - ReLU router output (to measure sparsity)
    - `opts` - Options:
      - `:target_active` - Target active experts (default: 2)
      - `:alpha` - Update multiplier (default: 1.2)

  ## Returns

    Updated lambda value.
  """
  @spec update_lambda(float(), Nx.Tensor.t(), keyword()) :: float()
  def update_lambda(lambda, router_weights, opts \\ []) do
    num_experts = Nx.axis_size(router_weights, 2)
    target_active = Keyword.get(opts, :target_active, @default_target_active)
    alpha = Keyword.get(opts, :alpha, 1.2)

    target_sparsity = 1.0 - target_active / num_experts

    # Measure actual sparsity: fraction of zero entries
    flat = Nx.reshape(router_weights, {:auto, num_experts})
    zero_count = Nx.sum(Nx.less_equal(flat, 0.0)) |> Nx.to_number()
    total = Nx.size(flat) |> Nx.to_number()
    measured_sparsity = zero_count / total

    # Update: if too dense (sparsity too low), increase lambda
    sign = if target_sparsity > measured_sparsity, do: 1.0, else: -1.0
    lambda * :math.pow(alpha, sign)
  end

  @doc "Get the output size of a ReMoE layer."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    Keyword.get(opts, :output_size, input_size)
  end

  # ===========================================================================
  # Private
  # ===========================================================================

  defp build_expert(input, hidden_size, output_size, dropout, name) do
    x =
      input
      |> Axon.dense(hidden_size, name: "#{name}_up")
      |> Axon.activation(:silu, name: "#{name}_silu")
      |> Axon.dense(output_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
    else
      x
    end
  end

  # Build a stack function for the given number of experts.
  # Axon.layer unpacks list elements as positional args, so the function
  # arity must be num_experts + 1 (for the opts keyword).
  defp build_stack_fn(num_experts) do
    case num_experts do
      2 -> fn a, b, _opts -> Nx.stack([a, b]) end
      3 -> fn a, b, c, _opts -> Nx.stack([a, b, c]) end
      4 -> fn a, b, c, d, _opts -> Nx.stack([a, b, c, d]) end
      6 -> fn a, b, c, d, e, f, _opts -> Nx.stack([a, b, c, d, e, f]) end
      8 -> fn a, b, c, d, e, f, g, h, _opts -> Nx.stack([a, b, c, d, e, f, g, h]) end
    end
  end

  # ReLU routing: weighted sum of expert outputs by ReLU gate values
  defnp relu_routing_forward(router_weights, experts_stacked, _opts) do
    # router_weights: {batch, seq_len, num_experts} (non-negative from ReLU)
    # experts_stacked: {num_experts, batch, seq_len, output_size}

    # Transpose experts to {batch, seq_len, num_experts, output_size}
    experts_t = Nx.transpose(experts_stacked, axes: [1, 2, 0, 3])

    # Weighted combination: {batch, seq_len, 1, num_experts} @ {batch, seq_len, num_experts, output_size}
    weights = Nx.new_axis(router_weights, 2)
    output = Nx.dot(weights, [3], [0, 1], experts_t, [2], [0, 1])
    Nx.squeeze(output, axes: [2])
  end
end
