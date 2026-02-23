defmodule Edifice.Meta.MoEv2 do
  @moduledoc """
  MoE v2: Expert Choice Routing + Shared Experts.

  Implements two key improvements to the Mixture of Experts architecture:
  expert choice routing (Zhou et al., 2022) and shared expert slots
  (DeepSeekMoE).

  ## Key Innovations

  ### 1. Expert Choice Routing

  Standard MoE: each **token** picks its top-K experts.
  Expert Choice: each **expert** picks its top-C tokens.

  ```
  Standard:  token -> selects experts   (load imbalance risk)
  Expert:    expert -> selects tokens   (perfect load balance)
  ```

  This eliminates the need for load balancing auxiliary loss since
  each expert processes exactly C tokens by construction.

  ### 2. Shared Expert Slots

  Some experts are "shared" (always active for every token), while
  others are "routed" (selected by expert choice). This ensures a
  base level of computation for every token while allowing
  specialization:

  ```
  Output = SharedExperts(x) + RoutedExperts(x)
  ```

  ## Architecture

  ```
  Input [batch, seq_len, input_size]
        |
        v
  +----------------------------------+
  |     Router (transposed scores)   |
  |  score = softmax(W_r * x)^T     |
  |  Each expert picks top-C tokens  |
  +----------------------------------+
        |
  +-----+-----+
  |             |
  v             v
  Shared      Routed
  Experts     Experts
  (always     (expert
  active)     choice)
  |             |
  +-----+-----+
        |
        v
  Output = shared + routed
  ```

  ## Usage

      model = MoEv2.build(
        input_size: 256,
        hidden_size: 512,
        num_shared_experts: 1,
        num_routed_experts: 4,
        tokens_per_expert: 4
      )

  ## References

  - Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (NeurIPS 2022)
  - DeepSeek-AI, "DeepSeekMoE: Towards Ultimate Expert Specialization" (2024)
  """

  import Nx.Defn

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:output_size, pos_integer()}
          | {:num_shared_experts, pos_integer()}
          | {:num_routed_experts, pos_integer()}
          | {:tokens_per_expert, pos_integer()}
          | {:dropout, float()}
          | {:expert_type, :ffn | :glu}
          | {:load_balance, :bias | :none}

  @default_num_shared 1
  @default_num_routed 4
  @default_tokens_per_expert 4

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a MoE v2 layer with expert choice routing and shared experts.

  ## Options

  - `:input_size` - Input dimension (required)
  - `:hidden_size` - Expert hidden dimension (default: input_size * 4)
  - `:output_size` - Output dimension (default: input_size)
  - `:num_shared_experts` - Always-active experts (default: 1)
  - `:num_routed_experts` - Expert-choice routed experts (default: 4)
  - `:tokens_per_expert` - Tokens each routed expert selects (default: 4)
  - `:dropout` - Dropout rate (default: 0.1)
  - `:expert_type` - `:ffn` or `:glu` (default: :ffn)
  - `:load_balance` - Load balancing strategy (default: `:none`)
    - `:none` — no load balancing (standard expert choice)
    - `:bias` — aux-loss-free bias term added to router logits before softmax.
      At training time, use `update_load_balance_bias/3` to adjust the bias
      based on expert utilization, eliminating the need for auxiliary loss.

  ## Returns

    An Axon model for the MoE v2 layer.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, input_size * 4)
    output_size = Keyword.get(opts, :output_size, input_size)
    num_shared = Keyword.get(opts, :num_shared_experts, @default_num_shared)
    num_routed = Keyword.get(opts, :num_routed_experts, @default_num_routed)
    tokens_per_expert = Keyword.get(opts, :tokens_per_expert, @default_tokens_per_expert)
    dropout = Keyword.get(opts, :dropout, 0.1)
    expert_type = Keyword.get(opts, :expert_type, :ffn)
    load_balance = Keyword.get(opts, :load_balance, :none)
    name = Keyword.get(opts, :name, "moe_v2")

    input = Axon.input("moe_input", shape: {nil, nil, input_size})

    # --- Shared experts (always active) ---
    shared_outputs =
      for i <- 0..(num_shared - 1) do
        build_expert(input, expert_type, hidden_size, output_size, dropout, "#{name}_shared_#{i}")
      end

    shared_sum =
      case shared_outputs do
        [single] -> single
        multiple -> Enum.reduce(multiple, fn x, acc -> Axon.add(x, acc) end)
      end

    # --- Routed experts (expert choice) ---
    # Router: projects input to num_routed scores
    router_logits = Axon.dense(input, num_routed, name: "#{name}_router")

    # Aux-loss-free load balancing: add learnable bias to router logits
    router_logits =
      if load_balance == :bias do
        Axon.bias(router_logits, name: "#{name}_load_balance_bias")
      else
        router_logits
      end

    # Build routed experts
    routed_experts =
      for i <- 0..(num_routed - 1) do
        build_expert(
          input,
          expert_type,
          hidden_size,
          output_size,
          dropout,
          "#{name}_routed_#{i}"
        )
      end

    # Stack routed experts
    stack_fn =
      case num_routed do
        2 -> fn a, b, _opts -> Nx.stack([a, b]) end
        4 -> fn a, b, c, d, _opts -> Nx.stack([a, b, c, d]) end
        8 -> fn a, b, c, d, e, f, g, h, _opts -> Nx.stack([a, b, c, d, e, f, g, h]) end
        _ -> fn a, b, _opts -> Nx.stack([a, b]) end
      end

    routed_stacked =
      Axon.layer(
        stack_fn,
        routed_experts,
        name: "#{name}_stack_routed",
        op_name: :stack_experts
      )

    # Expert choice routing: each expert picks its top-C tokens
    routed_output =
      Axon.layer(
        fn router_tensor, stacked, _opts ->
          expert_choice_forward(router_tensor, stacked, tokens_per_expert)
        end,
        [router_logits, routed_stacked],
        name: "#{name}_expert_choice",
        op_name: :expert_choice
      )

    # Combine: shared + routed
    Axon.add(shared_sum, routed_output, name: "#{name}_combine")
  end

  # ============================================================================
  # Expert Choice Routing
  # ============================================================================

  defnp expert_choice_forward(router_logits, experts_stacked, tokens_per_expert) do
    # router_logits: [batch, seq_len, num_routed]
    # experts_stacked: [num_routed, batch, seq_len, output_size]
    _batch = Nx.axis_size(router_logits, 0)
    seq_len = Nx.axis_size(router_logits, 1)
    _num_routed = Nx.axis_size(router_logits, 2)
    _output_size = Nx.axis_size(experts_stacked, 3)

    # Softmax over tokens for each expert (transposed routing)
    # scores: [batch, seq_len, num_routed] -> transpose -> [batch, num_routed, seq_len]
    scores = Nx.transpose(router_logits, axes: [0, 2, 1])

    # Softmax along token dimension for each expert
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(scores - max_scores)
    probs = exp_scores / Nx.sum(exp_scores, axes: [-1], keep_axes: true)

    # Each expert selects top-C tokens
    # Use top_k on each expert's probability distribution
    effective_c = min(tokens_per_expert, seq_len)
    {top_values, top_indices} = Nx.top_k(probs, k: effective_c)

    # Normalize top values as routing weights
    top_weights = top_values / Nx.sum(top_values, axes: [-1], keep_axes: true)

    # Build expert selection mask
    # For each expert, build a sparse assignment of tokens
    # one_hot: [batch, num_routed, C, seq_len]
    one_hot =
      Nx.equal(
        Nx.new_axis(top_indices, 3),
        Nx.iota({1, 1, 1, seq_len}, axis: 3)
      )

    # Weighted one_hot: [batch, num_routed, C, seq_len]
    weighted_mask = Nx.multiply(one_hot, Nx.new_axis(top_weights, 3))

    # Sum over C to get per-expert token weights: [batch, num_routed, seq_len]
    expert_token_weights = Nx.sum(weighted_mask, axes: [2])

    # Transpose experts: [num_routed, batch, seq_len, output_size] -> [batch, seq_len, num_routed, output_size]
    experts_t = Nx.transpose(experts_stacked, axes: [1, 2, 0, 3])

    # Weight and combine: [batch, seq_len, num_routed] @ [batch, seq_len, num_routed, output_size]
    # expert_token_weights: [batch, num_routed, seq_len] -> transpose -> [batch, seq_len, num_routed]
    weights = Nx.transpose(expert_token_weights, axes: [0, 2, 1])
    weights_expanded = Nx.new_axis(weights, 2)
    output = Nx.dot(weights_expanded, [3], [0, 1], experts_t, [2], [0, 1])
    Nx.squeeze(output, axes: [2])
  end

  # ============================================================================
  # Expert Builders
  # ============================================================================

  defp build_expert(input, :ffn, hidden_size, output_size, dropout, name) do
    x =
      input
      |> Axon.dense(hidden_size, name: "#{name}_up")
      |> Axon.gelu()
      |> Axon.dense(output_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
    else
      x
    end
  end

  defp build_expert(input, :glu, hidden_size, output_size, dropout, name) do
    gate =
      input
      |> Axon.dense(hidden_size, name: "#{name}_gate")
      |> Axon.sigmoid()

    value = Axon.dense(input, hidden_size, name: "#{name}_value")

    x =
      Axon.multiply(gate, value, name: "#{name}_glu")
      |> Axon.dense(output_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
    else
      x
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Compute expert utilization from router logits.

  Returns a tensor of shape `[num_experts]` representing the fraction of
  tokens assigned to each expert. Useful for monitoring load balance and
  for `update_load_balance_bias/3`.

  ## Parameters
    - `router_logits` - Router output tensor `[batch, seq_len, num_experts]`
    - `tokens_per_expert` - Number of tokens each expert selects

  ## Returns
    Tensor of shape `[num_experts]` with utilization ratios in [0, 1].
  """
  @spec compute_utilization(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def compute_utilization(router_logits, tokens_per_expert) do
    # Transpose to [batch, num_experts, seq_len] for expert-choice view
    scores = Nx.transpose(router_logits, axes: [0, 2, 1])

    # Softmax along token dimension
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    probs = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [-1], keep_axes: true))

    # Each expert selects top-C tokens; sum selected probabilities
    seq_len = Nx.axis_size(router_logits, 1)
    effective_c = min(tokens_per_expert, seq_len)
    {top_values, _top_indices} = Nx.top_k(probs, k: effective_c)

    # Utilization = mean weight per expert, averaged over batch
    Nx.divide(Nx.mean(Nx.sum(top_values, axes: [-1]), axes: [0]), seq_len)
  end

  @doc """
  Update the load-balance bias based on expert utilization.

  Adjusts the bias to route more tokens toward underutilized experts
  and fewer toward overutilized ones. Call this after each training step
  when using `load_balance: :bias`.

  ## Parameters
    - `params` - Model parameters (from `Axon.build`)
    - `utilization` - Expert utilization tensor from `compute_utilization/2`
    - `opts` - Options:
      - `:lr` - Bias learning rate (default: 0.001)
      - `:bias_key` - Parameter key for the bias (default: "moe_v2_load_balance_bias")

  ## Returns
    Updated model parameters.
  """
  @spec update_load_balance_bias(map(), Nx.Tensor.t(), keyword()) :: map()
  def update_load_balance_bias(params, utilization, opts \\ []) do
    lr = Keyword.get(opts, :lr, 0.001)
    bias_key = Keyword.get(opts, :bias_key, "moe_v2_load_balance_bias")

    num_experts = Nx.axis_size(utilization, 0)
    target = Nx.broadcast(Nx.tensor(1.0 / num_experts), {num_experts})

    # Gradient: push bias toward uniform utilization
    delta = Nx.multiply(lr, Nx.subtract(target, utilization))

    Map.update!(params, bias_key, fn bias_params ->
      Map.update!(bias_params, "bias", fn current_bias ->
        Nx.add(current_bias, delta)
      end)
    end)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_shared_experts: 1,
      num_routed_experts: 4,
      tokens_per_expert: 4,
      expert_type: :ffn,
      dropout: 0.1,
      load_balance: :none
    ]
  end
end
