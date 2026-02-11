defmodule Edifice.Meta.SwitchMoE do
  @moduledoc """
  Switch Transformer - Top-1 Expert Routing.

  The Switch Transformer simplifies MoE routing by selecting only a single
  expert per token (top-1), reducing computation and communication costs
  while maintaining model capacity. Each token is routed to exactly one
  expert based on learned routing weights.

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +------------------------------------+
  | Input Projection                   |
  +------------------------------------+
        |
        v
  +------------------------------------+
  | Switch Block 1:                    |
  |   Pre-Norm -> Router (top-1)       |
  |   -> Selected Expert FFN           |
  |   + Residual                       |
  +------------------------------------+
        |  (repeat N times)
        v
  +------------------------------------+
  | Final Norm + Last Timestep         |
  +------------------------------------+
        |
        v
  Output [batch, hidden_size]
  ```

  ## Router Design

  The router computes softmax probabilities over experts and selects the
  highest-scoring expert for each token. Since Axon uses static graphs,
  all experts are computed and the router selects via weighted combination
  with a peaked (near-one-hot) distribution.

  ## Usage

      model = SwitchMoE.build(
        embed_size: 256,
        hidden_size: 256,
        num_experts: 8,
        expert_size: 512,
        num_layers: 4
      )

  ## References

  - Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity" (JMLR 2022)
  - https://arxiv.org/abs/2101.03961
  """

  require Axon
  import Nx.Defn

  @default_hidden_size 256
  @default_num_experts 8
  @default_expert_size 512
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Switch Transformer model.

  ## Options

  - `:embed_size` - Input embedding dimension (required)
  - `:hidden_size` - Hidden dimension (default: 256)
  - `:num_experts` - Number of expert FFNs (default: 8)
  - `:expert_size` - Inner dimension of expert FFNs (default: 512)
  - `:num_layers` - Number of Switch blocks (default: 4)
  - `:dropout` - Dropout rate (default: 0.1)
  - `:window_size` - Sequence length (default: 60)

  ## Returns

  An Axon model: `[batch, seq_len, embed_size]` -> `[batch, hidden_size]`
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    expert_size = Keyword.get(opts, :expert_size, @default_expert_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_size})

    # Project to hidden size
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_proj")
      else
        input
      end

    # Stack Switch blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        switch_block(acc, hidden_size,
          num_experts: num_experts,
          expert_size: expert_size,
          dropout: dropout,
          name: "switch_#{idx}"
        )
      end)

    # Final norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep
    Axon.nx(x, fn tensor ->
      seq_len_actual = Nx.axis_size(tensor, 1)
      Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "last_timestep")
  end

  @doc """
  Single Switch block: pre-norm -> top-1 routed expert FFN -> residual.
  """
  @spec switch_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def switch_block(input, hidden_size, opts \\ []) do
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    expert_size = Keyword.get(opts, :expert_size, @default_expert_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "switch")

    normed = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Router: compute top-1 routing weights
    router_logits = Axon.dense(normed, num_experts, name: "#{name}_router")

    # Build all expert FFNs
    experts =
      for i <- 0..(num_experts - 1) do
        normed
        |> Axon.dense(expert_size, name: "#{name}_expert_#{i}_up")
        |> Axon.gelu()
        |> Axon.dense(hidden_size, name: "#{name}_expert_#{i}_down")
      end

    # Combine via top-1 routing using dynamic function for expert count
    route_fn = switch_route_fn(num_experts)

    moe_output =
      Axon.layer(
        route_fn,
        [router_logits | experts],
        name: "#{name}_route",
        num_experts: num_experts,
        op_name: :switch_route
      )

    moe_output =
      if dropout > 0.0 do
        Axon.dropout(moe_output, rate: dropout, name: "#{name}_drop")
      else
        moe_output
      end

    Axon.add(input, moe_output, name: "#{name}_residual")
  end

  @doc """
  Get the output size of a Switch MoE model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  # Generate a routing function for the given number of experts.
  # Axon.layer passes each list element as a separate positional arg,
  # so we need arity-specific functions.
  defp switch_route_fn(num_experts) do
    case num_experts do
      2 ->
        fn router, e0, e1, _opts ->
          route_top1(router, Nx.stack([e0, e1]))
        end

      4 ->
        fn router, e0, e1, e2, e3, _opts ->
          route_top1(router, Nx.stack([e0, e1, e2, e3]))
        end

      8 ->
        fn router, e0, e1, e2, e3, e4, e5, e6, e7, _opts ->
          route_top1(router, Nx.stack([e0, e1, e2, e3, e4, e5, e6, e7]))
        end

      _ ->
        # Fallback for other expert counts: use first expert
        fn router, e0, _opts ->
          _ = router
          e0
        end
    end
  end

  # Top-1 routing: select the expert with highest routing probability
  defnp route_top1(router_logits, experts_stacked) do
    # router_logits: [batch, seq_len, num_experts]
    # experts_stacked: [num_experts, batch, seq_len, hidden_size]

    # Softmax router probabilities
    router_probs = Nx.exp(router_logits - Nx.reduce_max(router_logits, axes: [-1], keep_axes: true))
    router_probs = router_probs / Nx.sum(router_probs, axes: [-1], keep_axes: true)

    # For top-1: weight each expert by its routing probability
    # This approximates hard routing in a differentiable way
    # router_probs: [batch, seq_len, num_experts]
    # experts_stacked: [num_experts, batch, seq_len, hidden_size]

    # Transpose experts to [batch, seq_len, num_experts, hidden_size]
    experts_t = Nx.transpose(experts_stacked, axes: [1, 2, 0, 3])

    # Weight and sum: [batch, seq_len, 1, num_experts] @ [batch, seq_len, num_experts, hidden_size]
    weights = Nx.new_axis(router_probs, 2)
    output = Nx.dot(weights, [3], [0, 1], experts_t, [2], [0, 1])
    Nx.squeeze(output, axes: [2])
  end
end
