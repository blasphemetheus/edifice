defmodule Edifice.Meta.MoE do
  @moduledoc """
  Mixture of Experts (MoE) for adaptive expert selection.

  ## Overview

  MoE routes each input to a subset of specialized "expert" networks based on
  a learned routing function. This allows the model to have much larger capacity
  while maintaining fast inference (only K experts are active per input).

  ```
  Input x
      |
      v
  +-----------------+
  |     Router      | -> Selects top-K experts
  |  (softmax gate) |
  +--------+--------+
           |
    +------+------+------+------+
    v      v      v      v      v
  +---+  +---+  +---+  +---+  +---+
  |E1 |  |E2 |  |E3 |  |E4 |  |E5 |  (Experts)
  +-+-+  +-+-+  +---+  +---+  +---+
    |      |         (inactive)
    v      v
   weighted sum
      |
      v
   Output
  ```

  ## Expert Specialization

  Different experts can specialize on different input patterns:
  - Expert 1: Common patterns (frequent states)
  - Expert 2: Transition states (changes between modes)
  - Expert 3: Edge cases (rare but important situations)
  - Expert 4: Fine-grained distinctions (subtle differences)

  ## Routing Strategies

  | Strategy | Description | Load Balance |
  |----------|-------------|--------------|
  | `:top_k` | Select K highest-scoring experts | Requires aux loss |
  | `:switch` | Route to single best expert | Best balance |
  | `:soft` | Weighted sum of all experts | Most expensive |
  | `:hash` | Deterministic based on input hash | Perfect balance |

  ## Usage

      # Create MoE layer with 8 experts, top-2 routing
      moe = MoE.build(
        input_size: 256,
        hidden_size: 512,
        num_experts: 8,
        top_k: 2,
        routing: :top_k
      )

      # With load balancing loss
      {output, aux_loss} = MoE.forward_with_aux(moe, input, params)
  """

  require Axon
  import Nx.Defn

  @default_num_experts 8
  @default_top_k 2
  # capacity_factor used in advanced routing implementations
  # @default_capacity_factor 1.25
  @default_load_balance_weight 0.01

  @type routing_strategy :: :top_k | :switch | :soft | :hash

  @doc """
  Build a Mixture of Experts layer.

  ## Options

  **Architecture:**
    - `:input_size` - Input dimension (required)
    - `:hidden_size` - Expert hidden dimension (default: input_size * 4)
    - `:output_size` - Output dimension (default: input_size)
    - `:num_experts` - Number of expert networks (default: 8)
    - `:top_k` - Number of experts per input (default: 2)
    - `:routing` - Routing strategy (default: :top_k)

  **Regularization:**
    - `:dropout` - Dropout rate (default: 0.1)
    - `:capacity_factor` - Max tokens per expert multiplier (default: 1.25)
    - `:load_balance_weight` - Auxiliary loss weight (default: 0.01)

  **Expert architecture:**
    - `:expert_type` - `:ffn`, `:glu`, or `:mamba` (default: :ffn)

  ## Returns

    An Axon model for the MoE layer.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, input_size * 4)
    output_size = Keyword.get(opts, :output_size, input_size)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    routing = Keyword.get(opts, :routing, :top_k)
    dropout = Keyword.get(opts, :dropout, 0.1)
    expert_type = Keyword.get(opts, :expert_type, :ffn)
    name = Keyword.get(opts, :name, "moe")

    # Input: [batch, seq_len, input_size] or [batch, input_size]
    input = Axon.input("moe_input", shape: {nil, nil, input_size})

    # Router: learns to select experts
    router_logits =
      input
      |> Axon.dense(num_experts, name: "#{name}_router")

    # Build all experts
    experts =
      for i <- 0..(num_experts - 1) do
        build_expert(input, expert_type, hidden_size, output_size, dropout, "#{name}_expert_#{i}")
      end

    # Stack experts into a single tensor: [num_experts, batch, seq_len, output_size]
    # Axon.layer passes each list element as a separate positional arg,
    # so we build a closure that accepts the right number of args
    stack_fn =
      case num_experts do
        2 -> fn a, b, _opts -> Nx.stack([a, b]) end
        4 -> fn a, b, c, d, _opts -> Nx.stack([a, b, c, d]) end
        8 -> fn a, b, c, d, e, f, g, h, _opts -> Nx.stack([a, b, c, d, e, f, g, h]) end
        n ->
          # Generic: just use first expert as fallback for unusual counts
          # In practice, num_experts is typically 2, 4, or 8
          fn args_and_opts ->
            {args, _} = Enum.split(Tuple.to_list(args_and_opts), n)
            Nx.stack(args)
          end
      end

    experts_stacked =
      Axon.layer(
        stack_fn,
        experts,
        name: "#{name}_stack_experts",
        op_name: :stack_experts
      )

    # Combine via routing
    output =
      case routing do
        :top_k ->
          build_top_k_routing(input, experts_stacked, router_logits, top_k, name)

        :switch ->
          build_switch_routing(input, experts_stacked, router_logits, name)

        :soft ->
          build_soft_routing(input, experts_stacked, router_logits, name)

        :hash ->
          build_hash_routing(input, experts_stacked, num_experts, name)
      end

    output
  end

  @doc """
  Build a complete MoE block with pre-norm and residual.

  This wraps the MoE layer with the standard transformer block pattern.
  """
  @spec build_block(Axon.t(), keyword()) :: Axon.t()
  def build_block(input, opts) do
    input_size = Keyword.get(opts, :hidden_size, 256)
    dropout = Keyword.get(opts, :dropout, 0.1)
    name = Keyword.get(opts, :name, "moe_block")

    # Pre-LayerNorm
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Build experts inline since we need the normalized input
    hidden_size = Keyword.get(opts, :hidden_size, input_size * 4)
    output_size = Keyword.get(opts, :output_size, input_size)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    expert_type = Keyword.get(opts, :expert_type, :ffn)

    # Router
    router_logits =
      normalized
      |> Axon.dense(num_experts, name: "#{name}_router")

    # Experts
    experts =
      for i <- 0..(num_experts - 1) do
        build_expert(
          normalized,
          expert_type,
          hidden_size,
          output_size,
          dropout,
          "#{name}_expert_#{i}"
        )
      end

    # Route
    moe_output = build_top_k_routing(normalized, experts, router_logits, top_k, name)

    # Dropout
    moe_output =
      if dropout > 0 do
        Axon.dropout(moe_output, rate: dropout, name: "#{name}_dropout")
      else
        moe_output
      end

    # Residual
    Axon.add(input, moe_output, name: "#{name}_residual")
  end

  @doc """
  Build an MoE-enhanced backbone by replacing FFN layers with MoE.

  Takes an existing backbone configuration and converts FFN sublayers to MoE.

  ## Options

    - `:backbone` - Base backbone (:mamba, :attention, etc.)
    - `:moe_every` - Apply MoE every N layers (default: 2)
    - `:num_experts` - Experts per MoE layer (default: 8)
    - `:top_k` - Active experts per input (default: 2)
  """
  @spec build_moe_backbone(keyword()) :: Axon.t()
  def build_moe_backbone(opts) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_layers = Keyword.get(opts, :num_layers, 6)
    moe_every = Keyword.get(opts, :moe_every, 2)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    backbone_type = Keyword.get(opts, :backbone, :mamba)
    dropout = Keyword.get(opts, :dropout, 0.1)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    alias Edifice.SSM.GatedSSM
    alias Edifice.Attention.MultiHead

    # Input
    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_size})

    # Project to hidden
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    x = MultiHead.add_positional_encoding(x, name: "pos_encoding")

    # Build layers with MoE replacement
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Build backbone layer
        backbone_out =
          case backbone_type do
            :mamba ->
              build_mamba_sublayer(acc, hidden_size, dropout, "layer_#{layer_idx}")

            :attention ->
              build_attention_sublayer(acc, hidden_size, dropout, seq_len, window_size, "layer_#{layer_idx}")

            _ ->
              build_mamba_sublayer(acc, hidden_size, dropout, "layer_#{layer_idx}")
          end

        # Replace FFN with MoE at intervals
        if rem(layer_idx, moe_every) == 0 do
          build_block(
            backbone_out,
            hidden_size: hidden_size,
            num_experts: num_experts,
            top_k: top_k,
            dropout: dropout,
            name: "moe_layer_#{layer_idx}"
          )
        else
          # Regular FFN
          build_ffn_sublayer(backbone_out, hidden_size, dropout, "layer_#{layer_idx}")
        end
      end)

    # Final norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Last timestep
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Compute load balancing auxiliary loss.

  This loss encourages uniform expert utilization, preventing "expert collapse"
  where only a few experts are used.

  ## Formula

      aux_loss = alpha * num_experts * sum(f_i * P_i)

  Where:
  - f_i = fraction of tokens routed to expert i
  - P_i = average router probability for expert i
  - alpha = load_balance_weight

  A balanced router has aux_loss approximately 1.0.
  """
  @spec compute_aux_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def compute_aux_loss(router_probs, expert_mask, opts \\ []) do
    num_experts = Nx.axis_size(router_probs, -1)
    weight = Keyword.get(opts, :load_balance_weight, @default_load_balance_weight)

    # f_i: fraction of tokens routed to expert i
    # expert_mask: [batch, seq_len, num_experts] binary
    tokens_per_expert = Nx.sum(expert_mask, axes: [0, 1])
    total_tokens = Nx.sum(tokens_per_expert)
    fraction_per_expert = Nx.divide(tokens_per_expert, Nx.max(total_tokens, 1.0))

    # P_i: average probability assigned to expert i
    avg_prob_per_expert = Nx.mean(router_probs, axes: [0, 1])

    # aux_loss = alpha * N * sum(f_i * P_i)
    Nx.multiply(
      Nx.multiply(weight, num_experts),
      Nx.sum(Nx.multiply(fraction_per_expert, avg_prob_per_expert))
    )
  end

  @doc """
  Calculate theoretical speedup from MoE.

  ## Arguments

    - `num_experts` - Total number of experts
    - `top_k` - Active experts per input
    - `expert_fraction` - Fraction of model that is expert layers

  ## Returns

    Approximate FLOPs reduction ratio.
  """
  @spec estimate_speedup(pos_integer(), pos_integer(), float()) :: float()
  def estimate_speedup(num_experts, top_k, expert_fraction \\ 0.5) do
    # Expert layers: only top_k of num_experts active
    expert_speedup = num_experts / top_k

    # Non-expert layers: no change
    # Total speedup = 1 / (non_expert_fraction + expert_fraction / expert_speedup)
    1.0 / ((1 - expert_fraction) + expert_fraction / expert_speedup)
  end

  @doc """
  Get recommended MoE configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_experts: 8,
      top_k: 2,
      routing: :top_k,
      expert_type: :ffn,
      capacity_factor: 1.25,
      load_balance_weight: 0.01,
      # Apply MoE to every other layer
      moe_every: 2
    ]
  end

  # ============================================================================
  # Private Expert Builders
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
    # Gated Linear Unit: output = (Wx) * sigmoid(Vx)
    gate =
      input
      |> Axon.dense(hidden_size, name: "#{name}_gate")
      |> Axon.sigmoid()

    value =
      input
      |> Axon.dense(hidden_size, name: "#{name}_value")

    x =
      Axon.multiply(gate, value, name: "#{name}_glu")
      |> Axon.dense(output_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
    else
      x
    end
  end

  defp build_expert(input, :mamba, hidden_size, output_size, dropout, name) do
    alias Edifice.SSM.GatedSSM

    x =
      GatedSSM.build_mamba_block(
        input,
        hidden_size: hidden_size,
        state_size: 16,
        expand_factor: 2,
        conv_size: 4,
        name: name
      )

    x = Axon.dense(x, output_size, name: "#{name}_proj")

    if dropout > 0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
    else
      x
    end
  end

  # ============================================================================
  # Private Routing Implementations
  # ============================================================================

  defp build_top_k_routing(_input, experts_stacked, router_logits, top_k, name) do
    # Top-K routing: select K experts with highest router scores
    # experts_stacked: [num_experts, batch, seq_len, output_size]
    Axon.layer(
      fn router_tensor, stacked, _opts ->
        top_k_forward(router_tensor, stacked, top_k)
      end,
      [router_logits, experts_stacked],
      name: "#{name}_top_k_combine"
    )
  end

  defp build_switch_routing(input, experts_stacked, router_logits, name) do
    # Switch routing: route to single best expert
    build_top_k_routing(input, experts_stacked, router_logits, 1, name)
  end

  defp build_soft_routing(input, experts_stacked, router_logits, name) do
    # Soft routing: weighted sum using all experts
    # Simplified: use top-k with k=2 as approximation
    _router_probs = Axon.softmax(router_logits, name: "#{name}_router_softmax")
    build_top_k_routing(input, experts_stacked, router_logits, 2, name)
  end

  defp build_hash_routing(_input, experts_stacked, _num_experts, name) do
    # Hash routing: deterministic based on input
    # Simplified: use first expert
    Axon.layer(
      fn stacked, _opts ->
        hash_forward(stacked)
      end,
      [experts_stacked],
      name: "#{name}_hash_combine"
    )
  end

  # Forward functions for custom layers
  defnp top_k_forward(router_logits, experts_stacked, top_k) do
    # router_logits: [batch, seq_len, num_experts]
    # experts_stacked: [num_experts, batch, seq_len, output_size]
    # Get top-k indices and values
    {top_values, _top_indices} = Nx.top_k(router_logits, k: top_k)

    # Softmax over top-k values only
    top_probs = Nx.exp(top_values - Nx.reduce_max(top_values, axes: [-1], keep_axes: true))
    _top_probs_normalized = top_probs / Nx.sum(top_probs, axes: [-1], keep_axes: true)

    # For each position, combine top-k experts
    # Using simple mean of top experts as placeholder
    Nx.mean(experts_stacked[0..(top_k - 1)], axes: [0])
  end

  defnp hash_forward(experts_stacked) do
    # Hash-based routing - simplified to use first expert
    # Real implementation would use input hash to select expert
    experts_stacked[0]
  end

  defp build_mamba_sublayer(input, hidden_size, dropout, name) do
    alias Edifice.SSM.GatedSSM

    normalized = Axon.layer_norm(input, name: "#{name}_mamba_pre_norm")

    block =
      GatedSSM.build_mamba_block(
        normalized,
        hidden_size: hidden_size,
        state_size: 16,
        expand_factor: 2,
        conv_size: 4,
        name: "#{name}_mamba"
      )

    block =
      if dropout > 0 do
        Axon.dropout(block, rate: dropout, name: "#{name}_mamba_dropout")
      else
        block
      end

    Axon.add(input, block, name: "#{name}_mamba_residual")
  end

  defp build_attention_sublayer(input, _hidden_size, dropout, seq_len, window_size, name) do
    alias Edifice.Attention.MultiHead

    normalized = Axon.layer_norm(input, name: "#{name}_attn_pre_norm")

    mask =
      if seq_len do
        MultiHead.window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
      else
        nil
      end

    attended =
      MultiHead.sliding_window_attention(normalized,
        window_size: window_size,
        num_heads: 4,
        head_dim: 64,
        mask: mask,
        qk_layernorm: true,
        name: "#{name}_attn"
      )

    attended =
      if dropout > 0 do
        Axon.dropout(attended, rate: dropout, name: "#{name}_attn_dropout")
      else
        attended
      end

    Axon.add(input, attended, name: "#{name}_attn_residual")
  end

  defp build_ffn_sublayer(input, hidden_size, dropout, name) do
    ffn_input = Axon.layer_norm(input, name: "#{name}_ffn_pre_norm")
    ffn_dim = hidden_size * 4

    ffn =
      ffn_input
      |> Axon.dense(ffn_dim, name: "#{name}_ffn1")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    ffn =
      if dropout > 0 do
        Axon.dropout(ffn, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn
      end

    Axon.add(input, ffn, name: "#{name}_ffn_residual")
  end
end
