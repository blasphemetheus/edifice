defmodule Edifice.Meta.MixtureOfExpertDepths do
  @moduledoc """
  Mixture of Expert Depths (MoED): Joint expert selection + depth routing.

  <!-- verified: true, date: 2026-02-28 -->

  Unifies dynamic width (MoE — which expert) and dynamic depth (MoD — whether
  to process or skip) into a single routing decision. A meta-controller router
  at each layer produces scores for E experts plus one "no-op" identity path.
  Tokens routed to the no-op effectively skip the layer via residual.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        v
  Projection -> [batch, seq, hidden_size]
        |
        v
  +-------------------------------------------+
  | Per Layer:                                 |
  |   Router: dense -> softmax over (E + 1)   |
  |                                            |
  |   Expert 1: SiLU FFN                       |
  |   Expert 2: SiLU FFN                       |
  |   ...                                      |
  |   Expert E: SiLU FFN                       |
  |   No-op:    identity (zero output)         |
  |                                            |
  |   Self-Attention sublayer                  |
  |   output = sum(gate_i * expert_i(x))       |
  +-------------------------------------------+
        | (repeat num_layers)
        v
  Final Norm -> Last Timestep
  Output [batch, hidden_size]
  ```

  ## Key Innovation

  The no-op expert participates in the softmax routing alongside real experts.
  When a token routes to the no-op with high probability, it effectively skips
  the layer's FFN computation. This is learned end-to-end — the model discovers
  which tokens need full processing and which can be skipped.

  ## Integrated MoDE Design

  Unlike staged approaches (MoD → MoE sequentially), the integrated design
  makes a single routing decision per token per layer. The no-op expert
  produces zero output, so tokens assigned to it only receive the attention
  sublayer computation plus residual.

  ## References

  - Raposo et al., "Mixture-of-Depths" (2024), Section 3.3 Integrated MoDE
  - https://arxiv.org/abs/2404.02258
  - Kodela, "Mixture-of-Experts-and-Depths" (2025)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_experts 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_expert_hidden_multiplier 4

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:expert_hidden_multiplier, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_experts, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Mixture of Expert Depths model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_experts` - Number of real experts, excluding no-op (default: 4)
    - `:num_layers` - Number of MoED layers (default: 4)
    - `:expert_hidden_multiplier` - FFN hidden size = hidden_size * multiplier
      (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    expert_mult = Keyword.get(opts, :expert_hidden_multiplier, @default_expert_hidden_multiplier)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)

    ModelBuilder.build_sequence_model(
      embed_dim: embed_dim,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      block_builder: fn input, block_opts ->
        layer_idx = block_opts[:layer_idx]
        name = "moed_#{layer_idx}"

        # Attention sublayer first (all tokens get attention)
        attn_out =
          TransformerBlock.layer(input,
            attention_fn: fn x, attn_name ->
              MultiHead.self_attention(x,
                hidden_size: hidden_size,
                num_heads: num_heads,
                dropout: dropout,
                causal: true,
                name: attn_name
              )
            end,
            hidden_size: hidden_size,
            dropout: dropout,
            name: "#{name}_attn"
          )

        # Router: scores for E experts + 1 no-op
        # [batch, seq, num_experts + 1] with softmax
        router_logits =
          Axon.dense(attn_out, num_experts + 1, name: "#{name}_router")

        router_weights =
          Axon.activation(router_logits, :softmax, name: "#{name}_router_softmax")

        # Build expert FFN outputs
        expert_hidden = hidden_size * expert_mult

        experts =
          for e <- 0..(num_experts - 1) do
            build_expert_ffn(attn_out, expert_hidden, hidden_size, dropout, "#{name}_expert_#{e}")
          end

        # Stack experts: use Axon.layer to combine
        # Each expert output: [batch, seq, hidden_size]
        # Route via weighted combination (soft routing)
        build_moed_combine(experts, router_weights, attn_out, num_experts, name)
      end
    )
  end

  # Build expert FFN: up -> SiLU -> down
  defp build_expert_ffn(input, hidden_size, output_size, dropout, name) do
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

  # Combine expert outputs with routing weights, including no-op path
  defp build_moed_combine(experts, router_weights, residual, num_experts, name) do
    # We need to handle variable number of experts + router_weights + residual
    # Using a 2-expert approach since Axon.layer unpacks list elements
    all_inputs = experts ++ [router_weights, residual]

    Axon.layer(
      build_combine_fn(num_experts),
      all_inputs,
      name: "#{name}_combine",
      num_experts: num_experts,
      op_name: :moed_combine
    )
  end

  # Build the combine function for the given number of experts.
  # Axon.layer unpacks list elements as positional args.
  # Args: expert_0, expert_1, ..., expert_{E-1}, router_weights, residual, opts
  defp build_combine_fn(num_experts) do
    case num_experts do
      2 ->
        fn e0, e1, weights, res, opts ->
          moed_forward([e0, e1], weights, res, opts)
        end

      3 ->
        fn e0, e1, e2, weights, res, opts ->
          moed_forward([e0, e1, e2], weights, res, opts)
        end

      4 ->
        fn e0, e1, e2, e3, weights, res, opts ->
          moed_forward([e0, e1, e2, e3], weights, res, opts)
        end

      6 ->
        fn e0, e1, e2, e3, e4, e5, weights, res, opts ->
          moed_forward([e0, e1, e2, e3, e4, e5], weights, res, opts)
        end

      8 ->
        fn e0, e1, e2, e3, e4, e5, e6, e7, weights, res, opts ->
          moed_forward([e0, e1, e2, e3, e4, e5, e6, e7], weights, res, opts)
        end
    end
  end

  # Weighted combination of expert outputs + no-op (identity/residual) path.
  # experts: list of [batch, seq, hidden] tensors
  # weights: [batch, seq, num_experts + 1] (last slot is no-op)
  # residual: [batch, seq, hidden] (attention output = identity path)
  defp moed_forward(experts, weights, residual, _opts) do
    num_experts = length(experts)

    # Stack expert outputs: [num_experts, batch, seq, hidden]
    stacked = Nx.stack(experts)

    # Transpose to [batch, seq, num_experts, hidden]
    stacked_t = Nx.transpose(stacked, axes: [1, 2, 0, 3])

    # Expert weights (first E columns): [batch, seq, num_experts, 1]
    expert_w = weights[[.., .., 0..(num_experts - 1)//1]]
    expert_w = Nx.new_axis(expert_w, 3)

    # Weighted expert sum: [batch, seq, hidden]
    expert_out = Nx.sum(Nx.multiply(stacked_t, expert_w), axes: [2])

    # No-op weight (last column): [batch, seq, 1]
    noop_w = weights[[.., .., num_experts..num_experts]]

    # Final: expert_weighted_sum + noop_weight * residual
    Nx.add(expert_out, Nx.multiply(noop_w, residual))
  end

  @doc "Get the output size of a MixtureOfExpertDepths model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
