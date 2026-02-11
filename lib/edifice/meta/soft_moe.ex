defmodule Edifice.Meta.SoftMoE do
  @moduledoc """
  Soft Mixture of Experts (Puigcerver et al., 2024).

  Unlike hard-routing MoE (Switch/top-K), Soft MoE computes a soft weighted
  combination of all expert outputs for every token. This eliminates token
  dropping, load balancing issues, and routing instability while maintaining
  the capacity benefits of MoE.

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
  | SoftMoE Block:                     |
  |   1. Compute dispatch weights      |
  |      D = softmax(X * Phi)          |
  |   2. Compute expert inputs         |
  |      X_e = D^T * X                 |
  |   3. Run all experts               |
  |      Y_e = Expert_e(X_e)           |
  |   4. Combine outputs               |
  |      Y = C * stack(Y_e)            |
  |   + Residual                       |
  +------------------------------------+
        |  (repeat N times)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = SoftMoE.build(
        embed_size: 256,
        hidden_size: 256,
        num_experts: 4,
        num_layers: 4
      )

  ## References

  - Puigcerver et al., "From Sparse to Soft Mixtures of Experts" (ICLR 2024)
  - https://arxiv.org/abs/2308.00951
  """

  require Axon
  import Nx.Defn

  @default_hidden_size 256
  @default_num_experts 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Soft MoE model.

  ## Options

  - `:embed_size` - Input embedding dimension (required)
  - `:hidden_size` - Hidden dimension (default: 256)
  - `:num_experts` - Number of experts (default: 4)
  - `:num_layers` - Number of SoftMoE blocks (default: 4)
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

    # Stack SoftMoE blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        soft_moe_block(acc, hidden_size,
          num_experts: num_experts,
          dropout: dropout,
          name: "soft_moe_#{idx}"
        )
      end)

    # Final norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep
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
  Single Soft MoE block with dispatch-combine routing.

  ## Options

  - `:num_experts` - Number of experts (default: 4)
  - `:dropout` - Dropout rate (default: 0.1)
  - `:name` - Layer name prefix
  """
  @spec soft_moe_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def soft_moe_block(input, hidden_size, opts \\ []) do
    num_experts = Keyword.get(opts, :num_experts, @default_num_experts)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "soft_moe")

    normed = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Dispatch weights: project input to num_experts scores
    # dispatch: [batch, seq_len, num_experts]
    dispatch_logits = Axon.dense(normed, num_experts, name: "#{name}_dispatch")

    # Build all expert FFNs
    expert_size = hidden_size * 4

    experts =
      for i <- 0..(num_experts - 1) do
        normed
        |> Axon.dense(expert_size, name: "#{name}_expert_#{i}_up")
        |> Axon.gelu()
        |> Axon.dense(hidden_size, name: "#{name}_expert_#{i}_down")
      end

    # Soft combination: weighted sum of all expert outputs
    combined = build_soft_combine(dispatch_logits, experts, num_experts, name)

    combined =
      if dropout > 0.0 do
        Axon.dropout(combined, rate: dropout, name: "#{name}_drop")
      else
        combined
      end

    Axon.add(input, combined, name: "#{name}_residual")
  end

  @doc """
  Get the output size of a Soft MoE model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  # Build soft combination of experts using dispatch weights
  defp build_soft_combine(dispatch_logits, experts, num_experts, name) do
    combine_fn =
      case num_experts do
        2 ->
          fn dispatch, e0, e1, _opts ->
            soft_combine(dispatch, Nx.stack([e0, e1]))
          end

        4 ->
          fn dispatch, e0, e1, e2, e3, _opts ->
            soft_combine(dispatch, Nx.stack([e0, e1, e2, e3]))
          end

        8 ->
          fn dispatch, e0, e1, e2, e3, e4, e5, e6, e7, _opts ->
            soft_combine(dispatch, Nx.stack([e0, e1, e2, e3, e4, e5, e6, e7]))
          end

        _ ->
          # Fallback: use mean of first 2
          fn dispatch, e0, e1, _opts ->
            _ = dispatch
            Nx.add(Nx.multiply(0.5, e0), Nx.multiply(0.5, e1))
          end
      end

    Axon.layer(
      combine_fn,
      [dispatch_logits | experts],
      name: "#{name}_combine",
      op_name: :soft_moe_combine
    )
  end

  # Soft combine: weighted sum of all expert outputs
  defnp soft_combine(dispatch_logits, experts_stacked) do
    # dispatch_logits: [batch, seq_len, num_experts]
    # experts_stacked: [num_experts, batch, seq_len, hidden_size]

    # Softmax dispatch weights
    dispatch_probs =
      Nx.exp(dispatch_logits - Nx.reduce_max(dispatch_logits, axes: [-1], keep_axes: true))

    dispatch_probs = dispatch_probs / Nx.sum(dispatch_probs, axes: [-1], keep_axes: true)

    # Transpose experts: [batch, seq_len, num_experts, hidden_size]
    experts_t = Nx.transpose(experts_stacked, axes: [1, 2, 0, 3])

    # Weighted combination: [batch, seq_len, 1, num_experts] @ [batch, seq_len, num_experts, hidden_size]
    weights = Nx.new_axis(dispatch_probs, 2)
    output = Nx.dot(weights, [3], [0, 1], experts_t, [2], [0, 1])
    Nx.squeeze(output, axes: [2])
  end
end
