defmodule Edifice.Meta.RouterNetwork do
  @moduledoc """
  Router Network: learned input-level dispatch to specialist models.

  Unlike MoE (which routes hidden states within a single model at the layer
  level), RouterNetwork routes entire inputs across separate specialist
  transformer stacks. A lightweight router scores each specialist, and the
  final output is a weighted combination (soft routing) or top-k selection
  (hard routing via straight-through estimator).

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        +---> Router (small MLP)
        |         |
        |         v
        |     Dispatch weights [batch, num_specialists]
        |
        +----+----+----+----+
        |    |    |    |    |
        v    v    v    v    v
       S1   S2   S3   S4  ...  (Specialist stacks)
        |    |    |    |    |
        v    v    v    v    v
  Weighted sum (soft) or top-k select (hard)
        |
        v
  Final Norm → Last Timestep
  Output [batch, aggregator_hidden_size]
  ```

  ## Routing Modes

  - **Soft routing** (default): output is `sum(weight_i * specialist_i(x))`.
    Fully differentiable, all specialists contribute.
  - **Hard routing** (`:top_k`): only the top-k specialists contribute.
    Uses straight-through estimator for gradient flow during training.

  ## Usage

      model = RouterNetwork.build(
        embed_dim: 64,
        num_specialists: 4,
        specialist_hidden_size: 32,
        specialist_layers: 2,
        routing: :soft,
        num_heads: 4
      )

  ## References
  - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (2017)
  - Fedus et al., "Switch Transformers" (2021)
  - Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @doc """
  Build a Router Network model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:num_specialists` - Number of specialist stacks (default: 4)
    - `:specialist_hidden_size` - Hidden size per specialist (default: 128)
    - `:specialist_layers` - Transformer layers per specialist (default: 2)
    - `:output_hidden_size` - Output projection size (default: specialist_hidden_size)
    - `:num_heads` - Attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:routing` - Routing mode: `:soft` or `{:top_k, k}` (default: `:soft`)
    - `:router_hidden_size` - Router MLP hidden size (default: 64)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, output_hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:num_specialists, pos_integer()}
          | {:specialist_hidden_size, pos_integer()}
          | {:specialist_layers, pos_integer()}
          | {:output_hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}
          | {:routing, :soft | {:top_k, pos_integer()}}
          | {:router_hidden_size, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_specialists = Keyword.get(opts, :num_specialists, 4)
    spec_hidden = Keyword.get(opts, :specialist_hidden_size, 128)
    spec_layers = Keyword.get(opts, :specialist_layers, 2)
    out_hidden = Keyword.get(opts, :output_hidden_size, spec_hidden)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, 0.1)
    routing = Keyword.get(opts, :routing, :soft)
    router_hidden = Keyword.get(opts, :router_hidden_size, 64)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to specialist hidden size
    projected =
      if embed_dim != spec_hidden do
        Axon.dense(input, spec_hidden, name: "input_projection")
      else
        input
      end

    # Router: input -> mean-pool -> MLP -> num_specialists logits
    # Mean-pool across sequence to get a sequence-level routing decision
    router_input =
      Axon.nx(input, fn x -> Nx.mean(x, axes: [1]) end, name: "router_pool")

    router_logits =
      router_input
      |> Axon.dense(router_hidden, name: "router_hidden", activation: :relu)
      |> Axon.dense(num_specialists, name: "router_logits")

    # Routing weights: softmax over specialists
    router_weights =
      Axon.nx(
        router_logits,
        fn x ->
          max_val = Nx.reduce_max(x, axes: [-1], keep_axes: true)
          exp_x = Nx.exp(Nx.subtract(x, max_val))
          Nx.divide(exp_x, Nx.sum(exp_x, axes: [-1], keep_axes: true))
        end,
        name: "router_softmax"
      )

    # Build specialist stacks
    specialist_outputs =
      for i <- 0..(num_specialists - 1) do
        spec =
          build_specialist_stack(projected, spec_hidden, spec_layers, num_heads, dropout,
            name: "specialist_#{i}"
          )

        # Project specialist to output size if needed, then take last timestep
        spec =
          if spec_hidden != out_hidden do
            Axon.dense(spec, out_hidden, name: "specialist_#{i}_output_proj")
          else
            spec
          end

        # Last timestep: [batch, seq, H] -> [batch, H]
        Axon.nx(
          spec,
          fn t ->
            seq_size = Nx.axis_size(t, 1)
            Nx.slice_along_axis(t, seq_size - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
          end,
          name: "specialist_#{i}_last_ts"
        )
      end

    # Concatenate specialist outputs along a new dim: [batch, num_specialists, H]
    concat_specs = Axon.concatenate(specialist_outputs, axis: 1, name: "concat_specialists")

    stacked_specs =
      Axon.nx(
        concat_specs,
        fn tensor ->
          {b, total} = Nx.shape(tensor)
          Nx.reshape(tensor, {b, num_specialists, div(total, num_specialists)})
        end,
        name: "reshape_specialists"
      )

    # Combine based on routing mode
    combined =
      case routing do
        :soft ->
          # Soft: weighted sum of all specialists
          # weights: [batch, num_specialists] -> [batch, num_specialists, 1]
          # stacked: [batch, num_specialists, H]
          Axon.layer(
            fn specs, weights, _opts ->
              w = Nx.new_axis(weights, 2)
              Nx.sum(Nx.multiply(specs, w), axes: [1])
            end,
            [stacked_specs, router_weights],
            name: "soft_routing"
          )

        {:top_k, k} ->
          # Hard: select top-k specialists, straight-through estimator
          Axon.layer(
            fn specs, weights, _opts ->
              top_k_route(specs, weights, k)
            end,
            [stacked_specs, router_weights],
            name: "top_k_routing"
          )
      end

    # Final layer norm
    Axon.layer(
      fn x, _opts ->
        mean = Nx.mean(x, axes: [-1], keep_axes: true)
        var = Nx.variance(x, axes: [-1], keep_axes: true)
        Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, 1.0e-5)))
      end,
      [combined],
      name: "final_norm"
    )
  end

  @doc """
  Get the output size of a RouterNetwork model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    spec_hidden = Keyword.get(opts, :specialist_hidden_size, 128)
    Keyword.get(opts, :output_hidden_size, spec_hidden)
  end

  @doc """
  Get recommended defaults for RouterNetwork.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_specialists: 4,
      specialist_hidden_size: 128,
      specialist_layers: 2,
      output_hidden_size: 128,
      num_heads: 4,
      dropout: 0.1,
      routing: :soft,
      router_hidden_size: 64,
      window_size: 60
    ]
  end

  # Build a single specialist transformer stack
  defp build_specialist_stack(input, hidden_size, num_layers, num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "specialist")

    Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
      block_name = "#{name}_block_#{layer_idx}"

      TransformerBlock.layer(acc,
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
        name: block_name
      )
    end)
  end

  # Top-k routing: only top-k specialists contribute with renormalized weights.
  # Gradient flows through the selected specialists' soft weights naturally
  # since the mask is derived from the weights themselves.
  defp top_k_route(specs, weights, k) do
    # specs: [batch, num_specialists, H]
    # weights: [batch, num_specialists]
    {_top_vals, top_indices} = Nx.top_k(weights, k: k)

    # Create mask: 1 for selected specialists, 0 otherwise
    num_specialists = Nx.axis_size(weights, 1)

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.iota({num_specialists}), 0) |> Nx.new_axis(0),
        Nx.new_axis(top_indices, 2)
      )

    # [batch, num_specialists]
    mask = Nx.reduce_max(one_hot, axes: [1])

    # Renormalize weights for selected specialists
    masked_weights = Nx.multiply(weights, mask)
    weight_sum = Nx.sum(masked_weights, axes: [-1], keep_axes: true)
    normalized = Nx.divide(masked_weights, Nx.add(weight_sum, 1.0e-8))

    w = Nx.new_axis(normalized, 2)
    Nx.sum(Nx.multiply(specs, w), axes: [1])
  end
end
