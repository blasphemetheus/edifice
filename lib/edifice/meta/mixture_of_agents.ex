defmodule Edifice.Meta.MixtureOfAgents do
  @moduledoc """
  Mixture of Agents: N proposer models feed into an aggregator.

  Implements a multi-agent architecture where N independent "proposer" transformer
  stacks process the same input in parallel, then their outputs are concatenated
  and fed into a larger "aggregator" transformer stack that combines the proposals.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        +----+----+----+----+
        |    |    |    |    |
        v    v    v    v    v
       P1   P2   P3   P4  ...  (Proposer stacks)
        |    |    |    |    |
        v    v    v    v    v
  Concatenate along feature dim
        |
        v
  [batch, seq, num_proposers * proposer_hidden]
        |
        v
  Dense projection to aggregator_hidden
        |
        v
  +-----------------------------+
  |   Aggregator Transformer    |
  |   (larger, combines all)    |
  +-----------------------------+
        |
        v
  Final Norm -> Last Timestep
  Output [batch, aggregator_hidden_size]
  ```

  ## Design

  Each proposer is a lightweight transformer stack that can specialize on
  different aspects of the input. The aggregator is typically larger and
  learns to combine the diverse proposals into a unified representation.

  ## Usage

      model = MixtureOfAgents.build(
        embed_dim: 287,
        num_proposers: 4,
        proposer_hidden_size: 128,
        aggregator_hidden_size: 256,
        proposer_layers: 2,
        aggregator_layers: 2
      )

  ## References
  - Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities" (2024)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @doc """
  Build a Mixture of Agents model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:num_proposers` - Number of proposer stacks (default: 4)
    - `:proposer_hidden_size` - Hidden size for each proposer (default: 128)
    - `:aggregator_hidden_size` - Hidden size for the aggregator (default: 256)
    - `:proposer_layers` - Number of layers per proposer (default: 2)
    - `:aggregator_layers` - Number of aggregator layers (default: 2)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, aggregator_hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:aggregator_hidden_size, pos_integer()}
          | {:aggregator_layers, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_proposers, pos_integer()}
          | {:proposer_hidden_size, pos_integer()}
          | {:proposer_layers, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_proposers = Keyword.get(opts, :num_proposers, 4)
    proposer_hidden = Keyword.get(opts, :proposer_hidden_size, 128)
    aggregator_hidden = Keyword.get(opts, :aggregator_hidden_size, 256)
    proposer_layers = Keyword.get(opts, :proposer_layers, 2)
    aggregator_layers = Keyword.get(opts, :aggregator_layers, 2)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, 0.1)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to proposer hidden size
    projected =
      if embed_dim != proposer_hidden do
        Axon.dense(input, proposer_hidden, name: "input_projection")
      else
        input
      end

    # Build N proposer stacks in parallel
    proposer_outputs =
      for i <- 0..(num_proposers - 1) do
        build_proposer_stack(projected, proposer_hidden, proposer_layers, num_heads, dropout,
          name: "proposer_#{i}"
        )
      end

    # Concatenate proposer outputs along feature dimension
    # Each is [batch, seq, proposer_hidden] -> [batch, seq, num_proposers * proposer_hidden]
    concatenated = Axon.concatenate(proposer_outputs, axis: 2, name: "concat_proposals")

    # Project to aggregator hidden size
    agg_input =
      Axon.dense(concatenated, aggregator_hidden, name: "aggregator_projection")

    # Aggregator transformer stack
    agg_output =
      Enum.reduce(1..aggregator_layers, agg_input, fn layer_idx, acc ->
        name = "aggregator_block_#{layer_idx}"

        TransformerBlock.layer(acc,
          attention_fn: fn x, attn_name ->
            MultiHead.self_attention(x,
              hidden_size: aggregator_hidden,
              num_heads: num_heads,
              dropout: dropout,
              causal: true,
              name: attn_name
            )
          end,
          hidden_size: aggregator_hidden,
          dropout: dropout,
          name: name
        )
      end)

    # Final norm
    agg_output = Axon.layer_norm(agg_output, name: "final_norm")

    # Last timestep: [batch, seq, hidden] -> [batch, hidden]
    Axon.nx(
      agg_output,
      fn tensor ->
        seq_size = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_size - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Get the output size of a MixtureOfAgents model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :aggregator_hidden_size, 256)
  end

  @doc """
  Get recommended defaults for MixtureOfAgents.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_proposers: 4,
      proposer_hidden_size: 128,
      aggregator_hidden_size: 256,
      proposer_layers: 2,
      aggregator_layers: 2,
      num_heads: 4,
      dropout: 0.1,
      window_size: 60
    ]
  end

  # Build a single proposer transformer stack
  defp build_proposer_stack(input, hidden_size, num_layers, num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "proposer")

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
end
