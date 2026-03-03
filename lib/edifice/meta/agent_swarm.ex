defmodule Edifice.Meta.AgentSwarm do
  @moduledoc """
  Agent Swarm: communication-augmented multi-agent ensemble.

  Extends `MixtureOfAgents` with inter-agent attention rounds. After proposer
  stacks produce independent representations, agents attend to each other's
  outputs over R communication rounds before aggregation. This captures the
  "multi-agent debate" pattern in a differentiable, end-to-end trainable form.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        +----+----+----+----+
        |    |    |    |    |
        v    v    v    v    v
       A1   A2   A3   A4  ...  (Agent proposer stacks)
        |    |    |    |    |
        v    v    v    v    v
  Stack: [batch, num_agents, seq, agent_hidden]
        |
        + Agent embeddings (optional)
        |
        v
  +---------------------------------+
  | Inter-agent attention (R rounds)|
  | Each agent attends to all       |
  | agents via cross-attention      |
  | + per-agent FFN + residual      |
  +---------------------------------+
        |
        v
  Flatten → Dense → [batch, seq, aggregator_hidden]
        |
        v
  +-----------------------------+
  |   Aggregator Transformer    |
  +-----------------------------+
        |
        v
  Final Norm → Last Timestep
  Output [batch, aggregator_hidden_size]
  ```

  ## Communication Rounds

  Each communication round applies multi-head attention where each agent's
  representation queries all agents' representations. This is implemented by
  reshaping the agent dimension into the sequence dimension for cross-attention,
  then reshaping back. Conceptually similar to message-passing GNN rounds.

  ## Usage

      model = AgentSwarm.build(
        embed_dim: 64,
        num_agents: 4,
        agent_hidden_size: 32,
        agent_layers: 2,
        communication_rounds: 2,
        aggregator_hidden_size: 64,
        aggregator_layers: 2,
        num_heads: 4
      )

  ## References
  - Du et al., "Improving Factuality and Reasoning through Multiagent Debate" (2023)
  - Wang et al., "Mixture-of-Agents Enhances LLM Capabilities" (2024)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @doc """
  Build an Agent Swarm model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:num_agents` - Number of agent proposer stacks (default: 4)
    - `:agent_hidden_size` - Hidden size per agent (default: 128)
    - `:agent_layers` - Transformer layers per agent proposer (default: 2)
    - `:communication_rounds` - Rounds of inter-agent attention (default: 2)
    - `:aggregator_hidden_size` - Aggregator transformer hidden size (default: 256)
    - `:aggregator_layers` - Aggregator transformer layers (default: 2)
    - `:num_heads` - Attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:use_agent_embeddings` - Add learned agent identity embeddings (default: true)
    - `:communication_gate` - Learnable per-agent gate on communication (default: false)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, aggregator_hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:num_agents, pos_integer()}
          | {:agent_hidden_size, pos_integer()}
          | {:agent_layers, pos_integer()}
          | {:communication_rounds, pos_integer()}
          | {:aggregator_hidden_size, pos_integer()}
          | {:aggregator_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}
          | {:use_agent_embeddings, boolean()}
          | {:communication_gate, boolean()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_agents = Keyword.get(opts, :num_agents, 4)
    agent_hidden = Keyword.get(opts, :agent_hidden_size, 128)
    agent_layers = Keyword.get(opts, :agent_layers, 2)
    comm_rounds = Keyword.get(opts, :communication_rounds, 2)
    agg_hidden = Keyword.get(opts, :aggregator_hidden_size, 256)
    agg_layers = Keyword.get(opts, :aggregator_layers, 2)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, 0.1)
    use_agent_emb = Keyword.get(opts, :use_agent_embeddings, true)
    comm_gate = Keyword.get(opts, :communication_gate, false)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to agent hidden size
    projected =
      if embed_dim != agent_hidden do
        Axon.dense(input, agent_hidden, name: "input_projection")
      else
        input
      end

    # Build N agent proposer stacks
    agent_outputs =
      for i <- 0..(num_agents - 1) do
        build_agent_stack(projected, agent_hidden, agent_layers, num_heads, dropout,
          name: "agent_#{i}"
        )
      end

    # Stack agent outputs: list of [batch, seq, H] -> [batch, num_agents, seq, H]
    # Concatenate along feature dim: [batch, seq, num_agents * H]
    # Then reshape to introduce agent dim
    concatenated = Axon.concatenate(agent_outputs, axis: 2, name: "concat_agents")

    stacked =
      Axon.nx(
        concatenated,
        fn tensor ->
          {b, s, _total} = Nx.shape(tensor)
          # Reshape [batch, seq, num_agents * H] -> [batch, seq, num_agents, H]
          # Then transpose to [batch, num_agents, seq, H]
          tensor
          |> Nx.reshape({b, s, num_agents, agent_hidden})
          |> Nx.transpose(axes: [0, 2, 1, 3])
        end,
        name: "stack_agents"
      )

    # Add learned agent embeddings: [1, num_agents, 1, H] broadcast-added
    stacked =
      if use_agent_emb do
        agent_emb =
          Axon.param("agent_embeddings", {1, num_agents, 1, agent_hidden},
            initializer: :glorot_uniform
          )

        Axon.layer(
          fn x, emb, _opts -> Nx.add(x, emb) end,
          [stacked, agent_emb],
          name: "add_agent_embeddings"
        )
      else
        stacked
      end

    # Inter-agent communication rounds
    # Reshape [batch, num_agents, seq, H] -> [batch*seq, num_agents, H]
    # Apply self-attention across agents, then reshape back
    communicated =
      Enum.reduce(1..comm_rounds, stacked, fn round_idx, acc ->
        build_communication_round(acc, num_agents, agent_hidden, num_heads, dropout,
          gate: comm_gate,
          name: "comm_round_#{round_idx}"
        )
      end)

    # Flatten agent dim: [batch, num_agents, seq, H] -> [batch, seq, num_agents * H]
    flat =
      Axon.nx(
        communicated,
        fn tensor ->
          {b, _a, s, _h} = Nx.shape(tensor)
          # Transpose to [batch, seq, num_agents, H] then reshape
          tensor
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({b, s, :auto})
        end,
        name: "flatten_agents"
      )

    # Project to aggregator hidden size
    agg_input = Axon.dense(flat, agg_hidden, name: "aggregator_projection")

    # Aggregator transformer stack
    agg_output =
      Enum.reduce(1..agg_layers, agg_input, fn layer_idx, acc ->
        TransformerBlock.layer(acc,
          attention_fn: fn x, attn_name ->
            MultiHead.self_attention(x,
              hidden_size: agg_hidden,
              num_heads: num_heads,
              dropout: dropout,
              causal: true,
              name: attn_name
            )
          end,
          hidden_size: agg_hidden,
          dropout: dropout,
          name: "aggregator_block_#{layer_idx}"
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
  Get the output size of an AgentSwarm model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :aggregator_hidden_size, 256)
  end

  @doc """
  Get recommended defaults for AgentSwarm.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_agents: 4,
      agent_hidden_size: 128,
      agent_layers: 2,
      communication_rounds: 2,
      aggregator_hidden_size: 256,
      aggregator_layers: 2,
      num_heads: 4,
      dropout: 0.1,
      use_agent_embeddings: true,
      communication_gate: false,
      window_size: 60
    ]
  end

  # Build a single agent proposer transformer stack
  defp build_agent_stack(input, hidden_size, num_layers, num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "agent")

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

  # One communication round: inter-agent attention + FFN with residual
  #
  # Input: [batch, num_agents, seq, H]
  # Process: reshape -> self-attention across agents -> reshape back
  defp build_communication_round(stacked, num_agents, hidden_size, num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "comm")
    gate = Keyword.get(opts, :gate, false)

    # Reshape [batch, num_agents, seq, H] -> [batch*seq, num_agents, H]
    # This lets standard self-attention operate across the agent dimension
    flat_for_attn =
      Axon.nx(
        stacked,
        fn tensor ->
          {b, a, s, h} = Nx.shape(tensor)
          tensor
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({b * s, a, h})
        end,
        name: "#{name}_reshape_in"
      )

    # Self-attention across agents (num_agents positions)
    attn_out =
      TransformerBlock.layer(flat_for_attn,
        attention_fn: fn x, attn_name ->
          MultiHead.self_attention(x,
            hidden_size: hidden_size,
            num_heads: num_heads,
            dropout: dropout,
            causal: false,
            name: attn_name
          )
        end,
        hidden_size: hidden_size,
        dropout: dropout,
        name: "#{name}_attn_block"
      )

    # Reshape back: [batch*seq, num_agents, H] -> [batch, num_agents, seq, H]
    reshaped =
      Axon.layer(
        fn attn_tensor, ref_tensor, _opts ->
          {b, _a, s, _h} = Nx.shape(ref_tensor)
          {_bs, a, h} = Nx.shape(attn_tensor)

          attn_tensor
          |> Nx.reshape({b, s, a, h})
          |> Nx.transpose(axes: [0, 2, 1, 3])
        end,
        [attn_out, stacked],
        name: "#{name}_reshape_out"
      )

    # Optional per-agent gate: sigmoid gate that controls how much communication to use
    if gate do
      # gate_param: [1, num_agents, 1, 1] — one scalar gate per agent
      gate_param =
        Axon.param("#{name}_gate", {1, num_agents, 1, 1},
          initializer: :zeros
        )

      Axon.layer(
        fn comm, original, g, _opts ->
          gate_val = Nx.sigmoid(g)
          # Blend: gate * communicated + (1 - gate) * original
          Nx.add(Nx.multiply(gate_val, comm), Nx.multiply(Nx.subtract(1.0, gate_val), original))
        end,
        [reshaped, stacked, gate_param],
        name: "#{name}_gated_blend"
      )
    else
      reshaped
    end
  end
end
