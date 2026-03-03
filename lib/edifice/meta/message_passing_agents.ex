defmodule Edifice.Meta.MessagePassingAgents do
  @moduledoc """
  Message-Passing Agents: GNN-inspired agent communication graph.

  Models a swarm of agents as nodes in a graph, with communication channels as
  edges. Each communication round applies graph neural network message passing:

  1. Each agent computes messages to its neighbors
  2. Each agent aggregates incoming messages
  3. Each agent updates its state via a GRU cell

  Unlike `AgentSwarm` (which uses dense all-to-all attention), this module
  supports sparse communication topologies via an adjacency matrix. This is
  useful for modeling hierarchical teams, ring topologies, or specialist
  clusters where not every agent should talk to every other agent.

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
  Stack: [batch, num_agents, hidden]  (mean-pooled over seq)
        |
        v
  +---------------------------------+
  | Message Passing (R rounds)      |
  | For each round:                 |
  |   1. sender_proj, receiver_proj |
  |   2. A @ sender_proj → agg     |
  |   3. GRU(state, agg) → state   |
  +---------------------------------+
        |
        v
  Pool agents → [batch, hidden]
        |
        v
  Output projection → [batch, output_size]
  ```

  ## Communication Topologies

  The adjacency matrix controls who talks to whom. Pass via the `"adjacency"`
  input or leave it out for fully-connected (default). Examples:

  - **Fully-connected**: every agent talks to every agent (default)
  - **Ring**: each agent talks to left/right neighbors only
  - **Hierarchical**: leader agent connected to all, workers connected to leader only
  - **Clusters**: specialist groups with intra-group edges + sparse inter-group

  ## Usage

      model = MessagePassingAgents.build(
        embed_dim: 64,
        num_agents: 4,
        agent_hidden_size: 32,
        agent_layers: 1,
        message_rounds: 3,
        output_size: 64,
        num_heads: 4
      )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates, Axon.ModelState.empty())

      # Fully-connected (all agents communicate):
      adj = Nx.broadcast(1.0, {batch, 4, 4})
      output = predict_fn.(params, %{
        "state_sequence" => input,
        "adjacency" => adj
      })

      # Ring topology (each agent talks to neighbors):
      adj = build_ring_adjacency(batch, 4)
      output = predict_fn.(params, %{
        "state_sequence" => input,
        "adjacency" => adj
      })

  ## References
  - Gilmer et al., "Neural Message Passing for Quantum Chemistry" (2017)
  - Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation" (2016)
  - Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation" (2018)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @doc """
  Build a Message-Passing Agents model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:num_agents` - Number of agent nodes (default: 4)
    - `:agent_hidden_size` - Hidden size per agent (default: 64)
    - `:agent_layers` - Transformer layers per agent proposer (default: 1)
    - `:message_rounds` - Rounds of message passing (default: 3)
    - `:output_size` - Final output dimension (default: agent_hidden_size)
    - `:num_heads` - Attention heads for agent proposers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:aggregation` - Message aggregation: :sum or :mean (default: :mean)
    - `:pool_mode` - Agent pooling: :mean, :sum, or :max (default: :mean)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, output_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:num_agents, pos_integer()}
          | {:agent_hidden_size, pos_integer()}
          | {:agent_layers, pos_integer()}
          | {:message_rounds, pos_integer()}
          | {:output_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}
          | {:aggregation, :sum | :mean}
          | {:pool_mode, :mean | :sum | :max}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_agents = Keyword.get(opts, :num_agents, 4)
    agent_hidden = Keyword.get(opts, :agent_hidden_size, 64)
    agent_layers = Keyword.get(opts, :agent_layers, 1)
    msg_rounds = Keyword.get(opts, :message_rounds, 3)
    out_size = Keyword.get(opts, :output_size, agent_hidden)
    num_heads = Keyword.get(opts, :num_heads, 4)
    dropout = Keyword.get(opts, :dropout, 0.1)
    aggregation = Keyword.get(opts, :aggregation, :mean)
    pool_mode = Keyword.get(opts, :pool_mode, :mean)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Adjacency input: [batch, num_agents, num_agents]
    # Pass ones for fully-connected, or custom topology
    adjacency = Axon.input("adjacency", shape: {nil, num_agents, num_agents})

    # Project input to agent hidden size
    projected =
      if embed_dim != agent_hidden do
        Axon.dense(input, agent_hidden, name: "input_projection")
      else
        input
      end

    # Build N agent proposer stacks, each produces [batch, seq, agent_hidden]
    # Then mean-pool over seq to get [batch, agent_hidden] per agent
    agent_features =
      for i <- 0..(num_agents - 1) do
        agent_seq = build_agent_stack(projected, agent_hidden, agent_layers, num_heads, dropout,
          name: "agent_#{i}"
        )

        # Mean-pool over sequence: [batch, seq, H] -> [batch, H]
        Axon.nx(agent_seq, fn tensor ->
          Nx.mean(tensor, axes: [1])
        end, name: "agent_#{i}_pool")
      end

    # Stack agent features: list of [batch, H] -> [batch, num_agents, H]
    # Concatenate along feature dim then reshape
    concatenated = Axon.concatenate(agent_features, axis: 1, name: "concat_agent_features")

    node_features =
      Axon.nx(
        concatenated,
        fn tensor ->
          {b, _total} = Nx.shape(tensor)
          Nx.reshape(tensor, {b, num_agents, agent_hidden})
        end,
        name: "stack_agent_features"
      )

    # Message passing rounds with GRU state updates
    updated_nodes =
      Enum.reduce(1..msg_rounds, node_features, fn round_idx, acc ->
        build_message_round(acc, adjacency, agent_hidden, aggregation,
          name: "msg_round_#{round_idx}"
        )
      end)

    # Pool over agents: [batch, num_agents, H] -> [batch, H]
    pooled =
      case pool_mode do
        :mean ->
          Axon.nx(updated_nodes, fn t -> Nx.mean(t, axes: [1]) end, name: "agent_pool_mean")

        :sum ->
          Axon.nx(updated_nodes, fn t -> Nx.sum(t, axes: [1]) end, name: "agent_pool_sum")

        :max ->
          Axon.nx(updated_nodes, fn t -> Nx.reduce_max(t, axes: [1]) end, name: "agent_pool_max")
      end

    # Output projection
    if out_size != agent_hidden do
      pooled
      |> Axon.dense(out_size, name: "output_projection")
      |> Axon.activation(:relu, name: "output_activation")
    else
      pooled
    end
  end

  @doc """
  Get the output size of a MessagePassingAgents model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    agent_hidden = Keyword.get(opts, :agent_hidden_size, 64)
    Keyword.get(opts, :output_size, agent_hidden)
  end

  @doc """
  Get recommended defaults for MessagePassingAgents.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      num_agents: 4,
      agent_hidden_size: 64,
      agent_layers: 1,
      message_rounds: 3,
      num_heads: 4,
      dropout: 0.1,
      aggregation: :mean,
      pool_mode: :mean,
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

  # One message passing round:
  # 1. Project node features for sender/receiver roles
  # 2. Aggregate neighbor messages via adjacency matrix
  # 3. Update node state via GRU cell
  defp build_message_round(node_features, adjacency, hidden_size, aggregation, opts) do
    name = Keyword.get(opts, :name, "msg")

    # Sender and receiver projections
    sender_proj = Axon.dense(node_features, hidden_size, name: "#{name}_sender_proj")
    receiver_proj = Axon.dense(node_features, hidden_size, name: "#{name}_receiver_proj")

    # Aggregate neighbor messages: A @ sender_proj -> [batch, num_agents, H]
    aggregated =
      Axon.layer(
        fn sender, receiver, adj, _opts ->
          # neighbor_msgs[i] = sum_j(A[i,j] * sender[j])
          neighbor_msgs = Nx.dot(adj, [2], [0], sender, [1], [0])

          case aggregation do
            :mean ->
              degree = Nx.sum(adj, axes: [2], keep_axes: true)
              degree = Nx.max(degree, 1.0)
              Nx.add(Nx.divide(neighbor_msgs, degree), receiver)

            :sum ->
              Nx.add(neighbor_msgs, receiver)
          end
        end,
        [sender_proj, receiver_proj, adjacency],
        name: "#{name}_aggregate"
      )

    # GRU update: state = node_features, input = aggregated messages
    # z = sigmoid(Wz @ [agg, state])
    # r = sigmoid(Wr @ [agg, state])
    # candidate = tanh(Wh @ [agg, r * state])
    # new_state = (1 - z) * state + z * candidate
    concat_for_gates =
      Axon.concatenate([aggregated, node_features], axis: 2, name: "#{name}_gru_concat")

    z_gate = Axon.dense(concat_for_gates, hidden_size, name: "#{name}_gru_z", activation: :sigmoid)
    r_gate = Axon.dense(concat_for_gates, hidden_size, name: "#{name}_gru_r", activation: :sigmoid)

    # Reset-gated state
    reset_state =
      Axon.layer(
        fn r, state, _opts -> Nx.multiply(r, state) end,
        [r_gate, node_features],
        name: "#{name}_gru_reset"
      )

    # Candidate
    candidate_input =
      Axon.concatenate([aggregated, reset_state], axis: 2, name: "#{name}_gru_cand_concat")

    candidate = Axon.dense(candidate_input, hidden_size, name: "#{name}_gru_candidate", activation: :tanh)

    # Final GRU update
    Axon.layer(
      fn z, state, cand, _opts ->
        Nx.add(
          Nx.multiply(Nx.subtract(1.0, z), state),
          Nx.multiply(z, cand)
        )
      end,
      [z_gate, node_features, candidate],
      name: "#{name}_gru_update"
    )
  end
end
