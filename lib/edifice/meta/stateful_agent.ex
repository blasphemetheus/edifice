defmodule Edifice.Meta.StatefulAgent do
  @moduledoc """
  Stateful Agent: multi-turn architecture wrapper with persistent memory.

  Wraps any Edifice sequence architecture with an explicit state tensor that
  persists across forward calls (turns). Each turn processes new input along
  with the previous state, producing output and updated state.

  This provides the stateful backbone that agents in a multi-agent swarm
  need for multi-turn interactions.

  ## Architecture

  ```
  Turn 1:  Input_1 → [Backbone + State_0] → Output_1, State_1
  Turn 2:  Input_2 → [Backbone + State_1] → Output_2, State_2
  Turn N:  Input_N → [Backbone + State_{N-1}] → Output_N, State_N
  ```

  The model takes two inputs: the current turn's input sequence and the
  previous state. It produces a tuple of `{output, new_state}`.

  ## State Modes

  - **`:compressive`** (default) — Fixed-size state updated via gated linear
    combination. State shape: `[batch, state_size]`. Each turn: read state
    via learned projection, blend with backbone output via sigmoid gate,
    write back via learned projection. Inspired by InfiniAttention's
    compressive memory.

  - **`:ema`** — Exponential moving average of backbone outputs. State shape:
    `[batch, state_size]`. Each turn: `state_new = alpha * state + (1 - alpha) * output`.
    Simplest option, good for slowly-changing context.

  - **`:gru`** — GRU-style gated update. State shape: `[batch, state_size]`.
    Each turn: backbone output acts as input to a GRU cell that updates state.
    Good for learning what to remember/forget.

  ## Usage

      model = StatefulAgent.build(
        embed_dim: 64,
        backbone: :decoder_only,
        backbone_opts: [hidden_size: 128, num_layers: 2, num_heads: 4],
        state_size: 64,
        state_mode: :compressive
      )

      # Model expects two inputs: "state_sequence" and "agent_state"
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates, Axon.ModelState.empty())

      # Turn 1: pass zero state
      {output_1, state_1} = predict_fn.(params, %{
        "state_sequence" => input_1,
        "agent_state" => zero_state
      })

      # Turn 2: pass state from turn 1
      {output_2, state_2} = predict_fn.(params, %{
        "state_sequence" => input_2,
        "agent_state" => state_1
      })

  ## References
  - Munkhdalai et al., "Leave No Context Behind: Efficient Infinite Context Transformers" (2024)
  - Peng et al., "Titans: Learning to Memorize at Test Time" (2025)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @doc """
  Build a Stateful Agent model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:backbone` - Backbone architecture atom or `:transformer` (default: `:transformer`)
    - `:backbone_opts` - Options passed to backbone (default: [])
    - `:hidden_size` - Backbone hidden size (default: 128)
    - `:num_layers` - Backbone transformer layers (default: 2)
    - `:num_heads` - Attention heads (default: 4)
    - `:state_size` - Persistent state dimension (default: 64)
    - `:state_mode` - State update mode: `:compressive`, `:ema`, `:gru` (default: `:compressive`)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `{output, new_state}` where output is
    `[batch, hidden_size]` and new_state is `[batch, state_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:backbone, atom()}
          | {:backbone_opts, keyword()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:state_size, pos_integer()}
          | {:state_mode, :compressive | :ema | :gru}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 128)
    num_layers = Keyword.get(opts, :num_layers, 2)
    num_heads = Keyword.get(opts, :num_heads, 4)
    state_size = Keyword.get(opts, :state_size, 64)
    state_mode = Keyword.get(opts, :state_mode, :compressive)
    dropout = Keyword.get(opts, :dropout, 0.1)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Two inputs: current turn sequence + previous state
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})
    prev_state = Axon.input("agent_state", shape: {nil, state_size})

    # Project input to hidden size
    projected =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Inject previous state into sequence via cross-attention-like conditioning
    # Project state to hidden_size, broadcast to [batch, 1, hidden], concat to input
    state_projected = Axon.dense(prev_state, hidden_size, name: "state_to_hidden")

    conditioned =
      Axon.layer(
        fn seq, state_h, _opts ->
          # state_h: [batch, hidden] -> [batch, 1, hidden]
          state_token = Nx.new_axis(state_h, 1)
          # Prepend state token to sequence: [batch, 1+seq, hidden]
          Nx.concatenate([state_token, seq], axis: 1)
        end,
        [projected, state_projected],
        name: "prepend_state_token"
      )

    # Backbone: transformer stack processing conditioned input
    backbone_out =
      Enum.reduce(1..num_layers, conditioned, fn layer_idx, acc ->
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
          name: "backbone_block_#{layer_idx}"
        )
      end)

    # Final norm on backbone
    backbone_out = Axon.layer_norm(backbone_out, name: "backbone_norm")

    # Extract output: last timestep of the original sequence positions
    # (skip the prepended state token)
    output =
      Axon.nx(backbone_out, fn tensor ->
        seq_size = Nx.axis_size(tensor, 1)
        # Last token of original sequence (index seq_size - 1)
        Nx.slice_along_axis(tensor, seq_size - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end, name: "extract_output")

    # Extract backbone summary for state update: mean-pool over original sequence
    backbone_summary =
      Axon.nx(backbone_out, fn tensor ->
        # Skip state token (index 0), pool over rest
        seq_size = Nx.axis_size(tensor, 1)
        seq_tokens = Nx.slice_along_axis(tensor, 1, seq_size - 1, axis: 1)
        Nx.mean(seq_tokens, axes: [1])
      end, name: "backbone_summary")

    # State update based on mode
    new_state =
      case state_mode do
        :compressive -> build_compressive_update(backbone_summary, prev_state, hidden_size, state_size)
        :ema -> build_ema_update(backbone_summary, prev_state, hidden_size, state_size)
        :gru -> build_gru_update(backbone_summary, prev_state, hidden_size, state_size)
      end

    # Return tuple: {output, new_state}
    Axon.container({output, new_state})
  end

  @doc """
  Get the output size of a StatefulAgent model.

  Returns `{output_size, state_size}`.
  """
  @spec output_size(keyword()) :: {pos_integer(), pos_integer()}
  def output_size(opts \\ []) do
    hidden = Keyword.get(opts, :hidden_size, 128)
    state = Keyword.get(opts, :state_size, 64)
    {hidden, state}
  end

  @doc """
  Get recommended defaults for StatefulAgent.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 128,
      num_layers: 2,
      num_heads: 4,
      state_size: 64,
      state_mode: :compressive,
      dropout: 0.1,
      window_size: 60
    ]
  end

  # Compressive state update: gated linear combination
  # new_state = gate * prev_state + (1 - gate) * W_write(backbone_summary)
  defp build_compressive_update(backbone_summary, prev_state, _hidden_size, state_size) do
    # Project backbone output to state space
    write_val = Axon.dense(backbone_summary, state_size, name: "state_write")

    # Gate: sigmoid(W_gate @ [backbone_summary, prev_state])
    gate_input = Axon.concatenate([backbone_summary, prev_state], axis: 1, name: "gate_concat")
    gate = Axon.dense(gate_input, state_size, name: "state_gate", activation: :sigmoid)

    # Gated update
    Axon.layer(
      fn write, prev, g, _opts ->
        Nx.add(Nx.multiply(g, prev), Nx.multiply(Nx.subtract(1.0, g), write))
      end,
      [write_val, prev_state, gate],
      name: "compressive_update"
    )
  end

  # EMA state update: exponential moving average
  # new_state = alpha * prev_state + (1 - alpha) * W_project(backbone_summary)
  defp build_ema_update(backbone_summary, prev_state, _hidden_size, state_size) do
    write_val = Axon.dense(backbone_summary, state_size, name: "state_write")

    # Learnable alpha per dimension
    alpha_param = Axon.param("ema_alpha", {1, state_size}, initializer: :zeros)

    Axon.layer(
      fn write, prev, alpha_raw, _opts ->
        alpha = Nx.sigmoid(alpha_raw)
        Nx.add(Nx.multiply(alpha, prev), Nx.multiply(Nx.subtract(1.0, alpha), write))
      end,
      [write_val, prev_state, alpha_param],
      name: "ema_update"
    )
  end

  # GRU-style state update
  # z = sigmoid(W_z @ [backbone_summary, prev_state])     — update gate
  # r = sigmoid(W_r @ [backbone_summary, prev_state])     — reset gate
  # candidate = tanh(W_h @ [backbone_summary, r * prev_state])
  # new_state = (1 - z) * prev_state + z * candidate
  defp build_gru_update(backbone_summary, prev_state, _hidden_size, state_size) do
    concat = Axon.concatenate([backbone_summary, prev_state], axis: 1, name: "gru_concat")

    # Update gate
    z_gate = Axon.dense(concat, state_size, name: "gru_z_gate", activation: :sigmoid)

    # Reset gate
    r_gate = Axon.dense(concat, state_size, name: "gru_r_gate", activation: :sigmoid)

    # Reset-gated state
    reset_state =
      Axon.layer(
        fn r, prev, _opts -> Nx.multiply(r, prev) end,
        [r_gate, prev_state],
        name: "gru_reset_state"
      )

    # Candidate: use backbone_summary + reset_state
    candidate_input =
      Axon.concatenate([backbone_summary, reset_state], axis: 1, name: "gru_candidate_concat")

    candidate = Axon.dense(candidate_input, state_size, name: "gru_candidate", activation: :tanh)

    # Final update: (1 - z) * prev + z * candidate
    Axon.layer(
      fn z, prev, cand, _opts ->
        Nx.add(
          Nx.multiply(Nx.subtract(1.0, z), prev),
          Nx.multiply(z, cand)
        )
      end,
      [z_gate, prev_state, candidate],
      name: "gru_update"
    )
  end
end
