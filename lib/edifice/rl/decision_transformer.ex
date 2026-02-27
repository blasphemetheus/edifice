defmodule Edifice.RL.DecisionTransformer do
  @moduledoc """
  Decision Transformer for offline reinforcement learning.

  Frames offline RL as conditional sequence generation. Given a desired
  return-to-go, past states, and past actions, predicts the next action using
  a causal GPT-style transformer over an interleaved token sequence.

  ## Architecture

  ```
  Returns-to-go [B, K]   States [B, K, state_dim]   Actions [B, K, action_dim]   Timesteps [B, K]
        |                       |                         |                           |
   Dense→hidden            Dense→hidden              Dense→hidden              Embed→hidden
        |                       |                         |                           |
        +--- add timestep_embed +--- add timestep_embed --+                           |
        |                       |                         |
   Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...] → [B, 3K, hidden_size]
        |
   Causal GPT (num_layers × TransformerBlock)
        |
   Extract action positions (indices 2, 5, 8, ...) → [B, K, hidden_size]
        |
   Dense → action_dim → [B, K, action_dim]
  ```

  For inference, feed context and extract the last action prediction.

  ## Usage

      model = DecisionTransformer.build(
        state_dim: 64,
        action_dim: 8,
        hidden_size: 128,
        num_heads: 4,
        num_layers: 3,
        context_len: 20
      )

  ## References

  - Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
  """

  alias Edifice.Blocks.TransformerBlock

  @default_hidden_size 128
  @default_num_layers 3
  @default_num_heads 4
  @default_context_len 20
  @default_max_timestep 1000
  @default_dropout 0.1
  @default_activation :gelu

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:state_dim, pos_integer()}
          | {:action_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:context_len, pos_integer()}
          | {:max_timestep, pos_integer()}
          | {:dropout, float()}
          | {:activation, :gelu | :relu | :silu | :tanh}

  @doc """
  Build a Decision Transformer model.

  ## Options

    - `:state_dim` - State observation dimension (required)
    - `:action_dim` - Action output dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 128)
    - `:num_layers` - Number of transformer blocks (default: 3)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:context_len` - Number of timesteps in context window (default: 20)
    - `:max_timestep` - Maximum timestep for embedding table (default: 1000)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:activation` - FFN activation (default: :gelu)

  ## Inputs

  The model expects four named inputs:
    - `"returns"` - `{batch, context_len}` scalar return-to-go per step
    - `"states"` - `{batch, context_len, state_dim}`
    - `"actions"` - `{batch, context_len, action_dim}`
    - `"timesteps"` - `{batch, context_len}` integer indices

  ## Returns

    An Axon model outputting `{batch, context_len, action_dim}`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    state_dim = Keyword.fetch!(opts, :state_dim)
    action_dim = Keyword.fetch!(opts, :action_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    context_len = Keyword.get(opts, :context_len, @default_context_len)
    max_timestep = Keyword.get(opts, :max_timestep, @default_max_timestep)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)

    # Inputs
    returns = Axon.input("returns", shape: {nil, context_len})
    states = Axon.input("states", shape: {nil, context_len, state_dim})
    actions = Axon.input("actions", shape: {nil, context_len, action_dim})
    timesteps = Axon.input("timesteps", shape: {nil, context_len})

    # Embed each modality to hidden_size
    # Returns: [B, K] -> [B, K, 1] -> dense -> [B, K, hidden_size]
    returns_embed =
      returns
      |> Axon.nx(fn r -> Nx.new_axis(r, -1) end, name: "dt_returns_expand")
      |> Axon.dense(hidden_size, name: "dt_returns_embed")

    # States: [B, K, state_dim] -> dense -> [B, K, hidden_size]
    states_embed = Axon.dense(states, hidden_size, name: "dt_states_embed")

    # Actions: [B, K, action_dim] -> dense -> [B, K, hidden_size]
    actions_embed = Axon.dense(actions, hidden_size, name: "dt_actions_embed")

    # Timestep embedding: [B, K] -> embedding lookup -> [B, K, hidden_size]
    timestep_embed =
      Axon.embedding(timesteps, max_timestep, hidden_size, name: "dt_timestep_embed")

    # Add timestep embeddings to each modality
    returns_embed = Axon.add(returns_embed, timestep_embed, name: "dt_returns_pos")
    states_embed = Axon.add(states_embed, timestep_embed, name: "dt_states_pos")
    actions_embed = Axon.add(actions_embed, timestep_embed, name: "dt_actions_pos")

    # Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...] -> [B, 3K, hidden_size]
    interleaved =
      Axon.layer(
        &interleave_impl/4,
        [returns_embed, states_embed, actions_embed],
        name: "dt_interleave",
        context_len: context_len,
        hidden_size: hidden_size,
        op_name: :interleave
      )

    # Apply dropout
    interleaved = maybe_dropout(interleaved, dropout, "dt_embed_dropout")

    # Stack transformer blocks with causal attention
    head_dim = div(hidden_size, num_heads)
    seq_len_3k = 3 * context_len

    transformed =
      TransformerBlock.stack(interleaved, num_layers,
        attention_fn: fn x, attn_name ->
          build_causal_attention(x,
            hidden_size: hidden_size,
            num_heads: num_heads,
            head_dim: head_dim,
            seq_len: seq_len_3k,
            name: attn_name
          )
        end,
        hidden_size: hidden_size,
        dropout: dropout,
        name: "dt_transformer"
      )

    # Final layer norm
    normed = Axon.layer_norm(transformed, name: "dt_final_norm")

    # Extract action-position tokens (indices 2, 5, 8, ... i.e. every 3rd starting at 2)
    extracted =
      Axon.layer(
        &extract_action_tokens_impl/2,
        [normed],
        name: "dt_extract_actions",
        context_len: context_len,
        hidden_size: hidden_size,
        op_name: :extract_action_tokens
      )

    # Project to action dimension
    Axon.dense(extracted, action_dim,
      activation: activation,
      name: "dt_action_head"
    )
  end

  @doc """
  Get the output size (action_dim).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :action_dim)
  end

  # ===========================================================================
  # Custom layer implementations
  # ===========================================================================

  # Interleave returns, states, actions into [B, 3K, hidden_size]
  defp interleave_impl(returns, states, actions, _opts) do
    batch = Nx.axis_size(returns, 0)
    context_len = Nx.axis_size(returns, 1)
    hidden_size = Nx.axis_size(returns, 2)

    # Stack along a new axis: [B, K, 3, hidden_size]
    stacked = Nx.stack([returns, states, actions], axis: 2)
    # Reshape to [B, 3*K, hidden_size]
    Nx.reshape(stacked, {batch, 3 * context_len, hidden_size})
  end

  # Extract action tokens at positions 2, 5, 8, ... from [B, 3K, hidden_size]
  defp extract_action_tokens_impl(x, opts) do
    context_len = opts[:context_len]
    # Action positions are at indices 2, 5, 8, ... (every 3rd starting at 2)
    indices = Nx.tensor(Enum.map(0..(context_len - 1), fn i -> 3 * i + 2 end))
    Nx.take(x, indices, axis: 1)
  end

  # Build causal multi-head self-attention
  defp build_causal_attention(input, opts) do
    hidden_size = opts[:hidden_size]
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    seq_len = opts[:seq_len]
    name = opts[:name]

    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        seq_len: seq_len,
        op_name: :causal_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # Causal multi-head attention implementation
  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(causal_mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    # Weighted sum: [batch, heads, seq, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden_size]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
