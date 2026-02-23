defmodule Edifice.RL.PolicyValue do
  @moduledoc """
  Policy-Value network for reinforcement learning.

  Shared-trunk actor-critic architecture with separate policy and value heads.
  Suitable for PPO, A2C, and other policy gradient methods.

  ## Architecture

  ```
  Input [batch, input_size]
        |
  +==================+
  |  Shared Trunk    |
  |  dense → GELU    |
  |  dense → GELU    |
  +==================+
        |
  +-----+-----+
  |           |
  v           v
  Policy     Value
  Head       Head
  |           |
  v           v
  [batch,    [batch]
  action_size]
  ```

  ## Action Types

  - `:discrete` — Policy outputs softmax probabilities over discrete actions
  - `:continuous` — Policy outputs tanh-squashed values in [-1, 1]

  ## Returns

  An Axon model outputting `%{policy: ..., value: ...}` via `Axon.container`.

  ## Usage

      model = PolicyValue.build(
        input_size: 64,
        action_size: 4,
        action_type: :discrete,
        hidden_size: 128
      )

  For a complete PPO training loop, see the exphil project which builds
  on these primitives.

  ## References

  - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
  - Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (A3C, 2016)
  """

  @default_hidden_size 64

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:action_size, pos_integer()}
          | {:action_type, :discrete | :continuous}
          | {:hidden_size, pos_integer()}

  @doc """
  Build a policy-value network.

  ## Options

    - `:input_size` - Input observation dimension (required)
    - `:action_size` - Number of actions (discrete) or action dimensions (continuous) (required)
    - `:action_type` - `:discrete` or `:continuous` (default: `:discrete`)
    - `:hidden_size` - Hidden layer size (default: 64)

  ## Returns

    An Axon model outputting `%{policy: [batch, action_size], value: [batch]}`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    action_size = Keyword.fetch!(opts, :action_size)
    action_type = Keyword.get(opts, :action_type, :discrete)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)

    input = Axon.input("observation", shape: {nil, input_size})

    # Shared trunk
    trunk =
      input
      |> Axon.dense(hidden_size, name: "pv_trunk_dense1")
      |> Axon.activation(:gelu, name: "pv_trunk_act1")
      |> Axon.dense(hidden_size, name: "pv_trunk_dense2")
      |> Axon.activation(:gelu, name: "pv_trunk_act2")

    # Policy head
    policy_logits = Axon.dense(trunk, action_size, name: "pv_policy_head")

    policy =
      case action_type do
        :discrete ->
          Axon.softmax(policy_logits, name: "pv_policy_softmax")

        :continuous ->
          Axon.tanh(policy_logits, name: "pv_policy_tanh")
      end

    # Value head
    value =
      trunk
      |> Axon.dense(1, name: "pv_value_head")
      |> Axon.nx(fn x -> Nx.squeeze(x, axes: [1]) end, name: "pv_value_squeeze")

    Axon.container(%{policy: policy, value: value})
  end

  @doc "Get the output size (action_size for policy head)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :action_size)
  end
end
