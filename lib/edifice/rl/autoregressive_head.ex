defmodule Edifice.RL.AutoregressiveHead do
  @moduledoc """
  Autoregressive action head with cross-component conditioning.

  Produces logits for a multi-component action space where each component's
  distribution is conditioned on the backbone hidden state *and* all
  previously selected components. During training, ground-truth previous
  actions are fed in (teacher forcing). During inference, components are
  sampled sequentially.

  ## Architecture

  ```
  Backbone hidden [batch, hidden_size]
        |
        v
  Component 0 logits (conditioned on: hidden)
        |  embed(action_0)
        v
  Component 1 logits (conditioned on: hidden + embed_0)
        |  embed(action_1)
        v
  Component 2 logits (conditioned on: hidden + embed_0 + embed_1)
        ...
  ```

  Each component uses a small MLP that takes `[hidden; prev_action_embeds]`
  and outputs logits. An embedding layer converts each discrete action into
  a vector for conditioning the next component.

  ## Usage

      # Melee controller: 6 components
      model = AutoregressiveHead.build(
        hidden_size: 256,
        components: [
          %{name: "buttons",  num_actions: 8, embed_dim: 16},
          %{name: "main_x",   num_actions: 17, embed_dim: 16},
          %{name: "main_y",   num_actions: 17, embed_dim: 16},
          %{name: "c_x",      num_actions: 17, embed_dim: 16},
          %{name: "c_y",      num_actions: 17, embed_dim: 16},
          %{name: "shoulder", num_actions: 5, embed_dim: 8}
        ]
      )

      # Training (teacher forcing): provide ground truth actions
      output = predict_fn.(params, %{
        "hidden" => backbone_output,
        "action_buttons" => gt_buttons,     # [batch] integer
        "action_main_x" => gt_main_x,       # [batch] integer
        ...
      })
      # output = %{buttons: [batch, 8], main_x: [batch, 17], ...}

  ## Game AI Context

  AlphaStar (Vinyals et al., 2019) showed that autoregressive action
  decomposition is critical for complex action spaces. In Melee, controller
  components are highly correlated — pressing B + stick angle = special move.
  Independent heads can't learn these correlations; autoregressive conditioning
  captures them explicitly.

  ## References

  - Vinyals et al., "Grandmaster level in StarCraft II using multi-agent RL" (2019)
  """

  @default_component_hidden 64

  @typedoc "Configuration for one action component."
  @type component_config :: %{
          name: String.t(),
          num_actions: pos_integer(),
          embed_dim: pos_integer()
        }

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:components, [component_config()]}
          | {:component_hidden, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build an autoregressive action head for training (teacher forcing).

  ## Options

    - `:hidden_size` - Backbone output dimension (required)
    - `:components` - List of component configs, each with `:name`,
      `:num_actions`, `:embed_dim` (required)
    - `:component_hidden` - Hidden size in per-component MLPs (default: #{@default_component_hidden})
    - `:dropout` - Dropout rate (default: 0.0)

  ## Inputs

    - `"hidden"` — Backbone output `[batch, hidden_size]`
    - `"action_{name}"` — Ground-truth action for each component `[batch]` (integer)
      Used for teacher forcing. Each component except the first needs the
      previous components' ground-truth actions.

  ## Returns

    An Axon model outputting a map of `%{component_name: logits}` via `Axon.container`.
    Each logits tensor has shape `[batch, num_actions]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    components = Keyword.fetch!(opts, :components)
    component_hidden = Keyword.get(opts, :component_hidden, @default_component_hidden)
    dropout = Keyword.get(opts, :dropout, 0.0)

    # Input: backbone hidden state
    hidden = Axon.input("hidden", shape: {nil, hidden_size})

    # Build each component's embedding layer and logit head
    # Accumulate action embeddings from previous components
    {logits_map, _} =
      Enum.reduce(components, {%{}, []}, fn component, {logits_acc, prev_embeds} ->
        comp_name = Map.fetch!(component, :name)
        num_actions = Map.fetch!(component, :num_actions)
        embed_dim = Map.fetch!(component, :embed_dim)

        # Condition on: hidden + all previous action embeddings
        conditioning =
          case prev_embeds do
            [] ->
              hidden

            embeds ->
              # Concatenate hidden + all previous embeddings
              Axon.concatenate([hidden | Enum.reverse(embeds)],
                axis: 1,
                name: "#{comp_name}_condition_concat"
              )
          end

        # Small MLP → logits
        logits =
          conditioning
          |> Axon.dense(component_hidden, name: "#{comp_name}_mlp1")
          |> Axon.activation(:relu)
          |> Axon.dropout(rate: dropout, name: "#{comp_name}_drop")
          |> Axon.dense(num_actions, name: "#{comp_name}_logits")

        # Ground-truth action input for teacher forcing
        action_input = Axon.input("action_#{comp_name}", shape: {nil})

        # Embed the action for conditioning next components
        action_embed =
          Axon.embedding(action_input, num_actions, embed_dim,
            name: "#{comp_name}_action_embed"
          )

        logits_map = Map.put(logits_acc, String.to_atom(comp_name), logits)
        {logits_map, [action_embed | prev_embeds]}
      end)

    Axon.container(logits_map)
  end

  @doc """
  Build an autoregressive head for inference (sequential sampling).

  Unlike `build/1`, this doesn't require ground-truth action inputs.
  Instead, it samples from each component's distribution and feeds the
  sampled action to the next component.

  Note: This uses argmax (greedy) selection. For stochastic sampling,
  use `build/1` with teacher forcing disabled at the application level.

  ## Options

  Same as `build/1` but no `"action_*"` inputs needed.

  ## Returns

    An Axon model outputting `%{logits: %{...}, actions: %{...}}`.
  """
  @spec build_inference(keyword()) :: Axon.t()
  def build_inference(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    components = Keyword.fetch!(opts, :components)
    component_hidden = Keyword.get(opts, :component_hidden, @default_component_hidden)
    dropout = Keyword.get(opts, :dropout, 0.0)

    hidden = Axon.input("hidden", shape: {nil, hidden_size})

    # Build sequentially: each component produces logits, samples, embeds
    {logits_map, actions_map, _} =
      Enum.reduce(components, {%{}, %{}, []}, fn component, {log_acc, act_acc, prev_embeds} ->
        comp_name = Map.fetch!(component, :name)
        num_actions = Map.fetch!(component, :num_actions)
        embed_dim = Map.fetch!(component, :embed_dim)

        conditioning =
          case prev_embeds do
            [] -> hidden
            embeds ->
              Axon.concatenate([hidden | Enum.reverse(embeds)],
                axis: 1,
                name: "#{comp_name}_inf_concat"
              )
          end

        logits =
          conditioning
          |> Axon.dense(component_hidden, name: "#{comp_name}_inf_mlp1")
          |> Axon.activation(:relu)
          |> Axon.dropout(rate: dropout, name: "#{comp_name}_inf_drop")
          |> Axon.dense(num_actions, name: "#{comp_name}_inf_logits")

        # Greedy selection: argmax
        sampled_action =
          Axon.nx(logits, fn l -> Nx.argmax(l, axis: 1) end,
            name: "#{comp_name}_inf_sample"
          )

        # Embed sampled action for next component
        action_embed =
          Axon.embedding(sampled_action, num_actions, embed_dim,
            name: "#{comp_name}_inf_embed"
          )

        log_acc = Map.put(log_acc, String.to_atom(comp_name), logits)
        act_acc = Map.put(act_acc, String.to_atom(comp_name), sampled_action)
        {log_acc, act_acc, [action_embed | prev_embeds]}
      end)

    Axon.container(%{logits: logits_map, actions: actions_map})
  end

  @doc """
  Return the total number of action dimensions across all components.
  """
  @spec total_actions(keyword()) :: pos_integer()
  def total_actions(opts) do
    components = Keyword.fetch!(opts, :components)
    Enum.reduce(components, 0, fn c, acc -> acc + Map.fetch!(c, :num_actions) end)
  end

  @doc """
  Recommended defaults for Melee controller (6 components).
  """
  @spec melee_defaults() :: keyword()
  def melee_defaults do
    [
      hidden_size: 256,
      components: [
        %{name: "buttons", num_actions: 8, embed_dim: 16},
        %{name: "main_x", num_actions: 17, embed_dim: 16},
        %{name: "main_y", num_actions: 17, embed_dim: 16},
        %{name: "c_x", num_actions: 17, embed_dim: 16},
        %{name: "c_y", num_actions: 17, embed_dim: 16},
        %{name: "shoulder", num_actions: 5, embed_dim: 8}
      ],
      component_hidden: 64,
      dropout: 0.1
    ]
  end
end
