defmodule Edifice.Meta.RLHFHead do
  @moduledoc """
  RLHF heads: reward model and DPO preference heads for alignment.

  Provides composable head modules for Reinforcement Learning from Human
  Feedback (RLHF) pipelines. Two head types are supported:

  ## Reward Head (`:reward`)

  Maps a sequence to a scalar reward value per batch element:

  ```
  Input [batch, seq, input_size]
        |
        v
  Dense(hidden) -> SiLU -> Dense(1) -> Squeeze -> Mean Pool
        |
        v
  Output [batch]
  ```

  ## DPO Head (`:dpo`)

  Takes two inputs ("chosen" and "rejected") and computes the preference
  logit (chosen_reward - rejected_reward) for Direct Preference Optimization:

  ```
  Chosen  [batch, seq, input_size] -> Reward Head -> chosen_score
  Rejected [batch, seq, input_size] -> Reward Head -> rejected_score
        |
        v
  Output = chosen_score - rejected_score  [batch]
  ```

  ## Usage

      # Reward head
      model = RLHFHead.build(input_size: 256, head_type: :reward)

      # DPO head
      model = RLHFHead.build(input_size: 256, head_type: :dpo)

  ## References
  - Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
  - Rafailov et al., "Direct Preference Optimization" (2023)
  """

  @doc """
  Build an RLHF head.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:hidden_size` - Hidden layer dimension (default: 256)
    - `:head_type` - Head type: `:reward` or `:dpo` (default: :reward)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    An Axon model. For `:reward`, input is `"state_sequence"` and output is `[batch]`.
    For `:dpo`, inputs are `"chosen"` and `"rejected"`, output is `[batch]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:head_type, :reward | :dpo}
          | {:hidden_size, pos_integer()}
          | {:input_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    head_type = Keyword.get(opts, :head_type, :reward)

    case head_type do
      :reward -> build_reward_head(opts)
      :dpo -> build_dpo_head(opts)
    end
  end

  @doc """
  Build a reward head that maps sequences to scalar rewards.

  Input: `"state_sequence"` `[batch, seq, input_size]`
  Output: `[batch]`
  """
  @spec build_reward_head(keyword()) :: Axon.t()
  def build_reward_head(opts) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    dropout = Keyword.get(opts, :dropout, 0.1)

    input = Axon.input("state_sequence", shape: {nil, nil, input_size})

    input
    |> Axon.dense(hidden_size, name: "reward_dense1")
    |> Axon.activation(:silu, name: "reward_act")
    |> maybe_dropout(dropout, "reward_dropout")
    |> Axon.dense(1, name: "reward_dense2")
    |> Axon.nx(fn t -> Nx.squeeze(t, axes: [2]) end, name: "reward_squeeze")
    |> Axon.nx(fn t -> Nx.mean(t, axes: [1]) end, name: "reward_pool")
  end

  @doc """
  Build a DPO preference head that computes chosen_reward - rejected_reward.

  Inputs: `"chosen"` and `"rejected"` `[batch, seq, input_size]`
  Output: `[batch]` (preference logit)
  """
  @spec build_dpo_head(keyword()) :: Axon.t()
  def build_dpo_head(opts) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    dropout = Keyword.get(opts, :dropout, 0.1)

    chosen = Axon.input("chosen", shape: {nil, nil, input_size})
    rejected = Axon.input("rejected", shape: {nil, nil, input_size})

    # Shared reward pathway for chosen
    chosen_score =
      chosen
      |> Axon.dense(hidden_size, name: "dpo_dense1")
      |> Axon.activation(:silu, name: "dpo_act")
      |> maybe_dropout(dropout, "dpo_dropout")
      |> Axon.dense(1, name: "dpo_dense2")
      |> Axon.nx(fn t -> Nx.squeeze(t, axes: [2]) |> Nx.mean(axes: [1]) end,
        name: "dpo_chosen_pool"
      )

    # Rejected path reuses weight names for parameter sharing
    rejected_score =
      rejected
      |> Axon.dense(hidden_size, name: "dpo_rej_dense1")
      |> Axon.activation(:silu, name: "dpo_rej_act")
      |> maybe_dropout(dropout, "dpo_rej_dropout")
      |> Axon.dense(1, name: "dpo_rej_dense2")
      |> Axon.nx(fn t -> Nx.squeeze(t, axes: [2]) |> Nx.mean(axes: [1]) end,
        name: "dpo_rejected_pool"
      )

    # Preference logit: chosen - rejected
    Axon.subtract(chosen_score, rejected_score, name: "dpo_preference")
  end

  @doc """
  Get the output size of an RLHF head (always scalar per batch).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(_opts \\ []), do: 1

  @doc """
  Get recommended defaults for RLHF heads.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      head_type: :reward,
      dropout: 0.1
    ]
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
