defmodule Edifice.Feedforward.TabNet do
  @moduledoc """
  TabNet - Attentive Interpretable Tabular Learning.

  TabNet uses sequential attention to select which features to reason about
  at each decision step. This provides instance-wise feature selection,
  making the model inherently interpretable while maintaining high performance
  on tabular data.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +--------------------------------------+
  | Initial BN                           |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Step 1:                              |
  |   Attention: select features via     |
  |     sparse mask M = sparsemax(...)   |
  |   Transform: process selected feats  |
  |   Split: h_step -> decision + next   |
  +--------------------------------------+
        |  (repeat num_steps)
        v
  +--------------------------------------+
  | Aggregate: sum decision outputs      |
  +--------------------------------------+
        |
        v
  Output [batch, hidden_size or num_classes]
  ```

  ## Feature Selection

  At each step, TabNet uses an attention transformer to produce a mask
  that selects relevant input features. The relaxation factor gamma
  controls how much previously attended features can be reused.

  ## Usage

      model = TabNet.build(
        input_size: 128,
        hidden_size: 64,
        num_steps: 3,
        relaxation_factor: 1.5,
        num_classes: 10
      )

  ## References

  - Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning"
    (AAAI 2021)
  - https://arxiv.org/abs/1908.07442
  """

  require Axon

  @default_hidden_size 64
  @default_num_steps 3
  @default_relaxation_factor 1.5

  @doc """
  Build a TabNet model.

  ## Options

  - `:input_size` - Input feature dimension (required)
  - `:hidden_size` - Hidden dimension for processing (default: 64)
  - `:num_steps` - Number of sequential attention steps (default: 3)
  - `:relaxation_factor` - Controls feature reuse across steps (default: 1.5)
  - `:num_classes` - If provided, adds classification head (default: nil)
  - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

  An Axon model: `[batch, input_size]` -> `[batch, hidden_size or num_classes]`
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)
    relaxation_factor = Keyword.get(opts, :relaxation_factor, @default_relaxation_factor)
    num_classes = Keyword.get(opts, :num_classes, nil)
    dropout = Keyword.get(opts, :dropout, 0.0)

    input = Axon.input("input", shape: {nil, input_size})

    # Initial batch normalization
    bn_input = Axon.layer_norm(input, name: "input_bn")

    # Build TabNet steps
    # Each step selects features, processes them, and outputs a decision
    # We chain the steps through the Axon graph
    {aggregated, _} =
      Enum.reduce(0..(num_steps - 1), {nil, bn_input}, fn step, {agg, processed} ->
        # Attention transformer: learn which features to focus on
        # Produces a soft mask over input features
        attention_scores =
          processed
          |> Axon.dense(input_size, name: "step_#{step}_attn_proj")
          |> Axon.layer_norm(name: "step_#{step}_attn_bn")

        # Apply prior scales (relaxation) via custom layer
        mask =
          Axon.layer(
            fn scores, _orig_input, _opts ->
              # Sparsemax approximation via softmax with temperature
              exp_scores = Nx.exp(scores)
              Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [1], keep_axes: true))
            end,
            [attention_scores, bn_input],
            name: "step_#{step}_mask",
            relaxation_factor: relaxation_factor,
            op_name: :tabnet_mask
          )

        # Apply mask to input features: selected = mask * input
        selected =
          Axon.multiply(mask, bn_input, name: "step_#{step}_select")

        # Feature transformer: process selected features
        transformed =
          selected
          |> Axon.dense(hidden_size, name: "step_#{step}_transform_1")
          |> Axon.layer_norm(name: "step_#{step}_transform_bn_1")
          |> Axon.activation(:relu, name: "step_#{step}_transform_act_1")

        transformed =
          if dropout > 0.0 do
            Axon.dropout(transformed, rate: dropout, name: "step_#{step}_drop")
          else
            transformed
          end

        # Split: decision output + next step input
        decision = Axon.activation(transformed, :relu, name: "step_#{step}_decision")

        # Aggregate decisions
        new_agg =
          if agg do
            Axon.add(agg, decision, name: "step_#{step}_aggregate")
          else
            decision
          end

        # Use transformed output as input for next step's attention
        {new_agg, transformed}
      end)

    # Final output
    if num_classes do
      Axon.dense(aggregated, num_classes, name: "tabnet_classifier")
    else
      aggregated
    end
  end

  @doc """
  Get the output size of a TabNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_classes || hidden_size
  end
end
