defmodule Edifice.Interpretability.ConceptBottleneck do
  @moduledoc """
  Concept Bottleneck Model (CBM) for inherently interpretable prediction.

  Forces predictions through a human-interpretable concept layer. The model
  first predicts a set of known concepts (e.g., "has wings", "is red"), then
  uses only those concept predictions to make the final task prediction.

  ## Architecture

  ```
  Input [batch, input_size]
        |
  Concept predictor: dense(num_concepts) + sigmoid
        |
  [batch, num_concepts]  (interpretable concept activations)
        |
  Task predictor: dense(num_classes) + softmax
        |
  Output [batch, num_classes]
  ```

  ## Key Properties

  - **Interpretable by design**: inspect which concepts activated for each prediction
  - **Concept intervention**: override a concept prediction at test time to change behavior
  - **Requires concept labels**: training needs both task labels AND concept annotations

  ## Usage

      # Bird classification via interpretable attributes
      model = ConceptBottleneck.build(
        input_size: 2048,
        num_concepts: 112,   # e.g., "has_crown", "wing_color_brown"
        num_classes: 200      # bird species
      )

  ## References

  - Koh et al., "Concept Bottleneck Models" (ICML 2020)
  """

  @default_num_classes 2

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:num_concepts, pos_integer()}
          | {:num_classes, pos_integer()}

  @doc """
  Build a concept bottleneck model.

  ## Options

    - `:input_size` - Dimension of input features (required)
    - `:num_concepts` - Number of interpretable concepts in the bottleneck (required)
    - `:num_classes` - Number of output classes (default: #{@default_num_classes})

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, num_classes]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    num_concepts = Keyword.fetch!(opts, :num_concepts)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)

    input = Axon.input("cbm_input", shape: {nil, input_size})

    # Concept predictor (bottleneck)
    concepts = Axon.dense(input, num_concepts, name: "cbm_concept_predictor")
    concepts = Axon.activation(concepts, :sigmoid, name: "cbm_concept_activation")

    # Task predictor
    logits = Axon.dense(concepts, num_classes, name: "cbm_task_predictor")
    Axon.activation(logits, :softmax, name: "cbm_task_output")
  end

  @doc """
  Build only the concept predictor (for inspecting or training concepts).

  Returns concept activations `[batch, num_concepts]` with sigmoid outputs.
  """
  @spec build_concept_predictor([build_opt()]) :: Axon.t()
  def build_concept_predictor(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    num_concepts = Keyword.fetch!(opts, :num_concepts)

    input = Axon.input("cbm_input", shape: {nil, input_size})
    concepts = Axon.dense(input, num_concepts, name: "cbm_concept_predictor")
    Axon.activation(concepts, :sigmoid, name: "cbm_concept_activation")
  end

  @doc "Get the output size of the concept bottleneck model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end
end
