defmodule Edifice.Interpretability.LinearProbe do
  @moduledoc """
  Linear Probe for interpretability analysis.

  A single linear layer trained to predict a target concept from frozen model
  activations. If a linear probe can decode concept X from layer L's activations,
  then layer L linearly represents X.

  ## Architecture

  ```
  Input [batch, input_size]  (frozen activations from target layer)
        |
  Dense: input_size → num_classes (classification)
     or: input_size → 1           (regression)
        |
  Output [batch, num_classes]  or  [batch, 1]
  ```

  ## Tasks

  - `:classification` — Multi-class probing with softmax output
  - `:binary` — Binary probing with sigmoid output
  - `:regression` — Continuous value probing with linear output

  ## Usage

      # Classification probe: does layer 6 encode part-of-speech?
      probe = LinearProbe.build(
        input_size: 768,
        num_classes: 17,
        task: :classification
      )

      # Binary probe: does layer 4 encode sentiment?
      probe = LinearProbe.build(
        input_size: 768,
        num_classes: 1,
        task: :binary
      )

      # Regression probe: does layer 8 encode word frequency?
      probe = LinearProbe.build(
        input_size: 768,
        num_classes: 1,
        task: :regression
      )

  ## References

  - Alain & Bengio, "Understanding intermediate layers using linear classifier
    probes" (2016)
  - Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" (2022)
  """

  @default_num_classes 2

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:task, :classification | :binary | :regression}

  @doc """
  Build a linear probe.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:num_classes` - Number of output classes/dimensions (default: #{@default_num_classes})
    - `:task` - Task type: `:classification`, `:binary`, or `:regression`
      (default: `:classification`)

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, num_classes]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    task = Keyword.get(opts, :task, :classification)

    input = Axon.input("probe_input", shape: {nil, input_size})
    logits = Axon.dense(input, num_classes, name: "probe_linear")

    case task do
      :classification ->
        Axon.activation(logits, :softmax, name: "probe_softmax")

      :binary ->
        Axon.activation(logits, :sigmoid, name: "probe_sigmoid")

      :regression ->
        logits
    end
  end

  @doc "Get the output size of the probe."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end
end
