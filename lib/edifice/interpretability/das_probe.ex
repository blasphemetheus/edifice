defmodule Edifice.Interpretability.DASProbe do
  @moduledoc """
  Distributed Alignment Search (DAS) Probe for causal concept detection.

  Extends the linear probe by first learning a linear subspace projection that
  aligns with the concept's distributed representation. Stronger than standard
  linear probes because concepts may be distributed across multiple dimensions
  rather than encoded in a single direction.

  ## Architecture

  ```
  Input [batch, input_size]  (frozen activations)
        |
  Subspace projection: dense(subspace_dim, no bias)
        |
  [batch, subspace_dim]  (concept-aligned subspace)
        |
  Probe: dense(num_classes)
        |
  Output [batch, num_classes]
  ```

  ## Comparison to Linear Probe

  A linear probe uses a single linear layer from full activation space. A DAS
  probe first projects into a learned low-dimensional subspace, then probes
  within that subspace. This finds distributed representations that span
  multiple dimensions — a concept encoded across 3 dimensions of a 768-dim
  space would be hard for a linear probe but easy for a DAS probe with
  subspace_dim >= 3.

  ## Usage

      # Does layer 6 causally encode part-of-speech in a 4-dim subspace?
      probe = DASProbe.build(
        input_size: 768,
        subspace_dim: 4,
        num_classes: 17
      )

  ## References

  - Geiger et al., "Finding Alignments Between Interpretable Causal Variables
    and Distributed Neural Representations" (ICLR 2024)
  """

  @default_subspace_dim 16
  @default_num_classes 2

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:subspace_dim, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:task, :classification | :binary | :regression}

  @doc """
  Build a DAS probe.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:subspace_dim` - Dimension of the learned concept subspace
      (default: #{@default_subspace_dim})
    - `:num_classes` - Number of output classes/dimensions (default: #{@default_num_classes})
    - `:task` - Task type: `:classification`, `:binary`, or `:regression`
      (default: `:classification`)

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, num_classes]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    subspace_dim = Keyword.get(opts, :subspace_dim, @default_subspace_dim)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    task = Keyword.get(opts, :task, :classification)

    input = Axon.input("das_probe_input", shape: {nil, input_size})

    # Learned subspace projection (no bias — pure linear subspace)
    projected = Axon.dense(input, subspace_dim, name: "das_projection", use_bias: false)

    # Probe within the aligned subspace
    logits = Axon.dense(projected, num_classes, name: "das_probe_linear")

    case task do
      :classification ->
        Axon.activation(logits, :softmax, name: "das_output")

      :binary ->
        Axon.activation(logits, :sigmoid, name: "das_output")

      :regression ->
        logits
    end
  end

  @doc """
  Build just the subspace projection (for inspecting the learned alignment).

  Returns projected activations `[batch, subspace_dim]`.
  """
  @spec build_projection([build_opt()]) :: Axon.t()
  def build_projection(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    subspace_dim = Keyword.get(opts, :subspace_dim, @default_subspace_dim)

    input = Axon.input("das_probe_input", shape: {nil, input_size})
    Axon.dense(input, subspace_dim, name: "das_projection", use_bias: false)
  end

  @doc "Get the output size of the DAS probe."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end
end
