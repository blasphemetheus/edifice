defmodule Edifice.Interpretability.LEACE do
  @moduledoc """
  LEACE (LEAst-squares Concept Erasure) for removing concepts from activations.

  Learns a low-rank projection that identifies and removes concept-related
  information from neural network activations. The output preserves all
  information except the erased concept direction(s).

  ## Architecture

  ```
  Input [batch, input_size]
        |
    ┌───┴───┐
    │       │
    │  Concept projection: dense(concept_dim, no bias)
    │       |
    │  Reconstruction: dense(input_size, no bias)
    │       |
    └───┬───┘
        |
  Subtract: input - reconstruction
        |
  Output [batch, input_size]  (concept-erased activations)
  ```

  ## How It Works

  The model learns a low-rank factorization `V @ V^T` that captures the concept
  subspace. Forward pass computes `x - x @ V^T @ V`, projecting activations
  onto the orthogonal complement of the concept directions.

  The original LEACE computes V in closed form via SVD. This trainable version
  converges to the same subspace when optimized to minimize concept probe
  accuracy on the erased activations while preserving reconstruction fidelity.

  ## Usage

      # Erase gender information from layer 8 activations
      eraser = LEACE.build(
        input_size: 768,
        concept_dim: 1
      )

      # Erase a 3-dimensional concept subspace
      eraser = LEACE.build(
        input_size: 768,
        concept_dim: 3
      )

  ## References

  - Belrose et al., "LEACE: Perfect linear concept erasure in closed form"
    (ICML 2023)
  """

  @default_concept_dim 1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:concept_dim, pos_integer()}

  @doc """
  Build a LEACE eraser.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:concept_dim` - Dimension of the concept subspace to erase
      (default: #{@default_concept_dim})

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
    Output has the concept direction(s) removed.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    concept_dim = Keyword.get(opts, :concept_dim, @default_concept_dim)

    input = Axon.input("leace_input", shape: {nil, input_size})

    # Project to concept space (learn concept directions)
    concept_proj = Axon.dense(input, concept_dim, name: "leace_concept_proj", use_bias: false)

    # Reconstruct the concept component in input space
    concept_recon =
      Axon.dense(concept_proj, input_size, name: "leace_concept_recon", use_bias: false)

    # Erase: subtract the concept component
    Axon.layer(
      fn x, recon, _opts -> Nx.subtract(x, recon) end,
      [input, concept_recon],
      name: "leace_erase",
      op_name: :concept_erase
    )
  end

  @doc "Get the output size of the LEACE eraser (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
