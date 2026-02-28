defmodule Edifice.Interpretability.JumpReluSAE do
  @moduledoc """
  JumpReLU Sparse Autoencoder for mechanistic interpretability.

  Replaces the standard ReLU + TopK sparsification with a JumpReLU activation
  that has a learnable per-feature threshold. Features only activate when they
  exceed their learned threshold, allowing the model to adaptively control
  sparsity without a rigid top-k constraint.

  ## Architecture

  ```
  Input [batch, input_size]
        |
  Encoder: dense(dict_size)
        |
  JumpReLU: x * σ(β(x - θ))   (θ learnable per-feature, β = temperature)
        |
  [batch, dict_size]  (sparse activations)
        |
  Decoder: dense(input_size)
        |
  Output [batch, input_size]  (reconstruction)
  ```

  ## JumpReLU Activation

  `JumpReLU(x) = x * H(x - θ)` where H is the Heaviside step function and θ
  is a learned threshold per dictionary feature. Approximated with a smooth
  sigmoid gate `σ(β(x - θ))` for differentiability, where β controls sharpness.

  - Common features learn lower thresholds (activate easily)
  - Rare features learn higher thresholds (activate only when strongly present)
  - At initialization (θ=0), this reduces to a soft ReLU

  ## Usage

      model = JumpReluSAE.build(
        input_size: 256,
        dict_size: 4096,
        temperature: 10.0
      )

  ## References

  - Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity
    with JumpReLU Sparse Autoencoders" (DeepMind, 2024)
  - Used in Gemma Scope (DeepMind, 2024) for production-scale SAE training
  """

  import Nx.Defn

  alias Edifice.Interpretability.SparseAutoencoder

  @default_dict_size 4096
  @default_temperature 10.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:temperature, float()}

  @doc """
  Build a JumpReLU sparse autoencoder.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:dict_size` - Number of dictionary features (default: #{@default_dict_size})
    - `:temperature` - Sharpness of the soft threshold gate; higher values
      approximate a hard step function more closely (default: #{@default_temperature})

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    temperature = Keyword.get(opts, :temperature, @default_temperature)

    input = Axon.input("jump_relu_sae_input", shape: {nil, input_size})

    # Encoder
    hidden = Axon.dense(input, dict_size, name: "jump_relu_sae_encoder")

    # JumpReLU with learnable threshold
    hidden = jump_relu_layer(hidden, dict_size, temperature)

    # Decoder
    Axon.dense(hidden, input_size, name: "jump_relu_sae_decoder")
  end

  @doc """
  Build the encoder portion only (for extracting sparse activations).

  Returns sparse hidden activations `[batch, dict_size]`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    temperature = Keyword.get(opts, :temperature, @default_temperature)

    input = Axon.input("jump_relu_sae_input", shape: {nil, input_size})
    hidden = Axon.dense(input, dict_size, name: "jump_relu_sae_encoder")
    jump_relu_layer(hidden, dict_size, temperature)
  end

  defp jump_relu_layer(hidden, dict_size, temperature) do
    threshold = Axon.param("threshold", {dict_size}, initializer: :zeros)

    Axon.layer(
      fn activations, thresh, _opts ->
        jump_relu(activations, thresh, temperature)
      end,
      [hidden, threshold],
      name: "jump_relu_sae_threshold",
      op_name: :jump_relu
    )
  end

  @doc """
  Compute JumpReLU SAE training loss: reconstruction MSE + L1 sparsity penalty.

  Same as standard SAE loss — the sparsity is controlled by the learned
  thresholds rather than by a separate penalty term, but L1 regularization
  on the hidden activations still helps during training.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(input, reconstruction, hidden_acts, opts \\ []) do
    SparseAutoencoder.loss(input, reconstruction, hidden_acts, opts)
  end

  @doc """
  JumpReLU activation: `x * σ(β(x - θ))`.

  Soft approximation of `x * H(x - θ)` where H is the Heaviside step function.
  The sigmoid gate `σ(β(x - θ))` is differentiable, allowing gradients to flow
  through both the activations and the learned threshold.
  """
  @spec jump_relu(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn jump_relu(activations, threshold, temperature) do
    gate = Nx.sigmoid(temperature * (activations - threshold))
    activations * gate
  end

  @doc "Get the output size of the JumpReLU SAE (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
