defmodule Edifice.Probabilistic.MCDropout do
  @moduledoc """
  MC Dropout for uncertainty estimation (Gal & Ghahramani, 2016).

  Standard dropout is disabled at inference time. MC Dropout keeps dropout
  active during inference and runs multiple forward passes to estimate
  prediction uncertainty. This provides a practical Bayesian approximation
  without modifying the training procedure.

  ## How It Works

  1. Train a standard network with dropout (nothing special)
  2. At inference time, keep dropout ON
  3. Run N forward passes with different dropout masks
  4. Mean of outputs = prediction, Variance = uncertainty

  ## Interpretation

  - **Low variance**: Model is confident (consistent predictions across dropout masks)
  - **High variance**: Model is uncertain (different subnetworks disagree)
  - **Out-of-distribution**: Typically shows high variance

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-------------------------------+
  | Dense + ReLU + Dropout (ON)  |  Layer 1
  +-------------------------------+
        |
        v
  +-------------------------------+
  | Dense + ReLU + Dropout (ON)  |  Layer 2
  +-------------------------------+
        |
        v
  Output [batch, output_size]

  (Run N times, compute mean + variance)
  ```

  ## Usage

      # Build model with always-on dropout
      model = MCDropout.build(
        input_size: 256,
        hidden_sizes: [128, 64],
        output_size: 10,
        dropout_rate: 0.2
      )

      # Get predictions with uncertainty
      {mean, variance} = MCDropout.predict_with_uncertainty(
        model, params, input, num_samples: 30
      )

  ## References
  - Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
  - https://arxiv.org/abs/1506.02142
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_dropout_rate 0.2
  @default_num_samples 20

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an MLP with dropout that stays active at inference time.

  Uses Axon's training mode to keep dropout active. The key difference
  from a standard MLP is that dropout is applied at every layer and
  is intended to remain active during inference for uncertainty estimation.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
    - `:output_size` - Output dimension (required)
    - `:dropout_rate` - Dropout probability (default: 0.2)
    - `:activation` - Activation function (default: :relu)

  ## Returns
    An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    output_size = Keyword.fetch!(opts, :output_size)
    dropout_rate = Keyword.get(opts, :dropout_rate, @default_dropout_rate)
    activation = Keyword.get(opts, :activation, :relu)

    input = Axon.input("input", shape: {nil, input_size})

    # Build hidden layers with always-on dropout
    x =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        build_mc_layer(acc, size,
          dropout_rate: dropout_rate,
          activation: activation,
          name: "mc_layer_#{idx}"
        )
      end)

    # Output layer (no dropout on the final output)
    Axon.dense(x, output_size, name: "mc_output")
  end

  @doc """
  Build a Dense + always-on Dropout layer.

  This is the building block for MC Dropout networks. Dropout is applied
  after activation and is kept active even in inference mode by running
  the model in training mode during prediction.

  ## Parameters
    - `input` - Axon node
    - `units` - Number of output units

  ## Options
    - `:dropout_rate` - Dropout probability (default: 0.2)
    - `:activation` - Activation function (default: :relu)
    - `:name` - Layer name prefix (default: "mc_layer")

  ## Returns
    An Axon node with shape `[batch, units]`
  """
  @spec build_mc_layer(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def build_mc_layer(input, units, opts \\ []) do
    dropout_rate = Keyword.get(opts, :dropout_rate, @default_dropout_rate)
    activation = Keyword.get(opts, :activation, :relu)
    name = Keyword.get(opts, :name, "mc_layer")

    input
    |> Axon.dense(units, name: "#{name}_dense")
    |> Axon.activation(activation, name: "#{name}_act")
    |> Axon.dropout(rate: dropout_rate, name: "#{name}_dropout")
  end

  # ============================================================================
  # Uncertainty Estimation
  # ============================================================================

  @doc """
  Run N forward passes with dropout and return mean prediction + variance.

  This is the core MC Dropout inference procedure. By running the model
  multiple times in training mode (dropout active), each pass uses a
  different random dropout mask, producing different outputs. The variance
  across these outputs quantifies the model's uncertainty.

  ## Parameters
    - `model` - Axon model built with `build/1`
    - `params` - Trained model parameters
    - `input` - Input tensor `[batch, input_size]`

  ## Options
    - `:num_samples` - Number of forward passes (default: 20)

  ## Returns
    Tuple of `{mean, variance}` where:
    - `mean` - Average prediction `[batch, output_size]`
    - `variance` - Prediction variance `[batch, output_size]`
  """
  @spec predict_with_uncertainty(Axon.t(), map(), Nx.Tensor.t(), keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def predict_with_uncertainty(model, params, input, opts \\ []) do
    num_samples = Keyword.get(opts, :num_samples, @default_num_samples)

    # Run N forward passes in training mode (dropout active)
    # Collect predictions from each stochastic forward pass
    predictions =
      Enum.map(1..num_samples, fn _i ->
        # Run model in training mode to keep dropout active
        Axon.predict(model, params, %{"input" => input}, mode: :train)
      end)

    # Stack into [num_samples, batch, output_size]
    stacked = Nx.stack(predictions, axis: 0)

    # Compute mean and variance across samples (axis 0)
    mean = Nx.mean(stacked, axes: [0])
    variance = Nx.variance(stacked, axes: [0])

    {mean, variance}
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  @doc """
  Compute predictive entropy from MC Dropout samples.

  For classification tasks, entropy provides an alternative uncertainty
  measure that captures total uncertainty (both aleatoric and epistemic).

  H[y|x] = -sum(p * log(p)) where p = mean softmax probabilities

  ## Parameters
    - `mean_probs` - Mean predicted probabilities `[batch, num_classes]`

  ## Returns
    Entropy values `[batch]`
  """
  @spec predictive_entropy(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn predictive_entropy(mean_probs) do
    # Clamp to avoid log(0)
    probs = Nx.max(mean_probs, 1.0e-10)
    -Nx.sum(probs * Nx.log(probs), axes: [1])
  end
end
