defmodule Edifice.Probabilistic.EvidentialNN do
  @moduledoc """
  Evidential Deep Learning with Dirichlet Priors.

  Evidential Neural Networks place a Dirichlet distribution over class
  probabilities, enabling principled uncertainty estimation in a single
  forward pass (no ensembles or MC sampling needed). The network outputs
  evidence parameters (alpha) that parameterize a Dirichlet distribution,
  from which both aleatoric and epistemic uncertainty can be derived.

  ## How It Works

  Instead of outputting softmax probabilities, the network outputs
  evidence parameters alpha_k >= 0 for each class:

      p(y|x) = Dir(p | alpha)
      alpha_k = evidence_k + 1

  The Dirichlet concentration parameters encode:
  - **Belief mass**: b_k = (alpha_k - 1) / S where S = sum(alpha)
  - **Uncertainty mass**: u = K / S (K = num_classes)
  - **Expected probability**: p_k = alpha_k / S

  ## Uncertainty Types

  | Type | Formula | Meaning |
  |------|---------|---------|
  | Epistemic | u = K / S | Lack of evidence (data uncertainty) |
  | Aleatoric | E[H[Cat(p)]] | Inherent class overlap |
  | Total | H[E[Cat(p)]] | Combined uncertainty |

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +--------------------------------------+
  | Backbone MLP                         |
  |   Dense -> Act -> Dense -> Act       |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Evidence Head:                       |
  |   Dense -> Softplus (ensures > 0)    |
  |   alpha = evidence + 1              |
  +--------------------------------------+
        |
        v
  Dirichlet Parameters [batch, num_classes]
  ```

  ## Usage

      model = EvidentialNN.build(
        input_size: 256,
        hidden_sizes: [128, 64],
        num_classes: 10
      )

      # Get predictions + uncertainty
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, state)
      alpha = predict_fn.(params, input)
      {epistemic, aleatoric} = EvidentialNN.uncertainty(alpha)

  ## References

  - Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty" (NeurIPS 2018)
  - https://arxiv.org/abs/1806.01768
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_activation :relu

  @doc """
  Build an Evidential Neural Network.

  ## Options

  - `:input_size` - Input feature dimension (required)
  - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
  - `:num_classes` - Number of output classes (required)
  - `:activation` - Activation function for hidden layers (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

  An Axon model outputting Dirichlet alpha parameters:
  `[batch, input_size]` -> `[batch, num_classes]`

  The output alpha_k values are always > 1 (evidence + 1).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    num_classes = Keyword.fetch!(opts, :num_classes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, 0.0)

    input = Axon.input("input", shape: {nil, input_size})

    # Backbone MLP
    backbone =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        layer =
          acc
          |> Axon.dense(size, name: "backbone_dense_#{idx}")
          |> Axon.layer_norm(name: "backbone_ln_#{idx}")
          |> Axon.activation(activation, name: "backbone_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "backbone_drop_#{idx}")
        else
          layer
        end
      end)

    # Evidence head: softplus ensures evidence >= 0
    evidence = Axon.dense(backbone, num_classes, name: "evidence_head")

    # alpha = evidence + 1 (Dirichlet concentration parameters)
    Axon.nx(
      evidence,
      fn e ->
        # Softplus for non-negative evidence, then add 1 for valid Dirichlet
        pos_evidence = Nx.log1p(Nx.exp(e))
        Nx.add(pos_evidence, 1.0)
      end,
      name: "alpha"
    )
  end

  @doc """
  Compute epistemic and aleatoric uncertainty from Dirichlet parameters.

  ## Parameters

  - `alpha` - Dirichlet concentration parameters `[batch, num_classes]`

  ## Returns

  Tuple `{epistemic, aleatoric}`:
  - `epistemic` - Uncertainty due to lack of evidence `[batch]`
  - `aleatoric` - Uncertainty due to inherent data noise `[batch]`
  """
  @spec uncertainty(Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def uncertainty(alpha) do
    {epistemic_uncertainty(alpha), aleatoric_uncertainty(alpha)}
  end

  @doc """
  Compute epistemic (knowledge) uncertainty.

  Epistemic uncertainty = K / S where K is the number of classes and
  S = sum(alpha) is the Dirichlet strength. High uncertainty when
  evidence is low (alpha values close to 1).

  ## Parameters

  - `alpha` - Dirichlet parameters `[batch, num_classes]`

  ## Returns

  Epistemic uncertainty `[batch]` in [0, 1].
  """
  @spec epistemic_uncertainty(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn epistemic_uncertainty(alpha) do
    num_classes = Nx.axis_size(alpha, 1)
    dirichlet_strength = Nx.sum(alpha, axes: [1])
    num_classes / dirichlet_strength
  end

  @doc """
  Compute aleatoric (data) uncertainty.

  Aleatoric uncertainty is the expected entropy of the categorical
  distribution under the Dirichlet: E_Dir[H[Cat(p)]].

  ## Parameters

  - `alpha` - Dirichlet parameters `[batch, num_classes]`

  ## Returns

  Aleatoric uncertainty `[batch]`.
  """
  @spec aleatoric_uncertainty(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn aleatoric_uncertainty(alpha) do
    dirichlet_strength = Nx.sum(alpha, axes: [1], keep_axes: true)
    probs = alpha / dirichlet_strength

    # Expected entropy: -sum(p_k * (digamma(alpha_k + 1) - digamma(S + 1)))
    # Approximation: use -sum(p_k * log(p_k)) as the expected entropy
    log_probs = Nx.log(Nx.max(probs, 1.0e-10))
    entropy = -Nx.sum(probs * log_probs, axes: [1])

    entropy
  end

  @doc """
  Compute the expected class probabilities from Dirichlet parameters.

  ## Parameters

  - `alpha` - Dirichlet parameters `[batch, num_classes]`

  ## Returns

  Expected probabilities `[batch, num_classes]` that sum to 1.
  """
  @spec expected_probability(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn expected_probability(alpha) do
    dirichlet_strength = Nx.sum(alpha, axes: [1], keep_axes: true)
    alpha / dirichlet_strength
  end

  @doc """
  Evidential Deep Learning loss (Type II Maximum Likelihood).

  Combines the negative log-likelihood of the Dirichlet-Categorical model
  with a KL divergence regularizer that penalizes evidence on incorrect
  classes.

  ## Parameters

  - `alpha` - Predicted Dirichlet parameters `[batch, num_classes]`
  - `targets` - One-hot encoded targets `[batch, num_classes]`

  ## Options

  - `:kl_weight` - Weight for KL regularization term (default: 0.01)

  ## Returns

  Scalar loss tensor.
  """
  @spec evidential_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn evidential_loss(alpha, targets, opts \\ []) do
    kl_weight =
      case opts[:kl_weight] do
        nil -> 0.01
        val -> val
      end

    dirichlet_strength = Nx.sum(alpha, axes: [1], keep_axes: true)

    # Type II Maximum Likelihood loss
    # L = sum(y_k * (log(S) - log(alpha_k)))
    nll =
      Nx.sum(
        targets * (Nx.log(dirichlet_strength) - Nx.log(alpha)),
        axes: [1]
      )

    # KL divergence regularizer: penalize evidence on wrong classes
    # Remove evidence for correct class
    alpha_tilde = targets + (1.0 - targets) * alpha

    # KL(Dir(alpha_tilde) || Dir(1, ..., 1))
    s_tilde = Nx.sum(alpha_tilde, axes: [1], keep_axes: true)
    num_classes = Nx.axis_size(alpha, 1)

    kl =
      Nx.sum(
        (alpha_tilde - 1.0) *
          (Nx.log(alpha_tilde) - Nx.log(s_tilde / num_classes)),
        axes: [1]
      )

    Nx.mean(nll + kl_weight * kl)
  end

  @doc """
  Get the output size of an Evidential NN.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :num_classes)
  end
end
