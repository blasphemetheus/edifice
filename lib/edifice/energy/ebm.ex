defmodule Edifice.Energy.EBM do
  @moduledoc """
  Energy-Based Model (EBM).

  Implements an energy function network that assigns scalar energy values to
  inputs. The model learns an energy landscape where low-energy regions
  correspond to high-probability data configurations. Training uses contrastive
  divergence to push down energy on real data and push up energy on negative
  samples generated via Langevin dynamics MCMC.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-----------------------------+
  | Energy Network (MLP):       |
  |   Dense -> Act -> Dense ... |
  +-----------------------------+
        |
        v
  +-----------------------------+
  | Scalar Output:              |
  |   Dense -> 1                |
  +-----------------------------+
        |
        v
  Energy [batch, 1]
  (lower = more likely)
  ```

  ## Training Loop

  ```
  1. Compute E(x_real) on real data
  2. Sample x_neg via Langevin dynamics from current model
  3. Compute E(x_neg) on negative samples
  4. Loss = E(x_real) - E(x_neg) (+ regularization)
  5. Update parameters to minimize loss
  ```

  ## Langevin Dynamics Sampling

  Starting from noise, iteratively refine samples by following the negative
  gradient of the energy function with added noise:

      x_{t+1} = x_t - step_size * grad_E(x_t) + sqrt(2 * step_size) * noise

  ## Usage

      # Build energy function
      model = EBM.build(input_size: 784, hidden_sizes: [256, 128])

      # Training
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({32, 784}, :f32), %{})

      # Compute energies
      real_energy = predict_fn.(params, %{"input" => real_data})
      neg_samples = EBM.langevin_sample(predict_fn, params, 784, steps: 60)
      neg_energy = predict_fn.(params, %{"input" => neg_samples})

      # Contrastive divergence loss
      loss = EBM.contrastive_divergence_loss(real_energy, neg_energy)

  ## References

  - "A Tutorial on Energy-Based Learning" (LeCun et al., 2006)
  - "Implicit Generation and Modeling with Energy-Based Models" (Du & Mordatch, 2019)
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_activation :silu
  @default_langevin_steps 60
  @default_langevin_step_size 10.0
  @default_langevin_noise_scale 0.005

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an Energy-Based Model.

  Constructs an MLP that maps inputs to scalar energy values. The network
  uses the specified hidden layers with activations, culminating in a single
  linear output neuron (the energy).

  ## Options

  - `:input_size` - Input feature dimension (required)
  - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
  - `:activation` - Activation function (default: :silu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:spectral_norm` - Apply spectral normalization for Lipschitz constraint (default: false)

  ## Returns

  An Axon model. Input: `{batch, input_size}`, Output: `{batch, 1}` (energy).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)

    input = Axon.input("input", shape: {nil, input_size})
    build_energy_fn(input, opts)
  end

  @doc """
  Build the energy function network from an existing Axon input.

  This is the core builder that can be composed into larger architectures.

  ## Parameters

  - `input` - Axon input node
  - `opts` - Options (same as `build/1` minus `:input_size`)

  ## Returns

  An Axon node outputting scalar energy `{batch, 1}`.
  """
  @spec build_energy_fn(Axon.t(), keyword()) :: Axon.t()
  def build_energy_fn(input, opts \\ []) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, 0.0)

    # Hidden layers
    hidden =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        layer =
          acc
          |> Axon.dense(size, name: "energy_dense_#{idx}")
          |> Axon.activation(activation, name: "energy_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "energy_drop_#{idx}")
        else
          layer
        end
      end)

    # Scalar energy output (no activation - energy can be any real value)
    Axon.dense(hidden, 1, name: "energy_output")
  end

  # ============================================================================
  # Training: Contrastive Divergence Loss
  # ============================================================================

  @doc """
  Contrastive divergence loss for training energy-based models.

  The loss pushes down energy on real data and pushes up energy on negative
  (generated) samples:

      L = E(x_real) - E(x_neg) + alpha * (E(x_real)^2 + E(x_neg)^2)

  The regularization term prevents energies from diverging to extreme values.

  ## Parameters

  - `real_energy` - Energy of real data samples `{batch, 1}`
  - `neg_energy` - Energy of negative (generated) samples `{batch, 1}`
  - `opts` - Options

  ## Options

  - `:reg_alpha` - Regularization strength on energy magnitudes (default: 0.01)

  ## Returns

  Scalar loss tensor.
  """
  @spec contrastive_divergence_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def contrastive_divergence_loss(real_energy, neg_energy, opts \\ []) do
    alpha = Keyword.get(opts, :reg_alpha, 0.01)
    cd_loss_impl(real_energy, neg_energy, alpha)
  end

  defnp cd_loss_impl(real_energy, neg_energy, alpha) do
    # Core CD loss: minimize real energy, maximize negative energy
    cd_loss = Nx.mean(real_energy) - Nx.mean(neg_energy)

    # Energy magnitude regularization to prevent divergence
    reg = alpha * (Nx.mean(Nx.pow(real_energy, 2)) + Nx.mean(Nx.pow(neg_energy, 2)))

    cd_loss + reg
  end

  # ============================================================================
  # Sampling: Langevin Dynamics
  # ============================================================================

  @doc """
  Generate samples via Langevin dynamics MCMC.

  Starting from random noise, iteratively refines samples by following the
  negative gradient of the energy function with injected Gaussian noise:

      x_{t+1} = x_t - (step_size / 2) * grad_E(x_t) + sqrt(step_size) * N(0, noise_scale)

  The noise term ensures exploration and, in the limit, samples from the
  Boltzmann distribution p(x) proportional to exp(-E(x)).

  ## Parameters

  - `predict_fn` - Compiled energy function `(params, input) -> energy`
  - `params` - Model parameters
  - `input_size` - Dimension of samples to generate
  - `opts` - Options

  ## Options

  - `:batch_size` - Number of samples to generate (default: 32)
  - `:steps` - Number of Langevin steps (default: 60)
  - `:step_size` - Langevin step size (default: 10.0)
  - `:noise_scale` - Scale of injected noise (default: 0.005)
  - `:init` - Initial samples, or nil for uniform noise (default: nil)
  - `:clamp` - Clamp range {min, max} or nil (default: nil)

  ## Returns

  Generated samples tensor `{batch_size, input_size}`.
  """
  @spec langevin_sample(function(), map(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def langevin_sample(predict_fn, params, input_size, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    steps = Keyword.get(opts, :steps, @default_langevin_steps)
    step_size = Keyword.get(opts, :step_size, @default_langevin_step_size)
    noise_scale = Keyword.get(opts, :noise_scale, @default_langevin_noise_scale)
    init = Keyword.get(opts, :init, nil)
    clamp = Keyword.get(opts, :clamp, nil)

    # Initialize from provided samples or uniform noise
    x =
      if init do
        init
      else
        Nx.Random.uniform(Nx.Random.key(System.os_time()), shape: {batch_size, input_size})
        |> elem(1)
        |> Nx.multiply(2.0)
        |> Nx.subtract(1.0)
      end

    # Run Langevin dynamics steps
    final_x =
      Enum.reduce(1..steps, x, fn _step, x_curr ->
        langevin_step(predict_fn, params, x_curr, step_size, noise_scale, clamp)
      end)

    # Detach from computation graph (stop gradients)
    Nx.backend_copy(final_x)
  end

  # Single Langevin dynamics step
  defp langevin_step(predict_fn, params, x, step_size, noise_scale, clamp) do
    # Compute gradient of energy with respect to input
    grad =
      Nx.Defn.grad(x, fn x_var ->
        input_map = %{"input" => x_var}
        energy = predict_fn.(params, input_map)
        Nx.sum(energy)
      end)

    # Langevin update: x' = x - (step/2) * grad_E(x) + sqrt(step) * noise
    noise_key = Nx.Random.key(System.os_time(:nanosecond))
    {noise, _key} = Nx.Random.normal(noise_key, shape: Nx.shape(x))

    x_new =
      x
      |> Nx.subtract(Nx.multiply(step_size / 2.0, grad))
      |> Nx.add(Nx.multiply(:math.sqrt(step_size) * noise_scale, noise))

    # Optional clamping
    case clamp do
      {min_val, max_val} -> Nx.clip(x_new, min_val, max_val)
      nil -> x_new
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Compute the energy of a batch of inputs using a compiled model.

  Convenience wrapper that handles the input map construction.

  ## Parameters

  - `predict_fn` - Compiled energy function
  - `params` - Model parameters
  - `input` - Input tensor `{batch, input_size}`

  ## Returns

  Energy tensor `{batch, 1}`.
  """
  @spec energy(function(), map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def energy(predict_fn, params, input) do
    predict_fn.(params, %{"input" => input})
  end
end
