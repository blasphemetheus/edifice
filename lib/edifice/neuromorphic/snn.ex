defmodule Edifice.Neuromorphic.SNN do
  @moduledoc """
  Spiking Neural Network with surrogate gradients.

  Spiking Neural Networks (SNNs) process information using discrete spikes
  rather than continuous activations. Neurons integrate input over time and
  fire when their membrane potential exceeds a threshold, then reset. This
  is biologically plausible and extremely energy-efficient on neuromorphic
  hardware (Intel Loihi, IBM TrueNorth).

  ## Leaky Integrate-and-Fire (LIF) Neuron

  The core compute unit:

      V[t] = beta * V[t-1] + W * x[t]    (leak + integrate)
      spike[t] = V[t] > threshold          (fire)
      V[t] = V[t] - spike[t] * threshold   (soft reset after spike)

  where:
  - beta = exp(-dt/tau) is the membrane decay factor
  - tau is the membrane time constant
  - threshold is the firing threshold

  ## Surrogate Gradients

  The spike function (Heaviside step) is non-differentiable. We use a
  surrogate gradient for backpropagation: the derivative of a smooth
  approximation (sigmoid or fast sigmoid) replaces the true derivative.

  ## Architecture

  ```
  Input [batch, input_size]
        |  (presented for num_timesteps)
        v
  +----------------------------+
  |   LIF Layer 1              |
  |   V = beta*V + W*x        |
  |   spike if V > threshold   |
  +----------------------------+
        |  (spike train)
        v
  +----------------------------+
  |   LIF Layer 2              |
  +----------------------------+
        |
        v
  +----------------------------+
  |   Rate Decoding            |
  |   output = mean(spikes)    |
  +----------------------------+
        |
        v
  Output [batch, output_size]
  ```

  ## Usage

      model = SNN.build(
        input_size: 256,
        hidden_sizes: [128, 64],
        output_size: 10,
        num_timesteps: 25,
        tau: 2.0,
        threshold: 1.0
      )

  ## References
  - Neftci et al., "Surrogate Gradient Learning in SNNs" (2019)
  - https://arxiv.org/abs/1901.09948
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_num_timesteps 25
  @default_tau 2.0
  @default_threshold 1.0
  @default_surrogate_slope 10.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Spiking Neural Network with LIF neurons and surrogate gradients.

  The network processes input through multiple LIF neuron layers over
  several timesteps, then rate-decodes the output spike train into
  a continuous output.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
    - `:output_size` - Output dimension (required)
    - `:num_timesteps` - Number of simulation timesteps (default: 25)
    - `:tau` - Membrane time constant (default: 2.0)
    - `:threshold` - Firing threshold (default: 1.0)

  ## Returns
    An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    output_size = Keyword.fetch!(opts, :output_size)
    num_timesteps = Keyword.get(opts, :num_timesteps, @default_num_timesteps)
    tau = Keyword.get(opts, :tau, @default_tau)
    threshold = Keyword.get(opts, :threshold, @default_threshold)

    input = Axon.input("input", shape: {nil, input_size})

    # Multi-layer SNN: each layer does dense projection then LIF simulation
    # The dense layer provides learnable weights W_l, and the LIF layer
    # simulates spiking dynamics with rate-coded output between layers
    hidden =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        # Dense projection (learnable synaptic weights)
        projected = Axon.dense(acc, size, name: "snn_layer_#{idx}_dense")

        # LIF neuron layer: simulate over timesteps, rate-decode output
        Axon.layer(
          &lif_layer_impl/2,
          [projected],
          name: "snn_layer_#{idx}_lif",
          num_timesteps: num_timesteps,
          tau: tau,
          threshold: threshold,
          op_name: :lif_layer
        )
      end)

    # Final readout projection (trainable)
    Axon.dense(hidden, output_size, name: "snn_output")
  end

  @doc """
  Leaky Integrate-and-Fire neuron step.

  Computes one timestep of LIF dynamics:

      V[t] = beta * V[t-1] + I[t]
      spike[t] = surrogate_gradient(V[t] - threshold)
      V[t] = V[t] * (1 - spike[t])   (reset)

  ## Parameters
    - `membrane` - Membrane potential from previous step `[batch, hidden_size]`
    - `input_current` - Weighted input current `[batch, hidden_size]`
    - `beta` - Membrane decay factor (= exp(-1/tau))
    - `threshold` - Firing threshold

  ## Returns
    Tuple `{new_membrane, spikes}`:
    - `new_membrane` - Updated membrane potential `[batch, hidden_size]`
    - `spikes` - Spike output `[batch, hidden_size]` (0 or ~1)
  """
  @spec lif_neuron(Nx.Tensor.t(), Nx.Tensor.t(), float(), float()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  defn lif_neuron(membrane, input_current, beta, threshold) do
    # Leaky integration
    new_membrane = beta * membrane + input_current

    # Spike generation with surrogate gradient
    spikes = surrogate_gradient(new_membrane - threshold, @default_surrogate_slope)

    # Reset membrane after spike (soft reset: subtract threshold)
    reset_membrane = new_membrane - spikes * threshold

    {reset_membrane, spikes}
  end

  @doc """
  Surrogate gradient for the non-differentiable spike function.

  The Heaviside step function has zero gradient almost everywhere.
  We use a fast sigmoid as a surrogate: during the forward pass we
  still get hard spikes, but gradients flow through the sigmoid
  approximation during backpropagation.

      forward: spike = (x > 0) ? 1 : 0
      backward: d_spike/dx = slope / (1 + slope * |x|)^2

  In practice with Nx, we use the sigmoid directly as a smooth
  approximation that is differentiable everywhere.

  ## Parameters
    - `x` - Input tensor (membrane - threshold)
    - `slope` - Steepness of the surrogate (default: 25.0)

  ## Returns
    Approximate spike values in [0, 1]
  """
  @spec surrogate_gradient(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn surrogate_gradient(x, slope) do
    # Standard sigmoid surrogate with slope scaling
    # Clamp to [-8, 8] to prevent exp() overflow → NaN gradients
    Nx.sigmoid(Nx.clip(slope * x, -8.0, 8.0))
  end

  @doc """
  Rate decoding: convert spike trains to firing rates.

  Computes the mean spike count over timesteps for each neuron.
  This is the simplest decoding scheme and works well for classification.

  ## Parameters
    - `spike_train` - Spike tensor `[batch, num_timesteps, hidden_size]`

  ## Returns
    Firing rates `[batch, hidden_size]` in [0, 1]
  """
  @spec rate_decode(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn rate_decode(spike_train) do
    Nx.mean(spike_train, axes: [1])
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Single LIF layer: simulate over timesteps with constant input current
  # Returns rate-decoded output (mean spike count over timesteps)
  defp lif_layer_impl(input_current, opts) do
    num_timesteps = opts[:num_timesteps]
    tau = opts[:tau]
    threshold = opts[:threshold]

    # Membrane decay factor: beta = exp(-1/tau)
    beta = :math.exp(-1.0 / tau)

    batch_size = Nx.axis_size(input_current, 0)
    dim = Nx.axis_size(input_current, 1)

    initial_membrane = Nx.broadcast(0.0, {batch_size, dim})
    initial_spikes = Nx.broadcast(0.0, {batch_size, dim})

    {_final_membrane, spike_sum} =
      Enum.reduce(1..num_timesteps, {initial_membrane, initial_spikes}, fn _t,
                                                                           {membrane, acc_spikes} ->
        # Leaky integration: V[t] = beta * V[t-1] + I[t]
        new_membrane = Nx.add(Nx.multiply(beta, membrane), input_current)

        # Spike generation via surrogate gradient (sigmoid approximation)
        # Clamp pre-sigmoid input to [-8, 8] to prevent exp() overflow → NaN gradients
        scaled =
          Nx.clip(
            Nx.multiply(@default_surrogate_slope, Nx.subtract(new_membrane, threshold)),
            -8.0,
            8.0
          )

        spikes = Nx.sigmoid(scaled)

        # Soft reset: V[t] = V[t] - spike[t] * threshold
        reset_membrane = Nx.subtract(new_membrane, Nx.multiply(spikes, threshold))

        {reset_membrane, Nx.add(acc_spikes, spikes)}
      end)

    # Rate decode: average spikes over timesteps
    Nx.divide(spike_sum, num_timesteps)
  end
end
