defmodule Edifice.Recurrent.Reservoir do
  @moduledoc """
  Echo State Networks / Reservoir Computing.

  Reservoir computing uses a fixed, randomly initialized recurrent network
  (the "reservoir") and only trains the output (readout) layer. This makes
  training extremely fast since only a linear layer is optimized.

  ## Architecture

  ```
  Input x[t]
       |
       v
  +------------------+
  | Fixed Reservoir  |  h[t] = tanh(W_in * x[t] + W_res * h[t-1])
  | (random weights) |  (NOT trained)
  +------------------+
       |
       v
  +------------------+
  | Readout Layer    |  y[t] = W_out * h[t]
  | (trained)        |  (ridge regression or gradient descent)
  +------------------+
       |
       v
  Output y[t]
  ```

  ## Key Properties

  - **Echo State Property**: reservoir state asymptotically depends only on input,
    not initial conditions. Achieved when spectral radius of W_res < 1.
  - **Separation Property**: different input sequences produce different reservoir states.
  - **Training**: Only W_out is trained (via linear regression or gradient descent).

  ## Usage

      model = Reservoir.build(
        input_size: 64,
        reservoir_size: 500,
        output_size: 10,
        spectral_radius: 0.9,
        sparsity: 0.1
      )
  """

  require Axon

  @doc """
  Build an Echo State Network.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:reservoir_size` - Number of reservoir neurons (default: 500)
    - `:output_size` - Output dimension (default: reservoir_size)
    - `:spectral_radius` - Spectral radius of reservoir matrix (default: 0.9)
    - `:sparsity` - Fraction of zero connections in reservoir (default: 0.9)
    - `:input_scaling` - Scale of input weights (default: 1.0)
    - `:leak_rate` - Leaky integration rate (default: 1.0, no leaking)
    - `:seq_len` - Sequence length (default: nil for dynamic)

  ## Returns
    An Axon model that processes sequences through a fixed reservoir
    and trainable readout layer.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    reservoir_size = Keyword.get(opts, :reservoir_size, 500)
    output_size = Keyword.get(opts, :output_size, reservoir_size)
    seq_len = Keyword.get(opts, :seq_len, nil)

    # Input: [batch, seq_len, input_size]
    input = Axon.input("input", shape: {nil, seq_len, input_size})

    # Reservoir layer (fixed random weights, not trained)
    reservoir_output =
      Axon.layer(
        &reservoir_forward/4,
        [input],
        name: "reservoir",
        reservoir_size: reservoir_size,
        input_size: input_size,
        spectral_radius: Keyword.get(opts, :spectral_radius, 0.9),
        sparsity: Keyword.get(opts, :sparsity, 0.9),
        input_scaling: Keyword.get(opts, :input_scaling, 1.0),
        leak_rate: Keyword.get(opts, :leak_rate, 1.0),
        op_name: :reservoir
      )

    # Trainable readout (only this gets optimized)
    Axon.dense(reservoir_output, output_size, name: "readout")
  end

  # Reservoir forward pass using fixed random weights
  # In practice, the reservoir weights should be initialized once and frozen.
  # Here we use a deterministic seed-based approach for reproducibility.
  defp reservoir_forward(input, _opts_or_params, _state, opts) do
    reservoir_size = opts[:reservoir_size]
    input_size = opts[:input_size]
    spectral_radius = opts[:spectral_radius]
    input_scaling = opts[:input_scaling]
    leak_rate = opts[:leak_rate]
    sparsity = opts[:sparsity]

    batch_size = Nx.axis_size(input, 0)
    seq_len = Nx.axis_size(input, 1)

    # Generate deterministic reservoir weights using a fixed key
    key = Nx.Random.key(42)

    # Input weights: [input_size, reservoir_size]
    {w_in, key} = Nx.Random.normal(key, shape: {input_size, reservoir_size})
    w_in = Nx.multiply(w_in, input_scaling)

    # Reservoir weights: [reservoir_size, reservoir_size]
    {w_res, key} = Nx.Random.normal(key, shape: {reservoir_size, reservoir_size})

    # Apply sparsity mask
    {mask_vals, _key} = Nx.Random.uniform(key, shape: {reservoir_size, reservoir_size})
    mask = Nx.greater(mask_vals, sparsity)
    w_res = Nx.multiply(w_res, mask)

    # Scale to target spectral radius
    # Approximate spectral radius scaling (exact eigenvalue computation is expensive)
    frobenius_norm = Nx.sqrt(Nx.sum(Nx.pow(w_res, 2)))
    estimated_spectral = Nx.divide(frobenius_norm, Nx.sqrt(reservoir_size * (1.0 - sparsity)))
    scale = Nx.divide(spectral_radius, Nx.add(estimated_spectral, 1.0e-8))
    w_res = Nx.multiply(w_res, scale)

    # Run reservoir dynamics
    h = Nx.broadcast(0.0, {batch_size, reservoir_size})

    # Process each timestep
    final_h =
      Enum.reduce(0..(seq_len - 1), h, fn t, h_prev ->
        x_t = Nx.slice_along_axis(input, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # h_new = tanh(W_in * x + W_res * h_prev)
        pre_activation = Nx.add(Nx.dot(x_t, w_in), Nx.dot(h_prev, w_res))
        h_new = Nx.tanh(pre_activation)

        # Leaky integration: h = (1 - alpha) * h_prev + alpha * h_new
        if leak_rate == 1.0 do
          h_new
        else
          Nx.add(
            Nx.multiply(1.0 - leak_rate, h_prev),
            Nx.multiply(leak_rate, h_new)
          )
        end
      end)

    # Return final hidden state: [batch, reservoir_size]
    final_h
  end

  @doc """
  Get the output size of the reservoir.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :output_size, Keyword.get(opts, :reservoir_size, 500))
  end
end
