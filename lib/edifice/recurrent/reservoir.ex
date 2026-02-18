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
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_scaling, float()}
          | {:input_size, pos_integer()}
          | {:leak_rate, float()}
          | {:output_size, pos_integer()}
          | {:reservoir_size, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:sparsity, float()}
          | {:spectral_radius, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    reservoir_size = Keyword.get(opts, :reservoir_size, 500)
    output_size = Keyword.get(opts, :output_size, reservoir_size)
    spectral_radius = Keyword.get(opts, :spectral_radius, 0.9)
    sparsity = Keyword.get(opts, :sparsity, 0.9)
    input_scaling = Keyword.get(opts, :input_scaling, 1.0)
    leak_rate = Keyword.get(opts, :leak_rate, 1.0)
    seq_len = Keyword.get(opts, :seq_len, nil)

    # Input: [batch, seq_len, input_size]
    input = Axon.input("input", shape: {nil, seq_len, input_size})

    # Generate fixed reservoir weights at BUILD TIME (computed once, frozen forever)
    # This is the key property of ESNs: reservoir weights are never trained
    seed = Keyword.get(opts, :seed, 42)
    key = Nx.Random.key(seed)

    # Input weights: [input_size, reservoir_size]
    {w_in_data, key} = Nx.Random.normal(key, shape: {input_size, reservoir_size})
    w_in_data = Nx.multiply(w_in_data, input_scaling)

    # Reservoir weights: [reservoir_size, reservoir_size] with sparsity
    {w_res_data, key} = Nx.Random.normal(key, shape: {reservoir_size, reservoir_size})
    {mask_vals, _key} = Nx.Random.uniform(key, shape: {reservoir_size, reservoir_size})
    sparse_mask = Nx.greater(mask_vals, sparsity)
    w_res_data = Nx.select(sparse_mask, w_res_data, Nx.tensor(0.0))

    # Scale to target spectral radius (Frobenius norm approximation)
    frobenius_norm = Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(w_res_data, 2))))
    est_spectral = frobenius_norm / :math.sqrt(max(reservoir_size * (1.0 - sparsity), 1.0))
    scale = spectral_radius / max(est_spectral, 1.0e-8)
    w_res_data = Nx.multiply(w_res_data, scale)

    # Inject as Axon.constant nodes (frozen, never trained)
    w_in_const = Axon.constant(w_in_data, name: "w_in")
    w_res_const = Axon.constant(w_res_data, name: "w_res")

    # Reservoir dynamics with fixed weights
    reservoir_output =
      Axon.layer(
        &reservoir_forward/4,
        [input, w_in_const, w_res_const],
        name: "reservoir",
        reservoir_size: reservoir_size,
        leak_rate: leak_rate,
        op_name: :reservoir
      )

    # Trainable readout (only this gets optimized)
    Axon.dense(reservoir_output, output_size, name: "readout")
  end

  # Reservoir forward pass using pre-computed fixed weights
  defp reservoir_forward(input, w_in, w_res, opts) do
    reservoir_size = opts[:reservoir_size]
    leak_rate = opts[:leak_rate]

    batch_size = Nx.axis_size(input, 0)
    seq_len = Nx.axis_size(input, 1)

    # Run reservoir dynamics: h[t] = tanh(W_in * x[t] + W_res * h[t-1])
    h = Nx.broadcast(0.0, {batch_size, reservoir_size})

    final_h =
      Enum.reduce(0..(seq_len - 1), h, fn t, h_prev ->
        x_t = Nx.slice_along_axis(input, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

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
