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
    not initial conditions. A spectral radius below one is a common initialization
    heuristic, not a sufficient guarantee for arbitrary driven nonlinear reservoirs.
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

    unless is_number(spectral_radius) and spectral_radius >= 0,
      do: raise(ArgumentError, "spectral_radius must be nonnegative")

    unless is_number(sparsity) and sparsity >= 0 and sparsity <= 1,
      do: raise(ArgumentError, "sparsity must be between zero and one")

    unless is_number(leak_rate) and leak_rate > 0 and leak_rate <= 1,
      do: raise(ArgumentError, "leak_rate must be greater than zero and at most one")

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

    # Use the eigenvalue magnitudes of the actual sparse, nonsymmetric matrix.
    # A zero-radius draw cannot be scaled to a positive requested radius.
    w_res_data = scale_spectral_radius(w_res_data, spectral_radius)

    # Reservoir dynamics: input projection + recurrent scan
    # Uses Axon.nx with closure-captured frozen weights. Must copy to BinaryBackend
    # so EXLA can inline them as constants in the compiled graph (avoids
    # "incompatible tensor implementations: Nx.Defn.Expr and EXLA.Backend" error).
    w_in_const = Nx.backend_copy(w_in_data, Nx.BinaryBackend)
    w_res_const = Nx.backend_copy(w_res_data, Nx.BinaryBackend)

    reservoir_output =
      Axon.nx(
        input,
        fn x ->
          batch_size = Nx.axis_size(x, 0)
          s_len = Nx.axis_size(x, 1)

          # Pre-compute W_in @ x for all timesteps: [batch, seq_len, reservoir_size]
          input_2d = Nx.reshape(x, {batch_size * s_len, Nx.axis_size(x, 2)})
          wx = Nx.dot(input_2d, w_in_const) |> Nx.reshape({batch_size, s_len, reservoir_size})

          # Reservoir scan (fallback path — EXLA compiles Nx ops to XLA graph directly)
          Edifice.CUDA.FusedScan.reservoir_scan_fallback(wx, w_res_const, leak_rate)
        end,
        name: "reservoir"
      )

    # Trainable readout (only this gets optimized)
    Axon.dense(reservoir_output, output_size, name: "readout")
  end

  @doc """
  Scale a real square reservoir matrix to the requested spectral radius.

  Uses host double-precision Hessenberg/QR eigenvalues, including complex pairs,
  once at build time. Raises on nonconvergence or a zero-radius matrix with a
  positive target. A zero target explicitly disables recurrent connections.
  This changes the frozen reservoir generated by older Edifice versions; old
  readout weights require the original reservoir to reproduce their predictions.
  """
  def scale_spectral_radius(matrix, target) when is_number(target) and target >= 0 do
    matrix = Nx.as_type(matrix, Nx.Type.to_floating(Nx.type(matrix)))

    if target == 0 do
      Nx.multiply(matrix, 0)
    else
      radius = Edifice.Utils.SpectralRadius.compute(matrix)

      if radius == 0,
        do:
          raise(
            ArgumentError,
            "cannot scale a zero-spectral-radius reservoir to a positive target; change seed or sparsity"
          )

      Nx.multiply(matrix, Nx.tensor(target / radius, type: Nx.type(matrix)))
    end
  end

  @doc """
  Get the output size of the reservoir.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :output_size, Keyword.get(opts, :reservoir_size, 500))
  end
end
