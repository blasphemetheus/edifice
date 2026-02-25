defmodule Edifice.Scientific.FNO do
  @moduledoc """
  FNO: Fourier Neural Operator.

  <!-- verified: true, date: 2026-02-23 -->

  Implements the Fourier Neural Operator from "Fourier Neural Operator for
  Parametric Partial Differential Equations" (Li et al., ICLR 2021). FNO
  learns operators mapping between infinite-dimensional function spaces via
  spectral convolutions, enabling 1000x faster PDE solving compared to
  traditional numerical methods.

  ## Key Innovation: Spectral Convolution

  Traditional convolutions operate in spatial domain with fixed kernel size.
  Spectral convolution operates in frequency domain:

  1. FFT input to frequency space
  2. Multiply by learned complex weights (only low frequencies)
  3. IFFT back to spatial domain
  4. Add pointwise linear bypass

  ```
  Input u(x) [batch, grid, channels]
        |
        +-------+-------+
        |               |
        v               v
  +-----------+    +---------+
  | FFT       |    | Linear  |   (bypass path)
  | Spectral  |    | W*u     |
  | Multiply  |    +---------+
  | IFFT      |         |
  +-----------+         |
        |               |
        +-------+-------+
                |
                v
  Output v(x) = IFFT(R * FFT(u)) + W*u
  ```

  ## Architecture

  ```
  Input [batch, grid_size, in_channels]
        |
        v
  +---------------------------+
  | Lifting: Linear(in -> h)  |
  +---------------------------+
        |
        v
  +---------------------------+
  | FNO Block x num_layers    |
  |   Spectral Conv           |
  |   + Pointwise Linear      |
  |   + Activation            |
  +---------------------------+
        |
        v
  +---------------------------+
  | Projection: Linear(h->out)|
  +---------------------------+
        |
        v
  Output [batch, grid_size, out_channels]
  ```

  ## Applications

  - Solving PDEs (Navier-Stokes, Burgers, Darcy flow)
  - Weather prediction (GraphCast uses spectral methods)
  - Fluid dynamics simulation
  - Any physics where solutions live in function spaces

  ## Usage

      model = FNO.build(
        in_channels: 1,
        out_channels: 1,
        modes: 16,
        hidden_channels: 64,
        num_layers: 4
      )

      # Input: discretized function on a grid
      # Output: solution of the PDE on the same grid

  ## Reference

  - Paper: "Fourier Neural Operator for Parametric Partial Differential Equations"
  - arXiv: https://arxiv.org/abs/2010.08895
  - ICLR 2021
  """

  @default_hidden_channels 64
  @default_num_layers 4
  @default_modes 16

  @doc """
  Build a Fourier Neural Operator for learning operators between function spaces.

  ## Options

    - `:in_channels` - Number of input channels (required)
    - `:out_channels` - Number of output channels (required)
    - `:modes` - Number of Fourier modes to keep (default: 16)
    - `:hidden_channels` - Hidden dimension (default: 64)
    - `:num_layers` - Number of FNO blocks (default: 4)
    - `:activation` - Activation function (default: :gelu)

  ## Returns

    An Axon model that maps input functions to output functions.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:hidden_channels, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:modes, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:out_channels, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    in_channels = Keyword.fetch!(opts, :in_channels)
    out_channels = Keyword.fetch!(opts, :out_channels)
    modes = Keyword.get(opts, :modes, @default_modes)
    hidden_channels = Keyword.get(opts, :hidden_channels, @default_hidden_channels)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    activation = Keyword.get(opts, :activation, :gelu)

    # Input: [batch, grid_size, in_channels]
    input = Axon.input("input", shape: {nil, nil, in_channels})

    # Lifting layer: project to hidden dimension
    x = Axon.dense(input, hidden_channels, name: "lift")

    # FNO blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_fno_block(acc,
          hidden_channels: hidden_channels,
          modes: modes,
          activation: activation,
          name: "fno_block_#{layer_idx}"
        )
      end)

    # Projection layer: project to output dimension
    x
    |> Axon.dense(hidden_channels, name: "proj_1")
    |> Axon.activation(activation, name: "proj_act")
    |> Axon.dense(out_channels, name: "proj_2")
  end

  @doc """
  Build a single FNO block with spectral convolution and pointwise bypass.
  """
  @spec build_fno_block(Axon.t(), keyword()) :: Axon.t()
  def build_fno_block(input, opts) do
    hidden_channels = Keyword.get(opts, :hidden_channels, @default_hidden_channels)
    modes = Keyword.get(opts, :modes, @default_modes)
    activation = Keyword.get(opts, :activation, :gelu)
    name = Keyword.get(opts, :name, "fno_block")

    # Spectral convolution path
    spectral_out = build_spectral_conv(input, hidden_channels, modes, "#{name}_spectral")

    # Pointwise linear bypass path
    bypass_out = Axon.dense(input, hidden_channels, name: "#{name}_bypass")

    # Combine paths and apply activation
    Axon.add(spectral_out, bypass_out, name: "#{name}_combine")
    |> Axon.activation(activation, name: "#{name}_act")
  end

  @doc """
  Build a spectral convolution layer.

  Applies FFT -> multiply by learned complex weights -> IFFT.
  Only keeps `modes` low-frequency components to learn.
  """
  @spec build_spectral_conv(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_spectral_conv(input, hidden_channels, modes, name) do
    # Learnable complex weights for spectral multiplication
    # Shape: [modes, hidden_channels, hidden_channels] for real part
    # Shape: [modes, hidden_channels, hidden_channels] for imaginary part
    weights_real =
      Axon.param("#{name}_weights_real", {modes, hidden_channels, hidden_channels},
        initializer: :glorot_uniform
      )

    weights_imag =
      Axon.param("#{name}_weights_imag", {modes, hidden_channels, hidden_channels},
        initializer: :glorot_uniform
      )

    Axon.layer(
      &spectral_conv_impl/4,
      [input, weights_real, weights_imag],
      name: "#{name}_conv",
      modes: modes,
      hidden_channels: hidden_channels,
      op_name: :spectral_conv
    )
  end

  # Spectral convolution implementation
  # 1. FFT the input
  # 2. Multiply low-frequency modes by learned weights
  # 3. IFFT back to spatial domain
  defp spectral_conv_impl(x, weights_real, weights_imag, opts) do
    modes = opts[:modes]
    hidden_channels = opts[:hidden_channels]

    batch = Nx.axis_size(x, 0)
    grid_size = Nx.axis_size(x, 1)

    # FFT along spatial dimension
    # x: [batch, grid_size, hidden_channels]
    # x_ft: [batch, grid_size, hidden_channels] (complex)
    x_ft = fft_1d(x)

    # Extract low-frequency modes
    # Only keep first `modes` frequencies (positive frequencies)
    actual_modes = min(modes, div(grid_size, 2) + 1)

    x_ft_modes = Nx.slice_along_axis(x_ft, 0, actual_modes, axis: 1)

    # Complex multiplication with learned weights
    # weights: [modes, hidden_channels, hidden_channels]
    # x_ft_modes: [batch, modes, hidden_channels]
    # Result: [batch, modes, hidden_channels]

    # Pad weights if needed
    {weights_real_padded, weights_imag_padded} =
      if actual_modes < modes do
        {
          Nx.slice_along_axis(weights_real, 0, actual_modes, axis: 0),
          Nx.slice_along_axis(weights_imag, 0, actual_modes, axis: 0)
        }
      else
        {weights_real, weights_imag}
      end

    out_ft_modes = complex_matmul(x_ft_modes, weights_real_padded, weights_imag_padded)

    # Reconstruct full spectrum by padding with zeros for high frequencies
    # Shape: [batch, grid_size, hidden_channels]
    out_ft = pad_spectrum(out_ft_modes, grid_size, hidden_channels, batch, actual_modes)

    # Inverse FFT
    ifft_1d(out_ft, grid_size)
  end

  # Simple 1D FFT using DFT matrix multiplication
  # Real implementation - works without Nx.fft which may not exist
  defp fft_1d(x) do
    # x: [batch, grid_size, channels]
    _batch = Nx.axis_size(x, 0)
    n = Nx.axis_size(x, 1)
    _channels = Nx.axis_size(x, 2)

    # DFT matrix: W[k,j] = exp(-2*pi*i*k*j/n)
    k = Nx.iota({n, 1}, type: :f32)
    j = Nx.iota({1, n}, type: :f32)
    angle = Nx.multiply(Nx.tensor(-2.0 * :math.pi() / n), Nx.multiply(k, j))

    # Real and imaginary parts of DFT matrix
    dft_real = Nx.cos(angle)
    dft_imag = Nx.sin(angle)

    # Apply DFT to each channel
    # x: [batch, n, channels] -> transpose to [batch, channels, n]
    x_t = Nx.transpose(x, axes: [0, 2, 1])

    # x_t: [batch, channels, n]
    # dft_real: [n, n]
    # result: [batch, channels, n]
    out_real = Nx.dot(x_t, [2], dft_real, [1])
    out_imag = Nx.dot(x_t, [2], dft_imag, [1])

    # Transpose back and combine as complex tensor (interleaved real/imag)
    # [batch, n, channels * 2] where even indices are real, odd are imaginary
    out_real_t = Nx.transpose(out_real, axes: [0, 2, 1])
    out_imag_t = Nx.transpose(out_imag, axes: [0, 2, 1])

    # Stack along last axis: [batch, n, channels, 2]
    Nx.stack([out_real_t, out_imag_t], axis: 3)
  end

  # Inverse FFT
  defp ifft_1d(x_ft, grid_size) do
    # x_ft: [batch, grid_size, channels, 2] where last dim is [real, imag]
    _batch = Nx.axis_size(x_ft, 0)
    n = grid_size
    _channels = Nx.axis_size(x_ft, 2)

    # Extract real and imaginary parts
    x_real = Nx.squeeze(Nx.slice_along_axis(x_ft, 0, 1, axis: 3), axes: [3])
    x_imag = Nx.squeeze(Nx.slice_along_axis(x_ft, 1, 1, axis: 3), axes: [3])

    # IDFT matrix: W[k,j] = exp(2*pi*i*k*j/n) / n
    k = Nx.iota({n, 1}, type: :f32)
    j = Nx.iota({1, n}, type: :f32)
    angle = Nx.multiply(Nx.tensor(2.0 * :math.pi() / n), Nx.multiply(k, j))

    idft_real = Nx.divide(Nx.cos(angle), n)
    idft_imag = Nx.divide(Nx.sin(angle), n)

    # Apply IDFT (only take real part of result)
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    # We want the real part: ac - bd
    x_real_t = Nx.transpose(x_real, axes: [0, 2, 1])
    x_imag_t = Nx.transpose(x_imag, axes: [0, 2, 1])

    # out_real = x_real * idft_real - x_imag * idft_imag
    out_real_part1 = Nx.dot(x_real_t, [2], idft_real, [1])
    out_real_part2 = Nx.dot(x_imag_t, [2], idft_imag, [1])
    out_real = Nx.subtract(out_real_part1, out_real_part2)

    Nx.transpose(out_real, axes: [0, 2, 1])
  end

  # Complex matrix multiplication
  # (a + bi) @ (c + di) = (ac - bd) + (ad + bc)i
  defp complex_matmul(x_ft, w_real, w_imag) do
    # x_ft: [batch, modes, channels, 2]
    # w_real, w_imag: [modes, channels, channels]

    _batch = Nx.axis_size(x_ft, 0)
    _modes_actual = Nx.axis_size(x_ft, 1)
    _channels = Nx.axis_size(x_ft, 2)

    # Extract real and imaginary parts of input
    x_real = Nx.squeeze(Nx.slice_along_axis(x_ft, 0, 1, axis: 3), axes: [3])
    x_imag = Nx.squeeze(Nx.slice_along_axis(x_ft, 1, 1, axis: 3), axes: [3])

    # For each mode m, compute: x[b,m,:] @ W[m,:,:]
    # x_real: [batch, modes, channels]
    # w_real: [modes, channels, channels]

    # Real part: ac - bd
    # [batch, modes, channels]
    ac = batched_mode_matmul(x_real, w_real)
    bd = batched_mode_matmul(x_imag, w_imag)
    out_real = Nx.subtract(ac, bd)

    # Imaginary part: ad + bc
    ad = batched_mode_matmul(x_real, w_imag)
    bc = batched_mode_matmul(x_imag, w_real)
    out_imag = Nx.add(ad, bc)

    # Stack back to complex format
    Nx.stack([out_real, out_imag], axis: 3)
  end

  # Batched matrix multiplication per mode
  # x: [batch, modes, channels]
  # w: [modes, channels, channels]
  # output: [batch, modes, channels]
  defp batched_mode_matmul(x, w) do
    _batch = Nx.axis_size(x, 0)
    _modes = Nx.axis_size(x, 1)
    _channels = Nx.axis_size(x, 2)

    # Transpose to [modes, batch, channels] to batch over modes
    x_t = Nx.transpose(x, axes: [1, 0, 2])

    # For batched matmul: x_t @ w where both have modes as batch dim
    # x_t: [modes, batch, channels_in]
    # w: [modes, channels_in, channels_out]
    # Result: [modes, batch, channels_out]
    # Use Nx.dot with batch_axes: [0] on both sides
    result = Nx.dot(x_t, [2], [0], w, [1], [0])

    # Transpose back to [batch, modes, channels]
    Nx.transpose(result, axes: [1, 0, 2])
  end

  # Pad spectrum with zeros for high frequencies
  defp pad_spectrum(x_ft_modes, grid_size, hidden_channels, batch, actual_modes) do
    # x_ft_modes: [batch, actual_modes, hidden_channels, 2]
    # Need: [batch, grid_size, hidden_channels, 2]

    # Create zero tensor for padding
    pad_size = grid_size - actual_modes
    zeros = Nx.broadcast(Nx.tensor(0.0), {batch, pad_size, hidden_channels, 2})

    # Concatenate along frequency axis
    Nx.concatenate([x_ft_modes, zeros], axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size for an FNO model (matches input grid).
  """
  @spec output_size(keyword()) :: :grid_dependent
  def output_size(_opts \\ []) do
    :grid_dependent
  end

  @doc """
  Calculate approximate parameter count for an FNO model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    in_channels = Keyword.get(opts, :in_channels, 1)
    out_channels = Keyword.get(opts, :out_channels, 1)
    hidden_channels = Keyword.get(opts, :hidden_channels, @default_hidden_channels)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    modes = Keyword.get(opts, :modes, @default_modes)

    # Lifting layer
    lift = in_channels * hidden_channels

    # Per FNO block: spectral weights (2x for complex) + bypass
    spectral = 2 * modes * hidden_channels * hidden_channels
    bypass = hidden_channels * hidden_channels
    per_block = spectral + bypass

    # Projection layers
    proj = hidden_channels * hidden_channels + hidden_channels * out_channels

    lift + num_layers * per_block + proj
  end

  @doc """
  Get recommended defaults for FNO.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_channels: @default_hidden_channels,
      num_layers: @default_num_layers,
      modes: @default_modes,
      activation: :gelu
    ]
  end
end
