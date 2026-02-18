defmodule Edifice.Vision.NeRF do
  @moduledoc """
  NeRF: Neural Radiance Fields network (Mildenhall et al., 2020).

  Maps 3D coordinates (and optionally viewing directions) to color and density
  values using Fourier positional encoding followed by an MLP with skip
  connections. This is the core network architecture used in Neural Radiance
  Fields for novel view synthesis.

  ## Architecture

  ```
  Coordinates [batch, 3]
        |
  +-----v--------------------+
  | Fourier Encoding          |  gamma(p) = [p, sin(2^0*pi*p), cos(2^0*pi*p), ...]
  +---------------------------+
        |
        v
  [batch, 3 * (2*L + 1)]
        |
  +-----v--------------------+
  | Dense Layer 1             |  ReLU
  | Dense Layer 2             |  ReLU
  | ...                       |
  | Dense Layer K (skip_layer)|  Concatenate encoded input, ReLU
  | ...                       |
  | Dense Layer N             |  ReLU
  +---------------------------+
        |
        +---> Density sigma [batch, 1]
        |
        +---> Feature -> concat(directions_encoded) -> Dense -> RGB [batch, 3]
        |
        v
  Output [batch, 4]  (RGB + density)
  ```

  ## Differences from Vision Models

  Unlike other vision models in Edifice, NeRF does not take image inputs.
  Instead, it takes raw 3D coordinates and optional viewing directions,
  making it fundamentally a coordinate-to-color mapping network.

  ## Usage

      # With viewing directions
      model = NeRF.build(
        hidden_size: 256,
        num_layers: 8,
        skip_layer: 4,
        num_frequencies: 10,
        use_viewdir: true
      )

      # Without viewing directions
      model = NeRF.build(use_viewdir: false)

  ## References

  - Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields
    for View Synthesis" (ECCV 2020)
  - https://arxiv.org/abs/2003.08934
  """

  @default_coord_dim 3
  @default_dir_dim 3
  @default_hidden_size 256
  @default_num_layers 8
  @default_skip_layer 4
  @default_num_frequencies 10

  @doc """
  Build a NeRF network.

  ## Options

    - `:coord_dim` - Coordinate input dimension (default: 3)
    - `:dir_dim` - Viewing direction dimension (default: 3)
    - `:hidden_size` - Hidden layer size (default: 256)
    - `:num_layers` - Number of MLP layers (default: 8)
    - `:skip_layer` - Layer index for skip connection (default: 4)
    - `:num_frequencies` - Number of Fourier frequency bands (default: 10)
    - `:use_viewdir` - Whether to use viewing direction input (default: true)

  ## Returns

    An Axon model that takes "coordinates" (and optionally "directions") inputs
    and outputs `[batch, 4]` (RGB + density).
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:coord_dim, pos_integer()}
          | {:dir_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_frequencies, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:skip_layer, pos_integer()}
          | {:use_viewdir, boolean()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    coord_dim = Keyword.get(opts, :coord_dim, @default_coord_dim)
    dir_dim = Keyword.get(opts, :dir_dim, @default_dir_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    skip_layer = Keyword.get(opts, :skip_layer, @default_skip_layer)
    num_frequencies = Keyword.get(opts, :num_frequencies, @default_num_frequencies)
    use_viewdir = Keyword.get(opts, :use_viewdir, true)

    # Encoded dimension: original + sin/cos for each frequency
    _encoded_dim = coord_dim * (2 * num_frequencies + 1)

    coords = Axon.input("coordinates", shape: {nil, coord_dim})

    # Fourier positional encoding for coordinates
    encoded = fourier_encoding(coords, num_frequencies, "pos_encoding")

    # MLP with skip connection at skip_layer
    x =
      Enum.reduce(0..(num_layers - 1), encoded, fn idx, acc ->
        layer_input =
          if idx == skip_layer do
            Axon.concatenate([acc, encoded], axis: 1, name: "skip_#{idx}")
          else
            acc
          end

        layer_input
        |> Axon.dense(hidden_size, name: "nerf_dense_#{idx}")
        |> Axon.relu(name: "nerf_relu_#{idx}")
      end)

    if use_viewdir do
      directions = Axon.input("directions", shape: {nil, dir_dim})

      # Fourier encoding for directions
      dir_encoded = fourier_encoding(directions, num_frequencies, "dir_encoding")

      # Density output from position features
      sigma = Axon.dense(x, 1, name: "sigma_out")

      # Color branch: position features + encoded direction -> RGB
      feature = Axon.dense(x, hidden_size, name: "feature_proj")

      rgb =
        Axon.concatenate([feature, dir_encoded], axis: 1, name: "color_input")
        |> Axon.dense(div(hidden_size, 2), name: "color_dense")
        |> Axon.relu(name: "color_relu")
        |> Axon.dense(3, name: "rgb_out")
        |> Axon.sigmoid(name: "rgb_sigmoid")

      # Concatenate: [batch, 4] = [R, G, B, sigma]
      Axon.concatenate([rgb, sigma], axis: 1, name: "nerf_output")
    else
      # Without viewdir: output 4 values directly
      Axon.dense(x, 4, name: "nerf_output")
    end
  end

  @doc """
  Get the output size of a NeRF model.

  Always returns 4 (RGB + density).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(_opts \\ []), do: 4

  # Fourier positional encoding: gamma(p) = [p, sin(2^0*pi*p), cos(2^0*pi*p), ...]
  defp fourier_encoding(input, num_frequencies, name) do
    # Pre-compute frequency scales
    freq_scales =
      for l <- 0..(num_frequencies - 1), do: :math.pow(2.0, l) * :math.pi()

    Axon.nx(
      input,
      fn x ->
        sin_cos =
          Enum.flat_map(freq_scales, fn scale ->
            scaled = Nx.multiply(x, scale)
            [Nx.sin(scaled), Nx.cos(scaled)]
          end)

        Nx.concatenate([x | sin_cos], axis: 1)
      end,
      name: name
    )
  end
end
