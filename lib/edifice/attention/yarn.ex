defmodule Edifice.Attention.YARN do
  @moduledoc """
  YaRN: Yet another RoPE extensioN for context window extension.

  <!-- verified: true, date: 2026-02-23 -->

  YaRN modifies RoPE frequency bands to handle longer sequences than the model
  was originally trained on. It achieves this by scaling different frequency
  components based on their wavelength relative to the original context length.

  ## Key Insight

  RoPE encodes position via rotations at different frequencies. For context
  extension, high-frequency (local position) info should be preserved while
  low-frequency (global position) info needs scaling:

  ```
  High-frequency bands (short wavelength):
    - Capture local positional relationships
    - Left unchanged (factor = 1)

  Low-frequency bands (long wavelength):
    - Capture global position in sequence
    - Scaled down by 1/scale factor

  Middle bands:
    - Linear interpolation between the two regimes
  ```

  ## Formula

  For each dimension i, compute a scaling factor:
  - If wavelength < high_freq_threshold: factor = 1 (unchanged)
  - If wavelength > low_freq_threshold: factor = 1/scale (full scaling)
  - Otherwise: linear interpolation between 1 and 1/scale

  The thresholds are derived from:
  - `high_freq_wavelen = original_max_position / high_freq_factor`
  - `low_freq_wavelen = original_max_position / low_freq_factor`

  ## Usage

      # Build a YaRN-modified RoPE layer
      model = YARN.build(
        embed_dim: 64,
        scale: 8,
        original_max_position: 2048
      )

      # Get the frequency table directly
      freqs = YARN.yarn_freqs(64,
        scale: 8,
        original_max_position: 2048
      )

      # Apply YaRN to query/key tensors
      {q_rotated, k_rotated} = YARN.apply_yarn(q, k,
        scale: 8,
        original_max_position: 2048
      )

  ## References

  - "YaRN: Efficient Context Window Extension of Large Language Models"
    (Peng et al., 2023) — https://arxiv.org/abs/2309.00071
  """

  import Nx.Defn

  @default_scale 8
  @default_original_max_position 2048
  @default_low_freq_factor 1
  @default_high_freq_factor 4
  @default_base 10_000.0
  # Pre-computed constant for use inside defn (where :math.pi/0 is not allowed)
  @two_pi 2 * :math.pi()

  @typedoc "Options for YaRN functions."
  @type yarn_opt ::
          {:embed_dim, pos_integer()}
          | {:scale, number()}
          | {:original_max_position, pos_integer()}
          | {:low_freq_factor, number()}
          | {:high_freq_factor, number()}
          | {:base, number()}
          | {:name, String.t()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an Axon model that applies YaRN-modified RoPE to input.

  ## Options

    - `:embed_dim` - Feature dimension (required, must be even)
    - `:scale` - Context extension scale factor (default: 8)
      For example, scale=8 extends 2048 to 16384 context length
    - `:original_max_position` - Original trained context length (default: 2048)
    - `:low_freq_factor` - Low frequency boundary factor (default: 1)
    - `:high_freq_factor` - High frequency boundary factor (default: 4)
    - `:base` - RoPE base frequency (default: 10000.0)
    - `:name` - Layer name prefix (default: "yarn")

  ## Returns

    An Axon model that applies YaRN-modified RoPE to the input tensor.
  """
  @spec build([yarn_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    scale = Keyword.get(opts, :scale, @default_scale)
    original_max_position = Keyword.get(opts, :original_max_position, @default_original_max_position)
    low_freq_factor = Keyword.get(opts, :low_freq_factor, @default_low_freq_factor)
    high_freq_factor = Keyword.get(opts, :high_freq_factor, @default_high_freq_factor)
    base = Keyword.get(opts, :base, @default_base)
    name = Keyword.get(opts, :name, "yarn")

    input = Axon.input("yarn_input", shape: {nil, nil, embed_dim})

    Axon.nx(
      input,
      fn tensor ->
        apply_yarn_impl(
          tensor,
          tensor,
          scale,
          original_max_position,
          low_freq_factor,
          high_freq_factor,
          base
        )
        |> elem(0)
      end,
      name: name
    )
  end

  # ============================================================================
  # Frequency Computation
  # ============================================================================

  @doc """
  Compute YaRN-scaled frequency table.

  Returns a tensor of shape `[embed_dim / 2]` containing the scaled frequencies
  for each dimension pair.

  ## Options

    - `:scale` - Context extension scale factor (default: 8)
    - `:original_max_position` - Original trained context length (default: 2048)
    - `:low_freq_factor` - Low frequency boundary factor (default: 1)
    - `:high_freq_factor` - High frequency boundary factor (default: 4)
    - `:base` - RoPE base frequency (default: 10000.0)

  ## Example

      freqs = YARN.yarn_freqs(64, scale: 8, original_max_position: 2048)
      # => Tensor of shape {32} with scaled frequencies
  """
  @spec yarn_freqs(pos_integer(), keyword()) :: Nx.Tensor.t()
  def yarn_freqs(embed_dim, opts \\ []) do
    scale = Keyword.get(opts, :scale, @default_scale)
    original_max_position = Keyword.get(opts, :original_max_position, @default_original_max_position)
    low_freq_factor = Keyword.get(opts, :low_freq_factor, @default_low_freq_factor)
    high_freq_factor = Keyword.get(opts, :high_freq_factor, @default_high_freq_factor)
    base = Keyword.get(opts, :base, @default_base)

    half_dim = div(embed_dim, 2)

    # Base frequencies: theta_i = base^(-2i/dim)
    base_freqs =
      Nx.pow(
        base,
        Nx.negate(Nx.divide(Nx.multiply(2, Nx.iota({half_dim})), embed_dim))
      )
      |> Nx.as_type(:f32)

    # Wavelength for each frequency: lambda_i = 2*pi / freq_i
    wavelengths = Nx.divide(2 * :math.pi(), base_freqs)

    # Compute thresholds
    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    # Compute ramp for interpolation
    # ramp = 0 for high-freq (short wavelength), 1 for low-freq (long wavelength)
    ramp =
      Nx.subtract(wavelengths, high_freq_wavelen)
      |> Nx.divide(low_freq_wavelen - high_freq_wavelen)
      |> Nx.clip(0.0, 1.0)

    # Factor: 1.0 for high-freq, 1/scale for low-freq, interpolated in between
    inv_scale = 1.0 / scale
    factor = Nx.add(Nx.multiply(ramp, inv_scale), Nx.multiply(Nx.subtract(1.0, ramp), 1.0))

    # Apply scaling factor to frequencies
    Nx.multiply(base_freqs, factor)
  end

  # ============================================================================
  # Application Functions
  # ============================================================================

  @doc """
  Apply YaRN-modified RoPE to query and key tensors.

  ## Parameters

    - `query` - Query tensor `[batch, seq_len, embed_dim]`
    - `key` - Key tensor `[batch, seq_len, embed_dim]`
    - `opts` - Options (see `yarn_freqs/2` for available options)

  ## Returns

    `{rotated_query, rotated_key}` with same shapes as input.

  ## Example

      {q_rot, k_rot} = YARN.apply_yarn(query, key, scale: 8)
  """
  @spec apply_yarn(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def apply_yarn(query, key, opts \\ []) do
    scale = Keyword.get(opts, :scale, @default_scale)
    original_max_position = Keyword.get(opts, :original_max_position, @default_original_max_position)
    low_freq_factor = Keyword.get(opts, :low_freq_factor, @default_low_freq_factor)
    high_freq_factor = Keyword.get(opts, :high_freq_factor, @default_high_freq_factor)
    base = Keyword.get(opts, :base, @default_base)

    apply_yarn_impl(query, key, scale, original_max_position, low_freq_factor, high_freq_factor, base)
  end

  defnp apply_yarn_impl(query, key, scale, original_max_position, low_freq_factor, high_freq_factor, base) do
    embed_dim = Nx.axis_size(query, 2)
    seq_len = Nx.axis_size(query, 1)
    half_dim = div(embed_dim, 2)

    # Compute YaRN-scaled frequencies
    base_freqs =
      Nx.pow(
        base,
        Nx.negate(Nx.divide(Nx.multiply(2.0, Nx.iota({half_dim}, type: :f32)), embed_dim))
      )

    wavelengths = Nx.divide(@two_pi, base_freqs)

    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    ramp =
      Nx.subtract(wavelengths, high_freq_wavelen)
      |> Nx.divide(low_freq_wavelen - high_freq_wavelen)
      |> Nx.clip(0.0, 1.0)

    inv_scale = 1.0 / scale
    factor = Nx.add(Nx.multiply(ramp, inv_scale), Nx.multiply(Nx.subtract(1.0, ramp), 1.0))
    freqs = Nx.multiply(base_freqs, factor)

    # Build rotation tables
    positions = Nx.iota({seq_len}, type: :f32)
    angles = Nx.outer(positions, freqs)

    cos_table = Nx.cos(angles) |> Nx.new_axis(0)
    sin_table = Nx.sin(angles) |> Nx.new_axis(0)

    # Apply rotation to query and key
    q_rotated = rotate_half(query, cos_table, sin_table, half_dim)
    k_rotated = rotate_half(key, cos_table, sin_table, half_dim)

    {q_rotated, k_rotated}
  end

  defnp rotate_half(x, cos_table, sin_table, half_dim) do
    x1 = Nx.slice_along_axis(x, 0, half_dim, axis: 2)
    x2 = Nx.slice_along_axis(x, half_dim, half_dim, axis: 2)

    rotated1 = Nx.subtract(Nx.multiply(x1, cos_table), Nx.multiply(x2, sin_table))
    rotated2 = Nx.add(Nx.multiply(x1, sin_table), Nx.multiply(x2, cos_table))

    Nx.concatenate([rotated1, rotated2], axis: 2)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get recommended defaults for YaRN.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      scale: @default_scale,
      original_max_position: @default_original_max_position,
      low_freq_factor: @default_low_freq_factor,
      high_freq_factor: @default_high_freq_factor,
      base: @default_base
    ]
  end

  @doc """
  Calculate the effective context length after YaRN scaling.

  ## Example

      YARN.effective_context_length(2048, 8)
      # => 16384
  """
  @spec effective_context_length(pos_integer(), number()) :: number()
  def effective_context_length(original_max_position, scale) do
    original_max_position * scale
  end
end
