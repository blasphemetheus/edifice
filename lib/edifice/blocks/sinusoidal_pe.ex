defmodule Edifice.Blocks.SinusoidalPE do
  @moduledoc """
  Sinusoidal Positional Encoding.

  The original positional encoding from "Attention Is All You Need", using
  sine and cosine functions at different frequencies to encode absolute
  position information. Deterministic (no learned parameters).

  ## Formula

      PE(pos, 2i)   = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  ## Usage

      # As an Axon layer that adds PE to input
      encoded = SinusoidalPE.layer(input, dim: 256)

      # Precompute PE table
      pe_table = SinusoidalPE.build_table(max_len: 512, dim: 256)

  ## References
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - https://arxiv.org/abs/1706.03762
  """

  @doc """
  Build a sinusoidal positional encoding table.

  Returns a tensor of shape [max_len, dim] with precomputed PE values.

  ## Options
    - `:max_len` - Maximum sequence length (default: 512)
    - `:dim` - Embedding dimension (required)

  ## Examples

      iex> table = Edifice.Blocks.SinusoidalPE.build_table(dim: 8, max_len: 16)
      iex> Nx.shape(table)
      {16, 8}
  """
  @spec build_table(keyword()) :: Nx.Tensor.t()
  def build_table(opts) do
    max_len = Keyword.get(opts, :max_len, 512)
    dim = Keyword.fetch!(opts, :dim)

    half_dim = div(dim, 2)

    # Frequency bands: 1 / 10000^(2i/dim)
    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim - 1),
          -:math.log(10_000.0)
        )
      )

    positions = Nx.iota({max_len}, type: :f32)

    # angles: [max_len, half_dim]
    angles = Nx.outer(positions, freqs)

    # Interleave sin and cos
    sin_pe = Nx.sin(angles)
    cos_pe = Nx.cos(angles)

    # [max_len, dim] with sin in even indices, cos in odd
    Nx.concatenate([sin_pe, cos_pe], axis: 1)
  end

  @doc """
  Build an Axon layer that computes sinusoidal timestep embedding.

  Takes a scalar timestep `[batch]` or `[batch, 1]` and produces an embedding
  `[batch, hidden_size]`. Used for diffusion model conditioning.

  ## Options
    - `:hidden_size` - Output embedding dimension (required)
    - `:num_steps` - If provided, normalizes timestep by dividing (default: nil)
    - `:name` - Layer name prefix (default: "time_sinusoidal")
  """
  @spec timestep_layer(Axon.t(), keyword()) :: Axon.t()
  def timestep_layer(timestep, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_steps = Keyword.get(opts, :num_steps, nil)
    name = Keyword.get(opts, :name, "time_sinusoidal")

    Axon.layer(
      &timestep_embed_impl/2,
      [timestep],
      name: name,
      hidden_size: hidden_size,
      num_steps: num_steps,
      op_name: :sinusoidal_embed
    )
  end

  defp timestep_embed_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    num_steps = opts[:num_steps]
    half_dim = div(hidden_size, 2)

    t_f =
      if num_steps do
        Nx.divide(Nx.as_type(t, :f32), num_steps)
      else
        Nx.as_type(t, :f32)
      end

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    angles = Nx.multiply(Nx.new_axis(t_f, 1), Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  @doc """
  Build an Axon layer that adds sinusoidal positional encoding to the input.

  ## Options
    - `:dim` - Feature dimension (required)
    - `:name` - Layer name prefix (default: "sinusoidal_pe")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    _dim = Keyword.fetch!(opts, :dim)
    name = Keyword.get(opts, :name, "sinusoidal_pe")

    Axon.nx(
      input,
      fn tensor ->
        seq_len = Nx.axis_size(tensor, 1)
        dim = Nx.axis_size(tensor, 2)
        half_dim = div(dim, 2)

        freqs =
          Nx.exp(
            Nx.multiply(
              Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1)),
              -:math.log(10_000.0)
            )
          )

        positions = Nx.iota({seq_len}, type: :f32)
        angles = Nx.outer(positions, freqs)

        pe = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
        pe = Nx.reshape(pe, {1, seq_len, dim})

        Nx.add(tensor, pe)
      end,
      name: name
    )
  end
end
