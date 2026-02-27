defmodule Edifice.Blocks.SinusoidalPE2D do
  @moduledoc """
  2D sinusoidal positional encoding for flattened spatial grids.

  Encodes x and y positions independently using sinusoidal frequency bands,
  then concatenates them. Used by detection models (DETR, RT-DETR, SAM 2) to
  provide spatial position information to transformer encoders/decoders.

  The encoding assumes tokens come from a roughly square feature map.
  Dimension is split into four quarters: sin(y), cos(y), sin(x), cos(x).

  ## Usage

      # Pure tensor function for use inside Axon.nx closures
      pe = SinusoidalPE2D.build_table(seq_len, dim)
      # => [1, seq_len, dim]
  """

  @doc """
  Build a 2D sinusoidal positional encoding table.

  Returns a tensor of shape `[1, seq_len, dim]` with sinusoidal position
  encodings for a flattened spatial grid of approximately `sqrt(seq_len)`
  height and width.

  ## Parameters

    - `seq_len` - Number of spatial positions (H' * W')
    - `dim` - Encoding dimension (must be divisible by 4)
  """
  @spec build_table(pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def build_table(seq_len, dim) do
    # Approximate grid dimensions (assume roughly square)
    h = seq_len |> :math.sqrt() |> ceil() |> trunc()
    w = ceil(seq_len / h) |> trunc()

    half_dim = div(dim, 2)
    quarter_dim = div(half_dim, 2)

    # Frequency bands
    freq_indices = Nx.iota({quarter_dim})

    inv_freq =
      Nx.exp(
        Nx.negate(Nx.multiply(freq_indices, Nx.divide(Nx.log(Nx.tensor(10_000.0)), quarter_dim)))
      )

    # Y positions (rows)
    y_pos = Nx.iota({h, 1}) |> Nx.broadcast({h, w}) |> Nx.reshape({h * w, 1})
    y_pos = Nx.divide(y_pos, Nx.tensor(max(h - 1, 1), type: :f32))

    # X positions (cols)
    x_pos = Nx.iota({1, w}) |> Nx.broadcast({h, w}) |> Nx.reshape({h * w, 1})
    x_pos = Nx.divide(x_pos, Nx.tensor(max(w - 1, 1), type: :f32))

    # Sinusoidal encoding for y: [h*w, quarter_dim] each
    y_angles = Nx.dot(y_pos, Nx.reshape(inv_freq, {1, quarter_dim}))
    y_pe = Nx.concatenate([Nx.sin(y_angles), Nx.cos(y_angles)], axis: 1)

    # Sinusoidal encoding for x: [h*w, quarter_dim] each
    x_angles = Nx.dot(x_pos, Nx.reshape(inv_freq, {1, quarter_dim}))
    x_pe = Nx.concatenate([Nx.sin(x_angles), Nx.cos(x_angles)], axis: 1)

    # Concatenate: [h*w, dim]
    pe = Nx.concatenate([y_pe, x_pe], axis: 1)

    # Truncate to exact seq_len (in case h*w > seq_len)
    pe = Nx.slice_along_axis(pe, 0, seq_len, axis: 0)

    # Add batch dimension: [1, seq_len, dim]
    Nx.reshape(pe, {1, seq_len, dim})
  end
end
