defmodule Edifice.Audio.EnCodec do
  @moduledoc """
  EnCodec: High-Fidelity Neural Audio Compression.

  EnCodec is Meta's neural audio codec that encodes raw waveforms into discrete
  tokens suitable for audio language models. It uses a convolutional encoder-decoder
  architecture with a Residual Vector Quantizer (RVQ) to produce multiple streams
  of discrete tokens at different levels of detail.

  ## Architecture

  ```
  Waveform [batch, 1, samples]
        |
  +-------------------+
  | Encoder           |
  | 1D Conv           |
  | Downsample blocks |
  | (stride 2, 4, 5,  |
  |  8 = 320x total)  |
  +-------------------+
        |
  Continuous embeddings [batch, T, dim]
        |
  +-------------------+
  | Residual VQ       |
  | Q codebooks       |
  | Each quantizes    |
  | the residual      |
  +-------------------+
        |
  Discrete tokens [batch, Q, T]
        |
  (Dequantize: sum codebook vectors)
        |
  +-------------------+
  | Decoder           |
  | Upsample blocks   |
  | (stride 8, 5, 4,  |
  |  2 = 320x total)  |
  | 1D Conv           |
  +-------------------+
        |
  Reconstructed waveform [batch, 1, samples]
  ```

  ## Residual Vector Quantization (RVQ)

  RVQ uses Q codebooks in sequence. The first codebook quantizes the encoder
  output; each subsequent codebook quantizes the *residual* (error) from the
  previous quantization. This provides coarse-to-fine representation:

  - Codebook 0: captures overall structure (coarse)
  - Codebooks 1-7: capture progressively finer details

  ## Bandwidth Control

  Different bandwidths correspond to different numbers of active codebooks:
  - 1.5 kbps: 2 codebooks (very compressed)
  - 3.0 kbps: 4 codebooks
  - 6.0 kbps: 8 codebooks (high quality)
  - 12.0 kbps: 16 codebooks (studio quality)

  ## Usage

      # Build full EnCodec
      model = EnCodec.build(
        num_codebooks: 8,
        codebook_size: 1024,
        hidden_dim: 128
      )

      # Encode waveform to tokens
      tokens = EnCodec.encode(encoder, rvq, params, waveform)

      # Decode tokens back to waveform
      waveform = EnCodec.decode(decoder, rvq, params, tokens)

  ## References

  - Défossez et al., "High Fidelity Neural Audio Compression"
    (Meta AI, 2022) — https://arxiv.org/abs/2210.13438
  - SoundStream (predecessor): https://arxiv.org/abs/2107.03312
  """

  import Nx.Defn

  @default_num_codebooks 8
  @default_codebook_size 1024
  @default_hidden_dim 128
  # Downsampling/upsampling strides (product = 320)
  # At 24kHz, this gives 75 tokens/second
  @strides [2, 4, 5, 8]

  @typedoc "Options for `build/1` and related functions."
  @type build_opt ::
          {:num_codebooks, pos_integer()}
          | {:codebook_size, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:sample_rate, pos_integer()}
          | {:bandwidth, float()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a complete EnCodec model (encoder + RVQ + decoder).

  ## Options

    - `:num_codebooks` - Number of RVQ codebooks (default: 8)
    - `:codebook_size` - Vocabulary size per codebook (default: 1024)
    - `:hidden_dim` - Base hidden dimension (default: 128)
    - `:sample_rate` - Audio sample rate in Hz (default: 24000)
    - `:bandwidth` - Target bandwidth in kbps (default: 6.0)

  ## Returns

    A tuple `{encoder, decoder}` of Axon models.
    - Encoder: `[batch, 1, samples]` -> `[batch, T, dim]`
    - Decoder: `[batch, T, dim]` -> `[batch, 1, samples]`

  Note: RVQ is implemented as stateful quantization functions, not as an Axon model.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    encoder = build_encoder(opts)
    decoder = build_decoder(opts)
    {encoder, decoder}
  end

  @doc """
  Build the EnCodec encoder.

  The encoder uses a stack of residual blocks with strided convolutions for
  downsampling. Each block doubles the channel count while reducing temporal
  resolution.

  ## Options

    - `:hidden_dim` - Base hidden dimension, scaled up through layers (default: 128)

  ## Returns

    An Axon model: `[batch, 1, samples]` -> `[batch, T, final_dim]`
    where T = samples / 320 and final_dim = hidden_dim * 16.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)

    # Input: raw waveform [batch, 1, samples]
    input = Axon.input("waveform", shape: {nil, 1, nil})

    # Initial convolution: 1 -> hidden_dim channels
    x =
      Axon.conv(input, hidden_dim,
        kernel_size: {7},
        padding: :same,
        channels: :first,
        name: "encoder_conv_in"
      )

    # Downsampling blocks with increasing channels
    # Channels: hidden_dim -> 2*hidden_dim -> 4*hidden_dim -> 8*hidden_dim -> 16*hidden_dim
    {x, _} =
      Enum.reduce(Enum.with_index(@strides), {x, hidden_dim}, fn {stride, idx}, {acc, ch_in} ->
        ch_out = ch_in * 2
        block_out = encoder_block(acc, ch_in, ch_out, stride, "encoder_block_#{idx}")
        {block_out, ch_out}
      end)

    # Final convolution and transpose for [batch, T, dim] output
    x = Axon.activation(x, :elu, name: "encoder_act_final")

    final_dim = hidden_dim * 16

    x =
      Axon.conv(x, final_dim,
        kernel_size: {3},
        padding: :same,
        channels: :first,
        name: "encoder_conv_out"
      )

    # Transpose from [batch, channels, T] to [batch, T, channels]
    Axon.nx(x, fn t -> Nx.transpose(t, axes: [0, 2, 1]) end, name: "encoder_transpose")
  end

  @doc """
  Build the EnCodec decoder.

  The decoder mirrors the encoder with transposed convolutions for upsampling.

  ## Options

    - `:hidden_dim` - Base hidden dimension (default: 128)

  ## Returns

    An Axon model: `[batch, T, dim]` -> `[batch, 1, samples]`
  """
  @spec build_decoder([build_opt()]) :: Axon.t()
  def build_decoder(opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    final_dim = hidden_dim * 16

    # Input: continuous embeddings [batch, T, dim]
    input = Axon.input("embeddings", shape: {nil, nil, final_dim})

    # Transpose to [batch, dim, T] for conv operations
    x = Axon.nx(input, fn t -> Nx.transpose(t, axes: [0, 2, 1]) end, name: "decoder_transpose_in")

    # Initial convolution
    x =
      Axon.conv(x, final_dim,
        kernel_size: {7},
        padding: :same,
        channels: :first,
        name: "decoder_conv_in"
      )

    # Upsampling blocks with decreasing channels (reverse order of strides)
    # Channels: 16*hidden_dim -> 8*hidden_dim -> 4*hidden_dim -> 2*hidden_dim -> hidden_dim
    reversed_strides = Enum.reverse(@strides)

    {x, _} =
      Enum.reduce(Enum.with_index(reversed_strides), {x, final_dim}, fn {stride, idx},
                                                                        {acc, ch_in} ->
        ch_out = div(ch_in, 2)
        block_out = decoder_block(acc, ch_in, ch_out, stride, "decoder_block_#{idx}")
        {block_out, ch_out}
      end)

    # Final convolution to 1 channel (mono audio)
    x = Axon.activation(x, :elu, name: "decoder_act_final")

    Axon.conv(x, 1,
      kernel_size: {7},
      padding: :same,
      channels: :first,
      name: "decoder_conv_out"
    )
  end

  @doc """
  Build the Residual Vector Quantizer configuration.

  RVQ is not an Axon model but a set of codebook parameters. This function
  returns the configuration; actual codebooks are initialized separately.

  ## Options

    - `:num_codebooks` - Number of quantization levels (default: 8)
    - `:codebook_size` - Entries per codebook (default: 1024)
    - `:hidden_dim` - Embedding dimension (default: 128, scaled to final_dim)

  ## Returns

    A map with RVQ configuration.
  """
  @spec build_rvq([build_opt()]) :: map()
  def build_rvq(opts \\ []) do
    num_codebooks = Keyword.get(opts, :num_codebooks, @default_num_codebooks)
    codebook_size = Keyword.get(opts, :codebook_size, @default_codebook_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    embedding_dim = hidden_dim * 16

    %{
      num_codebooks: num_codebooks,
      codebook_size: codebook_size,
      embedding_dim: embedding_dim
    }
  end

  # ============================================================================
  # Encoder/Decoder Blocks
  # ============================================================================

  # Encoder downsampling block: residual convs + strided conv
  defp encoder_block(input, ch_in, ch_out, stride, name) do
    # Residual units before downsampling
    x = residual_unit(input, ch_in, "#{name}_res1")
    x = residual_unit(x, ch_in, "#{name}_res2")

    # Strided convolution for downsampling
    x = Axon.activation(x, :elu, name: "#{name}_act")

    Axon.conv(x, ch_out,
      kernel_size: {stride * 2},
      strides: [stride],
      padding: :same,
      channels: :first,
      name: "#{name}_downsample"
    )
  end

  # Decoder upsampling block: transposed conv + residual convs
  defp decoder_block(input, _ch_in, ch_out, stride, name) do
    x = Axon.activation(input, :elu, name: "#{name}_act")

    # Transposed convolution for upsampling
    x =
      Axon.conv_transpose(x, ch_out,
        kernel_size: {stride * 2},
        strides: [stride],
        padding: :same,
        channels: :first,
        name: "#{name}_upsample"
      )

    # Residual units after upsampling
    x = residual_unit(x, ch_out, "#{name}_res1")
    residual_unit(x, ch_out, "#{name}_res2")
  end

  # Residual unit with dilated convolutions
  defp residual_unit(input, channels, name) do
    # First conv with dilation 1
    x = Axon.activation(input, :elu, name: "#{name}_act1")

    x =
      Axon.conv(x, channels,
        kernel_size: {3},
        padding: :same,
        kernel_dilation: [1],
        channels: :first,
        name: "#{name}_conv1"
      )

    # Second conv with dilation 1
    x = Axon.activation(x, :elu, name: "#{name}_act2")

    x =
      Axon.conv(x, channels,
        kernel_size: {1},
        channels: :first,
        name: "#{name}_conv2"
      )

    # Residual connection
    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # RVQ Operations (Defn functions)
  # ============================================================================

  @doc """
  Initialize RVQ codebooks.

  ## Parameters

    - `num_codebooks` - Number of codebooks
    - `codebook_size` - Entries per codebook
    - `embedding_dim` - Dimension of each embedding
    - `key` - PRNG key (default: Nx.Random.key(42))

  ## Returns

    Codebooks tensor `[num_codebooks, codebook_size, embedding_dim]`.
  """
  @spec init_codebooks(pos_integer(), pos_integer(), pos_integer(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def init_codebooks(num_codebooks, codebook_size, embedding_dim, key \\ Nx.Random.key(42)) do
    {codebooks, _key} =
      Nx.Random.normal(key,
        shape: {num_codebooks, codebook_size, embedding_dim},
        type: :f32
      )

    # Scale initialization
    scale = :math.sqrt(embedding_dim)
    Nx.divide(codebooks, scale)
  end

  @doc """
  Quantize encoder output using Residual Vector Quantization.

  Each codebook quantizes the residual from the previous quantization,
  building up a coarse-to-fine representation.

  ## Parameters

    - `z_e` - Encoder output `[batch, T, dim]`
    - `codebooks` - RVQ codebooks `[num_codebooks, codebook_size, dim]`

  ## Returns

    `{z_q, indices}` where:
    - `z_q` - Quantized vectors `[batch, T, dim]` (sum of all codebook contributions)
    - `indices` - Token indices `[batch, num_codebooks, T]`
  """
  @spec rvq_quantize(Nx.Tensor.t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  defn rvq_quantize(z_e, codebooks) do
    {num_codebooks, _codebook_size, embedding_dim} = Nx.shape(codebooks)
    {batch, seq_len, _dim} = Nx.shape(z_e)

    # Initialize outputs
    z_q = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(z_e)), {batch, seq_len, embedding_dim})
    indices = Nx.broadcast(Nx.tensor(0, type: :s32), {batch, num_codebooks, seq_len})

    residual = z_e

    # Process each codebook (codebooks must be in the while tuple for defn scoping)
    {z_q, indices, _residual, _codebooks} =
      while {z_q, indices, residual, codebooks}, i <- 0..(num_codebooks - 1) do
        # Get current codebook [codebook_size, embedding_dim]
        cb = Nx.slice_along_axis(codebooks, i, 1, axis: 0) |> Nx.squeeze(axes: [0])

        # Quantize residual against this codebook
        {z_q_i, idx_i} = vq_quantize(residual, cb)

        # Accumulate quantized output
        z_q_new = Nx.add(z_q, z_q_i)

        # Update residual for next codebook
        residual_new = Nx.subtract(residual, z_q_i)

        # Store indices [batch, seq_len] -> place in [batch, i, seq_len]
        indices_new = Nx.put_slice(indices, [0, i, 0], Nx.new_axis(idx_i, 1))

        {z_q_new, indices_new, residual_new, codebooks}
      end

    # Straight-through estimator
    z_q_st = z_e + stop_grad(z_q - z_e)

    {z_q_st, indices}
  end

  @doc """
  Dequantize RVQ tokens back to continuous embeddings.

  ## Parameters

    - `indices` - Token indices `[batch, num_codebooks, T]`
    - `codebooks` - RVQ codebooks `[num_codebooks, codebook_size, dim]`

  ## Returns

    Continuous embeddings `[batch, T, dim]`.
  """
  @spec rvq_dequantize(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn rvq_dequantize(indices, codebooks) do
    {num_codebooks, _codebook_size, embedding_dim} = Nx.shape(codebooks)
    {batch, _num_cb, seq_len} = Nx.shape(indices)

    # Initialize output
    z_q = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, seq_len, embedding_dim})

    # Sum contributions from each codebook (codebooks must be in while tuple for defn scoping)
    {z_q, _indices, _codebooks} =
      while {z_q, indices, codebooks}, i <- 0..(num_codebooks - 1) do
        # Get codebook
        cb = Nx.slice_along_axis(codebooks, i, 1, axis: 0) |> Nx.squeeze(axes: [0])

        # Get indices for this codebook [batch, seq_len]
        idx = Nx.slice_along_axis(indices, i, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Gather embeddings [batch, seq_len, dim]
        z_q_i = gather_codebook(idx, cb)

        {Nx.add(z_q, z_q_i), indices, codebooks}
      end

    z_q
  end

  # Single codebook VQ (helper for RVQ)
  defnp vq_quantize(z, codebook) do
    # z: [batch, seq_len, dim]
    # codebook: [K, dim]
    {batch, seq_len, dim} = Nx.shape(z)
    {_k, _} = Nx.shape(codebook)

    # Flatten batch and seq for distance computation
    z_flat = Nx.reshape(z, {batch * seq_len, dim})

    # Compute distances: ||z - e||^2 = ||z||^2 - 2*z*e^T + ||e||^2
    z_sq = Nx.sum(Nx.pow(z_flat, 2), axes: [-1], keep_axes: true)
    cb_sq = Nx.sum(Nx.pow(codebook, 2), axes: [-1], keep_axes: true)
    cross = Nx.dot(z_flat, Nx.transpose(codebook))

    distances = z_sq - 2.0 * cross + Nx.transpose(cb_sq)

    # Find nearest entry
    indices_flat = Nx.argmin(distances, axis: -1)
    indices = Nx.reshape(indices_flat, {batch, seq_len})

    # Gather quantized vectors
    z_q_flat = Nx.take(codebook, indices_flat, axis: 0)
    z_q = Nx.reshape(z_q_flat, {batch, seq_len, dim})

    {z_q, Nx.as_type(indices, :s32)}
  end

  # Gather embeddings from codebook for given indices
  defnp gather_codebook(indices, codebook) do
    # indices: [batch, seq_len]
    # codebook: [K, dim]
    {batch, seq_len} = Nx.shape(indices)
    {_k, dim} = Nx.shape(codebook)

    indices_flat = Nx.reshape(indices, {batch * seq_len})
    embeddings_flat = Nx.take(codebook, indices_flat, axis: 0)
    Nx.reshape(embeddings_flat, {batch, seq_len, dim})
  end

  # ============================================================================
  # High-level Encode/Decode Functions
  # ============================================================================

  @doc """
  Encode waveform to discrete tokens.

  ## Parameters

    - `encoder_fn` - Compiled encoder prediction function
    - `params` - Encoder parameters
    - `codebooks` - RVQ codebooks
    - `waveform` - Input waveform `[batch, 1, samples]`

  ## Returns

    Token indices `[batch, num_codebooks, T]`.
  """
  @spec encode(function(), map(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def encode(encoder_fn, params, codebooks, waveform) do
    z_e = encoder_fn.(params, %{"waveform" => waveform})
    {_z_q, indices} = rvq_quantize(z_e, codebooks)
    indices
  end

  @doc """
  Decode discrete tokens back to waveform.

  ## Parameters

    - `decoder_fn` - Compiled decoder prediction function
    - `params` - Decoder parameters
    - `codebooks` - RVQ codebooks
    - `tokens` - Token indices `[batch, num_codebooks, T]`

  ## Returns

    Reconstructed waveform `[batch, 1, samples]`.
  """
  @spec decode(function(), map(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def decode(decoder_fn, params, codebooks, tokens) do
    z_q = rvq_dequantize(tokens, codebooks)
    decoder_fn.(params, %{"embeddings" => z_q})
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute spectral reconstruction loss (multi-scale STFT).

  Simplified version using single-scale MSE in time domain.
  Full EnCodec uses multi-scale spectral loss + adversarial loss.

  ## Parameters

    - `reconstruction` - Reconstructed waveform `[batch, 1, samples]`
    - `target` - Original waveform `[batch, 1, samples]`

  ## Returns

    Reconstruction loss scalar.
  """
  @spec spectral_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn spectral_loss(reconstruction, target) do
    # Time-domain L1 loss
    time_loss = Nx.mean(Nx.abs(reconstruction - target))

    # Simplified spectral loss (MSE proxy)
    spectral_loss = Nx.mean(Nx.pow(reconstruction - target, 2))

    time_loss + 0.1 * spectral_loss
  end

  @doc """
  Compute commitment loss for RVQ training.

  Encourages encoder outputs to stay close to codebook entries.

  ## Parameters

    - `z_e` - Encoder output `[batch, T, dim]`
    - `z_q` - Quantized output `[batch, T, dim]`

  ## Returns

    Commitment loss scalar.
  """
  @spec commitment_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn commitment_loss(z_e, z_q) do
    diff = z_e - stop_grad(z_q)
    Nx.mean(Nx.pow(diff, 2))
  end

  @doc """
  Combined EnCodec loss.

  ## Parameters

    - `reconstruction` - Reconstructed waveform
    - `target` - Original waveform
    - `z_e` - Encoder output
    - `z_q` - Quantized output
    - `commitment_weight` - Weight for commitment loss (default: 0.25)

  ## Returns

    Combined loss scalar.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def loss(reconstruction, target, z_e, z_q, opts \\ []) do
    commitment_weight = Keyword.get(opts, :commitment_weight, 0.25)
    do_loss(reconstruction, target, z_e, z_q, commitment_weight)
  end

  defnp do_loss(reconstruction, target, z_e, z_q, commitment_weight) do
    spec = spectral_loss(reconstruction, target)
    commit = commitment_loss(z_e, z_q)
    spec + commitment_weight * commit
  end

  @doc "Get the output embedding dimension."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    hidden_dim * 16
  end
end
