defmodule Edifice.Generative.JanusFlow do
  @moduledoc """
  JanusFlow: Unified AR text + Rectified Flow image generation.

  Implements the velocity prediction network from JanusFlow (DeepSeek AI,
  November 2024). Combines a ShallowUViT encoder/decoder with a transformer
  backbone to predict flow velocity vectors in continuous VAE latent space.

  ## Key Innovations

  - **Unified multimodal**: Single LLM backbone handles both understanding
    (AR text) and generation (rectified flow images)
  - **Continuous generation**: Rectified flow replaces discrete VQ-VAE tokens,
    enabling higher-fidelity image synthesis
  - **ShallowUViT**: Lightweight ConvNeXt-based encoder/decoder with FiLM
    conditioning from timestep embeddings

  ## Architecture (Velocity Network)

  ```
  z_t [batch, latent_ch, H, W]  +  timestep [batch]  +  text_embed [batch, T, hidden]
        |                              |                        |
  [ShallowUViTEncoder]           TimestepEmbedding              |
    Patchify Conv2d                   |                         |
    ConvNeXt blocks (FiLM)      t_emb [batch, hidden]           |
        |                              |                         |
    flatten to [B, HW/4, dim]          |                         |
    project to hidden                  |                         |
        |                              |                         |
  +---------cat([text, t_emb, z_seq])---------+
  |                                            |
  |        Transformer Layers                  |
  |        (causal self-attention + FFN)       |
  |                                            |
  +--------------------------------------------+
        |
    Extract last HW/4 positions
    Project to decoder dim
    Reshape to [B, dim, H/2, W/2]
        |
  [ShallowUViTDecoder]
    Skip concat + ConvNeXt blocks (FiLM)
    PixelShuffle unpatchify
        |
  velocity [batch, latent_ch, H, W]
  ```

  ## Usage

      # Build velocity network
      {encoder, decoder} = JanusFlow.build(
        latent_channels: 4,
        latent_size: 16,
        encoder_dim: 64,
        hidden_size: 128,
        num_heads: 4,
        num_layers: 4,
        text_seq_len: 8,
        num_convnext_blocks: 2
      )

  ## References

  - Paper: "JanusFlow: Harmonizing Autoregression and Rectified Flow"
  - Authors: DeepSeek AI (November 2024)
  - ArXiv: 2411.07975
  """

  alias Edifice.Blocks.{FFN, SinusoidalPE}

  @default_latent_channels 4
  @default_latent_size 16
  @default_encoder_dim 64
  @default_hidden_size 128
  @default_num_heads 4
  @default_num_layers 4
  @default_text_seq_len 8
  @default_num_convnext_blocks 2
  @default_dropout 0.1
  @default_patch_size 2

  @doc """
  Build a JanusFlow velocity prediction network.

  Returns `{encoder_model, decoder_model}` where encoder takes noisy latent +
  timestep + text conditioning and decoder outputs velocity in latent space.

  For simplicity, this builds a single unified model that takes all inputs
  and returns velocity. Returns `{model, nil}` for registry compatibility.

  ## Options

    - `:latent_channels` - VAE latent channels (default: 4)
    - `:latent_size` - Spatial size of latent (default: 16)
    - `:encoder_dim` - ShallowUViT encoder/decoder channel dim (default: 64)
    - `:hidden_size` - Transformer hidden dimension (default: 128)
    - `:num_heads` - Transformer attention heads (default: 4)
    - `:num_layers` - Transformer layers (default: 4)
    - `:text_seq_len` - Expected text conditioning sequence length (default: 8)
    - `:num_convnext_blocks` - ConvNeXt blocks in encoder/decoder (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:patch_size` - Patchify stride/kernel (default: 2)

  ## Returns

    A `{velocity_model, nil}` tuple. The velocity model takes three inputs:
    `"z_t"`, `"timestep"`, and `"text_embed"`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:encoder_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:latent_channels, pos_integer()}
          | {:latent_size, pos_integer()}
          | {:num_convnext_blocks, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:text_seq_len, pos_integer()}

  @spec build([build_opt()]) :: {Axon.t(), nil}
  def build(opts \\ []) do
    latent_ch = Keyword.get(opts, :latent_channels, @default_latent_channels)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    enc_dim = Keyword.get(opts, :encoder_dim, @default_encoder_dim)
    hidden = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    text_seq_len = Keyword.get(opts, :text_seq_len, @default_text_seq_len)
    n_cnx = Keyword.get(opts, :num_convnext_blocks, @default_num_convnext_blocks)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)

    spatial = div(latent_size, patch_size)
    img_seq_len = spatial * spatial

    # --- Inputs ---
    # z_t: noisy latent in channels-last format [batch, H, W, latent_ch]
    z_input = Axon.input("z_t", shape: {nil, latent_size, latent_size, latent_ch})

    # timestep: [batch] — scalar per sample
    t_input = Axon.input("timestep", shape: {nil})

    # text_embed: [batch, text_seq_len, hidden]
    text_input = Axon.input("text_embed", shape: {nil, text_seq_len, hidden})

    # --- Timestep Embedding ---
    # Sinusoidal -> MLP (like diffusion models)
    t_emb = SinusoidalPE.timestep_layer(t_input, hidden_size: enc_dim, name: "time_sinusoidal")

    t_emb =
      t_emb
      |> Axon.dense(hidden, name: "time_mlp_1")
      |> Axon.activation(:silu, name: "time_silu")
      |> Axon.dense(hidden, name: "time_mlp_2")

    # --- ShallowUViT Encoder ---
    # Patchify: Conv2d with stride=patch_size
    z_enc =
      Axon.conv(z_input, enc_dim,
        kernel_size: patch_size,
        strides: patch_size,
        padding: :valid,
        name: "enc_patchify"
      )

    # ConvNeXt blocks with FiLM from t_emb
    # t_emb for FiLM needs to project to enc_dim*2 per block
    {z_enc, skip} =
      Enum.reduce(1..n_cnx, {z_enc, nil}, fn i, {x, _skip} ->
        x_out = convnext_2d_block(x, t_emb, enc_dim, hidden, "enc_cnx_#{i}")
        # Save first block output as skip connection for decoder
        skip_out = if i == 1, do: x_out, else: nil
        {x_out, skip_out || x}
      end)

    # Save encoder output as skip connection
    enc_skip = skip || z_enc

    # Flatten spatial to sequence: [batch, H', W', dim] -> [batch, H'*W', dim]
    z_seq =
      Axon.nx(
        z_enc,
        fn t ->
          {b, h, w, d} = Nx.shape(t)
          Nx.reshape(t, {b, h * w, d})
        end, name: "enc_flatten")

    # Project to transformer hidden size
    z_seq = Axon.dense(z_seq, hidden, name: "enc_proj")

    # --- Build Transformer Input ---
    # t_emb as single token: [batch, hidden] -> [batch, 1, hidden]
    t_token = Axon.nx(t_emb, fn t -> Nx.new_axis(t, 1) end, name: "t_token")

    # Concatenate: [text_embed, t_token, z_seq]
    # text: [B, text_seq_len, hidden]
    # t_token: [B, 1, hidden]
    # z_seq: [B, img_seq_len, hidden]
    combined = Axon.concatenate([text_input, t_token, z_seq], axis: 1, name: "seq_cat")

    # --- Transformer Backbone ---
    x = combined

    x =
      Enum.reduce(1..num_layers, x, fn i, acc ->
        build_transformer_layer(acc, hidden, num_heads, dropout, "tf_#{i}")
      end)

    # --- Extract Image Hidden States ---
    # Last img_seq_len positions
    img_hidden =
      Axon.nx(
        x,
        fn t ->
          start = Nx.axis_size(t, 1) - img_seq_len
          Nx.slice_along_axis(t, start, img_seq_len, axis: 1)
        end, name: "extract_img")

    # Project to decoder dimension
    img_hidden =
      img_hidden
      |> Axon.layer_norm(name: "dec_align_norm")
      |> Axon.dense(enc_dim, name: "dec_aligner")

    # Reshape to spatial: [batch, img_seq, dim] -> [batch, H', W', dim]
    img_spatial =
      Axon.nx(
        img_hidden,
        fn t ->
          {b, _s, d} = Nx.shape(t)
          Nx.reshape(t, {b, spatial, spatial, d})
        end, name: "dec_reshape")

    # --- ShallowUViT Decoder ---
    # Concat with encoder skip connection
    dec_in = Axon.concatenate([img_spatial, enc_skip], axis: 3, name: "dec_skip_cat")

    # ConvNeXt blocks on concatenated input
    dec_dim = enc_dim * 2

    x_dec =
      Enum.reduce(1..n_cnx, dec_in, fn i, acc ->
        convnext_2d_block(acc, t_emb, dec_dim, hidden, "dec_cnx_#{i}")
      end)

    # Unpatchify: Conv2d(dec_dim, latent_ch * patch_size^2) + PixelShuffle
    # First apply RMSNorm equivalent (layer_norm on channels-last)
    velocity =
      x_dec
      |> Axon.layer_norm(name: "dec_out_norm")
      |> Axon.conv(latent_ch * patch_size * patch_size,
        kernel_size: 1,
        padding: :valid,
        name: "dec_out_conv"
      )

    # PixelShuffle: [batch, H', W', C*r^2] -> [batch, H'*r, W'*r, C]
    velocity =
      Axon.nx(
        velocity,
        fn t ->
          pixel_shuffle(t, patch_size, latent_ch)
        end, name: "pixel_shuffle")

    {velocity, nil}
  end

  # ============================================================================
  # ConvNeXt V2 Block (2D) with FiLM conditioning
  # ============================================================================

  defp convnext_2d_block(x, t_emb, dim, _hidden_size, name) do
    intermediate = dim * 4

    h =
      x
      # Depthwise Conv2d (kernel=7, groups=dim)
      |> Axon.conv(dim,
        kernel_size: 7,
        padding: :same,
        feature_group_size: dim,
        name: "#{name}_dw_conv"
      )
      # LayerNorm on channels-last
      |> Axon.layer_norm(name: "#{name}_ln")
      # Pointwise expansion
      |> Axon.dense(intermediate, name: "#{name}_pw1")
      |> Axon.activation(:gelu, name: "#{name}_gelu")
      # Global Response Normalization
      |> grn_2d_layer(intermediate, "#{name}_grn")
      # Pointwise projection
      |> Axon.dense(dim, name: "#{name}_pw2")

    # Residual
    h = Axon.add(x, h, name: "#{name}_res")

    # FiLM conditioning from timestep
    film_proj =
      t_emb
      |> Axon.activation(:silu, name: "#{name}_film_silu")
      |> Axon.dense(dim * 2, name: "#{name}_film_proj")

    Axon.layer(
      &film_2d_fn/3,
      [h, film_proj],
      name: "#{name}_film",
      dim: dim,
      op_name: :film
    )
  end

  # FiLM: h * (1 + scale) + shift
  # h: [batch, H, W, dim], film: [batch, dim*2]
  defp film_2d_fn(h, film, opts) do
    dim = opts[:dim]
    shape = Nx.shape(h)
    batch = elem(shape, 0)

    scale = Nx.slice_along_axis(film, 0, dim, axis: -1)
    shift = Nx.slice_along_axis(film, dim, dim, axis: -1)

    # Reshape for spatial broadcasting: [batch, dim] -> [batch, 1, 1, dim]
    scale = Nx.reshape(scale, {batch, 1, 1, dim})
    shift = Nx.reshape(shift, {batch, 1, 1, dim})

    Nx.add(Nx.multiply(Nx.add(scale, 1.0), h), shift)
  end

  # ============================================================================
  # Global Response Normalization (2D spatial)
  # ============================================================================

  # GRN for 2D spatial: normalize by L2 norm along H,W dimensions
  # x: [batch, H, W, dim]
  defp grn_2d_layer(x, dim, name) do
    Axon.layer(
      fn t, gamma, beta, _opts ->
        # t: [batch, H, W, dim]
        # L2 norm along spatial dims (1, 2)
        gx = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(t, 2), axes: [1, 2], keep_axes: true), 1.0e-6))
        # Normalize: gx / mean(gx, channel_dim)
        nx_val = Nx.divide(gx, Nx.add(Nx.mean(gx, axes: [-1], keep_axes: true), 1.0e-6))
        Nx.add(Nx.add(Nx.multiply(gamma, Nx.multiply(t, nx_val)), beta), t)
      end,
      [x, Axon.param("gamma", {1, 1, 1, dim}), Axon.param("beta", {1, 1, 1, dim})],
      name: name,
      op_name: :grn_2d
    )
  end

  # ============================================================================
  # PixelShuffle: channels-last [B, H, W, C*r^2] -> [B, H*r, W*r, C]
  # ============================================================================

  defp pixel_shuffle(tensor, upscale_factor, out_channels) do
    {batch, h, w, _c} = Nx.shape(tensor)
    r = upscale_factor

    # Reshape: [B, H, W, C*r*r] -> [B, H, W, r, r, C]
    tensor = Nx.reshape(tensor, {batch, h, w, r, r, out_channels})

    # Transpose to interleave: [B, H, r, W, r, C]
    tensor = Nx.transpose(tensor, axes: [0, 1, 3, 2, 4, 5])

    # Reshape to upscaled: [B, H*r, W*r, C]
    Nx.reshape(tensor, {batch, h * r, w * r, out_channels})
  end

  # ============================================================================
  # Transformer Layer
  # ============================================================================

  defp build_transformer_layer(x, hidden_size, num_heads, dropout, name) do
    head_dim = div(hidden_size, num_heads)

    # Pre-norm attention
    attn_normed = Axon.layer_norm(x, name: "#{name}_attn_norm")

    q = Axon.dense(attn_normed, hidden_size, name: "#{name}_q")
    k = Axon.dense(attn_normed, hidden_size, name: "#{name}_k")
    v = Axon.dense(attn_normed, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &causal_attention_fn/4,
        [q, k, v],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim
      )

    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_attn_proj")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(x, attn_out, name: "#{name}_attn_res")

    # Pre-norm FFN
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, name: "#{name}_ffn")
    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_res")
  end

  defp causal_attention_fn(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))
    q = Nx.multiply(q, scale)

    scores = Nx.dot(q, [3], [0, 1], Nx.transpose(k, axes: [0, 1, 3, 2]), [2], [0, 1])

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.select(Nx.greater_equal(rows, cols), Nx.tensor(0.0), Nx.tensor(-1.0e9))
    mask = Nx.reshape(mask, {1, 1, seq_len, seq_len})
    scores = Nx.add(scores, mask)

    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [3], keep_axes: true)))
    weights = Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [3], keep_axes: true), 1.0e-6))

    out = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3])
    Nx.reshape(out, {batch, seq_len, num_heads * head_dim})
  end

  defp maybe_dropout(x, rate, name) when rate > 0, do: Axon.dropout(x, rate: rate, name: name)
  defp maybe_dropout(x, _rate, _name), do: x

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a JanusFlow model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
