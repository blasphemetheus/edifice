defmodule Edifice.Generative.LatentDiffusion do
  @moduledoc """
  Latent Diffusion: Diffusion in VAE latent space.

  Implements the Latent Diffusion Model (LDM) concept from "High-Resolution
  Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022).
  Instead of diffusing in the full input space, LDM first compresses data
  with a VAE encoder, runs diffusion in the compact latent space, then
  decodes back.

  ## Key Innovation: Perceptual Compression + Diffusion

  ```
  Training:
    1. Train VAE: input -> encoder -> z -> decoder -> reconstruction
    2. Freeze VAE
    3. Train diffusion in latent space:
       z_0 = encoder(input)
       z_t = add_noise(z_0, t)
       eps_hat = denoiser(z_t, t)
       loss = MSE(eps, eps_hat)

  Inference:
    1. Sample z_T ~ N(0, I) in latent space
    2. Denoise: z_0 = diffusion_sample(z_T)
    3. Decode: output = decoder(z_0)
  ```

  ## Advantages

  | Feature | Full-space Diffusion | Latent Diffusion |
  |---------|---------------------|-----------------|
  | Compute | O(input_dim) per step | O(latent_dim) per step |
  | Quality | Good | Good (perceptual compression) |
  | Speed | Slow | Fast (smaller dim) |
  | Memory | High | Low |

  ## Architecture

  Returns a tuple of three models: `{encoder, decoder, denoiser}`

  ```
  Input [batch, input_size]
        |
        v
  +-------------------+
  | Encoder (frozen)  |  -> z_0 [batch, latent_size]
  +-------------------+
        |
        v (add noise)
  +-------------------+
  | Denoiser          |  -> eps_hat [batch, latent_size]
  | (z_t, t) -> eps   |
  +-------------------+
        |
        v (denoise)
  +-------------------+
  | Decoder (frozen)  |  -> output [batch, input_size]
  +-------------------+
  ```

  ## Usage

      {encoder, decoder, denoiser} = LatentDiffusion.build(
        input_size: 287,
        latent_size: 32,
        hidden_size: 256,
        num_layers: 4
      )

      # Train VAE first, then freeze and train denoiser

  ## Reference

  - Paper: "High-Resolution Image Synthesis with Latent Diffusion Models"
  - arXiv: https://arxiv.org/abs/2112.10752
  """

  require Axon
  import Nx.Defn

  @default_latent_size 32
  @default_hidden_size 256
  @default_num_layers 4
  @default_num_steps 1000
  @default_beta_start 1.0e-4
  @default_beta_end 0.02

  @doc """
  Build a Latent Diffusion Model.

  Returns `{encoder, decoder, denoiser}` where:
  - Encoder: maps input to latent distribution (mu, log_var)
  - Decoder: maps latent vector to reconstructed input
  - Denoiser: predicts noise in latent space given (noisy_z, timestep)

  ## Options

    - `:input_size` - Input feature dimension (required)
    - `:latent_size` - Latent space dimension (default: 32)
    - `:hidden_size` - Hidden dimension for all sub-networks (default: 256)
    - `:num_layers` - Number of layers in denoiser (default: 4)
    - `:num_steps` - Number of diffusion timesteps (default: 1000)

  ## Returns

    `{encoder, decoder, denoiser}` - Tuple of Axon models.
  """
  @spec build(keyword()) :: {Axon.t(), Axon.t(), Axon.t()}
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)

    encoder = build_encoder(input_size, latent_size, hidden_size)
    decoder = build_decoder(input_size, latent_size, hidden_size)
    denoiser = build_denoiser(latent_size, hidden_size, num_layers, num_steps)

    {encoder, decoder, denoiser}
  end

  @doc """
  Build the VAE encoder.

  Maps input to latent distribution parameters (mu, log_var).
  """
  @spec build_encoder(pos_integer(), pos_integer(), pos_integer()) :: Axon.t()
  def build_encoder(input_size, latent_size, hidden_size) do
    input = Axon.input("input", shape: {nil, input_size})

    trunk =
      input
      |> Axon.dense(hidden_size, name: "enc_dense_0")
      |> Axon.activation(:silu, name: "enc_act_0")
      |> Axon.dense(hidden_size, name: "enc_dense_1")
      |> Axon.activation(:silu, name: "enc_act_1")

    mu = Axon.dense(trunk, latent_size, name: "enc_mu")
    log_var = Axon.dense(trunk, latent_size, name: "enc_log_var")

    Axon.container(%{mu: mu, log_var: log_var})
  end

  @doc """
  Build the VAE decoder.

  Maps a latent vector back to input space.
  """
  @spec build_decoder(pos_integer(), pos_integer(), pos_integer()) :: Axon.t()
  def build_decoder(input_size, latent_size, hidden_size) do
    latent = Axon.input("latent", shape: {nil, latent_size})

    latent
    |> Axon.dense(hidden_size, name: "dec_dense_0")
    |> Axon.activation(:silu, name: "dec_act_0")
    |> Axon.dense(hidden_size, name: "dec_dense_1")
    |> Axon.activation(:silu, name: "dec_act_1")
    |> Axon.dense(input_size, name: "dec_output")
  end

  @doc """
  Build the latent-space denoiser.

  Predicts noise from (noisy_z, timestep).
  """
  @spec build_denoiser(pos_integer(), pos_integer(), pos_integer(), pos_integer()) :: Axon.t()
  def build_denoiser(latent_size, hidden_size, num_layers, num_steps) do
    noisy_z = Axon.input("noisy_z", shape: {nil, latent_size})
    timestep = Axon.input("timestep", shape: {nil})

    # Timestep embedding
    time_embed =
      Axon.layer(
        &sinusoidal_embed_impl/2,
        [timestep],
        name: "time_embed",
        hidden_size: hidden_size,
        num_steps: num_steps,
        op_name: :sinusoidal_embed
      )

    time_mlp =
      time_embed
      |> Axon.dense(hidden_size, name: "time_mlp_1")
      |> Axon.activation(:silu, name: "time_mlp_silu")

    # Combine noisy latent with time embedding
    z_proj = Axon.dense(noisy_z, hidden_size, name: "z_proj")
    combined = Axon.add(z_proj, time_mlp, name: "combine")

    # Denoiser residual blocks
    x =
      Enum.reduce(1..num_layers, combined, fn idx, acc ->
        build_residual_block(acc, hidden_size, "denoiser_block_#{idx}")
      end)

    # Output: predict noise in latent space
    Axon.dense(x, latent_size, name: "noise_pred")
  end

  defp build_residual_block(input, hidden_size, name) do
    x = Axon.layer_norm(input, name: "#{name}_norm")
    x = Axon.dense(x, hidden_size * 4, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_silu")
    x = Axon.dense(x, hidden_size, name: "#{name}_down")
    Axon.add(input, x, name: "#{name}_residual")
  end

  defp sinusoidal_embed_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    num_steps = opts[:num_steps]
    half_dim = div(hidden_size, 2)

    t_norm = Nx.divide(Nx.as_type(t, :f32), num_steps)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    t_expanded = Nx.new_axis(t_norm, 1)
    angles = Nx.multiply(t_expanded, Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # ============================================================================
  # VAE Operations
  # ============================================================================

  @doc """
  Reparameterization trick for the encoder.

  Requires a PRNG `key` for sampling. Returns `{z, new_key}`.
  """
  @spec reparameterize(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  defn reparameterize(mu, log_var, key) do
    std = Nx.exp(0.5 * log_var)
    {eps, key} = Nx.Random.normal(key, shape: Nx.shape(mu), type: Nx.type(mu))
    {mu + eps * std, key}
  end

  @doc """
  KL divergence for VAE training.
  """
  @spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn kl_divergence(mu, log_var) do
    kl = -0.5 * Nx.sum(1.0 + log_var - Nx.pow(mu, 2) - Nx.exp(log_var), axes: [-1])
    Nx.mean(kl)
  end

  # ============================================================================
  # Diffusion Schedule
  # ============================================================================

  @doc """
  Create diffusion noise schedule.
  """
  @spec make_schedule(keyword()) :: map()
  def make_schedule(opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)
    beta_start = Keyword.get(opts, :beta_start, @default_beta_start)
    beta_end = Keyword.get(opts, :beta_end, @default_beta_end)

    betas = Nx.linspace(beta_start, beta_end, n: num_steps, type: :f32)
    alphas = Nx.subtract(1.0, betas)

    log_alphas = Nx.log(Nx.add(alphas, 1.0e-10))
    log_alphas_cumprod = Nx.cumulative_sum(log_alphas)
    alphas_cumprod = Nx.exp(log_alphas_cumprod)

    %{
      num_steps: num_steps,
      alphas_cumprod: alphas_cumprod,
      sqrt_alphas_cumprod: Nx.sqrt(alphas_cumprod),
      sqrt_one_minus_alphas_cumprod: Nx.sqrt(Nx.subtract(1.0, alphas_cumprod))
    }
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size (latent dimension for the denoiser).
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :latent_size, @default_latent_size)
  end

  @doc """
  Calculate approximate parameter count for the full system.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    input_size = Keyword.get(opts, :input_size, 287)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    # Encoder: 2 dense + mu + log_var
    encoder = 2 * input_size * hidden_size + 2 * hidden_size * latent_size
    # Decoder: 2 dense + output
    decoder = latent_size * hidden_size + hidden_size * hidden_size + hidden_size * input_size
    # Denoiser: z_proj + time_mlp + blocks + output
    denoiser_base = latent_size * hidden_size + hidden_size * hidden_size
    denoiser_blocks = num_layers * (hidden_size * hidden_size * 4 + hidden_size * 4 * hidden_size)
    denoiser_out = hidden_size * latent_size

    encoder + decoder + denoiser_base + denoiser_blocks + denoiser_out
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      input_size: 287,
      latent_size: 32,
      hidden_size: 256,
      num_layers: 4,
      num_steps: 1000
    ]
  end
end
