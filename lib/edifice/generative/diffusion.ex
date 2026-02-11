defmodule Edifice.Generative.Diffusion do
  @moduledoc """
  Diffusion Policy: Action generation via denoising diffusion.

  Implements Diffusion Policy from "Diffusion Policy: Visuomotor Policy Learning
  via Action Diffusion" (Chi et al., RSS 2023). Instead of directly predicting
  actions, we learn to denoise random noise into actions conditioned on observations.

  ## Key Innovation: DDPM for Actions

  Traditional policies: `a = pi(o)` - direct mapping
  Diffusion Policy: `a = denoise(noise | o)` - iterative refinement

  ```
  Training:
    1. Sample action sequence a_0 from data
    2. Add noise: a_t = sqrt(alpha_bar_t)*a_0 + sqrt(1-alpha_bar_t)*eps
    3. Predict noise: eps_hat = network(a_t, t, obs)
    4. Loss: ||eps - eps_hat||^2

  Inference:
    1. Sample a_T ~ N(0, I)
    2. For t = T...1: a_{t-1} = denoise(a_t, t, obs)
    3. Return a_0
  ```

  ## Architecture

  ```
  Observations [batch, obs_dim]
        |
        v
  +-------------------------------------+
  |  Observation Encoder                 |
  |  (MLP or temporal backbone)          |
  +-------------------------------------+
        |
        v obs_embed
  +-------------------------------------+
  |  Denoising Network                   |
  |  Input: (noisy_actions, timestep,    |
  |          obs_embed)                  |
  |  Output: predicted_noise             |
  +-------------------------------------+
        |
        v
  Denoised Actions [batch, action_horizon, action_dim]
  ```

  ## Advantages

  | Feature | Benefit |
  |---------|---------|
  | Multi-modal | Can represent multiple valid actions |
  | High-dim | Scales well to action sequences |
  | Stable | MSE loss is simple and stable |
  | Expressive | Captures complex action distributions |

  ## Usage

      # Build diffusion policy
      model = Diffusion.build(
        obs_size: 287,
        action_dim: 64,
        action_horizon: 8,
        num_diffusion_steps: 100
      )

      # Training: predict noise
      {loss, predicted_noise} = Diffusion.training_step(
        model, params, observations, actions, key
      )

      # Inference: denoise to get actions
      actions = Diffusion.sample(model, params, observations, key)

  ## References
  - Paper: https://arxiv.org/abs/2303.04137
  - Project: https://diffusion-policy.cs.columbia.edu/
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default number of diffusion timesteps"
  def default_num_steps, do: 100

  @doc "Default action prediction horizon"
  def default_action_horizon, do: 8

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default number of denoiser layers"
  def default_num_layers, do: 4

  @doc "Beta schedule start"
  def default_beta_start, do: 1.0e-4

  @doc "Beta schedule end"
  def default_beta_end, do: 0.02

  # ============================================================================
  # Noise Schedule
  # ============================================================================

  @doc """
  Precompute diffusion schedule constants.

  Returns a map with:
  - `:betas` - Noise schedule beta_t
  - `:alphas` - 1 - beta_t
  - `:alphas_cumprod` - alpha_bar_t = product of alpha_s
  - `:sqrt_alphas_cumprod` - sqrt(alpha_bar_t)
  - `:sqrt_one_minus_alphas_cumprod` - sqrt(1-alpha_bar_t)
  - `:sqrt_recip_alphas` - 1/sqrt(alpha_t)
  - `:posterior_variance` - beta_tilde_t for sampling
  """
  @spec make_schedule(keyword()) :: map()
  def make_schedule(opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, default_num_steps())
    beta_start = Keyword.get(opts, :beta_start, default_beta_start())
    beta_end = Keyword.get(opts, :beta_end, default_beta_end())

    # Linear beta schedule
    betas = Nx.linspace(beta_start, beta_end, n: num_steps, type: :f32)

    alphas = Nx.subtract(1.0, betas)
    alphas_cumprod = cumprod(alphas)

    # Precompute useful quantities
    sqrt_alphas_cumprod = Nx.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = Nx.sqrt(Nx.subtract(1.0, alphas_cumprod))
    sqrt_recip_alphas = Nx.rsqrt(alphas)

    # Posterior variance for sampling: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    alphas_cumprod_prev =
      Nx.concatenate([Nx.tensor([1.0]), Nx.slice(alphas_cumprod, [0], [num_steps - 1])])

    posterior_variance =
      Nx.multiply(
        betas,
        Nx.divide(
          Nx.subtract(1.0, alphas_cumprod_prev),
          Nx.add(Nx.subtract(1.0, alphas_cumprod), 1.0e-8)
        )
      )

    %{
      num_steps: num_steps,
      betas: betas,
      alphas: alphas,
      alphas_cumprod: alphas_cumprod,
      alphas_cumprod_prev: alphas_cumprod_prev,
      sqrt_alphas_cumprod: sqrt_alphas_cumprod,
      sqrt_one_minus_alphas_cumprod: sqrt_one_minus_alphas_cumprod,
      sqrt_recip_alphas: sqrt_recip_alphas,
      posterior_variance: posterior_variance
    }
  end

  # Cumulative product
  defp cumprod(tensor) do
    # Use log-sum-exp for numerical stability
    log_tensor = Nx.log(Nx.add(tensor, 1.0e-10))
    log_cumprod = Nx.cumulative_sum(log_tensor)
    Nx.exp(log_cumprod)
  end

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Diffusion Policy model.

  ## Options
    - `:obs_size` - Size of observation embedding (required)
    - `:action_dim` - Dimension of action space (required)
    - `:action_horizon` - Number of actions to predict (default: 8)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of denoiser layers (default: 4)
    - `:num_steps` - Number of diffusion timesteps (default: 100)

  ## Returns
    An Axon model that predicts noise given (noisy_actions, timestep, obs).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    num_steps = Keyword.get(opts, :num_steps, default_num_steps())

    # Inputs
    noisy_actions = Axon.input("noisy_actions", shape: {nil, action_horizon, action_dim})
    timestep = Axon.input("timestep", shape: {nil})
    observations = Axon.input("observations", shape: {nil, obs_size})

    # Build denoising network
    build_denoiser(noisy_actions, timestep, observations,
      action_dim: action_dim,
      action_horizon: action_horizon,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_steps: num_steps
    )
  end

  @doc """
  Build the observation encoder for temporal inputs.

  Processes sequence of observations into a single embedding.
  """
  @spec build_obs_encoder(keyword()) :: Axon.t()
  def build_obs_encoder(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    window_size = Keyword.get(opts, :window_size, 60)

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, window_size, embed_size})

    # Simple temporal pooling encoder
    # Could be replaced with a temporal backbone for better modeling
    input
    |> Axon.dense(hidden_size, name: "obs_proj")
    |> Axon.activation(:silu, name: "obs_silu")
    |> Axon.nx(
      fn x ->
        # Mean pool over time
        Nx.mean(x, axes: [1])
      end,
      name: "obs_pool"
    )
    |> Axon.dense(hidden_size, name: "obs_out")
  end

  # ============================================================================
  # Denoising Network
  # ============================================================================

  @doc """
  Build the denoising network (noise predictor).

  Architecture: MLP with sinusoidal timestep embedding and observation conditioning.
  """
  @spec build_denoiser(Axon.t(), Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def build_denoiser(noisy_actions, timestep, observations, opts) do
    action_dim = Keyword.fetch!(opts, :action_dim)
    action_horizon = Keyword.fetch!(opts, :action_horizon)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    num_steps = Keyword.get(opts, :num_steps, default_num_steps())

    # Flatten noisy actions: [batch, horizon, dim] -> [batch, horizon * dim]
    actions_flat =
      Axon.nx(
        noisy_actions,
        fn x ->
          batch = Nx.axis_size(x, 0)
          Nx.reshape(x, {batch, action_horizon * action_dim})
        end,
        name: "flatten_actions"
      )

    # Sinusoidal timestep embedding
    time_embed = build_timestep_embedding(timestep, hidden_size, num_steps)

    # Observation projection
    obs_embed = Axon.dense(observations, hidden_size, name: "obs_embed")

    # Concatenate all conditioning
    # [batch, horizon*dim + hidden + hidden]
    combined =
      Axon.concatenate([actions_flat, time_embed, obs_embed], axis: 1, name: "combine_inputs")

    # MLP denoiser layers
    x = Axon.dense(combined, hidden_size, name: "denoiser_in")
    x = Axon.activation(x, :silu, name: "denoiser_in_silu")

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_denoiser_block(acc, hidden_size, "denoiser_block_#{layer_idx}")
      end)

    # Output: predict noise with same shape as actions
    noise_flat = Axon.dense(x, action_horizon * action_dim, name: "noise_out")

    # Reshape back: [batch, horizon * dim] -> [batch, horizon, dim]
    Axon.nx(
      noise_flat,
      fn x ->
        batch = Nx.axis_size(x, 0)
        Nx.reshape(x, {batch, action_horizon, action_dim})
      end,
      name: "reshape_noise"
    )
  end

  # Single denoiser block with residual connection
  defp build_denoiser_block(input, hidden_size, name) do
    x = Axon.layer_norm(input, name: "#{name}_norm")
    x = Axon.dense(x, hidden_size * 4, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_silu")
    x = Axon.dense(x, hidden_size, name: "#{name}_down")
    Axon.add(input, x, name: "#{name}_residual")
  end

  # Sinusoidal timestep embedding (like Transformer positional encoding)
  defp build_timestep_embedding(timestep, hidden_size, num_steps) do
    Axon.layer(
      &sinusoidal_embedding/2,
      [timestep],
      name: "time_embed",
      hidden_size: hidden_size,
      num_steps: num_steps,
      op_name: :sinusoidal_embed
    )
  end

  defp sinusoidal_embedding(timestep, opts) do
    hidden_size = opts[:hidden_size]
    num_steps = opts[:num_steps]

    # timestep: [batch] with values in [0, num_steps)
    # Normalize to [0, 1]
    t_normalized = Nx.divide(Nx.as_type(timestep, :f32), num_steps)

    # Create frequency bands
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.pow(
        10_000.0,
        Nx.divide(
          Nx.iota({half_dim}, type: :f32),
          half_dim - 1
        )
      )

    # [batch, 1] * [half_dim] -> [batch, half_dim]
    t_expanded = Nx.new_axis(t_normalized, 1)
    angles = Nx.multiply(t_expanded, Nx.reshape(freqs, {1, half_dim}))

    # Concatenate sin and cos
    sin_embed = Nx.sin(angles)
    cos_embed = Nx.cos(angles)
    Nx.concatenate([sin_embed, cos_embed], axis: 1)
  end

  # ============================================================================
  # Diffusion Operations
  # ============================================================================

  @doc """
  Forward diffusion: add noise to actions.

  ```
  a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1-alpha_bar_t) * eps
  ```
  """
  @spec q_sample(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  defn q_sample(actions, timestep, noise, schedule) do
    # actions: [batch, horizon, dim]
    # timestep: [batch]
    # noise: [batch, horizon, dim]

    sqrt_alphas_cumprod = schedule.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod

    # Extract schedule values for this timestep
    # [batch] -> [batch, 1, 1] for broadcasting
    batch_size = Nx.axis_size(actions, 0)

    sqrt_alpha = Nx.gather(sqrt_alphas_cumprod, Nx.reshape(timestep, {batch_size, 1}))
    sqrt_alpha = Nx.reshape(sqrt_alpha, {batch_size, 1, 1})

    sqrt_one_minus_alpha =
      Nx.gather(sqrt_one_minus_alphas_cumprod, Nx.reshape(timestep, {batch_size, 1}))

    sqrt_one_minus_alpha = Nx.reshape(sqrt_one_minus_alpha, {batch_size, 1, 1})

    # a_t = sqrt(alpha_bar_t) * a_0 + sqrt(1-alpha_bar_t) * eps
    Nx.add(
      Nx.multiply(sqrt_alpha, actions),
      Nx.multiply(sqrt_one_minus_alpha, noise)
    )
  end

  @doc """
  Compute training loss: MSE between true and predicted noise.
  """
  @spec compute_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn compute_loss(true_noise, predicted_noise) do
    # MSE loss
    diff = Nx.subtract(true_noise, predicted_noise)
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Single denoising step (reverse process).

  ```
  a_{t-1} = (1/sqrt(alpha_t)) * (a_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_hat) + sqrt(beta_tilde_t) * z
  ```
  """
  @spec p_sample(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) ::
          Nx.Tensor.t()
  defn p_sample(noisy_actions, predicted_noise, timestep, random_noise, schedule) do
    # noisy_actions: [batch, horizon, dim]
    # predicted_noise: [batch, horizon, dim]
    # timestep: [batch]
    # random_noise: [batch, horizon, dim]

    batch_size = Nx.axis_size(noisy_actions, 0)

    # Extract schedule values
    betas = schedule.betas
    sqrt_recip_alphas = schedule.sqrt_recip_alphas
    sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod
    posterior_variance = schedule.posterior_variance

    # Gather values for current timestep
    beta_t = gather_and_expand(betas, timestep, batch_size)
    sqrt_recip_alpha_t = gather_and_expand(sqrt_recip_alphas, timestep, batch_size)

    sqrt_one_minus_alpha_t =
      gather_and_expand(sqrt_one_minus_alphas_cumprod, timestep, batch_size)

    posterior_var_t = gather_and_expand(posterior_variance, timestep, batch_size)

    # Compute mean: mu = (1/sqrt(alpha_t)) * (a_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_hat)
    coef = Nx.divide(beta_t, sqrt_one_minus_alpha_t)

    mean =
      Nx.multiply(
        sqrt_recip_alpha_t,
        Nx.subtract(noisy_actions, Nx.multiply(coef, predicted_noise))
      )

    # Add noise (except at t=0)
    # Check if t > 0, if so add noise
    t_is_zero = Nx.equal(timestep, 0)
    # Broadcast to match action shape [batch, horizon, dim]
    t_is_zero_broadcast =
      Nx.broadcast(Nx.reshape(t_is_zero, {batch_size, 1, 1}), Nx.shape(noisy_actions))

    noise_scale = Nx.sqrt(posterior_var_t)
    noise_term = Nx.multiply(noise_scale, random_noise)

    # Return mean if t=0, else mean + noise
    Nx.select(t_is_zero_broadcast, mean, Nx.add(mean, noise_term))
  end

  defnp gather_and_expand(tensor, indices, batch_size) do
    gathered = Nx.gather(tensor, Nx.reshape(indices, {batch_size, 1}))
    Nx.reshape(gathered, {batch_size, 1, 1})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Diffusion Policy model.

  Returns action_horizon * action_dim.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    action_horizon * action_dim
  end

  @doc """
  Calculate approximate parameter count for a Diffusion Policy model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    obs_size = Keyword.get(opts, :obs_size, 287)
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())

    action_flat = action_horizon * action_dim
    # actions + time + obs
    input_size = action_flat + hidden_size + hidden_size

    # Input projection
    input_proj = input_size * hidden_size

    # Denoiser blocks: each has up (h*4h) + down (4h*h)
    block_params = num_layers * (hidden_size * hidden_size * 4 + hidden_size * 4 * hidden_size)

    # Output projection
    output_proj = hidden_size * action_flat

    # Observation embedding
    obs_proj = obs_size * hidden_size

    input_proj + block_params + output_proj + obs_proj
  end

  @doc """
  Get recommended defaults for action generation.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      action_dim: 64,
      action_horizon: 8,
      hidden_size: 256,
      num_layers: 4,
      num_steps: 100,
      beta_start: 1.0e-4,
      beta_end: 0.02
    ]
  end

  @doc """
  Fast inference configuration with fewer diffusion steps.
  """
  @spec fast_inference_defaults() :: keyword()
  def fast_inference_defaults do
    [
      action_dim: 64,
      action_horizon: 4,
      hidden_size: 128,
      num_layers: 2,
      # Fewer steps for faster inference
      num_steps: 20,
      beta_start: 1.0e-4,
      beta_end: 0.02
    ]
  end
end
