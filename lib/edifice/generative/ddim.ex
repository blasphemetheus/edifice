defmodule Edifice.Generative.DDIM do
  @moduledoc """
  DDIM: Denoising Diffusion Implicit Models.

  Implements DDIM from "Denoising Diffusion Implicit Models" (Song et al.,
  ICLR 2021). Uses the same training objective as DDPM but enables
  deterministic sampling with far fewer steps via a non-Markovian
  reverse process.

  ## Key Innovation: Deterministic Sampling

  DDPM requires ~1000 steps because each step adds stochastic noise.
  DDIM reformulates the reverse process as deterministic:

  ```
  DDPM (stochastic, ~1000 steps):
    x_{t-1} = mu(x_t, eps_theta) + sigma_t * z,  z ~ N(0, I)

  DDIM (deterministic, ~50 steps):
    x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * pred_dir
    where:
      pred_x0 = (x_t - sqrt(1-alpha_t) * eps_theta) / sqrt(alpha_t)
      pred_dir = sqrt(1-alpha_{t-1}) * eps_theta
  ```

  The eta parameter interpolates between deterministic (eta=0) and
  stochastic DDPM (eta=1).

  ## Architecture

  Uses the same denoising network as DDPM (noise predictor conditioned
  on timestep and observations). The difference is in sampling, not training.

  ```
  Same training as DDPM:
    1. Sample x_0, noise, timestep
    2. Compute noisy x_t
    3. Predict noise: eps_theta(x_t, t, obs)
    4. Loss: MSE(eps, eps_theta)

  DDIM sampling (fewer steps):
    1. Choose stride S (e.g., skip every 20 steps)
    2. For t in [1000, 980, 960, ..., 20, 0]:
       x_{t-S} = ddim_step(x_t, t, eps_theta)
    3. Return x_0
  ```

  ## Usage

      # Build denoising network (same as DDPM)
      model = DDIM.build(
        obs_size: 287,
        action_dim: 64,
        action_horizon: 8,
        num_steps: 1000
      )

      # DDIM sampling with fewer steps
      schedule = DDIM.make_schedule(num_steps: 1000)
      actions = DDIM.ddim_sample(params, predict_fn, obs, noise,
        schedule: schedule,
        ddim_steps: 50,
        eta: 0.0
      )

  ## Reference

  - Paper: "Denoising Diffusion Implicit Models"
  - arXiv: https://arxiv.org/abs/2010.02502
  """

  require Axon
  import Nx.Defn

  @default_hidden_size 256
  @default_num_layers 4
  @default_action_horizon 8
  @default_num_steps 1000
  @default_beta_start 1.0e-4
  @default_beta_end 0.02

  @doc """
  Build a DDIM denoising network.

  The network architecture is identical to DDPM -- the difference is in
  the sampling procedure, not the model.

  ## Options

    - `:obs_size` - Size of observation embedding (required)
    - `:action_dim` - Dimension of action space (required)
    - `:action_horizon` - Number of actions to predict (default: 8)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of denoiser layers (default: 4)
    - `:num_steps` - Number of diffusion timesteps (default: 1000)

  ## Returns

    An Axon model that predicts noise given (noisy_actions, timestep, obs).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    action_horizon = Keyword.get(opts, :action_horizon, @default_action_horizon)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)

    # Inputs
    noisy_actions = Axon.input("noisy_actions", shape: {nil, action_horizon, action_dim})
    timestep = Axon.input("timestep", shape: {nil})
    observations = Axon.input("observations", shape: {nil, obs_size})

    # Flatten actions
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

    # Observation embedding
    obs_embed = Axon.dense(observations, hidden_size, name: "obs_embed")

    # Combine inputs
    combined =
      Axon.concatenate([actions_flat, time_embed, obs_embed], axis: 1, name: "combine_inputs")

    # Denoiser MLP
    x = Axon.dense(combined, hidden_size, name: "denoiser_in")
    x = Axon.activation(x, :silu, name: "denoiser_in_silu")

    x =
      Enum.reduce(1..num_layers, x, fn idx, acc ->
        build_denoiser_block(acc, hidden_size, "denoiser_block_#{idx}")
      end)

    # Output: predicted noise
    noise_flat = Axon.dense(x, action_horizon * action_dim, name: "noise_out")

    Axon.nx(
      noise_flat,
      fn x ->
        batch = Nx.axis_size(x, 0)
        Nx.reshape(x, {batch, action_horizon, action_dim})
      end,
      name: "reshape_noise"
    )
  end

  defp build_denoiser_block(input, hidden_size, name) do
    x = Axon.layer_norm(input, name: "#{name}_norm")
    x = Axon.dense(x, hidden_size * 4, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_silu")
    x = Axon.dense(x, hidden_size, name: "#{name}_down")
    Axon.add(input, x, name: "#{name}_residual")
  end

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

  defp sinusoidal_embedding(t, opts) do
    hidden_size = opts[:hidden_size]
    num_steps = opts[:num_steps]
    half_dim = div(hidden_size, 2)

    t_norm = Nx.divide(Nx.as_type(t, :f32), num_steps)
    freqs = Nx.pow(10_000.0, Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1)))
    t_expanded = Nx.new_axis(t_norm, 1)
    angles = Nx.multiply(t_expanded, Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # ============================================================================
  # Noise Schedule
  # ============================================================================

  @doc """
  Precompute diffusion schedule (same as DDPM).
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
      betas: betas,
      alphas: alphas,
      alphas_cumprod: alphas_cumprod,
      sqrt_alphas_cumprod: Nx.sqrt(alphas_cumprod),
      sqrt_one_minus_alphas_cumprod: Nx.sqrt(Nx.subtract(1.0, alphas_cumprod))
    }
  end

  # ============================================================================
  # DDIM Sampling
  # ============================================================================

  @doc """
  DDIM sampling: deterministic reverse process with stride.

  ## Parameters

    - `params` - Model parameters
    - `predict_fn` - Noise prediction function
    - `observations` - Conditioning [batch, obs_size]
    - `initial_noise` - Starting noise [batch, action_horizon, action_dim]
    - `opts`:
      - `:schedule` - Precomputed schedule from `make_schedule/1`
      - `:ddim_steps` - Number of DDIM steps (default: 50)
      - `:eta` - Stochasticity: 0.0 = deterministic, 1.0 = DDPM (default: 0.0)

  ## Returns

    Denoised actions [batch, action_horizon, action_dim].
  """
  @spec ddim_sample(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) ::
          Nx.Tensor.t()
  def ddim_sample(params, predict_fn, observations, initial_noise, opts \\ []) do
    schedule = Keyword.fetch!(opts, :schedule)
    ddim_steps = Keyword.get(opts, :ddim_steps, 50)
    eta = Keyword.get(opts, :eta, 0.0)

    num_steps = schedule.num_steps
    stride = max(div(num_steps, ddim_steps), 1)

    # Build timestep sequence: [num_steps-1, num_steps-1-stride, ..., 0]
    timesteps =
      Stream.iterate(num_steps - 1, &(&1 - stride))
      |> Enum.take_while(&(&1 >= 0))

    Enum.reduce(timesteps, initial_noise, fn t, x_t ->
      batch_size = Nx.axis_size(x_t, 0)
      t_tensor = Nx.broadcast(Nx.tensor(t, type: :s64), {batch_size})
      t_prev = max(t - stride, 0)

      # Predict noise
      predicted_noise =
        predict_fn.(params, %{
          "noisy_actions" => x_t,
          "timestep" => t_tensor,
          "observations" => observations
        })

      ddim_step(x_t, predicted_noise, t, t_prev, schedule, eta)
    end)
  end

  @doc """
  Single DDIM reverse step.

  ```
  pred_x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
  direction = sqrt(1-alpha_{t-1} - sigma^2) * eps
  x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + direction + sigma * noise
  ```
  """
  @spec ddim_step(Nx.Tensor.t(), Nx.Tensor.t(), integer(), integer(), map(), float()) ::
          Nx.Tensor.t()
  defn ddim_step(x_t, predicted_noise, t, t_prev, schedule, eta) do
    alphas_cumprod = schedule.alphas_cumprod

    alpha_t = alphas_cumprod[t]
    alpha_prev = alphas_cumprod[t_prev]

    # Predict clean sample
    pred_x0 =
      Nx.divide(
        Nx.subtract(x_t, Nx.multiply(Nx.sqrt(1.0 - alpha_t), predicted_noise)),
        Nx.sqrt(alpha_t)
      )

    # Clip for stability
    pred_x0 = Nx.clip(pred_x0, -1.0, 1.0)

    # Compute sigma for stochasticity
    sigma =
      eta *
        Nx.sqrt(
          Nx.divide(
            Nx.multiply(1.0 - alpha_prev, 1.0 - alpha_t / alpha_prev),
            Nx.add(1.0 - alpha_t, 1.0e-8)
          )
        )

    # Direction pointing to x_t
    dir_xt =
      Nx.multiply(
        Nx.sqrt(Nx.max(1.0 - alpha_prev - sigma * sigma, 0.0)),
        predicted_noise
      )

    # x_{t-1}
    Nx.add(
      Nx.multiply(Nx.sqrt(alpha_prev), pred_x0),
      dir_xt
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a DDIM model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, @default_action_horizon)
    action_horizon * action_dim
  end

  @doc """
  Calculate approximate parameter count.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    obs_size = Keyword.get(opts, :obs_size, 287)
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, @default_action_horizon)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    action_flat = action_horizon * action_dim
    input_size = action_flat + hidden_size + hidden_size
    input_proj = input_size * hidden_size
    block_params = num_layers * (hidden_size * hidden_size * 4 + hidden_size * 4 * hidden_size)
    output_proj = hidden_size * action_flat
    obs_proj = obs_size * hidden_size

    input_proj + block_params + output_proj + obs_proj
  end

  @doc """
  Get recommended defaults for fast DDIM sampling.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      action_dim: 64,
      action_horizon: 8,
      hidden_size: 256,
      num_layers: 4,
      num_steps: 1000,
      ddim_steps: 50,
      eta: 0.0
    ]
  end
end
