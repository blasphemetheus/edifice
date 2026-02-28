defmodule Edifice.Robotics.DiffusionPolicy do
  @moduledoc """
  Diffusion Policy — Visuomotor Policy Learning via Action Diffusion.

  <!-- verified: true, date: 2026-02-28 -->

  Generates action chunks by iterative denoising via DDPM. A noise prediction
  network takes noisy actions, a diffusion timestep, and observation features,
  then predicts the noise to remove. Predicting entire action *chunks* (Tp
  future timesteps) instead of single actions reduces compounding error in
  imitation learning.

  ## Architecture (U-Net 1D variant)

  ```
  Observations [batch, To, obs_dim]   Noisy Actions [batch, Tp, action_dim]
        |                                      |
  Flatten -> [batch, To * obs_dim]       Transpose -> [batch, action_dim, Tp]
        |                                      |
  (global conditioning)                  ConditionalUnet1D:
        |                                      |
  Timestep -> SinPosEmb -> MLP            Down path:
  -> [batch, embed_dim]                     ResBlock + FiLM(cond) x2 + Downsample
        |                                    (collect skip connections)
  cat([timestep_emb, obs_flat])              |
  -> [batch, cond_dim]  ─────────────>  Mid: ResBlock + FiLM x2
                                             |
                                        Up path:
                                          cat(skip) + ResBlock + FiLM x2 + Upsample
                                             |
                                        Final Conv -> [batch, action_dim, Tp]
                                             |
                                        Transpose -> [batch, Tp, action_dim]
  ```

  ## Horizons

  | Symbol | Default | Meaning |
  |--------|---------|---------|
  | To     | 2       | Observation steps (past frames) |
  | Tp     | 16      | Prediction horizon (chunk size) |
  | Ta     | 8       | Executed actions before re-plan |

  ## Usage

      model = DiffusionPolicy.build(
        action_dim: 7,
        obs_dim: 20,
        prediction_horizon: 16,
        observation_horizon: 2
      )

      # Inputs: "noisy_actions", "timestep", "observations"
      # Output: noise prediction [batch, Tp, action_dim]

      # Noise schedule for training/sampling
      schedule = DiffusionPolicy.make_cosine_schedule(num_steps: 100)

  ## References

  - Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action
    Diffusion" (RSS 2023)
  - https://arxiv.org/abs/2303.04137
  """

  import Nx.Defn
  alias Edifice.Blocks.SinusoidalPE

  @default_prediction_horizon 16
  @default_observation_horizon 2
  # Note: action_horizon (default 8) is a runtime parameter controlling how many
  # predicted actions to execute before re-planning. Not used in model building.
  @default_down_dims [256, 512, 1024]
  @default_kernel_size 5
  @default_n_groups 8
  @default_diffusion_step_embed_dim 256
  @default_num_train_timesteps 100

  @doc """
  Build a Diffusion Policy denoising network (ConditionalUnet1D).

  ## Options

    - `:action_dim` - Action space dimension (required)
    - `:obs_dim` - Observation dimension (required)
    - `:prediction_horizon` - Predicted action steps (default: 16)
    - `:observation_horizon` - Past observation steps (default: 2)
    - `:action_horizon` - Executed actions per re-plan (default: 8)
    - `:down_dims` - U-Net channel progression (default: [256, 512, 1024])
    - `:kernel_size` - Conv kernel size (default: 5)
    - `:n_groups` - GroupNorm groups (default: 8)
    - `:diffusion_step_embed_dim` - Timestep embedding dim (default: 256)
    - `:num_train_timesteps` - Diffusion steps (default: 100)

  ## Inputs

    - "noisy_actions": `[batch, Tp, action_dim]`
    - "timestep": `[batch]`
    - "observations": `[batch, To, obs_dim]`

  ## Returns

    Noise prediction `[batch, Tp, action_dim]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:action_dim, pos_integer()}
          | {:action_horizon, pos_integer()}
          | {:diffusion_step_embed_dim, pos_integer()}
          | {:down_dims, [pos_integer()]}
          | {:kernel_size, pos_integer()}
          | {:n_groups, pos_integer()}
          | {:num_train_timesteps, pos_integer()}
          | {:obs_dim, pos_integer()}
          | {:observation_horizon, pos_integer()}
          | {:prediction_horizon, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    action_dim = Keyword.fetch!(opts, :action_dim)
    obs_dim = Keyword.fetch!(opts, :obs_dim)
    tp = Keyword.get(opts, :prediction_horizon, @default_prediction_horizon)
    to = Keyword.get(opts, :observation_horizon, @default_observation_horizon)
    down_dims = Keyword.get(opts, :down_dims, @default_down_dims)
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)
    n_groups = Keyword.get(opts, :n_groups, @default_n_groups)
    embed_dim = Keyword.get(opts, :diffusion_step_embed_dim, @default_diffusion_step_embed_dim)
    num_steps = Keyword.get(opts, :num_train_timesteps, @default_num_train_timesteps)

    # Inputs
    noisy_actions = Axon.input("noisy_actions", shape: {nil, tp, action_dim})
    timestep = Axon.input("timestep", shape: {nil})
    observations = Axon.input("observations", shape: {nil, to, obs_dim})

    # Flatten observations for global conditioning
    obs_flat =
      Axon.nx(
        observations,
        fn t ->
          {batch, _to, _dim} = Nx.shape(t)
          Nx.reshape(t, {batch, :auto})
        end, name: "flatten_obs")

    # Timestep embedding: SinPosEmb -> MLP
    time_embed =
      SinusoidalPE.timestep_layer(timestep,
        hidden_size: embed_dim,
        num_steps: num_steps,
        name: "time_embed"
      )

    time_mlp =
      time_embed
      |> Axon.dense(embed_dim * 4, name: "time_mlp_1")
      |> Axon.activation(:silu, name: "time_mlp_silu")
      |> Axon.dense(embed_dim, name: "time_mlp_2")

    # Global conditioning: cat([timestep_emb, obs_flat])
    global_cond =
      Axon.layer(
        fn time_t, obs_t, _opts ->
          Nx.concatenate([time_t, obs_t], axis: -1)
        end,
        [time_mlp, obs_flat],
        name: "global_cond",
        op_name: :concatenate
      )

    cond_dim = embed_dim + to * obs_dim

    # Keep actions as [B, Tp, action_dim] (channels-last for Axon conv)
    x = noisy_actions

    # Build U-Net 1D
    all_dims = [action_dim | down_dims]

    # Down path: collect skip connections
    {x, skips} =
      all_dims
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.reduce({x, []}, fn {[dim_in, dim_out], idx}, {acc, skip_list} ->
        is_last = idx == length(down_dims) - 1

        # Two residual blocks with FiLM conditioning
        acc =
          cond_res_block(
            acc,
            global_cond,
            dim_in,
            dim_out,
            cond_dim,
            kernel_size,
            n_groups,
            "down_#{idx}_r0"
          )

        acc =
          cond_res_block(
            acc,
            global_cond,
            dim_out,
            dim_out,
            cond_dim,
            kernel_size,
            n_groups,
            "down_#{idx}_r1"
          )

        # Save skip connection
        skip_list = skip_list ++ [acc]

        # Downsample (except last level)
        acc =
          if is_last do
            acc
          else
            Axon.conv(acc, dim_out,
              kernel_size: 3,
              strides: [2],
              padding: :same,
              name: "down_#{idx}_ds"
            )
          end

        {acc, skip_list}
      end)

    # Mid blocks
    last_dim = List.last(down_dims)

    x =
      cond_res_block(
        x,
        global_cond,
        last_dim,
        last_dim,
        cond_dim,
        kernel_size,
        n_groups,
        "mid_r0"
      )

    x =
      cond_res_block(
        x,
        global_cond,
        last_dim,
        last_dim,
        cond_dim,
        kernel_size,
        n_groups,
        "mid_r1"
      )

    # Up path: mirror the down path in reverse
    # Down had N levels (indices 0..N-1), each with a skip.
    # Levels 0..N-2 had downsamples. Up path needs N-1 levels with upsamples matching.
    # Pair each skip (reversed) with the output dim to project to.
    num_down = length(down_dims)
    skips_rev = Enum.reverse(skips)

    # Build up dimension sequence: reversed down_dims pairs
    # e.g. down_dims=[16,32] -> up goes from 32 back to 16
    up_out_dims = Enum.reverse(down_dims)

    x =
      skips_rev
      |> Enum.with_index()
      |> Enum.reduce(x, fn {skip, idx}, acc ->
        # Current channel count of acc and skip
        skip_dim = Enum.at(up_out_dims, idx)
        out_dim = Enum.at(up_out_dims, min(idx + 1, num_down - 1))
        # For the last up level, output first_dim (= down_dims[0])
        out_dim = if idx == num_down - 1, do: List.first(down_dims), else: out_dim

        # Concatenate skip connection along channels (last axis)
        acc =
          Axon.layer(
            fn feat, sk, _opts -> Nx.concatenate([feat, sk], axis: -1) end,
            [acc, skip],
            name: "up_#{idx}_cat",
            op_name: :concatenate
          )

        # Input channels = current + skip (both should be skip_dim from the corresponding down level)
        in_ch = skip_dim * 2

        acc =
          cond_res_block(
            acc,
            global_cond,
            in_ch,
            out_dim,
            cond_dim,
            kernel_size,
            n_groups,
            "up_#{idx}_r0"
          )

        acc =
          cond_res_block(
            acc,
            global_cond,
            out_dim,
            out_dim,
            cond_dim,
            kernel_size,
            n_groups,
            "up_#{idx}_r1"
          )

        # Upsample to match the spatial resolution of the next skip connection.
        # Down path had downsamples at levels 0..N-2. Reversed: up idx 0..N-2 need upsample.
        if idx < num_down - 1 do
          upsample_1d(acc, out_dim, "up_#{idx}_us")
        else
          acc
        end
      end)

    # Final conv
    first_dim = List.first(down_dims)

    x =
      x
      |> conv1d_block(first_dim, first_dim, kernel_size, n_groups, "final_conv")
      |> Axon.conv(action_dim, kernel_size: 1, padding: :valid, name: "final_proj")

    # Output is already [B, Tp, action_dim] (channels-last)
    x
  end

  @doc "Get the output size of the policy."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.fetch!(opts, :action_dim)
    tp = Keyword.get(opts, :prediction_horizon, @default_prediction_horizon)
    action_dim * tp
  end

  # ===========================================================================
  # Conditional Residual Block with FiLM
  # ===========================================================================

  defp cond_res_block(x, cond, in_ch, out_ch, _cond_dim, kernel_size, n_groups, name) do
    # First conv block
    h = conv1d_block(x, in_ch, out_ch, kernel_size, n_groups, "#{name}_c0")

    # FiLM conditioning: cond -> scale, bias
    film =
      cond
      |> Axon.activation(:silu, name: "#{name}_film_silu")
      |> Axon.dense(out_ch * 2, name: "#{name}_film_proj")

    # Apply FiLM: scale * h + bias
    h =
      Axon.layer(
        &film_modulate/3,
        [h, film],
        name: "#{name}_film",
        out_ch: out_ch,
        op_name: :film
      )

    # Second conv block
    h = conv1d_block(h, out_ch, out_ch, kernel_size, n_groups, "#{name}_c1")

    # Residual connection (project if dimensions differ)
    residual =
      if in_ch == out_ch do
        x
      else
        Axon.conv(x, out_ch, kernel_size: 1, padding: :valid, name: "#{name}_skip")
      end

    Axon.add(h, residual, name: "#{name}_res")
  end

  # FiLM: split cond into scale and bias, apply channel-wise
  # h: [batch, time, out_ch] (channels-last), film: [batch, out_ch * 2]
  defp film_modulate(h, film, opts) do
    out_ch = opts[:out_ch]
    {batch, _time, _ch} = Nx.shape(h)

    # film: [batch, out_ch * 2] -> scale [batch, 1, out_ch], bias [batch, 1, out_ch]
    scale = Nx.slice_along_axis(film, 0, out_ch, axis: -1)
    bias = Nx.slice_along_axis(film, out_ch, out_ch, axis: -1)

    scale = Nx.reshape(scale, {batch, 1, out_ch})
    bias = Nx.reshape(bias, {batch, 1, out_ch})

    Nx.add(Nx.multiply(Nx.add(scale, 1.0), h), bias)
  end

  # ===========================================================================
  # Conv1d Block: Conv -> GroupNorm -> SiLU
  # ===========================================================================

  defp conv1d_block(x, _in_ch, out_ch, kernel_size, n_groups, name) do
    groups = valid_groups(out_ch, n_groups)

    x
    |> Axon.conv(out_ch, kernel_size: kernel_size, padding: :same, name: "#{name}_conv")
    |> Axon.group_norm(groups, name: "#{name}_gn")
    |> Axon.activation(:silu, name: "#{name}_act")
  end

  # Find largest valid group count <= target that divides channels
  defp valid_groups(channels, target) do
    target = min(target, channels)
    if rem(channels, target) == 0, do: target, else: valid_groups(channels, target - 1)
  end

  # ===========================================================================
  # Upsample 1D: nearest-neighbor interpolation + conv
  # ===========================================================================

  # Channels-last upsample: [B, T, C] -> [B, T*2, C]
  defp upsample_1d(x, dim, name) do
    Axon.layer(
      fn t, _opts ->
        {batch, time, channels} = Nx.shape(t)
        # Nearest-neighbor: repeat each time step
        t
        |> Nx.new_axis(2)
        |> Nx.tile([1, 1, 2, 1])
        |> Nx.reshape({batch, time * 2, channels})
      end,
      [x],
      name: "#{name}_nn",
      op_name: :upsample
    )
    |> Axon.conv(dim, kernel_size: 3, padding: :same, name: "#{name}_conv")
  end

  # ===========================================================================
  # Noise Schedule
  # ===========================================================================

  @doc """
  Create cosine noise schedule (squaredcos_cap_v2 from iDDPM).

  ## Options

    - `:num_steps` - Diffusion timesteps (default: 100)
    - `:s` - Cosine offset for numerical stability (default: 0.008)

  ## Returns

    Map with `:alphas_cumprod`, `:sqrt_alphas_cumprod`, `:sqrt_one_minus_alphas_cumprod`.
  """
  @spec make_cosine_schedule(keyword()) :: map()
  def make_cosine_schedule(opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, @default_num_train_timesteps)
    s = Keyword.get(opts, :s, 0.008)

    # alphas_cumprod = cos((t/T + s)/(1+s) * pi/2)^2
    steps = num_steps + 1
    t = Nx.linspace(0, num_steps, n: steps, type: :f32)
    ratio = Nx.divide(Nx.add(Nx.divide(t, num_steps), s), 1.0 + s)
    alphas_cumprod_full = Nx.pow(Nx.cos(Nx.multiply(ratio, :math.pi() / 2.0)), 2)

    # Normalize so alpha_0 = 1
    alpha_0 = Nx.slice_along_axis(alphas_cumprod_full, 0, 1, axis: 0)
    alphas_cumprod_full = Nx.divide(alphas_cumprod_full, alpha_0)

    # Compute betas from consecutive alphas_cumprod
    ac_prev = Nx.slice_along_axis(alphas_cumprod_full, 0, num_steps, axis: 0)
    ac_next = Nx.slice_along_axis(alphas_cumprod_full, 1, num_steps, axis: 0)
    betas = Nx.subtract(1.0, Nx.divide(ac_next, Nx.max(ac_prev, Nx.tensor(1.0e-8))))
    betas = Nx.clip(betas, 0.0, 0.999)

    alphas = Nx.subtract(1.0, betas)
    log_alphas = Nx.log(Nx.add(alphas, 1.0e-10))
    log_alphas_cumprod = Nx.cumulative_sum(log_alphas)
    alphas_cumprod = Nx.exp(log_alphas_cumprod)

    %{
      num_steps: num_steps,
      betas: betas,
      alphas_cumprod: alphas_cumprod,
      sqrt_alphas_cumprod: Nx.sqrt(alphas_cumprod),
      sqrt_one_minus_alphas_cumprod: Nx.sqrt(Nx.subtract(1.0, alphas_cumprod))
    }
  end

  @doc """
  Add noise to actions at timestep k.

  ```
  A^k = sqrt(alpha_bar_k) * A^0 + sqrt(1 - alpha_bar_k) * eps
  ```
  """
  @spec add_noise(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  defn add_noise(actions, noise, timesteps, schedule) do
    sqrt_alpha = Nx.take(schedule.sqrt_alphas_cumprod, timesteps)
    sqrt_one_minus = Nx.take(schedule.sqrt_one_minus_alphas_cumprod, timesteps)

    # Broadcast: [batch] -> [batch, 1, 1]
    sqrt_alpha = Nx.new_axis(Nx.new_axis(sqrt_alpha, -1), -1)
    sqrt_one_minus = Nx.new_axis(Nx.new_axis(sqrt_one_minus, -1), -1)

    sqrt_alpha * actions + sqrt_one_minus * noise
  end

  @doc """
  Compute training loss (MSE between predicted and actual noise).
  """
  @spec compute_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn compute_loss(predicted_noise, actual_noise) do
    diff = predicted_noise - actual_noise
    Nx.mean(diff * diff)
  end
end
