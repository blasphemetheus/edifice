defmodule Edifice.WorldModel.WorldModel do
  @moduledoc """
  World Model — learns a latent dynamics model of an environment.

  Encodes observations into a latent space, predicts next-state transitions
  given actions, and optionally decodes back to observation space. This is
  the core component for model-based RL and planning.

  ## Components

  - **Encoder:** `obs → z` — Maps raw observations to latent state
  - **Dynamics:** `(z, action) → next_z` — Predicts next latent state
  - **Reward head:** `z → scalar` — Predicts reward from latent state
  - **Decoder (optional):** `z → obs` — Reconstructs observations

  ## Dynamics Variants

  - `:mlp` — Standard two-layer MLP transition
  - `:neural_ode` — Shared-weight Euler integration (continuous dynamics)
  - `:gru` — Gated recurrent update (good for partially observable envs)

  ## Architecture

  ```
  obs [batch, obs_size]
        |
  +==============+
  |   Encoder    |  dense → GELU → dense
  +==============+
        |
  z [batch, latent_size]
        |
  +-----|-----+
  |     |     |
  v     v     v
  Dynamics  Reward  Decoder (optional)
  (z,a)→z'  z→r    z→obs
  ```

  ## Returns

  `{encoder, dynamics, reward_head}` or `{encoder, dynamics, reward_head, decoder}`
  when `use_decoder: true`.

  ## Usage

      {encoder, dynamics, reward_head} = WorldModel.build(
        obs_size: 64,
        action_size: 4,
        latent_size: 128,
        dynamics: :mlp
      )

      # With decoder for reconstruction loss
      {encoder, dynamics, reward_head, decoder} = WorldModel.build(
        obs_size: 64,
        action_size: 4,
        dynamics: :gru,
        use_decoder: true
      )

  ## References

  - Ha & Schmidhuber, "World Models" (2018)
  - Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer, 2020)
  - Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
  """

  @default_latent_size 128
  @default_hidden_size 256
  @default_num_ode_steps 4

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:obs_size, pos_integer()}
          | {:action_size, pos_integer()}
          | {:latent_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:dynamics, :mlp | :neural_ode | :gru}
          | {:use_decoder, boolean()}

  @doc """
  Build all world model components.

  ## Options

    - `:obs_size` - Observation dimension (required)
    - `:action_size` - Action dimension (required)
    - `:latent_size` - Latent state dimension (default: 128)
    - `:hidden_size` - Hidden layer size (default: 256)
    - `:dynamics` - Dynamics model type: `:mlp`, `:neural_ode`, or `:gru` (default: `:mlp`)
    - `:use_decoder` - Include observation decoder (default: false)

  ## Returns

    `{encoder, dynamics, reward_head}` or
    `{encoder, dynamics, reward_head, decoder}` if `use_decoder: true`.
  """
  @spec build([build_opt()]) ::
          {Axon.t(), Axon.t(), Axon.t()} | {Axon.t(), Axon.t(), Axon.t(), Axon.t()}
  def build(opts \\ []) do
    encoder = build_encoder(opts)
    dynamics = build_dynamics(opts)
    reward_head = build_reward_head(opts)

    if Keyword.get(opts, :use_decoder, false) do
      decoder = build_decoder(opts)
      {encoder, dynamics, reward_head, decoder}
    else
      {encoder, dynamics, reward_head}
    end
  end

  @doc """
  Build the observation encoder.

  Maps raw observations to latent state: `[batch, obs_size]` → `[batch, latent_size]`
  """
  @spec build_encoder(keyword()) :: Axon.t()
  def build_encoder(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)

    Axon.input("observation", shape: {nil, obs_size})
    |> Axon.dense(hidden_size, name: "wm_enc_dense1")
    |> Axon.activation(:gelu, name: "wm_enc_act1")
    |> Axon.dense(latent_size, name: "wm_enc_dense2")
    |> Axon.layer_norm(name: "wm_enc_norm")
  end

  @doc """
  Build the dynamics model.

  Predicts next latent state from current state and action:
  `[batch, latent_size + action_size]` → `[batch, latent_size]`

  ## Dynamics Variants

    - `:mlp` — Two dense layers with GELU
    - `:neural_ode` — Shared-weight Euler integration (4 steps)
    - `:gru` — Gated recurrent update
  """
  @spec build_dynamics(keyword()) :: Axon.t()
  def build_dynamics(opts \\ []) do
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    action_size = Keyword.fetch!(opts, :action_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dynamics_type = Keyword.get(opts, :dynamics, :mlp)
    concat_size = latent_size + action_size

    input = Axon.input("state_action", shape: {nil, concat_size})

    case dynamics_type do
      :mlp ->
        build_mlp_dynamics(input, hidden_size, latent_size)

      :neural_ode ->
        build_ode_dynamics(input, hidden_size, latent_size)

      :gru ->
        build_gru_dynamics(input, latent_size, hidden_size)
    end
  end

  @doc """
  Build the reward prediction head.

  Predicts scalar reward from latent state: `[batch, latent_size]` → `[batch]`
  """
  @spec build_reward_head(keyword()) :: Axon.t()
  def build_reward_head(opts \\ []) do
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)

    Axon.input("latent_state", shape: {nil, latent_size})
    |> Axon.dense(hidden_size, name: "wm_reward_dense1")
    |> Axon.activation(:gelu, name: "wm_reward_act")
    |> Axon.dense(1, name: "wm_reward_out")
    |> Axon.nx(fn x -> Nx.squeeze(x, axes: [1]) end, name: "wm_reward_squeeze")
  end

  @doc """
  Build the observation decoder.

  Reconstructs observations from latent state: `[batch, latent_size]` → `[batch, obs_size]`
  """
  @spec build_decoder(keyword()) :: Axon.t()
  def build_decoder(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)

    Axon.input("latent_state", shape: {nil, latent_size})
    |> Axon.dense(hidden_size, name: "wm_dec_dense1")
    |> Axon.activation(:gelu, name: "wm_dec_act")
    |> Axon.dense(obs_size, name: "wm_dec_out")
  end

  # ============================================================================
  # Dynamics Variants
  # ============================================================================

  # MLP dynamics: simple two-layer transition
  defp build_mlp_dynamics(input, hidden_size, latent_size) do
    input
    |> Axon.dense(hidden_size, name: "wm_dyn_dense1")
    |> Axon.activation(:gelu, name: "wm_dyn_act1")
    |> Axon.dense(latent_size, name: "wm_dyn_dense2")
    |> Axon.layer_norm(name: "wm_dyn_norm")
  end

  # Neural ODE dynamics: shared-weight Euler integration
  defp build_ode_dynamics(input, hidden_size, latent_size) do
    num_steps = @default_num_ode_steps
    step_size = 1.0 / num_steps

    # Project concat to latent dimension
    h0 = Axon.dense(input, latent_size, name: "wm_dyn_ode_proj")

    # Euler integration with shared weights
    h_final =
      Enum.reduce(0..(num_steps - 1), h0, fn step, h ->
        dh =
          h
          |> Axon.dense(hidden_size, name: "wm_dyn_ode_f1")
          |> Axon.activation(:silu, name: "wm_dyn_ode_f_act1")
          |> Axon.dense(latent_size, name: "wm_dyn_ode_f2")
          |> Axon.activation(:tanh, name: "wm_dyn_ode_f_act2")

        scaled_dh =
          Axon.nx(dh, fn x -> Nx.multiply(x, step_size) end, name: "wm_dyn_ode_scale_#{step}")

        Axon.add(h, scaled_dh, name: "wm_dyn_ode_step_#{step}")
      end)

    Axon.layer_norm(h_final, name: "wm_dyn_norm")
  end

  # GRU dynamics: gated update from concatenated (state, action)
  defp build_gru_dynamics(input, latent_size, hidden_size) do
    # Extract state from concatenated input
    z =
      Axon.nx(input, fn x -> Nx.slice_along_axis(x, 0, latent_size, axis: 1) end,
        name: "wm_dyn_gru_extract_z"
      )

    # Update gate
    update_gate =
      input
      |> Axon.dense(latent_size, name: "wm_dyn_gru_update")
      |> Axon.sigmoid(name: "wm_dyn_gru_update_sig")

    # Reset gate
    reset_gate =
      input
      |> Axon.dense(latent_size, name: "wm_dyn_gru_reset")
      |> Axon.sigmoid(name: "wm_dyn_gru_reset_sig")

    # Candidate: apply reset gate to state portion, then compute candidate
    reset_z =
      Axon.layer(
        fn z_val, reset_val, _opts -> Nx.multiply(z_val, reset_val) end,
        [z, reset_gate],
        name: "wm_dyn_gru_reset_z",
        op_name: :multiply
      )

    candidate =
      reset_z
      |> Axon.dense(hidden_size, name: "wm_dyn_gru_cand_dense1")
      |> Axon.activation(:gelu, name: "wm_dyn_gru_cand_act")
      |> Axon.dense(latent_size, name: "wm_dyn_gru_cand_dense2")
      |> Axon.tanh(name: "wm_dyn_gru_cand_tanh")

    # GRU update: z_new = (1 - update) * z + update * candidate
    Axon.layer(
      fn z_val, update_val, cand_val, _opts ->
        Nx.add(
          Nx.multiply(Nx.subtract(1.0, update_val), z_val),
          Nx.multiply(update_val, cand_val)
        )
      end,
      [z, update_gate, candidate],
      name: "wm_dyn_gru_output",
      op_name: :gru_update
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the latent size of the world model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :latent_size, @default_latent_size)
  end
end
