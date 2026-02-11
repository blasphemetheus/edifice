defmodule Edifice.Generative.ConsistencyModel do
  @moduledoc """
  Consistency Model: Single-step generation via consistency function.

  Implements the Consistency Model from "Consistency Models" (Song et al.,
  ICML 2023). Learns a function f(x_t, t) that maps any point on a
  probability flow ODE trajectory directly to its origin, enabling
  single-step generation without iterative denoising.

  ## Key Innovation: Self-Consistency Property

  A consistency function f satisfies: for all t, t' on the same trajectory,
  f(x_t, t) = f(x_{t'}, t'). In particular, f(x_t, t) = x_0 for all t.

  ```
  Diffusion (many steps):
    x_T -> x_{T-1} -> ... -> x_1 -> x_0    (N denoising steps)

  Consistency Model (one step):
    x_T -> x_0    (single forward pass!)

  Or few-step refinement:
    x_T -> x_0' -> add_noise(x_0', t') -> x_0''    (2 steps, better quality)
  ```

  ## Training Approaches

  1. **Consistency Distillation (CD)**: Distill from a pre-trained diffusion model
  2. **Consistency Training (CT)**: Train from scratch without a teacher

  Both enforce: f(x_{t+1}, t+1) = f(x_t, t) for adjacent timesteps.

  ## Architecture

  ```
  Input (x_t, sigma_t)
        |
        v
  +-----------------------+
  | Skip Connection       |
  | c_skip(t) * x_t +    |
  | c_out(t) * F(x_t, t) |
  +-----------------------+
        |
        v
  Output: predicted x_0
  ```

  The skip connection ensures the boundary condition f(x, sigma_min) = x.

  ## Usage

      model = ConsistencyModel.build(
        input_dim: 64,
        hidden_size: 256,
        num_layers: 4
      )

      # Single-step generation
      x_0 = ConsistencyModel.single_step_sample(model, params, noise)

      # Multi-step refinement
      x_0 = ConsistencyModel.multi_step_sample(model, params, noise, steps: 3)

  ## Reference

  - Paper: "Consistency Models"
  - arXiv: https://arxiv.org/abs/2303.01469
  """

  require Axon
  import Nx.Defn

  @default_hidden_size 256
  @default_num_layers 4
  @default_sigma_min 0.002
  @default_sigma_max 80.0

  @doc """
  Build a Consistency Model.

  The model learns f(x, sigma) that maps noisy input at any noise level
  to the clean data, while maintaining self-consistency across the trajectory.

  ## Options

    - `:input_dim` - Input feature dimension (required)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of residual blocks (default: 4)
    - `:sigma_min` - Minimum noise level (default: 0.002)
    - `:sigma_max` - Maximum noise level (default: 80.0)

  ## Returns

    An Axon model: (noisy_input, sigma) -> predicted clean input.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    sigma_min = Keyword.get(opts, :sigma_min, @default_sigma_min)
    sigma_max = Keyword.get(opts, :sigma_max, @default_sigma_max)

    # Inputs
    noisy_input = Axon.input("noisy_input", shape: {nil, input_dim})
    sigma = Axon.input("sigma", shape: {nil})

    # The network F(x, sigma) before skip connection
    f_net = build_f_network(noisy_input, sigma, input_dim, hidden_size, num_layers)

    # Skip connection for boundary condition: f(x, sigma_min) = x
    # f(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(x, sigma)
    Axon.layer(
      &consistency_skip_impl/4,
      [noisy_input, f_net, sigma],
      name: "consistency_output",
      sigma_min: sigma_min,
      sigma_max: sigma_max,
      op_name: :consistency_skip
    )
  end

  defp build_f_network(noisy_input, sigma, input_dim, hidden_size, num_layers) do
    # Sigma embedding (log-scale sinusoidal)
    sigma_embed =
      Axon.layer(
        &sigma_embedding_impl/2,
        [sigma],
        name: "sigma_embed",
        hidden_size: hidden_size,
        op_name: :sigma_embed
      )

    sigma_mlp =
      sigma_embed
      |> Axon.dense(hidden_size, name: "sigma_mlp_1")
      |> Axon.activation(:silu, name: "sigma_mlp_silu")

    # Project input
    x_proj = Axon.dense(noisy_input, hidden_size, name: "input_proj")

    # Combine
    combined = Axon.add(x_proj, sigma_mlp, name: "combine")

    # Residual blocks
    x =
      Enum.reduce(1..num_layers, combined, fn idx, acc ->
        build_residual_block(acc, hidden_size, "block_#{idx}")
      end)

    # Output projection
    Axon.dense(x, input_dim, name: "f_output")
  end

  defp build_residual_block(input, hidden_size, name) do
    x = Axon.layer_norm(input, name: "#{name}_norm")
    x = Axon.dense(x, hidden_size * 4, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_silu")
    x = Axon.dense(x, hidden_size, name: "#{name}_down")
    Axon.add(input, x, name: "#{name}_residual")
  end

  defp sigma_embedding_impl(sigma, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    # Log-scale embedding
    log_sigma = Nx.log(Nx.add(Nx.as_type(sigma, :f32), 1.0e-10))

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), max(half_dim - 1, 1))
        )
      )

    s_expanded = Nx.new_axis(log_sigma, 1)
    angles = Nx.multiply(s_expanded, Nx.reshape(freqs, {1, half_dim}))
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
  end

  # Consistency skip connection (Song et al., 2023, Appendix A)
  # c_skip(sigma) = sigma_data^2 / ((sigma - sigma_min)^2 + sigma_data^2)
  # c_out(sigma) = sigma_data * (sigma - sigma_min) / sqrt(sigma^2 + sigma_data^2)
  # Boundary: at sigma = sigma_min, c_skip = 1 and c_out = 0, so f(x, sigma_min) = x
  defp consistency_skip_impl(x, f_out, sigma, opts) do
    sigma_min = opts[:sigma_min]
    _sigma_max = opts[:sigma_max]

    sigma_data = 0.5
    sigma_2d = Nx.new_axis(sigma, 1)

    # (sigma - sigma_min) term — drives boundary condition
    sigma_diff = Nx.subtract(sigma_2d, sigma_min)
    sigma_diff_sq = Nx.multiply(sigma_diff, sigma_diff)
    sigma_sq = Nx.multiply(sigma_2d, sigma_2d)
    data_sq = sigma_data * sigma_data

    # c_skip = sigma_data^2 / ((sigma - sigma_min)^2 + sigma_data^2)
    # At sigma_min: numerator = data_sq, denominator = 0 + data_sq -> c_skip = 1
    c_skip = Nx.divide(data_sq, Nx.add(sigma_diff_sq, data_sq))

    # c_out = sigma_data * (sigma - sigma_min) / sqrt(sigma^2 + sigma_data^2)
    # At sigma_min: numerator = 0 -> c_out = 0
    c_out =
      Nx.divide(
        Nx.multiply(sigma_data, sigma_diff),
        Nx.sqrt(Nx.add(sigma_sq, data_sq))
      )

    Nx.add(Nx.multiply(c_skip, x), Nx.multiply(c_out, f_out))
  end

  # ============================================================================
  # Training Loss
  # ============================================================================

  @doc """
  Consistency training loss using pseudo-Huber function.

  Enforces f(x_{t+1}, t+1) = f(x_t, t) for adjacent timesteps.
  Uses a target network (EMA of the online network) for stability.

  The pseudo-Huber loss (Song et al., 2023) is: sqrt(||d||^2 + c^2) - c
  where d = f_theta(x_{t+dt}, t+dt) - f_target(x_t, t) and c = 0.00054*sqrt(d_dim).

  ## Parameters

    - `pred_current` - f_theta(x_{t+dt}, t+dt) from online network
    - `pred_target` - f_target(x_t, t) from target (EMA) network

  ## Returns

    Scalar loss.
  """
  @spec consistency_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn consistency_loss(pred_current, pred_target) do
    diff = Nx.subtract(pred_current, pred_target)
    d_sq = Nx.sum(Nx.multiply(diff, diff), axes: [-1])
    # Pseudo-Huber with c = 0.00054 * sqrt(dim), per Improved Consistency Training
    dim = Nx.axis_size(pred_current, Nx.rank(pred_current) - 1)
    c = 0.00054 * Nx.sqrt(dim)
    c_sq = Nx.multiply(c, c)
    Nx.mean(Nx.subtract(Nx.sqrt(Nx.add(d_sq, c_sq)), c))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Consistency Model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_dim, 64)
  end

  @doc """
  Calculate approximate parameter count.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    input_dim = Keyword.get(opts, :input_dim, 64)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    input_proj = input_dim * hidden_size
    sigma_mlp = hidden_size * hidden_size
    blocks = num_layers * (hidden_size * hidden_size * 4 + hidden_size * 4 * hidden_size)
    output_proj = hidden_size * input_dim

    input_proj + sigma_mlp + blocks + output_proj
  end

  @doc """
  Generate noise schedule using Karras et al. discretization.

  Produces N timesteps: t_i = (eps^{1/rho} + (i-1)/(N-1) * (T^{1/rho} - eps^{1/rho}))^rho
  where eps = sigma_min, T = sigma_max, and rho = 7 (default).

  ## Options
    - `:n_steps` - Number of timesteps (default: 40)
    - `:sigma_min` - Minimum sigma (default: 0.002)
    - `:sigma_max` - Maximum sigma (default: 80.0)
    - `:rho` - Schedule curvature (default: 7)
  """
  @spec noise_schedule(keyword()) :: Nx.Tensor.t()
  def noise_schedule(opts \\ []) do
    n = Keyword.get(opts, :n_steps, 40)
    sigma_min = Keyword.get(opts, :sigma_min, @default_sigma_min)
    sigma_max = Keyword.get(opts, :sigma_max, @default_sigma_max)
    rho = Keyword.get(opts, :rho, 7)

    inv_rho = 1.0 / rho
    min_inv = :math.pow(sigma_min, inv_rho)
    max_inv = :math.pow(sigma_max, inv_rho)

    steps =
      for i <- 0..(n - 1) do
        frac = if n > 1, do: i / (n - 1), else: 0.0
        :math.pow(min_inv + frac * (max_inv - min_inv), rho)
      end

    Nx.tensor(steps, type: :f32)
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      input_dim: 64,
      hidden_size: 256,
      num_layers: 4,
      sigma_min: 0.002,
      sigma_max: 80.0
    ]
  end
end
