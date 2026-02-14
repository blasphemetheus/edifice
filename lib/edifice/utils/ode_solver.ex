defmodule Edifice.Utils.ODESolver do
  @moduledoc """
  ODE Solver for Nx tensors - used by Liquid Neural Networks.

  Provides numerical integration methods for solving ordinary differential
  equations of the form: dx/dt = f(t, x)

  ## Available Solvers

  | Solver | Order | Adaptive | Best For |
  |--------|-------|----------|----------|
  | `:euler` | 1 | No | Fast, simple problems |
  | `:midpoint` | 2 | No | Better accuracy than Euler |
  | `:rk4` | 4 | No | Good accuracy, fixed step |
  | `:dopri5` | 4/5 | Yes | Best accuracy, adaptive step |

  ## Usage

      # Solve dx/dt = -x (exponential decay)
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

  ## Neural ODE Integration

  For Liquid Neural Networks, use `solve_ltc/5` which is optimized for
  the LTC (Liquid Time-Constant) ODE:

      dx/dt = (-x + f(x, input)) / tau

  ## Differentiability

  All solvers use pure Nx operations and are compatible with `Nx.Defn.grad/2`
  for automatic differentiation during training.

  ## Reference

  - Dormand & Prince (1980): "A family of embedded Runge-Kutta formulae"
  - Chen et al. (2018): "Neural Ordinary Differential Equations"
  """

  # Note: DOPRI5 coefficients are inlined in dopri5_step/4 for efficiency

  @doc """
  Solve an ODE from t0 to t1 with initial condition x0.

  ## Parameters

    - `f` - The ODE function `f(t, x)` returning dx/dt
    - `t0` - Initial time
    - `t1` - Final time
    - `x0` - Initial state (Nx tensor)
    - `opts` - Solver options

  ## Options

    - `:solver` - Solver type: `:euler`, `:midpoint`, `:rk4`, `:dopri5` (default: `:rk4`)
    - `:dt` - Time step for fixed-step methods (default: 0.01)
    - `:max_steps` - Maximum steps for adaptive methods (default: 1000)
    - `:atol` - Absolute tolerance for adaptive methods (default: 1.0e-6)
    - `:rtol` - Relative tolerance for adaptive methods (default: 1.0e-3)

  ## Returns

    The state at time t1.
  """
  @spec solve(
          (float(), Nx.Tensor.t() -> Nx.Tensor.t()),
          float(),
          float(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def solve(f, t0, t1, x0, opts \\ []) do
    solver = Keyword.get(opts, :solver, :rk4)
    dt = Keyword.get(opts, :dt, 0.01)

    case solver do
      :euler -> solve_euler(f, t0, t1, x0, dt)
      :midpoint -> solve_midpoint(f, t0, t1, x0, dt)
      :rk4 -> solve_rk4(f, t0, t1, x0, dt)
      :dopri5 -> solve_dopri5(f, t0, t1, x0, opts)
      _ -> solve_rk4(f, t0, t1, x0, dt)
    end
  end

  @doc """
  Solve the LTC (Liquid Time-Constant) ODE for one timestep.

  The LTC ODE is: dx/dt = (-x + activation) / tau

  This is optimized for Liquid Neural Networks where we integrate
  the state for a single frame (dt = 1.0 normalized).

  ## Parameters

    - `x` - Current hidden state [batch, hidden_size]
    - `activation` - The f(x, input) activation [batch, hidden_size]
    - `tau` - Time constants [batch, hidden_size]
    - `opts` - Solver options

  ## Options

    - `:solver` - Solver type (default: `:rk4`)
    - `:steps` - Number of integration sub-steps (default: 1)

  ## Returns

    The new hidden state after one frame.
  """
  @spec solve_ltc(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def solve_ltc(x, activation, tau, opts \\ []) do
    solver = Keyword.get(opts, :solver, :exact)
    steps = Keyword.get(opts, :steps, 1)

    # Total time for one frame = 1.0, split into steps
    dt = 1.0 / steps

    # LTC ODE: dx/dt = (-x + activation) / tau
    case solver do
      :exact ->
        # Analytical solution: x(t+dt) = activation + (x - activation) * exp(-dt/tau)
        # Unconditionally stable for any dt and tau values.
        # This is the preferred solver for the LTC equation since it's linear.
        exact_ltc_step(x, activation, tau, 1.0)

      :euler ->
        Enum.reduce(1..steps, x, fn _, h ->
          euler_ltc_step(h, activation, tau, dt)
        end)

      :midpoint ->
        Enum.reduce(1..steps, x, fn _, h ->
          midpoint_ltc_step(h, activation, tau, dt)
        end)

      :rk4 ->
        Enum.reduce(1..steps, x, fn _, h ->
          rk4_ltc_step(h, activation, tau, dt)
        end)

      :dopri5 ->
        # For DOPRI5, use adaptive stepping within the frame
        solve_ltc_adaptive(x, activation, tau, opts)

      _ ->
        exact_ltc_step(x, activation, tau, 1.0)
    end
  end

  # ============================================================================
  # Fixed-Step Solvers
  # ============================================================================

  defp solve_euler(f, t0, t1, x0, dt) do
    n_steps = max(1, round((t1 - t0) / dt))
    actual_dt = (t1 - t0) / n_steps

    Enum.reduce(0..(n_steps - 1), x0, fn i, x ->
      t = t0 + i * actual_dt
      k1 = f.(t, x)
      Nx.add(x, Nx.multiply(actual_dt, k1))
    end)
  end

  defp solve_midpoint(f, t0, t1, x0, dt) do
    n_steps = max(1, round((t1 - t0) / dt))
    actual_dt = (t1 - t0) / n_steps

    Enum.reduce(0..(n_steps - 1), x0, fn i, x ->
      t = t0 + i * actual_dt
      k1 = f.(t, x)
      x_mid = Nx.add(x, Nx.multiply(actual_dt / 2, k1))
      k2 = f.(t + actual_dt / 2, x_mid)
      Nx.add(x, Nx.multiply(actual_dt, k2))
    end)
  end

  defp solve_rk4(f, t0, t1, x0, dt) do
    n_steps = max(1, round((t1 - t0) / dt))
    actual_dt = (t1 - t0) / n_steps

    Enum.reduce(0..(n_steps - 1), x0, fn i, x ->
      t = t0 + i * actual_dt
      rk4_step(f, t, x, actual_dt)
    end)
  end

  defp rk4_step(f, t, x, dt) do
    k1 = f.(t, x)
    k2 = f.(t + dt / 2, Nx.add(x, Nx.multiply(dt / 2, k1)))
    k3 = f.(t + dt / 2, Nx.add(x, Nx.multiply(dt / 2, k2)))
    k4 = f.(t + dt, Nx.add(x, Nx.multiply(dt, k3)))

    # x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    weighted_sum =
      k1
      |> Nx.add(Nx.multiply(2.0, k2))
      |> Nx.add(Nx.multiply(2.0, k3))
      |> Nx.add(k4)

    Nx.add(x, Nx.multiply(dt / 6, weighted_sum))
  end

  # ============================================================================
  # Adaptive Solver (Dormand-Prince 4/5)
  # ============================================================================

  defp solve_dopri5(f, t0, t1, x0, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-6)
    rtol = Keyword.get(opts, :rtol, 1.0e-3)
    max_steps = Keyword.get(opts, :max_steps, 1000)

    # Initial step size estimate
    dt_init = (t1 - t0) / 10

    dopri5_loop(f, t0, t1, x0, dt_init, atol, rtol, max_steps, 0)
  end

  defp dopri5_loop(_f, t, t1, x, _dt, _atol, _rtol, _max_steps, _step) when t >= t1, do: x

  defp dopri5_loop(_f, _t, _t1, x, _dt, _atol, _rtol, max_steps, step) when step >= max_steps,
    do: x

  defp dopri5_loop(f, t, t1, x, dt, atol, rtol, max_steps, step) do
    # Clip dt to not overshoot t1
    dt = min(dt, t1 - t)

    # Compute DOPRI5 step
    {x_new, x_err, _k7} = dopri5_step(f, t, x, dt)

    # Estimate error
    error = estimate_error(x_err, x, x_new, atol, rtol)

    if error <= 1.0 do
      # Accept step
      # Compute new step size
      dt_new = dt * min(5.0, max(0.2, 0.9 * :math.pow(error, -0.2)))

      dopri5_loop(f, t + dt, t1, x_new, dt_new, atol, rtol, max_steps, step + 1)
    else
      # Reject step, reduce step size
      dt_new = dt * max(0.2, 0.9 * :math.pow(error, -0.25))

      dopri5_loop(f, t, t1, x, dt_new, atol, rtol, max_steps, step + 1)
    end
  end

  defp dopri5_step(f, t, x, dt) do
    # Compute the 7 stages
    k1 = f.(t, x)
    k2 = f.(t + dt / 5, Nx.add(x, Nx.multiply(dt / 5, k1)))

    k3 =
      f.(
        t + dt * 3 / 10,
        Nx.add(x, Nx.multiply(dt, Nx.add(Nx.multiply(3 / 40, k1), Nx.multiply(9 / 40, k2))))
      )

    k4_input =
      x
      |> Nx.add(Nx.multiply(dt * 44 / 45, k1))
      |> Nx.subtract(Nx.multiply(dt * 56 / 15, k2))
      |> Nx.add(Nx.multiply(dt * 32 / 9, k3))

    k4 = f.(t + dt * 4 / 5, k4_input)

    k5_input =
      x
      |> Nx.add(Nx.multiply(dt * 19_372 / 6561, k1))
      |> Nx.subtract(Nx.multiply(dt * 25_360 / 2187, k2))
      |> Nx.add(Nx.multiply(dt * 64_448 / 6561, k3))
      |> Nx.subtract(Nx.multiply(dt * 212 / 729, k4))

    k5 = f.(t + dt * 8 / 9, k5_input)

    k6_input =
      x
      |> Nx.add(Nx.multiply(dt * 9017 / 3168, k1))
      |> Nx.subtract(Nx.multiply(dt * 355 / 33, k2))
      |> Nx.add(Nx.multiply(dt * 46_732 / 5247, k3))
      |> Nx.add(Nx.multiply(dt * 49 / 176, k4))
      |> Nx.subtract(Nx.multiply(dt * 5103 / 18_656, k5))

    k6 = f.(t + dt, k6_input)

    # 5th order solution
    x_new =
      x
      |> Nx.add(Nx.multiply(dt * 35 / 384, k1))
      |> Nx.add(Nx.multiply(dt * 500 / 1113, k3))
      |> Nx.add(Nx.multiply(dt * 125 / 192, k4))
      |> Nx.subtract(Nx.multiply(dt * 2187 / 6784, k5))
      |> Nx.add(Nx.multiply(dt * 11 / 84, k6))

    # For FSAL (first same as last), compute k7
    k7 = f.(t + dt, x_new)

    # 4th order solution for error estimation
    x_4th =
      x
      |> Nx.add(Nx.multiply(dt * 5179 / 57_600, k1))
      |> Nx.add(Nx.multiply(dt * 7571 / 16_695, k3))
      |> Nx.add(Nx.multiply(dt * 393 / 640, k4))
      |> Nx.subtract(Nx.multiply(dt * 92_097 / 339_200, k5))
      |> Nx.add(Nx.multiply(dt * 187 / 2100, k6))
      |> Nx.add(Nx.multiply(dt / 40, k7))

    # Error = |x_5th - x_4th|
    x_err = Nx.subtract(x_new, x_4th)

    {x_new, x_err, k7}
  end

  defp estimate_error(x_err, x, x_new, atol, rtol) do
    # Scale = atol + rtol * max(|x|, |x_new|)
    scale =
      Nx.add(
        atol,
        Nx.multiply(rtol, Nx.max(Nx.abs(x), Nx.abs(x_new)))
      )

    # Error = sqrt(mean((x_err / scale)^2))
    scaled_err = Nx.divide(x_err, scale)
    Nx.mean(Nx.pow(scaled_err, 2)) |> Nx.sqrt() |> Nx.to_number()
  end

  # ============================================================================
  # LTC-Specific Solvers (Optimized for Liquid Neural Networks)
  # ============================================================================

  # LTC ODE: dx/dt = (-x + activation) / tau
  # These are specialized versions that don't need the general f(t, x) interface

  # Exact analytical solution for the linear LTC ODE: dx/dt = (activation - x) / tau
  # Solution: x(t+dt) = activation + (x(t) - activation) * exp(-dt/tau)
  # Unconditionally stable — no step size constraints.
  defp exact_ltc_step(x, activation, tau, dt) do
    decay = Nx.exp(Nx.negate(Nx.divide(dt, tau)))
    Nx.add(activation, Nx.multiply(Nx.subtract(x, activation), decay))
  end

  defp euler_ltc_step(x, activation, tau, dt) do
    # dx/dt = (activation - x) / tau
    dx = Nx.divide(Nx.subtract(activation, x), tau)
    Nx.add(x, Nx.multiply(dt, dx))
  end

  defp midpoint_ltc_step(x, activation, tau, dt) do
    # k1 = f(x)
    k1 = Nx.divide(Nx.subtract(activation, x), tau)
    # x_mid = x + (dt/2) * k1
    x_mid = Nx.add(x, Nx.multiply(dt / 2, k1))
    # k2 = f(x_mid)
    k2 = Nx.divide(Nx.subtract(activation, x_mid), tau)
    # x_new = x + dt * k2
    Nx.add(x, Nx.multiply(dt, k2))
  end

  defp rk4_ltc_step(x, activation, tau, dt) do
    # For LTC ODE: dx/dt = (activation - x) / tau
    # Note: activation is constant over the step (quasi-static)

    k1 = Nx.divide(Nx.subtract(activation, x), tau)

    x_mid1 = Nx.add(x, Nx.multiply(dt / 2, k1))
    k2 = Nx.divide(Nx.subtract(activation, x_mid1), tau)

    x_mid2 = Nx.add(x, Nx.multiply(dt / 2, k2))
    k3 = Nx.divide(Nx.subtract(activation, x_mid2), tau)

    x_end = Nx.add(x, Nx.multiply(dt, k3))
    k4 = Nx.divide(Nx.subtract(activation, x_end), tau)

    # x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    weighted_sum =
      k1
      |> Nx.add(Nx.multiply(2.0, k2))
      |> Nx.add(Nx.multiply(2.0, k3))
      |> Nx.add(k4)

    Nx.add(x, Nx.multiply(dt / 6, weighted_sum))
  end

  defp solve_ltc_adaptive(x, activation, tau, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    rtol = Keyword.get(opts, :rtol, 1.0e-2)
    max_steps = Keyword.get(opts, :max_steps, 100)

    # Solve from t=0 to t=1 (one frame)
    dt_init = 0.25

    ltc_dopri5_loop(x, activation, tau, 0.0, 1.0, dt_init, atol, rtol, max_steps, 0)
  end

  defp ltc_dopri5_loop(x, _activation, _tau, t, t1, _dt, _atol, _rtol, _max_steps, _step)
       when t >= t1,
       do: x

  defp ltc_dopri5_loop(x, _activation, _tau, _t, _t1, _dt, _atol, _rtol, max_steps, step)
       when step >= max_steps,
       do: x

  defp ltc_dopri5_loop(x, activation, tau, t, t1, dt, atol, rtol, max_steps, step) do
    dt = min(dt, t1 - t)

    {x_new, x_err} = ltc_dopri5_step(x, activation, tau, dt)

    error = estimate_error(x_err, x, x_new, atol, rtol)

    if error <= 1.0 do
      dt_new = dt * min(5.0, max(0.2, 0.9 * :math.pow(error, -0.2)))
      ltc_dopri5_loop(x_new, activation, tau, t + dt, t1, dt_new, atol, rtol, max_steps, step + 1)
    else
      dt_new = dt * max(0.2, 0.9 * :math.pow(error, -0.25))
      ltc_dopri5_loop(x, activation, tau, t, t1, dt_new, atol, rtol, max_steps, step + 1)
    end
  end

  defp ltc_dopri5_step(x, activation, tau, dt) do
    # Specialized DOPRI5 for LTC ODE
    ltc_f = fn h -> Nx.divide(Nx.subtract(activation, h), tau) end

    k1 = ltc_f.(x)
    k2 = ltc_f.(Nx.add(x, Nx.multiply(dt / 5, k1)))

    k3 =
      ltc_f.(Nx.add(x, Nx.multiply(dt, Nx.add(Nx.multiply(3 / 40, k1), Nx.multiply(9 / 40, k2)))))

    k4_input =
      x
      |> Nx.add(Nx.multiply(dt * 44 / 45, k1))
      |> Nx.subtract(Nx.multiply(dt * 56 / 15, k2))
      |> Nx.add(Nx.multiply(dt * 32 / 9, k3))

    k4 = ltc_f.(k4_input)

    k5_input =
      x
      |> Nx.add(Nx.multiply(dt * 19_372 / 6561, k1))
      |> Nx.subtract(Nx.multiply(dt * 25_360 / 2187, k2))
      |> Nx.add(Nx.multiply(dt * 64_448 / 6561, k3))
      |> Nx.subtract(Nx.multiply(dt * 212 / 729, k4))

    k5 = ltc_f.(k5_input)

    k6_input =
      x
      |> Nx.add(Nx.multiply(dt * 9017 / 3168, k1))
      |> Nx.subtract(Nx.multiply(dt * 355 / 33, k2))
      |> Nx.add(Nx.multiply(dt * 46_732 / 5247, k3))
      |> Nx.add(Nx.multiply(dt * 49 / 176, k4))
      |> Nx.subtract(Nx.multiply(dt * 5103 / 18_656, k5))

    k6 = ltc_f.(k6_input)

    # 5th order solution
    x_new =
      x
      |> Nx.add(Nx.multiply(dt * 35 / 384, k1))
      |> Nx.add(Nx.multiply(dt * 500 / 1113, k3))
      |> Nx.add(Nx.multiply(dt * 125 / 192, k4))
      |> Nx.subtract(Nx.multiply(dt * 2187 / 6784, k5))
      |> Nx.add(Nx.multiply(dt * 11 / 84, k6))

    k7 = ltc_f.(x_new)

    # 4th order for error
    x_4th =
      x
      |> Nx.add(Nx.multiply(dt * 5179 / 57_600, k1))
      |> Nx.add(Nx.multiply(dt * 7571 / 16_695, k3))
      |> Nx.add(Nx.multiply(dt * 393 / 640, k4))
      |> Nx.subtract(Nx.multiply(dt * 92_097 / 339_200, k5))
      |> Nx.add(Nx.multiply(dt * 187 / 2100, k6))
      |> Nx.add(Nx.multiply(dt / 40, k7))

    x_err = Nx.subtract(x_new, x_4th)

    {x_new, x_err}
  end
end
