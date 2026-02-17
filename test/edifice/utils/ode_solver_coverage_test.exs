defmodule Edifice.Utils.ODESolverCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Utils.ODESolver.
  Targets solve_ltc/4 with all solver variants, multi-step integration,
  adaptive LTC solver, step rejection paths, and edge cases.
  """
  use ExUnit.Case, async: true

  alias Edifice.Utils.ODESolver

  # ============================================================================
  # solve_ltc/4 — Euler solver
  # ============================================================================
  describe "solve_ltc/4 with :euler" do
    test "single step euler LTC integration" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :euler, steps: 1)
      assert Nx.shape(result) == {batch, hidden}

      # LTC ODE: dx/dt = (activation - x) / tau
      # With x=1, activation=0.5, tau=1: dx/dt = (0.5 - 1) / 1 = -0.5
      # After one step (dt=1): x_new = 1 + 1 * (-0.5) = 0.5
      val = Nx.to_number(result[0][0])
      assert_in_delta val, 0.5, 0.01
    end

    test "multi-step euler LTC integration" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.0, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :euler, steps: 10)
      assert Nx.shape(result) == {batch, hidden}

      # With more steps, euler should be more accurate
      # ODE: dx/dt = (0 - x)/1 = -x, solution: x(1) = exp(-1) ~ 0.3679
      val = Nx.to_number(result[0][0])
      assert val > 0.0 and val < 1.0
    end

    test "euler LTC with varying tau" do
      batch = 1
      hidden = 2
      x = Nx.tensor([[1.0, 1.0]])
      activation = Nx.tensor([[0.0, 0.0]])
      # Different time constants: fast vs slow decay
      tau = Nx.tensor([[0.5, 2.0]])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :euler, steps: 5)
      assert Nx.shape(result) == {batch, hidden}

      # Smaller tau means faster decay toward activation
      fast = Nx.to_number(result[0][0])
      slow = Nx.to_number(result[0][1])
      assert fast < slow
    end
  end

  # ============================================================================
  # solve_ltc/4 — Midpoint solver
  # ============================================================================
  describe "solve_ltc/4 with :midpoint" do
    test "single step midpoint LTC integration" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :midpoint, steps: 1)
      assert Nx.shape(result) == {batch, hidden}

      # Midpoint should give different (better) result than euler for same step count
      val = Nx.to_number(result[0][0])
      assert val > 0.0 and val < 1.0
    end

    test "multi-step midpoint LTC is more accurate than single step" do
      x = Nx.tensor([[1.0]])
      activation = Nx.tensor([[0.0]])
      tau = Nx.tensor([[1.0]])

      result_1 = ODESolver.solve_ltc(x, activation, tau, solver: :midpoint, steps: 1)
      result_10 = ODESolver.solve_ltc(x, activation, tau, solver: :midpoint, steps: 10)

      # Exact solution: exp(-1) ~ 0.3679
      exact = :math.exp(-1.0)
      err_1 = abs(Nx.to_number(result_1[0][0]) - exact)
      err_10 = abs(Nx.to_number(result_10[0][0]) - exact)

      # More steps should be more accurate
      assert err_10 < err_1
    end
  end

  # ============================================================================
  # solve_ltc/4 — RK4 solver (default)
  # ============================================================================
  describe "solve_ltc/4 with :rk4" do
    test "single step rk4 LTC integration" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 1)
      assert Nx.shape(result) == {batch, hidden}

      val = Nx.to_number(result[0][0])
      assert val > 0.0 and val < 1.0
    end

    test "rk4 LTC default solver (no opts)" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      # Default solver should be :rk4 with steps: 1
      result = ODESolver.solve_ltc(x, activation, tau)
      assert Nx.shape(result) == {batch, hidden}
    end

    test "rk4 LTC with multiple steps converges to activation" do
      x = Nx.tensor([[1.0]])
      activation = Nx.tensor([[0.0]])
      tau = Nx.tensor([[1.0]])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 5)

      # ODE: dx/dt = (0 - x)/1 = -x, x(1) = exp(-1) ~ 0.3679
      val = Nx.to_number(result[0][0])
      assert_in_delta val, :math.exp(-1.0), 0.01
    end

    test "rk4 LTC is more accurate than euler" do
      x = Nx.tensor([[1.0]])
      activation = Nx.tensor([[0.0]])
      tau = Nx.tensor([[1.0]])

      euler_result = ODESolver.solve_ltc(x, activation, tau, solver: :euler, steps: 1)
      rk4_result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 1)

      exact = :math.exp(-1.0)
      euler_err = abs(Nx.to_number(euler_result[0][0]) - exact)
      rk4_err = abs(Nx.to_number(rk4_result[0][0]) - exact)
      assert rk4_err < euler_err
    end
  end

  # ============================================================================
  # solve_ltc/4 — DOPRI5 adaptive solver
  # ============================================================================
  describe "solve_ltc/4 with :dopri5" do
    test "dopri5 LTC integration" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :dopri5)
      assert Nx.shape(result) == {batch, hidden}

      # Should converge toward activation
      val = Nx.to_number(result[0][0])
      assert val > 0.0 and val < 1.0
    end

    test "dopri5 LTC with custom tolerance" do
      batch = 1
      hidden = 2
      x = Nx.tensor([[1.0, 2.0]])
      activation = Nx.tensor([[0.5, 0.5]])
      tau = Nx.tensor([[1.0, 1.0]])

      result =
        ODESolver.solve_ltc(x, activation, tau,
          solver: :dopri5,
          atol: 1.0e-3,
          rtol: 1.0e-2
        )

      assert Nx.shape(result) == {batch, hidden}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "dopri5 LTC with max_steps limit" do
      batch = 1
      hidden = 1
      x = Nx.tensor([[1.0]])
      activation = Nx.tensor([[0.0]])
      tau = Nx.tensor([[1.0]])

      # Very few max_steps to trigger the max_steps guard
      result =
        ODESolver.solve_ltc(x, activation, tau,
          solver: :dopri5,
          max_steps: 2
        )

      assert Nx.shape(result) == {batch, hidden}
    end

    test "dopri5 LTC with very tight tolerance (triggers step rejection)" do
      batch = 1
      hidden = 2
      x = Nx.tensor([[10.0, -10.0]])
      activation = Nx.tensor([[0.0, 0.0]])
      # Very small tau forces rapid dynamics -> larger errors -> step rejections
      tau = Nx.tensor([[0.01, 0.01]])

      result =
        ODESolver.solve_ltc(x, activation, tau,
          solver: :dopri5,
          atol: 1.0e-8,
          rtol: 1.0e-6,
          max_steps: 200
        )

      assert Nx.shape(result) == {batch, hidden}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # solve_ltc/4 — Unknown/default solver fallback
  # ============================================================================
  describe "solve_ltc/4 with unknown solver" do
    test "falls back to rk4 for unknown solver" do
      batch = 2
      hidden = 4
      x = Nx.broadcast(1.0, {batch, hidden})
      activation = Nx.broadcast(0.5, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      # Unknown solver should fall back to rk4
      result = ODESolver.solve_ltc(x, activation, tau, solver: :unknown_solver, steps: 1)
      assert Nx.shape(result) == {batch, hidden}

      # Should match rk4 result
      rk4_result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 1)

      diff = Nx.subtract(result, rk4_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 0.01
    end

    test "falls back to rk4 for unknown solver with multiple steps" do
      x = Nx.tensor([[1.0, 0.5]])
      activation = Nx.tensor([[0.0, 1.0]])
      tau = Nx.tensor([[1.0, 1.0]])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :bogus, steps: 3)
      rk4_result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 3)

      diff = Nx.subtract(result, rk4_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 0.01
    end
  end

  # ============================================================================
  # solve/5 — Unknown solver fallback
  # ============================================================================
  describe "solve/5 with unknown solver" do
    test "falls back to rk4 for unknown solver" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :nonexistent, dt: 0.1)
      rk4_result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

      diff =
        Nx.subtract(result, rk4_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # solve/5 — DOPRI5 edge cases
  # ============================================================================
  describe "solve/5 dopri5 edge cases" do
    test "dopri5 with custom tolerances" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result =
        ODESolver.solve(f, 0.0, 1.0, x0,
          solver: :dopri5,
          atol: 1.0e-4,
          rtol: 1.0e-2,
          max_steps: 500
        )

      val = Nx.to_number(result[0])
      assert_in_delta val, :math.exp(-1.0), 0.05
    end

    test "dopri5 with max_steps=1 terminates early" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :dopri5, max_steps: 1)
      # With only 1 step allowed, result won't be very accurate but should not crash
      assert Nx.shape(result) == {1}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "dopri5 step rejection with stiff ODE" do
      # Very stiff ODE that forces step rejections
      # dx/dt = -100*x: fast dynamics require small steps
      f = fn _t, x -> Nx.multiply(-100.0, x) end
      x0 = Nx.tensor([1.0])

      result =
        ODESolver.solve(f, 0.0, 0.1, x0,
          solver: :dopri5,
          atol: 1.0e-8,
          rtol: 1.0e-6,
          max_steps: 500
        )

      assert Nx.shape(result) == {1}
      # exp(-100 * 0.1) = exp(-10) ~ 4.54e-5
      val = Nx.to_number(result[0])
      assert val >= 0.0 and val < 0.01
    end
  end

  # ============================================================================
  # solve/5 — Midpoint edge cases
  # ============================================================================
  describe "solve/5 midpoint edge cases" do
    test "midpoint with very large dt (single step)" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      # dt > (t1 - t0), should use a single step
      result = ODESolver.solve(f, 0.0, 0.5, x0, solver: :midpoint, dt: 10.0)
      assert Nx.shape(result) == {1}
    end

    test "midpoint with multi-dimensional state" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      result = ODESolver.solve(f, 0.0, 0.5, x0, solver: :midpoint, dt: 0.1)
      assert Nx.shape(result) == {4}

      # Each component should decay
      for i <- 0..3 do
        val = Nx.to_number(result[i])
        initial = Nx.to_number(x0[i])
        assert val < initial
        assert val > 0.0
      end
    end
  end

  # ============================================================================
  # solve/5 — Euler edge cases
  # ============================================================================
  describe "solve/5 euler edge cases" do
    test "euler with very small dt (many steps)" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.001)
      val = Nx.to_number(result[0])
      # With small dt, euler should be very accurate
      assert_in_delta val, :math.exp(-1.0), 0.01
    end
  end

  # ============================================================================
  # solve/5 — Time-dependent ODE
  # ============================================================================
  describe "solve/5 with time-dependent ODE" do
    test "euler handles time-dependent f(t, x)" do
      # dx/dt = t (solution: x = t^2/2)
      f = fn t, _x -> Nx.tensor([t]) end
      x0 = Nx.tensor([0.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.01)
      val = Nx.to_number(result[0])
      # x(1) = 0.5
      assert_in_delta val, 0.5, 0.05
    end

    test "rk4 handles time-dependent f(t, x)" do
      # dx/dt = t (solution: x = t^2/2)
      f = fn t, _x -> Nx.tensor([t]) end
      x0 = Nx.tensor([0.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)
      val = Nx.to_number(result[0])
      assert_in_delta val, 0.5, 0.001
    end

    test "midpoint handles time-dependent f(t, x)" do
      f = fn t, _x -> Nx.tensor([t]) end
      x0 = Nx.tensor([0.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :midpoint, dt: 0.1)
      val = Nx.to_number(result[0])
      assert_in_delta val, 0.5, 0.01
    end

    test "dopri5 handles time-dependent f(t, x)" do
      # dx/dt = t * sin(x + 1) — nonlinear, time-dependent, avoids exact zero error
      f = fn t, x -> Nx.multiply(t, Nx.sin(Nx.add(x, 1.0))) end
      x0 = Nx.tensor([0.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :dopri5)
      assert Nx.shape(result) == {1}
      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # solve_ltc/4 — Correctness tests
  # ============================================================================
  describe "solve_ltc/4 correctness" do
    test "all solvers converge to activation for large steps" do
      x = Nx.tensor([[5.0]])
      activation = Nx.tensor([[1.0]])
      tau = Nx.tensor([[0.1]])

      for solver <- [:euler, :midpoint, :rk4] do
        result = ODESolver.solve_ltc(x, activation, tau, solver: solver, steps: 20)
        val = Nx.to_number(result[0][0])
        # With small tau, state should converge close to activation (1.0)
        assert_in_delta val, 1.0, 0.1,
          "#{solver} should converge to activation, got #{val}"
      end
    end

    test "solve_ltc preserves shape for different batch sizes" do
      for {batch, hidden} <- [{1, 2}, {4, 8}, {2, 16}] do
        x = Nx.broadcast(1.0, {batch, hidden})
        activation = Nx.broadcast(0.5, {batch, hidden})
        tau = Nx.broadcast(1.0, {batch, hidden})

        result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 2)
        assert Nx.shape(result) == {batch, hidden}
      end
    end
  end
end
