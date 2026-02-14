defmodule Edifice.Utils.ODESolverTest do
  use ExUnit.Case, async: true

  alias Edifice.Utils.ODESolver

  describe "solve/5" do
    test "euler solver for exponential decay" do
      # dx/dt = -x, solution: x(t) = x0 * exp(-t)
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.01)
      # exp(-1) ≈ 0.3679
      val = Nx.to_number(result[0])
      assert abs(val - 0.3679) < 0.05
    end

    test "midpoint solver is more accurate than euler" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      euler = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.1)
      midpoint = ODESolver.solve(f, 0.0, 1.0, x0, solver: :midpoint, dt: 0.1)
      exact = :math.exp(-1.0)

      euler_err = abs(Nx.to_number(euler[0]) - exact)
      midpoint_err = abs(Nx.to_number(midpoint[0]) - exact)
      assert midpoint_err < euler_err
    end

    test "rk4 solver for exponential decay" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)
      val = Nx.to_number(result[0])
      assert abs(val - :math.exp(-1.0)) < 0.001
    end

    test "dopri5 solver for exponential decay" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :dopri5, dt: 0.1)
      val = Nx.to_number(result[0])
      assert abs(val - :math.exp(-1.0)) < 0.01
    end

    test "handles multi-dimensional state" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0, 2.0, 3.0])
      result = ODESolver.solve(f, 0.0, 0.5, x0, solver: :rk4, dt: 0.1)
      assert Nx.shape(result) == {3}
    end

    test "default solver works" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])
      result = ODESolver.solve(f, 0.0, 0.5, x0)
      assert Nx.shape(result) == {1}
    end
  end
end
