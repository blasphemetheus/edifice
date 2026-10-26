defmodule Edifice.Recurrent.SpectralRadiusTest do
  use ExUnit.Case, async: true
  alias Edifice.Utils.SpectralRadius
  alias Edifice.Recurrent.Reservoir

  test "nonsymmetric sparse and dense matrices match NumPy eigenvalues" do
    fixtures = File.read!("test/fixtures/reservoir_spectral_radius.json") |> Jason.decode!()

    for %{"matrix" => matrix, "radius" => radius} <- fixtures do
      tensor = Nx.tensor(matrix, type: :f64)
      assert_in_delta SpectralRadius.compute(tensor), radius, 1.0e-9 * max(radius, 1)

      if radius > 0 do
        scaled = Reservoir.scale_spectral_radius(tensor, 0.85)
        assert_in_delta SpectralRadius.compute(scaled), 0.85, 1.0e-9
      end
    end
  end

  test "complex pairs, negative eigenvalues, nonnormal and nilpotent matrices" do
    for {matrix, radius} <- [
          {[[0, -2], [2, 0]], 2},
          {[[-3, 100, 0], [0, 2, 100], [0, 0, 1]], 3},
          {[[0, 10, 0], [0, 0, 10], [0, 0, 0]], 0},
          {[[0, 1, 0], [0, 0, 1], [1, 0, 0]], 1},
          {[[0]], 0},
          {[[-2]], 2}
        ] do
      assert_in_delta SpectralRadius.compute(Nx.tensor(matrix, type: :f64)), radius, 1.0e-10
    end

    assert_raise ArgumentError, ~r/zero-spectral-radius/, fn ->
      Reservoir.scale_spectral_radius(Nx.tensor([[0.0, 1], [0, 0]]), 0.9)
    end

    assert Nx.to_flat_list(Reservoir.scale_spectral_radius(Nx.tensor([[2.0]]), 0)) == [0.0]

    assert_in_delta Nx.to_number(Reservoir.scale_spectral_radius(Nx.tensor([[2]]), 0.9)[0][0]),
                    0.9,
                    1.0e-6
  end

  test "known linearized decay agrees with target radius and leak rate" do
    matrix = Reservoir.scale_spectral_radius(Nx.tensor([[0.0, -2], [2, 0]]), 0.8)
    # A rotation has ||h_t|| = rho^t ||h_0||. Leaking changes the effective
    # eigenvalues to (1-leak) + leak*lambda, not simply leak*rho.
    for leak <- [1.0, 0.3] do
      effective = Nx.add(Nx.multiply(Nx.eye(2), 1 - leak), Nx.multiply(matrix, leak))
      expected_radius = :math.sqrt(:math.pow(1 - leak, 2) + :math.pow(leak * 0.8, 2))
      h = Enum.reduce(1..20, Nx.tensor([1.0, 0]), fn _, h -> Nx.dot(h, effective) end)
      norm = Nx.to_number(Nx.sqrt(Nx.sum(Nx.multiply(h, h))))
      assert_in_delta norm, :math.pow(expected_radius, 20), 1.0e-6
    end
  end

  test "impossible sparse draws and invalid options fail explicitly" do
    assert_raise ArgumentError, ~r/zero-spectral-radius/, fn ->
      Reservoir.build(input_size: 2, reservoir_size: 4, sparsity: 1.0)
    end

    for opts <- [[spectral_radius: -1], [sparsity: 1.1], [leak_rate: 0]] do
      assert_raise ArgumentError, fn -> Reservoir.build([input_size: 2] ++ opts) end
    end
  end
end
