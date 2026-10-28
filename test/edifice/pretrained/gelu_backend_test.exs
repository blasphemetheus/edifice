defmodule Edifice.Pretrained.GeluBackendTest do
  use ExUnit.Case, async: true
  import Nx.Testing

  # This checks the upstream XLA fix itself. Published EXLA 1.0 / XLA 0.10
  # has the documented defect; run this unchanged against the upstream profile.
  @tag :upstream_fix
  test "CPU erf and GELU saturate correctly at large f32 inputs" do
    erf = Nx.Defn.jit(&Nx.erf/1, compiler: EXLA, client: :host)
    gelu = Nx.Defn.jit(&Axon.Activations.gelu/1, compiler: EXLA, client: :host)

    for shape <- [{1}, {11}, {2, 16}],
        value <- [-59.02690505981445, -10.0, 10.0, 59.02690505981445] do
      input = Nx.broadcast(Nx.tensor(value, type: :f32), shape)
      # At these magnitudes the true results round to exactly ±1 for erf,
      # and to zero or the original input for GELU in f32.
      assert_equal(erf.(input), Nx.broadcast(if(value < 0, do: -1.0, else: 1.0), shape))
      expected = if value < 0, do: Nx.broadcast(0.0, shape), else: input
      assert_equal(gelu.(input), expected)
    end
  end
end
