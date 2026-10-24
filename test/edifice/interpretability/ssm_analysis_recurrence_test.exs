defmodule Edifice.Interpretability.SSMAnalysisRecurrenceTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias Edifice.Interpretability.SSMAnalysis
  alias Edifice.SSM.Common

  test "signed coefficients reconstruct the scan at every frame with varying dt and signed B/C" do
    {x, key} = Nx.Random.normal(Nx.Random.key(42), shape: {2, 5, 3})
    {b, key} = Nx.Random.normal(key, shape: {2, 5, 4})
    {c, key} = Nx.Random.normal(key, shape: {2, 5, 4})
    {dt, _} = Nx.Random.uniform(key, -0.1, 0.2, shape: {2, 5, 3})
    {a, bx} = Common.discretize_ssm(x, b, dt, 4)
    expected = Common.sequential_scan(a, bx) |> Common.compute_ssm_output(c)

    for t <- 0..4 do
      kernel = SSMAnalysis.frame_kernel(b, c, dt, t)
      actual = Nx.multiply(kernel, Nx.slice_along_axis(x, 0, t + 1, axis: 1)) |> Nx.sum(axes: [1])
      assert_all_close(actual, Nx.take(expected, t, axis: 1), atol: 1.0e-6)

      assert_all_close(
        SSMAnalysis.frame_attention(b, c, dt, t),
        Nx.sum(Nx.abs(kernel), axes: [2])
      )
    end
  end

  test "coefficients reconstruct actual Mamba probe output before gate and projection" do
    Process.put(:__edifice_force_fallback__, true)

    model =
      Edifice.SSM.Mamba.build_probe(
        embed_dim: 3,
        hidden_size: 4,
        state_size: 3,
        num_layers: 1,
        conv_size: 2,
        dropout: 0.0
      )

    {init, predict} = Axon.build(model, seed: 42)
    input = %{"state_sequence" => Nx.iota({2, 5, 3}) |> Nx.divide(30)}
    sites = predict.(init.(input, Axon.ModelState.empty()), input)
    {b_site, b} = Enum.find(sites, fn {name, _} -> String.ends_with?(name, ".B") end)
    prefix = String.replace_suffix(b_site, ".B", "")
    {_, x} = Enum.find(sites, fn {name, _} -> String.ends_with?(name, ".post_conv") end)
    {_, expected} = Enum.find(sites, fn {name, _} -> String.ends_with?(name, ".ssm_out") end)

    for t <- 0..4 do
      kernel = SSMAnalysis.frame_kernel(b, sites[prefix <> ".C"], sites[prefix <> ".dt"], t)
      actual = Nx.multiply(kernel, Nx.slice_along_axis(x, 0, t + 1, axis: 1)) |> Nx.sum(axes: [1])
      assert_all_close(actual, Nx.take(expected, t, axis: 1), atol: 1.0e-5)
    end
  end

  test "large state indices do not cause log-exp NaNs or future overflow" do
    b = Nx.broadcast(1.0, {1, 128, 2048})
    c = Nx.broadcast(1.0, {1, 128, 2048})
    dt = Nx.broadcast(0.1, {1, 128, 1})
    kernel = SSMAnalysis.frame_kernel(b, c, dt, 0)
    assert_all_close(kernel, Nx.tensor([[[204.8]]]), atol: 1.0e-3)
    times = SSMAnalysis.timescale_map(dt, 2048)
    assert Nx.to_number(Nx.any(Nx.is_nan(times.horizon_frames))) == 0
    assert_in_delta Nx.to_number(times.horizon_frames[0][2047]), 1 / 204.8, 1.0e-6
  end

  test "future capture values and gradients do not affect an earlier frame" do
    b = Nx.broadcast(1.0, {1, 8, 2})
    c = b
    dt = Nx.broadcast(0.01, {1, 8, 3})
    poisoned = Nx.put_slice(dt, [0, 3, 0], Nx.broadcast(Nx.Constants.nan(), {1, 5, 3}))

    assert_equal(
      SSMAnalysis.frame_kernel(b, c, dt, 2),
      SSMAnalysis.frame_kernel(b, c, poisoned, 2)
    )

    grad = Nx.Defn.grad(dt, fn d -> SSMAnalysis.frame_kernel(b, c, d, 2) |> Nx.sum() end)
    assert_equal(Nx.slice_along_axis(grad, 3, 5, axis: 1), Nx.broadcast(0.0, {1, 5, 3}))
  end

  test "timescales match a constant-dt closed form and validate capture shapes" do
    result = SSMAnalysis.timescale_map(Nx.broadcast(0.05, {2, 6, 1}), 3)
    assert_all_close(result.horizon_frames, Nx.tensor([[20.0, 10.0, 20.0 / 3]]), atol: 1.0e-4)

    for t <- [-1, 3, 0.5] do
      assert_raise ArgumentError, fn ->
        SSMAnalysis.frame_kernel(
          Nx.broadcast(1.0, {1, 3, 2}),
          Nx.broadcast(1.0, {1, 3, 2}),
          Nx.broadcast(0.1, {1, 3, 1}),
          t
        )
      end
    end

    assert_raise ArgumentError, fn -> SSMAnalysis.timescale_map(Nx.tensor([0.1]), 2) end
  end
end
