defmodule Edifice.Interpretability.SSMAnalysisTest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.SSMAnalysis

  @n 2
  @seq 8
  @hidden 4
  @state 3

  defp sites(dt_scale) do
    key = Nx.Random.key(11)
    {b, k2} = Nx.Random.normal(key, 0.0, 1.0, shape: {@n, @seq, @state}, type: :f32)
    {c, _} = Nx.Random.normal(k2, 0.0, 1.0, shape: {@n, @seq, @state}, type: :f32)
    dt = Nx.broadcast(Nx.tensor(dt_scale, type: :f32), {@n, @seq, @hidden})
    {b, c, dt}
  end

  describe "frame_attention/4" do
    test "returns causal {n, t+1} mass, all non-negative" do
      {b, c, dt} = sites(0.05)
      att = SSMAnalysis.frame_attention(b, c, dt, 5)

      assert Nx.shape(att) == {@n, 6}
      assert att |> Nx.reduce_min() |> Nx.to_number() >= 0.0
    end

    test "large dt (fast forgetting) concentrates mass near t; small dt spreads it" do
      {b, c, _} = sites(0.1)

      recency_share = fn dt_scale ->
        dt = Nx.broadcast(Nx.tensor(dt_scale, type: :f32), {@n, @seq, @hidden})
        att = SSMAnalysis.frame_attention(b, c, dt, @seq - 1)
        total = att |> Nx.sum(axes: [1])
        # Mass on the 2 most recent source frames / total
        recent = att |> Nx.slice_along_axis(@seq - 2, 2, axis: 1) |> Nx.sum(axes: [1])
        recent |> Nx.divide(Nx.max(total, 1.0e-9)) |> Nx.mean() |> Nx.to_number()
      end

      assert recency_share.(0.1) > recency_share.(0.001)
    end

    test "raises on out-of-range t" do
      {b, c, dt} = sites(0.05)

      assert_raise ArgumentError, ~r/out of range/, fn ->
        SSMAnalysis.frame_attention(b, c, dt, @seq)
      end
    end
  end

  describe "timescale_map/2" do
    test "shapes {hidden, state}; horizons positive and ordered by state index" do
      {_b, _c, dt} = sites(0.01)
      %{a_bar_mean: a, horizon_frames: h} = SSMAnalysis.timescale_map(dt, @state)

      assert Nx.shape(a) == {@hidden, @state}
      assert Nx.shape(h) == {@hidden, @state}
      assert h |> Nx.reduce_min() |> Nx.to_number() > 0.0

      # a_diag = -(j+1): higher state index decays FASTER -> shorter horizon
      h0 = h |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.mean() |> Nx.to_number()
      hl = h |> Nx.slice_along_axis(@state - 1, 1, axis: 1) |> Nx.mean() |> Nx.to_number()
      assert h0 > hl
    end

    test "larger dt shortens horizons" do
      {_b, _c, dt_small} = sites(0.001)
      {_b2, _c2, dt_big} = sites(0.1)

      h_small = SSMAnalysis.timescale_map(dt_small, @state).horizon_frames |> Nx.mean() |> Nx.to_number()
      h_big = SSMAnalysis.timescale_map(dt_big, @state).horizon_frames |> Nx.mean() |> Nx.to_number()

      assert h_small > h_big
    end
  end
end
