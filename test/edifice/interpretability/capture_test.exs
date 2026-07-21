defmodule Edifice.Interpretability.CaptureTest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.Capture
  alias Edifice.SSM.{Common, Mamba, MambaSSD}

  @opts [embed_dim: 8, hidden_size: 8, state_size: 4, num_layers: 2, window_size: 6, dropout: 0.0]

  defp batches(n_batches, batch, seq, d) do
    key = Nx.Random.key(7)

    Enum.map(1..n_batches, fn i ->
      {x, _} = Nx.Random.normal(Nx.Random.key(i), 0.0, 1.0, shape: {batch, seq, d}, type: :f32)
      _ = key
      x
    end)
  end

  defp init_params(model, batch, seq, d) do
    {init_fn, _} = Axon.build(model, mode: :inference)
    init_fn.(Nx.template({batch, seq, d}, :f32), Axon.ModelState.empty())
  end

  describe "probe_tap/with_probe_taps" do
    test "no active collection: tap is a no-op passthrough" do
      node = Axon.input("x", shape: {nil, 4})
      assert Common.probe_tap(node, "site") == node
    end

    test "collection gathers named sites and restores state" do
      {_result, taps} =
        Common.with_probe_taps(fn ->
          node = Axon.input("x", shape: {nil, 4})
          Common.probe_tap(node, "a")
          Common.probe_tap(node, "b")
          :done
        end)

      assert Map.keys(taps) |> Enum.sort() == ["a", "b"]
      # Collector cleared afterwards
      node = Axon.input("y", shape: {nil, 4})
      assert Common.probe_tap(node, "after") == node
    end
  end

  describe "build_probe/1" do
    test "mamba_ssd probe exposes trunk + per-block sites, same param names as build" do
      probe = MambaSSD.build_probe(@opts)
      plain = MambaSSD.build(@opts)

      probe_params = init_params(probe, 2, 6, 8)
      plain_params = init_params(plain, 2, 6, 8)

      assert Map.keys(probe_params.data) |> Enum.sort() ==
               Map.keys(plain_params.data) |> Enum.sort()
    end

    test "captured sites have full-sequence shapes; trunk output matches plain build" do
      probe = MambaSSD.build_probe(@opts)
      params = init_params(probe, 2, 6, 8)

      out = Capture.run(probe, params, batches(2, 2, 6, 8), compiler: Nx.Defn.Evaluator)

      assert Map.has_key?(out, "trunk")
      # 2 batches x 2 rows
      assert Nx.axis_size(out["trunk"], 0) == 4

      site_keys = Map.keys(out)
      assert Enum.any?(site_keys, &String.ends_with?(&1, ".gate_z"))
      assert Enum.any?(site_keys, &String.ends_with?(&1, ".dt"))

      # Sites keep {n, seq, ...}: per-timestep access included
      gate = out |> Enum.find(fn {k, _} -> String.ends_with?(k, ".gate_z") end) |> elem(1)
      assert Nx.axis_size(gate, 1) == 6
    end

    test "mamba (v1) probe shares the site vocabulary" do
      probe = Mamba.build_probe(@opts ++ [conv_size: 2])
      params = init_params(probe, 2, 6, 8)

      out =
        Capture.run(probe, params, batches(1, 2, 6, 8),
          compiler: Nx.Defn.Evaluator,
          only: ["trunk"]
        )

      assert Map.keys(out) == ["trunk"]
    end
  end

  describe "run/4" do
    test "single-output model yields %{\"output\" => _}" do
      model = MambaSSD.build(@opts)
      params = init_params(model, 2, 6, 8)

      out = Capture.run(model, params, batches(1, 2, 6, 8), compiler: Nx.Defn.Evaluator)

      assert Map.keys(out) == ["output"]
      assert Nx.shape(out["output"]) == {2, 8}
    end

    test "empty batch stream returns empty map" do
      model = MambaSSD.build(@opts)
      params = init_params(model, 2, 6, 8)
      assert Capture.run(model, params, [], compiler: Nx.Defn.Evaluator) == %{}
    end
  end
end
