defmodule Edifice.Energy.HopfieldTest do
  use ExUnit.Case, async: true
  @moduletag :energy

  alias Edifice.Energy.Hopfield

  describe "build/1" do
    test "produces correct output shape" do
      model =
        Hopfield.build(
          input_dim: 64,
          num_patterns: 16,
          pattern_dim: 32
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({4, 64}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {4, 64})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {4, 32}
    end

    test "uses default num_patterns=64 and pattern_dim=128" do
      model = Hopfield.build(input_dim: 32)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))
      assert Nx.shape(output) == {2, 128}
    end

    test "respects custom beta parameter" do
      # beta affects the softmax sharpness; model should still build and run
      model =
        Hopfield.build(
          input_dim: 32,
          num_patterns: 8,
          pattern_dim: 16,
          beta: 5.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))
      assert Nx.shape(output) == {2, 16}
    end
  end

  describe "hopfield_layer/2" do
    test "can be applied to an existing Axon node" do
      input = Axon.input("input", shape: {nil, 32})
      output = Hopfield.hopfield_layer(input, num_patterns: 8, pattern_dim: 16)

      {init_fn, predict_fn} = Axon.build(output)

      params =
        init_fn.(Nx.template({4, 32}, :f32), Axon.ModelState.empty())

      result = predict_fn.(params, Nx.broadcast(0.5, {4, 32}))
      assert Nx.shape(result) == {4, 16}
    end
  end

  describe "build_associative_memory/1" do
    test "produces hidden_size output" do
      model =
        Hopfield.build_associative_memory(
          input_dim: 32,
          num_patterns: 8,
          pattern_dim: 32,
          hidden_size: 64,
          num_layers: 2,
          num_heads: 1,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))
      assert Nx.shape(output) == {2, 64}
    end

    test "supports multi-head configuration" do
      model =
        Hopfield.build_associative_memory(
          input_dim: 64,
          num_patterns: 16,
          pattern_dim: 64,
          hidden_size: 64,
          num_heads: 4,
          num_layers: 1,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({2, 64}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 64}))
      assert Nx.shape(output) == {2, 64}
    end
  end

  describe "energy/3" do
    test "returns energy per batch element" do
      query = Nx.broadcast(0.5, {4, 16})
      patterns = Nx.broadcast(0.5, {8, 16})
      energies = Hopfield.energy(query, patterns, 1.0)

      assert Nx.shape(energies) == {4}
    end

    test "higher beta yields more negative energy for close matches" do
      # Pattern exactly matching query should have very negative energy
      pattern = Nx.broadcast(0.5, {1, 16})
      query = pattern

      energy_low_beta = Hopfield.energy(query, pattern, 0.1) |> Nx.squeeze() |> Nx.to_number()
      energy_high_beta = Hopfield.energy(query, pattern, 10.0) |> Nx.squeeze() |> Nx.to_number()

      # Higher beta amplifies the match, yielding more negative energy
      assert energy_high_beta < energy_low_beta
    end
  end
end
