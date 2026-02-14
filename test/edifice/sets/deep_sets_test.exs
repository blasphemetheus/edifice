defmodule Edifice.Sets.DeepSetsTest do
  use ExUnit.Case, async: true

  alias Edifice.Sets.DeepSets

  describe "build/1" do
    test "produces correct output shape" do
      model =
        DeepSets.build(
          input_dim: 3,
          output_dim: 10,
          hidden_size: 32,
          phi_sizes: [32],
          rho_sizes: [16]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({4, 20, 3}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {4, 20, 3})
      output = predict_fn.(params, input)

      # [batch, output_dim]
      assert Nx.shape(output) == {4, 10}
    end

    test "output is invariant to set element ordering" do
      model =
        DeepSets.build(
          input_dim: 3,
          output_dim: 5,
          hidden_size: 16,
          phi_sizes: [16],
          rho_sizes: [8],
          aggregation: :sum
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({1, 4, 3}, :f32), Axon.ModelState.empty())

      # Create a set and a permuted version
      set = Nx.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])

      permuted =
        Nx.tensor([[[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [4.0, 5.0, 6.0]]])

      output_original = predict_fn.(params, set)
      output_permuted = predict_fn.(params, permuted)

      # With sum aggregation, outputs should be identical for permuted inputs
      assert Nx.all_close(output_original, output_permuted, atol: 1.0e-5) |> Nx.to_number() == 1
    end

    test "works with different set sizes" do
      model =
        DeepSets.build(
          input_dim: 4,
          output_dim: 8,
          hidden_size: 16,
          phi_sizes: [16],
          rho_sizes: [8]
        )

      {init_fn, predict_fn} = Axon.build(model)

      # Set size 10
      params_10 =
        init_fn.(Nx.template({2, 10, 4}, :f32), Axon.ModelState.empty())

      output_10 = predict_fn.(params_10, Nx.broadcast(0.5, {2, 10, 4}))
      assert Nx.shape(output_10) == {2, 8}

      # Set size 30
      params_30 =
        init_fn.(Nx.template({2, 30, 4}, :f32), Axon.ModelState.empty())

      output_30 = predict_fn.(params_30, Nx.broadcast(0.5, {2, 30, 4}))
      assert Nx.shape(output_30) == {2, 8}
    end

    test "supports mean aggregation" do
      model =
        DeepSets.build(
          input_dim: 3,
          output_dim: 5,
          hidden_size: 16,
          phi_sizes: [16],
          rho_sizes: [8],
          aggregation: :mean
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, 8, 3}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 8, 3}))
      assert Nx.shape(output) == {2, 5}
    end

    test "supports max aggregation" do
      model =
        DeepSets.build(
          input_dim: 3,
          output_dim: 5,
          hidden_size: 16,
          phi_sizes: [16],
          rho_sizes: [8],
          aggregation: :max
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, 8, 3}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 8, 3}))
      assert Nx.shape(output) == {2, 5}
    end

    test "uses default options when not specified" do
      model = DeepSets.build(input_dim: 5, output_dim: 3)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({2, 10, 5}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {2, 10, 5}))
      assert Nx.shape(output) == {2, 3}
    end
  end
end
