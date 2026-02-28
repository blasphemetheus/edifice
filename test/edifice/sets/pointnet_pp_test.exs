defmodule Edifice.Sets.PointNetPPTest do
  use ExUnit.Case, async: true
  @moduletag :sets

  import Edifice.TestHelpers

  alias Edifice.Sets.PointNetPP

  @moduletag timeout: 120_000

  @batch 2
  @num_points 32
  @input_dim 3

  @small_opts [
    num_classes: 5,
    input_dim: @input_dim,
    sa_configs: [
      %{num_points: 16, radius: 0.5, max_neighbors: 8, mlp: [16, 16, 32]},
      %{num_points: 8, radius: 1.0, max_neighbors: 8, mlp: [32, 32, 64]}
    ],
    global_mlp: [64, 128],
    fc_dims: [64, 32],
    dropout: 0.0
  ]

  defp random_inputs(batch \\ @batch, num_points \\ @num_points) do
    key = Nx.Random.key(42)
    {points, _key} = Nx.Random.normal(key, shape: {batch, num_points, @input_dim})
    %{"points" => points}
  end

  defp build_and_predict(opts, inputs \\ nil) do
    inputs = inputs || random_inputs()
    model = PointNetPP.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "points" => Nx.template(Nx.shape(inputs["points"]), :f32)
    }

    params = init_fn.(template, %{})
    output = predict_fn.(params, inputs)
    {output, params}
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, _} = build_and_predict(@small_opts)
      assert Nx.shape(output) == {@batch, 5}
    end

    test "output values are finite" do
      {output, _} = build_and_predict(@small_opts)
      assert_finite!(output)
    end

    test "works with batch_size=1" do
      inputs = random_inputs(1)
      {output, _} = build_and_predict(@small_opts, inputs)
      assert Nx.shape(output) == {1, 5}
      assert_finite!(output)
    end

    test "with more classes" do
      opts = Keyword.put(@small_opts, :num_classes, 10)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 10}
      assert_finite!(output)
    end

    test "single SA layer" do
      opts =
        Keyword.put(@small_opts, :sa_configs, [
          %{num_points: 16, radius: 0.5, max_neighbors: 8, mlp: [16, 16, 32]}
        ])

      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 5}
      assert_finite!(output)
    end

    test "with 6D input (xyz + normals)" do
      opts =
        Keyword.merge(@small_opts,
          input_dim: 6,
          sa_configs: [
            %{num_points: 16, radius: 0.5, max_neighbors: 8, mlp: [16, 16, 32]}
          ]
        )

      key = Nx.Random.key(42)
      {points, _} = Nx.Random.normal(key, shape: {@batch, @num_points, 6})
      inputs = %{"points" => points}

      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, 5}
      assert_finite!(output)
    end

    test "with dropout" do
      opts = Keyword.put(@small_opts, :dropout, 0.2)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 5}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert PointNetPP.output_size(num_classes: 10) == 10
    end

    test "returns num_classes for different values" do
      assert PointNetPP.output_size(num_classes: 40) == 40
    end
  end
end
