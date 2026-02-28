defmodule Edifice.Scientific.DeepONetTest do
  use ExUnit.Case, async: true
  @moduletag :scientific

  import Edifice.TestHelpers

  alias Edifice.Scientific.DeepONet

  @moduletag timeout: 120_000

  @batch 4
  @num_sensors 50
  @num_queries 20
  @coord_dim 2

  @small_opts [
    num_sensors: @num_sensors,
    coord_dim: @coord_dim,
    branch_hidden: [32, 32],
    trunk_hidden: [32, 32],
    latent_dim: 16
  ]

  defp random_inputs(batch \\ @batch, num_queries \\ @num_queries) do
    key = Nx.Random.key(42)
    {sensors, key} = Nx.Random.normal(key, shape: {batch, @num_sensors})
    {queries, _key} = Nx.Random.normal(key, shape: {batch, num_queries, @coord_dim})

    %{"sensors" => sensors, "queries" => queries}
  end

  defp build_and_predict(opts, inputs) do
    model = DeepONet.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "sensors" => Nx.template(Nx.shape(inputs["sensors"]), :f32),
      "queries" => Nx.template(Nx.shape(inputs["queries"]), :f32)
    }

    params = init_fn.(template, %{})
    output = predict_fn.(params, inputs)

    {output, params}
  end

  describe "build/1" do
    test "produces correct output shape" do
      inputs = random_inputs()
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 1}
    end

    test "output values are finite" do
      inputs = random_inputs()
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert_finite!(output)
    end

    test "works with batch_size=1" do
      inputs = random_inputs(1, 10)
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert Nx.shape(output) == {1, 10, 1}
      assert_finite!(output)
    end

    test "supports multi-dimensional output" do
      opts = Keyword.put(@small_opts, :output_dim, 3)
      inputs = random_inputs()
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 3}
      assert_finite!(output)
    end

    test "supports 1D coordinates" do
      opts = Keyword.merge(@small_opts, coord_dim: 1)
      key = Nx.Random.key(42)
      {sensors, key} = Nx.Random.normal(key, shape: {@batch, @num_sensors})
      {queries, _key} = Nx.Random.normal(key, shape: {@batch, @num_queries, 1})
      inputs = %{"sensors" => sensors, "queries" => queries}

      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 1}
    end

    test "supports 3D coordinates" do
      opts = Keyword.merge(@small_opts, coord_dim: 3)
      key = Nx.Random.key(42)
      {sensors, key} = Nx.Random.normal(key, shape: {@batch, @num_sensors})
      {queries, _key} = Nx.Random.normal(key, shape: {@batch, @num_queries, 3})
      inputs = %{"sensors" => sensors, "queries" => queries}

      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 1}
    end

    test "works without bias" do
      opts = Keyword.put(@small_opts, :use_bias, false)
      inputs = random_inputs()
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 1}
      assert_finite!(output)
    end

    test "supports different hidden layer configs" do
      opts = Keyword.merge(@small_opts, branch_hidden: [64], trunk_hidden: [64, 32])
      inputs = random_inputs()
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, @num_queries, 1}
      assert_finite!(output)
    end

    test "supports variable number of query points" do
      inputs_10 = random_inputs(@batch, 10)
      inputs_50 = random_inputs(@batch, 50)

      {out_10, _} = build_and_predict(@small_opts, inputs_10)
      {out_50, _} = build_and_predict(@small_opts, inputs_50)

      assert Nx.shape(out_10) == {@batch, 10, 1}
      assert Nx.shape(out_50) == {@batch, 50, 1}
    end
  end

  describe "output_size/1" do
    test "returns 1 by default" do
      assert DeepONet.output_size([]) == 1
    end

    test "returns output_dim when specified" do
      assert DeepONet.output_size(output_dim: 3) == 3
    end
  end
end
