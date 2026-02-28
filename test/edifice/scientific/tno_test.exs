defmodule Edifice.Scientific.TNOTest do
  use ExUnit.Case, async: true
  @moduletag :scientific

  import Edifice.TestHelpers

  alias Edifice.Scientific.TNO

  @moduletag timeout: 120_000

  @batch 2
  @num_sensors 20
  @num_queries 10
  @history_dim 15
  @coord_dim 2

  @small_opts [
    num_sensors: @num_sensors,
    history_dim: @history_dim,
    coord_dim: @coord_dim,
    branch_hidden: [16, 16],
    temporal_hidden: [16, 16],
    trunk_hidden: [16, 16],
    output_hidden: [16],
    latent_dim: 8
  ]

  defp random_inputs(batch \\ @batch, num_queries \\ @num_queries) do
    %{
      "sensors" => random_tensor({batch, @num_sensors}),
      "history" => random_tensor({batch, @history_dim}),
      "queries" => random_tensor({batch, num_queries, @coord_dim + 1})
    }
  end

  defp build_and_predict(opts, inputs) do
    model = TNO.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "sensors" => Nx.template(Nx.shape(inputs["sensors"]), :f32),
      "history" => Nx.template(Nx.shape(inputs["history"]), :f32),
      "queries" => Nx.template(Nx.shape(inputs["queries"]), :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, inputs)
    {output, params}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = TNO.build(@small_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      inputs = random_inputs()
      {output, _} = build_and_predict(@small_opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 1}
    end

    test "output values are finite" do
      inputs = random_inputs()
      {output, _} = build_and_predict(@small_opts, inputs)
      assert_finite!(output)
    end

    test "batch=1 works" do
      inputs = random_inputs(1, 5)
      {output, _} = build_and_predict(@small_opts, inputs)
      assert Nx.shape(output) == {1, 5, 1}
      assert_finite!(output)
    end

    test "multi-dimensional output" do
      opts = Keyword.put(@small_opts, :output_dim, 3)
      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 3}
      assert_finite!(output)
    end

    test "output_steps > 1 (temporal bundling)" do
      opts = Keyword.put(@small_opts, :output_steps, 4)
      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 4}
      assert_finite!(output)
    end

    test "output_steps * output_dim" do
      opts = Keyword.merge(@small_opts, output_steps: 3, output_dim: 2)
      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 6}
      assert_finite!(output)
    end

    test "1D coordinates" do
      opts = Keyword.merge(@small_opts, coord_dim: 1)
      inputs = %{
        "sensors" => random_tensor({@batch, @num_sensors}),
        "history" => random_tensor({@batch, @history_dim}),
        "queries" => random_tensor({@batch, @num_queries, 2})
      }

      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 1}
    end

    test "without bias" do
      opts = Keyword.put(@small_opts, :use_bias, false)
      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 1}
      assert_finite!(output)
    end

    test "different hidden layer configs" do
      opts =
        Keyword.merge(@small_opts,
          branch_hidden: [32],
          temporal_hidden: [32, 16],
          trunk_hidden: [32],
          output_hidden: [32, 16]
        )

      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert Nx.shape(output) == {@batch, @num_queries, 1}
      assert_finite!(output)
    end

    test "tanh trunk activation" do
      opts = Keyword.put(@small_opts, :trunk_activation, :tanh)
      inputs = random_inputs()
      {output, _} = build_and_predict(opts, inputs)
      assert_finite!(output)
    end

    test "variable query counts" do
      inputs_5 = random_inputs(@batch, 5)
      inputs_20 = random_inputs(@batch, 20)

      {out_5, _} = build_and_predict(@small_opts, inputs_5)
      {out_20, _} = build_and_predict(@small_opts, inputs_20)

      assert Nx.shape(out_5) == {@batch, 5, 1}
      assert Nx.shape(out_20) == {@batch, 20, 1}
    end
  end

  describe "output_size/1" do
    test "returns 1 by default" do
      assert TNO.output_size([]) == 1
    end

    test "returns output_steps * output_dim" do
      assert TNO.output_size(output_steps: 4, output_dim: 3) == 12
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = TNO.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:latent_dim] == 64
      assert defaults[:trunk_activation] == :tanh
    end
  end

  describe "registry integration" do
    test "Edifice.build(:tno, ...) works" do
      model = Edifice.build(:tno, @small_opts)
      assert %Axon{} = model
    end
  end
end
