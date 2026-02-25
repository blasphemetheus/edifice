defmodule Edifice.Generative.LinearDiTTest do
  use ExUnit.Case, async: true

  alias Edifice.Generative.LinearDiT

  describe "build/1" do
    test "builds model with default options" do
      model =
        LinearDiT.build(
          input_dim: 32,
          hidden_size: 64,
          num_layers: 2,
          num_heads: 4
        )

      assert %Axon{} = model
    end

    test "builds model with class conditioning" do
      model =
        LinearDiT.build(
          input_dim: 32,
          hidden_size: 64,
          num_layers: 2,
          num_heads: 4,
          num_classes: 10
        )

      assert %Axon{} = model
    end

    test "model produces output with correct shape" do
      input_dim = 32
      hidden_size = 64
      batch_size = 2

      model =
        LinearDiT.build(
          input_dim: input_dim,
          hidden_size: hidden_size,
          num_layers: 2,
          num_heads: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_input" => Nx.template({batch_size, input_dim}, :f32),
            "timestep" => Nx.template({batch_size}, :f32)
          },
          %{}
        )

      input = %{
        "noisy_input" => Nx.broadcast(0.1, {batch_size, input_dim}),
        "timestep" => Nx.tensor([50, 100])
      }

      output = predict_fn.(params, input)

      assert {^batch_size, ^input_dim} = Nx.shape(output)
    end

    test "model with class conditioning produces correct output" do
      input_dim = 32
      hidden_size = 64
      batch_size = 2
      num_classes = 10

      model =
        LinearDiT.build(
          input_dim: input_dim,
          hidden_size: hidden_size,
          num_layers: 2,
          num_heads: 4,
          num_classes: num_classes
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "noisy_input" => Nx.template({batch_size, input_dim}, :f32),
            "timestep" => Nx.template({batch_size}, :f32),
            "class_label" => Nx.template({batch_size}, :f32)
          },
          %{}
        )

      input = %{
        "noisy_input" => Nx.broadcast(0.1, {batch_size, input_dim}),
        "timestep" => Nx.tensor([50, 100]),
        "class_label" => Nx.tensor([0, 5])
      }

      output = predict_fn.(params, input)

      assert {^batch_size, ^input_dim} = Nx.shape(output)
    end
  end

  describe "build_linear_dit_block/3" do
    test "builds a single block" do
      hidden_size = 64
      num_heads = 4

      input = Axon.input("input", shape: {nil, hidden_size})
      condition = Axon.input("condition", shape: {nil, hidden_size})

      output =
        LinearDiT.build_linear_dit_block(input, condition,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: 4.0,
          name: "test_block"
        )

      assert %Axon{} = output
    end
  end

  describe "output_size/1" do
    test "returns input_dim" do
      assert LinearDiT.output_size(input_dim: 128) == 128
    end

    test "returns default when not specified" do
      assert LinearDiT.output_size([]) == 64
    end
  end

  describe "param_count/1" do
    test "returns positive count" do
      count =
        LinearDiT.param_count(
          input_dim: 64,
          hidden_size: 256,
          num_layers: 6
        )

      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = LinearDiT.recommended_defaults()

      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :mlp_ratio)
    end
  end
end
