defmodule Edifice.Graph.DimeNetTest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Graph.DimeNet

  @moduletag timeout: 120_000

  @batch 2
  @num_atoms 8
  @input_dim 16

  # Small config for fast tests
  @small_opts [
    input_dim: @input_dim,
    hidden_size: 32,
    num_blocks: 2,
    num_radial: 4,
    num_spherical: 3,
    cutoff: 5.0,
    envelope_exponent: 5,
    int_emb_size: 16,
    basis_emb_size: 4,
    out_emb_size: 32,
    num_output_layers: 2
  ]

  defp random_inputs(batch \\ @batch, num_atoms \\ @num_atoms) do
    key = Nx.Random.key(42)
    {nodes, key} = Nx.Random.normal(key, shape: {batch, num_atoms, @input_dim})
    {positions, _key} = Nx.Random.normal(key, shape: {batch, num_atoms, 3})

    %{"nodes" => nodes, "positions" => positions}
  end

  defp build_and_predict(opts, inputs) do
    model = DimeNet.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "nodes" => Nx.template(Nx.shape(inputs["nodes"]), :f32),
      "positions" => Nx.template(Nx.shape(inputs["positions"]), :f32)
    }

    params = init_fn.(template, %{})
    output = predict_fn.(params, inputs)

    {output, params}
  end

  describe "build/1" do
    test "produces correct output shape (per-atom)" do
      inputs = random_inputs()
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert Nx.shape(output) == {@batch, @num_atoms, 32}
    end

    test "output values are finite" do
      inputs = random_inputs()
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert_finite!(output)
    end

    test "works with batch_size=1" do
      inputs = random_inputs(1, 6)
      {output, _params} = build_and_predict(@small_opts, inputs)

      assert Nx.shape(output) == {1, 6, 32}
      assert_finite!(output)
    end

    test "supports global sum pooling" do
      opts = Keyword.merge(@small_opts, pool: :sum)
      inputs = random_inputs()
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, 32}
    end

    test "supports output classification head" do
      opts = Keyword.merge(@small_opts, pool: :sum, num_classes: 1)
      inputs = random_inputs()
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {@batch, 1}
      assert_finite!(output)
    end

    test "works with single block" do
      opts = Keyword.put(@small_opts, :num_blocks, 1)
      inputs = random_inputs(1, 6)
      {output, _params} = build_and_predict(opts, inputs)

      assert Nx.shape(output) == {1, 6, 32}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns out_emb_size by default" do
      assert DimeNet.output_size(out_emb_size: 128) == 128
    end

    test "returns num_classes when specified" do
      assert DimeNet.output_size(num_classes: 5, out_emb_size: 128) == 5
    end

    test "returns default out_emb_size" do
      assert DimeNet.output_size([]) == 256
    end
  end
end
