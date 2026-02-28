defmodule Edifice.Meta.ManifoldHCTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  import Edifice.TestHelpers

  alias Edifice.Meta.ManifoldHC

  @moduletag timeout: 120_000

  @batch 2
  @seq_len 4
  @hidden_size 32

  @small_opts [
    hidden_size: @hidden_size,
    expansion_rate: 2,
    num_layers: 1,
    num_heads: 2,
    sinkhorn_iters: 3,
    seq_len: @seq_len
  ]

  defp random_input(batch \\ @batch, seq_len \\ @seq_len, hidden_size \\ @hidden_size) do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden_size})
    input
  end

  defp build_and_predict(opts, input) do
    model = ManifoldHC.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"sequence" => Nx.template(Nx.shape(input), :f32)}
    params = init_fn.(template, %{})
    output = predict_fn.(params, %{"sequence" => input})

    {output, params}
  end

  describe "build/1" do
    test "produces correct output shape" do
      input = random_input()
      {output, _params} = build_and_predict(@small_opts, input)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "output values are finite" do
      input = random_input()
      {output, _params} = build_and_predict(@small_opts, input)

      assert_finite!(output)
    end

    test "works with multiple layers" do
      opts = Keyword.put(@small_opts, :num_layers, 3)
      input = random_input(1)
      {output, _params} = build_and_predict(opts, input)

      assert Nx.shape(output) == {1, @seq_len, @hidden_size}
      assert_finite!(output)
    end

    test "supports expansion_rate of 4" do
      opts = Keyword.put(@small_opts, :expansion_rate, 4)
      input = random_input(1)
      {output, _params} = build_and_predict(opts, input)

      assert Nx.shape(output) == {1, @seq_len, @hidden_size}
      assert_finite!(output)
    end

    test "works with batch_size=1" do
      input = random_input(1)
      {output, _params} = build_and_predict(@small_opts, input)

      assert Nx.shape(output) == {1, @seq_len, @hidden_size}
    end

    test "uses default options" do
      model = ManifoldHC.build(seq_len: @seq_len)
      {init_fn, predict_fn} = Axon.build(model)

      key = Nx.Random.key(99)
      {input, _key} = Nx.Random.normal(key, shape: {1, @seq_len, 256})

      template = %{"sequence" => Nx.template({1, @seq_len, 256}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"sequence" => input})

      assert Nx.shape(output) == {1, @seq_len, 256}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert ManifoldHC.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert ManifoldHC.output_size([]) == 256
    end
  end

  describe "recommended_defaults/1" do
    test "returns small config" do
      config = ManifoldHC.recommended_defaults(:small)
      assert Keyword.fetch!(config, :hidden_size) == 256
      assert Keyword.fetch!(config, :expansion_rate) == 4
      assert Keyword.fetch!(config, :num_layers) == 4
    end

    test "returns medium config" do
      config = ManifoldHC.recommended_defaults(:medium)
      assert Keyword.fetch!(config, :hidden_size) == 512
    end

    test "returns large config" do
      config = ManifoldHC.recommended_defaults(:large)
      assert Keyword.fetch!(config, :hidden_size) == 1024
    end
  end

  describe "sinkhorn normalization" do
    test "different sinkhorn iterations both produce valid output" do
      input = random_input(1)

      opts_few = Keyword.put(@small_opts, :sinkhorn_iters, 1)
      opts_many = Keyword.put(@small_opts, :sinkhorn_iters, 10)

      {out_few, _} = build_and_predict(opts_few, input)
      {out_many, _} = build_and_predict(opts_many, input)

      assert_finite!(out_few)
      assert_finite!(out_many)
    end
  end
end
