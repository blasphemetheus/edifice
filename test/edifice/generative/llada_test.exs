defmodule Edifice.Generative.LLaDATest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.LLaDA

  @batch 2
  @seq_len 8
  @vocab_size 32
  @hidden_size 32
  @num_heads 4
  @num_layers 2

  @small_opts [
    vocab_size: @vocab_size,
    seq_len: @seq_len,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: @num_heads,
    intermediate_size: 64
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {tokens, _key} = Nx.Random.randint(key, 0, @vocab_size, shape: {@batch, @seq_len}, type: :s64)
    %{"tokens" => tokens}
  end

  defp build_and_run(opts) do
    model = LLaDA.build(opts)
    {init_fn, predict_fn} = Axon.build(model)
    template = %{"tokens" => Nx.template({@batch, @seq_len}, :s64)}
    params = init_fn.(template, Axon.ModelState.empty())
    output = predict_fn.(params, random_input())
    {model, output}
  end

  describe "LLaDA.build/1" do
    test "returns an Axon model" do
      model = LLaDA.build(@small_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@small_opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@small_opts)
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "GQA support" do
    test "works with grouped query attention (num_kv_heads < num_heads)" do
      opts = Keyword.merge(@small_opts, num_kv_heads: 2)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "works with multi-query attention (num_kv_heads: 1)" do
      opts = Keyword.merge(@small_opts, num_kv_heads: 1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end
  end

  describe "configuration variants" do
    test "custom intermediate_size" do
      opts = Keyword.put(@small_opts, :intermediate_size, 48)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "custom rope_theta" do
      opts = Keyword.put(@small_opts, :rope_theta, 500_000.0)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end
  end

  describe "output_size/1" do
    test "returns vocab_size" do
      assert LLaDA.output_size(vocab_size: 50_257) == 50_257
      assert LLaDA.output_size(vocab_size: 128) == 128
    end
  end
end
