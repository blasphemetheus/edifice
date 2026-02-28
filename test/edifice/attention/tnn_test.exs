defmodule Edifice.Attention.TNNTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  import Edifice.TestHelpers

  alias Edifice.Attention.TNN

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 16

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    expand_ratio: 2,
    rpe_dim: 8,
    rpe_layers: 1,
    window_size: @seq_len,
    dropout: 0.0
  ]

  defp build_and_run(opts) do
    model = TNN.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    embed = opts[:embed_dim] || @embed_dim
    seq = opts[:window_size] || @seq_len

    template = %{"state_sequence" => Nx.template({@batch, seq, embed}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())
    input = %{"state_sequence" => random_tensor({@batch, seq, embed})}
    output = predict_fn.(params, input)
    {model, output}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = TNN.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {_model, output} = build_and_run(@opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output values are finite" do
      {_model, output} = build_and_run(@opts)
      assert_finite!(output)
    end

    test "batch=1 works" do
      model = TNN.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({1, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())
      input = %{"state_sequence" => random_tensor({1, @seq_len, @embed_dim})}
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, @hidden_size}
      assert_finite!(output)
    end

    test "multiple layers" do
      opts = Keyword.put(@opts, :num_layers, 2)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "expand_ratio=3" do
      opts = Keyword.put(@opts, :expand_ratio, 3)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "rpe_layers=3" do
      opts = Keyword.put(@opts, :rpe_layers, 3)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "without decay" do
      opts = Keyword.put(@opts, :use_decay, false)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "rpe_activation=:silu" do
      opts = Keyword.put(@opts, :rpe_activation, :silu)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "with dropout" do
      opts = Keyword.put(@opts, :dropout, 0.1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert TNN.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert TNN.output_size([]) == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = TNN.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:hidden_size] == 256
      assert defaults[:expand_ratio] == 3
      assert defaults[:rpe_layers] == 3
      assert defaults[:gamma] == 0.99
    end
  end

  describe "registry integration" do
    test "Edifice.build(:tnn, ...) works" do
      model = Edifice.build(:tnn, @opts)
      assert %Axon{} = model
    end
  end
end
