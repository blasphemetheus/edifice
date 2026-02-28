defmodule Edifice.Attention.LogLinearTest do
  use ExUnit.Case, async: true

  @moduletag :attention

  alias Edifice.Attention.LogLinear

  import Edifice.TestHelpers

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_heads 4

  defp base_opts do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_heads: @num_heads,
      num_layers: 2,
      segment_size: 4,
      dropout: 0.0
    ]
  end

  defp build_and_run(opts) do
    model = LogLinear.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    input = random_tensor({@batch, @seq_len, @embed_dim}, 42)
    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  describe "build/1 shape tests" do
    test "returns an Axon model" do
      assert %Axon{} = LogLinear.build(base_opts())
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "works with different segment sizes" do
      opts = Keyword.merge(base_opts(), segment_size: 2)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "works with single layer" do
      opts = Keyword.merge(base_opts(), num_layers: 1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "numerical stability" do
    test "output is finite for random input" do
      {_model, output} = build_and_run(base_opts())
      assert_finite!(output, "log_linear_output")
    end

    test "output is finite for zero input" do
      model = LogLinear.build(base_opts())
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert_finite!(output, "log_linear_zero_input")
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert LogLinear.output_size(hidden_size: 128) == 128
    end

    test "returns default when no opts" do
      assert LogLinear.output_size() == 256
    end
  end

  describe "registry integration" do
    test "Edifice.build(:log_linear, ...) works" do
      model =
        Edifice.build(:log_linear,
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          num_layers: 1
        )

      assert %Axon{} = model
    end
  end
end
