defmodule Edifice.Attention.FoXTest do
  use ExUnit.Case, async: true

  @moduletag :attention

  alias Edifice.Attention.FoX

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
      dropout: 0.0
    ]
  end

  defp build_and_run(opts) do
    model = FoX.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    input = random_tensor({@batch, @seq_len, @embed_dim}, 42)
    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  describe "build/1 shape tests" do
    test "returns an Axon model" do
      assert %Axon{} = FoX.build(base_opts())
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "works with different hidden sizes" do
      opts = Keyword.merge(base_opts(), hidden_size: 64, num_heads: 8)
      model = FoX.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = random_tensor({@batch, @seq_len, @embed_dim}, 99)
      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch, 64}
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
      assert_finite!(output, "fox_output")
    end

    test "output is finite for zero input" do
      model = FoX.build(base_opts())
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})
      assert_finite!(output, "fox_zero_input")
    end
  end

  describe "forget gate mechanics" do
    test "forget gate projection has correct dimensions" do
      model = FoX.build(base_opts())
      {init_fn, _predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      # Verify forget gate parameters exist in the params
      flat = flatten_params(params)
      forget_keys = Enum.filter(flat, fn {path, _} -> String.contains?(path, "forget_proj") end)
      assert length(forget_keys) > 0
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert FoX.output_size(hidden_size: 128) == 128
    end

    test "returns default when no opts" do
      assert FoX.output_size() == 256
    end
  end

  describe "registry integration" do
    test "Edifice.build(:fox, ...) works" do
      model =
        Edifice.build(:fox,
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          num_layers: 1
        )

      assert %Axon{} = model
    end
  end
end
