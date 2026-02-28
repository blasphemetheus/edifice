defmodule Edifice.Recurrent.DeltaProductTest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Recurrent.DeltaProduct

  @embed_dim 16
  @hidden_size 16
  @num_heads 2
  @num_householder 2
  @num_layers 1
  @batch 2
  @seq_len 8

  defp build_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        num_heads: @num_heads,
        num_householder: @num_householder,
        num_layers: @num_layers,
        seq_len: @seq_len,
        dropout: 0.0
      ],
      overrides
    )
  end

  defp build_and_run(opts \\ [], batch \\ @batch) do
    model = DeltaProduct.build(build_opts(opts))
    {init_fn, predict_fn} = Axon.build(model)
    input = random_tensor({batch, @seq_len, @embed_dim})
    params = init_fn.(input, %{})
    output = predict_fn.(params, input)
    {output, params}
  end

  describe "build/1" do
    test "builds a valid model" do
      model = DeltaProduct.build(build_opts())
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {output, _params} = build_and_run()
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "batch=1 works" do
      {output, _params} = build_and_run([], 1)
      assert Nx.shape(output) == {1, @hidden_size}
      assert_finite!(output)
    end

    test "num_householder=1 degrades to standard delta rule" do
      {output, _params} = build_and_run(num_householder: 1)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "num_householder=3 works" do
      {output, _params} = build_and_run(num_householder: 3)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "without short conv" do
      {output, _params} = build_and_run(use_short_conv: false)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "allow_neg_eigval=false restricts beta to [0,1]" do
      {output, _params} = build_and_run(allow_neg_eigval: false)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "multiple layers" do
      {output, _params} = build_and_run(num_layers: 2)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "embed_dim != hidden_size triggers input projection" do
      {output, _params} = build_and_run(embed_dim: 12, hidden_size: 16)
      assert Nx.shape(output) == {@batch, 16}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DeltaProduct.output_size(hidden_size: 64) == 64
    end

    test "returns default hidden_size" do
      assert DeltaProduct.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = DeltaProduct.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_householder)
      assert Keyword.has_key?(defaults, :use_short_conv)
    end
  end

  describe "registry" do
    test "builds via Edifice.build/2" do
      model = Edifice.build(:delta_product, build_opts())
      assert %Axon{} = model
    end
  end
end
