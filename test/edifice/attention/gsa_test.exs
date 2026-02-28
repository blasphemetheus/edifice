defmodule Edifice.Attention.GSATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Attention.GSA

  @moduletag timeout: 120_000

  @embed_dim 16
  @hidden_size 16
  @num_heads 2
  @num_slots 4
  @num_layers 1
  @batch 2
  @seq_len 4

  defp build_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        num_heads: @num_heads,
        num_slots: @num_slots,
        num_layers: @num_layers,
        seq_len: @seq_len
      ],
      overrides
    )
  end

  defp build_and_run(opts \\ [], batch \\ @batch) do
    model = GSA.build(build_opts(opts))
    {init_fn, predict_fn} = Axon.build(model)
    input = random_tensor({batch, @seq_len, @embed_dim})
    params = init_fn.(input, Axon.ModelState.empty())
    output = predict_fn.(params, input)
    {output, params}
  end

  describe "build/1" do
    test "builds a valid Axon model" do
      model = GSA.build(build_opts())
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {output, _params} = build_and_run()
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output values are finite" do
      {output, _params} = build_and_run()
      assert_finite!(output)
    end

    test "works with batch size 1" do
      {output, _params} = build_and_run([], 1)
      assert Nx.shape(output) == {1, @hidden_size}
      assert_finite!(output)
    end

    test "works with multiple layers" do
      {output, _params} = build_and_run(num_layers: 2)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "works with different slot counts" do
      {output, _params} = build_and_run(num_slots: 8)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "works with embed_dim != hidden_size" do
      model = GSA.build(build_opts(embed_dim: 32, hidden_size: 16))
      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, 32})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)
      assert Nx.shape(output) == {@batch, 16}
      assert_finite!(output)
    end

    test "works with zero dropout" do
      {output, _params} = build_and_run(dropout: 0.0)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GSA.output_size(hidden_size: 128) == 128
    end

    test "returns default hidden_size" do
      assert GSA.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = GSA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_slots)
      assert Keyword.has_key?(defaults, :damping)
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = GSA.param_count(build_opts())
      assert is_integer(count)
      assert count > 0
    end
  end
end
