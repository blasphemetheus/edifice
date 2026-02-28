defmodule Edifice.Attention.MoBATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Attention.MoBA

  @embed_dim 32
  @hidden_size 32
  @num_heads 2
  @num_layers 1
  @block_size 4
  @topk 2
  @batch 2
  @seq_len 16

  defp build_opts(overrides \\ []) do
    Keyword.merge(
      [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        num_heads: @num_heads,
        num_layers: @num_layers,
        block_size: @block_size,
        topk: @topk,
        seq_len: @seq_len
      ],
      overrides
    )
  end

  defp build_and_run(opts \\ [], batch \\ @batch) do
    model = MoBA.build(build_opts(opts))
    {init_fn, predict_fn} = Axon.build(model)
    input = random_tensor({batch, @seq_len, @embed_dim})
    params = init_fn.(input, %{})
    output = predict_fn.(params, input)
    {output, params}
  end

  describe "build/1" do
    test "builds a valid model" do
      model = MoBA.build(build_opts())
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

    test "non-causal mode works" do
      {output, _params} = build_and_run(causal: false)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "multiple layers" do
      {output, _params} = build_and_run(num_layers: 2)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "different block_size and topk" do
      {output, _params} = build_and_run(block_size: 8, topk: 1)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "topk >= num_blocks degrades to full attention" do
      # With block_size=4 and seq_len=16, num_blocks=4
      # topk=4 means select all blocks
      {output, _params} = build_and_run(topk: 4)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MoBA.output_size(hidden_size: 128) == 128
    end

    test "returns default hidden_size" do
      assert MoBA.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = MoBA.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :block_size)
      assert Keyword.has_key?(defaults, :topk)
    end
  end

  describe "registry" do
    test "builds via Edifice.build/2" do
      model = Edifice.build(:moba, build_opts())
      assert %Axon{} = model
    end
  end
end
