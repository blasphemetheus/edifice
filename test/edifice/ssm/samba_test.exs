defmodule Edifice.SSM.SambaTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  import Edifice.TestHelpers

  alias Edifice.SSM.Samba

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 16
  @num_heads 4
  @head_dim 4

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_blocks: 1,
    num_heads: @num_heads,
    head_dim: @head_dim,
    window_size: @seq_len,
    state_size: 4,
    expand_factor: 2,
    mlp_dim: 32,
    seq_len: @seq_len
  ]

  defp build_and_run(opts) do
    model = Samba.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    embed = opts[:embed_dim] || @embed_dim
    seq = opts[:seq_len] || opts[:window_size] || @seq_len

    params =
      init_fn.(Nx.template({@batch, seq, embed}, :f32), Axon.ModelState.empty())

    input = random_tensor({@batch, seq, embed})
    output = predict_fn.(params, input)
    {model, output}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = Samba.build(@opts)
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

    test "works with multiple blocks" do
      {_model, output} = build_and_run(Keyword.put(@opts, :num_blocks, 2))
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "handles different embed and hidden sizes" do
      opts =
        Keyword.merge(@opts,
          embed_dim: 48,
          hidden_size: 24,
          num_heads: 4,
          head_dim: 6,
          mlp_dim: 48
        )

      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, 24}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Samba.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert Samba.output_size([]) == 256
    end
  end

  describe "registry integration" do
    test "Edifice.build(:samba, ...) works" do
      model = Edifice.build(:samba, @opts)
      assert %Axon{} = model
    end
  end
end
