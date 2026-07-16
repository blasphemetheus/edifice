defmodule Edifice.SSM.MambaTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  alias Edifice.SSM.Mamba

  @batch 4
  @seq_len 12
  @embed_dim 64
  @hidden_size 32
  @state_size 8
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    state_size: @state_size,
    num_layers: @num_layers,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    @tag :smoke
    test "builds an Axon model" do
      model = Mamba.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Mamba.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Mamba.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Mamba.output_size(@opts) == @hidden_size
    end
  end

  describe "Common.recommended_defaults/1 (overrides)" do
    alias Edifice.SSM.Common

    test "merges overrides over the base defaults" do
      defaults = Common.recommended_defaults(hidden_size: 128, seq_len: 30)

      assert defaults[:hidden_size] == 128
      assert defaults[:seq_len] == 30
      # untouched base keys survive the merge
      assert defaults[:state_size] == Common.recommended_defaults()[:state_size]
      assert defaults[:window_size] == Common.recommended_defaults()[:window_size]
    end

    test "with no overrides equals recommended_defaults/0" do
      assert Enum.sort(Common.recommended_defaults([])) ==
               Enum.sort(Common.recommended_defaults())
    end
  end

  describe ":embed_size legacy alias" do
    alias Edifice.SSM.Common

    test "build/1 accepts :embed_size in place of :embed_dim" do
      opts = @opts |> Keyword.delete(:embed_dim) |> Keyword.put(:embed_size, @embed_dim)
      model = Mamba.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "param_count/1 treats :embed_size like :embed_dim" do
      via_alias =
        @opts |> Keyword.delete(:embed_dim) |> Keyword.put(:embed_size, @embed_dim)

      assert Common.param_count(@opts) == Common.param_count(via_alias)
    end

    test ":embed_dim wins when both spellings are given" do
      both = Keyword.put(@opts, :embed_size, 999)
      assert Common.param_count(both) == Common.param_count(@opts)
    end

    test "build_model/2 still raises KeyError when neither spelling is given" do
      assert_raise KeyError, fn ->
        Mamba.build(Keyword.delete(@opts, :embed_dim))
      end
    end
  end
end
