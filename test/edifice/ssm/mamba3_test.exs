defmodule Edifice.SSM.Mamba3Test do
  use ExUnit.Case, async: true

  alias Edifice.SSM.Mamba3

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
    window_size: @seq_len,
    rank: 4,
    complex: true
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = Mamba3.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = Mamba3.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = Mamba3.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 without complex" do
    test "works with complex: false" do
      opts = Keyword.put(@opts, :complex, false)
      model = Mamba3.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with rank variations" do
    test "rank=1 works" do
      opts = Keyword.put(@opts, :rank, 1)
      model = Mamba3.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "rank=8 works" do
      opts = Keyword.put(@opts, :rank, 8)
      model = Mamba3.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with single layer" do
    test "single layer produces correct shape" do
      opts = Keyword.put(@opts, :num_layers, 1)
      model = Mamba3.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Mamba3.output_size(@opts) == @hidden_size
    end
  end

  describe "recommended_defaults/0" do
    test "includes rank and complex keys" do
      defaults = Mamba3.recommended_defaults()
      assert Keyword.has_key?(defaults, :rank)
      assert Keyword.has_key?(defaults, :complex)
      assert Keyword.has_key?(defaults, :hidden_size)
    end
  end
end
