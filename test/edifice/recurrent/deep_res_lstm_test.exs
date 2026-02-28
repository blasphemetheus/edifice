defmodule Edifice.Recurrent.DeepResLSTMTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.DeepResLSTM

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 64

  defp build_and_run(opts) do
    model = DeepResLSTM.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  defp base_opts do
    [
      embed_dim: @embed_dim,
      hidden_size: @hidden_size,
      num_layers: 2,
      seq_len: @seq_len,
      dropout: 0.0
    ]
  end

  describe "build/1 shape tests" do
    @tag :smoke
    test "returns an Axon model" do
      model = DeepResLSTM.build(base_opts())
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      opts = Keyword.merge(base_opts(), embed_dim: 48, hidden_size: 32)

      model = DeepResLSTM.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, 48}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, 48})

      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch, 32}
    end

    test "handles same embed and hidden sizes" do
      opts = Keyword.merge(base_opts(), embed_dim: @hidden_size, hidden_size: @hidden_size)

      model = DeepResLSTM.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})

      output = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 layer count variation" do
    test "different num_layers produce correct shape" do
      for num_layers <- [1, 2, 3, 4] do
        opts = Keyword.put(base_opts(), :num_layers, num_layers)
        {_model, output} = build_and_run(opts)

        assert Nx.shape(output) == {@batch, @hidden_size},
               "Failed for num_layers=#{num_layers}"
      end
    end
  end

  describe "numerical stability" do
    test "output is finite for random input" do
      {_model, output} = build_and_run(base_opts())
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      model = DeepResLSTM.build(base_opts())
      {init_fn, predict_fn} = Axon.build(model)

      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with dropout" do
    test "produces correct shape in inference mode" do
      opts = Keyword.put(base_opts(), :dropout, 0.1)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite with dropout" do
      opts = Keyword.put(base_opts(), :dropout, 0.1)
      {_model, output} = build_and_run(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DeepResLSTM.output_size(hidden_size: 256) == 256
    end

    test "returns default when no option given" do
      assert DeepResLSTM.output_size([]) == 512
    end
  end

  describe "registry integration" do
    test "Edifice.build(:deep_res_lstm, ...) works" do
      model =
        Edifice.build(:deep_res_lstm,
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end
end
