defmodule Edifice.Attention.MegaTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.Mega

  @batch_size 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 64
  @ema_dim 8

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    ema_dim: @ema_dim,
    num_layers: 2,
    window_size: @seq_len,
    dropout: 0.0
  ]

  defp build_and_run(opts) do
    model = Mega.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "state_sequence" => Nx.template({@batch_size, @seq_len, @embed_dim}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())

    input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_dim})
    output = predict_fn.(params, %{"state_sequence" => input})
    {model, output}
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = Mega.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      {_model, output} = build_and_run(@opts)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "output contains finite values" do
      {_model, output} = build_and_run(@opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with laplace_attention" do
      opts = Keyword.put(@opts, :laplace_attention, true)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different ema_dim" do
      opts = Keyword.put(@opts, :ema_dim, 4)
      {_model, output} = build_and_run(opts)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works when embed_dim equals hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: @hidden_size, hidden_size: @hidden_size)

      model = Mega.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "state_sequence" => Nx.template({@batch_size, @seq_len, @hidden_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Mega.output_size(@opts) == @hidden_size
    end
  end
end
