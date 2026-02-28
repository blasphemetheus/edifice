defmodule Edifice.Attention.RingAttentionTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.RingAttention

  @batch 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 32

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_chunks: 4,
    num_layers: 2,
    dropout: 0.0,
    window_size: @seq_len
  ]

  defp random_sequence do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  defp build_and_predict(opts) do
    model = RingAttention.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(
        Nx.template({@batch, @seq_len, @embed_dim}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_sequence())
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = RingAttention.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      output = build_and_predict(@base_opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with 2 chunks" do
      opts = Keyword.put(@base_opts, :num_chunks, 2)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with num_chunks equal to seq_len" do
      opts = Keyword.put(@base_opts, :num_chunks, @seq_len)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different hidden size" do
      opts =
        @base_opts
        |> Keyword.put(:hidden_size, 64)
        |> Keyword.put(:num_heads, 4)

      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 64}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert RingAttention.output_size(@base_opts) == @hidden_size
    end

    test "returns default when not specified" do
      assert RingAttention.output_size([]) == 256
    end
  end
end
