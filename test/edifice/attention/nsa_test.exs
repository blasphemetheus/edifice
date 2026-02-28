defmodule Edifice.Attention.NSATest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.NSA

  @batch 2
  @seq_len 32
  @embed_dim 64
  @hidden_size 32
  @num_heads 4
  @head_dim 8
  @window_size 16
  @block_size 8
  @num_selected_blocks 2
  @compression_ratio 4
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    head_dim: @head_dim,
    window_size: @window_size,
    block_size: @block_size,
    num_selected_blocks: @num_selected_blocks,
    compression_ratio: @compression_ratio,
    num_layers: @num_layers,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = NSA.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = NSA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = NSA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different sequence lengths" do
      # Test with longer sequence
      long_seq_len = 64
      model = NSA.build(Keyword.put(@opts, :seq_len, long_seq_len))

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, long_seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(123)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, long_seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "single layer model works" do
      model = NSA.build(Keyword.put(@opts, :num_layers, 1))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "works with different compression ratios" do
      model = NSA.build(Keyword.put(@opts, :compression_ratio, 8))
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build_nsa_block/2" do
    test "builds a single NSA block" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        NSA.build_nsa_block(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          window_size: @window_size,
          block_size: @block_size,
          num_selected_blocks: @num_selected_blocks,
          compression_ratio: @compression_ratio,
          dropout: 0.0,
          name: "test_block"
        )

      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(123)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block preserves sequence shape
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  describe "build_nsa_attention/2" do
    test "builds the NSA attention layer" do
      input = Axon.input("x", shape: {nil, @seq_len, @hidden_size})

      attn =
        NSA.build_nsa_attention(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          window_size: @window_size,
          block_size: @block_size,
          num_selected_blocks: @num_selected_blocks,
          compression_ratio: @compression_ratio,
          name: "test_attn"
        )

      assert %Axon{} = attn

      {init_fn, predict_fn} = Axon.build(attn)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(456)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Attention preserves sequence shape
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert NSA.output_size(@opts) == @hidden_size
    end

    test "returns default when no opts" do
      assert NSA.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = NSA.param_count(@opts)
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = NSA.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :head_dim)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :block_size)
      assert Keyword.has_key?(defaults, :num_selected_blocks)
      assert Keyword.has_key?(defaults, :compression_ratio)
    end
  end
end
