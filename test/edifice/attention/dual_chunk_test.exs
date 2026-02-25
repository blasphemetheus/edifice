defmodule Edifice.Attention.DualChunkTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.DualChunk

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @chunk_size 4

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_layers: 2,
    chunk_size: @chunk_size,
    dropout: 0.0,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  defp template,
    do: %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}

  describe "build/1" do
    test "builds an Axon model" do
      model = DualChunk.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = DualChunk.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = DualChunk.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_dual_chunk_attention/2" do
    test "builds attention sublayer" do
      input = Axon.input("x", shape: {nil, @seq_len, @hidden_size})

      attn =
        DualChunk.build_dual_chunk_attention(input,
          hidden_size: @hidden_size,
          num_heads: 4,
          head_dim: 8,
          chunk_size: @chunk_size,
          name: "test_attn"
        )

      assert %Axon{} = attn
    end

    test "produces correct output shape for sublayer" do
      input = Axon.input("x", shape: {nil, @seq_len, @hidden_size})

      attn =
        DualChunk.build_dual_chunk_attention(input,
          hidden_size: @hidden_size,
          num_heads: 4,
          head_dim: 8,
          chunk_size: @chunk_size,
          name: "test_attn"
        )

      template = %{"x" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)}
      {init_fn, predict_fn} = Axon.build(attn, mode: :inference)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(123)
      {input_tensor, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"x" => input_tensor})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DualChunk.output_size(@opts) == @hidden_size
    end

    test "returns default when not specified" do
      assert DualChunk.output_size([]) == 256
    end
  end

  describe "integration with decoder_only" do
    test "decoder_only with dual_chunk attention builds" do
      model =
        Edifice.Transformer.DecoderOnly.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
          attention_type: :dual_chunk,
          chunk_size: @chunk_size,
          window_size: @seq_len,
          dropout: 0.0
        )

      assert %Axon{} = model
    end

    test "decoder_only with dual_chunk attention produces correct output" do
      model =
        Edifice.Transformer.DecoderOnly.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
          attention_type: :dual_chunk,
          chunk_size: @chunk_size,
          window_size: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "registry" do
    test "dual_chunk_attention is in registry" do
      assert :dual_chunk_attention in Edifice.list_architectures()
    end

    test "can build from registry" do
      model =
        Edifice.build(:dual_chunk_attention,
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
          chunk_size: @chunk_size,
          window_size: @seq_len
        )

      assert %Axon{} = model
    end
  end
end
